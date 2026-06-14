"""
STRATA real-data run - critical-mineral district (minerals sector).

Points the wired Sentinel-2 L2A connector at a catalog deposit and runs the
SHAD-RD engine's MINERAL-domain features on real surface reflectance. Same
engine, same connector that produced the Tennessee water-quality run; only the
index domain (mineral / disturbance instead of water) and the target site
change - the cross-sector transfer, demonstrated on real imagery.

For each district it:
  1. pulls a cloud-masked Sentinel-2 L2A median composite over the deposit,
  2. computes alteration indices (iron-oxide, ferrous-iron, clay/hydroxyl,
     gossan) and the bare-ground disturbance index (BSI) on the real grids,
  3. quantifies how anomalous the KNOWN mine footprint is versus the district
     background (percentile + z-score) - a concrete, checkable result,
  4. runs an honest within-district detectability check (can the engine's real
     features separate the footprint from background under random CV?),
  5. writes real outputs to outputs/<slug>_*.{png,html,csv,json}.

Honesty: multispectral Sentinel-2 indices are coarse alteration proxies; this
detects and characterises the KNOWN footprint, it is not a trained
prospectivity model and does not, alone, find new deposits. Spatial
generalisation to unseen ground needs many districts (the catalog) plus
hyperspectral / Prithvi-EO embeddings and geophysics on the workstation.

Run:  python run_mineral_district.py "Mountain Pass"
      python run_mineral_district.py "Thacker Pass"
On a TLS-intercepting network first: export STRATA_RELAX_SSL=1
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strata.shadrd import SentinelL2AConnector, ShadrdModel, indices, features
from strata.sectors import minerals as M
from strata import viz

OUT = os.path.join(os.path.dirname(__file__), "outputs")
RELAX_SSL = os.environ.get("STRATA_RELAX_SSL", "0") == "1"
START, END = "2024-05-01", "2024-10-31"


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def truecolor_figure(bands, extent, title, centroid, out_path):
    """Percentile-stretched real Sentinel-2 true-colour context image."""
    rgb = np.dstack([bands["red"], bands["green"], bands["blue"]])
    out = np.zeros_like(rgb)
    for i in range(3):
        ch = rgb[:, :, i]
        lo, hi = np.nanpercentile(ch, 2), np.nanpercentile(ch, 98)
        out[:, :, i] = np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1)
    out = np.nan_to_num(out)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.imshow(out, extent=extent, origin="upper", aspect="auto")
    import matplotlib.patheffects as pe
    sc = ax.scatter([centroid[1]], [centroid[0]], s=260, facecolors="none",
                    edgecolors="white", linewidths=2.6, label="known deposit", zorder=6)
    sc.set_path_effects([pe.withStroke(linewidth=5.2, foreground="black")])
    ax.set_title(title)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_district(site_name, *, half_deg=0.09, resolution=40, max_scenes=5):
    os.makedirs(OUT, exist_ok=True)
    cat = M.load_catalog()
    row = cat[cat["name"].str.contains(site_name, case=False)]
    if row.empty:
        raise SystemExit(f"'{site_name}' not in catalog; options: {list(cat['name'])}")
    site = row.iloc[0]
    clat, clon = float(site["lat"]), float(site["lon"])
    bbox = [clon - half_deg, clat - half_deg, clon + half_deg, clat + half_deg]
    epsg = utm_epsg(clon, clat)
    s = slug(site["name"])

    print("=" * 64)
    print(f" {site['name']} ({site['state']}) - {site['commodity']}, {site['status']}")
    print(f" centroid ({clat:.4f}, {clon:.4f})  bbox {[round(v,3) for v in bbox]}  EPSG {epsg}")
    print("=" * 64)

    conn = SentinelL2AConnector(relax_ssl=RELAX_SSL)
    print("pulling REAL Sentinel-2 L2A composite ...")
    g = conn.reflectance_grid(bbox, START, END, max_cloud=8,
                              max_scenes=max_scenes, resolution=resolution,
                              epsg=epsg)
    meta = g["meta"]
    bands, extent = g["bands"], g["extent"]
    print(f"  scenes: {meta['n_scenes']}  dates: {meta['dates']}  res: {resolution} m")

    # engine MINERAL-domain indices + disturbance index on the real grids
    miner = indices.compute(bands, domain="mineral")   # iron_oxide, ferrous_iron, clay_minerals, gossan, ndvi
    bsi = indices.bsi(bands)
    lat2d, lon2d = g["lat"], g["lon"]

    # ---- anomaly of the KNOWN footprint vs district background ----
    d = np.sqrt((lat2d - clat) ** 2 + (lon2d - clon) ** 2)
    r_fp, r_bg = 0.012, 0.05            # ~1.3 km footprint, >5.5 km background
    fp = (d <= r_fp) & np.isfinite(bands["red"])
    bg = (d > r_bg) & np.isfinite(bands["red"])

    def anomaly(name, grid):
        gv = grid[np.isfinite(grid)]
        fpv, bgv = grid[fp], grid[bg]
        fpv, bgv = fpv[np.isfinite(fpv)], bgv[np.isfinite(bgv)]
        if len(fpv) == 0 or len(bgv) == 0:
            return None
        fp_med = float(np.median(fpv))
        pct = float((gv < fp_med).mean() * 100)          # percentile in district
        z = float((fp_med - bgv.mean()) / (bgv.std() + 1e-9))
        return name, fp_med, pct, z

    print(f"\nfootprint pixels: {int(fp.sum())}   background pixels: {int(bg.sum())}")
    print("KNOWN-FOOTPRINT ANOMALY vs district background (real reflectance):")
    rows = []
    for nm, grid in [("iron_oxide", miner["iron_oxide"]),
                     ("clay_minerals", miner["clay_minerals"]),
                     ("gossan", miner["gossan"]),
                     ("bsi_disturbance", bsi),
                     ("ndvi", miner["ndvi"])]:
        a = anomaly(nm, grid)
        if a:
            rows.append(a)
            print(f"  {a[0]:16s} footprint median={a[1]:.4f}  "
                  f"district pctile={a[2]:5.1f}  z={a[3]:+.2f}")
    anomaly_df = pd.DataFrame(rows, columns=["index", "footprint_median",
                                             "district_percentile", "z_score"])

    # ---- honest within-district detectability (random CV only) ----
    tbl = pd.DataFrame({
        "lat": lat2d.ravel(), "lon": lon2d.ravel(),
        **{c: bands[c].ravel() for c in ("blue", "green", "red", "nir", "swir1", "swir2")},
    })
    dd = d.ravel()
    label = np.where(dd <= r_fp, 1, np.where(dd > r_bg, 0, -1))
    tbl["label"] = label
    tbl = tbl.dropna().reset_index(drop=True)
    train = tbl[tbl["label"] >= 0].reset_index(drop=True)
    detect_auc = None
    if train["label"].nunique() == 2 and int((train["label"] == 1).sum()) >= 10:
        X, names = features.assemble(train, domain="mineral")
        y = train["label"].to_numpy()
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        aucs = []
        for tr, te in skf.split(X, y):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ShadrdModel(task="classification", feature_names=names).fit(X[tr], y[tr])
                aucs.append(roc_auc_score(y[te], m.score_surface(X[te])))
        detect_auc = float(np.mean(aucs))
        print(f"\nwithin-district footprint detectability (random 5-fold AUC): "
              f"{detect_auc:.3f} +/- {np.std(aucs):.3f}")
        print("  (separability of the KNOWN footprint from background on real "
              "features;\n   NOT prospectivity - spatial generalisation needs "
              "many districts + hyperspectral/Prithvi.)")

    # ---- figures ----
    title_sfx = f"{site['name']} ({site['commodity']}) - Sentinel-2 L2A {meta['dates'][0]} +{meta['n_scenes']-1}"
    truecolor_figure(bands, extent, f"True colour - {title_sfx}", (clat, clon),
                     os.path.join(OUT, f"{s}_truecolor.png"))
    io_grid = miner["iron_oxide"]
    viz.raster_map(
        io_grid, extent, f"Iron-oxide ratio (red/blue) - {title_sfx}",
        "iron-oxide ratio  (red = more ferric/oxidised, blue = less)",
        cmap="RdBu_r", center=float(np.nanmedian(io_grid)), smooth=3,
        marker=(clat, clon), out_path=os.path.join(OUT, f"{s}_iron_oxide.png"))
    viz.raster_map(
        bsi, extent, f"Bare-ground / disturbance (BSI) - {title_sfx}",
        "BSI (high = bare/disturbed ground)", cmap="cividis", smooth=3,
        marker=(clat, clon),
        vmin=float(np.nanpercentile(bsi, 2)), vmax=float(np.nanpercentile(bsi, 98)),
        out_path=os.path.join(OUT, f"{s}_bsi.png"))

    # ---- per-pixel CSV + folium map coloured by iron-oxide, deposit marked ----
    out_tbl = tbl.copy()
    out_tbl["iron_oxide"] = indices.iron_oxide(
        {c: out_tbl[c].to_numpy() for c in indices.BANDS})
    out_tbl["bsi"] = indices.bsi({c: out_tbl[c].to_numpy() for c in indices.BANDS})
    out_tbl.drop(columns=["label"]).to_csv(
        os.path.join(OUT, f"{s}_indices.csv"), index=False)

    import folium
    fmap = viz.value_points_map(out_tbl.sample(min(800, len(out_tbl)), random_state=0),
                                "iron_oxide", caption="Iron-oxide ratio (real S2)",
                                zoom_start=12)
    # high-visibility deposit marker: white ring with a thin black border
    folium.CircleMarker([clat, clon], radius=12, color="black", weight=6,
                        fill=False, opacity=1.0).add_to(fmap)
    folium.CircleMarker([clat, clon], radius=12, color="white", weight=2.5,
                        fill=False, opacity=1.0,
                        popup=f"{site['name']} ({site['commodity']})",
                        tooltip=site["name"]).add_to(fmap)
    fmap.save(os.path.join(OUT, f"{s}_map.html"))

    meta.update({"site": site["name"], "commodity": site["commodity"],
                 "centroid": [clat, clon], "footprint_pixels": int(fp.sum()),
                 "background_pixels": int(bg.sum()),
                 "detect_random_auc": detect_auc,
                 "anomaly": anomaly_df.to_dict(orient="records")})
    json.dump(meta, open(os.path.join(OUT, f"{s}_meta.json"), "w"), indent=2)
    print(f"\nwrote: {s}_truecolor.png, {s}_iron_oxide.png, {s}_bsi.png, "
          f"{s}_map.html, {s}_indices.csv, {s}_meta.json")
    return meta


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "Mountain Pass"
    run_district(site)
