"""
STRATA real-data run - Tennessee basin (Wheeler Lake / Tennessee River).

Validates the SHAD-RD engine's REAL imagery path end to end on Dr. Yates's
dissertation study area, using credential-free Sentinel-2 L2A surface
reflectance (AWS Open Data via Element84 earth-search). It:

  1. pulls a cloud-masked, multi-scene median reflectance composite,
  2. runs the engine's water-domain indices (MNDWI water mask, NDTI turbidity
     proxy, NDWI, NDVI) on the real grids,
  3. writes real outputs to outputs/:
       tn_water_mndwi.png        water extent (MNDWI)
       tn_turbidity_proxy.png    NDTI turbidity proxy over water
       tn_basin_map.html         per-pixel NDTI on a basemap
       tn_reflectance_indices.csv per-pixel reflectance + indices
       tn_run_meta.json          scenes, dates, provenance, stats

Scope note: this demonstrates the real feature pipeline on real imagery.
Turbidity *prediction* (a trained model) requires the dissertation's in-situ
NTU labels, which are not bundled here; plug them in as the regression target
to reproduce the SHAD-RD water-quality model on this composite.

Run:  python run_tn_basin.py
On a TLS-intercepting network (proxy/sandbox) first: export STRATA_RELAX_SSL=1
"""
from __future__ import annotations

import json
import os

from strata.shadrd import SentinelL2AConnector, indices
from strata import viz

OUT = os.path.join(os.path.dirname(__file__), "outputs")

# Tennessee River / Wheeler Lake reach near Huntsville-Decatur, AL.
AOI_BBOX = [-87.20, 34.45, -86.60, 34.82]
START, END = "2024-06-01", "2024-10-31"
RELAX_SSL = os.environ.get("STRATA_RELAX_SSL", "0") == "1"


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Tennessee basin AOI: {AOI_BBOX}  ({START}..{END})")
    if RELAX_SSL:
        print("(SSL verification relaxed for a TLS-intercepting network)")

    conn = SentinelL2AConnector(relax_ssl=RELAX_SSL)
    print("pulling REAL Sentinel-2 L2A composite ...")
    g = conn.reflectance_grid(AOI_BBOX, START, END, max_cloud=5, max_scenes=5,
                              resolution=150)
    meta = g["meta"]
    print(f"  scenes: {meta['n_scenes']}  dates: {meta['dates']}")
    print(f"  source: {meta['source']}")

    bands, extent = g["bands"], g["extent"]
    import numpy as np

    # engine water-domain indices on the REAL grids
    mndwi = indices.mndwi(bands)
    ndti = indices.ndti(bands)
    ndwi = indices.ndwi(bands)
    ndvi = indices.ndvi(bands)
    water = mndwi > 0.0
    ndti_water = np.where(water, ndti, np.nan)

    n_pix = int(np.isfinite(bands["red"]).sum())
    n_water = int(np.nansum(water))
    print(f"  cloud-free pixels: {n_pix}  water (MNDWI>0): {n_water} "
          f"({100 * n_water / max(n_pix, 1):.1f}%)")
    print(f"  REAL NDTI turbidity proxy over water: "
          f"mean={np.nanmean(ndti_water):.4f}  std={np.nanstd(ndti_water):.4f}")

    # ---- figures ----
    dlabel = f"{meta['n_scenes']}-scene composite, {meta['dates'][0]} .."
    viz.raster_map(mndwi, extent,
                   f"Tennessee basin - MNDWI water index\nSentinel-2 L2A {dlabel}",
                   "MNDWI (>0 = water)", cmap="BrBG", vmin=-0.6, vmax=0.6,
                   out_path=os.path.join(OUT, "tn_water_mndwi.png"))
    viz.raster_map(ndti_water, extent,
                   f"Tennessee basin - NDTI turbidity proxy (water only)\n"
                   f"Real Sentinel-2 L2A {dlabel}",
                   "NDTI (turbidity proxy)", cmap="turbo", base=bands["red"],
                   out_path=os.path.join(OUT, "tn_turbidity_proxy.png"))

    # ---- per-pixel table + map ----
    import pandas as pd
    tbl = pd.DataFrame({
        "lat": g["lat"].ravel(), "lon": g["lon"].ravel(),
        "mndwi": mndwi.ravel(), "ndti": ndti.ravel(),
        "ndwi": ndwi.ravel(), "ndvi": ndvi.ravel(),
        **{c: bands[c].ravel() for c in ("blue", "green", "red", "nir", "swir1", "swir2")},
    }).dropna().reset_index(drop=True)
    tbl.to_csv(os.path.join(OUT, "tn_reflectance_indices.csv"), index=False)

    wtbl = tbl[tbl["mndwi"] > 0].copy()
    m = viz.value_points_map(wtbl, "ndti",
                             caption="NDTI turbidity proxy (real Sentinel-2 L2A)")
    m.save(os.path.join(OUT, "tn_basin_map.html"))

    meta.update({"n_pixels": n_pix, "n_water_pixels": n_water,
                 "ndti_water_mean": float(np.nanmean(ndti_water)),
                 "ndti_water_std": float(np.nanstd(ndti_water))})
    json.dump(meta, open(os.path.join(OUT, "tn_run_meta.json"), "w"), indent=2)

    print("\nwrote: tn_water_mndwi.png, tn_turbidity_proxy.png, tn_basin_map.html, "
          "tn_reflectance_indices.csv, tn_run_meta.json")
    print("REAL imagery over the TN basin - engine validated end to end.")


if __name__ == "__main__":
    main()
