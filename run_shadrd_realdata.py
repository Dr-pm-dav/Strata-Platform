"""
Reproduce the SHAD-RD turbidity result by running STRATA's engine on the real
in-situ + HLS data (CPU-only).

This is the dissertation pipeline expressed through STRATA's components:
    realdata.load_insitu_hls  ->  features.assemble(domain="water")
    ->  ShadrdModel (LightGBM)  ->  random holdout vs station-grouped SBCV.

It reproduces the central finding: a random split looks predictive, but
station-grouped spatial cross-validation (the honest estimate) collapses,
because repeated readings at one gauge leak between train and test.

Prithvi-EO-2.0-300M embeddings are deliberately omitted: they need the CUDA
workstation, and this path is CPU-only. The published v10 numbers below were
produced with Prithvi + the full 1058-feature set, so this spectral-only run
sits lower on the optimistic metric while showing the same leakage gap.

Run:  python run_shadrd_realdata.py --data /path/to/hls_spectral_v10.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from strata.shadrd import features, indices
from strata.shadrd.model import ShadrdModel
from strata.shadrd.spatial_cv import SpatialBlockCV
from strata.shadrd.realdata import load_insitu_hls, seasonal_terms

# Published v10 reference (Prithvi + 1058 features) for cross-checking.
V10 = {"holdout_r2": 0.4232, "holdout_rmse": 6.694, "sbcv_r2": -0.0762,
       "sbcv_rmse": 9.571, "tennessee_r2": 0.158, "flint_r2": 0.643}


def _rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def build_features(df):
    """STRATA water-domain indices + bands, plus the cheap engineered features
    the v10 ranking favours (colour ratios, brightness, day-of-year)."""
    Xi, names = features.assemble(df, domain="water", include_bands=True, prithvi=False)
    b = {k: df[k].to_numpy() for k in indices.BANDS}
    eps = 1e-6
    ratios = np.column_stack([
        b["green"] / (b["blue"] + eps), b["red"] / (b["blue"] + eps),
        b["nir"] / (b["red"] + eps), b["red"] / (b["green"] + eps),
        (b["blue"] + b["green"] + b["red"]) / 3.0])
    rnames = ["G_B", "R_B", "NIR_R", "R_G", "brightness"]
    seas, snames = seasonal_terms(df["date"])
    return np.column_stack([Xi, ratios, seas]), names + rnames + snames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hls_spectral_v10.csv")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_insitu_hls(args.data)
    X, names = build_features(df)
    y = df["turbidity"].to_numpy()
    yl = np.log1p(y)                       # v10 used a log transform (right-skewed)
    lat, lon = df["latitude"].to_numpy(), df["longitude"].to_numpy()
    groups, basin = df["site_id"].to_numpy(), df["basin"].to_numpy()

    print("=" * 70)
    print(" SHAD-RD turbidity model on real data, via STRATA engine (CPU, no Prithvi)")
    print("=" * 70)
    print(f"  samples {len(df)} | stations {df['site_id'].nunique()} | "
          f"features {X.shape[1]} | basins {dict(df['basin'].value_counts())}")

    # 1) random holdout 80/20 - the optimistic estimate
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    m = ShadrdModel(task="regression", feature_names=names)
    m.fit(X[tr], yl[tr])
    pred, obs = np.expm1(m.predict(X[te])), np.expm1(yl[te])
    r2_h, rmse_h = r2_score(obs, pred), _rmse(obs, pred)
    print(f"\n  [random holdout]   R2 = {r2_h:+.3f}   RMSE = {rmse_h:.3f} FNU")

    # per-basin on the holdout (cross-checks v10 basin_metrics)
    basin_rows = {}
    for bname in np.unique(basin[te]):
        msk = basin[te] == bname
        if msk.sum() >= 5:
            basin_rows[bname] = {"r2": float(r2_score(obs[msk], pred[msk])),
                                 "rmse": _rmse(obs[msk], pred[msk]), "n": int(msk.sum())}
            print(f"      {bname:10s} R2 = {basin_rows[bname]['r2']:+.3f}  "
                  f"RMSE = {basin_rows[bname]['rmse']:.3f}  (n={basin_rows[bname]['n']})")

    # 2) station-grouped spatial CV - the honest estimate (matches v10 sbcv_type)
    scv = SpatialBlockCV(n_splits=5)
    r2s, rmses = [], []
    for trf, tef in scv.split(lat, lon, groups=groups):
        mm = ShadrdModel(task="regression")
        mm.fit(X[trf], yl[trf])
        p, o = np.expm1(mm.predict(X[tef])), np.expm1(yl[tef])
        r2s.append(r2_score(o, p))
        rmses.append(_rmse(o, p))
    r2s, rmses = np.array(r2s), np.array(rmses)
    print(f"\n  [station SBCV]     R2 = {r2s.mean():+.3f} +/- {r2s.std():.3f}   "
          f"RMSE = {rmses.mean():.3f} +/- {rmses.std():.3f} FNU")
    print(f"  [leakage gap]      random {r2_h:+.3f} - SBCV {r2s.mean():+.3f} "
          f"= {r2_h - r2s.mean():.3f}")

    # cross-check against published v10
    print(f"\n  cross-check vs published v10 (Prithvi + 1058 feats):")
    print(f"      holdout R2  this {r2_h:+.3f}  |  v10 {V10['holdout_r2']:+.3f}")
    print(f"      SBCV R2     this {r2s.mean():+.3f}  |  v10 {V10['sbcv_r2']:+.3f}")
    reproduced = (r2_h > r2s.mean()) and (r2s.mean() < 0.1)
    print(f"      leakage reproduced (random positive, SBCV ~zero/negative): "
          f"{'YES' if reproduced else 'partial'}")

    # feature importance (full-data fit)
    mf = ShadrdModel(task="regression", feature_names=names)
    mf.fit(X, yl)
    fi = np.asarray(mf._model.feature_importances_, dtype="float64")
    top_idx = np.argsort(fi)[::-1][:10]
    top_features = [names[i] for i in top_idx]
    print("\n  top features:", ", ".join(top_features))

    # figure
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10, 4.2))
    for bname, c in [("tennessee", "#1f6feb"), ("flint", "#2ca02c")]:
        msk = basin[te] == bname
        axa.scatter(obs[msk], pred[msk], s=10, alpha=0.5, color=c, label=bname)
    lim = max(obs.max(), pred.max()) * 1.05
    axa.plot([0, lim], [0, lim], "k--", lw=1)
    axa.set_xlim(0, lim)
    axa.set_ylim(0, lim)
    axa.set_xlabel("observed turbidity (FNU)")
    axa.set_ylabel("predicted (FNU)")
    axa.set_title(f"Random holdout  (R2 = {r2_h:.2f})")
    axa.legend(fontsize=8, frameon=False)
    axa.grid(alpha=0.25)
    labels = ["random\nholdout", "station\nSBCV"]
    vals = [r2_h, r2s.mean()]
    axb.bar(labels, vals, color=["#1f6feb", "#d62728"])
    axb.axhline(0, color="k", lw=0.8)
    axb.set_ylabel("R2")
    axb.set_ylim(min(min(vals), 0) - 0.10, max(max(vals), 0) + 0.12)
    axb.set_title(f"Spatial leakage gap = {r2_h - r2s.mean():.2f}")
    for i, v in enumerate(vals):
        axb.text(i, v + (0.012 if v >= 0 else -0.012), f"{v:+.2f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=10)
    axb.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "shadrd_realdata.png"), dpi=200)
    plt.close(fig)

    results = {
        "samples": int(len(df)), "stations": int(df["site_id"].nunique()),
        "features": int(X.shape[1]), "prithvi_used": False, "log_transform": True,
        "holdout_r2": r2_h, "holdout_rmse": rmse_h,
        "sbcv_r2_mean": float(r2s.mean()), "sbcv_r2_std": float(r2s.std()),
        "sbcv_rmse_mean": float(rmses.mean()), "sbcv_type": "GroupKFold_by_station",
        "leakage_gap": float(r2_h - r2s.mean()),
        "basin_metrics": basin_rows, "top_features": top_features,
        "v10_reference": V10,
    }
    json.dump(results, open(os.path.join(args.out, "shadrd_realdata_results.json"), "w"),
              indent=2)
    print("\n  wrote: shadrd_realdata.png, shadrd_realdata_results.json")
    print("  NOTE: CPU spectral-only path; Prithvi-EO embeddings add features on "
          "the 4090 workstation (set prithvi=True in features.assemble there).")


if __name__ == "__main__":
    main()
