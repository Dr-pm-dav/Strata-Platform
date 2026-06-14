"""
Two-basin water-quality characterization from the real in-situ record.

Profiles the Tennessee and Flint River basins across the four continuously
monitored parameters present for both (USGS NWIS): pH, dissolved oxygen,
specific conductance, and turbidity. These are gauge-measured quantities, so
they are characterised here as a water-quality signature, NOT used as features
for the satellite turbidity model (doing so would leak in-situ truth).

With millions of readings per parameter, statistical significance is a foregone
conclusion, so the comparison reports robust descriptives (median, IQR) and an
effect size (Cliff's delta, from the Mann-Whitney U on a large sample) rather
than p-values.

Run:  python run_waterquality_profile.py --data /path/to/uploads
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# param key -> (display label, log-scale plot)
PARAMS = {
    "ph": ("pH", False),
    "dissolved_o2": ("Dissolved O2 (mg/L)", False),
    "specific_cond": ("Specific conductance (uS/cm)", True),
    "turbidity_fnu": ("Turbidity (FNU)", True),
}
BASINS = ["tennessee", "flint"]
COLORS = {"tennessee": "#1f6feb", "flint": "#2ca02c"}
SAMPLE = 80_000


def _load_values(data_dir, basin, param, positive_only):
    path = os.path.join(data_dir, f"{basin}_{param}.csv")
    v = pd.read_csv(path, usecols=["value"])["value"]
    v = pd.to_numeric(v, errors="coerce").dropna().to_numpy()
    if param == "ph":
        v = v[(v >= 0) & (v <= 14)]
    if positive_only:
        v = v[v > 0]
    elif param == "dissolved_o2":
        v = v[v >= 0]
    return v


def _cliffs_delta(a, b, rng):
    """Cliff's delta via Mann-Whitney U on a capped sample: 2U/(n*m) - 1."""
    aa = a if len(a) <= SAMPLE else rng.choice(a, SAMPLE, replace=False)
    bb = b if len(b) <= SAMPLE else rng.choice(b, SAMPLE, replace=False)
    U, _ = stats.mannwhitneyu(aa, bb, alternative="two-sided")
    return 2.0 * U / (len(aa) * len(bb)) - 1.0, aa, bb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/user-data/uploads")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    rows = []
    samples = {}     # (param, basin) -> plotting sample
    print("=" * 74)
    print(" Two-basin water-quality signature (real USGS in-situ record)")
    print("=" * 74)
    for param, (label, logscale) in PARAMS.items():
        data = {b: _load_values(args.data, b, param, logscale) for b in BASINS}
        delta, sa, sb = _cliffs_delta(data["tennessee"], data["flint"], rng)
        samples[(param, "tennessee")] = sa
        samples[(param, "flint")] = sb
        for b in BASINS:
            v = data[b]
            rows.append({
                "parameter": label, "basin": b, "n": len(v),
                "median": float(np.median(v)), "mean": float(v.mean()),
                "sd": float(v.std()), "p25": float(np.percentile(v, 25)),
                "p75": float(np.percentile(v, 75)),
                "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
            })
        mt = np.median(data["tennessee"])
        mf = np.median(data["flint"])
        mag = ("negligible" if abs(delta) < 0.147 else "small" if abs(delta) < 0.33
               else "medium" if abs(delta) < 0.474 else "large")
        rows[-2]["cliffs_delta_vs_flint"] = round(delta, 3)
        rows[-2]["effect"] = mag
        print(f"\n  {label}")
        print(f"    Tennessee  median {mt:8.2f}  (n={len(data['tennessee']):>9,})")
        print(f"    Flint      median {mf:8.2f}  (n={len(data['flint']):>9,})")
        print(f"    Cliff's delta (TN vs FL) = {delta:+.3f}  [{mag}]")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "waterquality_comparison.csv"), index=False)

    # figure: 2x2 boxplots, TN vs FL, log y where skewed
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
    for ax, (param, (label, logscale)) in zip(axes.ravel(), PARAMS.items()):
        bp = ax.boxplot([samples[(param, "tennessee")], samples[(param, "flint")]],
                        showfliers=False, patch_artist=True, widths=0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Tennessee", "Flint"])
        for patch, b in zip(bp["boxes"], BASINS):
            patch.set_facecolor(COLORS[b])
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
        if logscale:
            ax.set_yscale("log")
        ax.set_title(label)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Tennessee vs Flint water-quality signature (USGS in-situ; boxes = IQR, "
                 "whiskers 1.5xIQR)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(args.out, "waterquality_profile.png"), dpi=200)
    plt.close(fig)
    print("\n  wrote: waterquality_profile.png, waterquality_comparison.csv")


if __name__ == "__main__":
    main()
