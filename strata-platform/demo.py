"""
STRATA end-to-end demo (headless).

Runs all three sectors on the offline synthetic substrate, prints a console
report, and writes every output to ``outputs/``:

    outputs/leakage_minerals.png        random vs spatial-CV figure
    outputs/supply_demand_gap.png       gap trajectories
    outputs/footprint_monitoring.png    mine-footprint trends
    outputs/minerals_map.html           sites + prospective targets
    outputs/siting_map.html             clean-firm-power sites
    outputs/prospective_targets.csv     ranked new locales (lat/lon/score)
    outputs/shortfall_ranking.csv       strategic shortfall table
    outputs/priority_sites.csv          priority commodities -> specific sites
    outputs/siting_sites.csv            ranked siting coordinates

This is the proof the platform runs cover-to-cover with no network, no GPU,
and no credentials. Everything here uses the clearly-labelled SYNTHETIC
substrate; swap in the workstation connectors (HLS, USGS, NREL) for real
results. Run:  python demo.py
"""
from __future__ import annotations

import os

import pandas as pd

from strata.sectors import minerals as M
from strata.sectors import supply as S
from strata.sectors import energy as E
from strata import viz

OUT = os.path.join(os.path.dirname(__file__), "outputs")


def banner(t):
    print("\n" + "=" * 64 + f"\n {t}\n" + "=" * 64)


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 100)

    # ---- SECTOR 1: critical minerals -------------------------------------
    banner("SECTOR 1  Critical-mineral prospectivity & monitoring")
    catalog = M.load_catalog()
    print(f"catalog: {len(catalog)} real U.S. critical-mineral sites")

    train, _ = M.build_training_frame(pos_per_site=80, neg_ratio=3, seed=0)
    pm = M.ProspectivityModel(block_deg=2.0, n_splits=5).fit(train)
    r = pm.report
    print(f"prospectivity model ({r['metric'].upper()}):")
    print(f"  random K-fold   = {r['random_mean']:.3f} +/- {r['random_std']:.3f}"
          f"  ({r['n_random_folds']} folds)")
    print(f"  spatial-block CV= {r['spatial_mean']:.3f} +/- {r['spatial_std']:.3f}"
          f"  ({r['n_spatial_folds']} folds)")
    print(f"  leakage gap     = {r['leakage_gap']:+.3f}  "
          f"(random overstates honest skill by this much)")

    grid = M.build_scoring_grid(step_deg=0.5)
    targets = pm.rank_targets(grid, top_n=10, min_sep_deg=1.0)
    targets.to_csv(os.path.join(OUT, "prospective_targets.csv"), index=False)
    print("\ntop prospective NEW locales (excluding known deposits):")
    print(targets.head(5).round(3).to_string(index=False))

    series = M.synth_footprint_series()
    flags = M.detect_expansion(series)
    expanding = flags[flags["is_expanding"]]["name"].tolist()
    print(f"\nfootprint monitoring: {len(expanding)} sites flagged expanding")
    print("  " + ", ".join(expanding[:8]))

    viz.leakage_figure(r, out_path=os.path.join(OUT, "leakage_minerals.png"))
    viz.monitoring_figure(series, flags,
                          out_path=os.path.join(OUT, "footprint_monitoring.png"))

    # ---- SECTOR 2: supply-demand gap -------------------------------------
    banner("SECTOR 2  Critical-mineral supply-demand gap")
    proj = S.project_gap(years=10, demand_multiplier=1.3, supply_ramp=0.06)
    ranking = S.rank_shortfalls(proj)
    ranking.to_csv(os.path.join(OUT, "shortfall_ranking.csv"), index=False)
    print("strategic shortfall ranking (accelerated-transition scenario):")
    print(ranking[["commodity", "name", "terminal_gap",
                   "terminal_import_reliance", "shortfall_score"]]
          .round(1).to_string(index=False))

    psites = S.priority_sites(ranking, top_k=3)
    psites.to_csv(os.path.join(OUT, "priority_sites.csv"), index=False)
    print("\ntop-3 priority commodities resolve to specific sites:")
    print(psites[["name", "state", "commodity", "status", "shortfall_rank"]]
          .to_string(index=False))

    viz.gap_figure(proj, out_path=os.path.join(OUT, "supply_demand_gap.png"))

    # ---- SECTOR 3: clean-firm-power siting -------------------------------
    banner("SECTOR 3  Clean-firm-power siting")
    cand = E.build_candidate_grid(step_deg=0.5)
    scored = E.score_sites(cand)
    top_sites = E.rank_sites(scored, top_n=10, min_sep_deg=1.5)
    top_sites.to_csv(os.path.join(OUT, "siting_sites.csv"), index=False)
    print(f"scored {len(scored)} candidate cells; top recommended sites:")
    print(top_sites.head(5).round(3).to_string(index=False))

    # ---- maps ------------------------------------------------------------
    banner("Maps")
    priority_commodities = set(ranking.head(3)["commodity"])
    mmap = viz.minerals_map(catalog, targets=targets, priority=priority_commodities)
    mmap.save(os.path.join(OUT, "minerals_map.html"))
    smap = viz.siting_map(top_sites, mineral_sites=catalog)
    smap.save(os.path.join(OUT, "siting_map.html"))
    print("saved minerals_map.html and siting_map.html")

    banner("DONE")
    print(f"all outputs written to: {OUT}")
    print("NOTE: results use the SYNTHETIC offline substrate (clearly labelled).")
    print("Swap in HLS / USGS / NREL connectors on the workstation for real output.")


if __name__ == "__main__":
    main()
