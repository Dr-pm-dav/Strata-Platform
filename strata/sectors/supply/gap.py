"""
Supply-demand gap scenario model.

Projects demand and domestic supply for each critical mineral over a horizon
and surfaces where the U.S. is most strategically short. Demand grows from a
clean-energy-driven CAGR (scaled by a scenario multiplier); domestic supply
grows from an assumed reshoring ramp. The gap, the import-reliance
trajectory, and a composite shortfall score rank the commodities.

Crucially, the model then LINKS each priority commodity back to the
mineral-site catalog, so the strategic answer ("lithium and rare earths are
the binding constraints") resolves to specific places to monitor and explore.
That hand-off from macro gap to map coordinates is the point of integrating
the supply sector with the minerals sector.

This is scenario analysis on illustrative parameters, not a forecast of
record. Swap in current USGS figures and your own demand scenarios to make it
decision-grade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .commodities import load_commodities


def project_gap(commodities=None, *, years=10, start_year=2026,
                demand_multiplier=1.0, supply_ramp=0.06):
    """Project demand, domestic supply, and the gap per commodity per year.

    Parameters
    ----------
    demand_multiplier : float
        Scales the baseline clean-energy demand CAGR. 1.0 = baseline,
        >1 = accelerated energy transition, <1 = slower.
    supply_ramp : float
        Assumed annual growth of domestic supply (reshoring).

    Returns long-format DataFrame: commodity, name, year, demand_index,
    domestic_index, gap_index, import_reliance.
    """
    if commodities is None:
        commodities = load_commodities()
    yrs = np.arange(years + 1)
    rows = []
    for _, c in commodities.iterrows():
        # index both demand and domestic supply to apparent consumption = 100
        nir = c["net_import_reliance"] / 100.0
        dom0 = 100.0 * (1 - nir)              # domestic share today
        dem_cagr = (c["baseline_demand_cagr"] / 100.0) * demand_multiplier
        for t in yrs:
            demand = 100.0 * (1 + dem_cagr) ** t
            domestic = dom0 * (1 + supply_ramp) ** t
            gap = max(demand - domestic, 0.0)
            reliance = 100.0 * gap / demand if demand > 0 else 0.0
            rows.append({
                "commodity": c["commodity"], "name": c["name"],
                "year": int(start_year + t),
                "demand_index": demand, "domestic_index": domestic,
                "gap_index": gap, "import_reliance": reliance,
            })
    return pd.DataFrame(rows)


def rank_shortfalls(projection: pd.DataFrame, commodities=None,
                    *, criticality_weight=None):
    """Rank commodities by a composite strategic-shortfall score.

    Score = terminal gap_index x terminal import_reliance/100 x criticality
    weight. Returns one row per commodity sorted high-to-low, with the terminal
    gap and reliance carried through for transparency.
    """
    if commodities is None:
        commodities = load_commodities()
    if criticality_weight is None:
        # heavier weight on minerals where a single foreign source dominates
        criticality_weight = {"REE": 1.4, "Gr": 1.3, "Li": 1.3, "Co": 1.2,
                              "Mn": 1.1, "Nb": 1.1, "Ni": 1.0, "Cu": 1.0}
    terminal = projection.sort_values("year").groupby("commodity").tail(1)
    out = []
    for _, r in terminal.iterrows():
        w = criticality_weight.get(r["commodity"], 1.0)
        score = r["gap_index"] * (r["import_reliance"] / 100.0) * w
        out.append({
            "commodity": r["commodity"], "name": r["name"],
            "terminal_gap": float(r["gap_index"]),
            "terminal_import_reliance": float(r["import_reliance"]),
            "criticality_weight": w,
            "shortfall_score": float(score),
        })
    return pd.DataFrame(out).sort_values("shortfall_score", ascending=False).reset_index(drop=True)


def priority_sites(ranking: pd.DataFrame, top_k=3):
    """Map the top-k priority commodities to specific catalog sites.

    Returns the catalog rows whose commodity is in the top-k shortfall list,
    annotated with the shortfall rank - the macro-to-map hand-off.
    """
    from ..minerals.catalog import load_catalog
    cat = load_catalog()
    top = ranking.head(top_k).reset_index(drop=True)
    rank_of = {c: i + 1 for i, c in enumerate(top["commodity"])}
    sel = cat[cat["commodity"].isin(rank_of)].copy()
    sel["shortfall_rank"] = sel["commodity"].map(rank_of)
    return sel.sort_values("shortfall_rank").reset_index(drop=True)
