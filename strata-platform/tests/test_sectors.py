"""Sector contract tests."""
import numpy as np

from strata.sectors import minerals as M
from strata.sectors import supply as S
from strata.sectors import energy as E


# --------------------------------- minerals --------------------------------
def test_catalog_integrity():
    cat = M.load_catalog()
    assert len(cat) >= 12
    assert list(cat.columns) == M.CATALOG_COLUMNS
    assert cat["lat"].between(15, 72).all()      # within U.S. incl. Alaska
    assert cat["lon"].between(-170, -65).all()
    assert cat["commodity"].notna().all()


def test_prospectivity_balanced_and_ranks():
    train, deposits = M.build_training_frame(pos_per_site=40, neg_ratio=3, seed=0)
    assert train["label"].nunique() == 2
    assert train["label"].sum() > 0
    pm = M.ProspectivityModel(block_deg=2.0, n_splits=5).fit(train)
    grid = M.build_scoring_grid(step_deg=0.75)
    targets = pm.rank_targets(grid, top_n=6, min_sep_deg=1.0)
    assert len(targets) <= 6
    assert {"lat", "lon", "prospectivity"} <= set(targets.columns)
    # targets must not sit on a known deposit
    dep = np.array(deposits)
    for _, t in targets.iterrows():
        d = np.min(np.sqrt((t.lat - dep[:, 0]) ** 2 + (t.lon - dep[:, 1]) ** 2))
        assert d >= 0.4


def test_monitoring_flags_expansion():
    series = M.synth_footprint_series(seed=3)
    flags = M.detect_expansion(series)
    assert "is_expanding" in flags.columns
    # development/construction sites are seeded with a positive trend
    assert flags["is_expanding"].sum() >= 1


# --------------------------------- supply ----------------------------------
def test_gap_monotonic_in_demand():
    g_lo = S.rank_shortfalls(S.project_gap(demand_multiplier=1.0))
    g_hi = S.rank_shortfalls(S.project_gap(demand_multiplier=1.6))
    lo = g_lo.set_index("commodity")["terminal_gap"]
    hi = g_hi.set_index("commodity")["terminal_gap"].reindex(lo.index)
    assert (hi >= lo - 1e-6).all()


def test_priority_sites_resolve_to_catalog():
    ranking = S.rank_shortfalls(S.project_gap())
    sites = S.priority_sites(ranking, top_k=3)
    top_commodities = set(ranking.head(3)["commodity"])
    assert set(sites["commodity"]) <= top_commodities
    assert "shortfall_rank" in sites.columns


# --------------------------------- energy ----------------------------------
def test_siting_scores_bounded():
    grid = E.build_candidate_grid(step_deg=1.0)
    scored = E.score_sites(grid)
    assert scored["score"].between(0.0, 1.0).all()
    for comp in ("s_slope", "s_water", "s_transmission", "s_load",
                 "s_exclusion", "s_minerals_link"):
        assert scored[comp].between(0.0, 1.0).all()


def test_siting_ranking_separated():
    grid = E.build_candidate_grid(step_deg=1.0)
    scored = E.score_sites(grid)
    top = E.rank_sites(scored, top_n=8, min_sep_deg=1.5)
    assert len(top) <= 8
    pts = top[["lat", "lon"]].to_numpy()
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert (abs(pts[i, 0] - pts[j, 0]) >= 1.5
                    or abs(pts[i, 1] - pts[j, 1]) >= 1.5)
