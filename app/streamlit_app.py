"""
STRATA console - the shippable tool.

Three sectors, one map-driven interface:
  1. Critical minerals   - known sites, prospectivity targets, spatial-CV
                           leakage report, footprint monitoring.
  2. Supply-demand gap   - scenario sliders, gap trajectories, strategic
                           shortfall ranking, and the resolve-to-sites table.
  3. Clean-firm-power     - criterion weights, suitability map, ranked
                           siting   coordinates.

Run:  streamlit run app/streamlit_app.py

Everything here runs on the offline SYNTHETIC substrate (clearly flagged in
the UI). On the workstation, point the engine connectors at HLS / USGS / NREL
to make the outputs real.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from strata.sectors import minerals as M
from strata.sectors import supply as S
from strata.sectors import energy as E
from strata import viz

st.set_page_config(page_title="STRATA console", page_icon="*", layout="wide")


# ----------------------------- cached compute -----------------------------
@st.cache_data(show_spinner=False)
def fit_prospectivity(pos_per_site, neg_ratio, block_deg, seed):
    train, _ = M.build_training_frame(pos_per_site=pos_per_site,
                                      neg_ratio=neg_ratio, seed=seed)
    pm = M.ProspectivityModel(block_deg=block_deg, n_splits=5).fit(train)
    grid = M.build_scoring_grid(step_deg=0.5, seed=1)
    targets = pm.rank_targets(grid, top_n=12, min_sep_deg=1.0)
    return pm.report, targets


@st.cache_data(show_spinner=False)
def run_gap(years, demand_multiplier, supply_ramp):
    proj = S.project_gap(years=years, demand_multiplier=demand_multiplier,
                         supply_ramp=supply_ramp)
    ranking = S.rank_shortfalls(proj)
    return proj, ranking


@st.cache_data(show_spinner=False)
def run_siting(weight_items, step_deg):
    weights = dict(weight_items)
    cand = E.build_candidate_grid(step_deg=step_deg)
    scored = E.score_sites(cand, weights=weights)
    top = E.rank_sites(scored, top_n=10, min_sep_deg=1.5)
    return top


def show_map(m, height=460):
    components.html(m._repr_html_(), height=height)


# --------------------------------- header ---------------------------------
st.title("STRATA")
st.caption("Strategic Terrain & Resource Analytics - cross-sector geospatial "
           "intelligence on the SHAD-RD engine")
st.info("Demonstration mode: results use a clearly-labelled SYNTHETIC offline "
        "substrate. Connect HLS / USGS / NREL on the workstation for real "
        "output.", icon="*")

tab1, tab2, tab3 = st.tabs(
    ["Critical minerals", "Supply-demand gap", "Clean-firm-power siting"]
)

# ============================ TAB 1: minerals =============================
with tab1:
    st.subheader("Critical-mineral prospectivity & monitoring")
    c = st.columns(4)
    pos = c[0].slider("Positives / site", 30, 120, 80, 10)
    neg = c[1].slider("Negative ratio", 1, 5, 3)
    blk = c[2].slider("Spatial block (deg)", 1.0, 3.0, 2.0, 0.25)
    seed = c[3].number_input("Seed", 0, 999, 0)

    report, targets = fit_prospectivity(pos, neg, blk, int(seed))
    catalog = M.load_catalog()

    m1, m2, m3 = st.columns(3)
    m1.metric("Random K-fold AUC", f"{report['random_mean']:.3f}",
              help="Optimistic baseline")
    m2.metric("Spatial-block CV AUC", f"{report['spatial_mean']:.3f}",
              help="Honest, leakage-controlled estimate")
    m3.metric("Leakage gap", f"{report['leakage_gap']:+.3f}",
              delta=f"{-report['leakage_gap']:+.3f} vs honest",
              delta_color="inverse",
              help="How much a random split overstates skill")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Sites & prospective targets**")
        priority = set(S.rank_shortfalls(S.project_gap()).head(3)["commodity"])
        show_map(viz.minerals_map(catalog, targets=targets, priority=priority))
        st.caption("Circles: known deposits (colour = commodity). "
                   "Stars: model-ranked new prospects.")
    with right:
        st.markdown("**Random vs spatial-block CV**")
        st.pyplot(viz.leakage_figure(report))
        st.markdown("**Top prospective locales**")
        st.dataframe(targets.round(3), use_container_width=True, height=220)

    st.divider()
    st.markdown("**Footprint monitoring** (bare-ground expansion)")
    series = M.synth_footprint_series()
    flags = M.detect_expansion(series)
    fc1, fc2 = st.columns([2, 3])
    fc1.dataframe(flags.round(3), use_container_width=True, height=300)
    fc2.pyplot(viz.monitoring_figure(series, flags))

# ========================= TAB 2: supply-demand ===========================
with tab2:
    st.subheader("Critical-mineral supply-demand gap")
    sc = st.columns(3)
    horizon = sc[0].slider("Horizon (years)", 5, 20, 10)
    dmult = sc[1].slider("Demand scenario (x baseline CAGR)", 0.5, 2.0, 1.3, 0.1,
                         help="Energy-transition pace; >1 accelerates demand")
    ramp = sc[2].slider("Domestic supply ramp (annual)", 0.0, 0.20, 0.06, 0.01)

    proj, ranking = run_gap(horizon, dmult, ramp)

    g1, g2 = st.columns([3, 2])
    with g1:
        st.markdown("**Gap trajectories**")
        st.pyplot(viz.gap_figure(proj))
    with g2:
        st.markdown("**Strategic shortfall ranking**")
        st.dataframe(
            ranking[["commodity", "name", "terminal_gap",
                     "terminal_import_reliance", "shortfall_score"]].round(1),
            use_container_width=True, height=320,
        )

    st.divider()
    topk = st.slider("Resolve top-k priority commodities to sites", 1, 5, 3)
    psites = S.priority_sites(ranking, top_k=topk)
    st.markdown("**Priority commodities resolve to specific sites**")
    p1, p2 = st.columns([2, 3])
    p1.dataframe(
        psites[["name", "state", "commodity", "status", "shortfall_rank"]],
        use_container_width=True, height=300,
    )
    with p2:
        priority = set(ranking.head(topk)["commodity"])
        show_map(viz.minerals_map(M.load_catalog(), priority=priority), height=380)

# ============================ TAB 3: siting ===============================
with tab3:
    st.subheader("Clean-firm-power siting")
    st.caption("Multi-criteria suitability for SMR / enhanced-geothermal "
               "siting. The minerals-link weight rewards co-location with "
               "critical-mineral demand.")
    w = E.DEFAULT_WEIGHTS.copy()
    wc = st.columns(6)
    keys = ["slope", "water", "transmission", "load", "exclusion", "minerals_link"]
    for i, k in enumerate(keys):
        w[k] = wc[i].slider(k, 0.0, 0.5, float(E.DEFAULT_WEIGHTS[k]), 0.02)
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}  # renormalise

    top_sites = run_siting(tuple(sorted(w.items())), 0.5)
    s1, s2 = st.columns([3, 2])
    with s1:
        st.markdown("**Recommended sites** (with critical-mineral context)")
        show_map(viz.siting_map(top_sites, mineral_sites=M.load_catalog()))
    with s2:
        st.markdown("**Ranked siting coordinates**")
        st.dataframe(top_sites.round(3), use_container_width=True, height=420)

st.divider()
st.caption("SHAD-RD engine (HLS + Prithvi-EO + LightGBM + spatial-block CV) "
           "applied across minerals, supply, and energy. One engine, three "
           "sectors - the cross-domain transferability claim, runnable.")
