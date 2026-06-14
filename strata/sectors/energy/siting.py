"""
Clean-firm-power siting.

Multi-criteria suitability scoring for siting clean firm generation (small
modular reactors, enhanced geothermal) at specific coordinates. Each candidate
cell is scored 0..1 on transparent, separately-weighted criteria, then the
best-separated high scorers are returned as a ranked site list.

Criteria (offline proxies; replace with real layers on the workstation):
    slope          terrain workability                 (USGS 3DEP DEM)
    water          cooling-water proximity             (NHD / NWIS)
    transmission   distance to existing grid           (HIFLD transmission)
    load           distance to demand centres          (load / metros)
    exclusion      protected / urban avoidance         (PAD-US, census)
    minerals_link  co-location with critical-mineral    (this platform's
                   demand to cut new transmission       mineral catalog)

The minerals_link criterion is the cross-sector tie: clean firm power sited
near the mineral operations it will electrify reduces new transmission and
couples the energy build-out to the resource base. Weights are caller-supplied
so the tool can explore siting strategy interactively.

Scoring is deliberately multi-criteria rather than ML: there are no siting
"labels" to learn, and a transparent weighted score is the honest, auditable
choice for a decision tool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate major U.S. demand centres (metro load anchors), lat/lon.
LOAD_CENTERS = [
    (34.05, -118.24), (37.77, -122.42), (47.61, -122.33), (33.45, -112.07),
    (39.74, -104.99), (32.78, -96.80), (29.76, -95.37), (41.88, -87.63),
    (33.75, -84.39), (40.71, -74.01), (38.91, -77.04), (47.92, -97.03),
]

DEFAULT_WEIGHTS = {
    "slope": 0.18, "water": 0.18, "transmission": 0.20,
    "load": 0.18, "exclusion": 0.12, "minerals_link": 0.14,
}


def _minmax(x):
    x = np.asarray(x, dtype="float64")
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _nearest_dist(lat, lon, anchors):
    a = np.asarray(anchors, dtype="float64")
    return np.min(
        np.sqrt((np.asarray(lat)[:, None] - a[:, 0]) ** 2
                + (np.asarray(lon)[:, None] - a[:, 1]) ** 2),
        axis=1,
    )


def build_candidate_grid(bbox=(-124.0, 31.0, -103.0, 49.0), step_deg=0.5, seed=7):
    """Candidate cells over a region with offline proxy criterion layers.

    Returns a DataFrame: lat, lon, slope, water, transmission_dist,
    load_dist, exclusion. Proxy layers are deterministic functions of space
    plus light noise; swap in real raster lookups on the workstation.
    """
    rng = np.random.default_rng(seed)
    lons = np.arange(bbox[0], bbox[2], step_deg)
    lats = np.arange(bbox[1], bbox[3], step_deg)
    glon, glat = np.meshgrid(lons, lats)
    lat, lon = glat.ravel(), glon.ravel()
    n = lat.size

    # proxy terrain ruggedness (lower = flatter = better)
    slope = (np.abs(np.sin(lat / 2.0) * np.cos(lon / 2.5))
             + 0.15 * rng.random(n))
    # proxy water proximity field (higher = more water nearby = better)
    water = (0.5 + 0.5 * np.sin(lon / 3.0) * np.cos(lat / 4.0)
             + 0.1 * rng.random(n))
    # distances to grid (proxied by load centers) and to load
    transmission_dist = _nearest_dist(lat, lon, LOAD_CENTERS) + 0.1 * rng.random(n)
    load_dist = _nearest_dist(lat, lon, LOAD_CENTERS)
    # proxy exclusion (urban/protected) penalty near dense metros
    exclusion = np.exp(-(_nearest_dist(lat, lon, LOAD_CENTERS) ** 2) / (2 * 0.4 ** 2))

    return pd.DataFrame({
        "lat": lat, "lon": lon, "slope": slope, "water": water,
        "transmission_dist": transmission_dist, "load_dist": load_dist,
        "exclusion": exclusion,
    })


def score_sites(grid, mineral_sites=None, weights=None):
    """Compute a 0..1 suitability score per candidate cell.

    ``mineral_sites`` (DataFrame with lat/lon) powers the cross-sector
    minerals_link criterion; if None, the platform catalog is used.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if mineral_sites is None:
        from ..minerals.catalog import load_catalog
        mineral_sites = load_catalog()
    g = grid.copy()

    # normalise each criterion to 0..1 where 1 = most suitable
    s_slope = 1 - _minmax(g["slope"])                      # flatter better
    s_water = _minmax(g["water"])                          # more water better
    s_trans = 1 - _minmax(g["transmission_dist"])          # closer better
    s_load = 1 - _minmax(g["load_dist"])                   # closer better
    s_excl = 1 - _minmax(g["exclusion"])                   # away from urban better
    mlink = _nearest_dist(g["lat"], g["lon"],
                          list(zip(mineral_sites["lat"], mineral_sites["lon"])))
    s_mlink = 1 - _minmax(mlink)                           # closer to minerals better

    g["score"] = (
        weights["slope"] * s_slope
        + weights["water"] * s_water
        + weights["transmission"] * s_trans
        + weights["load"] * s_load
        + weights["exclusion"] * s_excl
        + weights["minerals_link"] * s_mlink
    )
    # keep components for explainability
    g["s_slope"], g["s_water"], g["s_transmission"] = s_slope, s_water, s_trans
    g["s_load"], g["s_exclusion"], g["s_minerals_link"] = s_load, s_excl, s_mlink
    return g


def rank_sites(scored, top_n=10, min_sep_deg=1.0):
    """Top-N suitable, spatially-separated candidate sites."""
    s = scored.sort_values("score", ascending=False)
    chosen = []
    for _, row in s.iterrows():
        if any(abs(row.lat - c[0]) < min_sep_deg and abs(row.lon - c[1]) < min_sep_deg
               for c in chosen):
            continue
        chosen.append((row.lat, row.lon, row.score))
        if len(chosen) >= top_n:
            break
    return pd.DataFrame(chosen, columns=["lat", "lon", "score"])
