"""
Mine-footprint monitoring.

Tracks the bare-ground / disturbance signal (BSI from the engine's index
library) at each catalog site over time and flags sites whose extraction
footprint is expanding. The expansion test is an ordinary-least-squares trend
on the index time series with a significance threshold, so each flag is
defensible and reproducible.

Offline, the per-site time series is synthesised with a deterministic trend +
seasonal + noise model so the detector can be exercised and unit-tested. On
the workstation, ``footprint_series`` is replaced by a call that reduces real
HLS BSI over a polygon buffer around each site - the detector code is
unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .catalog import load_catalog


def synth_footprint_series(catalog=None, *, periods=32, seed=3):
    """Deterministic synthetic BSI time series per site (offline demo).

    Sites in development/construction status are given a positive expansion
    trend; producing/past-producing sites are flatter. Returns long-format
    DataFrame: name, t (period index), bsi.
    """
    if catalog is None:
        catalog = load_catalog()
    rng = np.random.default_rng(seed)
    rows = []
    t = np.arange(periods)
    season = 0.03 * np.sin(2 * np.pi * t / 12.0)
    for _, s in catalog.iterrows():
        if s["status"] in ("construction", "development"):
            slope = rng.uniform(0.004, 0.010)       # expanding footprint
            base = rng.uniform(0.05, 0.12)
        else:
            slope = rng.uniform(-0.001, 0.0015)     # stable / mature
            base = rng.uniform(0.12, 0.22)
        noise = rng.normal(0, 0.02, periods)
        bsi = base + slope * t + season + noise
        for ti, val in zip(t, bsi):
            rows.append((s["name"], int(ti), float(val)))
    return pd.DataFrame(rows, columns=["name", "t", "bsi"])


def detect_expansion(series: pd.DataFrame, *, alpha=0.05, min_slope=0.002):
    """OLS trend per site; flag significant positive expansion.

    Returns one row per site: slope (BSI/period), p_value, r2, and an
    ``expanding`` boolean (slope >= min_slope and p_value < alpha).
    """
    out = []
    for name, g in series.groupby("name"):
        g = g.sort_values("t")
        lr = stats.linregress(g["t"], g["bsi"])
        out.append({
            "name": name,
            "slope_per_period": float(lr.slope),
            "p_value": float(lr.pvalue),
            "r2": float(lr.rvalue ** 2),
            "is_expanding": bool(lr.slope >= min_slope and lr.pvalue < alpha),
        })
    res = pd.DataFrame(out).sort_values("slope_per_period", ascending=False)
    return res.reset_index(drop=True)
