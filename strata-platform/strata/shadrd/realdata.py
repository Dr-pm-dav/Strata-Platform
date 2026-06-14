"""
Load the real SHAD-RD in-situ + HLS co-located table into engine-native form.

The dissertation pipeline matched each in-situ turbidity reading to its HLS
surface-reflectance observation, producing one row per (station, date) with the
six HLS bands and the measured turbidity. This loader maps that table's
Sentinel-2/HLS band names (B2, B3, B4, B8A, B11, B12) onto the engine's
canonical band keys (blue, green, red, nir, swir1, swir2) so the existing
``shadrd.features`` / ``shadrd.indices`` code runs on it unchanged.

It also provides the day-of-year seasonal terms that the v10 feature-importance
ranking shows are the strongest non-spectral predictors of turbidity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# HLS / Sentinel-2 band names -> engine canonical band keys
BAND_MAP = {"B2": "blue", "B3": "green", "B4": "red",
            "B8A": "nir", "B11": "swir1", "B12": "swir2"}

_REQUIRED = ["blue", "green", "red", "nir", "swir1", "swir2",
             "turbidity", "site_id", "basin", "latitude", "longitude", "date"]


def load_insitu_hls(path: str) -> pd.DataFrame:
    """Read the co-located CSV and return it with canonical band columns.

    Drops rows missing any band or the turbidity label.
    """
    df = pd.read_csv(path).rename(columns=BAND_MAP)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"co-located table missing columns: {missing}")
    bands = ["blue", "green", "red", "nir", "swir1", "swir2"]
    df = df.dropna(subset=bands + ["turbidity"]).reset_index(drop=True)
    return df


def seasonal_terms(dates) -> tuple[np.ndarray, list[str]]:
    """Day-of-year sine/cosine encoding (captures the turbidity seasonal cycle)."""
    doy = pd.to_datetime(dates).dt.dayofyear.to_numpy()
    ang = 2.0 * np.pi * doy / 365.25
    return np.column_stack([np.sin(ang), np.cos(ang)]), ["doy_sin", "doy_cos"]
