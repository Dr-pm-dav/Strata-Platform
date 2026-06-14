"""
Critical-mineral commodity reference data.

Baseline figures for the supply/demand gap model: U.S. apparent supply, net
import reliance, and a clean-energy demand driver per commodity. The built-in
values are ILLUSTRATIVE round numbers derived from publicly reported USGS
Mineral Commodity Summaries magnitudes - they are here so the model runs out
of the box, not as figures of record. Refresh them from the latest USGS MCS
(https://www.usgs.gov/centers/national-minerals-information-center) via
``load_commodities(csv_path=...)`` before citing any result.

``net_import_reliance`` is the headline USGS metric: the share of apparent
consumption met by imports (0..100). High values are the strategic exposure
the U.S. critical-minerals policy is trying to reduce.

``eia_connector`` is the optional live-data path (energy-linked demand
drivers) and mirrors the EIA API v2 usage from the Helix project; it needs an
API key and is never required for the offline model.
"""
from __future__ import annotations

import pandas as pd

# commodity, name, us_supply_kt (relative magnitude), net_import_reliance %,
# primary foreign source, clean-energy demand driver
_COMMODITIES = [
    ("Li",  "Lithium",      5.0,   50, "Chile/Argentina", "EV & grid batteries"),
    ("REE", "Rare earths",  43.0,  95, "China",           "Permanent magnets (wind, EV motors)"),
    ("Co",  "Cobalt",       0.8,   76, "DR Congo/Norway", "Battery cathodes"),
    ("Ni",  "Nickel",       18.0,  48, "Canada/Indonesia","Battery cathodes, stainless"),
    ("Cu",  "Copper",       1100.0,41, "Chile/Mexico",    "Grid, electrification"),
    ("Gr",  "Graphite",     0.0,   100,"China",           "Battery anodes"),
    ("Mn",  "Manganese",    0.0,   100,"Gabon/S. Africa", "Battery chemistries, steel"),
    ("Nb",  "Niobium",      0.0,   100,"Brazil",          "HSLA steel, superalloys"),
]

COMMODITY_COLUMNS = [
    "commodity", "name", "us_supply_kt", "net_import_reliance",
    "import_source", "demand_driver",
]

# Illustrative clean-energy-driven demand growth (annual %, baseline scenario).
# Adjust with the scenario multiplier in the gap model.
_BASELINE_DEMAND_CAGR = {
    "Li": 14.0, "REE": 9.0, "Co": 7.0, "Ni": 8.0,
    "Cu": 4.0, "Gr": 13.0, "Mn": 8.0, "Nb": 5.0,
}


def load_commodities(csv_path: str | None = None) -> pd.DataFrame:
    """Return the commodity reference table.

    With ``csv_path`` given, load an override CSV (same columns) - the
    intended path for plugging in current USGS MCS figures.
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        missing = set(COMMODITY_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"override CSV missing columns: {sorted(missing)}")
        return df
    df = pd.DataFrame(_COMMODITIES, columns=COMMODITY_COLUMNS)
    df["baseline_demand_cagr"] = df["commodity"].map(_BASELINE_DEMAND_CAGR)
    return df


def baseline_demand_cagr() -> dict:
    """Illustrative baseline annual demand-growth rates by commodity (%)."""
    return dict(_BASELINE_DEMAND_CAGR)


def eia_connector(series_id: str, api_key: str | None = None):  # pragma: no cover
    """Optional live energy-demand driver via EIA API v2 (needs API key).

    Mirrors the Helix EIA usage. Returns a tidy DataFrame of period/value.
    """
    import os
    import requests
    key = api_key or os.environ.get("EIA_API_KEY")
    if not key:
        raise RuntimeError(
            "set EIA_API_KEY (env or arg) to pull live EIA series; the gap "
            "model runs fully offline without it"
        )
    url = f"https://api.eia.gov/v2/seriesid/{series_id}"
    r = requests.get(url, params={"api_key": key}, timeout=30)
    r.raise_for_status()
    rows = r.json()["response"]["data"]
    return pd.DataFrame(rows)
