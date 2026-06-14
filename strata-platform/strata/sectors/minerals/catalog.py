"""
Critical-mineral site catalog.

A curated, version-controlled table of major U.S. critical-mineral sites with
approximate centroid coordinates, primary commodity, and operational status.
These are real, publicly documented sites; coordinates are approximate site
centroids suitable for regional analysis, not survey-grade.

For the full national inventory, ``mrds_connector`` reads the USGS Mineral
Resources Data System (MRDS) export and normalises it to the same schema, so
the platform can scale from this hand-checked seed set to the complete
database without changing any downstream code.

Commodity codes follow the DOE/USGS critical-minerals vocabulary:
    Li lithium · REE rare earths · Co cobalt · Ni nickel · Cu copper
    PGM platinum-group · Gr graphite · Mn manganese · Nb niobium · U uranium
"""
from __future__ import annotations

import pandas as pd

# name, state, lat, lon, commodity, status
_SITES = [
    ("Mountain Pass", "CA", 35.4775, -115.5328, "REE", "producing"),
    ("Thacker Pass", "NV", 41.7050, -118.0530, "Li", "construction"),
    ("Silver Peak (Clayton Valley)", "NV", 37.7500, -117.6360, "Li", "producing"),
    ("Kings Mountain", "NC", 35.2120, -81.3410, "Li", "development"),
    ("Stillwater", "MT", 45.3800, -109.8800, "PGM", "producing"),
    ("Bingham Canyon", "UT", 40.5230, -112.1510, "Cu", "producing"),
    ("Resolution", "AZ", 33.3000, -111.1000, "Cu", "development"),
    ("Morenci", "AZ", 33.0500, -109.3600, "Cu", "producing"),
    ("Bagdad", "AZ", 34.5800, -113.2000, "Cu", "producing"),
    ("Bear Lodge", "WY", 44.5000, -104.4500, "REE", "development"),
    ("Round Top", "TX", 30.9900, -105.3400, "REE", "development"),
    ("Pea Ridge", "MO", 38.1200, -91.0500, "REE", "past-producing"),
    ("Elk Creek", "NE", 40.2900, -96.1300, "Nb", "development"),
    ("Graphite Creek", "AK", 64.9000, -161.1000, "Gr", "development"),
    ("Hermosa (Taylor)", "AZ", 31.4500, -110.5500, "Mn", "development"),
    ("Eagle Mine", "MI", 46.7700, -87.9300, "Ni", "producing"),
]

CATALOG_COLUMNS = ["name", "state", "lat", "lon", "commodity", "status"]


def load_catalog() -> pd.DataFrame:
    """Return the curated critical-mineral site catalog as a DataFrame."""
    return pd.DataFrame(_SITES, columns=CATALOG_COLUMNS)


def sites_for_commodity(commodity: str) -> pd.DataFrame:
    """Filter the catalog to a single commodity code (e.g. 'Li')."""
    cat = load_catalog()
    return cat[cat["commodity"] == commodity].reset_index(drop=True)


def bbox_of(catalog: pd.DataFrame, pad: float = 1.0):
    """Return (min_lon, min_lat, max_lon, max_lat) padded by ``pad`` degrees."""
    return (
        float(catalog["lon"].min() - pad),
        float(catalog["lat"].min() - pad),
        float(catalog["lon"].max() + pad),
        float(catalog["lat"].max() + pad),
    )


def mrds_connector(csv_path: str) -> pd.DataFrame:  # pragma: no cover - workstation
    """Normalise a USGS MRDS export to the catalog schema.

    Download the MRDS CSV from
    https://mrdata.usgs.gov/mrds/  (or the Earth MRI focus-area data) and point
    this at it. Column names below match the standard MRDS export; adjust if
    USGS revises the schema.
    """
    raw = pd.read_csv(csv_path)
    out = pd.DataFrame({
        "name": raw.get("site_name", raw.get("name")),
        "state": raw.get("state"),
        "lat": pd.to_numeric(raw.get("latitude"), errors="coerce"),
        "lon": pd.to_numeric(raw.get("longitude"), errors="coerce"),
        "commodity": raw.get("commod1", raw.get("commodity")),
        "status": raw.get("dev_stat", "unknown"),
    })
    return out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
