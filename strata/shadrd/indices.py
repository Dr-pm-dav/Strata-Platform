"""
Spectral index library for the SHAD-RD engine.

These are the physical features the engine derives from surface-reflectance
imagery (HLS v2.0 band naming). The same library serves multiple sectors:
the water-quality indices are the ones used in the original SHAD-RD
dissertation work; the mineral / bare-ground indices are the extension that
lets the engine cross into critical-minerals prospecting and mine monitoring.

All functions are pure NumPy and operate elementwise, so they run identically
on a single pixel, a flat table of pixels, or a full raster stack. A small
epsilon guards every ratio against divide-by-zero.

Reference HLS surface-reflectance bands (Sentinel-2 / Landsat harmonised):
    blue  -> B02      nir   -> B8A (narrow NIR, HLS-S30) / B05 (HLS-L30)
    green -> B03      swir1 -> B11 / B06
    red   -> B04      swir2 -> B12 / B07
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6

# Canonical band keys the engine expects after harmonisation.
BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")


def _ratio(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    return a / (b + EPS)


def _norm_diff(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    return (a - b) / (a + b + EPS)


# --- water / surface-water domain (original SHAD-RD lineage) ----------------

def ndvi(b):
    """Vegetation. Used as a masking layer across every sector."""
    return _norm_diff(b["nir"], b["red"])


def ndwi(b):
    """McFeeters NDWI - open water delineation."""
    return _norm_diff(b["green"], b["nir"])


def mndwi(b):
    """Modified NDWI (Xu) - water vs built/soil; SHAD-RD water masking."""
    return _norm_diff(b["green"], b["swir1"])


def ndti(b):
    """Normalised Difference Turbidity Index - the SHAD-RD water-quality target proxy."""
    return _norm_diff(b["red"], b["green"])


# --- mineral / alteration domain (critical-minerals extension) --------------

def iron_oxide(b):
    """Ferric-iron / gossan ratio (red/blue). High over oxidised, iron-stained ground."""
    return _ratio(b["red"], b["blue"])


def ferrous_iron(b):
    """Ferrous-iron ratio (swir1/nir)."""
    return _ratio(b["swir1"], b["nir"])


def clay_minerals(b):
    """Hydroxyl / clay-alteration ratio (swir1/swir2). Maps phyllic/argillic alteration."""
    return _ratio(b["swir1"], b["swir2"])


def gossan(b):
    """Simple gossan index combining iron staining and low vegetation."""
    return _ratio(b["swir1"], b["blue"])


# --- bare-ground / disturbance domain (mine-footprint monitoring) -----------

def bsi(b):
    """Bare Soil Index - exposed earth; the core mine-footprint monitoring signal."""
    num = (b["swir1"] + b["red"]) - (b["nir"] + b["blue"])
    den = (b["swir1"] + b["red"]) + (b["nir"] + b["blue"])
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    return num / (den + EPS)


def ndbi(b):
    """Normalised Difference Built-up Index - hard/altered surfaces."""
    return _norm_diff(b["swir1"], b["nir"])


# Domain -> ordered index functions. Sectors request a domain and get a
# reproducible, named feature block.
DOMAINS = {
    "water": {"ndvi": ndvi, "ndwi": ndwi, "mndwi": mndwi, "ndti": ndti},
    "mineral": {
        "iron_oxide": iron_oxide,
        "ferrous_iron": ferrous_iron,
        "clay_minerals": clay_minerals,
        "gossan": gossan,
        "ndvi": ndvi,
    },
    "disturbance": {"bsi": bsi, "ndbi": ndbi, "ndvi": ndvi, "mndwi": mndwi},
}

ALL_INDICES = {
    name: fn for domain in DOMAINS.values() for name, fn in domain.items()
}


def compute(bands: dict, domain: str = "mineral") -> dict:
    """Compute the named index block for a domain.

    Parameters
    ----------
    bands : dict
        Mapping of canonical band key -> array-like surface reflectance.
    domain : str
        One of ``DOMAINS`` ("water", "mineral", "disturbance").

    Returns
    -------
    dict[str, np.ndarray]
        Ordered index name -> computed array.
    """
    if domain not in DOMAINS:
        raise KeyError(f"unknown domain {domain!r}; choose from {sorted(DOMAINS)}")
    missing = [bk for bk in BANDS if bk not in bands]
    if missing:
        raise KeyError(f"missing bands {missing}; need {list(BANDS)}")
    return {name: fn(bands) for name, fn in DOMAINS[domain].items()}
