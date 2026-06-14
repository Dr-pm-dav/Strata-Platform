"""
SHAD-RD engine: the sector-agnostic remote-sensing ML core.

Originally built for satellite water-quality regression (Yates dissertation,
HLS v2.0 + Prithvi-EO + LightGBM + HUC-12 spatially-blocked CV), the engine
is factored here so the exact same components drive critical-minerals
prospecting, mine-footprint monitoring, and clean-firm-power siting. The
cross-sector reuse is the engineering thesis of the STRATA platform.

Public surface:
    indices            spectral index library (water / mineral / disturbance)
    features.assemble  reflectance table -> (feature matrix, names)
    ShadrdModel        LightGBM wrapper with honest spatial-CV evaluation
    SpatialBlockCV     spatially-blocked CV splitter
    leakage_report     random vs spatial-CV gap quantifier
    HLSConnector       real HLS access (workstation)
    synthetic_scene    deterministic offline reflectance generator
"""
from . import features, indices
from .imagery import (
    HLSConnector, SentinelL2AConnector, reflectance_at, synthetic_scene,
)
from .model import ShadrdModel
from .spatial_cv import SpatialBlockCV, leakage_report, make_blocks

__all__ = [
    "indices",
    "features",
    "ShadrdModel",
    "SpatialBlockCV",
    "leakage_report",
    "make_blocks",
    "HLSConnector",
    "SentinelL2AConnector",
    "reflectance_at",
    "synthetic_scene",
]

__version__ = "0.1.0"
