"""
STRATA - Strategic Terrain & Resource Analytics.

A cross-sector geospatial-ML platform built on the SHAD-RD engine (Yates,
Doctor of Computer Science dissertation: HLS v2.0 + Prithvi-EO + LightGBM +
spatially-blocked cross-validation, originally for satellite water-quality
regression). STRATA carries that same engine across three resource- and
energy-security sectors:

    minerals  - critical-mineral site monitoring and prospectivity ("best
                locales"), with the dissertation's spatial-CV leakage
                discipline applied to mineral targeting.
    supply    - critical-mineral supply-demand gap scenarios that rank
                strategic shortfalls and resolve them to specific sites.
    energy    - multi-criteria siting for clean firm power, tied back to the
                mineral demand it electrifies.

The shared SHAD-RD engine across sectors is the cross-domain transferability
claim made concrete. See README.md and METHODS.md.
"""
from . import shadrd
from .shadrd import __version__

__all__ = ["shadrd", "__version__"]
