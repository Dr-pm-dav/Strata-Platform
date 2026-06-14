"""Critical-minerals sector: site catalog, prospectivity, footprint monitoring."""
from .catalog import (
    load_catalog, sites_for_commodity, bbox_of, mrds_connector, CATALOG_COLUMNS,
)
from .monitoring import synth_footprint_series, detect_expansion
from .prospectivity import (
    ProspectivityModel, build_training_frame, build_scoring_grid,
)

__all__ = [
    "load_catalog", "sites_for_commodity", "bbox_of", "mrds_connector",
    "CATALOG_COLUMNS", "ProspectivityModel", "build_training_frame",
    "build_scoring_grid", "synth_footprint_series", "detect_expansion",
]
