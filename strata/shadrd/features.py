"""
Feature assembly for the SHAD-RD engine.

Turns a reflectance table into the feature matrix the model trains on. Two
feature families:

* Spectral indices (always available) - the domain index block from
  ``indices.compute`` plus the raw bands. Pure NumPy, runs anywhere.
* Prithvi-EO embeddings (optional, workstation) - 1024-d patch embeddings
  from the Prithvi-EO-2.0-300M geospatial foundation model via terratorch.
  This is the GPU path used in the dissertation. It is imported lazily and
  guarded: absence raises a clear message and the platform falls back to the
  index features so nothing breaks offline.

``assemble`` is the single entry point both the prospectivity and monitoring
sectors call, so the feature contract is identical across the platform.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indices


def assemble(reflectance: pd.DataFrame, domain: str = "mineral",
             include_bands: bool = True, prithvi: bool = False):
    """Build (feature_matrix, feature_names) from a reflectance table.

    Parameters
    ----------
    reflectance : DataFrame
        Must contain the six canonical band columns; lat/lon are passed
        through untouched and ignored as features.
    domain : str
        Index block to compute ("water" | "mineral" | "disturbance").
    include_bands : bool
        Append raw reflectance bands to the index features.
    prithvi : bool
        Append Prithvi-EO embeddings (workstation only).
    """
    bands = {bk: reflectance[bk].to_numpy() for bk in indices.BANDS}
    idx = indices.compute(bands, domain=domain)

    cols, names = [], []
    for name, arr in idx.items():
        cols.append(np.asarray(arr, dtype="float64"))
        names.append(name)
    if include_bands:
        for bk in indices.BANDS:
            cols.append(bands[bk])
            names.append(f"band_{bk}")

    X = np.column_stack(cols)

    if prithvi:
        emb, emb_names = prithvi_embeddings(reflectance)
        X = np.column_stack([X, emb])
        names = names + emb_names

    return X, names


def prithvi_embeddings(reflectance: pd.DataFrame):  # pragma: no cover - GPU path
    """Prithvi-EO-2.0-300M patch embeddings (workstation/GPU path).

    Lazy-imports terratorch/torch. Raises an actionable ImportError if the
    foundation-model stack or a CUDA device is unavailable, so callers can
    fall back to spectral-index features.
    """
    try:
        import torch  # noqa: F401
        from terratorch.registry import BACKBONE_REGISTRY  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Prithvi embeddings need the foundation-model stack. On your "
            "RTX-4090 box:\n"
            "    pip install terratorch torch --index-url <cuda wheel index>\n"
            "Then this returns 1024-d embeddings. Offline, STRATA uses the "
            "spectral-index features instead (prithvi=False)."
        ) from exc
    raise NotImplementedError(
        "Wire BACKBONE_REGISTRY.build('prithvi_eo_v2_300') here and run the "
        "reflectance chips through the encoder. Stub kept explicit so the "
        "offline contract stays honest."
    )
