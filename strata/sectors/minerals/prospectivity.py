"""
Mineral prospectivity modelling - the "best locales" engine.

Given a set of known deposits and a region, this fits the SHAD-RD model to
discriminate prospective from background ground, then scores a dense grid and
ranks the most prospective *new* locations (those not already a known
deposit). It is the direct cross-sector transfer of the dissertation pipeline:
identical engine, identical spatial-CV discipline, different target.

Honesty contract
----------------
Offline, the geophysical/geochemical signal is provided by
``shadrd.synthetic_scene``, which embeds alteration-halo structure around
known deposits. That is a *demonstration* substrate for exercising the full
modelling path deterministically - it is NOT a real prospectivity prediction.
On the workstation, replace ``build_training_frame``'s reflectance source with
``HLSConnector.load_reflectance`` and append USGS geophysical layers
(magnetics, radiometrics) and Earth MRI geochemistry; the ranked output then
becomes a real exploration target list. Every public result is labelled with
its data provenance so the two are never confused.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...shadrd import ShadrdModel, features, reflectance_at
from .catalog import load_catalog, bbox_of


def _nearest_deposit_dist(lat, lon, deposits):
    dep = np.asarray(deposits, dtype="float64")
    return np.min(
        np.sqrt((np.asarray(lat)[:, None] - dep[:, 0]) ** 2
                + (np.asarray(lon)[:, None] - dep[:, 1]) ** 2),
        axis=1,
    )


def build_training_frame(catalog=None, *, pos_per_site=60, neg_ratio=3,
                         halo_deg=0.25, seed=0, reflectance_fn=None):
    """Assemble a labelled presence / pseudo-absence training table.

    Positives: ``pos_per_site`` points jittered around each known deposit
    (within ~``halo_deg``). Negatives: background points sampled across the
    region beyond 2x the halo, kept at ``neg_ratio`` x the positive count.
    This is the standard prospectivity sampling design and guarantees both
    classes are present and spatially spread for honest spatial CV.

    ``reflectance_fn(lat, lon)`` overrides the offline substrate on the
    workstation (return a band table for the given coordinates).
    Returns (DataFrame with lat/lon/label, deposit_list).
    """
    if catalog is None:
        catalog = load_catalog()
    deposits = list(zip(catalog["lat"], catalog["lon"]))
    dep = np.asarray(deposits, dtype="float64")
    bbox = bbox_of(catalog, pad=1.0)
    rng = np.random.default_rng(seed)

    # --- positives: jitter around each deposit ---
    jit = halo_deg / 2.0
    plat, plon = [], []
    for la, lo in deposits:
        plat.append(rng.normal(la, jit, pos_per_site))
        plon.append(rng.normal(lo, jit, pos_per_site))
    plat = np.concatenate(plat)
    plon = np.concatenate(plon)

    # --- negatives: background beyond 2x halo ---
    n_neg_target = int(len(plat) * neg_ratio)
    nlat, nlon = [], []
    tries = 0
    while len(nlat) < n_neg_target and tries < 50:
        cand_lat = rng.uniform(bbox[1], bbox[3], n_neg_target)
        cand_lon = rng.uniform(bbox[0], bbox[2], n_neg_target)
        d = _nearest_deposit_dist(cand_lat, cand_lon, deposits)
        keep = d > 2 * halo_deg
        nlat.extend(cand_lat[keep].tolist())
        nlon.extend(cand_lon[keep].tolist())
        tries += 1
    nlat = np.array(nlat[:n_neg_target])
    nlon = np.array(nlon[:n_neg_target])

    lat = np.concatenate([plat, nlat])
    lon = np.concatenate([plon, nlon])
    label = np.concatenate([np.ones(len(plat), int), np.zeros(len(nlat), int)])

    refl = (reflectance_fn(lat, lon) if reflectance_fn is not None
            else reflectance_at(lat, lon, deposits=deposits, seed=seed, signal=0.6))
    frame = refl.copy()
    frame["lat"] = lat
    frame["lon"] = lon
    frame["dist_deg"] = _nearest_deposit_dist(lat, lon, deposits)
    frame["label"] = label
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True), deposits


def build_scoring_grid(catalog=None, *, step_deg=0.25, seed=1, reflectance_fn=None):
    """Dense grid over the region, with the same reflectance substrate."""
    if catalog is None:
        catalog = load_catalog()
    bbox = bbox_of(catalog, pad=1.0)
    lons = np.arange(bbox[0], bbox[2], step_deg)
    lats = np.arange(bbox[1], bbox[3], step_deg)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    glat, glon = grid_lat.ravel(), grid_lon.ravel()
    deposits = list(zip(catalog["lat"], catalog["lon"]))
    refl = (reflectance_fn(glat, glon) if reflectance_fn is not None
            else reflectance_at(glat, glon, deposits=deposits, seed=seed, signal=0.6))
    refl = refl.copy()
    refl["lat"] = glat
    refl["lon"] = glon
    return refl


class ProspectivityModel:
    """Fit, evaluate, score, and rank prospective locations."""

    def __init__(self, domain="mineral", block_deg=0.75, n_splits=5):
        self.domain = domain
        self.block_deg = block_deg
        self.n_splits = n_splits
        self.model = None
        self.names = None
        self.report = None

    def fit(self, train_frame):
        X, names = features.assemble(train_frame, domain=self.domain)
        y = train_frame["label"].to_numpy()
        self.names = names
        self.model = ShadrdModel(task="classification", feature_names=names)
        self.report = self.model.evaluate(
            X, y, train_frame["lat"].to_numpy(), train_frame["lon"].to_numpy(),
            n_splits=self.n_splits, block_deg=self.block_deg,
        )
        self.model.fit(X, y)
        return self

    def score_grid(self, grid):
        X, _ = features.assemble(grid, domain=self.domain)
        out = grid[["lat", "lon"]].copy()
        out["prospectivity"] = self.model.score_surface(X)
        return out

    def rank_targets(self, grid, catalog=None, top_n=10, min_sep_deg=0.4):
        """Return top-N prospective locations excluding existing deposits.

        Greedy spatial de-duplication keeps targets at least ``min_sep_deg``
        apart so the list is a set of distinct prospects, not one hot pixel.
        """
        if catalog is None:
            catalog = load_catalog()
        scored = self.score_grid(grid).sort_values("prospectivity", ascending=False)
        deposits = list(zip(catalog["lat"], catalog["lon"]))
        chosen = []
        for _, row in scored.iterrows():
            # skip cells essentially on a known deposit
            if _nearest_deposit_dist([row.lat], [row.lon], deposits)[0] < min_sep_deg:
                continue
            if any(abs(row.lat - c[0]) < min_sep_deg and abs(row.lon - c[1]) < min_sep_deg
                   for c in chosen):
                continue
            chosen.append((row.lat, row.lon, row.prospectivity))
            if len(chosen) >= top_n:
                break
        return pd.DataFrame(chosen, columns=["lat", "lon", "prospectivity"])
