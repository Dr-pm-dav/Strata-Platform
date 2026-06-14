"""
Spatially-blocked cross-validation for the SHAD-RD engine.

This is the methodological heart of the framework. Geospatial samples are
spatially autocorrelated: pixels (or sites) near each other carry nearly the
same information. A naive random train/test split therefore leaks
information across the split and reports an optimistic score that collapses
the moment the model sees a genuinely new area.

The original SHAD-RD work blocked by HUC-12 watershed. Here the blocking is
generalised: any caller can supply explicit group labels (e.g. HUC-12, a
geologic terrane, a 1-degree tile) OR let the splitter build square lat/lon
blocks of a chosen size. Folds are then assembled from whole blocks, so no
block is ever split between train and test.

``leakage_report`` runs the model under both random K-fold and spatial-block
CV and returns the gap between them - the honest measure of how much a random
split overstates performance. Surfacing that gap is itself a publishable
result and the reason the dissertation reported a negative spatial-CV R-square
where a random split looked strong.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_blocks(lat, lon, block_deg: float = 0.25) -> np.ndarray:
    """Assign each (lat, lon) to a square spatial block id.

    Block size is in decimal degrees. Returns an integer label per sample;
    samples in the same square share a label.
    """
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    iy = np.floor(lat / block_deg).astype("int64")
    ix = np.floor(lon / block_deg).astype("int64")
    # Cantor-style pairing into a single stable id.
    shift = ix - ix.min()
    return (iy - iy.min()) * (shift.max() + 1) + shift


@dataclass
class SpatialBlockCV:
    """K-fold over spatial blocks (whole blocks kept together).

    Parameters
    ----------
    n_splits : int
        Number of folds.
    block_deg : float
        Size of the lat/lon block in degrees when ``groups`` is not supplied.
    shuffle : bool
        Shuffle block order before partitioning into folds.
    random_state : int | None
        Seed for the block shuffle.
    """

    n_splits: int = 5
    block_deg: float = 0.25
    shuffle: bool = True
    random_state: int | None = 42

    def split(self, lat, lon, groups=None):
        """Yield (train_idx, test_idx) arrays.

        If ``groups`` is given it is used directly as the blocking label;
        otherwise blocks are built from ``lat``/``lon`` at ``block_deg``.
        """
        if groups is None:
            groups = make_blocks(lat, lon, self.block_deg)
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        if len(uniq) < self.n_splits:
            raise ValueError(
                f"only {len(uniq)} spatial blocks for {self.n_splits} folds; "
                "use a smaller block_deg or fewer splits"
            )
        rng = np.random.default_rng(self.random_state)
        if self.shuffle:
            rng.shuffle(uniq)
        fold_of_block = {b: i % self.n_splits for i, b in enumerate(uniq)}
        fold = np.array([fold_of_block[g] for g in groups])
        idx = np.arange(len(groups))
        for k in range(self.n_splits):
            test = idx[fold == k]
            train = idx[fold != k]
            if len(test) and len(train):
                yield train, test


def leakage_report(model_factory, X, y, lat, lon, *, task="regression",
                   n_splits=5, block_deg=0.25, random_state=42):
    """Compare random K-fold against spatial-block CV.

    ``model_factory`` is a zero-arg callable returning a fresh estimator with
    ``.fit(X, y)`` and ``.predict(X)``. Returns a dict of per-scheme metrics
    plus the leakage gap. Metric is R-square for regression, ROC-AUC for
    classification.
    """
    import warnings

    from sklearn.metrics import r2_score, roc_auc_score

    X = np.asarray(X, dtype="float64")
    y = np.asarray(y)
    n = len(y)
    idx = np.arange(n)

    def _fit_predict(train, test):
        # A fold is only scorable for classification if both train and test
        # carry both classes; otherwise AUC is undefined and the fold is
        # skipped rather than reported as a misleading value.
        if task == "classification":
            if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
                return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = model_factory()
            m.fit(X[train], y[train])
            pred = _proba_or_pred(m, X[test], task)
        if task == "regression":
            return r2_score(y[test], pred)
        return roc_auc_score(y[test], pred)

    # Random K-fold (the optimistic baseline).
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(idx)
    folds = np.array_split(perm, n_splits)
    rand_scores = []
    for k in range(n_splits):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        s = _fit_predict(train, test)
        if s is not None:
            rand_scores.append(s)

    # Spatial-block CV (the honest estimate).
    scv = SpatialBlockCV(n_splits=n_splits, block_deg=block_deg,
                         random_state=random_state)
    spatial_scores = []
    for train, test in scv.split(lat, lon):
        s = _fit_predict(train, test)
        if s is not None:
            spatial_scores.append(s)

    if not rand_scores or not spatial_scores:
        raise ValueError(
            "no scorable folds; for classification this usually means the "
            "positive class is too sparse or too clustered for the chosen "
            "block_deg / n_splits"
        )
    rand_scores = np.array(rand_scores, dtype="float64")
    spatial_scores = np.array(spatial_scores, dtype="float64")
    return {
        "task": task,
        "metric": "r2" if task == "regression" else "roc_auc",
        "random_mean": float(rand_scores.mean()),
        "random_std": float(rand_scores.std()),
        "spatial_mean": float(spatial_scores.mean()),
        "spatial_std": float(spatial_scores.std()),
        "leakage_gap": float(rand_scores.mean() - spatial_scores.mean()),
        "n_random_folds": int(len(rand_scores)),
        "n_spatial_folds": int(len(spatial_scores)),
        "random_folds": rand_scores.tolist(),
        "spatial_folds": spatial_scores.tolist(),
    }


def _proba_or_pred(model, X, task):
    if task == "classification" and hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)
