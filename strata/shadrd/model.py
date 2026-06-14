"""
The SHAD-RD model wrapper.

A thin, sector-agnostic layer over LightGBM that standardises the train /
predict / evaluate loop and bakes in spatially-honest validation. Every
sector in the platform (mineral prospectivity classification, water-quality
regression, siting scoring) instantiates this same class - that single
shared estimator is what makes the cross-sector claim concrete rather than
rhetorical.

Design choices:
* LightGBM, because it was the production model in the dissertation and runs
  fast on commodity hardware (no GPU required for the gradient-boosted layer).
* ``evaluate`` always reports BOTH random K-fold and spatial-block CV so the
  leakage gap is never hidden. If the spatial score is poor, the wrapper
  says so; it does not quietly report the random score.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .spatial_cv import leakage_report

# LightGBM 4.x emits a benign UserWarning when fed plain arrays after the
# sklearn wrapper recorded feature names. We pass NumPy throughout by design,
# so silence just that message to keep operational output clean.
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)

_REG_DEFAULTS = dict(
    objective="regression", n_estimators=400, learning_rate=0.03,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=30, reg_lambda=1.0, n_jobs=-1, verbosity=-1,
)
_CLF_DEFAULTS = dict(
    objective="binary", n_estimators=400, learning_rate=0.03,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=30, reg_lambda=1.0, n_jobs=-1, verbosity=-1,
)


@dataclass
class ShadrdModel:
    """LightGBM estimator with spatial-CV evaluation.

    Parameters
    ----------
    task : str
        "regression" or "classification".
    feature_names : list[str] | None
        Used for the importance report.
    params : dict
        Overrides merged onto the task defaults.
    """

    task: str = "classification"
    feature_names: list[str] | None = None
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'")
        base = _REG_DEFAULTS if self.task == "regression" else _CLF_DEFAULTS
        self._params = {**base, **self.params}
        self._model = None

    # -- core sklearn-style API ------------------------------------------
    def _new(self):
        import lightgbm as lgb
        cls = lgb.LGBMRegressor if self.task == "regression" else lgb.LGBMClassifier
        return cls(**self._params)

    def fit(self, X, y):
        self._model = self._new()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(np.asarray(X, dtype="float64"), np.asarray(y))
        return self

    def predict(self, X):
        self._check()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._model.predict(np.asarray(X, dtype="float64"))

    def predict_proba(self, X):
        self._check()
        if self.task != "classification":
            raise AttributeError("predict_proba only for classification")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._model.predict_proba(np.asarray(X, dtype="float64"))

    def score_surface(self, X):
        """Return a 0..1 score per row (probability for clf, raw pred for reg)."""
        if self.task == "classification":
            return self.predict_proba(X)[:, 1]
        return self.predict(X)

    # -- honest evaluation ------------------------------------------------
    def evaluate(self, X, y, lat, lon, *, n_splits=5, block_deg=0.25):
        """Random vs spatial-block CV. Returns the full leakage report."""
        return leakage_report(
            self._new, X, y, lat, lon, task=self.task,
            n_splits=n_splits, block_deg=block_deg,
        )

    def importances(self):
        self._check()
        imp = np.asarray(self._model.feature_importances_, dtype="float64")
        names = self.feature_names or [f"f{i}" for i in range(len(imp))]
        order = np.argsort(imp)[::-1]
        return [(names[i], float(imp[i])) for i in order]

    def _check(self):
        if self._model is None:
            raise RuntimeError("model is not fitted; call .fit(X, y) first")
