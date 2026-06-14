"""Engine contract tests."""
import numpy as np
import pytest

from strata.shadrd import (
    indices, features, ShadrdModel, SpatialBlockCV, make_blocks,
    leakage_report, reflectance_at,
)


def _bands(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return {b: rng.uniform(0.05, 0.4, n) for b in indices.BANDS}


def test_indices_known_values():
    b = {"blue": np.array([0.1]), "green": np.array([0.2]),
         "red": np.array([0.1]), "nir": np.array([0.3]),
         "swir1": np.array([0.2]), "swir2": np.array([0.1])}
    # NDVI = (nir-red)/(nir+red) = (0.3-0.1)/(0.4) = 0.5
    assert indices.ndvi(b)[0] == pytest.approx(0.5, abs=1e-4)
    # MNDWI = (green-swir1)/(green+swir1) = 0/0.4 = 0
    assert indices.mndwi(b)[0] == pytest.approx(0.0, abs=1e-4)
    # iron_oxide = red/blue = 1.0
    assert indices.iron_oxide(b)[0] == pytest.approx(1.0, abs=1e-3)


def test_compute_domains():
    b = _bands()
    for dom, names in (("water", 4), ("mineral", 5), ("disturbance", 4)):
        out = indices.compute(b, domain=dom)
        assert len(out) == names
        for v in out.values():
            assert np.all(np.isfinite(v))


def test_compute_missing_band_raises():
    with pytest.raises(KeyError):
        indices.compute({"blue": np.array([0.1])}, domain="mineral")


def test_features_assemble_shapes():
    import pandas as pd
    df = pd.DataFrame({**{b: np.random.rand(20) for b in indices.BANDS}})
    X, names = features.assemble(df, domain="mineral", include_bands=True)
    assert X.shape == (20, 5 + 6)
    assert len(names) == X.shape[1]


def test_spatial_blocks_disjoint():
    rng = np.random.default_rng(1)
    lat = rng.uniform(30, 45, 400)
    lon = rng.uniform(-120, -100, 400)
    blocks = make_blocks(lat, lon, block_deg=1.0)
    scv = SpatialBlockCV(n_splits=5, block_deg=1.0, random_state=0)
    for train, test in scv.split(lat, lon):
        # disjoint indices
        assert set(train).isdisjoint(set(test))
        # and crucially no spatial block straddles the split
        assert set(blocks[train]).isdisjoint(set(blocks[test]))


def test_model_regression_and_classification():
    X = np.random.default_rng(0).normal(size=(120, 5))
    yreg = X[:, 0] * 2 + np.random.default_rng(1).normal(scale=0.1, size=120)
    m = ShadrdModel(task="regression").fit(X, yreg)
    assert m.predict(X).shape == (120,)

    ycls = (X[:, 0] > 0).astype(int)
    mc = ShadrdModel(task="classification").fit(X, ycls)
    assert mc.predict_proba(X).shape == (120, 2)
    assert mc.score_surface(X).shape == (120,)


def test_sentinel_connector_offline_contract():
    # structural / offline only - never hits the network in CI
    from strata.shadrd import SentinelL2AConnector
    c = SentinelL2AConnector(relax_ssl=True)
    # asset map targets the canonical engine band keys
    assert set(c.ASSET_MAP.values()) == set(indices.BANDS)
    # SCL keep-set excludes cloud(8,9), shadow(3), cirrus(10), nodata(0)
    assert set(c.SCL_KEEP).isdisjoint({0, 1, 2, 3, 8, 9, 10})
    assert hasattr(c, "reflectance_grid") and hasattr(c, "reflectance_table")
    assert c.last_meta is None


def test_leakage_report_structure():
    dep = [(40.0, -112.0), (37.0, -117.0), (44.0, -109.0)]
    refl = reflectance_at(
        np.random.default_rng(0).uniform(35, 46, 300),
        np.random.default_rng(1).uniform(-119, -104, 300),
        deposits=dep, seed=0,
    )
    X, names = features.assemble(refl, domain="mineral")
    y = (np.random.default_rng(2).random(300) > 0.5).astype(int)
    m = ShadrdModel(task="classification", feature_names=names)
    rep = m.evaluate(X, y, refl["lat"].to_numpy(), refl["lon"].to_numpy(),
                     n_splits=3, block_deg=2.0)
    for k in ("random_mean", "spatial_mean", "leakage_gap",
              "n_random_folds", "n_spatial_folds"):
        assert k in rep
    assert 0.0 <= rep["random_mean"] <= 1.0
