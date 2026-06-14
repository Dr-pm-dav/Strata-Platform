"""
Imagery access for the SHAD-RD engine.

Two paths, by design:

1. ``HLSConnector`` - the real path for the RTX-4090 workstation. It queries
   NASA's CMR-STAC / Earthdata for HLS v2.0 granules over a bounding box and
   date range and reads the bands with rasterio/stackstac. These libraries
   (pystac_client, stackstac, rasterio) are heavy and are imported lazily, so
   the rest of the platform installs and runs without them. If they are
   absent the connector raises a clear, actionable message rather than
   failing obscurely.

2. ``synthetic_scene`` - the offline path. It fabricates a physically
   plausible reflectance table with embedded spatial structure so the whole
   pipeline (indices -> features -> model -> map) runs deterministically in
   CI, in this demo, and on any laptop with no network and no credentials.
   Synthetic reflectance is clearly labelled as such everywhere it surfaces;
   it is for exercising the code path, never for reporting real predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indices import BANDS

# NASA CMR-STAC endpoint for HLS (used by the real connector).
CMR_STAC_HLS = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
HLS_COLLECTIONS = ("HLSS30.v2.0", "HLSL30.v2.0")


class HLSConnector:
    """Real HLS v2.0 access (workstation path). Lazy heavy imports."""

    def __init__(self, earthdata_token: str | None = None):
        self.token = earthdata_token

    def _require(self):
        try:
            import pystac_client  # noqa: F401
            import stackstac  # noqa: F401
            import rasterio  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised on workstation
            raise ImportError(
                "HLS access needs the geospatial extras. On your workstation:\n"
                "    pip install pystac-client stackstac rasterio\n"
                "and set an Earthdata token. The rest of STRATA runs without "
                "these; use synthetic_scene() for offline development."
            ) from exc

    def search(self, bbox, start, end, max_cloud=20):  # pragma: no cover
        """Return a STAC ItemCollection of HLS granules over bbox/date range."""
        self._require()
        from pystac_client import Client
        client = Client.open(CMR_STAC_HLS)
        search = client.search(
            collections=list(HLS_COLLECTIONS),
            bbox=bbox, datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": max_cloud}},
        )
        return search.item_collection()

    def load_reflectance(self, bbox, start, end, max_cloud=20):  # pragma: no cover
        """Return a band-stacked xarray over bbox; bands renamed to canonical keys."""
        self._require()
        import stackstac
        items = self.search(bbox, start, end, max_cloud)
        if len(items) == 0:
            raise RuntimeError("no HLS granules matched; widen date or bbox")
        stack = stackstac.stack(items, assets=["B02", "B03", "B04", "B8A", "B11", "B12"])
        return stack  # caller reduces to a pixel table


class SentinelL2AConnector:
    """Credential-free REAL reflectance via Sentinel-2 L2A (AWS Open Data).

    HLSConnector needs an Earthdata token. This connector instead reads
    Sentinel-2 Level-2A surface reflectance from the AWS Open Data bucket via
    the Element84 earth-search STAC, which is public and needs no credentials.
    Sentinel-2 is the "S" half of HLS's harmonisation, so the canonical band
    keys and every downstream index are identical - the output is real surface
    reflectance, a drop-in for the synthetic substrate.

    Pipeline: STAC search -> least-cloudy scenes -> stackstac load -> SCL
    cloud/shadow mask -> per-pixel median composite -> scale to reflectance ->
    reproject pixel centres to lon/lat.

    On networks that intercept TLS (corporate proxies, some sandboxes), pass
    ``relax_ssl=True`` to disable GDAL's certificate check for these public
    reads. On a normal workstation leave it False.
    """

    STAC = "https://earth-search.aws.element84.com/v1"
    COLLECTION = "sentinel-2-l2a"
    # Sentinel-2 asset key -> canonical engine band key
    ASSET_MAP = {"blue": "blue", "green": "green", "red": "red",
                 "nir": "nir", "swir16": "swir1", "swir22": "swir2"}
    # SCL classes to KEEP (drop cloud, shadow, cirrus, nodata, saturated, dark)
    SCL_KEEP = (4, 5, 6, 7, 11)  # vegetation, bare, water, unclassified, snow

    def __init__(self, relax_ssl=False):
        self.relax_ssl = relax_ssl
        self.last_meta = None

    def _gdal_env(self):
        import os
        env = {"AWS_NO_SIGN_REQUEST": "YES",
               "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
               "CPL_VSIL_CURL_USE_HEAD": "NO", "VSI_CACHE": "TRUE"}
        if self.relax_ssl:
            env["GDAL_HTTP_UNSAFESSL"] = "YES"
        for k, v in env.items():
            os.environ.setdefault(k, v)

    def _require(self):
        try:
            import pystac_client  # noqa: F401
            import stackstac  # noqa: F401
            import rasterio  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Sentinel-2 access needs pystac-client, stackstac, rasterio:\n"
                "    pip install pystac-client stackstac rasterio\n"
                "Offline, use synthetic_scene() / reflectance_at()."
            ) from exc

    def search(self, bbox, start, end, max_cloud=5, max_scenes=5):
        """Return the ``max_scenes`` least-cloudy Sentinel-2 items over bbox."""
        self._require()
        from pystac_client import Client
        items = list(Client.open(self.STAC).search(
            collections=[self.COLLECTION], bbox=list(bbox),
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": max_cloud}}).items())
        return sorted(items, key=lambda x: x.properties["eo:cloud_cover"])[:max_scenes]

    def reflectance_grid(self, bbox, start, end, *, max_cloud=5, max_scenes=5,
                         resolution=150, epsg=32616, mask_clouds=True):
        """Load a cloud-masked median reflectance composite as 2D grids.

        Returns a dict: ``bands`` (canonical key -> 2D float array), ``lat``
        and ``lon`` (2D arrays), ``extent`` (lon/lat bounds for imshow), and
        ``meta``. Indices in the engine operate directly on these 2D arrays.
        """
        self._require()
        self._gdal_env()
        import warnings
        import stackstac
        from rasterio.warp import transform as warp_transform
        from rasterio.crs import CRS

        items = self.search(bbox, start, end, max_cloud, max_scenes)
        if not items:
            raise RuntimeError("no Sentinel-2 scenes matched; widen date/bbox/cloud")
        assets = list(self.ASSET_MAP) + (["scl"] if mask_clouds else [])
        arr = stackstac.stack(
            items, assets=assets, epsg=epsg, resolution=resolution,
            bounds_latlon=list(bbox), chunksize=512, dtype="float64",
            fill_value=np.nan, rescale=False).compute()

        data = arr.values
        ix = {b: i for i, b in enumerate(list(arr.band.values))}
        valid = (np.isin(data[:, ix["scl"], :, :], self.SCL_KEEP)
                 if mask_clouds else None)
        bands = {}
        for a, c in self.ASSET_MAP.items():
            bt = data[:, ix[a], :, :].copy()
            if mask_clouds:
                bt[~valid] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med = np.nanmedian(bt, axis=0)
            # Element84 Collection-1 DN are already offset-harmonised: apply
            # the 1e-4 scale only. The raster:bands offset=-0.1 tag must NOT be
            # applied here - raw green/red DN (~500-600, < 1000) would go
            # negative and clip to zero, corrupting every ratio index.
            bands[c] = np.clip(med * 0.0001, 0.0001, 1.0)

        ny, nx = bands["red"].shape
        X, Y = np.meshgrid(arr.x.values, arr.y.values)
        lon, lat = warp_transform(CRS.from_epsg(epsg), CRS.from_epsg(4326),
                                  X.ravel().tolist(), Y.ravel().tolist())
        lon = np.asarray(lon).reshape(ny, nx)
        lat = np.asarray(lat).reshape(ny, nx)
        meta = {"scenes": [i.id for i in items],
                "dates": sorted(str(i.datetime.date()) for i in items),
                "n_scenes": len(items), "epsg": epsg, "resolution_m": resolution,
                "bbox": list(bbox),
                "source": "Sentinel-2 L2A (AWS Open Data, Element84 earth-search)"}
        self.last_meta = meta
        return {"bands": bands, "lat": lat, "lon": lon,
                "extent": [float(lon.min()), float(lon.max()),
                           float(lat.min()), float(lat.max())], "meta": meta}

    def reflectance_table(self, bbox, start, end, *, dropna=True, **kw):
        """Real reflectance as a flat pixel table (drop-in for reflectance_at).

        Columns: lat, lon, blue, green, red, nir, swir1, swir2.
        """
        g = self.reflectance_grid(bbox, start, end, **kw)
        cols = {c: g["bands"][c].ravel() for c in self.ASSET_MAP.values()}
        tbl = pd.DataFrame({"lat": g["lat"].ravel(), "lon": g["lon"].ravel(), **cols})
        if dropna:
            tbl = tbl.dropna().reset_index(drop=True)
        self.last_meta = {**g["meta"], "n_pixels": int(len(tbl))}
        return tbl


def reflectance_at(lat, lon, deposits=None, seed=0, signal=0.6, regional=0.6):
    """Synthetic reflectance at explicit coordinates.

    Core offline substrate. Given arrays of ``lat``/``lon`` (and optional
    deposit locations), returns a reflectance DataFrame combining three
    components:

    * per-pixel noise (uncorrelated),
    * a deposit-driven alteration halo (the genuine, causal, transferable
      signal a prospectivity model *should* learn), scaled by ``signal``,
    * a low-frequency REGIONAL baseline (smooth in lat/lon), scaled by
      ``regional``. This stands in for the atmospheric / seasonal / sensor /
      regional-substrate variation that pervades real HLS reflectance.

    The regional term is why spatial CV matters: because deposits are
    clustered, a model under random CV can exploit the regional baseline as a
    location tell and post an inflated score, then fail when spatial CV holds
    out whole regions. Setting ``regional=0`` removes the confounder and the
    leakage gap collapses - a knob that makes the methodological point
    explicit and testable.

    NOTE: synthetic. Swap in HLSConnector.load_reflectance on the workstation.
    """
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    n = lat.size
    rng = np.random.default_rng(seed + int(abs(lat.sum() * 1000)) % 9973)

    base = {
        "blue": rng.uniform(0.05, 0.18, n),
        "green": rng.uniform(0.06, 0.22, n),
        "red": rng.uniform(0.07, 0.28, n),
        "nir": rng.uniform(0.15, 0.45, n),
        "swir1": rng.uniform(0.10, 0.38, n),
        "swir2": rng.uniform(0.06, 0.30, n),
    }

    if regional:
        # smooth regional baseline (deterministic in space); applied per band
        reg = (0.05 * np.sin(lat / 3.0) + 0.04 * np.cos(lon / 4.0)
               + 0.03 * np.sin((lat + lon) / 5.0))
        wb = {"blue": 0.6, "green": 0.7, "red": 0.9,
              "nir": 1.0, "swir1": 0.8, "swir2": 0.7}
        for bk in base:
            base[bk] = base[bk] + regional * wb[bk] * reg

    if deposits is not None and len(deposits):
        dep = np.asarray(deposits, dtype="float64")
        dists = np.sqrt((lat[:, None] - dep[:, 0]) ** 2 + (lon[:, None] - dep[:, 1]) ** 2)
        nearest = np.argmin(dists, axis=1)
        d = dists[np.arange(n), nearest]
        halo = np.exp(-(d ** 2) / (2 * 0.20 ** 2))  # ~0.2 deg alteration halo
        k = signal * halo

        # Per-deposit alteration STYLE, deterministic from the deposit's
        # coordinates. A common transferable component (iron-oxide / red
        # elevation) is shared by all deposits, so a model can generalise
        # *somewhat* across space. A cluster-specific component boosts a
        # different band per deposit, so the feature->label mapping is
        # non-stationary: random CV interpolates within seen clusters and
        # leaks; spatial CV extrapolates to unseen clusters and drops. This
        # non-stationarity is the realistic source of the leakage gap.
        style_band = np.array([
            ["green", "swir1", "swir2", "nir"][
                int(abs(la * 991 + lo * 757)) % 4
            ] for la, lo in deposits
        ])
        # shared transferable lift
        base["red"] = base["red"] * (1 + 0.6 * k)
        base["blue"] = base["blue"] * (1 - 0.25 * k)
        # cluster-specific lift on the deposit's style band
        for bname in ("green", "nir", "swir1", "swir2"):
            mask = (style_band[nearest] == bname)
            base[bname] = base[bname] * (1 + 0.9 * k * mask)

    df = pd.DataFrame({"lat": lat, "lon": lon, **base})
    for bk in BANDS:
        df[bk] = df[bk].clip(0.001, 0.7)
    return df


def synthetic_scene(n=2000, bbox=(-120.0, 35.0, -110.0, 45.0), seed=0,
                    deposits=None, signal=0.6):
    """Deterministic synthetic reflectance table over a bbox (random points)."""
    rng = np.random.default_rng(seed)
    lon = rng.uniform(bbox[0], bbox[2], n)
    lat = rng.uniform(bbox[1], bbox[3], n)
    return reflectance_at(lat, lon, deposits=deposits, seed=seed, signal=signal)
