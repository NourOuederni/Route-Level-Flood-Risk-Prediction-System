"""Terrain features from the DEM (guide 5.4, Phase 2 step 2).

reproject -> depth in sink (BEFORE filling) -> fill/resolve flats -> flow direction
-> flow accumulation -> streams -> slope, TWI, HAND, distance to drainage.
The DEM is a 30 m surface model: use the outputs as broad indicators, not street detail.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import distance_transform_edt

from .common import METRIC_CRS

log = logging.getLogger(__name__)
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)


def reproject_raster(src_path: Path, dst_path: Path, res: float,
                     resampling: Resampling = Resampling.bilinear) -> Path:
    with rasterio.open(src_path) as src:
        transform, w, h = calculate_default_transform(src.crs, f"EPSG:{METRIC_CRS}", src.width,
                                                      src.height, *src.bounds, resolution=res)
        prof = src.profile | {"crs": f"EPSG:{METRIC_CRS}", "transform": transform, "width": w,
                              "height": h, "driver": "GTiff", "compress": "deflate"}
        prof.pop("blockxsize", None); prof.pop("blockysize", None); prof["tiled"] = False
        with rasterio.open(dst_path, "w", **prof) as dst:
            reproject(rasterio.band(src, 1), rasterio.band(dst, 1), src_transform=src.transform,
                      src_crs=src.crs, dst_transform=transform, dst_crs=dst.crs, resampling=resampling)
    return dst_path


def _write(path: Path, arr: np.ndarray, profile: dict) -> Path:
    prof = profile | {"dtype": "float32", "count": 1, "nodata": np.nan, "compress": "deflate", "tiled": False}
    prof.pop("blockxsize", None); prof.pop("blockysize", None)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)
    return path


def derive(dem_metric: Path, out_dir: Path, waterways: gpd.GeoDataFrame | None = None,
           stream_area_m2: float = 500_000) -> dict[str, Path]:
    """Compute all terrain rasters; returns name -> path."""
    if not hasattr(np, "in1d"):   # pysheds still calls np.in1d, removed in recent NumPy
        np.in1d = lambda a, b, *args, **kw: np.isin(a, b, *args, **kw).ravel()
    from pysheds.grid import Grid  # heavy import, only needed here

    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dem_metric) as src:
        profile = src.profile.copy()
        res = abs(src.transform.a)
        transform = src.transform
        shape = (src.height, src.width)

    grid = Grid.from_raster(str(dem_metric))
    dem = grid.read_raster(str(dem_metric))
    nodata_mask = ~np.isfinite(np.asarray(dem)) | (np.asarray(dem) <= -1000)

    pit = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit)
    depth_in_sink = np.asarray(flooded) - np.asarray(dem)          # BEFORE filling: keep the sinks
    inflated = grid.resolve_flats(flooded)
    fdir = grid.flowdir(inflated, dirmap=DIRMAP)
    acc = np.asarray(grid.accumulation(fdir, dirmap=DIRMAP), dtype=float)

    z = np.asarray(dem, dtype=float)
    gy, gx = np.gradient(z, res)
    slope_rad = np.arctan(np.hypot(gx, gy))
    slope_deg = np.degrees(slope_rad)
    twi = np.log(((acc + 1.0) * res) / (np.tan(slope_rad) + 0.001))

    threshold_cells = max(10, stream_area_m2 / (res * res))
    streams = np.asarray(acc >= threshold_cells)
    from pysheds.view import Raster
    stream_r = Raster(streams.astype(bool), viewfinder=fdir.viewfinder)
    hand = np.asarray(grid.compute_hand(fdir, inflated, stream_r, dirmap=DIRMAP), dtype=float)

    drainage_mask = streams.copy()
    if waterways is not None and not waterways.empty:
        wm = rasterize(((g, 1) for g in waterways.to_crs(METRIC_CRS).geometry if g is not None),
                       out_shape=shape, transform=transform, fill=0, all_touched=True).astype(bool)
        drainage_mask |= wm
    dist = distance_transform_edt(~drainage_mask) * res

    for arr in (depth_in_sink, slope_deg, twi, hand, dist):
        arr[nodata_mask] = np.nan
    stream_layer = drainage_mask.astype("float32")

    paths = {
        "depth_in_sink": _write(out_dir / "depth_in_sink.tif", depth_in_sink, profile),
        "slope": _write(out_dir / "slope.tif", slope_deg, profile),
        "twi": _write(out_dir / "twi.tif", twi, profile),
        "hand": _write(out_dir / "hand.tif", hand, profile),
        "dist_stream": _write(out_dir / "dist_stream.tif", dist, profile),
        "streams": _write(out_dir / "streams.tif", stream_layer, profile),
    }
    log.info("terrain: %d stream cells (threshold %.0f cells), max sink depth %.2f m",
             int(streams.sum()), threshold_cells, float(np.nanmax(depth_in_sink)))
    return paths
