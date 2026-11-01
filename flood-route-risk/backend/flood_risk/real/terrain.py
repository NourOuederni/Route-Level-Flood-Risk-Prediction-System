"""Terrain and land-cover factors per segment (FR-06) from open rasters.

- Copernicus DEM GLO-30 (30 m): elevation, slope, fill-sink depression depth, D8 flow
  accumulation and topographic wetness index.
- ESA WorldCover 2021 (10 m): share of built-up surface around the segment.

Only the window covering the area is read from the remote cloud-optimised GeoTIFFs.
"""

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from scipy import ndimage
from skimage.morphology import reconstruction

GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}


def dem_urls(area):
    urls = []
    for lat in range(math.floor(area["south"]), math.floor(area["north"]) + 1):
        for lon in range(math.floor(area["west"]), math.floor(area["east"]) + 1):
            tile = (f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_"
                    f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM")
            urls.append(f"https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif")
    return urls


def usgs3dep_urls(area):
    """USGS 3DEP 1/3 arc-second bare-earth DEM (~10 m, United States only), tiles named by NW corner."""
    urls = []
    for lat in range(math.ceil(area["south"]), math.ceil(area["north"]) + 1):
        for lon in range(math.ceil(-area["east"]), math.ceil(-area["west"]) + 1):
            tile = f"n{lat:02d}w{lon:03d}"
            urls.append("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/"
                        f"{tile}/USGS_13_{tile}.tif")
    return urls


def worldcover_urls(area):
    urls = []
    for lat in range(math.floor(area["south"] / 3) * 3, math.floor(area["north"] / 3) * 3 + 1, 3):
        for lon in range(math.floor(area["west"] / 3) * 3, math.floor(area["east"] / 3) * 3 + 1, 3):
            tile = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            urls.append("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                        f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
    return urls


def to_grid(urls, area, res, resampling, cache: Path, pad_m=600):
    """Read the area from remote tiles and resample it on a metric UTM grid."""
    if cache.exists():
        z = np.load(cache)
        return z["data"], tuple(z["origin"])
    left, bottom, right, top = transform_bounds("EPSG:4326", area["crs"], area["west"], area["south"],
                                                area["east"], area["north"])
    left, bottom, right, top = left - pad_m, bottom - pad_m, right + pad_m, top + pad_m
    width, height = int((right - left) / res), int((top - bottom) / res)
    dst_transform = from_origin(left, top, res, res)
    dst = np.full((height, width), np.nan, dtype="float32")
    pad_deg = 0.02
    with rasterio.Env(**GDAL_ENV):
        for url in urls:
            with rasterio.open(url) as src:
                win = from_bounds(area["west"] - pad_deg, area["south"] - pad_deg, area["east"] + pad_deg,
                                  area["north"] + pad_deg, src.transform).round_offsets().round_lengths()
                win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                data = src.read(1, window=win).astype("float32")
                part = np.full_like(dst, np.nan)
                reproject(data, part, src_transform=src.window_transform(win), src_crs=src.crs,
                          src_nodata=src.nodata, dst_transform=dst_transform, dst_crs=area["crs"],
                          dst_nodata=np.nan, resampling=resampling)
                dst = np.where(np.isnan(dst), part, dst)
    np.savez_compressed(cache, data=dst, origin=np.array([left, top]))
    return dst, (left, top)


def d8_accumulation(z: np.ndarray) -> np.ndarray:
    """Number of upstream cells draining through each cell (steepest-descent D8)."""
    h, w = z.shape
    zp = np.pad(z, 1, constant_values=np.inf)
    best = np.zeros((h, w))
    recv = np.full((h, w), -1, dtype=np.int64)
    rows, cols = np.arange(h)[:, None], np.arange(w)[None, :]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            drop = (z - zp[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]) / math.hypot(dy, dx)
            better = drop > best
            best = np.where(better, drop, best)
            recv = np.where(better, (rows + dy) * w + (cols + dx), recv)
    acc = np.ones(h * w)
    r = recv.ravel()
    for i in np.argsort(-z, axis=None):
        if r[i] >= 0:
            acc[r[i]] += acc[i]
    return acc.reshape(h, w)


def terrain_layers(dem: np.ndarray, res: float, window_m: float = 1000) -> dict:
    dem = np.where(np.isnan(dem), np.nanmin(dem), dem).astype("float64")
    seed = dem.copy()
    seed[1:-1, 1:-1] = dem.max()
    filled = reconstruction(seed, dem, method="erosion")
    gy, gx = np.gradient(dem, res)
    slope = np.hypot(gx, gy)
    # Tiny tilt towards the grid edge lets water cross the flats left by sink filling.
    h, w = dem.shape
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx]).astype(float)
    acc = d8_accumulation(filled + 1e-4 * edge)
    return {
        "elevation_rel_m": dem - ndimage.minimum_filter(dem, size=int(window_m / res) | 1),
        "slope_pct": 100 * slope,
        "depression_depth_m": filled - dem,
        "twi": np.log(acc * res / np.maximum(slope, 1e-3)),
    }


def sample(layer, origin, res, xs, ys):
    left, top = origin
    c = np.clip(((np.asarray(xs) - left) / res).astype(int), 0, layer.shape[1] - 1)
    r = np.clip(((top - np.asarray(ys)) / res).astype(int), 0, layer.shape[0] - 1)
    return layer[r, c]


def add_terrain(seg, area, cache_dir: Path):
    """Add elevation_rel_m, slope_pct, depression_depth_m, twi and impervious_frac to the segments."""
    name = area["name"]
    xs, ys = seg["mid"].x.to_numpy(), seg["mid"].y.to_numpy()

    if area.get("dem") == "usgs3dep":
        res = 10
        dem, origin = to_grid(usgs3dep_urls(area), area, res, Resampling.bilinear, cache_dir / f"{name}_dem10.npz")
        dem = np.where(np.isnan(dem), np.nanmin(dem), dem)
    else:
        res = 30
        dem, origin = to_grid(dem_urls(area), area, res, Resampling.bilinear, cache_dir / f"{name}_dem30.npz")
        # Copernicus DEM is a surface model: buildings make every street look like a trench.
        # A 150 m grey opening removes objects narrower than the window and approximates the ground.
        dem = np.where(np.isnan(dem), np.nanmin(dem), dem)
        dem = ndimage.grey_opening(dem, size=(5, 5))
    seg["dem_res_m"] = res
    layers = terrain_layers(dem, res)
    for col, layer in layers.items():
        seg[col] = sample(layer, origin, res, xs, ys)
    # Road segments are not drawn on pixels exactly: keep the strongest depression within ~75 m.
    seg["depression_depth_m"] = sample(ndimage.maximum_filter(layers["depression_depth_m"], size=int(150 / res) | 1),
                                       origin, res, xs, ys)

    wc, wc_origin = to_grid(worldcover_urls(area), area, 10, Resampling.nearest, cache_dir / f"{name}_worldcover10.npz")
    built = ndimage.uniform_filter((wc == 50).astype("float32"), size=21)   # ~100 m around the point
    seg["impervious_frac"] = sample(built, wc_origin, 10, xs, ys)
    return seg
