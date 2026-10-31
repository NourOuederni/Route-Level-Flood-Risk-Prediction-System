"""Download inputs for a pilot area: OSM (OSMnx), Copernicus DEM GLO-30 and ESA WorldCover 2021.

Rasters are read as windows straight from the public AWS buckets (cloud-optimised GeoTIFFs,
no account needed), so only the pilot area is downloaded.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.merge import merge
from shapely.geometry import box

log = logging.getLogger(__name__)

DEM_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/"
           "Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM/Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM.tif")
WORLDCOVER_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                  "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")

SOURCES = [
    "© OpenStreetMap contributors (ODbL)",
    "Copernicus DEM GLO-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 "
    "provided under COPERNICUS by the European Union and ESA; all rights reserved (adapted)",
    "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) "
    "processed by ESA WorldCover consortium (CC BY 4.0)",
]


def _ns(lat: int) -> str:
    return f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"


def _ew(lon: int, width: int = 3) -> str:
    return f"E{lon:0{width}d}" if lon >= 0 else f"W{-lon:0{width}d}"


def dem_urls(bbox) -> list[str]:
    w, s, e, n = bbox
    return [DEM_URL.format(lat=_ns(la), lon=_ew(lo))
            for la in range(math.floor(s), math.floor(n) + 1)
            for lo in range(math.floor(w), math.floor(e) + 1)]


def worldcover_urls(bbox) -> list[str]:
    w, s, e, n = bbox
    tiles = {(math.floor(la / 3) * 3, math.floor(lo / 3) * 3)
             for la in (s, n) for lo in (w, e)}
    return [WORLDCOVER_URL.format(tile=f"{_ns(la)}{_ew(lo)}") for la, lo in sorted(tiles)]


def window_mosaic(urls: list[str], bbox, out_path: Path) -> Path:
    srcs = []
    for u in urls:
        try:
            srcs.append(rasterio.open(u))
        except rasterio.errors.RasterioIOError:
            log.warning("tile not available (ocean or missing): %s", u)
    if not srcs:
        raise RuntimeError("no raster tiles could be opened for this area")
    arr, transform = merge(srcs, bounds=tuple(bbox))
    prof = srcs[0].profile | {"driver": "GTiff", "height": arr.shape[1], "width": arr.shape[2],
                              "transform": transform, "compress": "deflate", "tiled": False}
    prof.pop("blockxsize", None); prof.pop("blockysize", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr)
    for s in srcs:
        s.close()
    return out_path


def fetch_osm(bbox, raw_dir: Path) -> None:
    import osmnx as ox

    ox.settings.useful_tags_way = list(dict.fromkeys(
        list(ox.settings.useful_tags_way) + ["ford", "layer", "surface", "tunnel", "bridge", "maxspeed"]))
    poly = box(*bbox)
    raw_dir.mkdir(parents=True, exist_ok=True)
    G = ox.graph_from_polygon(poly, network_type="drive", simplify=True, retain_all=False)
    G = ox.add_edge_speeds(G)          # km/h from maxspeed / road type
    G = ox.add_edge_travel_times(G)    # seconds
    ox.save_graphml(G, raw_dir / "graph.graphml")
    log.info("OSM graph: %d nodes, %d edges", len(G.nodes), len(G.edges))

    for name, tags in {
        "waterways": {"waterway": ["river", "stream", "canal", "drain", "ditch", "wadi"]},
        "drains": {"man_made": "storm_drain", "manhole": "drain"},
    }.items():
        try:
            gdf = ox.features_from_polygon(poly, tags=tags)
            gdf = gdf[["geometry"] + [c for c in ("waterway", "intermittent", "name") if c in gdf.columns]]
        except Exception as exc:  # osmnx raises when nothing matches
            log.info("no %s found in OSM (%s)", name, exc)
            gdf = gpd.GeoDataFrame(geometry=[], crs=4326)
        gdf.reset_index(drop=True).to_file(raw_dir / f"{name}.gpkg", driver="GPKG")
        log.info("OSM %s: %d features", name, len(gdf))


def fetch_rasters(bbox, raw_dir: Path, margin: float = 0.02) -> None:
    w, s, e, n = bbox
    big = (w - margin, s - margin, e + margin, n + margin)  # margin for upstream flow
    window_mosaic(dem_urls(big), big, raw_dir / "dem_glo30.tif")
    window_mosaic(worldcover_urls(big), big, raw_dir / "worldcover_2021.tif")
