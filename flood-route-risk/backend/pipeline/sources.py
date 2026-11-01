"""Download inputs for a pilot area: OSM (OSMnx), Copernicus DEM GLO-30 and ESA WorldCover 2021.

Rasters are read as windows straight from the public AWS buckets (cloud-optimised GeoTIFFs,
no account needed), so only the pilot area is downloaded.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import geopandas as gpd
import networkx as nx
import rasterio
from rasterio.merge import merge
from shapely.geometry import LineString, box

log = logging.getLogger(__name__)

# Local OSM extract (Geofabrik .osm.pbf), read instead of the Overpass API.
OSM_PBF = Path(__file__).resolve().parents[1] / "data" / "osm" / "tunisia-latest.osm.pbf"

# Free-flow speeds (km/h) by highway class, used when maxspeed is missing.
SPEED_KMH = {
    "motorway": 110, "motorway_link": 80, "trunk": 90, "trunk_link": 60,
    "primary": 70, "primary_link": 50, "secondary": 60, "secondary_link": 45,
    "tertiary": 50, "tertiary_link": 40, "unclassified": 40, "residential": 30,
    "living_street": 10, "service": 20,
}

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


def _str_or_none(v) -> str | None:
    """Pyrosm returns strings, NaN or None; keep the first value of 'a;b' or merged lists."""
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).split(";")[0].strip()
    return s or None


def _speed_kmh(highway: str | None, maxspeed: str | None) -> float:
    if maxspeed and "mph" not in maxspeed:
        digits = "".join(ch for ch in maxspeed if ch.isdigit())
        if digits:
            return float(digits)
    return float(SPEED_KMH.get(highway or "", 30))


def _road_graph(osm) -> nx.MultiDiGraph | None:
    """Driving network from the local extract, in the same shape OSMnx produced."""
    try:
        nodes, edges = osm.get_network(network_type="driving", nodes=True)
    except ValueError:            # pyrosm raises when the area has no roads
        return None
    if nodes is None or edges is None or len(edges) == 0:
        return None

    G = nx.MultiDiGraph(crs="epsg:4326")
    for nid, geom in zip(nodes["id"], nodes.geometry):
        G.add_node(int(nid), x=float(geom.x), y=float(geom.y))

    # Metric length (UTM 32N covers all of Tunisia)
    lengths = edges.to_crs(32632).length.values
    for rec, length in zip(edges.assign(length_m=lengths).to_dict("records"), lengths):
        u, v = int(rec["u"]), int(rec["v"])
        if u == v or u not in G or v not in G:
            continue
        geom = rec["geometry"]
        hw = _str_or_none(rec.get("highway"))
        ms = _str_or_none(rec.get("maxspeed"))
        attrs = {"length": float(length)}   # travel_time is set after simplifying
        for key, val in (("name", _str_or_none(rec.get("name"))), ("highway", hw), ("maxspeed", ms),
                         ("bridge", _str_or_none(rec.get("bridge"))),
                         ("tunnel", _str_or_none(rec.get("tunnel"))),
                         ("ford", _str_or_none(rec.get("ford"))),
                         ("layer", _str_or_none(rec.get("layer")))):
            if val is not None:
                attrs[key] = val

        oneway = (_str_or_none(rec.get("oneway")) or "no").lower()
        if oneway in ("yes", "true", "1"):
            G.add_edge(u, v, geometry=geom, **attrs)
        elif oneway == "-1":
            G.add_edge(v, u, geometry=LineString(list(geom.coords)[::-1]), **attrs)
        else:
            G.add_edge(u, v, geometry=geom, **attrs)
            G.add_edge(v, u, geometry=LineString(list(geom.coords)[::-1]), **attrs)
    return G


def _features(osm, custom_filter: dict, keep_nodes: bool) -> gpd.GeoDataFrame:
    try:
        gdf = osm.get_data_by_custom_criteria(
            custom_filter=custom_filter, filter_type="keep",
            keep_nodes=keep_nodes, keep_ways=True, keep_relations=False)
    except ValueError:            # nothing matched in this area
        gdf = None
    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    cols = [c for c in ("waterway", "intermittent", "name") if c in gdf.columns]
    return gdf[["geometry"] + cols].reset_index(drop=True)


def fetch_osm(bbox, raw_dir: Path) -> None:
    import osmnx as ox
    from pyrosm import OSM

    if not OSM_PBF.is_file():
        raise FileNotFoundError(f"OSM extract not found: {OSM_PBF}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    osm = OSM(str(OSM_PBF), bounding_box=list(bbox))

    G = _road_graph(osm)
    if G is None or G.number_of_edges() == 0:
        log.info("no roads in this area")
        return
    # Merge degree-2 nodes into single edges, as OSMnx's simplify=True did in the Overpass
    # version. Without this the graph is ~4x larger and the segment load is far slower.
    G = ox.simplify_graph(G)
    for _, _, d in G.edges(data=True):
        hw = _str_or_none(d.get("highway"))
        ms = _str_or_none(d.get("maxspeed"))
        d["travel_time"] = float(d["length"]) / (_speed_kmh(hw, ms) * 1000 / 3600)
    ox.save_graphml(G, raw_dir / "graph.graphml")
    log.info("OSM graph: %d nodes, %d edges", len(G.nodes), len(G.edges))

    waterways = _features(osm, {"waterway": ["river", "stream", "canal", "drain", "ditch", "wadi"]},
                          keep_nodes=False)
    drains = _features(osm, {"man_made": ["storm_drain"], "manhole": ["drain"]}, keep_nodes=True)
    for name, gdf in (("waterways", waterways), ("drains", drains)):
        gdf.to_file(raw_dir / f"{name}.gpkg", driver="GPKG")
        log.info("OSM %s: %d features", name, len(gdf))


def fetch_rasters(bbox, raw_dir: Path, margin: float = 0.02) -> None:
    w, s, e, n = bbox
    big = (w - margin, s - margin, e + margin, n + margin)  # margin for upstream flow
    window_mosaic(dem_urls(big), big, raw_dir / "dem_glo30.tif")
    window_mosaic(worldcover_urls(big), big, raw_dir / "worldcover_2021.tif")
