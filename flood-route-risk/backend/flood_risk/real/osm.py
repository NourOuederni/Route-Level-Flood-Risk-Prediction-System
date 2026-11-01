"""Road segments and hydrography from OpenStreetMap through the Overpass API (FR-03, FR-06)."""

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, Point
from shapely.ops import substring

ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter", "https://overpass.private.coffee/api/interpreter"]
ROAD_TYPES = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|"
              "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")
SEGMENT_M = 150.0


def overpass(query: str, cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    last = None
    for attempt in range(6):
        url = ENDPOINTS[attempt % len(ENDPOINTS)]
        try:
            r = requests.post(url, data={"data": query}, timeout=900,
                              headers={"User-Agent": "route-flood-risk-research/0.1"})
            if r.status_code == 200:
                cache.write_text(r.text, encoding="utf-8")
                return r.json()
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"Overpass failed: {last}")


def _bbox(area, pad=0.0) -> str:
    return f"{area['south'] - pad},{area['west'] - pad},{area['north'] + pad},{area['east'] + pad}"


def _lines(elements) -> list:
    out = []
    for el in elements:
        if el["type"] == "way" and "geometry" in el:
            out.append((el, LineString([(p["lon"], p["lat"]) for p in el["geometry"]])))
        elif el["type"] == "relation":
            for m in el.get("members", []):
                if "geometry" in m and len(m["geometry"]) > 1:
                    out.append((el, LineString([(p["lon"], p["lat"]) for p in m["geometry"]])))
    return [(el, g) for el, g in out if len(g.coords) > 1]


def _is_underpass(tags) -> int:
    layer = tags.get("layer", "0")
    try:
        below = int(float(layer)) < 0
    except ValueError:
        below = False
    return int(tags.get("tunnel") in ("yes", "building_passage", "culvert") or tags.get("covered") == "yes" or below)


def fetch(area, cache_dir: Path):
    name = area["name"]
    roads = overpass(f'[out:json][timeout:800];way["highway"~"^({ROAD_TYPES})$"]({_bbox(area)});out body geom;',
                     cache_dir / f"{name}_roads.json")
    water = overpass(f'[out:json][timeout:800];(way["waterway"~"^(river|stream|canal|drain|ditch|wadi)$"]({_bbox(area, 0.03)});'
                     f'way["natural"~"^(water|coastline)$"]({_bbox(area, 0.03)});'
                     f'relation["natural"="water"]({_bbox(area, 0.03)}););out body geom;',
                     cache_dir / f"{name}_water.json")
    drains = overpass(f'[out:json][timeout:800];(node["manhole"~"drain|storm"]({_bbox(area)});'
                      f'node["man_made"="storm_drain"]({_bbox(area)});node["inlet"]({_bbox(area)}););out body;',
                      cache_dir / f"{name}_drains.json")
    return roads, water, drains


def build_segments(area, cache_dir: Path, clip=None) -> gpd.GeoDataFrame:
    """Split OSM roads into ~150 m segments with OSM-derived static factors, in the area's UTM CRS.

    `clip` (WGS84 geometry) keeps only segments whose midpoint falls inside it.
    """
    roads, water, drains = fetch(area, cache_dir)
    crs = area["crs"]

    ways = gpd.GeoDataFrame(
        [{"way_id": el["id"], "highway": el["tags"].get("highway"), "underpass": _is_underpass(el["tags"]),
          "ford": int(el["tags"].get("ford") == "yes"), "geometry": g} for el, g in _lines(roads["elements"])],
        crs="EPSG:4326").to_crs(crs)

    rows = []
    for w in ways.itertuples():
        n = max(1, int(round(w.geometry.length / SEGMENT_M)))
        step = w.geometry.length / n
        for i in range(n):
            g = substring(w.geometry, i * step, (i + 1) * step)
            if g.length >= 10:
                rows.append({"segment_id": f"{w.way_id}_{i}", "way_id": w.way_id, "highway": w.highway,
                             "underpass": w.underpass, "ford": w.ford, "length_m": g.length, "geometry": g})
    seg = gpd.GeoDataFrame(rows, crs=crs)
    seg["mid"] = seg.geometry.interpolate(0.5, normalized=True)
    if clip is not None:
        clip_utm = gpd.GeoSeries([clip], crs="EPSG:4326").to_crs(crs).iloc[0]
        seg = seg[seg["mid"].within(clip_utm)].reset_index(drop=True)

    water_lines = gpd.GeoSeries([g for _, g in _lines(water["elements"])], crs="EPSG:4326").to_crs(crs)
    if len(water_lines):
        idx = water_lines.sindex
        near, dist = idx.nearest(seg["mid"], return_distance=True, return_all=False)
        seg["dist_wadi_m"] = np.clip(dist, 0, 5000)
    else:
        seg["dist_wadi_m"] = 5000.0

    pts = [Point(el["lon"], el["lat"]) for el in drains["elements"] if el["type"] == "node"]
    if pts:
        drain_pts = gpd.GeoSeries(pts, crs="EPSG:4326").to_crs(crs)
        buf = gpd.GeoDataFrame(geometry=seg.geometry.buffer(50), crs=crs)
        hits = gpd.sjoin(gpd.GeoDataFrame(geometry=drain_pts), buf, predicate="within")
        counts = hits.groupby("index_right").size()
        seg["drain_inlet_density"] = seg.index.map(counts).fillna(0) / (seg["length_m"] / 1000)
    else:
        seg["drain_inlet_density"] = 0.0
    seg["drain_inlet_density"] = seg["drain_inlet_density"].clip(upper=60)
    return seg
