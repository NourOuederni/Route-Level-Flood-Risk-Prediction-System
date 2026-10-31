"""V2 national pipeline: process all of Tunisia in ~0.5° tiles, load into PostGIS.

Usage (from backend/):
    python -m pipeline.national                     # all tiles
    python -m pipeline.national --tile 10.5_36.5    # one tile (debug)
    python -m pipeline.national --from terrain      # resume after OSM+rasters

Tiles are ~0.5° squares covering the Tunisia bounding box [7.5, 30.2, 11.6, 37.5].
Each tile runs the same pipeline as V1 (OSM, rasters, terrain, features, normalise)
then inserts segments into PostGIS.

The final step builds the national igraph routing graph from PostGIS.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import yaml

from . import common, features, sources, terrain
from .load_postgis import load_tile_segments, finalise_weather_cells

log = logging.getLogger("pipeline.national")

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["osm", "rasters", "terrain", "features"]

# Tunisia bounding box (WGS84)
TUNISIA_BBOX = [7.5, 30.2, 11.6, 37.5]
TILE_SIZE = 0.5          # degrees
TILE_OVERLAP = 0.02      # small overlap avoids missing roads at borders


def generate_tiles(bbox: list[float], step: float = TILE_SIZE) -> list[dict]:
    """Generate tile definitions covering the bbox."""
    w, s, e, n = bbox
    tiles = []
    lat = s
    while lat < n:
        lon = w
        while lon < e:
            tile_id = f"{lon:.1f}_{lat:.1f}"
            tiles.append({
                "tile_id": tile_id,
                "bbox": [
                    lon - TILE_OVERLAP,
                    lat - TILE_OVERLAP,
                    min(lon + step + TILE_OVERLAP, e + TILE_OVERLAP),
                    min(lat + step + TILE_OVERLAP, n + TILE_OVERLAP),
                ],
            })
            lon += step
        lat += step
    return tiles


def _read(path: Path) -> gpd.GeoDataFrame | None:
    if not path.is_file():
        return None
    g = gpd.read_file(path)
    return None if g.empty else g


def process_tile(tile: dict, data_dir: Path, start_step: str = "osm",
                 stream_area_m2: float = 500_000) -> gpd.GeoDataFrame | None:
    """Run the V1 pipeline for one tile, return the normalised segments GeoDataFrame."""
    tile_id = tile["tile_id"]
    bbox = tile["bbox"]
    raw = data_dir / "raw" / "national" / tile_id
    interim = data_dir / "interim" / "national" / tile_id
    processed = data_dir / "processed" / "national" / tile_id
    steps = STEPS[STEPS.index(start_step):]

    if "osm" in steps:
        try:
            sources.fetch_osm(bbox, raw)
        except Exception as exc:
            log.warning("tile %s: OSM fetch failed (%s), skipping", tile_id, exc)
            return None

    # Check if we have a graph
    if not (raw / "graph.graphml").is_file():
        log.info("tile %s: no road graph, skipping (likely sea/desert)", tile_id)
        return None

    if "rasters" in steps:
        sources.fetch_rasters(bbox, raw)

    if "terrain" in steps:
        interim.mkdir(parents=True, exist_ok=True)
        try:
            dem_m = terrain.reproject_raster(raw / "dem_glo30.tif", interim / "dem_utm.tif", 30)
            terrain.reproject_raster(raw / "worldcover_2021.tif", interim / "worldcover_utm.tif", 10,
                                     resampling=terrain.Resampling.nearest)
            terrain.derive(dem_m, interim, _read(raw / "waterways.gpkg"), stream_area_m2)
        except Exception as exc:
            log.warning("tile %s: terrain processing failed (%s)", tile_id, exc)
            return None

    if "features" in steps:
        import osmnx as ox

        G = ox.load_graphml(raw / "graph.graphml")
        D = common.collapse_to_digraph(G)

        if D.number_of_edges() == 0:
            log.info("tile %s: empty graph, skipping", tile_id)
            return None

        seg = common.build_segments(D)
        log.info("tile %s: %d segments from %d edges", tile_id, len(seg), D.number_of_edges())

        rasters = {k: interim / f"{k}.tif" for k in
                   ["depth_in_sink", "slope", "twi", "hand", "dist_stream", "streams"]}

        # Check that rasters exist
        missing = [k for k, v in rasters.items() if not v.is_file()]
        if missing:
            log.warning("tile %s: missing rasters %s, skipping", tile_id, missing)
            return None

        seg = seg.join(features.terrain_measurements(seg, rasters))
        seg["impervious_share"] = features.impervious_share(seg, interim / "worldcover_utm.tif")

        # Authority inlets take precedence over sparse OSM ones
        drains = _read(raw / "drainage" / "inlets.geojson")
        if drains is None:
            drains = _read(raw / "drains.gpkg")
        seg["drain_count"] = features.drain_count(seg, drains)
        seg["dv_maintenance"] = features.maintenance_factor(
            seg, _read(raw / "drainage" / "maintenance.geojson"))

        # Events for this tile
        events = common.load_events(processed / "events.geojson")
        seg, params = common.normalise(seg, events)

        # Write V1-style output for QA
        processed.mkdir(parents=True, exist_ok=True)
        common.write_area(processed, tile_id, D, seg, sources.SOURCES, params)

        return seg

    return None


def compute_susceptibility(seg: gpd.GeoDataFrame, cfg_path: Path | None = None) -> gpd.GeoDataFrame:
    """Pre-compute S and S' for every segment using the engine config.

    This is the key V2 optimisation: S' is static and computed once offline.
    At request time, only H (a single division) is needed per segment.
    """
    from app.engine import default_config

    cfg = default_config() if cfg_path is None else default_config()
    w = cfg.s_weights
    s_min = cfg.s_min

    s = seg.copy()
    # S = sum_i w_i * f_i
    S = (
        w["depression"] * s["f_depression"].clip(0, 1) +
        w["drainage_position"] * s["f_drainage_position"].clip(0, 1) +
        w["low_slope"] * s["f_low_slope"].clip(0, 1) +
        w["impervious"] * s["f_impervious"].clip(0, 1) +
        w["crossing"] * s["f_crossing"].clip(0, 1) +
        w["past_flood"] * s["f_past_flood"].clip(0, 1)
    ).clip(0, 1)
    s["susceptibility"] = S
    s["susceptibility_floored"] = s_min + (1.0 - s_min) * S
    return s


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--from", dest="start", choices=STEPS, default="osm")
    ap.add_argument("--tile", help="Process only this tile (e.g. 10.5_36.5)")
    ap.add_argument("--db-url", default=os.environ.get(
        "DATABASE_URL", "postgresql://flood:flood_dev_2024@localhost:5433/floodrisk"))
    ap.add_argument("--stream-area-km2", type=float, default=0.5)
    ap.add_argument("--skip-graph", action="store_true", help="Skip igraph build at the end")
    args = ap.parse_args()

    tiles = generate_tiles(TUNISIA_BBOX)
    log.info("Tunisia grid: %d tiles of %.1f°", len(tiles), TILE_SIZE)

    if args.tile:
        tiles = [t for t in tiles if t["tile_id"] == args.tile]
        if not tiles:
            log.error("tile %s not found", args.tile)
            return

    # Process tiles
    total_segments = 0
    successful_tiles = 0
    for i, tile in enumerate(tiles, 1):
        log.info("=== tile %d/%d: %s ===", i, len(tiles), tile["tile_id"])
        try:
            seg = process_tile(tile, args.data_dir, args.start, args.stream_area_km2 * 1e6)
            if seg is not None and len(seg) > 0:
                seg = compute_susceptibility(seg)
                load_tile_segments(seg, tile["tile_id"], args.db_url)
                total_segments += len(seg)
                successful_tiles += 1
                log.info("tile %s: loaded %d segments into PostGIS", tile["tile_id"], len(seg))
        except Exception:
            log.exception("tile %s failed", tile["tile_id"])

    log.info("=== done: %d tiles, %d total segments ===", successful_tiles, total_segments)

    # Assign weather cells
    finalise_weather_cells(args.db_url)
    log.info("weather cells assigned")

    # Build the national igraph
    if not args.skip_graph:
        from .build_graph import build_national_graph
        graph_path = args.data_dir / "processed" / "national" / "graph.pickle"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        n_vertices, n_edges = build_national_graph(args.db_url, graph_path)
        log.info("national graph: %d vertices, %d edges -> %s", n_vertices, n_edges, graph_path)


if __name__ == "__main__":
    main()
