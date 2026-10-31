"""Offline pipeline for one pilot area (guide Phase 2). Run from backend/:

    python -m pipeline.run --area nabeul                 # everything
    python -m pipeline.run --area nabeul --from terrain  # resume (inputs already downloaded)
    python -m pipeline.run --area nabeul --only features # re-run after importing events / drainage data

Steps: osm -> rasters -> terrain -> features
Layout: data/raw/<area>/ (untouched downloads), data/interim/<area>/ (derived rasters),
        data/processed/<area>/ (what the API loads).
Look at every layer in QGIS (data/interim/<area>/*.tif and processed/<area>/segments_qa.gpkg)
before trusting it.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import yaml

from . import common, features, sources, terrain

log = logging.getLogger("pipeline")
ROOT = Path(__file__).resolve().parents[1]
STEPS = ["osm", "rasters", "terrain", "features"]


def _read(path: Path) -> gpd.GeoDataFrame | None:
    if not path.is_file():
        return None
    g = gpd.read_file(path)
    return None if g.empty else g


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="nabeul")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--from", dest="start", choices=STEPS, default="osm")
    ap.add_argument("--only", choices=STEPS)
    ap.add_argument("--stream-area-km2", type=float, default=0.5,
                    help="contributing area that starts a DEM stream (default 0.5 km2)")
    args = ap.parse_args()

    areas = yaml.safe_load((Path(__file__).parent / "areas.yaml").read_text(encoding="utf-8"))
    area = areas[args.area]
    bbox = area["bbox"]
    raw = args.data_dir / "raw" / args.area
    interim = args.data_dir / "interim" / args.area
    processed = args.data_dir / "processed" / args.area
    steps = [args.only] if args.only else STEPS[STEPS.index(args.start):]

    if "osm" in steps:
        sources.fetch_osm(bbox, raw)
    if "rasters" in steps:
        sources.fetch_rasters(bbox, raw)
    if "terrain" in steps:
        interim.mkdir(parents=True, exist_ok=True)
        dem_m = terrain.reproject_raster(raw / "dem_glo30.tif", interim / "dem_utm.tif", 30)
        terrain.reproject_raster(raw / "worldcover_2021.tif", interim / "worldcover_utm.tif", 10,
                                 resampling=terrain.Resampling.nearest)
        terrain.derive(dem_m, interim, _read(raw / "waterways.gpkg"), args.stream_area_km2 * 1e6)
    if "features" in steps:
        import osmnx as ox

        G = ox.load_graphml(raw / "graph.graphml")
        D = common.collapse_to_digraph(G)
        seg = common.build_segments(D)
        log.info("%d segments from %d edges", len(seg), D.number_of_edges())
        rasters = {k: interim / f"{k}.tif" for k in
                   ["depth_in_sink", "slope", "twi", "hand", "dist_stream", "streams"]}
        seg = seg.join(features.terrain_measurements(seg, rasters))
        seg["impervious_share"] = features.impervious_share(seg, interim / "worldcover_utm.tif")
        # authority inlets (FR-10) take precedence over the sparse OSM ones
        drains = _read(raw / "drainage" / "inlets.geojson")
        if drains is None:
            drains = _read(raw / "drains.gpkg")
        seg["drain_count"] = features.drain_count(seg, drains)
        seg["dv_maintenance"] = features.maintenance_factor(seg, _read(raw / "drainage" / "maintenance.geojson"))
        events = common.load_events(processed / "events.geojson")
        seg, params = common.normalise(seg, events)
        common.write_area(processed, area["name"], D, seg, sources.SOURCES, params)
        log.info("wrote %s", processed)


if __name__ == "__main__":
    main()
