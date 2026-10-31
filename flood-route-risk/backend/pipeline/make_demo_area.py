"""Build a SYNTHETIC demo area that runs fully offline (tests, first demo, no downloads).

    python -m pipeline.make_demo_area            # -> data/processed/demo/
    FLOOD_AREA=demo uvicorn app.main:app --reload

It fakes a small grid town near Nabeul with a wadi valley, depressions, an underpass and a
ford, then runs the REAL terrain and feature code on synthetic rasters. Streets do not match
the real basemap: use `python -m pipeline.run --area nabeul` for real data.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from . import common, features, terrain

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("demo")

CENTRE = (36.452, 10.735)        # lat, lon (Nabeul)
N = 14                           # grid nodes per side
SPACING = 260.0                  # metres
STREETS_EW = ["Avenue Habib Bourguiba", "Rue de Tunis", "Rue Farhat Hached", "Avenue Hedi Nouira",
              "Rue de la Plage", "Rue des Potiers", "Avenue de la Republique", "Rue Mongi Slim",
              "Rue du Souk", "Avenue Taieb Mhiri", "Rue Ibn Khaldoun", "Rue d'Alger",
              "Rue de Kelibia", "Route de Hammamet"]
STREETS_NS = ["Rue de l'Oued", "Avenue Ali Belhouane", "Rue Jamel Abdennaceur", "Rue de Beni Khiar",
              "Avenue Mohamed V", "Rue des Orangers", "Rue de Grombalia", "Rue Ennakhil",
              "Avenue de l'Environnement", "Rue Ibn Rachiq", "Rue de Dar Chaabane", "Rue Errimel",
              "Rue de Korba", "Boulevard du 7 Novembre"]


def _grid_xy(cx: float, cy: float):
    off = (N - 1) * SPACING / 2
    return {(i, j): (cx - off + i * SPACING, cy - off + j * SPACING) for i in range(N) for j in range(N)}


def _wadi_y(x: float, cx: float, cy: float) -> float:
    """A meandering wadi crossing the town from west to east."""
    return cy - 300 + 220 * np.sin((x - cx) / 700.0)


def build(out_root: Path) -> Path:
    to_m = Transformer.from_crs(4326, common.METRIC_CRS, always_xy=True)
    to_ll = Transformer.from_crs(common.METRIC_CRS, 4326, always_xy=True)
    cx, cy = to_m.transform(CENTRE[1], CENTRE[0])
    xy = _grid_xy(cx, cy)

    # --- synthetic DEM and land cover --------------------------------------
    res = 30.0
    pad = 1500.0
    x0, y1 = cx - (N * SPACING) / 2 - pad, cy + (N * SPACING) / 2 + pad
    w = h = int(((N * SPACING) + 2 * pad) / res)
    xs = x0 + (np.arange(w) + 0.5) * res
    ys = y1 - (np.arange(h) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    rng = np.random.default_rng(42)
    z = 25 - 0.004 * (X - cx) + 0.003 * (Y - cy)                        # slopes down to the sea (east)
    wy = _wadi_y(X, cx, cy)
    z -= 6.0 * np.exp(-0.5 * ((Y - wy) / 120.0) ** 2)                    # wadi valley
    for (dx, dy, depth, r) in [(-700, 500, 2.5, 160), (400, -150, 2.0, 140), (900, 800, 1.6, 120),
                               (-200, -900, 1.8, 150)]:
        z -= depth * np.exp(-0.5 * (((X - cx - dx) ** 2 + (Y - cy - dy) ** 2) / r ** 2))  # closed depressions
    z += rng.normal(0, 0.15, z.shape)
    built = (np.hypot(X - cx, Y - cy) < 1300).astype("uint8") * 50        # dense core
    built[(built == 0) & (rng.random(built.shape) < 0.35)] = 50
    built[built == 0] = 40                                                # cropland around

    tmp = out_root.parent.parent / "interim" / "demo"
    tmp.mkdir(parents=True, exist_ok=True)
    tr = from_origin(x0, y1, res, res)
    prof = {"driver": "GTiff", "width": w, "height": h, "count": 1, "crs": f"EPSG:{common.METRIC_CRS}",
            "transform": tr}
    with rasterio.open(tmp / "dem_utm.tif", "w", dtype="float32", nodata=-9999, **prof) as dst:
        dst.write(z.astype("float32"), 1)
    wc_prof = prof | {"transform": tr}
    with rasterio.open(tmp / "worldcover_utm.tif", "w", dtype="uint8", nodata=0, **wc_prof) as dst:
        dst.write(built, 1)

    # wadi as an OSM-like waterway line
    wx = np.linspace(x0 + pad / 2, x0 + w * res - pad / 2, 80)
    wadi_line_m = LineString([(x, _wadi_y(x, cx, cy)) for x in wx])
    waterways = gpd.GeoDataFrame({"waterway": ["stream"], "intermittent": ["yes"], "name": ["Oued Souhil"]},
                                 geometry=[wadi_line_m], crs=common.METRIC_CRS)
    rasters = terrain.derive(tmp / "dem_utm.tif", tmp, waterways, stream_area_m2=2_000_000)

    # --- synthetic road graph ------------------------------------------------
    G = nx.MultiDiGraph(crs="EPSG:4326")
    nid = {}
    for k, (x, y) in xy.items():
        lon, lat = to_ll.transform(x, y)
        nid[k] = 1000 + k[0] * N + k[1]
        G.add_node(nid[k], x=lon, y=lat)

    def add(a, b, name, highway, speed, **tags):
        (xa, ya), (xb, yb) = xy[a], xy[b]
        la = to_ll.transform(xa, ya)
        lb = to_ll.transform(xb, yb)
        length = math.hypot(xb - xa, yb - ya)
        for u, v, g in ((a, b, LineString([la, lb])), (b, a, LineString([lb, la]))):
            G.add_edge(nid[u], nid[v], length=length, travel_time=length / (speed / 3.6),
                       name=name, highway=highway, maxspeed=str(speed) if highway != "residential" else None,
                       geometry=g, **tags)

    rng2 = np.random.default_rng(7)
    for j in range(N):
        main = j % 4 == 1
        for i in range(N - 1):
            if not main and rng2.random() < 0.08:
                continue                                                     # some missing links
            add((i, j), (i + 1, j), STREETS_EW[j], "primary" if main else "residential", 50 if main else 30)
    for i in range(N):
        main = i % 5 == 2
        for j in range(N - 1):
            if not main and rng2.random() < 0.08:
                continue
            name = STREETS_NS[i]
            tags = {}
            ya, yb = xy[(i, j)][1], xy[(i, j + 1)][1]
            wyy = _wadi_y(xy[(i, j)][0], cx, cy)
            crosses = min(ya, yb) <= wyy <= max(ya, yb)
            if crosses and main:
                tags["bridge"] = "yes"
            elif crosses and i % 3 == 0:
                tags["ford"] = "yes"
            if i == 7 and j == 9:
                tags["tunnel"] = "yes"; tags["layer"] = "-1"                 # an underpass
            hw = "secondary" if main else ("unclassified" if i % 6 == 4 else "residential")
            add((i, j), (i, j + 1), None if hw == "unclassified" else name, hw, 50 if main else 30, **tags)

    D = common.collapse_to_digraph(G)
    seg = common.build_segments(D)
    seg = seg.join(features.terrain_measurements(seg, rasters))
    seg["impervious_share"] = features.impervious_share(seg, tmp / "worldcover_utm.tif")
    seg["drain_count"] = 0.0

    # synthetic past flood events (one held out as test set)
    ev_pts = []
    for (dx, dy, rel, test) in [(-700, 500, "high", False), (400, -150, "medium", False),
                                (0, -300, "high", False), (900, 800, "low", True)]:
        lon, lat = to_ll.transform(cx + dx, cy + dy)
        ev_pts.append({"event_date": "2025-12-26T15:00:00Z", "source": "synthetic demo",
                       "reliability": rel, "test_set": test, "geometry": Point(lon, lat)})
    events = gpd.GeoDataFrame(ev_pts, crs=4326)
    out = out_root
    out.mkdir(parents=True, exist_ok=True)
    events.to_file(out / "events.geojson", driver="GeoJSON")

    seg, params = common.normalise(seg, events)
    common.write_area(out, "Nabeul (synthetic demo)", D, seg,
                      ["Synthetic demo data - not real roads or terrain"], params, synthetic=True)
    log.info("demo area: %d segments, %d nodes -> %s", len(seg), D.number_of_nodes(), out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()
    build(args.data_dir / "processed" / "demo")


if __name__ == "__main__":
    main()
