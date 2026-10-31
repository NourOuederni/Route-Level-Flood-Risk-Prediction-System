"""Load the precomputed pilot area produced by the pipeline (data/processed/<area>/).

Files
  area.json         name, bbox, centre, sources, feature version
  graph.pkl         networkx.DiGraph; nodes: x (lon), y (lat); edges: travel_time [s],
                    length [m], edge_uid, name, highway, forward (True if u -> v follows
                    the canonical direction of the edge's segments)
  segments.parquet  one row per <= 200 m segment, WGS84 geometry in canonical direction,
                    normalised factors f_* and drainage proxies dv_*
"""
from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from ..engine.types import SegmentStatic
from ..weather.store import Cell, build_grid, nearest_cell

STATIC_FIELDS = [
    "f_depression", "f_drainage_position", "f_low_slope", "f_impervious", "f_crossing",
    "f_past_flood", "dv_urban", "dv_blackspot", "dv_drain_density_inv",
]


@dataclass
class SegmentRecord:
    static: SegmentStatic
    edge_uid: str
    seq: int
    length_m: float
    name: str | None
    road_type: str | None
    coords: list[tuple[float, float]]       # (lat, lon), canonical direction
    cell_id: str
    mid: tuple[float, float]


class Area:
    def __init__(self, directory: Path, grid_step: float):
        self.dir = directory
        self.meta = json.loads((directory / "area.json").read_text(encoding="utf-8"))
        with open(directory / "graph.pkl", "rb") as fh:
            self.graph: nx.DiGraph = pickle.load(fh)
        self.cells: list[Cell] = build_grid(tuple(self.meta["bbox"]), grid_step)
        self.segments: dict[int, SegmentRecord] = {}
        self.edge_segments: dict[str, list[int]] = {}
        self._load_segments(directory / "segments.parquet")
        self._build_node_index()

    @property
    def centre(self) -> tuple[float, float]:
        return tuple(self.meta["centre"])  # (lat, lon)

    def _load_segments(self, path: Path) -> None:
        gdf = gpd.read_parquet(path).to_crs(4326)
        has_maint = "dv_maintenance" in gdf.columns
        for row in gdf.itertuples(index=False):
            coords = [(float(y), float(x)) for x, y in row.geometry.coords]
            mid = coords[len(coords) // 2]
            maint = getattr(row, "dv_maintenance", None) if has_maint else None
            if maint is not None and not (isinstance(maint, float) and math.isnan(maint)):
                maint = float(maint)
            else:
                maint = None
            static = SegmentStatic(
                segment_id=int(row.segment_id),
                **{f: float(getattr(row, f)) for f in STATIC_FIELDS},
                dv_maintenance=maint,
                poorly_mapped=bool(row.poorly_mapped),
                past_flood_confirmed=bool(row.past_flood_confirmed),
                crossing_kind=row.crossing_kind if isinstance(row.crossing_kind, str) else None,
            )
            rec = SegmentRecord(
                static=static, edge_uid=str(row.edge_uid), seq=int(row.seq),
                length_m=float(row.length_m),
                name=row.name if isinstance(row.name, str) else None,
                road_type=row.road_type if isinstance(row.road_type, str) else None,
                coords=coords, cell_id=nearest_cell(self.cells, mid[0], mid[1]), mid=mid)
            self.segments[static.segment_id] = rec
            self.edge_segments.setdefault(rec.edge_uid, []).append(static.segment_id)
        for uid, ids in self.edge_segments.items():
            ids.sort(key=lambda i: self.segments[i].seq)

    def _build_node_index(self) -> None:
        self._nodes = list(self.graph.nodes)
        lat0 = self.centre[0]
        self._kx = 111_320 * math.cos(math.radians(lat0))
        self._ky = 110_570
        xy = np.array([[self.graph.nodes[n]["x"] * self._kx, self.graph.nodes[n]["y"] * self._ky]
                       for n in self._nodes])
        self._tree = cKDTree(xy)

    def nearest_node(self, lat: float, lon: float) -> tuple[int, float]:
        dist, idx = self._tree.query([lon * self._kx, lat * self._ky])
        return self._nodes[int(idx)], float(dist)

    def segments_for_edge(self, u, v) -> list[SegmentRecord]:
        """Segments of directed edge u->v, ordered and oriented in travel direction."""
        d = self.graph.edges[u, v]
        recs = [self.segments[i] for i in self.edge_segments.get(d["edge_uid"], [])]
        if d.get("forward", True):
            return recs
        return [_reversed(r) for r in reversed(recs)]

    def in_bbox(self, lat: float, lon: float, margin: float = 0.02) -> bool:
        w, s, e, n = self.meta["bbox"]
        return (w - margin) <= lon <= (e + margin) and (s - margin) <= lat <= (n + margin)


def _reversed(rec: SegmentRecord) -> SegmentRecord:
    return SegmentRecord(static=rec.static, edge_uid=rec.edge_uid, seq=rec.seq,
                         length_m=rec.length_m, name=rec.name, road_type=rec.road_type,
                         coords=list(reversed(rec.coords)), cell_id=rec.cell_id, mid=rec.mid)
