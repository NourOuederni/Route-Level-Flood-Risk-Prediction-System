"""National routing graph backed by igraph + PostGIS.

Replaces V1's in-memory Area class. The graph lives in igraph for fast
shortest-path, while segment data is fetched from PostGIS on demand.
"""
from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import igraph as ig
import numpy as np
from scipy.spatial import cKDTree
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db.models import Segment, make_engine, make_session
from ..engine.types import SegmentStatic

log = logging.getLogger(__name__)

STATIC_FIELDS = [
    "f_depression", "f_drainage_position", "f_low_slope", "f_impervious", "f_crossing",
    "f_past_flood", "dv_urban", "dv_blackspot", "dv_drain_density_inv",
]


@dataclass
class SegmentRecord:
    """Same shape as V1 SegmentRecord — the planner doesn't care where data came from."""
    static: SegmentStatic
    edge_uid: str
    seq: int
    length_m: float
    name: str | None
    road_type: str | None
    coords: list[tuple[float, float]]       # (lat, lon)
    cell_id: str
    mid: tuple[float, float]
    susceptibility_floored: float            # V2: pre-computed S'


class NationalGraph:
    """National-scale routing backed by igraph + PostGIS segments."""

    def __init__(self, graph_path: Path, db_url: str):
        with open(graph_path, "rb") as fh:
            self.graph: ig.Graph = pickle.load(fh)
        self._engine = make_engine(db_url)
        self._Session = make_session(self._engine)
        self._build_node_index()

        # Cache for segments along an edge (edge_uid -> list[SegmentRecord])
        self._segment_cache: dict[str, list[SegmentRecord]] = {}
        log.info("national graph loaded: %d vertices, %d edges",
                 self.graph.vcount(), self.graph.ecount())

    def _build_node_index(self) -> None:
        """KDTree over graph vertices for fast nearest-node snapping."""
        lats = self.graph.vs["lat"]
        lons = self.graph.vs["lon"]
        # Approximate metric coords for distance computation
        self._lat0 = np.mean(lats)
        self._kx = 111_320 * math.cos(math.radians(self._lat0))
        self._ky = 110_570
        xy = np.column_stack([np.array(lons) * self._kx, np.array(lats) * self._ky])
        self._tree = cKDTree(xy)

    @property
    def centre(self) -> tuple[float, float]:
        return (33.8, 9.5)    # Tunisia centre

    @property
    def bbox(self) -> list[float]:
        return [7.5, 30.2, 11.6, 37.5]

    def nearest_node(self, lat: float, lon: float) -> tuple[int, float]:
        """Find nearest graph vertex. Returns (vertex_index, distance_metres)."""
        dist, idx = self._tree.query([lon * self._kx, lat * self._ky])
        return int(idx), float(dist)

    def in_bbox(self, lat: float, lon: float, margin: float = 0.1) -> bool:
        w, s, e, n = self.bbox
        return (w - margin) <= lon <= (e + margin) and (s - margin) <= lat <= (n + margin)

    def shortest_path(self, source: int, target: int, weights: str = "travel_time") -> list[int]:
        """Return the vertex sequence of the shortest path, or raise if unreachable."""
        paths = self.graph.get_shortest_paths(source, to=target, weights=weights, output="vpath")
        if not paths or not paths[0]:
            raise ValueError("no path between source and target")
        return paths[0]

    def k_shortest_paths(self, source: int, target: int, k: int = 3,
                         max_factor: float = 1.5) -> list[list[int]]:
        """Up to k diverse shortest paths using penalty method."""
        best = self.shortest_path(source, target)
        best_tt = self._path_travel_time(best)
        accepted = [best]
        weights = list(self.graph.es["travel_time"])
        penalty_factor = 2.0

        for _ in range(k * 3):
            if len(accepted) >= k:
                break
            # Penalise edges used by accepted paths
            pen_weights = list(weights)
            for path in accepted:
                for i in range(len(path) - 1):
                    eid = self.graph.get_eid(path[i], path[i + 1], error=False)
                    if eid >= 0:
                        pen_weights[eid] *= penalty_factor
            try:
                paths = self.graph.get_shortest_paths(
                    source, to=target, weights=pen_weights, output="vpath")
                if not paths or not paths[0]:
                    break
                cand = paths[0]
            except Exception:
                break
            tt = self._path_travel_time(cand)
            if tt > best_tt * max_factor:
                break
            # Check overlap
            if all(self._overlap(cand, p) < 0.7 for p in accepted):
                accepted.append(cand)

        return accepted

    def _path_travel_time(self, path: list[int]) -> float:
        total = 0.0
        for i in range(len(path) - 1):
            eid = self.graph.get_eid(path[i], path[i + 1], error=False)
            if eid >= 0:
                total += self.graph.es[eid]["travel_time"]
        return total

    def _path_length(self, path: list[int]) -> float:
        total = 0.0
        for i in range(len(path) - 1):
            eid = self.graph.get_eid(path[i], path[i + 1], error=False)
            if eid >= 0:
                total += self.graph.es[eid]["length"]
        return total

    def _overlap(self, a: list[int], b: list[int]) -> float:
        eb = set(zip(b, b[1:]))
        shared = 0.0
        total = 0.0
        for i in range(len(a) - 1):
            eid = self.graph.get_eid(a[i], a[i + 1], error=False)
            if eid < 0:
                continue
            length = self.graph.es[eid]["length"]
            total += length
            if (a[i], a[i + 1]) in eb:
                shared += length
        return shared / max(total, 1.0)

    def edge_info(self, u: int, v: int) -> dict:
        """Get edge attributes between two vertices."""
        eid = self.graph.get_eid(u, v, error=False)
        if eid < 0:
            return {}
        return {
            "edge_uid": self.graph.es[eid]["edge_uid"],
            "travel_time": self.graph.es[eid]["travel_time"],
            "length": self.graph.es[eid]["length"],
        }

    def segments_for_edge(self, u: int, v: int) -> list[SegmentRecord]:
        """Fetch segments for this edge from PostGIS (with caching)."""
        eid = self.graph.get_eid(u, v, error=False)
        if eid < 0:
            return []
        edge_uid = self.graph.es[eid]["edge_uid"]

        cached = self._segment_cache.get(edge_uid)
        if cached is not None:
            return cached

        session = self._Session()
        try:
            rows = session.query(Segment).filter(
                Segment.edge_uid == edge_uid
            ).order_by(Segment.seq).all()

            records = []
            for row in rows:
                static = SegmentStatic(
                    segment_id=row.segment_id,
                    **{f: float(getattr(row, f)) for f in STATIC_FIELDS},
                    dv_maintenance=row.dv_maintenance,
                    poorly_mapped=bool(row.poorly_mapped),
                    past_flood_confirmed=bool(row.past_flood_confirmed),
                    crossing_kind=row.crossing_kind,
                )
                # Get coordinates from geometry
                from shapely import wkb
                geom = wkb.loads(bytes(row.geom.data))
                coords = [(y, x) for x, y in geom.coords]    # (lat, lon)
                mid = coords[len(coords) // 2] if coords else (row.mid_lat, row.mid_lon)

                records.append(SegmentRecord(
                    static=static,
                    edge_uid=row.edge_uid,
                    seq=row.seq,
                    length_m=row.length_m,
                    name=row.name,
                    road_type=row.road_type,
                    coords=coords,
                    cell_id=row.cell_id or "",
                    mid=mid,
                    susceptibility_floored=row.susceptibility_floored,
                ))
            self._segment_cache[edge_uid] = records
            return records
        finally:
            session.close()

    def segments_in_bbox(self, west: float, south: float, east: float, north: float,
                         limit: int = 2000) -> list[SegmentRecord]:
        """Spatial query: segments whose midpoint falls within a bounding box."""
        session = self._Session()
        try:
            rows = session.query(Segment).filter(
                Segment.mid_lat >= south,
                Segment.mid_lat <= north,
                Segment.mid_lon >= west,
                Segment.mid_lon <= east,
            ).limit(limit).all()
            records = []
            for row in rows:
                static = SegmentStatic(
                    segment_id=row.segment_id,
                    **{f: float(getattr(row, f)) for f in STATIC_FIELDS},
                    dv_maintenance=row.dv_maintenance,
                    poorly_mapped=bool(row.poorly_mapped),
                    past_flood_confirmed=bool(row.past_flood_confirmed),
                    crossing_kind=row.crossing_kind,
                )
                records.append(SegmentRecord(
                    static=static,
                    edge_uid=row.edge_uid,
                    seq=row.seq,
                    length_m=row.length_m,
                    name=row.name,
                    road_type=row.road_type,
                    coords=[],      # not needed for hotspots
                    cell_id=row.cell_id or "",
                    mid=(row.mid_lat, row.mid_lon),
                    susceptibility_floored=row.susceptibility_floored,
                ))
            return records
        finally:
            session.close()

    def clear_cache(self) -> None:
        self._segment_cache.clear()
