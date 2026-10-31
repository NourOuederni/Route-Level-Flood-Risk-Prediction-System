"""Build a national igraph routing graph from PostGIS edges.

The graph is serialised to a pickle file that the V2 API loads at startup.
igraph is ~10x faster than NetworkX for shortest-path on 200k+ edges.
"""
from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path

import igraph as ig
from sqlalchemy import text

from app.db.models import make_engine

log = logging.getLogger("pipeline.build_graph")


def build_national_graph(db_url: str, out_path: Path) -> tuple[int, int]:
    """Build igraph from the road network stored in PostGIS segments.

    Each unique edge_uid becomes one undirected edge in the graph, weighted by
    the sum of its segment lengths (converted to an approximate travel time
    using a speed lookup by road_type).
    """
    engine = make_engine(db_url)

    # Speed assumptions by road_type (km/h) for travel_time weight
    SPEED_KMH = {
        "motorway": 110, "motorway_link": 80,
        "trunk": 90, "trunk_link": 60,
        "primary": 70, "primary_link": 50,
        "secondary": 50, "secondary_link": 40,
        "tertiary": 40, "tertiary_link": 30,
        "residential": 30, "living_street": 20,
        "unclassified": 30, "service": 20,
        "track": 15, "road": 30,
    }
    DEFAULT_SPEED = 30

    with engine.connect() as conn:
        # Get all edges: aggregate segments back to edges
        rows = conn.execute(text("""
            SELECT edge_uid,
                   MIN(tile_id) as tile_id,
                   MAX(road_type) as road_type,
                   MAX(name) as name,
                   SUM(length_m) as total_length_m,
                   -- Endpoints: first segment start, last segment end
                   (SELECT ST_Y(ST_StartPoint(s2.geom))
                    FROM segment s2
                    WHERE s2.edge_uid = s.edge_uid AND s2.seq = 0
                    LIMIT 1) as start_lat,
                   (SELECT ST_X(ST_StartPoint(s2.geom))
                    FROM segment s2
                    WHERE s2.edge_uid = s.edge_uid AND s2.seq = 0
                    LIMIT 1) as start_lon,
                   (SELECT ST_Y(ST_EndPoint(s2.geom))
                    FROM segment s2
                    WHERE s2.edge_uid = s.edge_uid AND s2.seq = (
                        SELECT MAX(s3.seq) FROM segment s3 WHERE s3.edge_uid = s.edge_uid)
                    LIMIT 1) as end_lat,
                   (SELECT ST_X(ST_EndPoint(s2.geom))
                    FROM segment s2
                    WHERE s2.edge_uid = s.edge_uid AND s2.seq = (
                        SELECT MAX(s3.seq) FROM segment s3 WHERE s3.edge_uid = s.edge_uid)
                    LIMIT 1) as end_lon
            FROM segment s
            GROUP BY edge_uid
        """)).fetchall()

    engine.dispose()
    log.info("loaded %d edges from PostGIS", len(rows))

    # Build node index: snap coordinates to a grid (~1m resolution) for dedup
    SNAP = 5      # decimal places for lat/lon snapping (~1.1m)
    node_map: dict[tuple[float, float], int] = {}
    nodes: list[tuple[float, float]] = []

    def get_node(lat: float, lon: float) -> int:
        key = (round(lat, SNAP), round(lon, SNAP))
        idx = node_map.get(key)
        if idx is None:
            idx = len(nodes)
            node_map[key] = idx
            nodes.append(key)
        return idx

    edges_data = []
    for row in rows:
        if row.start_lat is None or row.end_lat is None:
            continue
        u = get_node(row.start_lat, row.start_lon)
        v = get_node(row.end_lat, row.end_lon)
        if u == v:
            continue
        length_m = float(row.total_length_m)
        speed = SPEED_KMH.get(row.road_type, DEFAULT_SPEED)
        travel_time = length_m / (speed * 1000 / 3600)    # seconds
        edges_data.append({
            "source": u, "target": v,
            "edge_uid": row.edge_uid,
            "length": length_m,
            "travel_time": travel_time,
            "road_type": row.road_type,
            "name": row.name,
        })

    log.info("graph: %d nodes, %d edges", len(nodes), len(edges_data))

    # Build igraph
    g = ig.Graph(n=len(nodes), directed=True)
    g.vs["lat"] = [n[0] for n in nodes]
    g.vs["lon"] = [n[1] for n in nodes]

    if edges_data:
        g.add_edges([(e["source"], e["target"]) for e in edges_data])
        g.es["edge_uid"] = [e["edge_uid"] for e in edges_data]
        g.es["length"] = [e["length"] for e in edges_data]
        g.es["travel_time"] = [e["travel_time"] for e in edges_data]
        g.es["road_type"] = [e["road_type"] for e in edges_data]
        g.es["name"] = [e["name"] for e in edges_data]

        # Add reverse edges for bidirectional travel (most roads are two-way)
        reverse_edges = [(e["target"], e["source"]) for e in edges_data]
        n_forward = len(edges_data)
        g.add_edges(reverse_edges)
        g.es[n_forward:]["edge_uid"] = [e["edge_uid"] for e in edges_data]
        g.es[n_forward:]["length"] = [e["length"] for e in edges_data]
        g.es[n_forward:]["travel_time"] = [e["travel_time"] for e in edges_data]
        g.es[n_forward:]["road_type"] = [e["road_type"] for e in edges_data]
        g.es[n_forward:]["name"] = [e["name"] for e in edges_data]

    # Simplify: keep only the fastest of parallel edges
    g = g.simplify(multiple=True, loops=True, combine_edges={
        "edge_uid": "first",
        "length": "min",
        "travel_time": "min",
        "road_type": "first",
        "name": "first",
    })

    log.info("simplified graph: %d vertices, %d edges", g.vcount(), g.ecount())

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(g, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # Also save the node coordinate mapping for fast lookup
    meta = {
        "vertices": g.vcount(),
        "edges": g.ecount(),
        "node_snap_decimals": SNAP,
    }
    meta_path = out_path.with_suffix(".json")
    import json
    meta_path.write_text(json.dumps(meta, indent=2))

    return g.vcount(), g.ecount()


if __name__ == "__main__":
    import argparse, os
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=os.environ.get(
        "DATABASE_URL", "postgresql://flood:flood_dev_2024@localhost:5433/floodrisk"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/national/graph.pickle"))
    args = ap.parse_args()
    v, e = build_national_graph(args.db_url, args.out)
    print(f"Built graph: {v} vertices, {e} edges")
