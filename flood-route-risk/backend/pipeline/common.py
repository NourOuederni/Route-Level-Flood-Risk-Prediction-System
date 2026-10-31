"""Shared pipeline helpers: graph collapse, segmentation, normalisation, output writing.

Used by both the real pipeline (pipeline.run) and the offline demo (pipeline.make_demo_area)
so the backend always reads the same format.
"""
from __future__ import annotations

import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import substring, transform

METRIC_CRS = 32632          # UTM 32N covers all of Tunisia (guide 3.3)
MAX_SEGMENT_M = 200.0
FEATURES_VERSION = "v1.0"
POORLY_MAPPED_TYPES = {"unclassified", "road", "track", "service"}
RELIABILITY_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}

_to_metric = Transformer.from_crs(4326, METRIC_CRS, always_xy=True).transform


def _first(v):
    return v[0] if isinstance(v, list) and v else v


def collapse_to_digraph(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Keep the fastest of parallel edges so routing can use a simple DiGraph."""
    D = nx.DiGraph()
    D.graph.update(G.graph)
    for n, d in G.nodes(data=True):
        D.add_node(n, x=float(d["x"]), y=float(d["y"]))
    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        if D.has_edge(u, v) and D.edges[u, v]["travel_time"] <= d["travel_time"]:
            continue
        geom = d.get("geometry")
        if geom is None:
            geom = LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
        D.add_edge(u, v, travel_time=float(d["travel_time"]), length=float(d["length"]),
                   geometry=geom, name=_first(d.get("name")), highway=_first(d.get("highway")),
                   maxspeed=_first(d.get("maxspeed")), bridge=_first(d.get("bridge")),
                   tunnel=_first(d.get("tunnel")), ford=_first(d.get("ford")),
                   layer=_first(d.get("layer")))
    return D


def split_line(line: LineString, max_len: float = MAX_SEGMENT_M) -> list[LineString]:
    n = max(1, math.ceil(line.length / max_len))
    step = line.length / n
    return [substring(line, i * step, (i + 1) * step) for i in range(n)]


def build_segments(D: nx.DiGraph) -> gpd.GeoDataFrame:
    """Cut each undirected road edge into <= 200 m segments (metric CRS).

    Sets edge attributes `edge_uid` and `forward` on D. Both directions of a two-way
    road share the same segments (one risk per piece of road).
    """
    rows = []
    seen: dict[str, None] = {}
    sid = 0
    for u, v, d in D.edges(data=True):
        a, b = (u, v) if str(u) <= str(v) else (v, u)
        uid = f"{a}_{b}_{round(d['length'])}"
        d["edge_uid"] = uid
        d["forward"] = (u == a)
        if uid in seen:
            continue
        seen[uid] = None
        geom = d["geometry"] if u == a else LineString(list(d["geometry"].coords)[::-1])
        metric = transform(_to_metric, geom)
        tunnel = str(d.get("tunnel") or "no")
        layer = d.get("layer")
        try:
            layer_v = int(str(layer).split(";")[0]) if layer is not None else 0
        except ValueError:
            layer_v = 0
        for seq, piece in enumerate(split_line(metric)):
            sid += 1
            rows.append({
                "segment_id": sid, "edge_uid": uid, "seq": seq, "length_m": piece.length,
                "name": d.get("name"), "road_type": d.get("highway"),
                "has_maxspeed": d.get("maxspeed") is not None,
                "is_bridge": str(d.get("bridge") or "no") not in ("no", "nan", "None"),
                "is_underpass": tunnel not in ("no", "nan", "None") or layer_v < 0,
                "is_ford": str(d.get("ford") or "no") not in ("no", "nan", "None"),
                "geometry": piece,
            })
    for _, _, d in D.edges(data=True):
        d.pop("geometry", None)            # keep the pickled graph small
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC_CRS)


# --- normalisation (guide 7.2) ---------------------------------------------
def prank(s: pd.Series) -> pd.Series:
    """Percentile rank in [0,1] over the pilot area (robust to outliers)."""
    s = s.astype(float)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=s.index)
    return s.rank(pct=True, method="average").fillna(0.0)


def prank_positive(s: pd.Series) -> pd.Series:
    """Percentile rank among positive values; zeros stay 0 (e.g. depth in sink)."""
    s = s.astype(float).fillna(0.0)
    out = pd.Series(0.0, index=s.index)
    pos = s > 0
    if pos.sum() > 1:
        out[pos] = s[pos].rank(pct=True, method="average")
    elif pos.sum() == 1:
        out[pos] = 1.0
    return out


def event_features(seg: gpd.GeoDataFrame, events: gpd.GeoDataFrame | None) -> pd.DataFrame:
    """Past-flood proximity f6, black-spot recurrence B, confirmed flag. Test-set events excluded."""
    out = pd.DataFrame({"f_past_flood": 0.0, "dv_blackspot": 0.0, "past_flood_confirmed": False},
                       index=seg.index)
    if events is None or events.empty:
        return out
    ev = events[~events.get("test_set", pd.Series(False, index=events.index)).fillna(False).astype(bool)]
    if ev.empty:
        return out
    ev = ev.to_crs(METRIC_CRS)
    mids = seg.geometry.interpolate(0.5, normalized=True)
    f6 = np.zeros(len(seg))
    cnt = np.zeros(len(seg))
    conf = np.zeros(len(seg), dtype=bool)
    for _, e in ev.iterrows():
        w = RELIABILITY_WEIGHT.get(str(e.get("reliability", "medium")), 0.7)
        d = mids.distance(e.geometry).to_numpy()
        f6 = np.maximum(f6, w * np.exp(-d / 150.0))
        cnt += w * (d <= 100.0)
        conf |= (d <= 50.0) & (w >= 0.7)
    out["f_past_flood"] = np.clip(f6, 0, 1)
    out["dv_blackspot"] = np.clip(cnt / 3.0, 0, 1)
    out["past_flood_confirmed"] = conf
    return out


def normalise(seg: gpd.GeoDataFrame, events: gpd.GeoDataFrame | None) -> tuple[gpd.GeoDataFrame, dict]:
    """Raw per-segment measurements -> f_* and dv_* in [0,1]."""
    s = seg.copy()
    s["f_depression"] = 0.5 * prank_positive(s["depth_in_sink_m"]) + 0.5 * prank(s["twi_mean"])
    s["f_drainage_position"] = 0.5 * (1 - prank(s["hand_m"])) + 0.5 * (1 - prank(s["dist_stream_m"]))
    s["f_low_slope"] = 1 - prank(s["slope_deg"])
    s["f_impervious"] = prank(s["impervious_share"])
    crossing = np.where(s["is_underpass"] | s["is_ford"], 1.0,
                        np.where(s["wadi_crossing"] & ~s["is_bridge"], 0.8, 0.0))
    s["f_crossing"] = crossing
    s["crossing_kind"] = np.select(
        [s["is_underpass"], s["is_ford"], s["wadi_crossing"] & ~s["is_bridge"]],
        ["underpass", "ford", "low_crossing"], default=None)
    s["dv_urban"] = s["f_impervious"]
    if s["drain_count"].sum() > 0:
        s["dv_drain_density_inv"] = 1 - prank(s["drain_count"] / s["length_m"].clip(lower=1))
    else:
        s["dv_drain_density_inv"] = 0.5       # no inlets mapped: carries no information
    if "dv_maintenance" not in s:
        s["dv_maintenance"] = np.nan
    ef = event_features(s, events)
    s[ef.columns] = ef
    s["poorly_mapped"] = s["name"].isna() & ~s["has_maxspeed"] & s["road_type"].isin(POORLY_MAPPED_TYPES)

    params = {c: [float(np.nanpercentile(s[c], q)) for q in (5, 25, 50, 75, 95)]
              for c in ["depth_in_sink_m", "twi_mean", "hand_m", "dist_stream_m", "slope_deg",
                        "impervious_share"]}
    for c in ["f_depression", "f_drainage_position", "f_low_slope", "f_impervious", "f_crossing",
              "f_past_flood", "dv_urban", "dv_blackspot", "dv_drain_density_inv"]:
        s[c] = s[c].astype(float).clip(0, 1)
    return s, params


def write_area(out_dir: Path, name: str, D: nx.DiGraph, seg: gpd.GeoDataFrame, sources: list[str],
               norm_params: dict, synthetic: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_out = seg.to_crs(4326)
    keep = ["segment_id", "edge_uid", "seq", "length_m", "name", "road_type",
            "f_depression", "f_drainage_position", "f_low_slope", "f_impervious", "f_crossing",
            "f_past_flood", "dv_urban", "dv_blackspot", "dv_drain_density_inv", "dv_maintenance",
            "poorly_mapped", "past_flood_confirmed", "crossing_kind",
            # raw measurements kept for QA in QGIS
            "depth_in_sink_m", "twi_mean", "hand_m", "dist_stream_m", "slope_deg",
            "impervious_share", "is_underpass", "is_ford", "is_bridge", "wadi_crossing", "geometry"]
    seg_out = seg_out[[c for c in keep if c in seg_out.columns]]
    seg_out.to_parquet(out_dir / "segments.parquet")
    seg_out.to_file(out_dir / "segments_qa.gpkg", driver="GPKG")  # open in QGIS
    with open(out_dir / "graph.pkl", "wb") as fh:
        pickle.dump(D, fh, protocol=pickle.HIGHEST_PROTOCOL)
    xs = [d["x"] for _, d in D.nodes(data=True)]
    ys = [d["y"] for _, d in D.nodes(data=True)]
    bbox = [min(xs), min(ys), max(xs), max(ys)]
    meta = {
        "name": name, "bbox": bbox, "centre": [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2],
        "features_version": FEATURES_VERSION, "built_at": datetime.now(timezone.utc).isoformat(),
        "segments": int(len(seg_out)), "nodes": D.number_of_nodes(), "edges": D.number_of_edges(),
        "synthetic": synthetic, "sources": sources, "normalisation": norm_params,
    }
    (out_dir / "area.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_events(path: Path) -> gpd.GeoDataFrame | None:
    if not path.is_file():
        return None
    ev = gpd.read_file(path)
    return ev if not ev.empty else None


def point(lon: float, lat: float) -> Point:
    return Point(lon, lat)
