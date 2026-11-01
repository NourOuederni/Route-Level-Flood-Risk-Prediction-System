"""V2 route planner: igraph-backed, PostGIS segments, same algorithmic engine.

Key difference from V1: uses pre-computed S' from PostGIS. At request time,
only H (hazard = R_eff / T_eff) is computed per segment — a single division.

Score = 100 * H * S'   (same formula, S' is just pre-stored)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..engine import EvalContext, SegmentRisk, evaluate_segment, level_at_least
from ..engine.config import LEVEL_RANK, EngineConfig
from .national_graph import NationalGraph, SegmentRecord


class PlanningError(ValueError):
    pass


@dataclass
class RoutedSegment:
    rec: SegmentRecord
    eta: datetime
    risk: SegmentRisk


class RiskScorer:
    """Scores segments at an ETA window, with a per-request memo.
    Same as V1, but works with NationalGraph's SegmentRecord."""

    def __init__(self, graph: NationalGraph, weather, cfg: EngineConfig, now: datetime,
                 feedback: dict[int, str] | None = None):
        self.graph, self.weather, self.cfg, self.now = graph, weather, cfg, now
        self.feedback = feedback or {}
        r = cfg.section("rainfall")
        self.min_window = timedelta(minutes=float(r["window"]["min_minutes"]))
        self.frac = float(r["window"]["fraction_of_elapsed"])
        self.ante_h = int(r["antecedent_hours"])
        self.refresh = float(cfg.section("weather")["refresh_interval_minutes"])
        self._memo: dict[tuple, SegmentRisk] = {}

    def window(self, eta: datetime, elapsed_s: float) -> tuple[datetime, datetime]:
        half = max(self.min_window, timedelta(seconds=self.frac * max(elapsed_s, 0.0)))
        return eta - half, eta + half

    def score(self, rec: SegmentRecord, eta: datetime, elapsed_s: float) -> SegmentRisk:
        start, end = self.window(eta, elapsed_s)
        key = (rec.static.segment_id, int(start.timestamp() // 300), int(end.timestamp() // 300))
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        rain = self.weather.window(rec.cell_id, start, end, self.now, self.ante_h)
        ctx = EvalContext(refresh_interval_minutes=self.refresh,
                          recent_feedback=self.feedback.get(rec.static.segment_id))
        risk = evaluate_segment(rec.static, rain, ctx, self.cfg)
        self._memo[key] = risk
        return risk


class ZoneRiskScorer(RiskScorer):
    """Terrain model inside the pilot zone, algorithmic engine everywhere else."""

    def __init__(self, graph: NationalGraph, weather, cfg: EngineConfig, now: datetime,
                 feedback: dict[int, str] | None, zone, trigger_on: bool):
        super().__init__(graph, weather, cfg, now, feedback)
        self.zone, self.trigger_on = zone, trigger_on

    def score(self, rec: SegmentRecord, eta: datetime, elapsed_s: float) -> SegmentRisk:
        idx = self.zone.lookup(rec.mid[0], rec.mid[1])
        if idx is None:
            return super().score(rec, eta, elapsed_s)
        return self.zone.risk(idx, rec.static.segment_id, self.trigger_on)


class PlannerV2:
    """National-scale planner using igraph for routing."""

    def __init__(self, graph: NationalGraph, weather, cfg: EngineConfig, zone=None):
        self.graph, self.weather, self.cfg = graph, weather, cfg
        self.rcfg = cfg.section("routing")
        self.zone = zone          # ZoneModel or None

    def make_scorer(self, now: datetime, feedback: dict[int, str] | None = None):
        """Returns (scorer, trigger status). The trigger is None when no zone model is loaded."""
        if self.zone is None:
            return RiskScorer(self.graph, self.weather, self.cfg, now, feedback), None
        trigger = self.zone.trigger(self.weather.store, now)
        scorer = ZoneRiskScorer(self.graph, self.weather, self.cfg, now, feedback,
                                self.zone, trigger["status"] == "on")
        return scorer, trigger

    def _snap(self, lat: float, lon: float, what: str) -> int:
        if not self.graph.covers(lat, lon):
            w, s, e, n = self.graph.bbox
            raise PlanningError(
                f"{what} is outside the mapped area. Road data is currently loaded for "
                f"{s:.2f}–{n:.2f}°N, {w:.2f}–{e:.2f}°E.")
        node, dist = self.graph.nearest_node(lat, lon)
        max_dist = float(self.rcfg["max_snap_distance_m"])
        if dist > max_dist:
            near = self.graph.nearest_road(lat, lon)
            if near is not None:
                name, road_m = near
                label = f'"{name}"' if name else "an unnamed road"
                raise PlanningError(
                    f"{what} is {road_m / 1000:.1f} km from the nearest mapped road ({label}). "
                    f"Pick a point within {max_dist / 1000:.1f} km of a road.")
            raise PlanningError(f"{what} is too far from any mapped road.")
        return node

    def _path_stats(self, path: list[int]) -> tuple[float, float]:
        tt = 0.0
        ln = 0.0
        for i in range(len(path) - 1):
            info = self.graph.edge_info(path[i], path[i + 1])
            tt += info.get("travel_time", 0)
            ln += info.get("length", 0)
        return tt, ln

    def _route_segments(self, path: list[int], departure: datetime,
                        scorer: RiskScorer) -> list[RoutedSegment]:
        out: list[RoutedSegment] = []
        elapsed = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            info = self.graph.edge_info(u, v)
            recs = self.graph.segments_for_edge(u, v)
            edge_tt = info.get("travel_time", 0)
            total_len = sum(r.length_m for r in recs) or 1.0

            for rec in recs:
                dt = edge_tt * rec.length_m / total_len
                eta_s = elapsed + dt / 2
                eta = departure + timedelta(seconds=eta_s)
                out.append(RoutedSegment(rec, eta, scorer.score(rec, eta, eta_s)))
                elapsed += dt

            if not recs:
                elapsed += edge_tt
        return out

    def _risk_aware_path(self, o: int, d: int, departure: datetime,
                         scorer: RiskScorer, fastest_tt: float, alert_level: str):
        """Build a risk-penalised weight vector and find the safest path."""
        g = self.graph.graph
        base_weights = list(g.es["travel_time"])
        base_penalty = self.rcfg["risk_penalty"]
        cutoff_tt = fastest_tt * float(self.rcfg["corridor_factor"])

        # Estimate risk level per edge at approximate ETA
        pen_weights = list(base_weights)
        excl = set(self.rcfg["exclude_levels"])

        # Get approximate arrival times via shortest path from origin
        dists = g.shortest_paths(source=o, weights="travel_time")[0]

        for eid in range(g.ecount()):
            edge = g.es[eid]
            src = edge.source
            if dists[src] > cutoff_tt:
                continue
            eta = departure + timedelta(seconds=dists[src])
            recs = self.graph.segments_for_edge(edge.source, edge.target)
            worst = "low"
            for rec in recs:
                lvl = scorer.score(rec, eta, dists[src]).level
                if LEVEL_RANK[lvl] > LEVEL_RANK[worst]:
                    worst = lvl
            if worst in excl:
                pen_weights[eid] = float("inf")    # effectively remove
            else:
                pen = float(base_penalty.get(worst, 0.0))
                if level_at_least(worst, alert_level):
                    pen = max(pen, float(base_penalty.get("high", 0.3)))
                pen_weights[eid] = base_weights[eid] * (1.0 + pen)

        try:
            paths = g.get_shortest_paths(o, to=d, weights=pen_weights, output="vpath")
            if not paths or not paths[0]:
                return None
            path = paths[0]
            # Check it doesn't use excluded edges
            for i in range(len(path) - 1):
                eid = g.get_eid(path[i], path[i + 1], error=False)
                if eid >= 0 and pen_weights[eid] == float("inf"):
                    return None
            return path
        except Exception:
            return None

    def plan(self, origin: tuple[float, float], destination: tuple[float, float],
             departure: datetime | None, alert_level: str,
             feedback: dict[int, str] | None = None) -> tuple[dict, list[dict]]:
        now = datetime.now(timezone.utc)
        departure = (departure or now).astimezone(timezone.utc)
        o = self._snap(*origin, "origin")
        d = self._snap(*destination, "destination")
        if o == d:
            raise PlanningError("origin and destination are the same place")

        scorer, trigger = self.make_scorer(now, feedback)

        # Candidate paths via penalty method
        paths = self.graph.k_shortest_paths(
            o, d, k=int(self.rcfg["max_candidates"]),
            max_factor=float(self.rcfg["candidate_duration_factor"]))

        routes = []
        for i, p in enumerate(paths):
            segs = self._route_segments(p, departure, scorer)
            routes.append(self._summarise(p, segs, "fastest" if i == 0 else "alternative", alert_level))

        fastest = routes[0]
        safer = None
        fallback = None

        if fastest["alert"]["count"] > 0:
            rp = self._risk_aware_path(o, d, departure, scorer, fastest["duration_s"], alert_level)
            if rp is not None:
                segs = self._route_segments(rp, departure, scorer)
                cand = self._summarise(rp, segs, "safer", alert_level)
                excl = set(self.rcfg["exclude_levels"])
                still_excluded = any(s["level"] in excl for s in cand["segments"])
                improved = (cand["alert"]["exposure_m"] <= 0.9 * fastest["alert"]["exposure_m"]
                            or LEVEL_RANK[cand["max_level"]] < LEVEL_RANK[fastest["max_level"]])
                improved = improved and cand["_path"] != fastest["_path"]
                if improved and not still_excluded:
                    cand["extra_duration_s"] = round(cand["duration_s"] - fastest["duration_s"])
                    cand["extra_distance_m"] = round(cand["distance_m"] - fastest["distance_m"])
                    safer = cand
                    for r in routes:
                        if r["_path"] == cand["_path"]:
                            r["kind"] = "safer"
                            r["extra_duration_s"] = cand["extra_duration_s"]
                            r["extra_distance_m"] = cand["extra_distance_m"]
                            safer = r
                    if safer is cand:
                        routes.append(cand)
            if safer is None:
                fallback = {
                    "code": "no_safe_alternative",
                    "text": "No safer route was found. The original route is kept: "
                            "consider delaying the trip and avoid flooded roads.",
                }

        request_id = str(uuid.uuid4())
        log_rows = []
        for r in routes:
            for s in r["segments"]:
                log_rows.append({
                    "request_id": request_id, "segment_id": s["segment_id"],
                    "eta": datetime.fromisoformat(s["eta"]), "score": s["score"],
                    "level": s["level"], "confidence": s["confidence"], "mode": s["mode"],
                    "factors": {"contributions": s["contributions"], "dv": s["drainage_vulnerability"],
                                "threshold": s["threshold_mm_h"], "rain": s["effective_rain_mm_h"]},
                    "issued_at": now})
            r.pop("_path", None)

        result = {
            "request_id": request_id,
            "issued_at": now.isoformat(),
            "departure": departure.isoformat(),
            "alert_level": alert_level,
            "routes": routes,
            "safer_route_id": safer["id"] if safer else None,
            "fallback": fallback,
            "engine_version": self.cfg.version,
            "disclaimer": self.cfg.raw["disclaimer"],
            "trigger": trigger,
        }
        return result, log_rows

    def _summarise(self, path: list[int], segs: list[RoutedSegment], kind: str,
                   alert_level: str) -> dict:
        tt, ln = self._path_stats(path)
        seg_out = []
        alert_ids = []
        exposure_m = 0.0
        worst: RoutedSegment | None = None
        for s in segs:
            r = s.risk
            if worst is None or r.score > worst.risk.score:
                worst = s
            if level_at_least(r.level, alert_level):
                alert_ids.append(r.segment_id)
                exposure_m += s.rec.length_m
            dd = r.to_dict()
            dd.update({
                "eta": s.eta.isoformat(),
                "name": s.rec.name,
                "road_type": s.rec.road_type,
                "crossing": s.rec.static.crossing_kind,
                "coords": [[round(a, 6), round(b, 6)] for a, b in s.rec.coords],
            })
            seg_out.append(dd)

        names: dict[str, float] = {}
        for s in segs:
            if s.rec.name:
                names[s.rec.name] = names.get(s.rec.name, 0) + s.rec.length_m
        via = [n for n, _ in sorted(names.items(), key=lambda kv: -kv[1])[:2]]
        confs = [s.risk.confidence for s in segs] or [0.0]
        first_alert = next((s for s in segs if s.risk.segment_id in alert_ids), None)

        return {
            "id": uuid.uuid5(uuid.NAMESPACE_OID, "-".join(map(str, path))).hex[:12],
            "kind": kind,
            "duration_s": round(tt),
            "distance_m": round(ln),
            "via": via,
            "max_score": round(worst.risk.score, 1) if worst else 0.0,
            "max_level": worst.risk.level if worst else "low",
            "min_confidence": round(min(confs), 2),
            "confidence_label": min((s.risk.confidence_label for s in segs),
                                    key=lambda l: {"low": 0, "medium": 1, "high": 2}[l], default="low"),
            "static_only": any(s.risk.mode == "static_only" for s in segs),
            "alert": {
                "count": len(alert_ids),
                "exposure_m": round(exposure_m),
                "segment_ids": alert_ids,
                "first_eta": first_alert.eta.isoformat() if first_alert else None,
                "worst_segment_id": worst.risk.segment_id if worst and alert_ids else None,
                "reasons": [e.__dict__ for e in worst.risk.explanations] if worst and alert_ids else [],
            },
            "segments": seg_out,
            "_path": tuple(path),
        }
