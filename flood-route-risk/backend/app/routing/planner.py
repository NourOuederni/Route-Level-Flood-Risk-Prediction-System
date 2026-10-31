"""Route planning with per-segment flood risk (FR-01..FR-04, FR-16..FR-19).

Steps for one request
  1. snap origin/destination to the graph
  2. up to N candidate routes (penalty method on travel time)
  3. per route: split into segments, propagate ETA, score each segment at its ETA window
  4. if the fastest route reaches the alert level: risk-aware search where
       cost = travel_time * (1 + penalty[level]), excluded levels removed,
     levels estimated at the ETA each edge would have (Dijkstra from the origin),
     then the result is re-scored with its true ETAs (second pass)
  5. if no acceptable alternative exists: keep the original route and warn (FR-19)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import networkx as nx

from ..engine import EvalContext, SegmentRisk, evaluate_segment, level_at_least
from ..engine.config import LEVEL_RANK, EngineConfig
from .area import Area, SegmentRecord


class PlanningError(ValueError):
    pass


@dataclass
class RoutedSegment:
    rec: SegmentRecord
    eta: datetime
    risk: SegmentRisk


class RiskScorer:
    """Scores segments at an ETA window, with a per-request memo."""

    def __init__(self, area: Area, weather, cfg: EngineConfig, now: datetime,
                 feedback: dict[int, str] | None = None):
        self.area, self.weather, self.cfg, self.now = area, weather, cfg, now
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


class Planner:
    def __init__(self, area: Area, weather, cfg: EngineConfig):
        self.area, self.weather, self.cfg = area, weather, cfg
        self.rcfg = cfg.section("routing")

    # --- graph helpers -------------------------------------------------------
    def _snap(self, lat: float, lon: float, what: str):
        if not self.area.in_bbox(lat, lon):
            raise PlanningError(f"{what} is outside the pilot area")
        node, dist = self.area.nearest_node(lat, lon)
        if dist > float(self.rcfg["max_snap_distance_m"]):
            raise PlanningError(f"{what} is too far from any mapped road ({dist:.0f} m)")
        return node

    def _path_length(self, path) -> tuple[float, float]:
        g = self.area.graph
        tt = sum(g.edges[u, v]["travel_time"] for u, v in zip(path, path[1:]))
        ln = sum(g.edges[u, v]["length"] for u, v in zip(path, path[1:]))
        return tt, ln

    def _candidates(self, o, d) -> list[list]:
        g = self.area.graph
        try:
            best = nx.shortest_path(g, o, d, weight="travel_time")
        except nx.NetworkXNoPath as exc:
            raise PlanningError("no drivable path between origin and destination") from exc
        accepted = [best]
        best_tt, _ = self._path_length(best)
        penal: dict[tuple, float] = {}
        factor = float(self.rcfg["candidate_penalty"])
        max_n = int(self.rcfg["max_candidates"])
        for _ in range(max_n * 3):
            if len(accepted) >= max_n:
                break
            for p in accepted:
                for e in zip(p, p[1:]):
                    penal[e] = penal.get(e, 1.0) * factor
            try:
                cand = nx.shortest_path(
                    g, o, d, weight=lambda u, v, dd: dd["travel_time"] * penal.get((u, v), 1.0))
            except nx.NetworkXNoPath:
                break
            tt, ln = self._path_length(cand)
            if tt > best_tt * float(self.rcfg["candidate_duration_factor"]):
                break
            if all(self._overlap(cand, p) < float(self.rcfg["candidate_overlap_max"]) for p in accepted):
                accepted.append(cand)
        return accepted

    def _overlap(self, a, b) -> float:
        g = self.area.graph
        eb = set(zip(b, b[1:]))
        shared = sum(g.edges[e]["length"] for e in zip(a, a[1:]) if e in eb)
        total = sum(g.edges[e]["length"] for e in zip(a, a[1:])) or 1.0
        return shared / total

    # --- scoring a path ------------------------------------------------------
    def _route_segments(self, path, departure: datetime, scorer: RiskScorer) -> list[RoutedSegment]:
        g = self.area.graph
        out: list[RoutedSegment] = []
        elapsed = 0.0
        for u, v in zip(path, path[1:]):
            d = g.edges[u, v]
            recs = self.area.segments_for_edge(u, v)
            total_len = sum(r.length_m for r in recs) or 1.0
            for rec in recs:
                dt = d["travel_time"] * rec.length_m / total_len
                eta_s = elapsed + dt / 2          # ETA at the middle of the segment
                eta = departure + timedelta(seconds=eta_s)
                out.append(RoutedSegment(rec, eta, scorer.score(rec, eta, eta_s)))
                elapsed += dt
            if not recs:
                elapsed += d["travel_time"]
        return out

    def _risk_aware_path(self, o, d, departure: datetime, scorer: RiskScorer,
                         fastest_tt: float, alert_level: str):
        g = self.area.graph
        cutoff = fastest_tt * float(self.rcfg["corridor_factor"])
        arrival = nx.single_source_dijkstra_path_length(g, o, cutoff=cutoff, weight="travel_time")
        base = self.rcfg["risk_penalty"]
        # every level the user wants to be alerted about gets at least the "high" penalty
        pen = {lvl: (max(float(base.get(lvl, 0.0)), float(base["high"]))
                     if level_at_least(lvl, alert_level) else float(base.get(lvl, 0.0)))
               for lvl in LEVEL_RANK}
        excl = set(self.rcfg["exclude_levels"])
        edge_level: dict[tuple, str] = {}
        for u, t_u in arrival.items():
            eta = departure + timedelta(seconds=t_u)
            for v in g.successors(u):
                worst = "low"
                for rec in self.area.segments_for_edge(u, v):
                    lvl = scorer.score(rec, eta, t_u).level
                    if LEVEL_RANK[lvl] > LEVEL_RANK[worst]:
                        worst = lvl
                edge_level[(u, v)] = worst

        def cost(u, v, dd):
            lvl = edge_level.get((u, v), "low")
            if lvl in excl:
                return None                     # excluded: networkx hides the edge
            return dd["travel_time"] * (1.0 + float(pen.get(lvl, 0.0)))

        try:
            return nx.shortest_path(g, o, d, weight=cost)
        except nx.NetworkXNoPath:
            return None

    # --- public --------------------------------------------------------------
    def plan(self, origin: tuple[float, float], destination: tuple[float, float],
             departure: datetime | None, alert_level: str,
             feedback: dict[int, str] | None = None) -> tuple[dict, list[dict]]:
        now = datetime.now(timezone.utc)
        departure = (departure or now).astimezone(timezone.utc)
        o = self._snap(*origin, "origin")
        d = self._snap(*destination, "destination")
        if o == d:
            raise PlanningError("origin and destination are the same place")
        scorer = RiskScorer(self.area, self.weather, self.cfg, now, feedback)

        paths = self._candidates(o, d)
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
                segs = self._route_segments(rp, departure, scorer)      # second pass, true ETAs
                cand = self._summarise(rp, segs, "safer", alert_level)
                excl = set(self.rcfg["exclude_levels"])
                still_excluded = any(s["level"] in excl for s in cand["segments"])
                # better = clearly less road at/above the alert level, or a lower worst level
                improved = (cand["alert"]["exposure_m"] <= 0.9 * fastest["alert"]["exposure_m"]
                            or LEVEL_RANK[cand["max_level"]] < LEVEL_RANK[fastest["max_level"]])
                # an "alternative" that is barely different is not worth showing
                improved = improved and cand["_path"] != fastest["_path"]
                if improved and not still_excluded:
                    cand["extra_duration_s"] = round(cand["duration_s"] - fastest["duration_s"])
                    cand["extra_distance_m"] = round(cand["distance_m"] - fastest["distance_m"])
                    safer = cand
                    # if an existing candidate is the same path, label it rather than duplicating
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
        }
        return result, log_rows

    def _summarise(self, path, segs: list[RoutedSegment], kind: str, alert_level: str) -> dict:
        tt, ln = self._path_length(path)
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
