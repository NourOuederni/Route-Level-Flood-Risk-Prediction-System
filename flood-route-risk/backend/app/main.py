"""FastAPI application - development server for the Version 1 pilot.

Run:  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .api.schemas import FeedbackRequest, RouteRequest, WeatherModeRequest
from .engine import default_config
from .engine.config import LEVEL_RANK
from .routing.area import Area
from .routing.planner import Planner, PlanningError, RiskScorer
from .settings import get_settings
from .storage.db import Database
from .weather.scenarios import SCENARIOS
from .weather.service import WeatherService

log = logging.getLogger("flood")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class State:
    settings = None
    cfg = None
    area: Area | None = None
    weather: WeatherService | None = None
    planner: Planner | None = None
    db: Database | None = None
    hotspot_cache: tuple[float, str, list] | None = None
    geocode_cache: dict[str, list] = {}


S = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    S.settings = get_settings()
    S.cfg = default_config()
    area_dir = S.settings.data_dir / "processed" / S.settings.area
    if not (area_dir / "area.json").is_file():
        raise RuntimeError(
            f"No processed area at {area_dir}. Run the pipeline "
            f"(python -m pipeline.run --area {S.settings.area}) or the offline demo "
            f"(python -m pipeline.make_demo_area).")
    S.settings.data_dir.mkdir(parents=True, exist_ok=True)
    S.db = Database(S.settings.database_url)
    wcfg = S.cfg.section("weather")
    S.area = Area(area_dir, float(wcfg["grid_step_deg"]))
    S.weather = WeatherService(
        S.area.cells, S.area.centre, wcfg, S.settings.data_dir / "raw" / "weather",
        rain_day_mm=float(S.cfg.section("drainage")["dry_spell"]["rain_day_mm"]),
        mode=S.settings.weather_mode, recorder=S.db.record_rain)
    S.planner = Planner(S.area, S.weather, S.cfg)
    S.weather.start()
    log.info("area %s: %d segments, %d grid cells, weather mode %s",
             S.area.meta["name"], len(S.area.segments), len(S.area.cells), S.weather.mode)
    yield
    await S.weather.stop()


app = FastAPI(title="Flood Route Risk - V1 pilot API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _require_admin(token: str | None) -> None:
    if token != S.settings.admin_token:
        raise HTTPException(status_code=403, detail="admin token required")


@app.get("/health")
def health():
    return {"status": "ok", "segments": len(S.area.segments), "weather": S.weather.status()}


@app.get("/meta")
def meta():
    """Everything the client needs at start-up: area, levels, attributions, demo modes."""
    m = S.area.meta
    return {
        "area": {"name": m["name"], "bbox": m["bbox"], "centre": m["centre"],
                 "synthetic": m.get("synthetic", False)},
        "engine_version": S.cfg.version,
        "levels": S.cfg.level_cuts,
        "default_alert_level": S.cfg.section("alerts")["default_level"],
        "reevaluate_minutes": 5,
        "weather": S.weather.status(),
        "weather_modes": ["live", "outage"] + [f"scenario:{k}" for k in SCENARIOS]
                         + [f"replay:{f}" for f in S.weather.available_replays()],
        "scenarios": {k: v.description for k, v in SCENARIOS.items()},
        "attributions": m.get("sources", []) + ([S.weather.status()["attribution"]]
                                                if S.weather.status().get("attribution") else []),
        "disclaimer": S.cfg.raw["disclaimer"],
    }


@app.post("/routes")
def routes(req: RouteRequest):
    t0 = time.perf_counter()
    since = datetime.now(timezone.utc) - timedelta(hours=6)
    try:
        result, log_rows = S.planner.plan(
            (req.origin.lat, req.origin.lon), (req.destination.lat, req.destination.lon),
            req.departure_time, req.alert_level,
            feedback=S.db.recent_feedback(None, since))   # consistent recent reports -> confidence bonus
    except PlanningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    S.db.log_predictions(r | {"rain_source": S.weather.status().get("source"),
                              "rain_retrieved_at": S.weather.store.retrieved_at if S.weather.store else None,
                              "engine_version": S.cfg.version} for r in log_rows)
    result["weather"] = S.weather.status()
    result["compute_ms"] = round((time.perf_counter() - t0) * 1000)
    return result


@app.get("/segments/{segment_id}")
def segment(segment_id: int, eta: datetime | None = None):
    rec = S.area.segments.get(segment_id)
    if rec is None:
        raise HTTPException(404, "unknown segment")
    now = datetime.now(timezone.utc)
    scorer = RiskScorer(S.area, S.weather, S.cfg, now)
    at = (eta or now).astimezone(timezone.utc)
    risk = scorer.score(rec, at, 0.0)
    return risk.to_dict() | {
        "eta": at.isoformat(),
        "name": rec.name, "road_type": rec.road_type, "crossing": rec.static.crossing_kind,
        "coords": rec.coords, "cell_id": rec.cell_id, "static": rec.static.__dict__,
        "weather": S.weather.status()}


@app.get("/hotspots")
def hotspots(min_level: str = Query("moderate", pattern="^(moderate|high|very_high)$"),
             limit: int = Query(400, le=2000)):
    """Segments at or above a level right now - the dots on the map."""
    now = datetime.now(timezone.utc)
    key = f"{S.weather.mode}|{S.weather.status().get('retrieved_at')}|{min_level}"
    if S.hotspot_cache and S.hotspot_cache[1] == key and time.time() - S.hotspot_cache[0] < 300:
        return {"items": S.hotspot_cache[2][:limit], "cached": True}
    scorer = RiskScorer(S.area, S.weather, S.cfg, now)
    items = []
    for rec in S.area.segments.values():
        r = scorer.score(rec, now, 0.0)
        if LEVEL_RANK[r.level] >= LEVEL_RANK[min_level]:
            items.append({"segment_id": rec.static.segment_id, "lat": rec.mid[0], "lon": rec.mid[1],
                          "score": round(r.score, 1), "level": r.level,
                          "confidence_label": r.confidence_label})
    items.sort(key=lambda x: -x["score"])
    S.hotspot_cache = (time.time(), key, items)
    return {"items": items[:limit], "cached": False}


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    sid = req.segment_id
    if sid is None:  # snap to the nearest segment midpoint (within 60 m)
        best, best_d = None, 60.0
        for rec in S.area.segments.values():
            d = ((rec.mid[0] - req.lat) * 110_570) ** 2 + ((rec.mid[1] - req.lon) * 89_000) ** 2
            if d < best_d ** 2:
                best, best_d = rec.static.segment_id, d ** 0.5
        sid = best
    fid = S.db.add_feedback({"segment_id": sid, "kind": req.kind, "comment": req.comment,
                             "lat": round(req.lat, 4), "lon": round(req.lon, 4)})
    return {"id": fid, "segment_id": sid}


@app.get("/geocode")
def geocode(q: str = Query(min_length=2, max_length=120)):
    """Search a place inside the pilot area (Nominatim, bounded, cached; light dev use only)."""
    key = q.strip().lower()
    if key in S.geocode_cache:
        return {"items": S.geocode_cache[key]}
    w, s, e, n = S.area.meta["bbox"]
    try:
        r = httpx.get(S.settings.nominatim_url, timeout=8, headers={"User-Agent": S.settings.user_agent},
                      params={"q": q, "format": "jsonv2", "limit": 6, "bounded": 1,
                              "viewbox": f"{w},{n},{e},{s}", "accept-language": "fr,ar,en"})
        r.raise_for_status()
        items = [{"name": it.get("name") or it["display_name"].split(",")[0],
                  "address": it["display_name"], "lat": float(it["lat"]), "lon": float(it["lon"])}
                 for it in r.json()]
    except Exception as exc:
        raise HTTPException(503, f"geocoding unavailable: {exc}") from exc
    S.geocode_cache[key] = items
    return {"items": items}


# --- development / admin --------------------------------------------------
@app.get("/admin/weather")
def weather_status():
    return S.weather.status() | {"modes": ["live", "outage"] + [f"scenario:{k}" for k in SCENARIOS]}


@app.post("/admin/weather")
def set_weather(req: WeatherModeRequest, x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    try:
        S.weather.set_mode(req.mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    S.hotspot_cache = None
    if S.weather.mode == "live" and S.weather.store is None:
        S.weather.refresh()
    return S.weather.status()
