"""Admin dashboard: model validation results, live zone status and alert history.

GET /admin/metrics    JSON for admins (Bearer token from /auth/login)
GET /admin/dashboard  HTML page (app/admin_dashboard.html) that shows the metrics

Live accuracy is not computed here: the system has no confirmed flood labels yet. The model's
validation results (fitted and checked offline, see models/final/trigger.json) are shown as published.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import select

from . import accounts
from .accounts import AlertEvent, CurrentUser, PushToken, User, require_admin

router = APIRouter()
_ctx: dict = {"S": None, "monitor": None}
_PAGE = Path(__file__).with_name("admin_dashboard.html")
_EVAL = Path(__file__).resolve().parent.parent / "models" / "final" / "evaluation.json"


def _evaluation() -> dict | None:
    """Threshold sweeps and validation counts (built from reports/, see models/final/evaluation.json)."""
    try:
        return json.loads(_EVAL.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def init(state, monitor) -> None:
    """Called from main at startup so the dashboard can read the live state."""
    _ctx["S"] = state
    _ctx["monitor"] = monitor


@router.get("/admin/metrics")
def metrics(user: CurrentUser = Depends(require_admin)):
    S = _ctx["S"]
    zone = S.planner.zone if S and S.planner else None
    now = datetime.now(timezone.utc)
    trig = zone.trigger(S.weather.store, now) if zone and S.weather else None

    levels: dict[str, int] = {}
    validation: dict = {}
    if zone is not None:
        levels = dict(Counter(r["level"] for r in zone.rows))
        validation = zone.trigger_cfg

    with accounts._sf()() as db:
        recent = db.execute(select(AlertEvent).order_by(AlertEvent.started_at.desc()).limit(20)).scalars().all()
        users = len(db.execute(select(User.id)).all())
        devices = len(db.execute(select(PushToken.id)).all())

    weather = S.weather.status() if S and S.weather else None
    return {
        "model": {
            "name": "v2_terrain (zone terrain classifier + rain trigger)",
            "zone_segments": len(zone.rows) if zone else 0,
            "level_counts": levels,
            "trigger_threshold_mm": zone.threshold_mm if zone else None,
            "trigger_status": validation.get("status", "provisional") if zone else None,
            "validation": validation.get("validation", {}),
            "limits": validation.get("limits", []),
            "live_accuracy": "not measured: no confirmed flood labels yet",
        },
        "live": {
            "trigger": trig,
            "monitor_last_checked": _ctx["monitor"].last_checked if _ctx["monitor"] else None,
            "weather": weather,
            "routes_served_since_start": S.routes_served if S else 0,
            "users": users,
            "devices_for_alerts": devices,
        },
        "evaluation": _evaluation(),
        "alerts": [{"started_at": a.started_at.isoformat(), "max_rain_24h_mm": a.max_rain_24h_mm,
                    "threshold_mm": a.threshold_mm, "devices_notified": a.pushes_sent} for a in recent],
        "requested_by": user.email,
    }


@router.get("/admin/dashboard", response_class=HTMLResponse)
def dashboard_page():
    return _PAGE.read_text(encoding="utf-8")
