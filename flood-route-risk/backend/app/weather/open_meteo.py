"""Open-Meteo client: one multi-location call per poll for the whole pilot grid (guide 4.5).

Free tier is non-commercial; data are CC BY 4.0: credit "Weather data by Open-Meteo.com".
We request a fixed grid and share the cache - never one call per segment or per user.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .store import CellSeries, Cell, RainStore

log = logging.getLogger(__name__)
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"


def _as_list(payload: Any) -> list[dict]:
    return payload if isinstance(payload, list) else [payload]


def _precip_keys(block: dict, models: list[str]) -> dict[str, str]:
    """Map model -> response key. Single-model answers use plain 'precipitation'."""
    keys = {}
    for m in models:
        if f"precipitation_{m}" in block:
            keys[m] = f"precipitation_{m}"
    if not keys and "precipitation" in block:
        keys[models[0] if models else "default"] = "precipitation"
    return keys


def parse_forecast(payload: Any, cells: list[Cell], models: list[str]) -> dict[str, CellSeries]:
    out: dict[str, CellSeries] = {}
    for cell, loc in zip(cells, _as_list(payload)):
        hourly = loc.get("hourly", {})
        times = np.asarray(hourly.get("time", []), dtype=np.int64)
        values = {}
        for model, key in _precip_keys(hourly, models).items():
            arr = np.asarray([np.nan if v is None else float(v) for v in hourly[key]], dtype=float)
            if np.isfinite(arr).any():          # drop models with no data for this point
                values[model] = arr
        out[cell.cell_id] = CellSeries(times=times, values=values)
    return out


def parse_daily(payload: Any, cells: list[Cell], into: dict[str, CellSeries]) -> None:
    for cell, loc in zip(cells, _as_list(payload)):
        daily = loc.get("daily", {})
        dates = np.asarray(daily.get("time", []), dtype=np.int64)
        sums = [v for k, v in daily.items() if k.startswith("precipitation_sum")]
        if cell.cell_id in into and dates.size and sums:
            arr = np.nanmean(np.array([[np.nan if x is None else x for x in s] for s in sums], dtype=float), axis=0)
            into[cell.cell_id].daily_dates = dates
            into[cell.cell_id].daily_mm = np.nan_to_num(arr)


def fetch(cells: list[Cell], wcfg: dict, raw_dir: Path | None = None) -> RainStore:
    """Fetch hourly forecast (+ recent past) and daily history for dry-spell computation."""
    models = list(wcfg["models"])
    common = {
        "latitude": ",".join(str(c.lat) for c in cells),
        "longitude": ",".join(str(c.lon) for c in cells),
        "timeformat": "unixtime",
        "timezone": "GMT",
    }
    retrieved = datetime.now(timezone.utc)
    with httpx.Client(timeout=wcfg.get("timeout_s", 20)) as client:
        r1 = client.get(wcfg["url"], params={
            **common, "hourly": "precipitation", "models": ",".join(models),
            "past_days": wcfg["past_days"], "forecast_days": wcfg["forecast_days"]})
        r1.raise_for_status()
        hourly_payload = r1.json()
        r2 = client.get(wcfg["url"], params={
            **common, "daily": "precipitation_sum",
            "past_days": wcfg["dry_spell_past_days"], "forecast_days": 1})
        r2.raise_for_status()
        daily_payload = r2.json()

    if raw_dir is not None:  # keep raw responses: replay and validation depend on them
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = retrieved.strftime("%Y%m%dT%H%M%SZ")
        (raw_dir / f"openmeteo_{stamp}.json").write_text(json.dumps({
            "retrieved_at": retrieved.isoformat(), "models": models,
            "cells": [c.__dict__ for c in cells],
            "hourly": hourly_payload, "daily": daily_payload}), encoding="utf-8")

    series = parse_forecast(hourly_payload, cells, models)
    parse_daily(daily_payload, cells, series)
    _check_interval(hourly_payload)
    return RainStore(source="open-meteo", retrieved_at=retrieved, series=series,
                     interval_minutes=60, interpolated=True, models=tuple(models))


def load_raw(path: Path) -> RainStore:
    """Rebuild a store from a saved raw response (replay mode / validation)."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    cells = [Cell(**c) for c in doc["cells"]]
    series = parse_forecast(doc["hourly"], cells, doc["models"])
    if doc.get("daily"):
        parse_daily(doc["daily"], cells, series)
    return RainStore(source=f"replay:{path.name}",
                     retrieved_at=datetime.fromisoformat(doc["retrieved_at"]),
                     series=series, interval_minutes=60, interpolated=True,
                     models=tuple(doc["models"]))


def _check_interval(payload: Any) -> None:
    """Read the interval metadata instead of assuming it (guide 4.5)."""
    first = _as_list(payload)[0] if payload else {}
    units = first.get("hourly_units", {})
    for k, v in units.items():
        if k.startswith("precipitation") and v != "mm":
            log.warning("unexpected precipitation unit %s=%s", k, v)
    times = first.get("hourly", {}).get("time", [])
    if len(times) > 1 and int(times[1]) - int(times[0]) != 3600:
        log.warning("hourly interval is not 60 minutes: %s s", int(times[1]) - int(times[0]))
