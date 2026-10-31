"""V2 national weather service: polls Open-Meteo for a 0.25° grid covering all Tunisia.

Same algorithmic approach as V1 — no AI. Just polls more grid cells (~500 vs ~20).
Uses the same RainStore and Open-Meteo client; only the grid and refresh logic differ.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..engine.types import RainWindow
from . import open_meteo, scenarios
from .store import Cell, RainStore

log = logging.getLogger(__name__)

# Tunisia bounding box and grid
TUNISIA_BBOX = [7.5, 30.2, 11.6, 37.5]
GRID_STEP = 0.25              # 0.25° grid ~ 500 cells


def build_national_grid() -> list[Cell]:
    """Build the weather grid covering all Tunisia at 0.25° spacing."""
    w, s, e, n = TUNISIA_BBOX
    cells = []
    lat = s
    while lat <= n:
        lon = w
        while lon <= e:
            cell_id = f"{lon:.2f}_{lat:.2f}"
            cells.append(Cell(cell_id=cell_id, lat=round(lat, 4), lon=round(lon, 4)))
            lon += GRID_STEP
        lat += GRID_STEP
    return cells


class NationalWeatherService:
    """Weather service scaled to cover all of Tunisia.

    Same modes as V1 (live, scenario, replay, outage). Polls Open-Meteo
    in batches (API supports up to ~30 locations per call), cycling through
    the ~500-cell grid so the whole country refreshes every few minutes.
    """

    def __init__(self, wcfg: dict, raw_dir: Path, rain_day_mm: float = 1.0,
                 mode: str = "live", recorder=None):
        self.cells = build_national_grid()
        self.centre = (33.8, 9.5)     # Tunisia centre
        self.wcfg = wcfg
        self.raw_dir = raw_dir
        self.rain_day_mm = rain_day_mm
        self.recorder = recorder
        self.store: RainStore | None = None
        self.last_error: str | None = None
        self.last_attempt: datetime | None = None
        self.mode = "live"
        self._task: asyncio.Task | None = None
        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        now = datetime.now(timezone.utc)
        if mode.startswith("scenario:"):
            name = mode.split(":", 1)[1]
            if name not in scenarios.SCENARIOS:
                raise ValueError(f"unknown scenario {name!r}")
            self.store = scenarios.build_store(name, self.cells, self.centre, now, self.rain_day_mm)
        elif mode.startswith("replay:"):
            path = self.raw_dir / mode.split(":", 1)[1]
            if not path.is_file():
                raise ValueError(f"replay file not found: {path.name}")
            self.store = open_meteo.load_raw(path)
            self.store.simulated_age_minutes = 30.0
        elif mode == "outage":
            if self.store is not None:
                refresh = float(self.wcfg["refresh_interval_minutes"])
                self.store.simulated_age_minutes = 4 * refresh
        elif mode == "live":
            if self.store is not None:
                self.store.simulated_age_minutes = None
                if not self.store.source.startswith("open-meteo"):
                    self.store = None
        else:
            raise ValueError(f"unknown weather mode {mode!r}")
        self.mode = mode
        self.last_error = None

    def available_replays(self) -> list[str]:
        if not self.raw_dir.is_dir():
            return []
        return sorted(p.name for p in self.raw_dir.glob("openmeteo_*.json"))[-20:]

    def needs_fetch(self, now: datetime) -> bool:
        if self.mode != "live":
            return False
        if self.store is None:
            return True
        fetch_every = timedelta(minutes=float(self.wcfg["fetch_interval_minutes"]))
        return now - self.store.retrieved_at >= fetch_every

    def refresh(self) -> None:
        """Fetch weather for all ~500 cells. Open-Meteo handles batching internally."""
        now = datetime.now(timezone.utc)
        self.last_attempt = now
        try:
            store = open_meteo.fetch(self.cells, self.wcfg, self.raw_dir)
            store.rain_day_mm = self.rain_day_mm
            self.store = store
            self.last_error = None
            if self.recorder:
                self.recorder(store)
            log.info("national weather refreshed: %d cells", len(store.series))
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("national weather refresh failed, keeping cache: %s", self.last_error)

    async def run(self) -> None:
        interval = float(self.wcfg["check_interval_minutes"]) * 60
        while True:
            if self.needs_fetch(datetime.now(timezone.utc)):
                await asyncio.to_thread(self.refresh)
            await asyncio.sleep(interval)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    def window(self, cell_id: str, start: datetime, end: datetime, now: datetime,
               antecedent_hours: int) -> RainWindow | None:
        if self.store is None:
            return None
        return self.store.window(cell_id, start, end, now, antecedent_hours)

    def status(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        out = {
            "mode": self.mode,
            "simulated": self.mode != "live",
            "replay": self.mode.startswith("replay:"),
            "available": self.store is not None,
            "last_error": self.last_error,
            "cells": len(self.cells),
            "attribution": open_meteo.ATTRIBUTION if self.mode in ("live", "outage") or self.mode.startswith("replay") else None,
        }
        if self.store is not None:
            out.update(self.store.summary(now))
        return out
