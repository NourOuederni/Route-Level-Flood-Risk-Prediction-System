"""Weather service: owns the current RainStore and the refresh policy.

Modes
  live              Open-Meteo, checked every check_interval, fetched at most every fetch_interval
  scenario:<name>   synthetic storm relative to now (demo)
  replay:<file>     a saved raw Open-Meteo response from data/raw/weather (validation / demo)
  outage            simulate the source going down: keep last data, flag it as very old (demo, FR-15)
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


class WeatherService:
    def __init__(self, cells: list[Cell], centre: tuple[float, float], wcfg: dict,
                 raw_dir: Path, rain_day_mm: float = 1.0, mode: str = "live",
                 recorder=None):
        self.cells = cells
        self.centre = centre
        self.wcfg = wcfg
        self.raw_dir = raw_dir
        self.rain_day_mm = rain_day_mm
        self.recorder = recorder          # callable(RainStore) -> None, writes clean records to DB
        self.store: RainStore | None = None
        self.last_error: str | None = None
        self.last_attempt: datetime | None = None
        self.mode = "live"
        self._task: asyncio.Task | None = None
        self.set_mode(mode)

    # --- modes ---------------------------------------------------------------
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
            self.store.simulated_age_minutes = 30.0   # fresh relative to the replayed event
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

    # --- refresh -------------------------------------------------------------
    def needs_fetch(self, now: datetime) -> bool:
        if self.mode != "live":
            return False
        if self.store is None:
            return True
        fetch_every = timedelta(minutes=float(self.wcfg["fetch_interval_minutes"]))
        return now - self.store.retrieved_at >= fetch_every

    def refresh(self) -> None:
        now = datetime.now(timezone.utc)
        self.last_attempt = now
        try:
            store = open_meteo.fetch(self.cells, self.wcfg, self.raw_dir)
            store.rain_day_mm = self.rain_day_mm
            self.store = store
            self.last_error = None
            if self.recorder:
                self.recorder(store)
            log.info("weather refreshed: %d cells", len(store.series))
        except Exception as exc:  # keep the old cache, confidence drops with age (FR-15)
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("weather refresh failed, keeping cache: %s", self.last_error)

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

    # --- queries -------------------------------------------------------------
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
            "attribution": open_meteo.ATTRIBUTION if self.mode in ("live", "outage") or self.mode.startswith("replay") else None,
        }
        if self.store is not None:
            out.update(self.store.summary(now))
        return out
