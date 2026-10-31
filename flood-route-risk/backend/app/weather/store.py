"""In-memory rainfall store: per grid cell, hourly series per weather model.

Time convention: each hourly value is stamped with the END of its hour and holds the
precipitation accumulated during the preceding hour (Open-Meteo convention), in mm.
All timestamps are UTC unix seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ..engine.types import RainWindow

HOUR = 3600
DAY = 86400


@dataclass(frozen=True)
class Cell:
    cell_id: str
    lat: float
    lon: float


def build_grid(bbox: tuple[float, float, float, float], step: float) -> list[Cell]:
    """Regular grid of cell centres covering bbox = (west, south, east, north)."""
    w, s, e, n = bbox
    nx = max(1, math.ceil((e - w) / step))
    ny = max(1, math.ceil((n - s) / step))
    cells = []
    for j in range(ny):
        for i in range(nx):
            lat = round(s + (j + 0.5) * step, 4)
            lon = round(w + (i + 0.5) * step, 4)
            cells.append(Cell(f"r{j}c{i}", lat, lon))
    return cells


def nearest_cell(cells: list[Cell], lat: float, lon: float) -> str:
    k = math.cos(math.radians(lat))
    best = min(cells, key=lambda c: (c.lat - lat) ** 2 + ((c.lon - lon) * k) ** 2)
    return best.cell_id


@dataclass
class CellSeries:
    times: np.ndarray                          # end-of-hour unix seconds, sorted
    values: dict[str, np.ndarray]              # model -> mm per hour
    daily_dates: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    daily_mm: np.ndarray = field(default_factory=lambda: np.array([]))  # ensemble daily sums


@dataclass
class RainStore:
    source: str
    retrieved_at: datetime
    series: dict[str, CellSeries]
    interval_minutes: int = 60
    interpolated: bool = True
    simulated_age_minutes: float | None = None  # used by the simulated outage mode
    rain_day_mm: float = 1.0
    models: tuple[str, ...] = ()

    def age_minutes(self, now: datetime) -> float:
        if self.simulated_age_minutes is not None:
            return self.simulated_age_minutes
        return max(0.0, (now - self.retrieved_at).total_seconds() / 60.0)

    def dry_spell_days(self, cs: CellSeries, at: float) -> float | None:
        """Days without a wet day (>= rain_day_mm) before the current rain event.

        Measured up to 24 h before `at`, so the storm being scored does not reset its own
        dry-spell counter (sediment accumulated before the storm is what matters).
        """
        at = at - DAY
        last_wet: float | None = None
        if cs.daily_dates.size:
            mask = (cs.daily_mm >= self.rain_day_mm) & (cs.daily_dates <= at)
            if mask.any():
                last_wet = float(cs.daily_dates[mask].max()) + DAY  # end of that day
            elif cs.daily_dates.size:
                last_wet = float(cs.daily_dates.min())  # dry for the whole history
        if cs.times.size and cs.values:
            ens = np.mean(np.vstack(list(cs.values.values())), axis=0)
            # roll hourly into 24 h sums ending at each hour (only needs a rough signal)
            past = cs.times <= at
            if past.any():
                t, v = cs.times[past], ens[past]
                csum = np.concatenate([[0.0], np.cumsum(v)])
                idx = np.searchsorted(t, t - DAY, side="right")
                sums = csum[1:] - csum[idx]
                wet = sums >= self.rain_day_mm
                if wet.any():
                    last_wet = max(last_wet or 0.0, float(t[wet].max()))
                elif last_wet is None:
                    last_wet = float(t.min())
        if last_wet is None:
            return None
        return max(0.0, (at - last_wet) / DAY)

    def window(self, cell_id: str, start: datetime, end: datetime, now: datetime,
               antecedent_hours: int = 6) -> RainWindow | None:
        cs = self.series.get(cell_id)
        if cs is None or cs.times.size == 0 or not cs.values:
            return None
        t0, t1 = start.timestamp(), end.timestamp()
        overl = (cs.times > t0) & (cs.times - HOUR < t1)
        if not overl.any():
            return None  # ETA beyond forecast horizon or before history
        ante = (cs.times > t0 - antecedent_hours * HOUR) & (cs.times <= t0)
        models = tuple(cs.values.keys())
        intens = tuple(float(np.nanmax(cs.values[m][overl])) if np.isfinite(cs.values[m][overl]).any() else 0.0
                       for m in models)
        antec = tuple(float(np.nansum(cs.values[m][ante])) for m in models)
        return RainWindow(
            intensity_mm_h=intens,
            antecedent_mm=antec,
            source=self.source,
            retrieved_at=self.retrieved_at,
            age_minutes=self.age_minutes(now),
            interval_minutes=self.interval_minutes,
            interpolated=self.interpolated,
            dry_spell_days=self.dry_spell_days(cs, t0),
            models=models,
        )

    def summary(self, now: datetime) -> dict:
        return {
            "source": self.source,
            "retrieved_at": self.retrieved_at.astimezone(timezone.utc).isoformat(),
            "age_minutes": round(self.age_minutes(now), 1),
            "interval_minutes": self.interval_minutes,
            "interpolated": self.interpolated,
            "models": list(self.models),
            "cells": len(self.series),
            "simulated_outage": self.simulated_age_minutes is not None,
        }
