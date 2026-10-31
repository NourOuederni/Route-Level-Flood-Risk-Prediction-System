"""Synthetic rainfall scenarios for demos and tests (guide 2.3: replay / simulate a heavy-rain case).

Scenarios are generated relative to "now" so the demo works at any time. They are clearly
labelled as simulated in every API response.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .store import DAY, HOUR, Cell, CellSeries, RainStore

MODEL_FACTORS = {"model_a": 0.8, "model_b": 1.0, "model_c": 1.25}  # fake ensemble spread


@dataclass(frozen=True)
class Storm:
    peak_mm_h: float
    peak_offset_h: float        # hours from now
    duration_sigma_h: float     # gaussian width in time
    radius_km: float            # gaussian width in space
    centre_offset_km: tuple[float, float] = (0.0, 0.0)   # (east, north) from area centre
    motion_km_h: tuple[float, float] = (0.0, 0.0)        # (east, north) storm speed
    background_mm_h: float = 0.0


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    storms: tuple[Storm, ...]
    dry_spell_days: float


SCENARIOS: dict[str, Scenario] = {
    "dry": Scenario("dry", "No rain. Every segment should be Low.", (), 3.0),
    "showers": Scenario(
        "showers", "Light-to-moderate showers after a short dry spell.",
        (Storm(peak_mm_h=8, peak_offset_h=1.0, duration_sigma_h=2.0, radius_km=15, background_mm_h=0.5),),
        6.0),
    "storm": Scenario(
        "storm", "First autumn convective storm after a long dry summer (drains likely clogged).",
        (Storm(peak_mm_h=32, peak_offset_h=0.75, duration_sigma_h=1.2, radius_km=6,
               centre_offset_km=(-1.5, 0.5), motion_km_h=(2.0, 0.5), background_mm_h=1.0),),
        45.0),
}


def _km_offsets(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    return ((lon - lon0) * 111.32 * math.cos(math.radians(lat0)), (lat - lat0) * 110.57)


def build_store(name: str, cells: list[Cell], centre: tuple[float, float],
                now: datetime | None = None, rain_day_mm: float = 1.0) -> RainStore:
    sc = SCENARIOS[name]
    now = now or datetime.now(timezone.utc)
    base = int(now.timestamp() // HOUR * HOUR)
    times = np.arange(base - 48 * HOUR, base + 49 * HOUR, HOUR, dtype=np.int64)
    rel_h = (times - now.timestamp()) / HOUR - 0.5     # centre of each hourly interval
    lat0, lon0 = centre
    series: dict[str, CellSeries] = {}
    for c in cells:
        dx, dy = _km_offsets(lat0, lon0, c.lat, c.lon)
        rain = np.zeros_like(rel_h)
        for st in sc.storms:
            cx = st.centre_offset_km[0] + st.motion_km_h[0] * (rel_h - st.peak_offset_h)
            cy = st.centre_offset_km[1] + st.motion_km_h[1] * (rel_h - st.peak_offset_h)
            d2 = (dx - cx) ** 2 + (dy - cy) ** 2
            temporal = np.exp(-0.5 * ((rel_h - st.peak_offset_h) / st.duration_sigma_h) ** 2)
            spatial = np.exp(-0.5 * d2 / st.radius_km ** 2)
            rain += st.peak_mm_h * temporal * spatial + st.background_mm_h * temporal
        rain[rel_h < -24] = 0.0
        values = {m: np.round(rain * f, 2) for m, f in MODEL_FACTORS.items()}
        days = np.arange(int(now.timestamp() // DAY * DAY) - 60 * DAY,
                         int(now.timestamp() // DAY * DAY), DAY, dtype=np.int64)
        daily = np.zeros(days.size)
        last_wet = now.timestamp() - sc.dry_spell_days * DAY
        daily[days <= last_wet - DAY] = 0.0
        daily[(days > last_wet - DAY) & (days <= last_wet)] = 5.0
        series[c.cell_id] = CellSeries(times=times, values=values, daily_dates=days, daily_mm=daily)
    return RainStore(source=f"scenario:{name}", retrieved_at=now, series=series,
                     interval_minutes=60, interpolated=True, rain_day_mm=rain_day_mm,
                     models=tuple(MODEL_FACTORS))
