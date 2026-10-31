from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["low", "moderate", "high", "very_high"]


class LatLon(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class RouteRequest(BaseModel):
    origin: LatLon
    destination: LatLon
    departure_time: datetime | None = None   # default: now
    alert_level: Level = "high"


class FeedbackRequest(BaseModel):
    kind: Literal["flooded", "not_flooded", "drain_blocked"]
    lat: float
    lon: float
    segment_id: int | None = None
    comment: str | None = Field(default=None, max_length=500)


class WeatherModeRequest(BaseModel):
    mode: str = Field(examples=["live", "scenario:storm", "outage", "replay:openmeteo_20261030T120000Z.json"])
