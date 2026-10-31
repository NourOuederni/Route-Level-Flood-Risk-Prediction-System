"""Runtime settings from environment variables (development defaults)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    area: str
    database_url: str
    weather_mode: str
    admin_token: str
    nominatim_url: str
    user_agent: str


def get_settings() -> Settings:
    data_dir = Path(os.environ.get("FLOOD_DATA_DIR", ROOT / "data"))
    return Settings(
        data_dir=data_dir,
        area=os.environ.get("FLOOD_AREA", "nabeul"),
        database_url=os.environ.get("DATABASE_URL", f"sqlite:///{data_dir / 'flood.db'}"),
        # live | scenario:storm | scenario:showers | scenario:dry | replay:<file> | outage
        weather_mode=os.environ.get("WEATHER_MODE", "live"),
        admin_token=os.environ.get("ADMIN_TOKEN", "dev-admin"),
        nominatim_url=os.environ.get("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search"),
        user_agent=os.environ.get("FLOOD_USER_AGENT", "flood-route-risk-pilot/0.1 (academic prototype)"),
    )
