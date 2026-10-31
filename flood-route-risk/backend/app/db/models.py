"""SQLAlchemy + GeoAlchemy2 models for the V2 national-scale database.

Tables
  segment          Pre-computed static factors and geometry for every road segment in the country.
                   Written once by the pipeline, read at request time.
  weather_cell     Grid cells for weather lookups.
  prediction_log   Same as V1 — logged predictions for auditing and replay validation.
  feedback         Same as V1 — crowdsourced flood/drain reports.
  flood_event      Same as V1 — authoritative past flood events.
  hotspot_cache    Pre-computed hotspot snapshots (refreshed every weather update).
"""
from __future__ import annotations

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float, Index, Integer, String, Text,
    create_engine, text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from geoalchemy2 import Geometry


class Base(DeclarativeBase):
    pass


class Segment(Base):
    """One <= 200 m road segment with pre-computed static risk factors."""
    __tablename__ = "segment"

    segment_id = Column(BigInteger, primary_key=True)
    tile_id = Column(String(20), nullable=False, index=True)
    edge_uid = Column(Text, nullable=False, index=True)
    seq = Column(Integer, nullable=False)
    road_type = Column(String(40))
    name = Column(Text)
    length_m = Column(Float, nullable=False)

    # Geometry in WGS84 for spatial queries and in UTM 32N for measurements
    geom = Column(Geometry("LINESTRING", srid=4326), nullable=False)

    # Midpoint for fast point lookups and weather cell assignment
    mid_lat = Column(Float, nullable=False)
    mid_lon = Column(Float, nullable=False)
    cell_id = Column(String(20), index=True)

    # Raw terrain measurements (for QA and recalibration)
    depth_in_sink_m = Column(Float)
    twi_mean = Column(Float)
    hand_m = Column(Float)
    slope_deg = Column(Float)
    dist_stream_m = Column(Float)
    impervious_share = Column(Float)

    # Normalised static factors f_i in [0,1]  (guide Section 7.2)
    f_depression = Column(Float, nullable=False, default=0.0)
    f_drainage_position = Column(Float, nullable=False, default=0.0)
    f_low_slope = Column(Float, nullable=False, default=0.0)
    f_impervious = Column(Float, nullable=False, default=0.0)
    f_crossing = Column(Float, nullable=False, default=0.0)
    f_past_flood = Column(Float, nullable=False, default=0.0)

    # Pre-computed static susceptibility S' = s_min + (1 - s_min) * S
    susceptibility = Column(Float, nullable=False)           # S  in [0,1]
    susceptibility_floored = Column(Float, nullable=False)   # S' in [s_min, 1]

    # Drainage vulnerability proxies
    dv_urban = Column(Float, nullable=False, default=0.0)
    dv_blackspot = Column(Float, nullable=False, default=0.0)
    dv_drain_density_inv = Column(Float, nullable=False, default=0.5)
    dv_maintenance = Column(Float)    # NULL = no authority data

    # Boolean flags
    is_underpass = Column(Boolean, default=False)
    is_ford = Column(Boolean, default=False)
    is_bridge = Column(Boolean, default=False)
    wadi_crossing = Column(Boolean, default=False)
    poorly_mapped = Column(Boolean, default=False)
    past_flood_confirmed = Column(Boolean, default=False)
    crossing_kind = Column(String(20))   # underpass | ford | low_crossing | NULL

    __table_args__ = (
        Index("ix_segment_geom", "geom", postgresql_using="gist"),
        Index("ix_segment_mid", "mid_lat", "mid_lon"),
    )


class WeatherCell(Base):
    """Weather grid cell — the grid the weather service polls."""
    __tablename__ = "weather_cell"
    cell_id = Column(String(20), primary_key=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)


class HotspotCache(Base):
    """Pre-computed hotspot for current conditions, refreshed every weather update."""
    __tablename__ = "hotspot_cache"
    segment_id = Column(BigInteger, primary_key=True)
    score = Column(Float, nullable=False)
    level = Column(String(12), nullable=False, index=True)
    confidence_label = Column(String(8))
    mid_lat = Column(Float, nullable=False)
    mid_lon = Column(Float, nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False)


class PredictionLog(Base):
    __tablename__ = "prediction_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(40), index=True)
    segment_id = Column(BigInteger, index=True)
    eta = Column(DateTime(timezone=True))
    score = Column(Float)
    level = Column(String(12))
    confidence = Column(Float)
    mode = Column(String(16))
    factors = Column(JSONB)
    rain_source = Column(Text)
    rain_retrieved_at = Column(DateTime(timezone=True))
    engine_version = Column(String(20))
    issued_at = Column(DateTime(timezone=True))


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, autoincrement=True)
    segment_id = Column(BigInteger, index=True)
    kind = Column(String(20), nullable=False)
    comment = Column(Text)
    lat = Column(Float)
    lon = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=text("now()"))


class FloodEvent(Base):
    __tablename__ = "flood_event"
    id = Column(Integer, primary_key=True, autoincrement=True)
    geom = Column(Geometry("POINT", srid=4326))
    date = Column(DateTime(timezone=True))
    source = Column(Text)
    reliability = Column(String(10))
    description = Column(Text)
    test_set = Column(Boolean, default=False)


# --- session factory --------------------------------------------------------
def make_engine(url: str):
    return create_engine(url, pool_size=10, max_overflow=20, pool_pre_ping=True)


def make_session(engine) -> sessionmaker:
    return sessionmaker(bind=engine)


def init_db(engine) -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
