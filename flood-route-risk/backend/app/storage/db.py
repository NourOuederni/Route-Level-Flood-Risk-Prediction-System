"""Logs and inventories (FR-22, FR-23, FR-21).

SQLAlchemy Core so the same code runs on SQLite (default, zero setup) and on
PostgreSQL/PostGIS (set DATABASE_URL). See sql/postgis_schema.sql for the spatial
version of these tables. Privacy (NFR-07): no user id, no origin/destination and no
trip trace is ever stored - only segment ids, scores and timestamps.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, Integer, MetaData, String,
                        Table, Text, create_engine, func, insert, select)
from sqlalchemy.engine import Engine

metadata = MetaData()

prediction_log = Table(
    "prediction_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("request_id", String(36), index=True),
    Column("segment_id", Integer, index=True),
    Column("eta", DateTime(timezone=True), index=True),
    Column("score", Float), Column("level", String(16)), Column("confidence", Float),
    Column("mode", String(16)),
    Column("factors", JSON),
    Column("rain_source", String(64)),
    Column("rain_retrieved_at", DateTime(timezone=True)),
    Column("engine_version", String(32)),
    Column("issued_at", DateTime(timezone=True), index=True),
)

feedback = Table(
    "feedback", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("segment_id", Integer, index=True, nullable=True),
    Column("lat", Float), Column("lon", Float),   # rounded to ~10 m by the API
    Column("kind", String(24)),                  # flooded | not_flooded | drain_blocked
    Column("comment", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), index=True),
)

flood_event = Table(
    "flood_event", metadata,
    Column("event_id", Integer, primary_key=True, autoincrement=True),
    Column("event_date", DateTime(timezone=True), index=True),
    Column("duration_h", Float, nullable=True),
    Column("geom_wkt", Text),                    # WGS84; geometry(Geometry,4326) in PostGIS
    Column("spatial_acc_m", Float, nullable=True),
    Column("source", String(128)), Column("source_url", Text, nullable=True),
    Column("reliability", String(8)),            # high | medium | low
    Column("observed_rain_mm", Float, nullable=True),
    Column("test_set", Boolean, default=False),  # held out for validation: never feeds f6 / DV
    Column("notes", Text, nullable=True),
)

rain_record = Table(
    "rain_obs_forecast", metadata,
    Column("cell_id", String(16), primary_key=True),
    Column("valid_time", DateTime(timezone=True), primary_key=True),
    Column("model", String(48), primary_key=True),
    Column("retrieved_at", DateTime(timezone=True), primary_key=True),
    Column("mm", Float), Column("interval_min", Integer), Column("source", String(64)),
)


class Database:
    def __init__(self, url: str):
        kwargs: dict[str, Any] = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self.engine: Engine = create_engine(url, future=True, **kwargs)
        metadata.create_all(self.engine)

    def log_predictions(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if rows:
            with self.engine.begin() as cx:
                cx.execute(insert(prediction_log), rows)

    def add_feedback(self, row: dict) -> int:
        with self.engine.begin() as cx:
            res = cx.execute(insert(feedback).values(**row, created_at=datetime.now(timezone.utc)))
            return int(res.inserted_primary_key[0])

    def recent_feedback(self, segment_ids: list[int] | None, since: datetime) -> dict[int, str]:
        """'consistent_flooded' / 'consistent_dry' when >= 2 reports agree (confidence bonus).

        segment_ids=None looks at every segment with recent feedback.
        """
        q = (select(feedback.c.segment_id, feedback.c.kind, func.count())
             .where(feedback.c.created_at >= since, feedback.c.segment_id.is_not(None))
             .group_by(feedback.c.segment_id, feedback.c.kind))
        if segment_ids is not None:
            q = q.where(feedback.c.segment_id.in_(segment_ids))
        counts: dict[int, dict[str, int]] = {}
        with self.engine.connect() as cx:
            for sid, kind, n in cx.execute(q):
                counts.setdefault(sid, {})[kind] = n
        out = {}
        for sid, c in counts.items():
            wet = c.get("flooded", 0) + c.get("drain_blocked", 0)
            dry = c.get("not_flooded", 0)
            if wet >= 2 and dry == 0:
                out[sid] = "consistent_flooded"
            elif dry >= 2 and wet == 0:
                out[sid] = "consistent_dry"
        return out

    def record_rain(self, store) -> None:
        rows = []
        for cell_id, cs in store.series.items():
            for model, values in cs.values.items():
                for t, v in zip(cs.times, values):
                    if v == v:  # not NaN
                        rows.append({
                            "cell_id": cell_id, "model": model, "mm": float(v),
                            "valid_time": datetime.fromtimestamp(int(t), timezone.utc),
                            "retrieved_at": store.retrieved_at, "interval_min": store.interval_minutes,
                            "source": store.source})
        if rows:
            with self.engine.begin() as cx:
                cx.execute(insert(rain_record), rows)

    def insert_events(self, rows: list[dict]) -> int:
        with self.engine.begin() as cx:
            cx.execute(insert(flood_event), rows)
        return len(rows)

    def events(self) -> list[dict]:
        with self.engine.connect() as cx:
            return [dict(r._mapping) for r in cx.execute(select(flood_event))]
