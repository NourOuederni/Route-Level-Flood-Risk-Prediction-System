"""Database configuration and connection helpers for V2."""
from __future__ import annotations

import os
from functools import lru_cache

from .models import make_engine, make_session, init_db

DEFAULT_DB_URL = "postgresql://flood:flood_dev_2024@localhost:5433/floodrisk"


def get_db_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DB_URL)


@lru_cache(maxsize=1)
def get_engine():
    return make_engine(get_db_url())


@lru_cache(maxsize=1)
def get_session_factory():
    return make_session(get_engine())


def get_db():
    """Yield a session for FastAPI dependency injection."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def ensure_tables():
    """Create all tables if they don't exist (safe to call repeatedly)."""
    init_db(get_engine())
