"""Engine and session handling.

SQLite today, Postgres by connection string tomorrow. No dialect-specific SQL
lives anywhere in this package.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from the_oracle.config import Settings, get_settings
from the_oracle.store import models as _models  # noqa: F401  (registers tables)

_ENGINES: dict[str, Engine] = {}


def _enable_sqlite_fks(engine: Engine) -> None:
    """Turn on foreign keys for SQLite. A pragma, not SQL in our code path."""
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection: Any, _record: Any) -> None:  # pragma: no cover
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def build_engine(url: str, *, echo: bool = False) -> Engine:
    """Create an engine for ``url`` without touching the module cache."""
    kwargs: dict[str, Any] = {"echo": echo}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
        else:
            target = url.split("sqlite:///", 1)[-1]
            if target and target != ":memory:":
                Path(target).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, **kwargs)
    _enable_sqlite_fks(engine)
    return engine


def get_engine(settings: Settings | None = None, *, url: str | None = None) -> Engine:
    """Return the cached engine for ``url`` or for the configured database."""
    settings = settings or get_settings()
    target = url or settings.sqlalchemy_url
    engine = _ENGINES.get(target)
    if engine is None:
        engine = build_engine(target)
        _ENGINES[target] = engine
    return engine


def create_all(engine: Engine | None = None) -> Engine:
    """Create every table that does not exist yet. Safe to call repeatedly."""
    engine = engine or get_engine()
    SQLModel.metadata.create_all(engine)
    return engine


def drop_all(engine: Engine | None = None) -> None:
    """Drop every table. Destructive. Tests and a full reset only."""
    engine = engine or get_engine()
    SQLModel.metadata.drop_all(engine)


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Transactional session. Commits on success, rolls back on error."""
    engine = engine or get_engine()
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_cache() -> None:
    """Dispose and forget every cached engine."""
    for engine in _ENGINES.values():
        engine.dispose()
    _ENGINES.clear()
