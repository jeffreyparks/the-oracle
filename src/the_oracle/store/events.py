"""The append-only event log.

This is the raw signal. Everything else in the system is derived from it.

Rules, enforced here and by tests:

* ``append`` is the only writer.
* There is no update and no delete. Correcting a fact means appending a new
  event, never editing an old one.
* Readers get typed :class:`Event` records, ordered by ``(occurred_at, id)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import Engine
from sqlmodel import Session, asc, select

from the_oracle.store import models
from the_oracle.store.db import get_engine, session_scope


class EventKind(StrEnum):
    """Known event kinds. Unknown strings are accepted; reducers ignore them."""

    LEARNER_CREATED = "learner.created"
    ENROLLED = "learner.enrolled"
    GOAL_SET = "learner.goal_set"
    PREFERENCES_SET = "learner.preferences_set"
    SESSION_STARTED = "session.started"
    SESSION_ENDED = "session.ended"
    ITEM_PRESENTED = "item.presented"
    RESPONSE_GRADED = "response.graded"
    MISCONCEPTION_DETECTED = "misconception.detected"
    RESOURCE_ATTACHED = "resource.attached"
    REVIEW_SCHEDULED = "review.scheduled"
    NUDGE_SENT = "nudge.sent"


class ImmutableEventLogError(RuntimeError):
    """Raised on any attempt to mutate a recorded event."""


@dataclass(frozen=True, slots=True)
class Event:
    """A typed, read-only view of one row of the log."""

    id: int
    learner_id: str
    kind: str
    payload: dict[str, Any]
    occurred_at: datetime
    recorded_at: datetime

    @classmethod
    def from_row(cls, row: models.EventLog) -> "Event":
        if row.id is None:  # pragma: no cover - unreachable after flush
            raise ValueError("event row has no id; it was never persisted")
        return cls(
            id=row.id,
            learner_id=row.learner_id,
            kind=row.kind,
            payload=dict(row.payload or {}),
            occurred_at=row.occurred_at,
            recorded_at=row.recorded_at,
        )


class EventLog:
    """Append-only writer and reader over :class:`models.EventLog`."""

    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine or get_engine()

    @property
    def engine(self) -> Engine:
        return self._engine

    # -- write -------------------------------------------------------------

    def append(
        self,
        kind: str | EventKind,
        learner_id: str,
        payload: dict[str, Any] | None = None,
        *,
        occurred_at: datetime | None = None,
    ) -> Event:
        """Record one event. The only write path into the log."""
        row = models.EventLog(
            learner_id=learner_id,
            kind=str(kind),
            payload=dict(payload or {}),
            occurred_at=occurred_at or models.utcnow(),
        )
        with session_scope(self._engine) as session:
            session.add(row)
            session.flush()
            session.refresh(row)
            return Event.from_row(row)

    def append_many(
        self,
        events: Iterable[tuple[str | EventKind, str, dict[str, Any]]],
    ) -> list[Event]:
        """Append a batch in one transaction. Order is preserved."""
        rows = [
            models.EventLog(learner_id=learner_id, kind=str(kind), payload=dict(payload or {}))
            for kind, learner_id, payload in events
        ]
        with session_scope(self._engine) as session:
            for row in rows:
                session.add(row)
            session.flush()
            for row in rows:
                session.refresh(row)
            return [Event.from_row(row) for row in rows]

    # -- forbidden ---------------------------------------------------------

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        """Always raises. The log is append-only."""
        raise ImmutableEventLogError("the event log is append-only; append a correction instead")

    def delete(self, *_args: Any, **_kwargs: Any) -> None:
        """Always raises. The log is append-only."""
        raise ImmutableEventLogError("the event log is append-only; events are never deleted")

    # -- read --------------------------------------------------------------

    def read(
        self,
        learner_id: str | None = None,
        *,
        kinds: Sequence[str | EventKind] | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Return matching events in replay order."""
        return list(self.stream(learner_id, kinds=kinds, since=since, limit=limit))

    def stream(
        self,
        learner_id: str | None = None,
        *,
        kinds: Sequence[str | EventKind] | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> Iterator[Event]:
        """Yield matching events in replay order, oldest first."""
        statement = select(models.EventLog)
        if learner_id is not None:
            statement = statement.where(models.EventLog.learner_id == learner_id)
        if kinds:
            statement = statement.where(models.EventLog.kind.in_([str(k) for k in kinds]))  # type: ignore[attr-defined]
        if since is not None:
            statement = statement.where(models.EventLog.occurred_at >= since)
        statement = statement.order_by(
            asc(models.EventLog.occurred_at), asc(models.EventLog.id)
        )
        if limit is not None:
            statement = statement.limit(limit)
        with Session(self._engine) as session:
            for row in session.exec(statement):
                yield Event.from_row(row)

    def count(self, learner_id: str | None = None) -> int:
        """Number of events, optionally for one learner."""
        return len(self.read(learner_id))
