"""Replay: drop derived state, read the log, recompute.

This is the recovery path, the model-swap path, and the primary integration
test. If a replay does not reproduce current state, something is broken.

The pipeline is real and pluggable. Phase 0 ships no mastery model, so the
default registry holds no-op reducers. Phase 1 registers the BKT reducer and
nothing here changes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import Engine, delete
from sqlmodel import SQLModel

from the_oracle.store import models
from the_oracle.store.db import get_engine, session_scope
from the_oracle.store.events import Event, EventLog


@dataclass(slots=True)
class ReplayState:
    """Scratch space carried through one replay.

    Reducers mutate ``rows`` and ``scratch``; the writer persists ``rows`` at
    the end. Reducers never touch the database themselves.
    """

    learner_id: str
    rows: list[SQLModel] = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)
    handled: int = 0
    skipped: int = 0

    def emit(self, row: SQLModel) -> None:
        self.rows.append(row)


@runtime_checkable
class Reducer(Protocol):
    """Fold one event into replay state."""

    def __call__(self, state: ReplayState, event: Event) -> None: ...


class ReducerRegistry:
    """Event kind -> reducers. Later phases register real ones here."""

    def __init__(self) -> None:
        self._by_kind: dict[str, list[Reducer]] = {}

    def register(self, kind: str, reducer: Reducer) -> Reducer:
        self._by_kind.setdefault(str(kind), []).append(reducer)
        return reducer

    def on(self, kind: str) -> Callable[[Reducer], Reducer]:
        """Decorator form of :meth:`register`."""

        def _wrap(reducer: Reducer) -> Reducer:
            return self.register(kind, reducer)

        return _wrap

    def reducers_for(self, kind: str) -> list[Reducer]:
        return list(self._by_kind.get(str(kind), ()))

    def kinds(self) -> list[str]:
        return sorted(self._by_kind)

    def apply(self, state: ReplayState, event: Event) -> None:
        reducers = self.reducers_for(event.kind)
        if not reducers:
            state.skipped += 1
            return
        for reducer in reducers:
            reducer(state, event)
        state.handled += 1


def noop_reducer(state: ReplayState, event: Event) -> None:
    """Phase 0 placeholder. Counts the event and derives nothing."""
    counts: dict[str, int] = state.scratch.setdefault("counts", {})
    counts[event.kind] = counts.get(event.kind, 0) + 1


#: Default registry. Phase 0 wires every known kind to the no-op reducer so the
#: pipeline is exercised end to end before a mastery model exists.
REGISTRY = ReducerRegistry()
for _kind in (
    "learner.created",
    "learner.enrolled",
    "learner.goal_set",
    "learner.preferences_set",
    "session.started",
    "session.ended",
    "item.presented",
    "response.graded",
    "misconception.detected",
    "resource.attached",
    "review.scheduled",
    "nudge.sent",
):
    REGISTRY.register(_kind, noop_reducer)


@dataclass(frozen=True, slots=True)
class RebuildReport:
    """What one replay did."""

    learner_id: str
    events_read: int
    events_handled: int
    events_skipped: int
    rows_written: int
    tables_cleared: tuple[str, ...]

    def summary(self) -> str:
        return (
            f"{self.learner_id}: {self.events_read} events, "
            f"{self.events_handled} handled, {self.events_skipped} unhandled, "
            f"{self.rows_written} derived rows"
        )


def drop_derived(
    learner_id: str,
    engine: Engine | None = None,
    tables: Iterable[type[SQLModel]] | None = None,
) -> tuple[str, ...]:
    """Delete this learner's derived rows. The event log is never touched."""
    engine = engine or get_engine()
    targets = tuple(tables or models.DERIVED_TABLES)
    cleared: list[str] = []
    with session_scope(engine) as session:
        for table in targets:
            if table is models.EventLog:
                raise ValueError("refusing to drop the event log")
            column = getattr(table, "learner_id", None)
            if column is None:
                continue
            session.exec(delete(table).where(column == learner_id))  # type: ignore[call-overload]
            cleared.append(str(table.__tablename__))
    return tuple(cleared)


def rebuild(
    learner_id: str,
    engine: Engine | None = None,
    *,
    registry: ReducerRegistry | None = None,
) -> RebuildReport:
    """Drop derived state for ``learner_id``, replay the log, recompute."""
    engine = engine or get_engine()
    registry = registry or REGISTRY

    cleared = drop_derived(learner_id, engine)

    log = EventLog(engine)
    state = ReplayState(learner_id=learner_id)
    read = 0
    for event in log.stream(learner_id):
        read += 1
        registry.apply(state, event)

    if state.rows:
        with session_scope(engine) as session:
            for row in state.rows:
                session.add(row)

    return RebuildReport(
        learner_id=learner_id,
        events_read=read,
        events_handled=state.handled,
        events_skipped=state.skipped,
        rows_written=len(state.rows),
        tables_cleared=cleared,
    )
