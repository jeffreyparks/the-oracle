"""Reducers: rebuild MasteryState from the event log alone.

Registered on the default :data:`REGISTRY` at import time, so
``rebuild(learner_id)`` recomputes every derived mastery row from the log with
no other input.

Purity contract
---------------
These reducers are deterministic functions of the event stream. They:

* never read the database (``ReplayState`` is the only scratch space),
* never call ``utcnow()`` - every timestamp comes from ``event.occurred_at``,
* never depend on iteration order of a set or a dict built from unsorted input,
* emit each derived row **once** and then mutate that same object in place, so
  a learner-objective pair keeps its primary key across a whole replay.

Same log in, same mastery out, every time. ``tests/test_mastery.py`` proves it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from the_oracle.mastery import bkt
from the_oracle.mastery.scheduler import Review, plan
from the_oracle.store import models
from the_oracle.store.events import Event, EventKind
from the_oracle.store.rebuild import REGISTRY, ReducerRegistry, ReplayState

#: Scratch key holding the per-objective accumulators.
SCRATCH_KEY = "mastery"


def params_for(objective_id: str) -> bkt.BKTParams:
    """Knowledge-tracing parameters for one objective.

    One global set today. When per-objective fitting arrives, only this
    function changes, and it must stay a pure function of the objective id so
    replay stays deterministic.
    """
    del objective_id
    return bkt.DEFAULT_PARAMS


@dataclass(slots=True)
class ObjectiveAccumulator:
    """Everything replay knows about one learner-objective pair."""

    objective_id: str
    params: bkt.BKTParams
    p: float
    observations: int = 0
    correct: int = 0
    misconception_ids: list[str] = field(default_factory=list)
    reviews: list[Review] = field(default_factory=list)
    last_seen_at: datetime | None = None

    @classmethod
    def start(cls, objective_id: str) -> "ObjectiveAccumulator":
        params = params_for(objective_id)
        return cls(objective_id=objective_id, params=params, p=bkt.clamp(params.p_init))

    def observe(self, *, correct: bool, at: datetime, difficulty: int) -> None:
        self.p = bkt.update(self.p, correct, self.params)
        self.observations += 1
        self.correct += int(bool(correct))
        self.reviews.append(Review(correct=bool(correct), at=at, difficulty=difficulty))
        self.last_seen_at = at

    def tag(self, misconception_id: str | None) -> None:
        if misconception_id and misconception_id not in self.misconception_ids:
            self.misconception_ids.append(misconception_id)

    @property
    def confidence(self) -> float:
        return bkt.confidence(self.observations)

    def review_plan(self):  # type: ignore[no-untyped-def]
        return plan(self.reviews, self.p)


# -- helpers ---------------------------------------------------------------


def _accumulators(state: ReplayState) -> dict[str, ObjectiveAccumulator]:
    return state.scratch.setdefault(SCRATCH_KEY, {})


def _objective_id(payload: dict[str, Any]) -> str | None:
    value = payload.get("objective_id")
    return value if isinstance(value, str) and value else None


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _for(state: ReplayState, objective_id: str) -> ObjectiveAccumulator:
    """Accumulator for one objective, creating its MasteryState row on first sight."""
    accumulators = _accumulators(state)
    accumulator = accumulators.get(objective_id)
    if accumulator is not None:
        return accumulator

    accumulator = ObjectiveAccumulator.start(objective_id)
    accumulators[objective_id] = accumulator

    rows = state.scratch.setdefault("mastery_rows", {})
    rows[objective_id] = models.MasteryState(
        learner_id=state.learner_id,
        objective_id=objective_id,
        p_mastery=accumulator.p,
        confidence=0.0,
        observations=0,
        misconception_ids=[],
        updated_at=models.utcnow(),
    )
    state.emit(rows[objective_id])
    return accumulator


def _sync(state: ReplayState, accumulator: ObjectiveAccumulator, at: datetime) -> None:
    """Write the accumulator back onto the emitted rows."""
    row: models.MasteryState = state.scratch["mastery_rows"][accumulator.objective_id]
    row.p_mastery = accumulator.p
    row.confidence = accumulator.confidence
    row.observations = accumulator.observations
    row.misconception_ids = list(accumulator.misconception_ids)
    row.last_seen_at = accumulator.last_seen_at
    row.updated_at = at

    review = accumulator.review_plan()
    if review is None:
        return
    row.next_review_at = review.due_at

    schedules = state.scratch.setdefault("schedule_rows", {})
    schedule = schedules.get(accumulator.objective_id)
    if schedule is None:
        schedule = models.ReviewSchedule(
            learner_id=state.learner_id, objective_id=accumulator.objective_id
        )
        schedules[accumulator.objective_id] = schedule
        state.emit(schedule)
    schedule.due_at = review.due_at
    schedule.interval_days = review.interval_days
    schedule.stability = review.stability
    schedule.difficulty = review.difficulty
    schedule.lapses = review.lapses


# -- reducers --------------------------------------------------------------


def reduce_response_graded(state: ReplayState, event: Event) -> None:
    """Fold one graded answer into mastery, the response table, and the schedule."""
    payload = event.payload
    objective_id = _objective_id(payload)
    if objective_id is None:
        return

    correct = bool(payload.get("correct", False))
    difficulty = _int(payload.get("difficulty"), 1)
    accumulator = _for(state, objective_id)
    accumulator.observe(correct=correct, at=event.occurred_at, difficulty=difficulty)
    accumulator.tag(payload.get("misconception_id"))

    seconds = payload.get("seconds")
    latency_ms: int | None
    try:
        latency_ms = int(round(float(seconds) * 1000)) if seconds is not None else None
    except (TypeError, ValueError):
        latency_ms = None

    misconception_id = payload.get("misconception_id")
    state.emit(
        models.Response(
            learner_id=state.learner_id,
            objective_id=objective_id,
            item_id=payload.get("item_id"),
            session_id=payload.get("session_id"),
            correct=correct,
            score=1.0 if correct else 0.0,
            latency_ms=latency_ms,
            misconception_ids=[misconception_id] if misconception_id else [],
            created_at=event.occurred_at,
        )
    )
    _sync(state, accumulator, event.occurred_at)


def reduce_misconception_detected(state: ReplayState, event: Event) -> None:
    """Record a named wrong model against an objective.

    It does not move ``p_mastery``. The grading event that exposed the
    misconception already carried the evidence; counting it twice would
    double-penalise one mistake.
    """
    objective_id = _objective_id(event.payload)
    misconception_id = event.payload.get("misconception_id")
    if objective_id is None or not misconception_id:
        return
    accumulator = _for(state, objective_id)
    accumulator.tag(str(misconception_id))
    _sync(state, accumulator, event.occurred_at)


def register(registry: ReducerRegistry) -> None:
    """Attach the mastery reducers to ``registry``. Idempotent per registry."""
    if getattr(registry, "_mastery_registered", False):
        return
    registry.register(EventKind.RESPONSE_GRADED, reduce_response_graded)
    registry.register(EventKind.MISCONCEPTION_DETECTED, reduce_misconception_detected)
    registry._mastery_registered = True  # type: ignore[attr-defined]  # noqa: SLF001


register(REGISTRY)


__all__ = [
    "SCRATCH_KEY",
    "ObjectiveAccumulator",
    "params_for",
    "reduce_misconception_detected",
    "reduce_response_graded",
    "register",
]
