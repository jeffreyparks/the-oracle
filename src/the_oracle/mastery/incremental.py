"""Incremental mastery: fold one event without replaying the log.

Phase 1 debt item 2, paid here.

``run_diagnostic`` calls ``rebuild()`` once per item: it drops every derived row
for the learner, re-reads the whole event log, and rewrites everything. That is
fine at a 12-item cap and hopeless across a study session, where the cost grows
with the learner's entire history on every single answer.

What this module is, and is not
-------------------------------
This is a **fast path**, not a second model of the world. Full replay
(:func:`the_oracle.store.rebuild.rebuild`) stays the source of truth. Every
function here is built from the same pieces the reducers use -
:func:`the_oracle.mastery.reducers.params_for`, :func:`the_oracle.mastery.bkt.update`,
:func:`the_oracle.mastery.bkt.confidence`, :func:`the_oracle.mastery.scheduler.plan` -
so the two cannot drift apart without one of those changing underneath both.
``tests/test_gate.py`` drives random response streams through both paths over
several seeds and asserts the mastery they produce is identical to 1e-9.

Why the schedule still reads a little history
---------------------------------------------
The BKT posterior is a true fold: new ``p`` depends only on old ``p`` and one
observation. The scheduler is not. ``scheduler.plan`` recomputes per-objective
difficulty over the whole review history and its stability growth uses the
*current* posterior at every step, so a correct schedule needs that objective's
responses. We read them back with one filtered query over the log, for that one
objective, instead of replaying every event and rewriting every table. Exactness
first; the speed comes from doing less work, not from approximating.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Engine

from the_oracle.mastery import bkt
from the_oracle.mastery.reducers import params_for
from the_oracle.mastery.scheduler import Review, plan
from the_oracle.store import models
from the_oracle.store.db import get_engine, session_scope
from the_oracle.store.events import Event, EventKind, EventLog


def _objective_id(payload: dict[str, Any]) -> str | None:
    value = payload.get("objective_id")
    return value if isinstance(value, str) and value else None


def _int(value: Any, default: int) -> int:
    """Same lenient cast the reducers use, so bad payloads land the same way."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _latency_ms(seconds: Any) -> int | None:
    try:
        return int(round(float(seconds) * 1000)) if seconds is not None else None
    except (TypeError, ValueError):
        return None


def _start(learner_id: str, objective_id: str) -> models.MasteryState:
    """A fresh row at the BKT prior - what replay emits on first sight."""
    return models.MasteryState(
        learner_id=learner_id,
        objective_id=objective_id,
        p_mastery=bkt.clamp(params_for(objective_id).p_init),
        confidence=0.0,
        observations=0,
        misconception_ids=[],
        updated_at=models.utcnow(),
    )


def _copy(state: models.MasteryState) -> models.MasteryState:
    return models.MasteryState(
        learner_id=state.learner_id,
        objective_id=state.objective_id,
        p_mastery=float(state.p_mastery),
        confidence=float(state.confidence),
        observations=int(state.observations),
        misconception_ids=list(state.misconception_ids or []),
        last_seen_at=state.last_seen_at,
        next_review_at=state.next_review_at,
        updated_at=state.updated_at,
    )


def apply_event(state: models.MasteryState | None, event: Event) -> models.MasteryState:
    """Fold one event into one mastery row and return the result.

    Pure: ``state`` is never mutated, a new row is returned. ``None`` means the
    objective has no evidence yet, so the fold starts from the BKT prior,
    exactly as replay does on first sight of an objective.

    Only ``response.graded`` moves the posterior. ``misconception.detected``
    records the tag and leaves ``p`` alone - the grading event that exposed the
    misconception already carried that evidence, and counting it twice would
    punish one mistake twice. Any other event kind is returned unchanged.

    ``next_review_at`` is not set here: the schedule needs the objective's
    review history, which one event does not carry. :func:`apply_response`
    fills it in.
    """
    objective_id = _objective_id(event.payload)
    if state is None:
        if objective_id is None:
            raise ValueError("cannot start a mastery row from an event with no objective_id")
        state = _start(event.learner_id, objective_id)
    updated = _copy(state)
    if objective_id is None or objective_id != updated.objective_id:
        return updated

    misconception_id = event.payload.get("misconception_id")

    if event.kind == EventKind.RESPONSE_GRADED:
        correct = bool(event.payload.get("correct", False))
        params = params_for(objective_id)
        updated.p_mastery = bkt.update(float(updated.p_mastery), correct, params)
        updated.observations = int(updated.observations) + 1
        updated.confidence = bkt.confidence(updated.observations)
        updated.last_seen_at = event.occurred_at
        updated.updated_at = event.occurred_at
    elif event.kind == EventKind.MISCONCEPTION_DETECTED:
        if not misconception_id:
            return updated
        updated.updated_at = event.occurred_at
    else:
        return updated

    if misconception_id and str(misconception_id) not in updated.misconception_ids:
        updated.misconception_ids = [*updated.misconception_ids, str(misconception_id)]
    return updated


def _review_history(
    log: EventLog, learner_id: str, objective_id: str
) -> list[Review]:
    """Every graded opportunity on one objective, oldest first.

    One filtered read over the log, not a replay. The scheduler needs it
    because its difficulty term and its stability fold are not incremental.
    """
    return [
        Review(
            correct=bool(event.payload.get("correct", False)),
            at=event.occurred_at,
            difficulty=_int(event.payload.get("difficulty"), 1),
        )
        for event in log.stream(learner_id, kinds=[EventKind.RESPONSE_GRADED])
        if _objective_id(event.payload) == objective_id
    ]


def apply_response(
    learner_id: str,
    payload: dict[str, Any],
    *,
    engine: Engine | None = None,
    occurred_at: datetime | None = None,
    append: bool = True,
) -> float:
    """Record one graded response and return the learner's new ``p``.

    This is the study loop's write path. By default it appends the
    ``response.graded`` event itself, so the log and the derived rows can never
    disagree; pass ``append=False`` when the caller has already appended the
    event and only wants the derived rows moved forward.

    It updates ``MasteryState``, writes the ``Response`` row, and refreshes
    ``ReviewSchedule`` - the same three rows a full replay would produce, with
    the same values. It never writes a misconception event; the caller appends
    ``misconception.detected`` when it wants the tag in the log, and replay and
    this path both treat that event as a tag, not as evidence.
    """
    engine = engine or get_engine()
    objective_id = _objective_id(payload)
    if objective_id is None:
        raise ValueError("response payload needs a non-empty objective_id")

    log = EventLog(engine)
    if append:
        event = log.append(
            EventKind.RESPONSE_GRADED, learner_id, payload, occurred_at=occurred_at
        )
    else:
        event = Event(
            id=0,
            learner_id=learner_id,
            kind=str(EventKind.RESPONSE_GRADED),
            payload=dict(payload),
            occurred_at=occurred_at or models.utcnow(),
            recorded_at=models.utcnow(),
        )

    reviews = _review_history(log, learner_id, objective_id)
    if not append:
        reviews.append(
            Review(
                correct=bool(payload.get("correct", False)),
                at=event.occurred_at,
                difficulty=_int(payload.get("difficulty"), 1),
            )
        )

    misconception_id = payload.get("misconception_id")
    with session_scope(engine) as session:
        row = session.get(models.MasteryState, (learner_id, objective_id))
        folded = apply_event(row, event)
        if row is None:
            row = folded
            session.add(row)
        else:
            row.p_mastery = folded.p_mastery
            row.confidence = folded.confidence
            row.observations = folded.observations
            row.misconception_ids = list(folded.misconception_ids)
            row.last_seen_at = folded.last_seen_at
            row.updated_at = folded.updated_at

        session.add(
            models.Response(
                learner_id=learner_id,
                objective_id=objective_id,
                item_id=payload.get("item_id"),
                session_id=payload.get("session_id"),
                correct=bool(payload.get("correct", False)),
                score=1.0 if payload.get("correct", False) else 0.0,
                latency_ms=_latency_ms(payload.get("seconds")),
                misconception_ids=[misconception_id] if misconception_id else [],
                created_at=event.occurred_at,
            )
        )

        review = plan(reviews, row.p_mastery)
        if review is not None:
            row.next_review_at = review.due_at
            schedule = session.get(models.ReviewSchedule, (learner_id, objective_id))
            if schedule is None:
                schedule = models.ReviewSchedule(
                    learner_id=learner_id, objective_id=objective_id
                )
                session.add(schedule)
            schedule.due_at = review.due_at
            schedule.interval_days = review.interval_days
            schedule.stability = review.stability
            schedule.difficulty = review.difficulty
            schedule.lapses = review.lapses
        new_p = float(row.p_mastery)
    return new_p


def apply_misconception(
    learner_id: str,
    payload: dict[str, Any],
    *,
    engine: Engine | None = None,
    occurred_at: datetime | None = None,
) -> None:
    """Record a named wrong model against an objective. ``p`` does not move."""
    engine = engine or get_engine()
    objective_id = _objective_id(payload)
    if objective_id is None or not payload.get("misconception_id"):
        return
    event = EventLog(engine).append(
        EventKind.MISCONCEPTION_DETECTED, learner_id, payload, occurred_at=occurred_at
    )
    with session_scope(engine) as session:
        row = session.get(models.MasteryState, (learner_id, objective_id))
        folded = apply_event(row, event)
        if row is None:
            session.add(folded)
            return
        row.misconception_ids = list(folded.misconception_ids)
        row.updated_at = folded.updated_at


def mastery_for(
    learner_id: str, objective_id: str, *, engine: Engine | None = None
) -> float:
    """Current posterior for one objective, straight off the derived row."""
    engine = engine or get_engine()
    with session_scope(engine) as session:
        row = session.get(models.MasteryState, (learner_id, objective_id))
        return float(row.p_mastery) if row is not None else bkt.clamp(
            params_for(objective_id).p_init
        )


__all__ = [
    "apply_event",
    "apply_misconception",
    "apply_response",
    "mastery_for",
]
