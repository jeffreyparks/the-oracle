"""Store tests: the event log is append-only, rebuild round-trips,
MasteryState has no domain column, and the idempotency cache works.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from pydantic import BaseModel
from sqlalchemy import inspect as sa_inspect
from sqlmodel import Session, select

from the_oracle.agents.base import Agent, Usage, hash_input
from the_oracle.config import Settings, Task
from the_oracle.context import AuthError, LearnerContext
from the_oracle.store import models
from the_oracle.store.db import build_engine, create_all
from the_oracle.store.events import Event, EventKind, EventLog, ImmutableEventLogError
from the_oracle.store.rebuild import RebuildReport, ReducerRegistry, ReplayState, rebuild

LEARNER = "learner_test"


@pytest.fixture
def engine():
    eng = build_engine("sqlite:///:memory:")
    create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def log(engine):
    return EventLog(engine)


@pytest.fixture
def ctx():
    return LearnerContext.for_learner(LEARNER, Settings(home="/tmp/oracle-test-home"))


# -- event log ------------------------------------------------------------


def test_append_returns_typed_event(log):
    event = log.append(EventKind.RESPONSE_GRADED, LEARNER, {"objective_id": "obj_a", "correct": True})
    assert isinstance(event, Event)
    assert event.id is not None
    assert event.kind == "response.graded"
    assert event.learner_id == LEARNER
    assert event.payload["correct"] is True


def test_read_is_ordered_and_filtered(log):
    log.append(EventKind.SESSION_STARTED, LEARNER, {"n": 1})
    log.append(EventKind.RESPONSE_GRADED, LEARNER, {"n": 2})
    log.append(EventKind.SESSION_ENDED, LEARNER, {"n": 3})
    log.append(EventKind.SESSION_STARTED, "someone_else", {"n": 4})

    mine = log.read(LEARNER)
    assert [e.payload["n"] for e in mine] == [1, 2, 3]
    assert all(e.learner_id == LEARNER for e in mine)

    graded = log.read(LEARNER, kinds=[EventKind.RESPONSE_GRADED])
    assert [e.payload["n"] for e in graded] == [2]


def test_log_has_no_update_or_delete_api(log):
    log.append(EventKind.SESSION_STARTED, LEARNER, {"n": 1})
    with pytest.raises(ImmutableEventLogError):
        log.update(1, {"n": 99})
    with pytest.raises(ImmutableEventLogError):
        log.delete(1)
    assert log.count(LEARNER) == 1


def test_events_are_read_only_records(log):
    event = log.append(EventKind.SESSION_STARTED, LEARNER, {"n": 1})
    with pytest.raises(Exception):
        event.kind = "tampered"  # type: ignore[misc]


def test_append_never_overwrites(log):
    payload = {"objective_id": "obj_a", "correct": False}
    first = log.append(EventKind.RESPONSE_GRADED, LEARNER, payload)
    second = log.append(EventKind.RESPONSE_GRADED, LEARNER, payload)
    assert first.id != second.id
    assert log.count(LEARNER) == 2


def test_correction_is_a_new_event_not_an_edit(log):
    log.append(EventKind.RESPONSE_GRADED, LEARNER, {"objective_id": "obj_a", "correct": False})
    log.append(EventKind.RESPONSE_GRADED, LEARNER, {"objective_id": "obj_a", "correct": True, "corrects": 1})
    events = log.read(LEARNER)
    assert len(events) == 2
    assert events[0].payload["correct"] is False


# -- schema ---------------------------------------------------------------


def test_mastery_state_has_no_domain_column(engine):
    columns = {c["name"] for c in sa_inspect(engine).get_columns("mastery_state")}
    assert "domain_id" not in columns
    assert not any("domain" in name for name in columns)
    assert {"learner_id", "objective_id"} <= columns


def test_mastery_state_primary_key_is_learner_and_objective(engine):
    pk = sa_inspect(engine).get_pk_constraint("mastery_state")["constrained_columns"]
    assert set(pk) == {"learner_id", "objective_id"}


def test_mastery_is_shared_across_domains(engine):
    """One row per (learner, objective). Two domains cannot fork it."""
    with Session(engine) as session:
        session.add(models.MasteryState(learner_id=LEARNER, objective_id="obj_a", p_mastery=0.9))
        session.commit()
        rows = session.exec(
            select(models.MasteryState).where(models.MasteryState.objective_id == "obj_a")
        ).all()
    assert len(rows) == 1
    assert rows[0].p_mastery == pytest.approx(0.9)


def test_event_log_is_not_a_derived_table():
    assert models.EventLog not in models.DERIVED_TABLES
    assert models.MasteryState in models.DERIVED_TABLES


# -- rebuild --------------------------------------------------------------


def test_rebuild_reads_every_event(engine, log):
    for i in range(5):
        log.append(EventKind.RESPONSE_GRADED, LEARNER, {"objective_id": "obj_a", "i": i})
    report = rebuild(LEARNER, engine)
    assert isinstance(report, RebuildReport)
    assert report.events_read == 5
    assert report.events_handled == 5
    assert report.events_skipped == 0


def test_rebuild_never_touches_the_log(engine, log):
    for i in range(3):
        log.append(EventKind.SESSION_STARTED, LEARNER, {"i": i})
    before = log.read(LEARNER)
    rebuild(LEARNER, engine)
    rebuild(LEARNER, engine)
    after = log.read(LEARNER)
    assert [(e.id, e.kind, e.payload) for e in before] == [(e.id, e.kind, e.payload) for e in after]


def test_rebuild_round_trips_derived_state(engine, log):
    """Replay reproduces derived state exactly, twice over."""
    registry = ReducerRegistry()

    def grade(state: ReplayState, event: Event) -> None:
        objective_id = event.payload["objective_id"]
        seen = state.scratch.setdefault("mastery", {})
        row = seen.get(objective_id)
        if row is None:
            row = models.MasteryState(learner_id=state.learner_id, objective_id=objective_id)
            seen[objective_id] = row
            state.emit(row)
        row.observations += 1
        row.p_mastery = min(1.0, row.p_mastery + (0.25 if event.payload["correct"] else 0.0))

    registry.register(EventKind.RESPONSE_GRADED, grade)

    for correct in (True, True, False, True):
        log.append(EventKind.RESPONSE_GRADED, LEARNER, {"objective_id": "obj_a", "correct": correct})
    log.append(EventKind.RESPONSE_GRADED, LEARNER, {"objective_id": "obj_b", "correct": True})

    first = rebuild(LEARNER, engine, registry=registry)
    snapshot_one = _mastery_snapshot(engine)

    second = rebuild(LEARNER, engine, registry=registry)
    snapshot_two = _mastery_snapshot(engine)

    assert first.rows_written == second.rows_written == 2
    assert snapshot_one == snapshot_two
    assert snapshot_one == {("obj_a", 4, 0.75), ("obj_b", 1, 0.25)}


def _mastery_snapshot(engine) -> set[tuple[str, int, float]]:
    with Session(engine) as session:
        rows = session.exec(select(models.MasteryState)).all()
    return {(r.objective_id, r.observations, round(r.p_mastery, 6)) for r in rows}


def test_rebuild_drops_stale_derived_rows(engine, log):
    with Session(engine) as session:
        session.add(models.MasteryState(learner_id=LEARNER, objective_id="ghost", p_mastery=1.0))
        session.commit()
    log.append(EventKind.SESSION_STARTED, LEARNER, {})
    rebuild(LEARNER, engine)
    assert _mastery_snapshot(engine) == set()


def test_rebuild_isolates_learners(engine, log):
    with Session(engine) as session:
        session.add(models.MasteryState(learner_id="other", objective_id="obj_a", p_mastery=0.5))
        session.commit()
    rebuild(LEARNER, engine)
    with Session(engine) as session:
        rows = session.exec(select(models.MasteryState)).all()
    assert [r.learner_id for r in rows] == ["other"]


def test_registry_counts_unhandled_kinds(engine, log):
    registry = ReducerRegistry()
    log.append("something.new", LEARNER, {})
    report = rebuild(LEARNER, engine, registry=registry)
    assert report.events_read == 1
    assert report.events_handled == 0
    assert report.events_skipped == 1


# -- idempotency ----------------------------------------------------------


class _Payload(BaseModel):
    question: str


class _Answer(BaseModel):
    text: str


class _CountingAgent(Agent[_Payload, _Answer]):
    name = "counting"
    task = Task.SUMMARISE
    input_type = _Payload
    output_type = _Answer

    def __init__(self, engine) -> None:
        super().__init__(engine, model="test:stub")
        self.calls = 0

    async def _run(self, ctx, payload):  # type: ignore[no-untyped-def]
        self.calls += 1
        return _Answer(text=f"{payload.question}:{self.calls}"), Usage(10, 5)


def test_hash_input_is_stable_and_order_independent():
    assert hash_input(_Payload(question="a")) == hash_input(_Payload(question="a"))
    assert hash_input({"a": 1, "b": 2}) == hash_input({"b": 2, "a": 1})
    assert hash_input(_Payload(question="a")) != hash_input(_Payload(question="b"))


def test_idempotency_cache_returns_the_cached_result(engine, ctx):
    agent = _CountingAgent(engine)
    payload = _Payload(question="q")

    first = asyncio.run(agent.run(ctx, payload, objective_id="obj_a"))
    second = asyncio.run(agent.run(ctx, payload, objective_id="obj_a"))

    assert agent.calls == 1
    assert first.cached is False
    assert second.cached is True
    assert second.output == first.output
    assert second.usage.total_tokens == 15


def test_idempotency_key_includes_objective_and_input(engine, ctx):
    agent = _CountingAgent(engine)
    asyncio.run(agent.run(ctx, _Payload(question="q"), objective_id="obj_a"))
    asyncio.run(agent.run(ctx, _Payload(question="q"), objective_id="obj_b"))
    asyncio.run(agent.run(ctx, _Payload(question="other"), objective_id="obj_a"))
    assert agent.calls == 3

    with Session(engine) as session:
        rows = session.exec(select(models.AgentCall)).all()
    assert len(rows) == 3
    assert {(r.agent, r.objective_id) for r in rows} == {
        ("counting", "obj_a"),
        ("counting", "obj_b"),
    }


def test_force_bypasses_the_cache_but_keeps_the_first_row(engine, ctx):
    agent = _CountingAgent(engine)
    payload = _Payload(question="q")
    asyncio.run(agent.run(ctx, payload, objective_id="obj_a"))
    forced = asyncio.run(agent.run(ctx, payload, objective_id="obj_a", force=True))
    assert agent.calls == 2
    assert forced.cached is False
    with Session(engine) as session:
        rows = session.exec(select(models.AgentCall)).all()
    assert len(rows) == 1


# -- context seam ---------------------------------------------------------


def test_context_owns_only_its_learner(ctx):
    assert ctx.owns(LEARNER)
    assert not ctx.owns("someone_else")
    with pytest.raises(AuthError):
        ctx.require("someone_else")


def test_context_resolves_from_settings():
    settings = Settings(learner_id="from_config")
    assert LearnerContext.resolve(settings).learner_id == "from_config"
    with pytest.raises(AuthError):
        LearnerContext.resolve(Settings(learner_id="  "))
