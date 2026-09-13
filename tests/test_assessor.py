"""Assessor tests. Offline, free, deterministic.

Every LLM call runs through the cassette harness in ``tests/cassettes``. No
API key was available when this suite was written, so the cassettes are
hand-written by :mod:`tests.cassettes.offline` and replayed strictly. A miss
fails loudly; it never falls through to the network.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, select

from the_oracle.agents.assessor import (
    Grade,
    GradeRequest,
    Grader,
    Item,
    ItemRequest,
    ItemWriter,
    feedback_is_immediate,
    kind_for_bloom,
    misconceptions_for,
    next_difficulty,
    next_objective,
    run_diagnostic,
)
from the_oracle.config import reset_settings_cache
from the_oracle.context import LearnerContext
from the_oracle.domains.registry import load_domain
from the_oracle.domains.schema import Domain
from the_oracle.store import models
from the_oracle.store.db import build_engine, create_all
from the_oracle.store.events import EventKind, EventLog

from tests.cassettes import record as cassette_record
from tests.cassettes.offline import synthesize
from tests.cassettes.replay import CassetteMiss, cassette

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PACKS = REPO_ROOT / "data" / "packs"
DOMAIN_ID = "bayesian_forecasting"
LEARNER = "test-learner"

pytest.importorskip(
    "the_oracle.mastery",
    reason="the mastery package lands with the `mastery` worker",
)


@pytest.fixture(autouse=True)
def packs_home(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point every path lookup at the repo's packs, not the user's home."""
    monkeypatch.setenv("ORACLE_HOME", str(DATA_PACKS))
    monkeypatch.setenv("ORACLE_LEARNER_ID", LEARNER)
    reset_settings_cache()
    yield DATA_PACKS
    reset_settings_cache()


@pytest.fixture
def engine() -> Engine:
    """A throwaway in-memory database with every table created."""
    engine = build_engine("sqlite://")
    create_all(engine)
    return engine


@pytest.fixture
def domain() -> Domain:
    return load_domain(DOMAIN_ID)


@pytest.fixture
def ctx() -> LearnerContext:
    return LearnerContext.for_learner(LEARNER)


# --- pedagogy mapping ------------------------------------------------------


def test_bloom_level_picks_the_item_kind() -> None:
    assert kind_for_bloom("remember") == "recall"
    assert kind_for_bloom("understand") == "short"
    assert kind_for_bloom("apply") == "short"
    assert kind_for_bloom("analyze") == "multi_step"
    assert kind_for_bloom("create") == "multi_step"


def test_feedback_is_immediate_except_for_multi_step() -> None:
    """PLAN.md section 5: immediate, but never mid-attempt."""
    assert feedback_is_immediate("recall")
    assert feedback_is_immediate("short")
    assert not feedback_is_immediate("multi_step")


def test_difficulty_steers_toward_the_target_accuracy() -> None:
    assert next_difficulty(3, 0, 0, 0.7) == 3
    assert next_difficulty(3, 10, 10, 0.7) == 4  # too easy, push
    assert next_difficulty(3, 10, 2, 0.7) == 2  # too hard, ease off
    assert next_difficulty(3, 10, 7, 0.7) == 3  # on target, hold
    assert next_difficulty(5, 10, 10, 0.7) == 5  # clamped
    assert next_difficulty(1, 10, 0, 0.7) == 1


# --- the two agents, replayed ---------------------------------------------


def test_item_writer_replays_a_cassette(engine: Engine, ctx: LearnerContext) -> None:
    request = next(p for name, p in cassette_record.unit_cases() if name == "item_writer")
    assert isinstance(request, ItemRequest)
    with cassette("assessor_units") as tape:
        result = anyio_run(ItemWriter(engine).run(ctx, request, objective_id=request.objective_id))
    assert [call["source"] for call in tape.calls] == ["cassette"]

    item = result.output
    assert isinstance(item, Item)
    assert item.objective_id == request.objective_id
    assert item.bloom == request.bloom
    assert item.difficulty == request.difficulty
    assert item.kind == kind_for_bloom(request.bloom)
    assert item.stem and item.expected


def test_item_writer_caches_on_retry(engine: Engine, ctx: LearnerContext) -> None:
    """The idempotency key stops a retry from authoring a second item."""
    request = next(p for name, p in cassette_record.unit_cases() if name == "item_writer")
    writer = ItemWriter(engine)
    with cassette("assessor_units") as tape:
        first = anyio_run(writer.run(ctx, request))
        second = anyio_run(writer.run(ctx, request))
    assert len(tape.calls) == 1
    assert second.cached and not first.cached
    assert second.output.id == first.output.id


def test_grader_marks_a_correct_answer(engine: Engine, ctx: LearnerContext) -> None:
    right = [p for name, p in cassette_record.unit_cases() if name == "grader"][0]
    with cassette("assessor_units"):
        grade = anyio_run(Grader(engine).run(ctx, right)).output
    assert grade.correct
    assert grade.misconception_id is None
    assert 0.0 <= grade.confidence <= 1.0
    assert grade.feedback and grade.reasoning


def test_grader_names_the_misconception(engine: Engine, ctx: LearnerContext) -> None:
    wrong = [p for name, p in cassette_record.unit_cases() if name == "grader"][1]
    known = {m["id"] for m in wrong.misconceptions}
    with cassette("assessor_units"):
        grade = anyio_run(Grader(engine).run(ctx, wrong)).output
    assert not grade.correct
    assert grade.misconception_id in known


def test_grader_leaves_misconception_null_when_nothing_matches(
    engine: Engine, ctx: LearnerContext
) -> None:
    empty = [p for name, p in cassette_record.unit_cases() if name == "grader"][2]
    with cassette("assessor_units"):
        grade = anyio_run(Grader(engine).run(ctx, empty)).output
    assert not grade.correct
    assert grade.misconception_id is None


def test_grader_drops_an_invented_misconception_id(engine: Engine) -> None:
    """A model may hallucinate an id. The pack decides what exists."""
    request = [p for name, p in cassette_record.unit_cases() if name == "grader"][1]
    invented = Grade(
        correct=False,
        confidence=1.4,
        misconception_id="mis_not_in_any_pack",
        feedback="wrong",
        reasoning="made it up",
    )
    cleaned = Grader(engine).normalise(invented, request)
    assert cleaned.misconception_id is None
    assert cleaned.confidence == 1.0


def test_a_cassette_miss_fails_loudly(engine: Engine, ctx: LearnerContext) -> None:
    """No silent fallback to the network."""
    request = ItemRequest(objective_id="never_recorded", bloom="apply", difficulty=2, stems=[])
    with cassette("assessor_units"), pytest.raises(CassetteMiss):
        anyio_run(ItemWriter(engine).run(ctx, request))


def test_misconceptions_come_from_the_pack(domain: Domain) -> None:
    found = misconceptions_for(domain, "bayes_theorem")
    assert found
    assert all(set(m) == {"id", "wrong_model", "diagnostic"} for m in found)
    assert misconceptions_for(domain, "prob_sample_space_events") != found


# --- selection policy ------------------------------------------------------


@dataclass
class FakeProfile:
    """Stands in for MasteryProfile in the pure selection tests."""

    values: dict[str, float]
    blocked: set[str]

    def p(self, objective_id: str) -> float:
        return self.values.get(objective_id, 0.2)

    def known_prerequisites_met(self, domain: Domain, objective_id: str) -> bool:
        return objective_id not in self.blocked


def test_next_objective_picks_the_estimate_nearest_a_coin_flip(domain: Domain) -> None:
    order = domain.teaching_order()
    a, b, c = order[0], order[1], order[2]
    profile = FakeProfile({a: 0.2, b: 0.48, c: 0.35}, blocked=set())
    assert next_objective(domain, profile, asked=set()) == b


def test_next_objective_never_repeats(domain: Domain) -> None:
    order = domain.teaching_order()
    a, b = order[0], order[1]
    profile = FakeProfile({a: 0.5, b: 0.49}, blocked=set())
    assert next_objective(domain, profile, asked={a}) == b


def test_next_objective_respects_prerequisites(domain: Domain) -> None:
    order = domain.teaching_order()
    a, b = order[0], order[1]
    profile = FakeProfile({a: 0.5, b: 0.49}, blocked={a})
    assert next_objective(domain, profile, asked=set()) == b


def test_next_objective_stops_when_every_estimate_is_settled(domain: Domain) -> None:
    settled = {ref.id: 0.97 for ref in domain.objectives}
    profile = FakeProfile(settled, blocked=set())
    assert next_objective(domain, profile, asked=set()) is None


# --- the loop, end to end --------------------------------------------------


def synthetic_answer_fn(domain: Domain, *, seed: int = 7) -> "Any":
    """An answer source backed by :class:`SyntheticLearner`.

    Strong on the early objectives, weak later on: a learner with a real
    frontier rather than one who knows everything. Slip and guess make the
    stream stochastic; the seed keeps it deterministic.
    """
    from the_oracle.mastery import SyntheticLearner

    order = domain.teaching_order()
    true_mastery = {oid: (0.9 if i < 8 else 0.25) for i, oid in enumerate(order)}
    learner = SyntheticLearner(true_mastery, slip=0.1, guess=0.2, seed=seed)

    def answer(item: Item) -> str:
        if learner.answer(item.objective_id, item.difficulty):
            return item.expected
        wrong = misconceptions_for(domain, item.objective_id)
        if wrong:
            return str(wrong[0]["wrong_model"])
        return "I am not sure; I would guess and move on."

    return answer


def test_run_diagnostic_end_to_end(engine: Engine, domain: Domain) -> None:
    from the_oracle.mastery import profile_for
    from the_oracle.store.rebuild import rebuild

    seen: list[tuple[Item, Grade]] = []
    with cassette("assessor_e2e", synthesize=synthesize, write=False):
        profile = run_diagnostic(
            DOMAIN_ID,
            LEARNER,
            max_items=12,
            answer_fn=synthetic_answer_fn(domain),
            engine=engine,
            on_graded=lambda item, answer, grade: seen.append((item, grade)),
        )

    log = EventLog(engine)
    presented = log.read(LEARNER, kinds=[EventKind.ITEM_PRESENTED])
    graded = log.read(LEARNER, kinds=[EventKind.RESPONSE_GRADED])

    # caps and non-repetition
    assert 0 < len(presented) <= 12
    assert len(graded) == len(presented) == len(seen)
    asked = [e.payload["objective_id"] for e in graded]
    assert len(set(asked)) == len(asked)

    # no objective was asked before its prerequisites were cleared
    right = {e.payload["objective_id"] for e in graded if e.payload["correct"]}
    for objective_id in asked:
        for prerequisite in domain.prerequisites(objective_id):
            cleared = prerequisite in right or profile.p(prerequisite) >= 0.85
            assert cleared, (objective_id, prerequisite)

    # the graded payload matches the contract the reducers read
    for event in graded:
        assert set(event.payload) == {
            "objective_id",
            "item_id",
            "correct",
            "difficulty",
            "bloom",
            "seconds",
            "misconception_id",
        }
        assert isinstance(event.payload["correct"], bool)

    # accuracy lands near the 70% target
    accuracy = sum(1 for e in graded if e.payload["correct"]) / len(graded)
    assert 0.4 <= accuracy <= 0.9

    # the returned profile is the derived one, and it saw every graded objective
    assert set(profile.as_dict()) == set(asked)
    before = profile.as_dict()
    rebuild(LEARNER, engine)
    assert profile_for(LEARNER, engine).as_dict() == before


def test_mastery_is_derivable_from_the_log_alone(engine: Engine, domain: Domain) -> None:
    """The loop may write events only. Everything else must replay from them."""
    from the_oracle.store.rebuild import drop_derived, rebuild

    with cassette("assessor_e2e", synthesize=synthesize, write=False):
        run_diagnostic(
            DOMAIN_ID,
            LEARNER,
            max_items=6,
            answer_fn=synthetic_answer_fn(domain, seed=3),
            engine=engine,
        )

    snapshot = _mastery_rows(engine)
    assert snapshot, "the diagnostic produced no mastery at all"

    drop_derived(LEARNER, engine)
    assert not _mastery_rows(engine)

    rebuild(LEARNER, engine)
    assert _mastery_rows(engine) == snapshot

    # and nothing outside the event log carries the signal
    log = EventLog(engine)
    kinds = {e.kind for e in log.read(LEARNER)}
    assert EventKind.ITEM_PRESENTED in kinds
    assert EventKind.RESPONSE_GRADED in kinds


def _mastery_rows(engine: Engine) -> dict[tuple[str, str], tuple[float, int]]:
    """MasteryState as a plain comparable mapping."""
    with Session(engine) as session:
        return {
            (row.learner_id, row.objective_id): (round(row.p_mastery, 9), row.observations)
            for row in session.exec(select(models.MasteryState))
        }


def anyio_run(coro: Any) -> Any:
    """Run one coroutine. The agents are async; these tests are not."""
    import asyncio

    return asyncio.run(coro)
