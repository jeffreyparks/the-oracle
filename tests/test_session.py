"""Session tests. Offline, no key, no network.

Every model call replays through the cassette harness with the hand-written
offline responder, and every learner is a :class:`SyntheticLearner` on a fixed
seed. A whole four-phase session runs in milliseconds, which is the only way
pedagogy gets tested at all (PLAN.md section 9).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, select

from the_oracle.agents.assessor import Item, misconceptions_for
from the_oracle.config import reset_settings_cache
from the_oracle.domains.registry import load_domain
from the_oracle.domains.schema import Domain
from the_oracle.mastery import MASTERY_THRESHOLD, SyntheticLearner, profile_for
from the_oracle.session import (
    ALLOWED_MINUTES,
    DEFAULT_SHAPE,
    Phase,
    SessionPlan,
    fatigue,
    plan_session,
    prerequisites_met,
    run_session,
    shape_for,
    should_interrupt,
    teachable,
)
from the_oracle.session.loop import ITEM_MINUTES, TEACH_MINUTES
from the_oracle.session.shape import budget_for, snap_minutes
from the_oracle.store import models
from the_oracle.store.db import build_engine, create_all
from the_oracle.store.events import EventKind, EventLog

from tests.cassettes.offline import synthesize
from tests.cassettes.replay import cassette

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PACKS = REPO_ROOT / "data" / "packs"
DOMAIN_ID = "bayesian_forecasting"
LEARNER = "session-learner"

ROOT = "prob_sample_space_events"
CHILD = "prob_random_variables"
BAYES = "bayes_theorem"

pytest.importorskip(
    "the_oracle.mastery.incremental",
    reason="the incremental apply path lands with the `mastery2` worker",
)


@pytest.fixture(autouse=True)
def packs_home(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point every path lookup at the repo\'s packs, not the user\'s home."""
    monkeypatch.setenv("ORACLE_HOME", str(DATA_PACKS))
    monkeypatch.setenv("ORACLE_LEARNER_ID", LEARNER)
    reset_settings_cache()
    yield DATA_PACKS
    reset_settings_cache()


@pytest.fixture
def engine() -> Engine:
    engine = build_engine("sqlite://")
    create_all(engine)
    return engine


@pytest.fixture
def domain() -> Domain:
    return load_domain(DOMAIN_ID)


# --- helpers ---------------------------------------------------------------


def seed(
    engine: Engine,
    objective_id: str,
    results: list[bool],
    *,
    at: datetime | None = None,
    difficulty: int = 2,
    learner_id: str = LEARNER,
) -> None:
    """Write real graded history through the same path the session uses."""
    from the_oracle.mastery.incremental import apply_response

    moment = at or datetime.now(UTC) - timedelta(days=30)
    for offset, correct in enumerate(results):
        apply_response(
            learner_id,
            {
                "objective_id": objective_id,
                "item_id": f"seed-{objective_id}-{offset}",
                "correct": bool(correct),
                "difficulty": difficulty,
                "bloom": "understand",
                "seconds": 5.0,
                "misconception_id": None,
            },
            engine=engine,
            occurred_at=moment + timedelta(seconds=offset),
        )


def answer_fn_for(domain: Domain, learner: SyntheticLearner) -> Callable[[Item], str]:
    """A synthetic learner as an answer source.

    A correct answer returns the expected text, which the offline grader marks
    right. A wrong answer returns the pack\'s wrong model when one exists, so
    the grader names the misconception the way a real one would.
    """

    def answer(item: Item) -> str:
        if learner.answer(item.objective_id, item.difficulty):
            return item.expected
        wrong = misconceptions_for(domain, item.objective_id)
        if wrong:
            return str(wrong[0]["wrong_model"])
        return "I am not sure; I would guess and move on."

    return answer


def always_wrong(domain: Domain) -> Callable[[Item], str]:
    return answer_fn_for(domain, SyntheticLearner({}, slip=0.0, guess=0.0, seed=1))


def presented(engine: Engine) -> list[dict[str, Any]]:
    return [e.payload for e in EventLog(engine).read(LEARNER, kinds=[EventKind.ITEM_PRESENTED])]


def mastery_rows(engine: Engine) -> dict[str, tuple[float, int]]:
    with Session(engine) as session:
        return {
            row.objective_id: (round(row.p_mastery, 9), row.observations)
            for row in session.exec(select(models.MasteryState))
        }


# --- the shape -------------------------------------------------------------


def test_the_shape_is_fixed_and_carries_the_reference_budgets() -> None:
    """PLAN.md section 5: 4 / 14 / 5 / 2, in that order, at 25 minutes."""
    assert [b.phase for b in DEFAULT_SHAPE] == [
        Phase.WARMUP,
        Phase.NEW,
        Phase.PRACTICE,
        Phase.CONSOLIDATE,
    ]
    assert [b.minutes for b in DEFAULT_SHAPE] == [4.0, 14.0, 5.0, 2.0]
    assert sum(b.minutes for b in DEFAULT_SHAPE) == 25


def test_only_the_length_moves_and_only_onto_supported_values() -> None:
    for minutes in ALLOWED_MINUTES:
        shape = shape_for(minutes)
        assert [b.phase for b in shape] == [b.phase for b in DEFAULT_SHAPE]
        assert sum(b.minutes for b in shape) == pytest.approx(minutes)
        for scaled, base in zip(shape, DEFAULT_SHAPE, strict=True):
            assert scaled.minutes == pytest.approx(base.minutes * minutes / 25, abs=0.02)
    assert snap_minutes(27) == 25
    assert snap_minutes(3) == 10
    assert snap_minutes(90) == 50


# --- fatigue ---------------------------------------------------------------


def test_fatigue_never_fires_before_min_items() -> None:
    """Five bad answers is a bad patch, not decay."""
    signal = fatigue([False] * 5, baseline=0.9, min_items=6)
    assert not signal.stop
    assert "too early" in signal.reason.lower()


def test_fatigue_fires_on_a_real_decay_against_the_learners_own_baseline() -> None:
    signal = fatigue([True] * 4 + [False] * 6, baseline=0.9)
    assert signal.stop
    assert signal.observed < signal.baseline
    assert "%" in signal.reason and "stopping here" in signal.reason.lower()


def test_fatigue_does_not_punish_a_learner_with_a_low_baseline() -> None:
    """A 45% learner running at 50% is having a normal day, not fading."""
    responses = [True, False, True, False, True, False, True, False]
    assert not fatigue(responses, baseline=0.45).stop
    assert fatigue(responses[:4] + [False] * 4, baseline=0.9).stop


# --- the interrupt ---------------------------------------------------------


def test_the_same_wrong_model_twice_interrupts_but_the_first_sighting_does_not() -> None:
    assert not should_interrupt([], "mis_a")
    assert not should_interrupt(["mis_b"], "mis_a")
    assert should_interrupt(["mis_a"], "mis_a")
    assert not should_interrupt(["mis_a"], None)


# --- planning --------------------------------------------------------------


def test_warmup_carries_due_reviews_oldest_first(engine: Engine) -> None:
    long_ago = datetime.now(UTC) - timedelta(days=90)
    seed(engine, ROOT, [True, True], at=long_ago)
    seed(engine, "prob_conditional_independence", [True, True], at=long_ago + timedelta(days=10))
    seed(engine, "prob_mc_integration", [True, True], at=long_ago + timedelta(days=20))

    plan = plan_session(DOMAIN_ID, LEARNER, minutes=25, engine=engine)
    assert plan.warmup == [ROOT, "prob_conditional_independence", "prob_mc_integration"]
    assert plan.minutes == 25
    assert plan.shape == DEFAULT_SHAPE


def test_warmup_falls_back_to_recent_objectives_when_nothing_is_due(engine: Engine) -> None:
    just_now = datetime.now(UTC) - timedelta(minutes=2)
    seed(engine, ROOT, [True, True], at=just_now - timedelta(minutes=5))
    seed(engine, "prob_conditional_independence", [True, True], at=just_now)

    from the_oracle.session import due_reviews

    profile = profile_for(LEARNER, engine)
    domain = load_domain(DOMAIN_ID)
    assert due_reviews(domain, profile, LEARNER, engine=engine) == []

    plan = plan_session(DOMAIN_ID, LEARNER, minutes=25, engine=engine)
    assert plan.warmup, "a session must still open with retrieval practice"
    assert plan.warmup[0] == "prob_conditional_independence"  # most recent first


def test_focus_only_ever_contains_teachable_objectives(engine: Engine, domain: Domain) -> None:
    seed(engine, ROOT, [True, True])
    plan = plan_session(DOMAIN_ID, LEARNER, minutes=25, engine=engine)
    profile = profile_for(LEARNER, engine)

    assert plan.focus
    allowed = set(teachable(domain, profile))
    for objective_id in plan.focus:
        assert objective_id in allowed
        assert prerequisites_met(domain, objective_id, profile)
        for prerequisite in domain.prerequisites(objective_id):
            assert profile.p(prerequisite) >= MASTERY_THRESHOLD


def test_focus_uses_the_strict_gate_not_the_loose_diagnostic_one(
    engine: Engine, domain: Domain
) -> None:
    """One correct answer is not mastery. The loose gate would let it through."""
    seed(engine, ROOT, [True])  # p ~= 0.6, below the bar
    profile = profile_for(LEARNER, engine)
    assert profile.p(ROOT) < MASTERY_THRESHOLD
    assert not prerequisites_met(domain, CHILD, profile)

    plan = plan_session(DOMAIN_ID, LEARNER, minutes=25, engine=engine)
    assert CHILD not in plan.focus


def test_a_long_session_may_carry_a_second_new_objective(engine: Engine) -> None:
    seed(engine, ROOT, [True, True])
    short = plan_session(DOMAIN_ID, LEARNER, minutes=10, engine=engine)
    long = plan_session(DOMAIN_ID, LEARNER, minutes=50, engine=engine)
    assert len(short.focus) == 1
    assert len(long.focus) <= 2


# --- running a session -----------------------------------------------------


def test_the_four_phases_run_in_order_and_respect_their_budgets(
    engine: Engine, domain: Domain
) -> None:
    long_ago = datetime.now(UTC) - timedelta(days=60)
    seed(engine, ROOT, [True, True], at=long_ago)
    seed(engine, "prob_conditional_independence", [True, True], at=long_ago)

    plan = plan_session(DOMAIN_ID, LEARNER, minutes=25, engine=engine)
    learner = SyntheticLearner(
        {oid: 0.95 for oid in domain.teaching_order()}, slip=0.05, guess=0.2, seed=11
    )
    with cassette("session_e2e", synthesize=synthesize, write=False):
        result = run_session(plan, answer_fn=answer_fn_for(domain, learner), engine=engine)

    payloads = presented(engine)
    assert payloads, "a session that asks nothing is not a session"

    order = [Phase.WARMUP, Phase.NEW, Phase.PRACTICE]
    rank = {str(phase): i for i, phase in enumerate(order)}
    seen = [rank[p["phase"]] for p in payloads]
    assert seen == sorted(seen), "phases must never run out of order"

    for phase in order:
        spent = sum(
            ITEM_MINUTES[p["kind"]] for p in payloads if p["phase"] == str(phase)
        )
        if phase is Phase.NEW:
            spent += TEACH_MINUTES * len(plan.focus)
        assert spent <= budget_for(plan.shape, phase) + 1e-6, phase

    assert result.asked == len(payloads)
    assert set(result.objectives_touched) >= set(plan.focus)


def test_every_run_is_bracketed_by_session_started_and_ended(
    engine: Engine, domain: Domain
) -> None:
    seed(engine, ROOT, [True, True], at=datetime.now(UTC) - timedelta(days=60))
    plan = plan_session(DOMAIN_ID, LEARNER, minutes=10, engine=engine)
    learner = SyntheticLearner({oid: 0.9 for oid in domain.teaching_order()}, seed=3)
    with cassette("session_e2e", synthesize=synthesize, write=False):
        result = run_session(plan, answer_fn=answer_fn_for(domain, learner), engine=engine)

    events = EventLog(engine).read(LEARNER)
    kinds = [e.kind for e in events]
    assert kinds.index(str(EventKind.SESSION_STARTED)) < kinds.index(str(EventKind.ITEM_PRESENTED))
    assert kinds[-1] == str(EventKind.SESSION_ENDED)

    started = next(e for e in events if e.kind == str(EventKind.SESSION_STARTED))
    ended = events[-1]
    assert started.payload["session_id"] == ended.payload["session_id"] == result.session_id
    assert ended.payload["asked"] == result.asked
    assert EventKind.REVIEW_SCHEDULED in {e.kind for e in events}


def test_a_session_ends_early_on_measured_fatigue_and_says_why(
    engine: Engine, domain: Domain
) -> None:
    """The learner\'s own baseline is high, and today nothing lands."""
    seed(engine, ROOT, [True] * 10, at=datetime.now(UTC) - timedelta(days=60))
    plan = SessionPlan(
        learner_id=LEARNER,
        domain_id=DOMAIN_ID,
        minutes=25,
        warmup=[ROOT],
        focus=[CHILD],
        shape=DEFAULT_SHAPE,
    )
    with cassette("session_e2e", synthesize=synthesize, write=False):
        result = run_session(plan, answer_fn=always_wrong(domain), engine=engine)

    assert result.ended_early
    assert result.correct == 0
    assert result.asked >= 6, "fatigue may not fire before min_items"
    assert "usually run at" in result.reason
    assert str(Phase.PRACTICE) not in {p["phase"] for p in presented(engine)}


def test_a_repeated_misconception_interrupts_and_reteaches(
    engine: Engine, domain: Domain
) -> None:
    for prerequisite in domain.prerequisites(BAYES):
        seed(engine, prerequisite, [True, True])
    plan = SessionPlan(
        learner_id=LEARNER,
        domain_id=DOMAIN_ID,
        minutes=25,
        warmup=[],
        focus=[BAYES],
        shape=DEFAULT_SHAPE,
    )
    with cassette("session_e2e", synthesize=synthesize, write=False):
        result = run_session(plan, answer_fn=always_wrong(domain), engine=engine)

    tags = [
        e.payload
        for e in EventLog(engine).read(LEARNER, kinds=[EventKind.MISCONCEPTION_DETECTED])
    ]
    assert len(tags) >= 2
    assert tags[0]["repeat"] is False, "a first sighting must not interrupt"
    assert any(tag["repeat"] for tag in tags[1:]), "the second sighting must"
    assert "twice" in result.reason
    assert tags[0]["misconception_id"] in result.reason


def test_drift_drops_a_focus_objective_rather_than_teaching_blind(
    engine: Engine, domain: Domain
) -> None:
    """A prerequisite can decay inside the session. The warm-up is real evidence."""
    seed(engine, ROOT, [True, True], at=datetime.now(UTC) - timedelta(days=60))
    profile = profile_for(LEARNER, engine)
    assert prerequisites_met(domain, CHILD, profile), "the plan was sound when it was made"

    plan = SessionPlan(
        learner_id=LEARNER,
        domain_id=DOMAIN_ID,
        minutes=25,
        warmup=[ROOT],
        focus=[CHILD],
        shape=DEFAULT_SHAPE,
    )
    with cassette("session_e2e", synthesize=synthesize, write=False):
        result = run_session(plan, answer_fn=always_wrong(domain), engine=engine)

    after = profile_for(LEARNER, engine)
    assert after.p(ROOT) < MASTERY_THRESHOLD, "the warm-up really did knock it down"
    assert CHILD not in {p["objective_id"] for p in presented(engine)}
    assert f"Dropped {CHILD}" in result.reason
    assert ROOT in result.reason


def test_the_session_writes_mastery_only_through_events(
    engine: Engine, domain: Domain
) -> None:
    """Incremental apply on the real path must equal a full replay, exactly."""
    from the_oracle.store.rebuild import drop_derived, rebuild

    seed(engine, ROOT, [True, True], at=datetime.now(UTC) - timedelta(days=60))
    plan = plan_session(DOMAIN_ID, LEARNER, minutes=25, engine=engine)
    learner = SyntheticLearner({oid: 0.8 for oid in domain.teaching_order()}, seed=5)
    with cassette("session_e2e", synthesize=synthesize, write=False):
        run_session(plan, answer_fn=answer_fn_for(domain, learner), engine=engine)

    snapshot = mastery_rows(engine)
    assert snapshot

    drop_derived(LEARNER, engine)
    rebuild(LEARNER, engine)
    assert mastery_rows(engine) == snapshot


# --- the whole point -------------------------------------------------------


def test_studying_across_sessions_raises_mastery_without_ever_skipping_a_prerequisite(
    engine: Engine, domain: Domain
) -> None:
    """Several sessions of honest work move the learner, and never teach blind.

    This is the claim the whole system makes. A synthetic learner who can learn
    anything studies four spaced sessions; mastery must rise, and no objective
    may be taught before every one of its prerequisites cleared 0.85.
    """
    learner = SyntheticLearner(
        {oid: 0.95 for oid in domain.teaching_order()}, slip=0.05, guess=0.2, seed=17
    )
    answer = answer_fn_for(domain, learner)

    start = profile_for(LEARNER, engine)
    mastered_at_start = len(start.mastered())
    taught: list[str] = []

    with cassette("session_e2e", synthesize=synthesize, write=False):
        for index in range(4):
            now = datetime.now(UTC) + timedelta(days=3 * index)
            plan = plan_session(DOMAIN_ID, LEARNER, minutes=25, now=now, engine=engine)
            before = profile_for(LEARNER, engine)
            for objective_id in plan.focus:
                assert prerequisites_met(domain, objective_id, before), (
                    f"session {index} planned {objective_id} with a prerequisite unmet"
                )
            result = run_session(plan, answer_fn=answer, engine=engine)
            taught.extend(plan.focus)
            assert result.asked > 0

    end = profile_for(LEARNER, engine)
    assert len(end.mastered()) > mastered_at_start
    assert sum(end.as_dict().values()) > sum(start.as_dict().values())

    # every objective taught had a clear road behind it at the moment it was taught
    order = domain.teaching_order()
    for position, objective_id in enumerate(taught):
        earlier = set(taught[:position]) | end.mastered()
        for prerequisite in domain.prerequisites(objective_id):
            assert prerequisite in earlier or end.p(prerequisite) >= MASTERY_THRESHOLD, (
                f"{objective_id} was taught before {prerequisite}"
            )
        assert objective_id in order

    # and the warm-up kept doing its job: review items rode inside the session
    graded = EventLog(engine).read(LEARNER, kinds=[EventKind.RESPONSE_GRADED])
    phases = {e.payload.get("phase") for e in graded}
    assert str(Phase.WARMUP) in phases


def test_a_session_with_nothing_to_do_says_so_and_invents_no_work(engine: Engine) -> None:
    plan = SessionPlan(
        learner_id=LEARNER, domain_id=DOMAIN_ID, minutes=25, warmup=[], focus=[],
        shape=DEFAULT_SHAPE,
    )
    with cassette("session_e2e", synthesize=synthesize, write=False):
        result = run_session(plan, answer_fn=lambda item: "", engine=engine)
    assert result.asked == 0
    assert not result.ended_early
    assert "no session to run" in result.reason
    assert presented(engine) == []


# --- reachability ----------------------------------------------------------


def test_the_study_command_reaches_run_session() -> None:
    """The recurring failure of this project is code that ships unreachable."""
    import inspect

    study = pytest.importorskip(
        "the_oracle.commands.study",
        reason="the study command lands with the `commands4` worker",
    )
    source = inspect.getsource(study)
    assert "run_session" in source
    assert "plan_session" in source
