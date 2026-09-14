"""The strict teaching gate, and the incremental mastery fast path.

Two pieces of Phase 1 debt are paid here, and both have a claim that must be
proved rather than asserted in a comment:

* the strict gate and the diagnostic's loose gate **disagree** (they must never
  quietly converge), and
* the incremental apply path produces **exactly** what a full replay produces.
"""

from __future__ import annotations

import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlmodel import Session, select

from the_oracle.agents.assessor import prerequisites_ready
from the_oracle.domains.schema import Domain, Edge, Module, ObjectiveRef
from the_oracle.mastery import incremental
from the_oracle.mastery.bkt import MASTERY_THRESHOLD, BKTParams
from the_oracle.mastery.gate import blocked_by, prerequisites_met, teachable
from the_oracle.mastery.profile import MasteryProfile, profile_for
from the_oracle.store import models
from the_oracle.store.db import build_engine, create_all
from the_oracle.store.events import EventKind, EventLog
from the_oracle.store.rebuild import rebuild

LEARNER = "learner_gate"
START = datetime(2025, 3, 1, 9, 0, tzinfo=UTC)
OBJECTIVES = ("o_one", "o_two", "o_three", "o_four")


@pytest.fixture
def engine():
    eng = build_engine("sqlite:///:memory:")
    create_all(eng)
    yield eng
    eng.dispose()


def domain() -> Domain:
    """A single-root chain with one branch: o_one -> o_two -> {o_three, o_four}."""
    return Domain(
        id="d_gate",
        version=1,
        title="Gate manifest",
        objectives=[ObjectiveRef(id=o, version=1) for o in OBJECTIVES],
        edges=[
            Edge(from_id="o_one", to_id="o_two"),
            Edge(from_id="o_two", to_id="o_three"),
            Edge(from_id="o_two", to_id="o_four"),
            Edge(from_id="o_three", to_id="o_four"),
        ],
        modules=[Module(id="m1", title="M", goal="g", objectives=list(OBJECTIVES))],
    )


# -- the strict gate -------------------------------------------------------


def test_a_root_objective_is_always_teachable():
    profile = MasteryProfile({})
    assert prerequisites_met(domain(), "o_one", profile) is True
    assert blocked_by(domain(), "o_one", profile) == []


def test_an_unseen_prerequisite_blocks_teaching():
    """"We have never looked" is not "they know it". The prior is 0.2."""
    profile = MasteryProfile({})
    assert profile.p("o_one") == pytest.approx(BKTParams().p_init)
    assert prerequisites_met(domain(), "o_two", profile) is False
    assert blocked_by(domain(), "o_two", profile) == ["o_one"]


def test_blocked_by_names_every_unmet_prerequisite_in_manifest_order():
    profile = MasteryProfile({"o_one": 0.99, "o_two": 0.3, "o_three": 0.2})
    assert blocked_by(domain(), "o_four", profile) == ["o_two", "o_three"]
    profile = MasteryProfile({"o_one": 0.99, "o_two": 0.9, "o_three": 0.2})
    assert blocked_by(domain(), "o_four", profile) == ["o_three"]


def test_the_gate_uses_the_mastery_threshold_exactly():
    just_under = MasteryProfile({"o_one": MASTERY_THRESHOLD - 1e-9})
    just_over = MasteryProfile({"o_one": MASTERY_THRESHOLD})
    assert prerequisites_met(domain(), "o_two", just_under) is False
    assert prerequisites_met(domain(), "o_two", just_over) is True


def test_teachable_is_unmastered_and_unblocked_in_teaching_order():
    profile = MasteryProfile({"o_one": 0.95, "o_two": 0.4, "o_three": 0.1})
    assert teachable(domain(), profile) == ["o_two"]

    profile = MasteryProfile({"o_one": 0.95, "o_two": 0.9, "o_three": 0.1})
    assert teachable(domain(), profile) == ["o_three"]

    mastered = MasteryProfile({oid: 0.99 for oid in OBJECTIVES})
    assert teachable(domain(), mastered) == []  # nothing left; never invent work


def test_a_custom_threshold_is_honoured():
    profile = MasteryProfile({"o_one": 0.6})
    assert prerequisites_met(domain(), "o_two", profile) is False
    assert prerequisites_met(domain(), "o_two", profile, threshold=0.5) is True


# -- the divergence that must never close ---------------------------------


def test_strict_and_loose_gates_disagree_on_a_just_answered_prerequisite():
    """The required test: the two gates must not silently converge.

    Constructed case: the learner has answered ``o_one`` correctly once, so the
    diagnostic counts it as passed, but one correct answer only lifts the
    posterior to about 0.63 - far below the 0.85 mastery bar.

    The diagnostic may probe ``o_two`` on that evidence. Teaching may not.
    """
    dom = domain()
    from the_oracle.mastery.bkt import update

    p_after_one_correct = update(BKTParams().p_init, True)
    assert 0.5 < p_after_one_correct < MASTERY_THRESHOLD  # the whole point

    profile = MasteryProfile({"o_one": p_after_one_correct})
    passed = {"o_one"}  # the diagnostic just saw a correct answer

    loose = prerequisites_ready(dom, profile, "o_two", passed)
    strict = prerequisites_met(dom, "o_two", profile)

    assert loose is True, "the diagnostic gate must stay loose, or probing stalls"
    assert strict is False, "the teaching gate must require p >= 0.85, always"
    assert loose != strict
    assert blocked_by(dom, "o_two", profile) == ["o_one"]
    assert "o_two" not in teachable(dom, profile)


def test_the_two_gates_only_agree_once_the_prerequisite_is_truly_mastered():
    dom = domain()
    profile = MasteryProfile({"o_one": 0.95})
    assert prerequisites_ready(dom, profile, "o_two", set()) is True
    assert prerequisites_met(dom, "o_two", profile) is True


# -- incremental apply -----------------------------------------------------


def graded(
    objective_id: str,
    correct: bool,
    *,
    item_id: str = "item_1",
    difficulty: int = 2,
    seconds: float = 11.0,
    misconception_id: str | None = None,
) -> dict:
    return {
        "objective_id": objective_id,
        "item_id": item_id,
        "correct": correct,
        "difficulty": difficulty,
        "bloom": "understand",
        "seconds": seconds,
        "misconception_id": misconception_id,
    }


def test_apply_event_does_not_mutate_the_row_it_is_given(engine):
    log = EventLog(engine)
    event = log.append(EventKind.RESPONSE_GRADED, LEARNER, graded("o_one", True))
    before = incremental.apply_event(None, event)
    after = incremental.apply_event(before, event)
    assert before.observations == 1
    assert after.observations == 2
    assert after.p_mastery > before.p_mastery


def test_apply_event_starts_an_unseen_objective_at_the_prior(engine):
    log = EventLog(engine)
    event = log.append(EventKind.MISCONCEPTION_DETECTED, LEARNER, {
        "objective_id": "o_one", "misconception_id": "mc_1"
    })
    state = incremental.apply_event(None, event)
    assert state.p_mastery == pytest.approx(BKTParams().p_init)
    assert state.observations == 0  # a tag is not evidence
    assert state.misconception_ids == ["mc_1"]


def _script(seed: int, length: int) -> list[tuple[str, bool, int, str | None]]:
    """A random response stream: mixed correct/incorrect, repeated objectives."""
    rng = random.Random(seed)
    out = []
    for _ in range(length):
        objective_id = rng.choice(OBJECTIVES)
        correct = rng.random() < 0.6
        difficulty = rng.randint(1, 5)
        misconception_id = (
            rng.choice(["mc_1", "mc_2"]) if not correct and rng.random() < 0.5 else None
        )
        out.append((objective_id, correct, difficulty, misconception_id))
    return out


def _replay_engine(script, learner: str = LEARNER):
    """Append the same stream to a fresh log, then rebuild. The source of truth."""
    eng = build_engine("sqlite:///:memory:")
    create_all(eng)
    log = EventLog(eng)
    for index, (objective_id, correct, difficulty, misconception_id) in enumerate(script):
        at = START + timedelta(minutes=index)
        log.append(
            EventKind.RESPONSE_GRADED,
            learner,
            graded(objective_id, correct, item_id=f"item_{index}",
                   difficulty=difficulty, misconception_id=misconception_id),
            occurred_at=at,
        )
        if misconception_id:
            log.append(
                EventKind.MISCONCEPTION_DETECTED,
                learner,
                {"objective_id": objective_id, "misconception_id": misconception_id,
                 "item_id": f"item_{index}"},
                occurred_at=at,
            )
    rebuild(learner, eng)
    return eng


def _incremental_engine(script, learner: str = LEARNER):
    eng = build_engine("sqlite:///:memory:")
    create_all(eng)
    for index, (objective_id, correct, difficulty, misconception_id) in enumerate(script):
        at = START + timedelta(minutes=index)
        incremental.apply_response(
            learner,
            graded(objective_id, correct, item_id=f"item_{index}",
                   difficulty=difficulty, misconception_id=misconception_id),
            engine=eng,
            occurred_at=at,
        )
        if misconception_id:
            incremental.apply_misconception(
                learner,
                {"objective_id": objective_id, "misconception_id": misconception_id,
                 "item_id": f"item_{index}"},
                engine=eng,
                occurred_at=at,
            )
    return eng


def _schedules(eng, learner: str = LEARNER) -> dict[str, tuple]:
    with Session(eng) as session:
        rows = session.exec(
            select(models.ReviewSchedule).where(models.ReviewSchedule.learner_id == learner)
        )
        return {
            row.objective_id: (row.due_at, row.interval_days, row.stability,
                               row.difficulty, row.lapses)
            for row in rows
        }


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11, 42])
def test_incremental_apply_matches_full_replay_exactly(seed):
    """The required test. Full replay is the truth; this path must reproduce it."""
    script = _script(seed, 40)
    fast = _incremental_engine(script)
    truth = _replay_engine(script)
    try:
        fast_profile = profile_for(LEARNER, fast)
        truth_profile = profile_for(LEARNER, truth)

        assert set(fast_profile.as_dict()) == set(truth_profile.as_dict())
        for objective_id, p in truth_profile.as_dict().items():
            assert fast_profile.p(objective_id) == pytest.approx(p, abs=1e-9), objective_id
            assert fast_profile.observations(objective_id) == truth_profile.observations(objective_id)
            assert fast_profile.confidence(objective_id) == pytest.approx(
                truth_profile.confidence(objective_id), abs=1e-9
            )
            assert fast_profile.misconceptions(objective_id) == truth_profile.misconceptions(objective_id)
            assert fast_profile.last_seen(objective_id) == truth_profile.last_seen(objective_id)

        assert _schedules(fast).keys() == _schedules(truth).keys()
        for objective_id, expected in _schedules(truth).items():
            got = _schedules(fast)[objective_id]
            assert got[0] == expected[0], objective_id          # due_at
            for a, b in zip(got[1:], expected[1:], strict=True):
                assert a == pytest.approx(b, abs=1e-9), objective_id
    finally:
        fast.dispose()
        truth.dispose()


def test_incremental_and_replay_agree_on_the_response_table():
    script = _script(5, 25)
    fast = _incremental_engine(script)
    truth = _replay_engine(script)
    try:
        def rows(eng):
            with Session(eng) as session:
                return [
                    (r.objective_id, r.item_id, r.correct, r.score, r.latency_ms,
                     r.misconception_ids, r.created_at)
                    for r in session.exec(
                        select(models.Response).order_by(models.Response.id)
                    )
                ]
        assert rows(fast) == rows(truth)
    finally:
        fast.dispose()
        truth.dispose()


def test_apply_response_returns_the_new_posterior(engine):
    first = incremental.apply_response(LEARNER, graded("o_one", True), engine=engine)
    second = incremental.apply_response(LEARNER, graded("o_one", True), engine=engine)
    assert second > first > BKTParams().p_init
    assert incremental.mastery_for(LEARNER, "o_one", engine=engine) == pytest.approx(second)
    rebuild(LEARNER, engine)
    assert profile_for(LEARNER, engine).p("o_one") == pytest.approx(second, abs=1e-9)


def test_apply_response_rejects_a_payload_with_no_objective(engine):
    with pytest.raises(ValueError, match="objective_id"):
        incremental.apply_response(LEARNER, {"correct": True}, engine=engine)


def test_incremental_is_faster_than_rebuilding_per_response(capsys):
    """The debt was worth clearing only if the fast path is actually fast."""
    script = _script(99, 120)

    fast = build_engine("sqlite:///:memory:")
    create_all(fast)
    started = time.perf_counter()
    for index, (objective_id, correct, difficulty, _mc) in enumerate(script):
        incremental.apply_response(
            LEARNER,
            graded(objective_id, correct, item_id=f"item_{index}", difficulty=difficulty),
            engine=fast,
            occurred_at=START + timedelta(minutes=index),
        )
    fast_seconds = time.perf_counter() - started

    slow = build_engine("sqlite:///:memory:")
    create_all(slow)
    log = EventLog(slow)
    started = time.perf_counter()
    for index, (objective_id, correct, difficulty, _mc) in enumerate(script):
        log.append(
            EventKind.RESPONSE_GRADED,
            LEARNER,
            graded(objective_id, correct, item_id=f"item_{index}", difficulty=difficulty),
            occurred_at=START + timedelta(minutes=index),
        )
        rebuild(LEARNER, slow)  # what run_diagnostic does today
    slow_seconds = time.perf_counter() - started

    try:
        fast_p = profile_for(LEARNER, fast).as_dict()
        slow_p = profile_for(LEARNER, slow).as_dict()
        assert fast_p.keys() == slow_p.keys()
        for objective_id, p in slow_p.items():
            assert fast_p[objective_id] == pytest.approx(p, abs=1e-9)
        print(
            f"\n120 responses: incremental {fast_seconds:.3f}s, "
            f"rebuild-per-item {slow_seconds:.3f}s, "
            f"speedup {slow_seconds / fast_seconds:.1f}x"
        )
        assert fast_seconds * 2 < slow_seconds
    finally:
        fast.dispose()
        slow.dispose()


# -- reachability ----------------------------------------------------------


def test_the_gate_and_the_fast_path_are_used_by_the_study_loop():
    """This project's recurring failure is code that is built and never wired.

    The study loop is owned by other workers in this phase. When their modules
    land, they must reach these two. If they are not here yet the test says so
    out loud rather than passing quietly.
    """
    root = Path(__file__).resolve().parents[1] / "src" / "the_oracle"
    candidates = sorted(root.glob("session/*.py")) + sorted(root.glob("commands/study.py"))
    if not candidates:
        pytest.skip("study loop modules not landed yet; orchestrator must wire them")
    source = "\n".join(path.read_text() for path in candidates)
    assert "mastery.gate" in source or "from the_oracle.mastery import gate" in source, (
        "the study loop must use the STRICT gate"
    )
    if "def run_session" not in source:
        pytest.skip(
            "session runner not landed yet; it must call mastery.incremental.apply_response"
        )
    assert "incremental" in source, "the study loop must update mastery incrementally"
