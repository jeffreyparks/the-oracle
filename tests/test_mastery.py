"""Mastery tests.

The BKT reducer is the heart of Phase 1, so the two claims that matter most -
the update is correct, and replay is pure - are proved here, not asserted in a
comment.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session, select

from the_oracle.domains.schema import Domain, Edge, Module, ObjectiveRef
from the_oracle.mastery import bkt, reducers
from the_oracle.mastery.bkt import MASTERY_THRESHOLD, BKTParams, posterior, update
from the_oracle.mastery.profile import MasteryProfile, profile_for
from the_oracle.mastery.scheduler import Review, next_review_at, plan
from the_oracle.mastery.synthetic import SyntheticLearner, recovery_error
from the_oracle.store import models
from the_oracle.store.db import build_engine, create_all
from the_oracle.store.events import EventKind, EventLog
from the_oracle.store.rebuild import REGISTRY, ReplayState, drop_derived, rebuild

LEARNER = "learner_mastery"
OTHER = "learner_other"
START = datetime(2025, 1, 1, 9, 0, tzinfo=UTC)


@pytest.fixture
def engine():
    eng = build_engine("sqlite:///:memory:")
    create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def log(engine):
    return EventLog(engine)


def graded(
    objective_id: str,
    correct: bool,
    *,
    item_id: str = "item_1",
    difficulty: int = 2,
    bloom: str = "understand",
    seconds: float = 12.5,
    misconception_id: str | None = None,
) -> dict:
    """A RESPONSE_GRADED payload exactly as the contract defines it."""
    return {
        "objective_id": objective_id,
        "item_id": item_id,
        "correct": correct,
        "difficulty": difficulty,
        "bloom": bloom,
        "seconds": seconds,
        "misconception_id": misconception_id,
    }


def write_stream(log: EventLog, learner: str, outcomes: list[tuple[str, bool]]) -> None:
    """Append one graded event per outcome, at fixed increasing times."""
    for i, (objective_id, correct) in enumerate(outcomes):
        log.append(
            EventKind.RESPONSE_GRADED,
            learner,
            graded(objective_id, correct, item_id=f"item_{i}"),
            occurred_at=START + timedelta(minutes=i),
        )


def mastery_rows(engine, learner: str = LEARNER) -> dict[str, models.MasteryState]:
    with Session(engine) as session:
        rows = session.exec(
            select(models.MasteryState).where(models.MasteryState.learner_id == learner)
        ).all()
    return {row.objective_id: row for row in rows}


def snapshot(engine, learner: str = LEARNER) -> list[tuple]:
    """A comparable, order-free view of derived mastery."""
    return sorted(
        (
            oid,
            round(row.p_mastery, 12),
            round(row.confidence, 12),
            row.observations,
            tuple(row.misconception_ids or ()),
            row.last_seen_at,
            row.next_review_at,
        )
        for oid, row in mastery_rows(engine, learner).items()
    )


# -- the update ------------------------------------------------------------


def test_posterior_matches_the_hand_computed_bayes_update():
    params = BKTParams(p_init=0.2, p_transit=0.15, p_slip=0.10, p_guess=0.20)
    p = 0.2
    # correct: (0.2 * 0.9) / (0.2 * 0.9 + 0.8 * 0.2) = 0.18 / 0.34
    conditional = 0.18 / 0.34
    expected = conditional + (1 - conditional) * 0.15
    assert posterior(p, True, params) == pytest.approx(expected)

    # wrong: (0.2 * 0.1) / (0.2 * 0.1 + 0.8 * 0.8) = 0.02 / 0.66
    conditional = 0.02 / 0.66
    expected = conditional + (1 - conditional) * 0.15
    assert posterior(p, False, params) == pytest.approx(expected)


def test_update_defaults_to_the_contract_parameters():
    assert update(0.2, True) == pytest.approx(posterior(0.2, True, BKTParams()))
    assert update(0.2, False) == pytest.approx(posterior(0.2, False, BKTParams()))


def test_a_correct_answer_raises_the_belief_and_a_wrong_one_lowers_it():
    params = BKTParams(p_transit=0.0)
    for p in (0.05, 0.2, 0.5, 0.8, 0.95):
        assert update(p, True, params) > p
        assert update(p, False, params) < p


def test_repeated_correct_answers_converge_upward_past_threshold():
    p = BKTParams().p_init
    path = []
    for _ in range(12):
        p = update(p, True)
        path.append(p)
    assert path == sorted(path)          # monotonic
    assert path[-1] > MASTERY_THRESHOLD  # and it gets there
    assert path[-1] < 1.0


def test_repeated_wrong_answers_converge_downward():
    p = 0.9
    path = []
    for _ in range(12):
        p = update(p, False, BKTParams(p_transit=0.0))
        path.append(p)
    assert path == sorted(path, reverse=True)
    assert path[-1] < 0.05
    assert path[-1] > 0.0


def test_probability_stays_strictly_inside_zero_and_one():
    params = BKTParams(p_transit=0.0)
    for observations in ([True] * 500, [False] * 500, [True, False] * 250):
        p = params.p_init
        for correct in observations:
            p = update(p, correct, params)
            assert 0.0 < p < 1.0


def test_the_belief_is_monotonic_in_the_prior():
    params = BKTParams()
    for correct in (True, False):
        values = [update(p / 20, correct, params) for p in range(1, 20)]
        assert values == sorted(values)


def test_learning_rate_lifts_the_result():
    slow = BKTParams(p_transit=0.0)
    fast = BKTParams(p_transit=0.5)
    assert update(0.3, False, fast) > update(0.3, False, slow)


def test_bad_parameters_are_rejected():
    with pytest.raises(ValueError):
        BKTParams(p_slip=1.5)
    with pytest.raises(ValueError):
        BKTParams(p_slip=0.6, p_guess=0.6)  # correct answer would be bad news


def test_confidence_grows_with_evidence_and_stays_bounded():
    values = [bkt.confidence(n) for n in range(0, 40)]
    assert values[0] == 0.0
    assert values == sorted(values)
    assert max(values) < 1.0


# -- replay ----------------------------------------------------------------


def test_rebuild_derives_mastery_from_the_log_alone(engine, log):
    write_stream(log, LEARNER, [("o_a", True)] * 10 + [("o_b", False)] * 6)

    assert mastery_rows(engine) == {}
    report = rebuild(LEARNER, engine)

    rows = mastery_rows(engine)
    assert set(rows) == {"o_a", "o_b"}
    assert rows["o_a"].p_mastery > MASTERY_THRESHOLD
    assert rows["o_b"].p_mastery < 0.2
    assert rows["o_a"].observations == 10
    assert rows["o_b"].observations == 6
    assert report.events_read == 16
    assert report.rows_written >= 16


def test_replay_is_pure_same_log_same_mastery(engine, log):
    write_stream(
        log,
        LEARNER,
        [("o_a", True), ("o_b", False), ("o_a", False), ("o_a", True), ("o_b", True)] * 3,
    )

    rebuild(LEARNER, engine)
    first = snapshot(engine)
    rebuild(LEARNER, engine)
    second = snapshot(engine)
    rebuild(LEARNER, engine)
    third = snapshot(engine)

    assert first == second == third
    assert first  # and it is not trivially empty


def test_replay_is_pure_without_a_database(log):
    """The reducers alone are a deterministic fold over the event stream."""
    write_stream(log, LEARNER, [("o_a", True), ("o_a", False), ("o_b", True)])
    events = log.read(LEARNER)

    def fold() -> list[tuple[str, float, int]]:
        state = ReplayState(learner_id=LEARNER)
        for event in events:
            REGISTRY.apply(state, event)
        return sorted(
            (oid, acc.p, acc.observations)
            for oid, acc in state.scratch[reducers.SCRATCH_KEY].items()
        )

    assert fold() == fold() == fold()


def test_rebuild_reproduces_mastery_after_dropping_derived_tables(engine, log):
    write_stream(log, LEARNER, [("o_a", True), ("o_a", True), ("o_b", False), ("o_c", True)])
    rebuild(LEARNER, engine)
    before = snapshot(engine)

    cleared = drop_derived(LEARNER, engine)
    assert "mastery_state" in cleared
    assert mastery_rows(engine) == {}
    assert log.count(LEARNER) == 4  # the log survived

    rebuild(LEARNER, engine)
    assert snapshot(engine) == before


def test_replay_order_matters_but_the_result_does_not_drift(engine, log):
    write_stream(log, LEARNER, [("o_a", True), ("o_a", False)])
    rebuild(LEARNER, engine)
    a_then_b = mastery_rows(engine)["o_a"].p_mastery

    write_stream(log, OTHER, [("o_a", False), ("o_a", True)])
    rebuild(OTHER, engine)
    b_then_a = mastery_rows(engine, OTHER)["o_a"].p_mastery

    assert a_then_b != pytest.approx(b_then_a)  # order is real information
    rebuild(LEARNER, engine)
    assert mastery_rows(engine)["o_a"].p_mastery == pytest.approx(a_then_b)


def test_rebuild_touches_only_one_learner(engine, log):
    write_stream(log, LEARNER, [("o_a", True)] * 5)
    write_stream(log, OTHER, [("o_a", False)] * 5)
    rebuild(LEARNER, engine)
    rebuild(OTHER, engine)

    mine = mastery_rows(engine)["o_a"].p_mastery
    rebuild(OTHER, engine)
    assert mastery_rows(engine)["o_a"].p_mastery == pytest.approx(mine)
    assert mastery_rows(engine, OTHER)["o_a"].p_mastery < mine


def test_responses_are_derived_from_the_log(engine, log):
    log.append(
        EventKind.RESPONSE_GRADED,
        LEARNER,
        graded("o_a", True, seconds=4.25, misconception_id="mc_x"),
        occurred_at=START,
    )
    rebuild(LEARNER, engine)
    with Session(engine) as session:
        responses = session.exec(
            select(models.Response).where(models.Response.learner_id == LEARNER)
        ).all()
    assert len(responses) == 1
    assert responses[0].correct is True
    assert responses[0].score == 1.0
    assert responses[0].latency_ms == 4250
    assert responses[0].misconception_ids == ["mc_x"]


def test_misconceptions_are_collected_and_deduplicated(engine, log):
    log.append(EventKind.RESPONSE_GRADED, LEARNER, graded("o_a", False, misconception_id="mc_x"),
               occurred_at=START)
    log.append(EventKind.RESPONSE_GRADED, LEARNER, graded("o_a", False, misconception_id="mc_x"),
               occurred_at=START + timedelta(minutes=1))
    log.append(EventKind.MISCONCEPTION_DETECTED, LEARNER,
               {"objective_id": "o_a", "misconception_id": "mc_y"},
               occurred_at=START + timedelta(minutes=2))
    rebuild(LEARNER, engine)

    row = mastery_rows(engine)["o_a"]
    assert row.misconception_ids == ["mc_x", "mc_y"]
    assert row.observations == 2  # the tag event is not a new opportunity


def test_a_bare_misconception_event_does_not_move_mastery(engine, log):
    log.append(EventKind.MISCONCEPTION_DETECTED, LEARNER,
               {"objective_id": "o_a", "misconception_id": "mc_x"}, occurred_at=START)
    rebuild(LEARNER, engine)
    row = mastery_rows(engine)["o_a"]
    assert row.p_mastery == pytest.approx(BKTParams().p_init)
    assert row.observations == 0


def test_malformed_payloads_are_ignored_not_fatal(engine, log):
    log.append(EventKind.RESPONSE_GRADED, LEARNER, {}, occurred_at=START)
    log.append(EventKind.RESPONSE_GRADED, LEARNER, {"objective_id": ""},
               occurred_at=START + timedelta(minutes=1))
    log.append(EventKind.RESPONSE_GRADED, LEARNER,
               {"objective_id": "o_a", "correct": True, "seconds": "slow"},
               occurred_at=START + timedelta(minutes=2))
    rebuild(LEARNER, engine)
    rows = mastery_rows(engine)
    assert set(rows) == {"o_a"}
    assert rows["o_a"].observations == 1


def test_derived_timestamps_come_from_the_event_not_the_clock(engine, log):
    write_stream(log, LEARNER, [("o_a", True), ("o_a", True)])
    rebuild(LEARNER, engine)
    row = mastery_rows(engine)["o_a"]
    assert row.last_seen_at.replace(tzinfo=UTC) == START + timedelta(minutes=1)
    assert row.updated_at.replace(tzinfo=UTC) == START + timedelta(minutes=1)


# -- profile ---------------------------------------------------------------


def test_profile_for_reads_mastery_state(engine, log):
    write_stream(log, LEARNER, [("o_a", True)] * 8 + [("o_b", False)] * 4)
    rebuild(LEARNER, engine)

    profile = profile_for(LEARNER, engine)
    assert profile.learner_id == LEARNER
    assert profile.mastered() == {"o_a"}
    assert profile.p("o_a") > MASTERY_THRESHOLD
    assert profile.p("o_b") < 0.2
    assert profile.observations("o_a") == 8
    assert profile.as_dict() == {"o_a": profile.p("o_a"), "o_b": profile.p("o_b")}
    assert list(profile.as_dict()) == ["o_a", "o_b"]  # sorted, stable output


def test_an_unseen_objective_answers_with_the_prior_not_zero():
    profile = MasteryProfile({"o_a": 0.9})
    assert profile.p("o_never_seen") == pytest.approx(BKTParams().p_init)
    assert "o_never_seen" not in profile
    assert profile.mastered() == {"o_a"}


def test_empty_profile_for_a_learner_with_no_events(engine):
    profile = profile_for("nobody", engine)
    assert profile.as_dict() == {}
    assert profile.mastered() == set()


def _domain() -> Domain:
    return Domain(
        id="d_test",
        version=1,
        title="Test manifest",
        objectives=[ObjectiveRef(id=o, version=1) for o in ("o_one", "o_two", "o_three")],
        edges=[Edge(from_id="o_one", to_id="o_two"), Edge(from_id="o_two", to_id="o_three")],
        modules=[Module(id="m1", title="M", goal="g", objectives=["o_one", "o_two", "o_three"])],
    )


def test_known_prerequisites_met_follows_the_manifest():
    domain = _domain()
    profile = MasteryProfile({"o_one": 0.95, "o_two": 0.4, "o_three": 0.1})

    assert profile.known_prerequisites_met(domain, "o_one") is True   # no prerequisites
    assert profile.known_prerequisites_met(domain, "o_two") is True   # o_one mastered
    assert profile.known_prerequisites_met(domain, "o_three") is False  # o_two is not

    assert profile.ready(domain) == ["o_two"]


def test_prerequisites_use_the_mastery_threshold_exactly():
    domain = _domain()
    just_under = MasteryProfile({"o_one": MASTERY_THRESHOLD - 1e-9})
    just_over = MasteryProfile({"o_one": MASTERY_THRESHOLD})
    assert just_under.known_prerequisites_met(domain, "o_two") is False
    assert just_over.known_prerequisites_met(domain, "o_two") is True


# -- scheduler -------------------------------------------------------------


def _reviews(pattern: list[bool]) -> list[Review]:
    return [
        Review(correct=c, at=START + timedelta(days=i), difficulty=2)
        for i, c in enumerate(pattern)
    ]


def test_no_history_means_no_schedule():
    assert plan([], 0.5) is None
    assert next_review_at([], 0.5) is None


def test_intervals_grow_with_a_correct_streak():
    intervals = [plan(_reviews([True] * n), 0.9).interval_days for n in range(1, 6)]
    assert intervals == sorted(intervals)
    assert intervals[0] < intervals[-1]


def test_a_lapse_collapses_the_interval_and_counts():
    strong = plan(_reviews([True] * 5), 0.9)
    lapsed = plan(_reviews([True] * 5 + [False]), 0.9)
    assert lapsed.interval_days < strong.interval_days
    assert lapsed.lapses == 1
    assert strong.lapses == 0


def test_shaky_mastery_earns_a_shorter_interval_than_strong_mastery():
    shaky = plan(_reviews([True] * 4), 0.3)
    strong = plan(_reviews([True] * 4), 0.95)
    assert shaky.interval_days < strong.interval_days


def test_the_schedule_is_deterministic_and_bounded():
    history = _reviews([True, False, True, True, True])
    first, second = plan(history, 0.8), plan(history, 0.8)
    assert first == second
    assert 0.5 <= first.interval_days <= 365.0
    assert first.due_at == history[-1].at + timedelta(days=first.interval_days)


def test_rebuild_writes_the_review_schedule(engine, log):
    write_stream(log, LEARNER, [("o_a", True), ("o_a", True), ("o_a", True)])
    rebuild(LEARNER, engine)
    with Session(engine) as session:
        schedules = session.exec(
            select(models.ReviewSchedule).where(models.ReviewSchedule.learner_id == LEARNER)
        ).all()
    assert len(schedules) == 1
    assert schedules[0].objective_id == "o_a"
    assert schedules[0].interval_days > 1.0
    assert mastery_rows(engine)["o_a"].next_review_at is not None


# -- synthetic learners ----------------------------------------------------


def test_a_synthetic_learner_is_reproducible_from_its_seed():
    a = SyntheticLearner({"o_a": 0.7}, seed=11)
    b = SyntheticLearner({"o_a": 0.7}, seed=11)
    first = a.answers("o_a", 50)
    assert first == b.answers("o_a", 50)

    a.reset()  # rewinds the stream to the seed
    assert a.answers("o_a", 50) == first
    assert len(a.asked) == 50

    different = SyntheticLearner({"o_a": 0.7}, seed=12)
    assert different.answers("o_a", 50) != first


def test_a_synthetic_learner_hits_its_own_true_accuracy():
    learner = SyntheticLearner({"o_a": 0.9, "o_b": 0.1}, slip=0.1, guess=0.2, seed=5)
    high = learner.answers("o_a", 2000)
    low = learner.answers("o_b", 2000)
    # P(correct) = m * (1 - slip) + (1 - m) * guess
    assert sum(high) / len(high) == pytest.approx(0.9 * 0.9 + 0.1 * 0.2, abs=0.03)
    assert sum(low) / len(low) == pytest.approx(0.1 * 0.9 + 0.9 * 0.2, abs=0.03)


def test_harder_items_lower_the_odds_of_knowing():
    learner = SyntheticLearner({"o_a": 0.5}, seed=1)
    easy = learner.effective_mastery("o_a", 1)
    neutral = learner.effective_mastery("o_a", 3)
    hard = learner.effective_mastery("o_a", 5)
    assert easy > neutral > hard
    assert neutral == pytest.approx(0.5)


def test_recovery_error_is_mean_absolute_error():
    assert recovery_error({"a": 1.0, "b": 0.0}, {"a": 0.8, "b": 0.4}) == pytest.approx(0.3)
    assert recovery_error({}, {}) == 0.0
    # an objective the estimator never saw is scored against the prior
    assert recovery_error({"a": 0.2}, {}) == pytest.approx(0.0)
