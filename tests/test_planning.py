"""Planner tests: order, honesty of totals, and stability.

The planner is pure, so everything here runs offline with no API key and no
model. A plan that changes between two identical runs is a bug, and the last
test in this file is the one that proves it does not.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from the_oracle.domains.registry import load_domain
from the_oracle.mastery import MASTERY_THRESHOLD
from the_oracle.mastery.profile import MasteryProfile, profile_for
from the_oracle.planning import (
    REVIEW_FACTOR,
    Syllabus,
    build_checkpoints,
    build_syllabus,
    capacity,
    classify,
    finish_date,
    planned_minutes,
)
from the_oracle.store.db import build_engine, create_all

DOMAIN_ID = "planner_domain"
LEARNER = "learner_planner"

# A tiny, subject-neutral pack. Three modules and a prerequisite chain that
# crosses module boundaries, so teaching order has real work to do.
OBJECTIVES: list[dict] = [
    {"id": "obj_a1", "est_minutes": 30, "difficulty": 1},
    {"id": "obj_a2", "est_minutes": 20, "difficulty": 3},
    {"id": "obj_b1", "est_minutes": 45, "difficulty": 2},
    {"id": "obj_b2", "est_minutes": 25, "difficulty": 5},
    {"id": "obj_b3", "est_minutes": 15, "difficulty": 4},
    {"id": "obj_c1", "est_minutes": 60, "difficulty": 3},
    {"id": "obj_c2", "est_minutes": 10, "difficulty": 2},
    {"id": "obj_c3", "est_minutes": 40, "difficulty": 5},
    {"id": "obj_c4", "est_minutes": 35, "difficulty": 4},
    {"id": "obj_c5", "est_minutes": 5, "difficulty": 1},
    {"id": "obj_c6", "est_minutes": 50, "difficulty": 3},
]
MODULES: list[tuple[str, list[str]]] = [
    ("mod_a", ["obj_a1", "obj_a2"]),
    ("mod_b", ["obj_b1", "obj_b3", "obj_b2"]),
    ("mod_c", ["obj_c1", "obj_c2", "obj_c3", "obj_c4", "obj_c5", "obj_c6"]),
]
EDGES: list[tuple[str, str]] = [
    ("obj_a1", "obj_a2"),
    ("obj_a2", "obj_b1"),
    ("obj_b1", "obj_b3"),
    ("obj_b3", "obj_b2"),
    ("obj_b2", "obj_c1"),
]
EST = {o["id"]: o["est_minutes"] for o in OBJECTIVES}


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the synthetic pack and point ORACLE_HOME at it."""
    objectives_dir = tmp_path / "objectives"
    objectives_dir.mkdir()
    for spec in OBJECTIVES:
        body = {
            "id": spec["id"],
            "version": 1,
            "title": f"Skill {spec['id']}",
            "description": f"A neutral placeholder skill named {spec['id']}.",
            "bloom": "apply",
            "difficulty": spec["difficulty"],
            "est_minutes": spec["est_minutes"],
            "assessment_stems": [f"Do the thing for {spec['id']}."],
            "tags": ["placeholder"],
        }
        (objectives_dir / f"{spec['id']}.yaml").write_text(
            yaml.safe_dump(body, sort_keys=False), encoding="utf-8"
        )

    manifest = {
        "id": DOMAIN_ID,
        "version": 1,
        "title": "Planner Test Domain",
        "description": "Synthetic pack used only by the planner tests.",
        "objectives": [{"id": o["id"], "version": 1} for o in OBJECTIVES],
        "edges": [{"from": a, "to": b} for a, b in EDGES],
        "modules": [
            {"id": mid, "title": mid.upper(), "goal": "goal", "objectives": list(oids)}
            for mid, oids in MODULES
        ],
        "misconceptions": [],
    }
    domains_dir = tmp_path / "domains"
    domains_dir.mkdir()
    (domains_dir / f"{DOMAIN_ID}.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("ORACLE_HOME", str(tmp_path))
    return tmp_path


def plan_with(scores: dict[str, float], **kwargs) -> Syllabus:
    """Build a syllabus from an explicit mastery state, no database involved."""
    profile = MasteryProfile(scores, learner_id=LEARNER)
    return build_syllabus(DOMAIN_ID, LEARNER, profile=profile, **kwargs)


# --- ordering --------------------------------------------------------------


def test_teaching_order_is_never_violated(home: Path) -> None:
    syllabus = plan_with({})
    position = {item.objective_id: i for i, item in enumerate(syllabus.items)}
    for source, target in EDGES:
        assert position[source] < position[target], (source, target)


def test_plan_order_matches_the_domain_order(home: Path) -> None:
    domain = load_domain(DOMAIN_ID)
    syllabus = plan_with({})
    assert [i.objective_id for i in syllabus.items] == domain.teaching_order()
    assert [i.objective_id for i in syllabus.for_module("mod_b")] == [
        "obj_b1", "obj_b3", "obj_b2",
    ]


def test_order_survives_skipping_a_mastered_objective(home: Path) -> None:
    syllabus = plan_with({"obj_b3": 0.99})
    position = {item.objective_id: i for i, item in enumerate(syllabus.items)}
    assert "obj_b3" not in position
    assert position["obj_b1"] < position["obj_b2"] < position["obj_c1"]


# --- statuses --------------------------------------------------------------


def test_mastered_objective_is_skipped_not_planned(home: Path) -> None:
    syllabus = plan_with({"obj_a1": 0.95, "obj_c5": MASTERY_THRESHOLD})
    assert syllabus.skipped_mastered == ["obj_a1", "obj_c5"]
    planned = {i.objective_id for i in syllabus.items}
    assert planned.isdisjoint({"obj_a1", "obj_c5"})
    assert all(i.status in {"learn", "review"} for i in syllabus.items)


def test_partial_mastery_is_review_with_reduced_time(home: Path) -> None:
    syllabus = plan_with({"obj_c1": 0.6})
    item = next(i for i in syllabus.items if i.objective_id == "obj_c1")
    assert item.status == "review"
    assert item.est_minutes == round(EST["obj_c1"] * REVIEW_FACTOR)
    assert 0 < item.est_minutes < EST["obj_c1"]


def test_status_boundaries_are_exact(home: Path) -> None:
    assert classify(MASTERY_THRESHOLD) == "mastered"
    assert classify(MASTERY_THRESHOLD - 0.0001) == "review"
    assert classify(0.5) == "review"
    assert classify(0.4999) == "learn"
    assert planned_minutes(1, "review") == 1  # never free, never zero


def test_below_the_review_floor_costs_full_time(home: Path) -> None:
    syllabus = plan_with({"obj_c1": 0.49})
    item = next(i for i in syllabus.items if i.objective_id == "obj_c1")
    assert item.status == "learn"
    assert item.est_minutes == EST["obj_c1"]


# --- totals ----------------------------------------------------------------


def test_totals_add_up_exactly(home: Path) -> None:
    syllabus = plan_with({"obj_a1": 0.9, "obj_b1": 0.7})
    assert syllabus.total_minutes == sum(i.est_minutes for i in syllabus.items)

    expected = sum(EST.values()) - EST["obj_a1"]
    expected -= EST["obj_b1"] - round(EST["obj_b1"] * REVIEW_FACTOR)
    assert syllabus.total_minutes == expected
    assert sum(syllabus.module_minutes(m) for m in syllabus.module_ids()) == expected
    assert syllabus.counts() == {"learn": 9, "review": 1, "mastered": 1}


def test_weeks_scale_with_hours_per_week(home: Path) -> None:
    slow = plan_with({}, hours_per_week=1.5)
    fast = plan_with({}, hours_per_week=3.0)
    assert fast.total_minutes == slow.total_minutes
    assert slow.weeks == pytest.approx(fast.weeks * 2.0)
    assert fast.weeks == pytest.approx(sum(EST.values()) / 60.0 / 3.0)
    with pytest.raises(ValueError):
        plan_with({}, hours_per_week=0.0)


def test_finish_date_rounds_partial_weeks_up(home: Path) -> None:
    start = date(2025, 1, 1)
    assert finish_date(1.0, start) == date(2025, 1, 8)
    assert finish_date(1.01, start) == date(2025, 1, 9)


# --- no data is not an error ----------------------------------------------


def test_learner_with_no_mastery_data_gets_the_whole_domain(home: Path) -> None:
    engine = build_engine("sqlite:///:memory:")
    create_all(engine)
    try:
        profile = profile_for("nobody", engine)
        assert len(profile) == 0
        syllabus = build_syllabus(DOMAIN_ID, "nobody", engine=engine)
    finally:
        engine.dispose()

    assert len(syllabus.items) == len(OBJECTIVES)
    assert syllabus.skipped_mastered == []
    assert syllabus.total_minutes == sum(EST.values())
    assert all(i.status == "learn" for i in syllabus.items)


def test_everything_mastered_leaves_an_empty_plan(home: Path) -> None:
    syllabus = plan_with(dict.fromkeys(EST, 0.99))
    assert syllabus.items == []
    assert syllabus.total_minutes == 0
    assert syllabus.weeks == 0.0
    assert len(syllabus.skipped_mastered) == len(OBJECTIVES)


# --- checkpoints -----------------------------------------------------------


def test_a_checkpoint_lands_after_every_module(home: Path) -> None:
    domain = load_domain(DOMAIN_ID)
    syllabus = plan_with({})
    checkpoints = build_checkpoints(syllabus, domain=domain)
    assert [c.module_id for c in checkpoints] == [mid for mid, _ in MODULES]
    for checkpoint in checkpoints:
        last_of_module = max(
            i for i, item in enumerate(syllabus.items)
            if item.module_id == checkpoint.module_id
        )
        assert checkpoint.after_item_index == last_of_module + 1


def test_a_checkpoint_only_draws_from_its_own_module(home: Path) -> None:
    domain = load_domain(DOMAIN_ID)
    syllabus = plan_with({})
    by_module = {mid: set(oids) for mid, oids in MODULES}
    for checkpoint in build_checkpoints(syllabus, domain=domain):
        assert checkpoint.objective_ids
        assert set(checkpoint.objective_ids) <= by_module[checkpoint.module_id]
        assert len(set(checkpoint.objective_ids)) == len(checkpoint.objective_ids)


def test_a_checkpoint_fits_one_session(home: Path) -> None:
    domain = load_domain(DOMAIN_ID)
    syllabus = plan_with({})
    for minutes in (10, 25, 50):
        for checkpoint in build_checkpoints(syllabus, domain=domain, session_minutes=minutes):
            assert checkpoint.est_minutes <= minutes
            assert checkpoint.item_count <= capacity(minutes)
    # module c has six objectives but a 25 minute session holds four items
    big = next(c for c in build_checkpoints(syllabus, domain=domain) if c.module_id == "mod_c")
    assert big.item_count == capacity(25) == 4


def test_a_fully_mastered_module_gets_no_checkpoint(home: Path) -> None:
    domain = load_domain(DOMAIN_ID)
    syllabus = plan_with({oid: 0.99 for oid in by_module_ids("mod_a")})
    checkpoints = build_checkpoints(syllabus, domain=domain)
    assert [c.module_id for c in checkpoints] == ["mod_b", "mod_c"]


def by_module_ids(module_id: str) -> list[str]:
    return next(oids for mid, oids in MODULES if mid == module_id)


# --- stability -------------------------------------------------------------


def test_the_syllabus_is_stable_across_identical_runs(home: Path) -> None:
    """Same mastery state in, same syllabus out. Twice.

    A plan that shifts under the learner destroys trust, so this is asserted
    on the whole structure, not on the total alone.
    """
    scores = {"obj_a1": 0.9, "obj_b2": 0.62, "obj_c3": 0.51, "obj_c5": 0.2}
    domain = load_domain(DOMAIN_ID)

    first = plan_with(scores)
    second = plan_with(scores)

    assert first == second
    assert first.items == second.items
    assert first.skipped_mastered == second.skipped_mastered
    assert (first.total_minutes, first.weeks) == (second.total_minutes, second.weeks)
    assert build_checkpoints(first, domain=domain) == build_checkpoints(second, domain=domain)


def test_the_plan_does_not_depend_on_mastery_insertion_order(home: Path) -> None:
    forward = {"obj_a1": 0.9, "obj_b2": 0.62, "obj_c3": 0.51}
    backward = dict(reversed(list(forward.items())))
    assert plan_with(forward) == plan_with(backward)
