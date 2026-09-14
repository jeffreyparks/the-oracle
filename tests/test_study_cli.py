"""The study loop's learner-facing surface: ``study`` and ``review``.

Everything here is offline. No key, no network, no model, and — deliberately —
no dependency on the real ``the_oracle.session`` or on a live scheduler. The
command reaches the session engine through two seams, ``study._session`` and
``study._gate``, which these tests replace with fakes built only from the
frozen Phase 4 contract.

What this file is really guarding:

1. **Reachability.** Both commands are exercised through the shipped CLI app,
   not only through their own module. This project has now shipped five
   components that were unit-tested and never wired in.
2. **Honesty.** An empty corpus stops the session and names the command to run.
   Nothing due says nothing due. The close-out reports the measured numbers.
3. **Voice.** No banned phrase from ``style.py`` reaches the learner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
import yaml
from sqlmodel import Session
from typer.testing import CliRunner

from the_oracle import style
from the_oracle.commands import review as review_cmd
from the_oracle.commands import study as study_cmd
from the_oracle.config import reset_settings_cache
from the_oracle.corpus import store as corpus_store
from the_oracle.store import models
from the_oracle.store.db import create_all, get_engine

DOMAIN_ID = "study_domain"
OTHER_DOMAIN_ID = "study_other"
MODULE_ID = "mod_one"
LEARNER = "learner_study"
OBJECTIVES = ["obj_one", "obj_two"]

NOW = datetime.now(timezone.utc)


# --- a subject-neutral pack and a clean home --------------------------------


def _objective(oid: str) -> dict[str, Any]:
    return {
        "id": oid,
        "version": 1,
        "title": f"Skill {oid}",
        "description": f"A neutral placeholder skill named {oid}.",
        "bloom": "apply",
        "difficulty": 2,
        "est_minutes": 30,
        "assessment_stems": [f"Do the thing for {oid}."],
        "tags": ["placeholder"],
    }


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    objectives_dir = tmp_path / "objectives"
    objectives_dir.mkdir()
    for oid in OBJECTIVES:
        (objectives_dir / f"{oid}.yaml").write_text(
            yaml.safe_dump(_objective(oid), sort_keys=False), encoding="utf-8"
        )
    domains_dir = tmp_path / "domains"
    domains_dir.mkdir()
    (domains_dir / f"{DOMAIN_ID}.yaml").write_text(
        yaml.safe_dump(
            {
                "id": DOMAIN_ID,
                "version": 1,
                "title": "Study Test Domain",
                "description": "Synthetic pack used only by the study tests.",
                "objectives": [{"id": o, "version": 1} for o in OBJECTIVES],
                "edges": [{"from": OBJECTIVES[0], "to": OBJECTIVES[1]}],
                "modules": [
                    {
                        "id": MODULE_ID,
                        "title": "Module One",
                        "goal": "goal",
                        "objectives": list(OBJECTIVES),
                    }
                ],
                "misconceptions": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ORACLE_HOME", str(tmp_path))
    monkeypatch.setenv("ORACLE_LEARNER_ID", LEARNER)
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SERPER_API_KEY", "VOYAGE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield tmp_path
    reset_settings_cache()


@pytest.fixture
def engine(home: Path):
    eng = get_engine()
    create_all(eng)
    return eng


# --- fakes for the two seams ------------------------------------------------


@dataclass(frozen=True)
class FakeBudget:
    phase: str
    minutes: float


@dataclass(frozen=True)
class FakePlan:
    """Same shape as ``session.SessionPlan``; ``study`` calls ``replace`` on it."""

    learner_id: str
    domain_id: str
    minutes: int
    warmup: list[str] = field(default_factory=list)
    focus: list[str] = field(default_factory=list)
    shape: tuple[FakeBudget, ...] = ()


@dataclass(frozen=True)
class FakeResult:
    session_id: str = "sess_test"
    asked: int = 0
    correct: int = 0
    objectives_touched: list[str] = field(default_factory=list)
    mastery_before: dict[str, float] = field(default_factory=dict)
    mastery_after: dict[str, float] = field(default_factory=dict)
    ended_early: bool = False
    reason: str = ""


def _shape(minutes: int) -> tuple[FakeBudget, ...]:
    scale = minutes / 25.0
    return (
        FakeBudget("warmup", 4 * scale),
        FakeBudget("new", 14 * scale),
        FakeBudget("practice", 5 * scale),
        FakeBudget("consolidate", 2 * scale),
    )


class FakeSession:
    """The session engine, reduced to the frozen contract."""

    def __init__(
        self,
        *,
        focus: list[str] | None = None,
        warmup: list[str] | None = None,
        result: FakeResult | None = None,
        script: Any = None,
    ) -> None:
        self._focus = list(focus if focus is not None else [OBJECTIVES[0]])
        self._warmup = list(warmup or [])
        self._result = result or FakeResult()
        self._script = script
        self.plan_calls: list[dict[str, Any]] = []
        self.run_calls: list[Any] = []

    def plan_session(
        self,
        domain_id: str,
        learner_id: str,
        *,
        minutes: int = 25,
        now: datetime | None = None,
        engine: Any = None,
    ) -> FakePlan:
        self.plan_calls.append({"domain_id": domain_id, "minutes": minutes})
        return FakePlan(
            learner_id=learner_id,
            domain_id=domain_id,
            minutes=minutes,
            warmup=list(self._warmup),
            focus=list(self._focus),
            shape=_shape(minutes),
        )

    def run_session(
        self, plan: Any, *, answer_fn: Any = None, on_event: Any = None, engine: Any = None
    ) -> FakeResult:
        self.run_calls.append(plan)
        if self._script is not None:
            self._script(answer_fn, on_event)
        return self._result

    @staticmethod
    def should_interrupt(seen: Any, new_id: str | None) -> bool:
        return bool(new_id) and new_id in tuple(seen)


class FakeGate:
    @staticmethod
    def blocked_by(domain: Any, objective_id: str, profile: Any, **_: Any) -> list[str]:
        return list(domain.prerequisites(objective_id))


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch):
    """Install a fake session engine behind the command's seams."""

    def install(session: FakeSession) -> FakeSession:
        monkeypatch.setattr(study_cmd, "_session", lambda: session)
        monkeypatch.setattr(study_cmd, "_gate", lambda: FakeGate())
        return session

    return install


# --- the shipped path -------------------------------------------------------


def _registered(app: typer.Typer) -> dict[str, Any]:
    return {
        info.name or (info.callback.__name__ if info.callback else ""): info.callback
        for info in app.registered_commands
    }


def _standalone_app() -> typer.Typer:
    """Wire the two commands exactly the way ``cli.py`` must.

    A Typer app holding one command collapses into that command, so the real
    CLI keeps several. This mirrors the shipped wiring line for line.
    """
    app = typer.Typer()
    app.command("study", help="Run a study session.")(study_cmd.study)
    app.command("review", help="Work through everything that is due.")(review_cmd.review)
    return app


def _shipped_cli() -> typer.Typer:
    from the_oracle.cli import app

    return app


def _wired_into_cli() -> bool:
    registered = _registered(_shipped_cli())
    return (
        registered.get("study") is study_cmd.study
        and registered.get("review") is review_cmd.review
    )


def _cli_app() -> typer.Typer:
    """Run every behaviour test through the real CLI once it is wired.

    Until the orchestrator wires ``cli.py``, these tests run through an app
    built with the identical registration calls, so the day the wiring lands
    nothing about these tests changes.
    """
    return _shipped_cli() if _wired_into_cli() else _standalone_app()


def test_study_and_review_are_wired_into_the_real_cli() -> None:
    """Reachability, asserted on the shipped app and not on a stand-in.

    This project has shipped five components that were unit-tested and never
    wired in. ``cli.py`` belongs to the orchestrator, so until it lands this
    reports as an expected failure rather than a red suite - but it reports.
    """
    registered = _registered(_shipped_cli())
    stub = registered.get("study")
    if stub is not None and stub.__module__ == "the_oracle.cli":
        pytest.xfail(
            "cli.py still ships the Phase 4 stub. Wire it with: "
            'app.command("study")(_study_cmd.study)' " and "
            'app.command("review")(_review_cmd.review)'
        )
    assert "study" in registered, "cli.py does not register a study command"
    assert "review" in registered, "cli.py does not register a review command"
    assert registered["study"] is study_cmd.study, (
        "cli.py registers a study command, but not commands/study.py"
    )
    assert registered["review"] is review_cmd.review, (
        "cli.py registers a review command, but not commands/review.py"
    )


def _invoke(args: list[str], stdin: str | None = None):
    return CliRunner().invoke(_cli_app(), args, input=stdin)


# --- helpers ----------------------------------------------------------------


def _accept(
    objective_id: str,
    engine: Any,
    *,
    title: str = "A Written Introduction",
    slug: str = "a",
) -> str:
    resource_id = corpus_store.put_resource(
        objective_id,
        url=f"https://example.test/{objective_id}/{slug}",
        title=title,
        engine=engine,
    )
    corpus_store.record_review(
        resource_id,
        SimpleNamespace(verdict="accept", score=0.9, rubric={}, notes="fits the objective"),
        engine,
    )
    return resource_id


def _mastery(engine: Any, objective_id: str, p: float, due_at: datetime, observations: int = 4) -> None:
    with Session(engine) as db:
        db.add(
            models.MasteryState(
                learner_id=LEARNER,
                objective_id=objective_id,
                p_mastery=p,
                confidence=0.6,
                observations=observations,
                last_seen_at=NOW - timedelta(days=2),
                next_review_at=due_at,
            )
        )
        db.commit()


def assert_voice(text: str) -> None:
    """No banned phrase, no emoji, no exclamation-mark enthusiasm."""
    lowered = text.lower()
    for phrase in style.BANNED_PHRASES:
        assert phrase not in lowered, f"banned phrase in output: {phrase!r}"
    assert "!" not in text, "exclamation marks are banned by style.py"


# --- study ------------------------------------------------------------------


def test_study_with_an_empty_corpus_tells_the_learner_what_to_run(engine, wire) -> None:
    """No material means no teaching. It names the exact command instead."""
    session = wire(FakeSession(focus=[OBJECTIVES[0]]))
    result = _invoke(["study", DOMAIN_ID])

    assert result.exit_code == 1, result.output
    assert "no accepted material" in result.output.lower()
    assert f"the-oracle resources {DOMAIN_ID} --module {MODULE_ID}" in " ".join(result.output.split())
    assert session.run_calls == [], "it must not teach without material"
    assert_voice(result.output)


def test_study_shows_the_phase_and_why_it_exists(engine, wire) -> None:
    """A learner is told why last week's work comes before new work."""
    _accept(OBJECTIVES[0], engine)
    wire(FakeSession(focus=[OBJECTIVES[0]], warmup=[OBJECTIVES[1]]))
    result = _invoke(["study", DOMAIN_ID])

    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split()).lower()
    for phase in ("warmup", "new", "practice", "consolidate"):
        assert phase in flat
    assert "retrieving something is what makes it stick" in flat
    assert "You start with 1 item from earlier work: obj_two." in " ".join(result.output.split())
    assert_voice(result.output)


def test_study_shows_the_first_accepted_resource(engine, wire) -> None:
    _accept(OBJECTIVES[0], engine, title="First Accepted Source", slug="first")
    _accept(OBJECTIVES[0], engine, title="Second Accepted Source", slug="second")

    def script(answer_fn: Any, on_event: Any) -> None:
        on_event(
            SimpleNamespace(
                kind="item_presented", payload={"phase": "new", "objective_id": OBJECTIVES[0]}
            )
        )

    wire(FakeSession(focus=[OBJECTIVES[0]], script=script))
    result = _invoke(["study", DOMAIN_ID])

    flat = " ".join(result.output.split())
    assert "First Accepted Source" in flat
    assert "Second Accepted Source" not in flat
    assert_voice(result.output)


def test_minutes_is_honoured(engine, wire) -> None:
    """``--minutes`` reaches the planner and shows up in the rendered shape."""
    _accept(OBJECTIVES[0], engine)
    session = wire(FakeSession(focus=[OBJECTIVES[0]]))
    result = _invoke(["study", DOMAIN_ID, "--minutes", "50"])

    assert result.exit_code == 0, result.output
    assert session.plan_calls == [{"domain_id": DOMAIN_ID, "minutes": 50}]
    assert "50 min" in " ".join(result.output.split())
    assert_voice(result.output)


def test_multi_step_attempt_is_never_interrupted(engine, wire) -> None:
    """The critique lands after the attempt, and the learner is told so first."""
    _accept(OBJECTIVES[0], engine)

    def script(answer_fn: Any, on_event: Any) -> None:
        item = SimpleNamespace(
            stem="Derive the result in full.",
            kind="multi_step",
            difficulty=3,
            objective_id=OBJECTIVES[0],
            phase="practice",
        )
        answer = answer_fn(item)
        assert answer == "my derivation"
        on_event(
            SimpleNamespace(
                kind="response_graded",
                payload={
                    "objective_id": OBJECTIVES[0],
                    "correct": False,
                    "feedback": "You dropped the constant at the third line.",
                },
            )
        )

    wire(FakeSession(focus=[OBJECTIVES[0]], script=script))
    result = _invoke(["study", DOMAIN_ID], stdin="my derivation\n")

    flat = " ".join(result.output.split())
    assert "hold the critique until you submit" in flat.lower()
    assert "wrong" in flat.lower()
    assert flat.lower().index("hold the critique") < flat.lower().index("you dropped the constant")
    assert_voice(result.output)


def test_same_misconception_twice_interrupts(engine, wire) -> None:
    _accept(OBJECTIVES[0], engine)

    def script(answer_fn: Any, on_event: Any) -> None:
        for _ in range(2):
            on_event(
                SimpleNamespace(
                    kind="response_graded",
                    payload={
                        "objective_id": OBJECTIVES[0],
                        "correct": False,
                        "feedback": "Wrong model.",
                        "misconception_id": "mis_swap",
                    },
                )
            )

    wire(FakeSession(focus=[OBJECTIVES[0]], script=script))
    result = _invoke(["study", DOMAIN_ID])

    flat = " ".join(result.output.split()).lower()
    assert "second time this session" in flat
    assert_voice(result.output)


def test_closeout_reports_the_real_numbers(engine, wire) -> None:
    """Honest arithmetic: 4 of 6, the measured delta, and a real return date."""
    _accept(OBJECTIVES[0], engine)
    due_at = NOW + timedelta(days=3)
    _mastery(engine, OBJECTIVES[0], 0.62, due_at)

    result_obj = FakeResult(
        asked=6,
        correct=4,
        objectives_touched=[OBJECTIVES[0]],
        mastery_before={OBJECTIVES[0]: 0.40},
        mastery_after={OBJECTIVES[0]: 0.62},
    )
    wire(FakeSession(focus=[OBJECTIVES[0]], result=result_obj))
    result = _invoke(["study", DOMAIN_ID])

    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "You answered 4 of 6 right. That is 67%." in flat
    assert "0.40" in flat and "0.62" in flat and "+0.22" in flat
    assert "Come back" in flat
    assert due_at.strftime("%Y-%m-%d") in flat
    assert_voice(result.output)


def test_closeout_does_not_flatter_a_session_that_moved_nothing(engine, wire) -> None:
    _accept(OBJECTIVES[0], engine)
    wire(
        FakeSession(
            focus=[OBJECTIVES[0]],
            result=FakeResult(
                asked=4,
                correct=1,
                mastery_before={OBJECTIVES[0]: 0.30},
                mastery_after={OBJECTIVES[0]: 0.30},
            ),
        )
    )
    result = _invoke(["study", DOMAIN_ID])

    flat = " ".join(result.output.split())
    assert "You answered 1 of 4 right. That is 25%." in flat
    assert "No objective moved enough to report" in flat
    assert_voice(result.output)


def test_study_ended_early_says_why(engine, wire) -> None:
    _accept(OBJECTIVES[0], engine)
    wire(
        FakeSession(
            focus=[OBJECTIVES[0]],
            result=FakeResult(
                asked=8,
                correct=3,
                ended_early=True,
                reason="Your accuracy dropped well below your own baseline, so stopping here.",
            ),
        )
    )
    result = _invoke(["study", DOMAIN_ID])
    flat = " ".join(result.output.split())
    assert "ended early" in flat.lower()
    assert "below your own baseline" in flat
    assert_voice(result.output)


def test_study_with_nothing_teachable_says_what_is_blocking(engine, wire) -> None:
    wire(FakeSession(focus=[]))
    result = _invoke(["study", DOMAIN_ID])

    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "Nothing here is teachable right now." in flat
    assert f"the-oracle assess {DOMAIN_ID}" in flat
    assert_voice(result.output)


# --- review -----------------------------------------------------------------


def test_review_with_nothing_scheduled_says_so_and_exits_zero(engine) -> None:
    result = _invoke(["review"])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "Nothing is due" in flat
    assert_voice(result.output)


def test_review_with_nothing_due_names_the_next_date(engine) -> None:
    later = NOW + timedelta(days=4)
    _mastery(engine, OBJECTIVES[0], 0.7, later)

    result = _invoke(["review"])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "Nothing is due right now." in flat
    assert OBJECTIVES[0] in flat
    assert later.strftime("%Y-%m-%d") in flat
    assert_voice(result.output)


def test_review_lists_everything_due_with_its_domain(engine) -> None:
    _mastery(engine, OBJECTIVES[0], 0.55, NOW - timedelta(days=2))
    _mastery(engine, OBJECTIVES[1], 0.61, NOW - timedelta(hours=5))

    result = _invoke(["review"])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "2 objectives due." in flat
    assert DOMAIN_ID in flat
    assert "0.55" in flat and "0.61" in flat
    assert f"the-oracle study {DOMAIN_ID}" in flat
    assert_voice(result.output)


def test_review_invents_no_work(engine) -> None:
    """Nothing due means no table, no suggestions, no busywork."""
    result = _invoke(["review"])
    flat = " ".join(result.output.split()).lower()
    assert "due now" not in flat
    assert "objectives due" not in flat and "objective due" not in flat


# --- voice ------------------------------------------------------------------


def test_no_banned_phrase_reaches_the_learner(engine, wire) -> None:
    """Every surface this module owns, checked against style.BANNED_PHRASES."""
    _accept(OBJECTIVES[0], engine)
    _mastery(engine, OBJECTIVES[0], 0.5, NOW - timedelta(days=1))
    wire(
        FakeSession(
            focus=[OBJECTIVES[0]],
            warmup=[OBJECTIVES[1]],
            result=FakeResult(
                asked=5,
                correct=2,
                mastery_before={OBJECTIVES[0]: 0.30},
                mastery_after={OBJECTIVES[0]: 0.50},
            ),
        )
    )
    outputs = [
        _invoke(["study", DOMAIN_ID]).output,
        _invoke(["study", DOMAIN_ID, "--minutes", "10"]).output,
        _invoke(["review"]).output,
    ]
    for text in outputs:
        assert_voice(text)

    source = (
        Path(study_cmd.__file__).read_text(encoding="utf-8")
        + Path(review_cmd.__file__).read_text(encoding="utf-8")
    )
    for phrase in style.BANNED_PHRASES:
        assert phrase not in source.lower(), f"banned phrase in source: {phrase!r}"


class FakeSessionWithFeedback(FakeSession):
    """The real engine also offers ``on_feedback``; the command must prefer it."""

    def run_session(  # type: ignore[override]
        self,
        plan: Any,
        *,
        answer_fn: Any = None,
        on_event: Any = None,
        engine: Any = None,
        on_feedback: Any = None,
    ) -> FakeResult:
        self.run_calls.append(plan)
        if self._script is not None:
            self._script(answer_fn, on_event, on_feedback)
        return self._result


def test_grader_prose_is_shown_through_on_feedback(engine, wire) -> None:
    """The verdict carries the grader's words, not just correct or wrong."""

    def script(answer_fn: Any, on_event: Any, on_feedback: Any) -> None:
        item = SimpleNamespace(
            stem="State it.", kind="recall", difficulty=1, objective_id=OBJECTIVES[0], phase="new"
        )
        grade = SimpleNamespace(
            correct=False,
            feedback="You used the sample mean where the prior mean belongs.",
            misconception_id=None,
        )
        on_feedback(item, "an answer", grade, "new")
        # The event stream must not repeat a verdict that on_feedback reported.
        on_event(
            SimpleNamespace(
                kind="response_graded",
                payload={"objective_id": OBJECTIVES[0], "correct": False, "phase": "new"},
            )
        )

    _accept(OBJECTIVES[0], engine)
    wire(FakeSessionWithFeedback(focus=[OBJECTIVES[0]], script=script))
    result = _invoke(["study", DOMAIN_ID])

    flat = " ".join(result.output.split())
    assert "You used the sample mean where the prior mean belongs." in flat
    assert flat.lower().count("wrong —") == 1, "the verdict is reported once, not twice"
    assert_voice(result.output)


def test_the_command_calls_the_real_session_and_gate_signatures() -> None:
    """The fakes above are only honest if they match the shipped engine.

    This binds the exact call ``study`` makes against the real signatures, so a
    drift in ``the_oracle.session`` or ``mastery.gate`` fails here rather than
    the first time a learner runs the command.
    """
    import inspect

    session = pytest.importorskip("the_oracle.session")
    gate = pytest.importorskip("the_oracle.mastery.gate")

    inspect.signature(session.plan_session).bind(
        DOMAIN_ID, LEARNER, minutes=25, now=NOW, engine=None
    )
    run = inspect.signature(session.run_session)
    run.bind(object(), answer_fn=print, on_event=print, engine=None)
    assert "on_feedback" in run.parameters, (
        "study.py prefers on_feedback for grader prose; the session dropped it"
    )
    run.bind(object(), answer_fn=print, on_event=print, engine=None, on_feedback=print)
    inspect.signature(gate.blocked_by).bind(object(), "obj_one", object())
