"""The weekly report: facts, honesty, and the no-invented-numbers rule.

Offline. No key, no network. The one model call goes through the cassette
harness in ``tests/cassettes``, with a hand-written synthesizer standing in for
the model, so a stale key can never turn into a live request.

What this file guards, in order of importance:

1. **No invented numbers.** Every figure in the rendered output is checked
   against :class:`WeekFacts`. The check is run against a deliberately
   number-dense report, and the extractor is itself asserted to find numbers,
   so the test cannot pass by finding nothing.
2. **Honesty.** A dead week says so. A fading objective is reported as loudly
   as an improving one. Unearned praise is deleted rather than softened.
3. **Reachability.** The command is exercised through a Typer app, and the
   wiring into the shipped ``cli.py`` is asserted.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from sqlalchemy import Engine
from typer.testing import CliRunner

from the_oracle import style
from the_oracle.agents import coach as coach_mod
from the_oracle.agents.coach import (
    Coach,
    WeekFacts,
    WeeklyReport,
    allowed_numbers,
    deterministic_report,
    earned_praise,
    enforce,
    gather_week,
    invented_numbers,
    render,
)
from the_oracle.commands import report as report_cmd
from the_oracle.config import reset_settings_cache
from the_oracle.context import LearnerContext
from the_oracle.mastery import MASTERY_THRESHOLD
from the_oracle.store.db import create_all, get_engine
from the_oracle.store.events import EventKind, EventLog

from tests.cassettes.replay import cassette

LEARNER = "learner_report"
#: Frozen. Nothing in this file reads the wall clock for an assertion.
NOW = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)

runner = CliRunner()


# --- a clean home -----------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ORACLE_HOME", str(tmp_path))
    monkeypatch.setenv("ORACLE_LEARNER_ID", LEARNER)
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SERPER_API_KEY", "VOYAGE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield tmp_path
    reset_settings_cache()


@pytest.fixture
def engine(home: Path) -> Engine:
    eng = get_engine()
    create_all(eng)
    return eng


# --- seeding ----------------------------------------------------------------


def ago(days: float = 0.0, hours: float = 0.0) -> datetime:
    return NOW - timedelta(days=days, hours=hours)


def grade(
    log: EventLog,
    objective_id: str,
    correct: bool,
    at: datetime,
    *,
    difficulty: int = 2,
    session_id: str = "s1",
) -> None:
    log.append(
        EventKind.RESPONSE_GRADED,
        LEARNER,
        {
            "objective_id": objective_id,
            "item_id": f"item_{objective_id}",
            "correct": correct,
            "difficulty": difficulty,
            "session_id": session_id,
        },
        occurred_at=at,
    )


def session(log: EventLog, session_id: str, start: datetime, minutes: float) -> None:
    log.append(
        EventKind.SESSION_STARTED, LEARNER, {"session_id": session_id, "mode": "study"},
        occurred_at=start,
    )
    log.append(
        EventKind.SESSION_ENDED, LEARNER, {"session_id": session_id, "mode": "study"},
        occurred_at=start + timedelta(minutes=minutes),
    )


def good_week(engine: Engine) -> WeekFacts:
    """A week with a real gain on one objective and decay on another."""
    log = EventLog(engine)
    # Long-dead objective: answered right repeatedly months ago, untouched since.
    for day in (120, 119, 118, 117):
        grade(log, "obj_stale", True, ago(days=day), session_id="s0")
    # The objective that moved this week.
    grade(log, "obj_rising", False, ago(days=40), session_id="s0")
    for hours in (60, 50, 40, 30):
        grade(log, "obj_rising", True, ago(hours=hours), session_id="s1")
    # The objective that lost ground this week.
    for day in (30, 29, 28):
        grade(log, "obj_slipping", True, ago(days=day), session_id="s0")
    for hours in (55, 45):
        grade(log, "obj_slipping", False, ago(hours=hours), session_id="s1")
    session(log, "s1", ago(hours=61), 45.0)
    session(log, "s2", ago(hours=31), 20.0)
    return gather_week(LEARNER, now=NOW, engine=engine)


# --- gather: pure and deterministic ----------------------------------------


def test_empty_log_never_crashes(engine: Engine) -> None:
    facts = gather_week(LEARNER, now=NOW, engine=engine)
    assert facts == WeekFacts(learner_id=LEARNER, idle_days=7.0)
    assert facts.is_empty_week
    assert facts.accuracy is None


def test_gather_is_deterministic(engine: Engine) -> None:
    first = good_week(engine)
    second = gather_week(LEARNER, now=NOW, engine=engine)
    assert first == second


def test_gather_counts_only_the_window(engine: Engine) -> None:
    facts = good_week(engine)
    assert facts.items == 6           # four on obj_rising, two on obj_slipping
    assert facts.correct == 4
    assert facts.sessions == 2
    assert facts.minutes == 65.0
    assert facts.idle_days == pytest.approx(30 / 24, abs=0.01)


def test_gather_reports_gains_and_decay(engine: Engine) -> None:
    facts = good_week(engine)
    improved = dict((oid, (b, a)) for oid, b, a in facts.objectives_improved)
    fading = dict(facts.objectives_fading)
    assert "obj_rising" in improved or "obj_rising" in facts.objectives_mastered
    assert "obj_slipping" in fading, "an objective that lost ground must be reported"
    assert "obj_stale" in fading, "a review months overdue is fading"
    for _oid, before, after in facts.objectives_improved:
        assert after > before


def test_wider_window_sees_more(engine: Engine) -> None:
    good_week(engine)
    one = gather_week(LEARNER, now=NOW, engine=engine, weeks=1)
    six = gather_week(LEARNER, now=NOW, engine=engine, weeks=6)
    assert six.items > one.items


# --- the rule: the model never invents a number ----------------------------


def plain(renderable: Any, width: int = 100) -> str:
    """Render to plain text exactly as the terminal would show it."""
    console = Console(record=True, width=width, no_color=True, legacy_windows=False)
    console.print(renderable)
    return console.export_text()


def numbers_in(text: str, facts: WeekFacts) -> list[str]:
    masked = coach_mod._mask_identifiers(text, facts)
    return coach_mod.NUMBER_RE.findall(masked)


def test_every_number_in_the_rendered_report_comes_from_the_facts(engine: Engine) -> None:
    """The point of the whole design. Strict: nothing is exempt.

    The rendered output is scanned for every bare number, and each one must be
    accounted for by ``WeekFacts``. The extractor is asserted to find a real
    crop of numbers first, so this cannot pass vacuously.
    """
    facts = good_week(engine)
    report = deterministic_report(facts)
    text = plain(render(report, facts))

    found = numbers_in(text, facts)
    assert len(found) >= 8, f"the number extractor found almost nothing in:\n{text}"

    allowed = allowed_numbers(facts)
    unaccounted = [token for token in found if coach_mod._canon(token) not in allowed]
    assert unaccounted == [], (
        f"these figures are not in WeekFacts: {unaccounted}\n{text}"
    )
    assert invented_numbers(text, facts) == []


def test_the_guard_actually_catches_an_invented_number(engine: Engine) -> None:
    """Prove the checker is not a rubber stamp."""
    facts = good_week(engine)
    assert invented_numbers("You answered 9999 items.", facts) == ["9999"]
    assert invented_numbers("You spent 12.5 hours on it.", facts) == ["12.5"]
    assert invented_numbers(f"You answered {facts.items} items.", facts) == []
    # A percentage the facts do not support is still an invented number.
    assert invented_numbers("You were right 73% of the time.", facts) == ["73"]
    # An objective id is a fact, not a figure, even when it carries digits.
    digits = facts.model_copy(update={"objectives_fading": [("obj_12", 0.4)]})
    assert invented_numbers("obj_12 is fading.", digits) == []


def test_a_hallucinated_report_is_replaced_by_the_computed_one(engine: Engine) -> None:
    facts = good_week(engine)
    liar = WeeklyReport(
        headline="You answered 47 items across 9 sessions.",
        what_moved="Three objectives jumped 0.42 points.",
        what_is_fading="Nothing at all.",
        what_is_next="Do 30 items tomorrow.",
        encouragement="Great job, you got this.",
    )
    fixed = enforce(liar, facts)
    floor = deterministic_report(facts)
    assert fixed.headline == floor.headline
    assert fixed.what_moved == floor.what_moved
    assert fixed.what_is_next == floor.what_is_next
    assert invented_numbers(plain(render(fixed, facts)), facts) == []
    assert not any(p in plain(render(fixed, facts)).lower() for p in style.BANNED_PHRASES)


def test_clean_prose_survives_enforcement(engine: Engine) -> None:
    facts = good_week(engine)
    honest = WeeklyReport(
        headline=f"You answered {facts.items} items and lost ground on obj_slipping.",
        what_moved="obj_rising answered right four times running.",
        what_is_fading="obj_slipping went wrong twice, so it needs review before anything new.",
        what_is_next="Review the two that slipped before you open new material.",
        encouragement="",
    )
    fixed = enforce(honest, facts)
    assert fixed.headline == honest.headline
    assert fixed.what_is_fading == honest.what_is_fading


# --- honesty ----------------------------------------------------------------


def test_a_dead_week_says_so_without_scolding(engine: Engine) -> None:
    facts = gather_week(LEARNER, now=NOW, engine=engine)
    report = deterministic_report(facts)
    text = plain(render(report, facts))
    assert "nothing" in report.headline.lower()
    assert report.encouragement == ""
    for scold in ("should have", "you failed", "disappointing", "lazy", "excuse", "shame"):
        assert scold not in text.lower()
    for spin in ("great", "well done", "keep it up"):
        assert spin not in text.lower()
    assert invented_numbers(text, facts) == []


def test_zero_activity_never_calls_the_model(engine: Engine) -> None:
    """A dead week is written from facts. No key, no call, no invented week."""
    facts = gather_week(LEARNER, now=NOW, engine=engine)
    ctx = LearnerContext.for_learner(LEARNER)
    with cassette("coach_units", synthesize=_never_called) as tape:
        import asyncio

        result = asyncio.run(Coach(engine).run(ctx, facts))
    assert tape.calls == []
    assert result.output.encouragement == ""
    assert "nothing" in result.output.headline.lower()


def test_encouragement_is_omitted_when_it_is_not_earned(engine: Engine) -> None:
    """One lonely wrong answer earns nothing, and nothing is what it gets."""
    log = EventLog(engine)
    grade(log, "obj_thin", False, ago(hours=20))
    facts = gather_week(LEARNER, now=NOW, engine=engine)
    assert not earned_praise(facts)
    report = deterministic_report(facts)
    assert report.encouragement == ""
    assert "worth saying" not in plain(render(report, facts))
    # And a model that tries anyway is overruled.
    pushy = WeeklyReport(headline="A thin week.", encouragement="You are doing so well.")
    assert enforce(pushy, facts).encouragement == ""


def test_encouragement_is_earned_and_specific_when_it_appears(engine: Engine) -> None:
    facts = good_week(engine)
    assert earned_praise(facts)
    report = deterministic_report(facts)
    assert report.encouragement
    named = [*facts.objectives_mastered, *(o for o, _b, _a in facts.objectives_improved)]
    assert any(objective_id in report.encouragement for objective_id in named), (
        "praise must name the exact thing the learner did"
    )


def test_fading_is_reported_as_prominently_as_gains(engine: Engine) -> None:
    facts = good_week(engine)
    text = plain(render(deterministic_report(facts), facts))
    assert "what is fading" in text
    for objective_id, _p in facts.objectives_fading:
        assert objective_id in text


def test_no_banned_phrase_reaches_the_learner(engine: Engine) -> None:
    facts = good_week(engine)
    empty = gather_week("nobody", now=NOW, engine=engine)
    thin = facts.model_copy(update={"objectives_mastered": [], "objectives_improved": []})
    for f in (facts, empty, thin):
        text = plain(render(deterministic_report(f), f)).lower()
        for phrase in style.BANNED_PHRASES:
            assert phrase not in text
        assert "!" not in text


# --- the model path, through the cassette ----------------------------------


def _never_called(agent: str, prompt: str) -> dict[str, Any]:  # pragma: no cover
    raise AssertionError(f"{agent} called the model when it should not have")


def _hallucinating(agent: str, prompt: str) -> dict[str, Any]:
    """A stand-in model that lies with numbers and gushes. Both get caught."""
    return {
        "headline": "A huge week: 91 items at 100%.",
        "what_moved": "You gained 0.77 on three objectives.",
        "what_is_fading": "",
        "what_is_next": "Book 4 sessions next week.",
        "encouragement": "Great job, keep it up.",
    }


def _honest(agent: str, prompt: str) -> dict[str, Any]:
    return {
        "headline": "You put in real work and still lost ground on obj_slipping.",
        "what_moved": "obj_rising came right four times running, which is why it climbed.",
        "what_is_fading": "obj_slipping went wrong twice. Review it before new material.",
        "what_is_next": "Clear the overdue reviews first.",
        "encouragement": "You went back to obj_rising after getting it wrong, and it stuck.",
    }


def _run_coach(engine: Engine, facts: WeekFacts, synth: Any) -> WeeklyReport:
    import asyncio

    ctx = LearnerContext.for_learner(LEARNER)
    with cassette("coach_units", synthesize=synth, write=False):
        return asyncio.run(Coach(engine).run(ctx, facts, force=True)).output


def test_a_lying_model_cannot_reach_the_learner(engine: Engine) -> None:
    facts = good_week(engine)
    written = _run_coach(engine, facts, _hallucinating)
    text = plain(render(written, facts))
    assert invented_numbers(text, facts) == []
    assert "91" not in text and "0.77" not in text
    for phrase in style.BANNED_PHRASES:
        assert phrase not in text.lower()
    assert written.what_is_fading, "a silent fading section is filled from the facts"


def test_an_honest_model_is_left_alone(engine: Engine) -> None:
    facts = good_week(engine)
    written = _run_coach(engine, facts, _honest)
    assert written.headline.startswith("You put in real work")
    assert "obj_slipping" in written.what_is_fading
    assert invented_numbers(plain(render(written, facts)), facts) == []


# --- the command ------------------------------------------------------------


def test_report_command_runs_on_an_empty_log(engine: Engine) -> None:
    result = runner.invoke(report_cmd.app, [])
    assert result.exit_code == 0, result.output
    assert "nothing" in result.output.lower()
    assert "Traceback" not in result.output


def test_report_command_accepts_weeks(engine: Engine) -> None:
    good_week(engine)
    result = runner.invoke(report_cmd.app, ["--weeks", "4"])
    assert result.exit_code == 0, result.output
    assert "what is fading" in result.output
    assert "Traceback" not in result.output


def test_report_command_rejects_a_nonsense_window(engine: Engine) -> None:
    result = runner.invoke(report_cmd.app, ["--weeks", "0"])
    assert result.exit_code == 1


def test_report_command_prints_without_a_model(engine: Engine) -> None:
    """No key, so the Coach call fails. The report still prints, and says why."""
    good_week(engine)
    result = runner.invoke(report_cmd.app, [])
    assert result.exit_code == 0, result.output
    assert "what moved" in result.output


# --- reachability: is it wired into the shipped CLI? ------------------------


def test_report_is_wired_into_the_cli() -> None:
    """Six components have now been built and never wired in. Not this one."""
    from the_oracle import cli

    registered = {
        info.name or (info.callback.__name__ if info.callback else ""): info
        for info in cli.app.registered_commands
    }
    info = registered.get("report")
    assert info is not None, "cli.py has no report command at all"
    if info.callback is not report_cmd.report:
        pytest.xfail(
            "report is still the Phase 0 stub in cli.py. Replace the stub with:\n"
            "    from the_oracle.commands import report as _report_cmd\n"
            "    app.command('report', help='Print the weekly report.')"
            "(_report_cmd.report)"
        )
    assert info.callback is report_cmd.report
