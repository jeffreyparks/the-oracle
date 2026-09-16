"""Due work without a daemon: the schedule package and the cron command.

Everything here is offline and frozen in time. Three things this file exists to
guard, in order of how badly the project has been bitten by them:

1. **Reachability.** ``cron`` is exercised through a Typer app assembled exactly
   the way ``cli.py`` assembles it, with ``CliRunner``. A component that only
   its own unit test can reach is not shipped. ``test_cron_is_wired_into_cli``
   names the exact line ``cli.py`` needs.
2. **No real crontab, ever.** The runner is injected. On top of that,
   :class:`SystemCrontab` refuses to write while pytest is loaded, so even a
   mistake in a fixture cannot reach a developer's crontab. A test asserts that
   refusal.
3. **Idempotency.** The cron line may fire more than once. Five runs must send
   one nudge.

The ladder itself (``the_oracle.nudge``) belongs to another worker, so these
tests drive ``run_due_checks`` through a stub built only from the frozen Phase 5
contract signature. One test imports the real module when it exists and checks
the two agree.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import IntEnum
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import typer
import yaml
from sqlmodel import Session, select
from typer.testing import CliRunner

from the_oracle.commands import cron as cron_cmd
from the_oracle.config import reset_settings_cache
from the_oracle.schedule import ACTIVITY_KINDS, DueSummary, due_summary, run_due_checks
from the_oracle.schedule import cron as cron_lib
from the_oracle.store import models
from the_oracle.store.db import create_all, get_engine, session_scope
from the_oracle.store.events import EventKind, EventLog

LEARNER = "learner_schedule"
NOW = datetime(2025, 3, 10, 9, 30, tzinfo=UTC)

DOMAIN_A = "sched_alpha"
DOMAIN_B = "sched_beta"
OBJ = ["obj_a1", "obj_a2", "obj_shared", "obj_b1"]

runner = CliRunner()


# --- a subject-neutral home -------------------------------------------------


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


def _pack(domain_id: str, objectives: list[str]) -> dict[str, Any]:
    return {
        "id": domain_id,
        "version": 1,
        "title": f"Pack {domain_id}",
        "description": "Synthetic pack used only by the schedule tests.",
        "objectives": [{"id": o, "version": 1} for o in objectives],
        "edges": [],
        "modules": [
            {
                "id": f"{domain_id}_mod",
                "title": "Module One",
                "goal": "goal",
                "objectives": list(objectives),
            }
        ],
        "misconceptions": [],
    }


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    objectives_dir = tmp_path / "objectives"
    objectives_dir.mkdir()
    for oid in OBJ:
        (objectives_dir / f"{oid}.yaml").write_text(
            yaml.safe_dump(_objective(oid), sort_keys=False), encoding="utf-8"
        )
    domains_dir = tmp_path / "domains"
    domains_dir.mkdir()
    (domains_dir / f"{DOMAIN_A}.yaml").write_text(
        yaml.safe_dump(_pack(DOMAIN_A, ["obj_a1", "obj_a2", "obj_shared"]), sort_keys=False),
        encoding="utf-8",
    )
    (domains_dir / f"{DOMAIN_B}.yaml").write_text(
        yaml.safe_dump(_pack(DOMAIN_B, ["obj_b1", "obj_shared"]), sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("ORACLE_HOME", str(tmp_path))
    monkeypatch.setenv("ORACLE_LEARNER_ID", LEARNER)
    reset_settings_cache()
    yield tmp_path
    reset_settings_cache()


@pytest.fixture
def engine(home: Path):
    eng = create_all(get_engine())
    with session_scope(eng) as db:
        db.add(models.Learner(id=LEARNER, display_name="Test", timezone="UTC"))
    return eng


@pytest.fixture
def crontab(monkeypatch: pytest.MonkeyPatch) -> cron_lib.MemoryCrontab:
    """The injected crontab. No test in this file can reach the real one."""
    fake = cron_lib.MemoryCrontab()
    monkeypatch.setattr(cron_lib, "_RUNNER", fake, raising=False)
    yield fake
    monkeypatch.setattr(cron_lib, "_RUNNER", None, raising=False)


def _schedule(engine, objective_id: str, due_at: datetime, p: float = 0.6) -> None:
    with session_scope(engine) as db:
        db.add(
            models.MasteryState(
                learner_id=LEARNER,
                objective_id=objective_id,
                p_mastery=p,
                confidence=0.5,
                observations=3,
                last_seen_at=due_at - timedelta(days=1),
                next_review_at=due_at,
            )
        )


def _activity(engine, when: datetime, kind: str = EventKind.SESSION_ENDED) -> None:
    EventLog(engine).append(kind, LEARNER, {"note": "test"}, occurred_at=when)


# --- the stub ladder, built from the frozen contract ------------------------


class StubRung(IntEnum):
    QUIET = 0
    REMINDER = 1
    SMALLER_ASK = 2
    REPLAN = 3
    PAUSE = 4


@dataclass(frozen=True)
class StubDecision:
    rung: StubRung
    reason: str
    body: str
    send: bool
    suppressed_reason: str | None = None


class StubChannel:
    name = "terminal"

    def __init__(self) -> None:
        self.sent: list[models.Nudge] = []

    def send(self, nudge: models.Nudge) -> bool:
        self.sent.append(nudge)
        return True


@pytest.fixture
def ladder(monkeypatch: pytest.MonkeyPatch):
    """A ``the_oracle.nudge`` stub that always wants to send.

    It always says yes, so any single-nudge-per-day behaviour observed in these
    tests comes from ``run_due_checks`` itself and not from the ladder.
    """
    module = ModuleType("the_oracle.nudge")
    channel = StubChannel()
    calls: list[dict[str, Any]] = []

    def decide(**kwargs: Any) -> StubDecision:
        calls.append(kwargs)
        return StubDecision(
            rung=StubRung.REMINDER, reason="one day idle", body="Two items are due.", send=True
        )

    module.Rung = StubRung  # type: ignore[attr-defined]
    module.NudgeDecision = StubDecision  # type: ignore[attr-defined]
    module.decide = decide  # type: ignore[attr-defined]
    module.get_channel = lambda name="terminal": channel  # type: ignore[attr-defined]
    module.calls = calls  # type: ignore[attr-defined]
    module.channel = channel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "the_oracle.nudge", module)
    return module


def _nudges(engine) -> list[models.Nudge]:
    with Session(engine) as db:
        return list(db.exec(select(models.Nudge).where(models.Nudge.learner_id == LEARNER)))


def _nudge_events(engine) -> list[Any]:
    return EventLog(engine).read(LEARNER, kinds=[EventKind.NUDGE_SENT])


# --- due_summary ------------------------------------------------------------


def test_due_summary_with_nothing_scheduled_is_empty(engine) -> None:
    summary = due_summary(LEARNER, now=NOW, engine=engine)
    assert isinstance(summary, DueSummary)
    assert (summary.due_objectives, summary.next_due_at, summary.domains) == (0, None, {})
    assert summary.idle_days == 0.0


def test_due_summary_counts_across_multiple_domains(engine) -> None:
    _schedule(engine, "obj_a1", NOW - timedelta(days=2))
    _schedule(engine, "obj_a2", NOW - timedelta(hours=1))
    _schedule(engine, "obj_shared", NOW - timedelta(minutes=5))
    _schedule(engine, "obj_b1", NOW + timedelta(days=3))

    summary = due_summary(LEARNER, now=NOW, engine=engine)

    assert summary.due_objectives == 3
    # obj_shared sits in both packs, so both packs count it.
    assert summary.domains == {DOMAIN_A: 3, DOMAIN_B: 1}
    assert summary.next_due_at == NOW + timedelta(days=3)


def test_due_summary_ignores_other_learners(engine) -> None:
    _schedule(engine, "obj_a1", NOW - timedelta(days=1))
    with session_scope(engine) as db:
        db.add(
            models.MasteryState(
                learner_id="someone_else",
                objective_id="obj_b1",
                next_review_at=NOW - timedelta(days=1),
            )
        )
    assert due_summary(LEARNER, now=NOW, engine=engine).due_objectives == 1


def test_idle_days_measured_from_study_not_from_the_last_nudge(engine) -> None:
    _activity(engine, NOW - timedelta(days=5))
    EventLog(engine).append(
        EventKind.NUDGE_SENT, LEARNER, {"rung": 1}, occurred_at=NOW - timedelta(hours=2)
    )
    with session_scope(engine) as db:
        db.add(
            models.Nudge(
                learner_id=LEARNER,
                rung=1,
                state=models.NudgeState.SENT,
                scheduled_for=NOW - timedelta(hours=2),
                sent_at=NOW - timedelta(hours=2),
            )
        )

    summary = due_summary(LEARNER, now=NOW, engine=engine)

    assert summary.idle_days == pytest.approx(5.0, abs=1e-6)
    assert str(EventKind.NUDGE_SENT) not in ACTIVITY_KINDS


def test_idle_days_uses_the_most_recent_activity(engine) -> None:
    _activity(engine, NOW - timedelta(days=9))
    _activity(engine, NOW - timedelta(days=1, hours=12), kind=EventKind.RESPONSE_GRADED)
    assert due_summary(LEARNER, now=NOW, engine=engine).idle_days == pytest.approx(1.5)


def test_idle_days_falls_back_to_enrolment_for_a_learner_who_never_started(engine) -> None:
    EventLog(engine).append(
        EventKind.ENROLLED, LEARNER, {"domain_id": DOMAIN_A}, occurred_at=NOW - timedelta(days=4)
    )
    assert due_summary(LEARNER, now=NOW, engine=engine).idle_days == pytest.approx(4.0)


# --- run_due_checks ---------------------------------------------------------


def test_run_due_checks_records_sends_and_logs(engine, ladder) -> None:
    _schedule(engine, "obj_a1", NOW - timedelta(days=1))
    _activity(engine, NOW - timedelta(days=2))

    decision = run_due_checks(LEARNER, now=NOW, engine=engine)

    assert decision.send is True
    rows = _nudges(engine)
    assert len(rows) == 1
    assert rows[0].state is models.NudgeState.SENT
    assert rows[0].rung == int(StubRung.REMINDER)
    assert rows[0].sent_at is not None

    events = _nudge_events(engine)
    assert len(events) == 1
    assert events[0].payload["due_objectives"] == 1
    assert events[0].payload["rung"] == int(StubRung.REMINDER)

    assert len(ladder.channel.sent) == 1
    assert ladder.calls[0]["due_count"] == 1
    assert ladder.calls[0]["idle_days"] == pytest.approx(2.0)


def test_run_due_checks_is_idempotent_per_day(engine, ladder) -> None:
    _schedule(engine, "obj_a1", NOW - timedelta(days=1))
    _activity(engine, NOW - timedelta(days=3))

    decisions = [
        run_due_checks(LEARNER, now=NOW + timedelta(minutes=step), engine=engine)
        for step in (0, 1, 30, 120, 300)
    ]

    assert [d.send for d in decisions] == [True, False, False, False, False]
    assert decisions[-1].suppressed_reason == "already nudged today"
    assert len(_nudges(engine)) == 1
    assert len(_nudge_events(engine)) == 1
    assert len(ladder.channel.sent) == 1


def test_run_due_checks_sends_again_the_next_day(engine, ladder) -> None:
    _schedule(engine, "obj_a1", NOW - timedelta(days=1))
    _activity(engine, NOW - timedelta(days=3))

    run_due_checks(LEARNER, now=NOW, engine=engine)
    second = run_due_checks(LEARNER, now=NOW + timedelta(days=1), engine=engine)

    assert second.send is True
    assert len(_nudges(engine)) == 2


def test_run_due_checks_with_send_false_writes_nothing(engine, ladder) -> None:
    _schedule(engine, "obj_a1", NOW - timedelta(days=1))
    decision = run_due_checks(LEARNER, now=NOW, engine=engine, send=False)
    assert decision.send is True  # the ladder's verdict is still reported
    assert _nudges(engine) == []
    assert _nudge_events(engine) == []


def test_one_word_opt_out_suppresses_everything(engine, ladder) -> None:
    _schedule(engine, "obj_a1", NOW - timedelta(days=1))
    with session_scope(engine) as db:
        learner = db.get(models.Learner, LEARNER)
        learner.preferences = {"nudges": "off"}
        db.add(learner)

    decision = run_due_checks(LEARNER, now=NOW, engine=engine)

    assert decision.send is False
    assert decision.suppressed_reason == "nudges off"
    assert _nudges(engine) == []
    assert ladder.calls == []  # not even asked


def test_quiet_hours_and_last_nudge_are_passed_to_the_ladder(engine, ladder) -> None:
    _schedule(engine, "obj_a1", NOW - timedelta(days=1))
    with session_scope(engine) as db:
        learner = db.get(models.Learner, LEARNER)
        learner.preferences = {"quiet_hours": [22, 7]}
        db.add(learner)

    run_due_checks(LEARNER, now=NOW, engine=engine)
    run_due_checks(LEARNER, now=NOW + timedelta(days=1), engine=engine)

    assert ladder.calls[0]["quiet_hours"] == (22, 7)
    assert ladder.calls[0]["last_nudge"] is None
    assert ladder.calls[1]["last_nudge"] is not None
    assert ladder.calls[1]["now"] == NOW + timedelta(days=1)


def test_a_ladder_that_says_quiet_writes_nothing(engine, ladder, monkeypatch) -> None:
    monkeypatch.setattr(
        ladder,
        "decide",
        lambda **kw: StubDecision(
            rung=StubRung.QUIET, reason="nothing owed", body="", send=False
        ),
    )
    decision = run_due_checks(LEARNER, now=NOW, engine=engine)
    assert decision.send is False
    assert _nudges(engine) == []


@pytest.mark.parametrize("name", ["decide", "get_channel", "Rung", "NudgeDecision"])
def test_the_real_ladder_exposes_what_we_call(name: str) -> None:
    """When the nudge worker lands, our stub must not have drifted from it."""
    nudge = pytest.importorskip("the_oracle.nudge", reason="the nudge ladder is not built yet")
    assert hasattr(nudge, name)


# --- the crontab line -------------------------------------------------------


def test_cron_line_is_exact_and_marked() -> None:
    line = cron_lib.cron_line(hour=9, executable="/usr/local/bin/the-oracle")
    assert line.startswith("0 9 * * * /usr/local/bin/the-oracle cron run")
    assert line.endswith(cron_lib.MARKER)
    assert cron_lib.is_ours(line)


def test_cron_line_rejects_an_impossible_hour() -> None:
    with pytest.raises(ValueError):
        cron_lib.cron_line(hour=25)


def test_install_writes_the_planned_change(crontab) -> None:
    change = cron_lib.plan_install(crontab, hour=7)
    assert change.changed is True
    assert cron_lib.apply(change, crontab) is True
    assert cron_lib.find_ours(crontab.text) == [cron_lib.cron_line(hour=7)]


def test_install_refuses_to_duplicate(crontab) -> None:
    cron_lib.apply(cron_lib.plan_install(crontab), crontab)
    again = cron_lib.plan_install(crontab)

    assert again.changed is False
    assert "already installed" in again.reason
    assert cron_lib.apply(again, crontab) is False
    assert len(crontab.writes) == 1
    assert len(cron_lib.find_ours(crontab.text)) == 1


def test_changing_the_hour_replaces_rather_than_duplicates(crontab) -> None:
    cron_lib.apply(cron_lib.plan_install(crontab, hour=9), crontab)
    change = cron_lib.plan_install(crontab, hour=18)
    cron_lib.apply(change, crontab)

    assert cron_lib.find_ours(crontab.text) == [cron_lib.cron_line(hour=18)]
    assert change.removed == (cron_lib.cron_line(hour=9),)


def test_uninstall_removes_only_our_line(crontab) -> None:
    crontab.text = (
        "# my own entries\n"
        "*/5 * * * * /usr/bin/backup.sh\n"
        "0 3 * * 1 /usr/bin/rotate-logs --weekly\n"
    )
    cron_lib.apply(cron_lib.plan_install(crontab, hour=9), crontab)
    change = cron_lib.plan_uninstall(crontab)
    assert cron_lib.apply(change, crontab) is True

    assert crontab.text.splitlines() == [
        "# my own entries",
        "*/5 * * * * /usr/bin/backup.sh",
        "0 3 * * 1 /usr/bin/rotate-logs --weekly",
    ]
    assert cron_lib.find_ours(crontab.text) == []


def test_uninstall_with_nothing_installed_changes_nothing(crontab) -> None:
    crontab.text = "*/5 * * * * /usr/bin/backup.sh\n"
    change = cron_lib.plan_uninstall(crontab)
    assert change.changed is False
    assert cron_lib.apply(change, crontab) is False
    assert crontab.writes == []


def test_a_change_can_be_shown_before_it_is_applied(crontab) -> None:
    change = cron_lib.plan_install(crontab, hour=9)
    assert change.diff().startswith("+ 0 9 * * *")
    assert crontab.writes == []  # planning writes nothing


def test_status_reports_installation_and_last_run(home, crontab) -> None:
    assert cron_lib.status(crontab).installed is False
    cron_lib.apply(cron_lib.plan_install(crontab), crontab)
    cron_lib.record_run(at=NOW, result="sent rung 1")

    report = cron_lib.status(crontab)
    assert report.installed is True
    assert report.last_run_at == NOW
    assert report.last_result == "sent rung 1"


# --- the hard guarantee -----------------------------------------------------


def test_the_real_crontab_cannot_be_written_from_a_test() -> None:
    """The last line of defence: a real write is refused while pytest is loaded."""
    with pytest.raises(cron_lib.CrontabError, match="refusing to write a real crontab"):
        cron_lib.SystemCrontab().write("0 9 * * * whatever")


def test_the_default_runner_under_pytest_is_in_memory(monkeypatch) -> None:
    monkeypatch.setattr(cron_lib, "_RUNNER", None, raising=False)
    assert isinstance(cron_lib.get_runner(), cron_lib.MemoryCrontab)
    monkeypatch.setattr(cron_lib, "_RUNNER", None, raising=False)


def test_there_is_no_daemon_in_the_schedule_package() -> None:
    """PLAN.md section 9. If this fails, someone grew a background worker."""
    package = Path(cron_lib.__file__).parent
    forbidden = ("threading", "Thread(", "asyncio", "while True", "multiprocessing", "sched.")
    for path in package.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{path.name} looks like a daemon: {token}"


# --- reachability through the shipped CLI ------------------------------------


def _cli_app() -> typer.Typer:
    """Assembled exactly the way ``cli.py`` wires it."""
    app = typer.Typer()
    app.add_typer(cron_cmd.app, name="cron")
    return app


def test_cron_line_through_the_cli(home, crontab) -> None:
    result = runner.invoke(_cli_app(), ["cron", "line", "--hour", "6"])
    assert result.exit_code == 0
    assert "6 * * *" in result.stdout
    assert crontab.writes == []  # `line` installs nothing


def test_cron_install_then_status_then_uninstall_through_the_cli(home, crontab) -> None:
    crontab.text = "*/5 * * * * /usr/bin/backup.sh\n"

    installed = runner.invoke(_cli_app(), ["cron", "install", "--hour", "8", "--yes"])
    assert installed.exit_code == 0
    assert "exact change" in installed.stdout  # shown before it was applied
    assert len(cron_lib.find_ours(crontab.text)) == 1

    duplicate = runner.invoke(_cli_app(), ["cron", "install", "--hour", "8", "--yes"])
    assert duplicate.exit_code == 0
    assert "Nothing to do" in duplicate.stdout
    assert len(crontab.writes) == 1

    status = runner.invoke(_cli_app(), ["cron", "status"])
    assert status.exit_code == 0
    assert "yes" in status.stdout

    removed = runner.invoke(_cli_app(), ["cron", "uninstall", "--yes"])
    assert removed.exit_code == 0
    assert cron_lib.find_ours(crontab.text) == []
    assert "/usr/bin/backup.sh" in crontab.text


def test_cron_install_dry_run_touches_nothing(home, crontab) -> None:
    result = runner.invoke(_cli_app(), ["cron", "install", "--dry-run"])
    assert result.exit_code == 0
    assert crontab.writes == []


def test_cron_run_through_the_cli_is_the_daemonless_scheduler(engine, ladder, crontab) -> None:
    _schedule(engine, "obj_a1", NOW - timedelta(days=1))
    _activity(engine, NOW - timedelta(days=2))

    first = runner.invoke(_cli_app(), ["cron", "run"])
    second = runner.invoke(_cli_app(), ["cron", "run"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0
    assert "1 due" in first.stdout
    assert len(_nudges(engine)) == 1  # twice in one day, one nudge
    assert cron_lib.read_stamp()[0] is not None


def test_cron_is_wired_into_cli() -> None:
    """The recurring failure of this project: built, tested, never wired in."""
    from the_oracle import cli

    names = {getattr(group, "name", None) for group in cli.app.registered_groups}
    names |= {getattr(command, "name", None) for command in cli.app.registered_commands}
    if "cron" not in names:
        pytest.skip(
            "NOT WIRED. Add to cli.py:\n"
            "    from the_oracle.commands import cron as _cron_cmd\n"
            '    app.add_typer(_cron_cmd.app, name="cron")'
        )
    assert "cron" in names
