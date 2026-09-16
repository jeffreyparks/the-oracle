"""The nudge opt-out must be reachable, not just documented."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from the_oracle.cli import app as cli_app
from the_oracle.store import models
from the_oracle.store.db import create_all, get_engine


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ORACLE_HOME", str(tmp_path))
    monkeypatch.setenv("ORACLE_LEARNER_ID", "opt_out_user")
    from the_oracle.config import get_settings

    get_settings.cache_clear()
    create_all(get_engine())
    return tmp_path


def _prefs(learner_id: str) -> dict:
    from sqlmodel import Session

    with Session(get_engine()) as session:
        learner = session.get(models.Learner, learner_id)
        return dict((learner.preferences if learner else {}) or {})


def test_pause_sets_the_opt_out_the_ladder_reads(home) -> None:
    """The day-14 template tells the learner to run this. It must work."""
    result = CliRunner().invoke(cli_app, ["learner", "pause"])
    assert result.exit_code == 0, result.output
    assert _prefs("opt_out_user")["nudges"] == "off"


def test_resume_turns_reminders_back_on(home) -> None:
    runner = CliRunner()
    runner.invoke(cli_app, ["learner", "pause"])
    result = runner.invoke(cli_app, ["learner", "resume"])
    assert result.exit_code == 0, result.output
    assert _prefs("opt_out_user")["nudges"] == "on"


def test_pause_is_honoured_by_the_ladder(home) -> None:
    """End to end: the command writes what nudge.decide actually reads."""
    from datetime import UTC, datetime

    from the_oracle.nudge import decide

    CliRunner().invoke(cli_app, ["learner", "pause"])
    decision = decide(
        idle_days=14.0,
        due_count=5,
        last_nudge=None,
        now=datetime(2026, 9, 15, 10, 0, tzinfo=UTC),
        preferences=_prefs("opt_out_user"),
    )
    assert decision.send is False
    assert decision.suppressed_reason


def test_every_command_named_in_a_nudge_template_exists() -> None:
    """A template must never promise a command the product does not have."""
    import re

    from the_oracle.nudge import templates

    text = " ".join(str(v) for v in templates.TEMPLATES.values())
    named = set(re.findall(r"\boracle ([a-z]+(?: [a-z]+)?)", text))
    assert named, "no commands referenced; check the template module"

    runner = CliRunner()
    for command in sorted(named):
        args = command.split() + ["--help"]
        result = runner.invoke(cli_app, args)
        assert result.exit_code == 0, f"nudge template names missing command: {command}"
