"""Tests for the nudge ladder, its templates, and its channels.

Time is frozen everywhere. Nothing here touches a model, a network, or a clock.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from the_oracle import style
from the_oracle.nudge import (
    IDLE_DAYS,
    TEMPLATES,
    MemoryChannel,
    NudgeChannel,
    NudgeDecision,
    Rung,
    TerminalChannel,
    decide,
    get_channel,
    render,
    rung_for,
)
from the_oracle.store.models import Nudge, NudgeState

NOW = datetime(2025, 3, 12, 9, 0, tzinfo=UTC)


def _nudge(*, sent_at: datetime | None = None, rung: Rung = Rung.REMINDER) -> Nudge:
    return Nudge(
        learner_id="learner-1",
        rung=int(rung),
        channel="memory",
        state=NudgeState.SENT if sent_at else NudgeState.QUEUED,
        body="x",
        scheduled_for=sent_at or NOW,
        sent_at=sent_at,
    )


# ---------------------------------------------------------------- the ladder


@pytest.mark.parametrize(
    ("idle_days", "expected"),
    [
        (0.0, Rung.QUIET),
        (0.9, Rung.QUIET),
        (1.0, Rung.REMINDER),
        (2.9, Rung.REMINDER),
        (3.0, Rung.SMALLER_ASK),
        (6.5, Rung.SMALLER_ASK),
        (7.0, Rung.REPLAN),
        (13.9, Rung.REPLAN),
        (14.0, Rung.PAUSE),
        (40.0, Rung.PAUSE),
    ],
)
def test_rung_for_climbs_with_idle_days(idle_days: float, expected: Rung) -> None:
    assert rung_for(idle_days=idle_days, due_count=3) is expected


def test_idle_days_thresholds_match_the_contract() -> None:
    assert IDLE_DAYS == {
        Rung.REMINDER: 1,
        Rung.SMALLER_ASK: 3,
        Rung.REPLAN: 7,
        Rung.PAUSE: 14,
    }


def test_reminder_is_sent_and_states_the_cost() -> None:
    decision = decide(idle_days=1.0, due_count=3, last_nudge=None, now=NOW)
    assert decision.send is True
    assert decision.rung is Rung.REMINDER
    assert decision.suppressed_reason is None
    assert "3 items due, about 10 minutes" in decision.body


# --------------------------------------------------- one nudge per calendar day


def test_second_nudge_the_same_day_is_suppressed() -> None:
    earlier = _nudge(sent_at=NOW.replace(hour=7))
    decision = decide(idle_days=3.0, due_count=2, last_nudge=earlier, now=NOW)
    assert decision.send is False
    assert decision.suppressed_reason == "already nudged today"


def test_a_nudge_yesterday_does_not_block_today() -> None:
    yesterday = _nudge(sent_at=NOW - timedelta(days=1))
    decision = decide(idle_days=3.0, due_count=2, last_nudge=yesterday, now=NOW)
    assert decision.send is True


def test_a_queued_nudge_today_still_blocks() -> None:
    queued = _nudge(sent_at=None)  # scheduled_for is today
    decision = decide(idle_days=3.0, due_count=2, last_nudge=queued, now=NOW)
    assert decision.send is False
    assert decision.suppressed_reason == "already nudged today"


# ------------------------------------------------------------------ quiet hours


def test_quiet_hours_suppress_the_send() -> None:
    night = NOW.replace(hour=23)
    decision = decide(
        idle_days=3.0, due_count=2, last_nudge=None, now=night, quiet_hours=(22, 7)
    )
    assert decision.send is False
    assert decision.suppressed_reason == "quiet hours"


def test_quiet_hours_wrap_past_midnight() -> None:
    early = NOW.replace(hour=3)
    decision = decide(
        idle_days=3.0, due_count=2, last_nudge=None, now=early, quiet_hours=(22, 7)
    )
    assert decision.send is False


def test_outside_quiet_hours_the_nudge_goes() -> None:
    decision = decide(
        idle_days=3.0, due_count=2, last_nudge=None, now=NOW, quiet_hours=(22, 7)
    )
    assert decision.send is True


# ------------------------------------------------------------- de-escalation


def test_study_activity_resets_the_rung_to_quiet() -> None:
    """Fourteen days idle, then a session. The next decision is silence."""
    hard = decide(idle_days=14.0, due_count=5, last_nudge=None, now=NOW)
    assert hard.rung is Rung.PAUSE

    after_study = decide(
        idle_days=0.0,
        due_count=5,
        last_nudge=_nudge(sent_at=NOW - timedelta(days=1), rung=Rung.PAUSE),
        now=NOW,
    )
    assert after_study.rung is Rung.QUIET
    assert after_study.send is False
    assert after_study.body == ""


def test_a_returning_learner_gets_the_gentlest_rung_not_the_last_one() -> None:
    """Day 15 return, one day idle again: a plain reminder, not "pause?"."""
    decision = decide(
        idle_days=1.0,
        due_count=2,
        last_nudge=_nudge(sent_at=NOW - timedelta(days=2), rung=Rung.PAUSE),
        now=NOW,
    )
    assert decision.rung is Rung.REMINDER
    assert decision.send is True
    assert "pause" not in decision.body.lower()


# ------------------------------------------------------------- nothing is owed


def test_nothing_owed_says_nothing() -> None:
    decision = decide(idle_days=0.2, due_count=0, last_nudge=None, now=NOW)
    assert decision.send is False
    assert decision.rung is Rung.QUIET
    assert decision.suppressed_reason == "nothing owed"
    assert decision.body == ""


def test_due_work_but_studied_today_says_nothing() -> None:
    decision = decide(idle_days=0.0, due_count=9, last_nudge=None, now=NOW)
    assert decision.send is False
    assert decision.rung is Rung.QUIET


# ------------------------------------------------------------------- opt-out


@pytest.mark.parametrize("value", ["off", "OFF", " Off "])
def test_one_word_opt_out_suppresses_everything(value: str) -> None:
    decision = decide(
        idle_days=30.0,
        due_count=40,
        last_nudge=None,
        now=NOW,
        preferences={"nudges": value},
    )
    assert decision.send is False
    assert decision.rung is Rung.QUIET
    assert decision.suppressed_reason == "nudges are off"
    assert decision.body == ""


def test_opt_out_is_not_argued_with() -> None:
    decision = decide(
        idle_days=30.0, due_count=40, last_nudge=None, now=NOW, preferences={"nudges": "off"}
    )
    text = f"{decision.body} {decision.reason} {decision.suppressed_reason}".lower()
    for phrase in ("are you sure", "sure?", "really", "miss out", "reconsider"):
        assert phrase not in text


def test_absent_preference_means_nudges_are_on() -> None:
    assert decide(idle_days=1.0, due_count=1, last_nudge=None, now=NOW, preferences={}).send


# --------------------------------------------------------------------- tone


def test_no_banned_phrase_appears_in_any_template() -> None:
    for rung, template in TEMPLATES.items():
        lowered = template.lower()
        for phrase in style.BANNED_PHRASES:
            assert phrase not in lowered, f"{rung.name} uses a banned phrase: {phrase}"


def test_no_banned_phrase_appears_in_any_rendered_body() -> None:
    for rung in Rung:
        for due_count in (0, 1, 3, 12):
            body = render(rung, due_count=due_count, idle_days=IDLE_DAYS.get(rung, 0)).lower()
            for phrase in style.BANNED_PHRASES:
                assert phrase not in body


def test_templates_never_guilt_trip_or_manufacture_urgency() -> None:
    banned = (
        "!",
        "streak",
        "don't lose",
        "falling behind",
        "fallen behind",
        "you should have",
        "disappoint",
        "let us down",
        "let yourself down",
        "last chance",
        "urgent",
        "hurry",
        "act now",
        "failing",
        "lazy",
        "guilt",
        "emoji",
    )
    for rung, template in TEMPLATES.items():
        lowered = template.lower()
        for phrase in banned:
            assert phrase not in lowered, f"{rung.name} uses {phrase!r}"


def test_every_rung_has_a_template_and_only_quiet_is_empty() -> None:
    assert set(TEMPLATES) == set(Rung)
    assert TEMPLATES[Rung.QUIET] == ""
    for rung in (Rung.REMINDER, Rung.SMALLER_ASK, Rung.REPLAN, Rung.PAUSE):
        assert TEMPLATES[rung].strip()


def test_bodies_are_short() -> None:
    for rung in (Rung.REMINDER, Rung.SMALLER_ASK, Rung.REPLAN, Rung.PAUSE):
        body = render(rung, due_count=3, idle_days=IDLE_DAYS[rung])
        assert len(body.splitlines()) <= 3
        assert len(body) <= 260


def test_the_estimate_is_honest_about_one_item() -> None:
    body = render(Rung.REMINDER, due_count=1, idle_days=1)
    assert "1 item due" in body


# ------------------------------------------------------------------ channels


def test_terminal_channel_satisfies_the_protocol() -> None:
    channel = get_channel("terminal")
    assert isinstance(channel, TerminalChannel)
    assert isinstance(channel, NudgeChannel)
    assert channel.name == "terminal"


def test_terminal_channel_writes_the_body(capsys: pytest.CaptureFixture[str]) -> None:
    nudge = _nudge()
    nudge.body = "2 items due, about 10 minutes."
    assert TerminalChannel().send(nudge) is True
    assert "2 items due" in capsys.readouterr().out


def test_a_channel_refuses_an_empty_body() -> None:
    nudge = _nudge()
    nudge.body = ""
    assert TerminalChannel().send(nudge) is False
    assert MemoryChannel().send(nudge) is False


def test_memory_channel_collects_nudges() -> None:
    channel = get_channel("memory")
    assert isinstance(channel, MemoryChannel)
    nudge = _nudge()
    nudge.body = "3 items due, about 10 minutes."
    assert channel.send(nudge) is True
    assert channel.sent == [nudge]


def test_unknown_channel_fails_loud() -> None:
    with pytest.raises(ValueError, match="unknown nudge channel"):
        get_channel("carrier-pigeon")


# ------------------------------------------------------- decision is a value


def test_decision_is_frozen() -> None:
    decision = decide(idle_days=1.0, due_count=1, last_nudge=None, now=NOW)
    assert isinstance(decision, NudgeDecision)
    with pytest.raises(Exception):
        decision.send = False  # type: ignore[misc]
