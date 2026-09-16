"""The nudge ladder. Deterministic, no model, no network.

The ladder climbs with idle days and falls the moment the learner studies
again. Someone who comes back on day 15 is met with a plain reminder, never
with the day-14 "should we pause your plan?" message.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Any

from the_oracle.store.models import Nudge

__all__ = [
    "IDLE_DAYS",
    "NudgeDecision",
    "Rung",
    "decide",
    "in_quiet_hours",
    "nudges_enabled",
    "rung_for",
]


class Rung(IntEnum):
    """How hard the system is allowed to push. Higher is rarer."""

    QUIET = 0  # nothing owed, say nothing
    REMINDER = 1  # day 1 idle: a plain reminder
    SMALLER_ASK = 2  # day 3 idle: shrink the ask, "just 5 minutes"
    REPLAN = 3  # day 7 idle: offer to re-plan, "was the pace wrong?"
    PAUSE = 4  # day 14 idle: pause the plan and ask


IDLE_DAYS: dict[Rung, int] = {
    Rung.REMINDER: 1,
    Rung.SMALLER_ASK: 3,
    Rung.REPLAN: 7,
    Rung.PAUSE: 14,
}
"""Idle-day threshold at which each rung becomes eligible."""


@dataclass(frozen=True)
class NudgeDecision:
    """The whole decision, including the reason it was not sent."""

    rung: Rung
    reason: str
    body: str
    send: bool
    suppressed_reason: str | None = None


def nudges_enabled(preferences: Mapping[str, Any] | None) -> bool:
    """False when the learner has opted out. One word, permanent, absolute."""
    if not preferences:
        return True
    return str(preferences.get("nudges", "on")).strip().lower() != "off"


def in_quiet_hours(now: datetime, quiet_hours: tuple[int, int] | None) -> bool:
    """True when ``now`` falls inside ``[start, end)``, wrapping past midnight."""
    if quiet_hours is None:
        return False
    start, end = quiet_hours
    if start == end:
        return False
    hour = now.hour
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def rung_for(*, idle_days: float, due_count: int) -> Rung:
    """The rung earned by current state alone. No memory, so it de-escalates."""
    if due_count <= 0 and idle_days < IDLE_DAYS[Rung.REMINDER]:
        return Rung.QUIET
    if idle_days < IDLE_DAYS[Rung.REMINDER]:
        return Rung.QUIET
    earned = Rung.QUIET
    for rung in (Rung.REMINDER, Rung.SMALLER_ASK, Rung.REPLAN, Rung.PAUSE):
        if idle_days >= IDLE_DAYS[rung]:
            earned = rung
    return earned


def _nudge_day(nudge: Nudge) -> datetime | None:
    return nudge.sent_at or nudge.scheduled_for


def decide(
    *,
    idle_days: float,
    due_count: int,
    last_nudge: Nudge | None,
    now: datetime,
    quiet_hours: tuple[int, int] | None = None,
    preferences: Mapping[str, Any] | None = None,
) -> NudgeDecision:
    """Decide whether to nudge, at what rung, and with what body.

    ``preferences`` is the learner's preference mapping; ``{"nudges": "off"}``
    suppresses every nudge, with no follow-up question.
    """
    from . import templates

    if not nudges_enabled(preferences):
        return NudgeDecision(
            rung=Rung.QUIET,
            reason="the learner turned nudges off",
            body="",
            send=False,
            suppressed_reason="nudges are off",
        )

    rung = rung_for(idle_days=idle_days, due_count=due_count)
    if rung is Rung.QUIET:
        reason = (
            "nothing is owed"
            if due_count <= 0
            else f"{due_count} due but studied within the last day"
        )
        return NudgeDecision(
            rung=Rung.QUIET,
            reason=reason,
            body="",
            send=False,
            suppressed_reason="nothing owed",
        )

    body = templates.render(rung, due_count=due_count, idle_days=idle_days)
    reason = f"idle {idle_days:g} days, {due_count} due -> {rung.name.lower()}"

    if last_nudge is not None:
        previous = _nudge_day(last_nudge)
        if previous is not None and previous.date() == now.date():
            return NudgeDecision(
                rung=rung,
                reason=reason,
                body=body,
                send=False,
                suppressed_reason="already nudged today",
            )

    if in_quiet_hours(now, quiet_hours):
        return NudgeDecision(
            rung=rung,
            reason=reason,
            body=body,
            send=False,
            suppressed_reason="quiet hours",
        )

    return NudgeDecision(rung=rung, reason=reason, body=body, send=True)
