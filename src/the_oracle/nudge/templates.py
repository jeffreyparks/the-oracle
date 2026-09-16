"""Nudge bodies. Plain text, no model, no guilt.

Every body is written for an adult who is busy. A nudge states what is owed,
how long it takes, and the one command that does it. It never implies the
learner has let anyone down, and it never manufactures urgency.
"""

from __future__ import annotations

from .ladder import Rung

MINUTES_PER_ITEM: int = 3
"""Rough cost of one review item. Used only to state an honest estimate."""


def _plural(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def estimate_minutes(due_count: int) -> int:
    """Minutes for ``due_count`` items, rounded to the nearest 5, at least 5."""
    if due_count <= 0:
        return 0
    raw = due_count * MINUTES_PER_ITEM
    return max(5, 5 * round(raw / 5))


def due_clause(due_count: int) -> str:
    """"3 items due, about 10 minutes" — or a plain phrase when nothing is due."""
    if due_count <= 0:
        return "Nothing is due right now"
    item = _plural(due_count, "item", "items")
    return f"{due_count} {item} due, about {estimate_minutes(due_count)} minutes"


TEMPLATES: dict[Rung, str] = {
    Rung.QUIET: "",
    Rung.REMINDER: (
        "{due_clause}.\n"
        "Run `oracle review` when you have a gap."
    ),
    Rung.SMALLER_ASK: (
        "{idle_days} days since your last session. {due_clause}.\n"
        "If that is too much today, do one item. Five minutes still counts."
    ),
    Rung.REPLAN: (
        "A week without a session. That is usually the plan, not you.\n"
        "Run `oracle plan` to change the pace, or `oracle review` "
        "to pick up where you stopped."
    ),
    Rung.PAUSE: (
        "Two weeks idle. Do you want to pause this plan?\n"
        "`oracle learner pause` stops the reminders and keeps your progress. "
        "`oracle review` starts again. Either is a fine answer."
    ),
}
"""One body per rung. Keys cover every :class:`Rung` member."""


def render(rung: Rung, *, due_count: int = 0, idle_days: float = 0.0) -> str:
    """Fill the template for ``rung``. Deterministic; no model is called."""
    template = TEMPLATES[rung]
    if not template:
        return ""
    return template.format(
        due_clause=due_clause(due_count),
        due_count=due_count,
        idle_days=int(idle_days),
        minutes=estimate_minutes(due_count),
    )
