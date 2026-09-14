"""Measured fatigue: end the session when this learner decays, not on a clock.

PLAN.md section 5: "the Oracle tracks in-session accuracy against the learner's
own baseline and ends the session when decay is detected, with a plain
explanation. Measured fatigue beats an assumed constant."

Two guards keep this honest:

* ``min_items`` - a short streak of bad luck is not fatigue. Nothing may fire
  before enough answers are in.
* the learner's own baseline - a learner who runs at 55% is not fatigued at
  55%. The comparison is always against that person's history.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Engine

#: How many recent answers the observed accuracy is measured over.
WINDOW: int = 6
#: How far below baseline the recent window must fall before we call it decay.
MARGIN: float = 0.25
#: Baseline used when the learner has too little history of their own.
DEFAULT_BASELINE: float = 0.7
#: Observations needed before a learner's own history replaces the default.
MIN_HISTORY: int = 8


@dataclass(frozen=True, slots=True)
class FatigueSignal:
    """The verdict on whether to stop, and the numbers behind it."""

    stop: bool
    reason: str
    observed: float
    baseline: float


def _percent(value: float) -> str:
    return f"{round(value * 100)}%"


def fatigue(
    responses: Sequence[bool],
    *,
    baseline: float,
    min_items: int = 6,
    window: int = WINDOW,
    margin: float = MARGIN,
) -> FatigueSignal:
    """Decide whether accuracy has decayed far enough to stop the session.

    ``responses`` is this session's answers, oldest first. ``baseline`` is the
    learner's own usual accuracy. The signal never fires before ``min_items``
    answers, and it is only ever read between attempts, never inside one.
    """
    baseline = max(0.0, min(1.0, float(baseline)))
    answered = list(responses)
    if len(answered) < max(1, min_items):
        observed = sum(answered) / len(answered) if answered else 0.0
        return FatigueSignal(
            stop=False,
            reason=(
                f"Too early to judge. {len(answered)} answers in, and I need "
                f"{min_items} before I trust a dip."
            ),
            observed=observed,
            baseline=baseline,
        )

    recent = answered[-window:]
    observed = sum(recent) / len(recent)
    if observed > baseline - margin:
        return FatigueSignal(
            stop=False,
            reason=(
                f"You are running at {_percent(observed)} over your last "
                f"{len(recent)} answers, against your usual {_percent(baseline)}. "
                "That is normal for you."
            ),
            observed=observed,
            baseline=baseline,
        )

    return FatigueSignal(
        stop=True,
        reason=(
            f"Your last {len(recent)} answers ran at {_percent(observed)}. You "
            f"usually run at {_percent(baseline)}. That is a real drop, not noise, "
            "so I am stopping here. Come back fresh and the spacing does the work."
        ),
        observed=observed,
        baseline=baseline,
    )


def baseline_for(
    learner_id: str,
    *,
    engine: Engine | None = None,
    default: float = DEFAULT_BASELINE,
    min_history: int = MIN_HISTORY,
) -> float:
    """The learner's own historical accuracy, from the event log.

    Falls back to ``default`` while the history is too thin to mean anything.
    """
    from the_oracle.store.events import EventKind, EventLog

    graded = EventLog(engine).read(learner_id, kinds=[EventKind.RESPONSE_GRADED])
    if len(graded) < min_history:
        return default
    correct = sum(1 for event in graded if bool(event.payload.get("correct")))
    return correct / len(graded)


__all__ = [
    "DEFAULT_BASELINE",
    "MARGIN",
    "MIN_HISTORY",
    "WINDOW",
    "FatigueSignal",
    "baseline_for",
    "fatigue",
]
