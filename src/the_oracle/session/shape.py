"""Session shape: four phases, fixed order, adaptive length.

PLAN.md section 5. Research gives no magic session length - Bradbury (2016)
debunks the fixed attention span and the 25-minute figure is Pomodoro folklore.
What is well supported is spacing (distributed practice, d = 0.54). So the
*shape* is fixed and the *length* adapts: a default of 25 minutes, settable to
10 or 50, and an early exit on measured fatigue.

The minute figures below are budgets, not a clock. They cap how much work each
phase may spend before the session moves on.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

#: Session lengths a learner may choose, in minutes.
ALLOWED_MINUTES: tuple[int, ...] = (10, 25, 50)
#: The length the shape below is written for.
BASE_MINUTES: int = 25


class Phase(StrEnum):
    """The four phases, always in this order."""

    WARMUP = "warmup"
    NEW = "new"
    PRACTICE = "practice"
    CONSOLIDATE = "consolidate"


@dataclass(frozen=True, slots=True)
class PhaseBudget:
    """How many minutes one phase may spend."""

    phase: Phase
    minutes: float


#: The 25-minute reference shape: 4 / 14 / 5 / 2.
DEFAULT_SHAPE: tuple[PhaseBudget, ...] = (
    PhaseBudget(Phase.WARMUP, 4.0),
    PhaseBudget(Phase.NEW, 14.0),
    PhaseBudget(Phase.PRACTICE, 5.0),
    PhaseBudget(Phase.CONSOLIDATE, 2.0),
)


def snap_minutes(minutes: int) -> int:
    """Snap a requested length onto the nearest supported one."""
    return min(ALLOWED_MINUTES, key=lambda allowed: (abs(allowed - int(minutes)), allowed))


def shape_for(minutes: int) -> tuple[PhaseBudget, ...]:
    """Scale :data:`DEFAULT_SHAPE` to ``minutes``. Proportions never change."""
    total = snap_minutes(minutes)
    if total == BASE_MINUTES:
        return DEFAULT_SHAPE
    factor = total / BASE_MINUTES
    return tuple(
        PhaseBudget(budget.phase, round(budget.minutes * factor, 2)) for budget in DEFAULT_SHAPE
    )


def budget_for(shape: tuple[PhaseBudget, ...], phase: Phase) -> float:
    """Minutes ``shape`` gives ``phase``. Zero when the phase is absent."""
    for budget in shape:
        if budget.phase is phase:
            return budget.minutes
    return 0.0


__all__ = [
    "ALLOWED_MINUTES",
    "BASE_MINUTES",
    "DEFAULT_SHAPE",
    "Phase",
    "PhaseBudget",
    "budget_for",
    "shape_for",
    "snap_minutes",
]
