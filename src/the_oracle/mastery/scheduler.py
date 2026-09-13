"""Review scheduling: when should this objective come back?

What this is
------------
An FSRS-*style* scheduler, not FSRS. Real FSRS (Free Spaced Repetition
Scheduler) fits ~17 parameters to a large review corpus and predicts retention
with a power-law forgetting curve. We have no corpus yet, so fitting those
parameters would be theatre.

What we approximate, and how honest each part is
------------------------------------------------
* **Stability** - FSRS keeps a memory-stability term in days that grows after a
  successful review and collapses after a lapse. We keep the same variable and
  the same two moves, but with fixed multipliers instead of fitted ones.
* **Difficulty** - FSRS keeps a per-item difficulty in [1, 10] that drifts with
  every grade. We keep a per-objective difficulty in [0, 1] driven by the
  observed error rate and the item difficulty, and use it to damp stability
  growth.
* **Retrievability / forgetting curve** - NOT implemented. FSRS solves the
  power-law curve for the interval that hits a target retention. We simply set
  ``interval = stability`` and lean on the growth rule. This is the largest
  approximation here.
* **Grades** - FSRS takes four grades (again / hard / good / easy). We only have
  a boolean from the grader, so ``correct`` maps to "good" and ``not correct``
  maps to "again".
* **Fuzz** - FSRS randomises intervals to spread load. We do not: replay must be
  deterministic.

The whole module is pure. Same history in, same schedule out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

#: First interval after a single success, in days.
INITIAL_STABILITY: float = 1.0
#: Stability multiplier floor and ceiling on a success.
MIN_EASE: float = 1.3
MAX_EASE: float = 3.4
#: What a lapse leaves behind, as a fraction of current stability.
LAPSE_RETENTION: float = 0.35
#: Never schedule closer than this or further than this.
MIN_INTERVAL_DAYS: float = 0.5
MAX_INTERVAL_DAYS: float = 365.0


@dataclass(frozen=True, slots=True)
class Review:
    """One graded opportunity, as the scheduler sees it."""

    correct: bool
    at: datetime
    difficulty: int = 1


@dataclass(frozen=True, slots=True)
class ReviewPlan:
    """The derived schedule for one learner-objective pair."""

    due_at: datetime
    interval_days: float
    stability: float
    difficulty: float
    lapses: int
    reviews: int


def _ease(p_mastery: float, difficulty: float) -> float:
    """Growth multiplier for a successful review.

    High mastery and low difficulty earn a long jump; shaky mastery or a hard
    objective earns a short one.
    """
    span = MAX_EASE - MIN_EASE
    return MIN_EASE + span * max(0.0, min(1.0, p_mastery)) * (1.0 - 0.5 * difficulty)


def _difficulty(reviews: Sequence[Review]) -> float:
    """Per-objective difficulty in [0, 1] from error rate and item difficulty.

    Equal weight to "how often was this wrong" and "how hard were the items",
    where item difficulty is the 1-5 scale from the pack format.
    """
    if not reviews:
        return 0.5
    errors = sum(1 for r in reviews if not r.correct) / len(reviews)
    load = sum(max(1, min(5, r.difficulty)) for r in reviews) / len(reviews)
    return max(0.0, min(1.0, 0.5 * errors + 0.5 * (load - 1.0) / 4.0))


def plan(reviews: Sequence[Review], p_mastery: float) -> ReviewPlan | None:
    """Fold a response history into a review plan. ``None`` when there is none."""
    ordered = list(reviews)
    if not ordered:
        return None

    difficulty = _difficulty(ordered)
    stability = 0.0
    lapses = 0
    for review in ordered:
        if review.correct:
            stability = (
                INITIAL_STABILITY
                if stability <= 0.0
                else stability * _ease(p_mastery, difficulty)
            )
        else:
            lapses += 1
            stability = max(MIN_INTERVAL_DAYS, stability * LAPSE_RETENTION)

    interval = max(MIN_INTERVAL_DAYS, min(MAX_INTERVAL_DAYS, stability or MIN_INTERVAL_DAYS))
    last = ordered[-1].at
    return ReviewPlan(
        due_at=last + timedelta(days=interval),
        interval_days=interval,
        stability=stability,
        difficulty=difficulty,
        lapses=lapses,
        reviews=len(ordered),
    )


def next_review_at(reviews: Sequence[Review], p_mastery: float) -> datetime | None:
    """Just the date. ``None`` when the objective has never been seen."""
    result = plan(reviews, p_mastery)
    return result.due_at if result else None


def is_due(plan_: ReviewPlan, now: datetime) -> bool:
    """True when this objective should be injected into the next session."""
    return now >= plan_.due_at


__all__ = [
    "INITIAL_STABILITY",
    "LAPSE_RETENTION",
    "MAX_INTERVAL_DAYS",
    "MIN_INTERVAL_DAYS",
    "Review",
    "ReviewPlan",
    "is_due",
    "next_review_at",
    "plan",
]
