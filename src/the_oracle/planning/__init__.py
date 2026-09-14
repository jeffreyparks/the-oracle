"""Planning: a deterministic syllabus over a domain for one learner.

Nothing in this package calls a model. Planning is arithmetic over the pack
and the mastery profile, so the same inputs always give the same plan.
"""

from __future__ import annotations

from the_oracle.planning.checkpoints import (
    DEFAULT_SESSION_MINUTES,
    Checkpoint,
    build_checkpoints,
    capacity,
    checkpoint_minutes,
)
from the_oracle.planning.syllabus import (
    REVIEW_FACTOR,
    REVIEW_FLOOR,
    PlannedObjective,
    Status,
    Syllabus,
    build_syllabus,
    classify,
    finish_date,
    planned_minutes,
    weeks_for,
)

__all__ = [
    "DEFAULT_SESSION_MINUTES",
    "REVIEW_FACTOR",
    "REVIEW_FLOOR",
    "Checkpoint",
    "PlannedObjective",
    "Status",
    "Syllabus",
    "build_checkpoints",
    "build_syllabus",
    "capacity",
    "checkpoint_minutes",
    "classify",
    "finish_date",
    "planned_minutes",
    "weeks_for",
]
