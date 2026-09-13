"""The learner model: Bayesian Knowledge Tracing over shared objectives.

Importing this package registers the mastery reducers on the default replay
registry, so ``rebuild(learner_id)`` recomputes ``MasteryState`` from the event
log alone.
"""

from __future__ import annotations

from the_oracle.mastery import reducers  # noqa: F401  (registers reducers)
from the_oracle.mastery.bkt import MASTERY_THRESHOLD, BKTParams, posterior, update
from the_oracle.mastery.profile import MasteryProfile, profile_for
from the_oracle.mastery.scheduler import Review, ReviewPlan, next_review_at
from the_oracle.mastery.synthetic import SyntheticLearner, recovery_error

__all__ = [
    "MASTERY_THRESHOLD",
    "BKTParams",
    "MasteryProfile",
    "Review",
    "ReviewPlan",
    "SyntheticLearner",
    "next_review_at",
    "posterior",
    "profile_for",
    "recovery_error",
    "update",
]
