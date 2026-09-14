"""The study session: the four-phase loop at the heart of the tutor.

Everything the earlier phases built - the domain pack, the learner model, the
corpus, the plan - exists to make this loop possible. PLAN.md section 5.
"""

from __future__ import annotations

from the_oracle.session.fatigue import FatigueSignal, baseline_for, fatigue
from the_oracle.session.gating import blocked_by, prerequisites_met, teachable
from the_oracle.session.loop import (
    SessionResult,
    run_session,
    run_session_async,
    should_interrupt,
)
from the_oracle.session.planner import (
    SessionPlan,
    due_reviews,
    plan_session,
    recently_studied,
)
from the_oracle.session.shape import (
    ALLOWED_MINUTES,
    DEFAULT_SHAPE,
    Phase,
    PhaseBudget,
    shape_for,
)

__all__ = [
    "ALLOWED_MINUTES",
    "DEFAULT_SHAPE",
    "FatigueSignal",
    "Phase",
    "PhaseBudget",
    "SessionPlan",
    "SessionResult",
    "baseline_for",
    "blocked_by",
    "due_reviews",
    "fatigue",
    "plan_session",
    "prerequisites_met",
    "recently_studied",
    "run_session",
    "run_session_async",
    "shape_for",
    "should_interrupt",
    "teachable",
]
