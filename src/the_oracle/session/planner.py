"""Planning one session: what to warm up on, what to learn, how long.

PLAN.md section 5. Review items are injected into a normal session as the
warm-up - never a separate chore - and new material is gated strictly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Engine

from the_oracle.mastery import MASTERY_THRESHOLD
from the_oracle.mastery.scheduler import Review, is_due, plan as review_plan
from the_oracle.session.gating import teachable
from the_oracle.session.shape import PhaseBudget, shape_for, snap_minutes
from the_oracle.store.events import EventKind, EventLog

if TYPE_CHECKING:  # pragma: no cover
    from the_oracle.domains.schema import Domain
    from the_oracle.mastery import MasteryProfile

#: How many objectives the warm-up carries. PLAN.md says 2-3 items.
WARMUP_MIN: int = 2
WARMUP_MAX: int = 3
#: One new objective, rarely two. The second only appears in a long session.
FOCUS_MAX_SHORT: int = 1
FOCUS_MAX_LONG: int = 2
#: A session this long may carry a second new objective.
LONG_SESSION_MINUTES: int = 50


@dataclass(frozen=True, slots=True)
class SessionPlan:
    """The shape of one session, decided before a single item is written."""

    learner_id: str
    domain_id: str
    minutes: int
    warmup: list[str] = field(default_factory=list)
    focus: list[str] = field(default_factory=list)
    shape: tuple[PhaseBudget, ...] = ()

    @property
    def objectives(self) -> list[str]:
        """Warm-up then focus, first appearance only."""
        seen: list[str] = []
        for objective_id in [*self.warmup, *self.focus]:
            if objective_id not in seen:
                seen.append(objective_id)
        return seen


def _graded_by_objective(
    learner_id: str, engine: Engine | None
) -> dict[str, list[Review]]:
    """Every graded answer, as scheduler reviews, keyed by objective."""
    history: dict[str, list[Review]] = {}
    for event in EventLog(engine).read(learner_id, kinds=[EventKind.RESPONSE_GRADED]):
        objective_id = event.payload.get("objective_id")
        if not isinstance(objective_id, str) or not objective_id:
            continue
        at = event.occurred_at
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        history.setdefault(objective_id, []).append(
            Review(
                correct=bool(event.payload.get("correct")),
                at=at,
                difficulty=int(event.payload.get("difficulty") or 1),
            )
        )
    return history


def due_reviews(
    domain: "Domain",
    profile: "MasteryProfile",
    learner_id: str,
    *,
    now: datetime | None = None,
    engine: Engine | None = None,
) -> list[str]:
    """Objectives whose review is due, oldest due date first."""
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    in_domain = set(domain.teaching_order())
    due: list[tuple[datetime, str]] = []
    for objective_id, reviews in _graded_by_objective(learner_id, engine).items():
        if objective_id not in in_domain:
            continue
        schedule = review_plan(reviews, profile.p(objective_id))
        if schedule is None:
            continue
        due_at = schedule.due_at
        if due_at.tzinfo is None:
            due_at = due_at.replace(tzinfo=UTC)
        if is_due(schedule, moment) or due_at <= moment:
            due.append((due_at, objective_id))
    due.sort(key=lambda pair: (pair[0], pair[1]))
    return [objective_id for _, objective_id in due]


def recently_studied(
    domain: "Domain", learner_id: str, *, engine: Engine | None = None
) -> list[str]:
    """Most recently studied objectives in this domain, newest first.

    The warm-up falls back to these when nothing is due, so the session still
    opens with retrieval practice instead of cold new material.
    """
    in_domain = set(domain.teaching_order())
    last: dict[str, datetime] = {}
    for objective_id, reviews in _graded_by_objective(learner_id, engine).items():
        if objective_id in in_domain and reviews:
            last[objective_id] = max(review.at for review in reviews)
    return [oid for oid, _ in sorted(last.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]


def plan_session(
    domain_id: str,
    learner_id: str,
    *,
    minutes: int = 25,
    now: datetime | None = None,
    engine: Engine | None = None,
    threshold: float = MASTERY_THRESHOLD,
) -> SessionPlan:
    """Decide the warm-up, the focus, and the shape for one session."""
    from the_oracle.domains.registry import load_domain
    from the_oracle.mastery import profile_for

    domain = load_domain(domain_id)
    profile = profile_for(learner_id, engine)
    length = snap_minutes(minutes)

    warmup = due_reviews(domain, profile, learner_id, now=now, engine=engine)[:WARMUP_MAX]
    if not warmup:
        warmup = recently_studied(domain, learner_id, engine=engine)[:WARMUP_MAX]

    ready = [
        objective_id
        for objective_id in teachable(domain, profile, threshold=threshold)
        if objective_id not in warmup
    ]
    limit = FOCUS_MAX_LONG if length >= LONG_SESSION_MINUTES else FOCUS_MAX_SHORT
    focus = ready[:limit]

    return SessionPlan(
        learner_id=learner_id,
        domain_id=domain_id,
        minutes=length,
        warmup=list(warmup),
        focus=focus,
        shape=shape_for(length),
    )


__all__ = [
    "FOCUS_MAX_LONG",
    "FOCUS_MAX_SHORT",
    "WARMUP_MAX",
    "WARMUP_MIN",
    "SessionPlan",
    "due_reviews",
    "plan_session",
    "recently_studied",
]
