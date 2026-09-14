"""The planner: turn a domain plus what a learner knows into an ordered plan.

Pure and deterministic. No LLM, no network, no clock inside the planning maths.
The same domain and the same mastery state always produce the same syllabus,
because a plan that shifts under the learner destroys trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING, Literal

from the_oracle.domains.registry import load_domain
from the_oracle.mastery import MASTERY_THRESHOLD
from the_oracle.mastery.profile import MasteryProfile, profile_for

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy import Engine

    from the_oracle.domains.schema import Domain

Status = Literal["mastered", "review", "learn"]

REVIEW_FLOOR: float = 0.5
"""Below this, the objective is taught again from the start, not reviewed."""

REVIEW_FACTOR: float = 0.4
"""Review costs 40% of first-teaching time.

Justification, not a guess. A first pass pays for exposition, worked examples,
and applied practice. A review pays for retrieval and repair only: the learner
has the model already, it is just weak. PLAN.md section 5 budgets ~14 of 25
session minutes to new material and ~4 to retrieval warm-up plus ~5 to applied
practice; dropping the exposition and keeping retrieval and practice lands near
0.4. It is also the conservative side of the savings effect: relearning is
cheaper than learning, but never free, so we never plan a review at zero.
"""

MINIMUM_MINUTES: int = 1
"""No planned item is ever costed at zero. Honest totals only."""


def classify(p: float, *, threshold: float = MASTERY_THRESHOLD) -> Status:
    """Bucket one mastery estimate. ``>= threshold`` is mastered, ``>= 0.5`` is review."""
    if p >= threshold:
        return "mastered"
    if p >= REVIEW_FLOOR:
        return "review"
    return "learn"


def planned_minutes(est_minutes: int, status: Status) -> int:
    """Minutes to plan for one objective given its status."""
    if status == "mastered":
        return 0
    if status == "review":
        return max(MINIMUM_MINUTES, round(est_minutes * REVIEW_FACTOR))
    return max(MINIMUM_MINUTES, int(est_minutes))


@dataclass(frozen=True, slots=True)
class PlannedObjective:
    """One unit of planned work, in teaching order."""

    objective_id: str
    module_id: str
    est_minutes: int
    p_mastery: float
    status: Status


@dataclass(frozen=True, slots=True)
class Syllabus:
    """An ordered plan over one domain for one learner."""

    domain_id: str
    learner_id: str
    items: list[PlannedObjective] = field(default_factory=list)
    total_minutes: int = 0
    weeks: float = 0.0
    skipped_mastered: list[str] = field(default_factory=list)

    # -- reads -------------------------------------------------------------

    def module_ids(self) -> list[str]:
        """Modules that carry planned work, in plan order, without repeats."""
        seen: list[str] = []
        for item in self.items:
            if item.module_id not in seen:
                seen.append(item.module_id)
        return seen

    def for_module(self, module_id: str) -> list[PlannedObjective]:
        """Planned items belonging to one module, in plan order."""
        return [item for item in self.items if item.module_id == module_id]

    def module_minutes(self, module_id: str) -> int:
        return sum(item.est_minutes for item in self.for_module(module_id))

    @property
    def total_hours(self) -> float:
        return self.total_minutes / 60.0

    def counts(self) -> dict[str, int]:
        """How many items of each status, including what was skipped."""
        return {
            "learn": sum(1 for i in self.items if i.status == "learn"),
            "review": sum(1 for i in self.items if i.status == "review"),
            "mastered": len(self.skipped_mastered),
        }


def weeks_for(total_minutes: int, hours_per_week: float) -> float:
    """Calendar weeks of study at a given weekly budget."""
    if hours_per_week <= 0:
        msg = "hours_per_week must be greater than zero"
        raise ValueError(msg)
    return (total_minutes / 60.0) / hours_per_week


def finish_date(weeks: float, start: date | None = None) -> date:
    """A realistic finish date. Partial weeks round up to a whole day."""
    start = start or date.today()
    return start + timedelta(days=int(-(-(weeks * 7.0) // 1)))


def _module_of(domain: "Domain") -> dict[str, str]:
    """objective id -> module id. Rule 4 guarantees exactly one module each."""
    return {oid: module.id for module in domain.modules for oid in module.objectives}


def build_syllabus(
    domain_id: str,
    learner_id: str,
    *,
    hours_per_week: float = 3.0,
    engine: "Engine | None" = None,
    profile: MasteryProfile | None = None,
    domain: "Domain | None" = None,
) -> Syllabus:
    """Plan one domain for one learner.

    ``profile`` and ``domain`` are optional injection seams for callers that
    already hold them; both default to a read. A learner with no recorded
    evidence is not an error - every objective answers with the learner model's
    default, so the plan is simply the whole domain.
    """
    domain = domain if domain is not None else load_domain(domain_id)
    profile = profile if profile is not None else profile_for(learner_id, engine)

    module_of = _module_of(domain)
    items: list[PlannedObjective] = []
    skipped: list[str] = []

    for objective_id in domain.teaching_order():
        p = round(float(profile.p(objective_id)), 6)
        status = classify(p)
        if status == "mastered":
            skipped.append(objective_id)
            continue
        items.append(
            PlannedObjective(
                objective_id=objective_id,
                module_id=module_of.get(objective_id, ""),
                est_minutes=planned_minutes(domain.objective(objective_id).est_minutes, status),
                p_mastery=p,
                status=status,
            )
        )

    total_minutes = sum(item.est_minutes for item in items)
    return Syllabus(
        domain_id=domain.id,
        learner_id=learner_id,
        items=items,
        total_minutes=total_minutes,
        weeks=weeks_for(total_minutes, hours_per_week),
        skipped_mastered=skipped,
    )


__all__ = [
    "MINIMUM_MINUTES",
    "REVIEW_FACTOR",
    "REVIEW_FLOOR",
    "PlannedObjective",
    "Status",
    "Syllabus",
    "build_syllabus",
    "classify",
    "finish_date",
    "planned_minutes",
    "weeks_for",
]
