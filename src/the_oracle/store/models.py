"""SQLModel tables.

Two families:

* **Shared** - the moat. Objectives, domains, modules, resources, items.
  Not owned by any learner.
* **Learner-scoped** - private, exportable, deletable.

Non-negotiable: :class:`MasteryState` is keyed ``(learner_id, objective_id)``.
It has no ``domain_id``, because objectives are shared atoms and mastery
transfers across domains.

Portable SQL only. Every column type works on SQLite and Postgres alike.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Column, Text, UniqueConstraint
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    """Timezone-aware now. Every timestamp column uses this."""
    return datetime.now(UTC)


def _json_column() -> Column:  # type: ignore[type-arg]
    return Column(JSON, nullable=False, default=dict)


def _text_column(*, nullable: bool = True) -> Column:  # type: ignore[type-arg]
    return Column(Text, nullable=nullable)


class Bloom(StrEnum):
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class ResourceKind(StrEnum):
    FOUND = "found"
    AUTHORED = "authored"


class ReviewVerdict(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"


class NudgeState(StrEnum):
    QUEUED = "queued"
    SENT = "sent"
    ACKED = "acked"
    SUPPRESSED = "suppressed"


# --------------------------------------------------------------------------
# Shared tables
# --------------------------------------------------------------------------


class Objective(SQLModel, table=True):
    """Current head of a shared objective. Domain-neutral by construction."""

    __tablename__ = "objective"

    id: str = Field(primary_key=True)
    version: int = Field(default=1, index=True)
    title: str
    description: str = Field(sa_column=_text_column(nullable=False))
    bloom: Bloom = Field(default=Bloom.UNDERSTAND)
    difficulty: int = Field(default=1)
    est_minutes: int = Field(default=30)
    assessment_stems: list[str] = Field(default_factory=list, sa_column=_json_column())
    tags: list[str] = Field(default_factory=list, sa_column=_json_column())
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class ObjectiveVersion(SQLModel, table=True):
    """Immutable snapshot of an objective at one version. Domains pin these."""

    __tablename__ = "objective_version"
    __table_args__ = (UniqueConstraint("objective_id", "version", name="uq_objective_version"),)

    id: int | None = Field(default=None, primary_key=True)
    objective_id: str = Field(index=True, foreign_key="objective.id")
    version: int
    body: dict[str, Any] = Field(default_factory=dict, sa_column=_json_column())
    created_at: datetime = Field(default_factory=utcnow)


class Misconception(SQLModel, table=True):
    """A named wrong model. The Assessor detects it; the Author targets it."""

    __tablename__ = "misconception"

    id: str = Field(primary_key=True)
    wrong_model: str = Field(sa_column=_text_column(nullable=False))
    diagnostic: str | None = Field(default=None, sa_column=_text_column())
    objective_ids: list[str] = Field(default_factory=list, sa_column=_json_column())
    created_at: datetime = Field(default_factory=utcnow)


class Domain(SQLModel, table=True):
    """A manifest: an ordered view over shared objectives.

    Distinct from ``the_oracle.domains.schema.Domain``, which is the on-disk
    pack model. This is the indexed copy.
    """

    __tablename__ = "domain"

    id: str = Field(primary_key=True)
    version: int = Field(default=1)
    title: str
    description: str | None = Field(default=None, sa_column=_text_column())
    objective_pins: list[dict[str, Any]] = Field(default_factory=list, sa_column=_json_column())
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class DomainEdge(SQLModel, table=True):
    """A prerequisite edge. Belongs to the domain, never to the objective."""

    __tablename__ = "domain_edge"
    __table_args__ = (
        UniqueConstraint("domain_id", "from_objective_id", "to_objective_id", name="uq_domain_edge"),
    )

    id: int | None = Field(default=None, primary_key=True)
    domain_id: str = Field(index=True, foreign_key="domain.id")
    from_objective_id: str
    to_objective_id: str


class Module(SQLModel, table=True):
    """An ordered chunk of a domain. Resources are fetched per module, lazily."""

    __tablename__ = "module"

    id: str = Field(primary_key=True)
    domain_id: str = Field(index=True, foreign_key="domain.id", primary_key=True)
    position: int = Field(default=0)
    title: str
    goal: str | None = Field(default=None, sa_column=_text_column())
    objective_ids: list[str] = Field(default_factory=list, sa_column=_json_column())


class Resource(SQLModel, table=True):
    """Corpus entry, keyed by objective and shared across domains and learners."""

    __tablename__ = "resource"

    id: str = Field(primary_key=True)
    objective_id: str = Field(index=True)
    kind: ResourceKind = Field(default=ResourceKind.FOUND)
    url: str | None = Field(default=None, sa_column=_text_column())
    title: str | None = Field(default=None, sa_column=_text_column())
    body: str | None = Field(default=None, sa_column=_text_column())
    meta: dict[str, Any] = Field(default_factory=dict, sa_column=_json_column())
    created_at: datetime = Field(default_factory=utcnow)


class ResourceReview(SQLModel, table=True):
    """A Critic verdict on one resource."""

    __tablename__ = "resource_review"

    id: int | None = Field(default=None, primary_key=True)
    resource_id: str = Field(index=True, foreign_key="resource.id")
    verdict: ReviewVerdict = Field(default=ReviewVerdict.REJECT)
    score: float = Field(default=0.0)
    rubric: dict[str, Any] = Field(default_factory=dict, sa_column=_json_column())
    notes: str | None = Field(default=None, sa_column=_text_column())
    created_at: datetime = Field(default_factory=utcnow)


class Item(SQLModel, table=True):
    """An assessment item. Shared, reusable, tied to one objective."""

    __tablename__ = "item"

    id: str = Field(primary_key=True)
    objective_id: str = Field(index=True)
    bloom: Bloom = Field(default=Bloom.UNDERSTAND)
    difficulty: float = Field(default=0.0)
    stem: str = Field(sa_column=_text_column(nullable=False))
    answer_key: dict[str, Any] = Field(default_factory=dict, sa_column=_json_column())
    misconception_ids: list[str] = Field(default_factory=list, sa_column=_json_column())
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# Learner-scoped tables
# --------------------------------------------------------------------------


class Learner(SQLModel, table=True):
    __tablename__ = "learner"

    id: str = Field(primary_key=True)
    display_name: str | None = None
    timezone: str = Field(default="UTC")
    preferences: dict[str, Any] = Field(default_factory=dict, sa_column=_json_column())
    created_at: datetime = Field(default_factory=utcnow)


class Enrollment(SQLModel, table=True):
    """Joins a learner to a domain. One learner may study several at once."""

    __tablename__ = "enrollment"
    __table_args__ = (UniqueConstraint("learner_id", "domain_id", name="uq_enrollment"),)

    id: int | None = Field(default=None, primary_key=True)
    learner_id: str = Field(index=True)
    domain_id: str = Field(index=True)
    started_at: datetime = Field(default_factory=utcnow)
    active: bool = Field(default=True)


class Goal(SQLModel, table=True):
    __tablename__ = "goal"

    id: int | None = Field(default=None, primary_key=True)
    learner_id: str = Field(index=True)
    domain_id: str | None = Field(default=None, index=True)
    statement: str = Field(sa_column=_text_column(nullable=False))
    target_date: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)


class MasteryState(SQLModel, table=True):
    """Derived mastery for one learner on one shared objective.

    Keyed ``(learner_id, objective_id)``. There is no ``domain_id`` and there
    never will be: mastery transfers across every domain that references the
    objective.

    Every row here is derived. ``rebuild`` may drop the table and replay it.
    """

    __tablename__ = "mastery_state"

    learner_id: str = Field(primary_key=True)
    objective_id: str = Field(primary_key=True)
    p_mastery: float = Field(default=0.0)
    confidence: float = Field(default=0.0)
    observations: int = Field(default=0)
    misconception_ids: list[str] = Field(default_factory=list, sa_column=_json_column())
    last_seen_at: datetime | None = None
    next_review_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utcnow)


class StudySession(SQLModel, table=True):
    __tablename__ = "study_session"

    id: str = Field(primary_key=True)
    learner_id: str = Field(index=True)
    domain_id: str | None = Field(default=None, index=True)
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    planned_minutes: int = Field(default=25)
    ended_early: bool = Field(default=False)
    tokens_used: int = Field(default=0)


class Response(SQLModel, table=True):
    """One graded attempt. Derived from the event log, never authoritative."""

    __tablename__ = "response"

    id: int | None = Field(default=None, primary_key=True)
    learner_id: str = Field(index=True)
    objective_id: str = Field(index=True)
    item_id: str | None = Field(default=None, index=True)
    session_id: str | None = Field(default=None, index=True)
    correct: bool = Field(default=False)
    score: float = Field(default=0.0)
    latency_ms: int | None = None
    misconception_ids: list[str] = Field(default_factory=list, sa_column=_json_column())
    created_at: datetime = Field(default_factory=utcnow)


class ReviewSchedule(SQLModel, table=True):
    """Derived spacing schedule. One row per learner-objective."""

    __tablename__ = "review_schedule"

    learner_id: str = Field(primary_key=True)
    objective_id: str = Field(primary_key=True)
    due_at: datetime = Field(default_factory=utcnow)
    interval_days: float = Field(default=1.0)
    stability: float = Field(default=0.0)
    difficulty: float = Field(default=0.0)
    lapses: int = Field(default=0)


class Nudge(SQLModel, table=True):
    __tablename__ = "nudge"

    id: int | None = Field(default=None, primary_key=True)
    learner_id: str = Field(index=True)
    rung: int = Field(default=0)
    channel: str = Field(default="terminal")
    state: NudgeState = Field(default=NudgeState.QUEUED)
    body: str | None = Field(default=None, sa_column=_text_column())
    scheduled_for: datetime = Field(default_factory=utcnow)
    sent_at: datetime | None = None


class EventLog(SQLModel, table=True):
    """Append-only raw signal. Nothing in the system may update or delete a row.

    Use :class:`the_oracle.store.events.EventLog` to write and read it.
    """

    __tablename__ = "event_log"

    id: int | None = Field(default=None, primary_key=True)
    learner_id: str = Field(index=True)
    kind: str = Field(index=True)
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=_json_column())
    occurred_at: datetime = Field(default_factory=utcnow, index=True)
    recorded_at: datetime = Field(default_factory=utcnow)


class AgentCall(SQLModel, table=True):
    """Idempotency cache, keyed ``(agent, objective_id, input_hash)``.

    A retry returns the cached output instead of paying for a second call.
    """

    __tablename__ = "agent_call"
    __table_args__ = (
        UniqueConstraint("agent", "objective_id", "input_hash", name="uq_agent_call"),
    )

    id: int | None = Field(default=None, primary_key=True)
    agent: str = Field(index=True)
    objective_id: str = Field(default="", index=True)
    input_hash: str = Field(index=True)
    model: str | None = None
    output: dict[str, Any] = Field(default_factory=dict, sa_column=_json_column())
    request_tokens: int = Field(default=0)
    response_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    created_at: datetime = Field(default_factory=utcnow)


#: Tables that ``rebuild`` may drop and recompute. The event log is never here.
DERIVED_TABLES: tuple[type[SQLModel], ...] = (
    MasteryState,
    Response,
    ReviewSchedule,
)

__all__ = [
    "DERIVED_TABLES",
    "AgentCall",
    "Bloom",
    "Domain",
    "DomainEdge",
    "Enrollment",
    "EventLog",
    "Goal",
    "Item",
    "Learner",
    "MasteryState",
    "Misconception",
    "Module",
    "NudgeState",
    "Nudge",
    "Objective",
    "ObjectiveVersion",
    "Resource",
    "ResourceKind",
    "ResourceReview",
    "ReviewSchedule",
    "ReviewVerdict",
    "StudySession",
    "utcnow",
]
