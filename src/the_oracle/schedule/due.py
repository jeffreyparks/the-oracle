"""Due work, computed on demand. No daemon, no queue, nothing to restart.

PLAN.md section 9 is the whole design here: due work is a *query*, not a
process. Nothing in this package starts a thread, opens a socket, or holds a
loop. The CLI asks what is owed when it starts, and one crontab line asks once
a day. If this file ever grows a background worker, the design has been lost.

Two entry points:

* :func:`due_summary` - a pure read: what is owed, how long the learner has
  been away, when the next thing lands, and which packs it falls in.
* :func:`run_due_checks` - the one thing a cron line calls. It summarises,
  asks :func:`the_oracle.nudge.decide`, records the ``Nudge`` row, appends
  ``NUDGE_SENT``, and delivers through the channel. Idempotent per day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import Engine
from sqlmodel import Session, select

from the_oracle.store import models
from the_oracle.store.events import EventKind, EventLog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from the_oracle.nudge import NudgeDecision

#: Event kinds that count as real study activity. A nudge is not activity:
#: being reminded is not the same as showing up, so ``NUDGE_SENT`` is absent
#: from this tuple on purpose, and a test pins that.
ACTIVITY_KINDS: tuple[str, ...] = (
    str(EventKind.SESSION_STARTED),
    str(EventKind.SESSION_ENDED),
    str(EventKind.ITEM_PRESENTED),
    str(EventKind.RESPONSE_GRADED),
)

#: Kinds that start the clock when a learner has never studied at all. Someone
#: who enrolled a week ago and never began is idle, not brand new.
ENROLMENT_KINDS: tuple[str, ...] = (
    str(EventKind.ENROLLED),
    str(EventKind.GOAL_SET),
    str(EventKind.LEARNER_CREATED),
)


@dataclass(frozen=True, slots=True)
class DueSummary:
    """What one learner owes right now.

    ``domains`` counts *due* objectives per domain pack, so a due objective
    that appears in two packs is counted under both. Objectives that no
    installed pack references are still in ``due_objectives``; they simply
    have no pack to be listed under.
    """

    learner_id: str
    due_objectives: int
    idle_days: float
    next_due_at: datetime | None
    domains: dict[str, int] = field(default_factory=dict)

    @property
    def anything_owed(self) -> bool:
        return self.due_objectives > 0


def _aware(moment: datetime | None) -> datetime | None:
    if moment is None:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _engine(engine: Engine | None) -> Engine:
    if engine is not None:
        return engine
    from the_oracle.store.db import create_all, get_engine

    return create_all(get_engine())


def _now(now: datetime | None) -> datetime:
    return _aware(now) or datetime.now(UTC)


def _schedule_rows(learner_id: str, engine: Engine) -> list[models.MasteryState]:
    statement = select(models.MasteryState).where(models.MasteryState.learner_id == learner_id)
    with Session(engine) as session:
        rows = list(session.exec(statement))
    return [row for row in rows if row.next_review_at is not None]


def _domains_by_objective() -> dict[str, list[str]]:
    """Objective id -> packs that reference it. Best effort, never raises.

    A broken pack must not hide due work in the others, so every failure here
    is swallowed and the objective simply lists no domain.
    """
    try:
        from the_oracle.domains import registry
    except Exception:  # pragma: no cover - domains is always installed
        return {}

    mapping: dict[str, list[str]] = {}
    try:
        domain_ids = registry.list_domains()
    except Exception:
        return mapping
    for domain_id in sorted(domain_ids):
        try:
            domain = registry.load_domain(domain_id)
        except Exception:
            continue
        for ref in getattr(domain, "objectives", []):
            mapping.setdefault(getattr(ref, "id", str(ref)), []).append(domain_id)
    return mapping


def last_activity_at(learner_id: str, *, engine: Engine | None = None) -> datetime | None:
    """When the learner last did real work. ``None`` when they never have.

    Read from the event log, never from the ``Nudge`` table. Idle time is
    measured from study, not from the last time we spoke.
    """
    engine = _engine(engine)
    log = EventLog(engine)
    events = [e for e in log.read(learner_id, kinds=ACTIVITY_KINDS)]
    if events:
        return _aware(events[-1].occurred_at)
    starts = log.read(learner_id, kinds=ENROLMENT_KINDS)
    if starts:
        return _aware(starts[0].occurred_at)
    return None


def idle_days_for(learner_id: str, *, now: datetime | None = None, engine: Engine | None = None) -> float:
    """Days since the last real study activity. ``0.0`` with no history."""
    moment = _now(now)
    last = last_activity_at(learner_id, engine=_engine(engine))
    if last is None:
        return 0.0
    return max(0.0, (moment - last).total_seconds() / 86400.0)


def due_summary(
    learner_id: str,
    *,
    now: datetime | None = None,
    engine: Engine | None = None,
) -> DueSummary:
    """Compute what is owed, on demand. Pure read, no writes, no process."""
    engine = _engine(engine)
    moment = _now(now)

    rows = _schedule_rows(learner_id, engine)
    due = [row for row in rows if (_aware(row.next_review_at) or moment) <= moment]
    later = sorted(
        (d for d in (_aware(r.next_review_at) for r in rows) if d is not None and d > moment)
    )

    by_objective = _domains_by_objective()
    domains: dict[str, int] = {}
    for row in due:
        for domain_id in by_objective.get(row.objective_id, []):
            domains[domain_id] = domains.get(domain_id, 0) + 1

    return DueSummary(
        learner_id=learner_id,
        due_objectives=len(due),
        idle_days=idle_days_for(learner_id, now=moment, engine=engine),
        next_due_at=later[0] if later else None,
        domains=dict(sorted(domains.items())),
    )


# -- the daily check ---------------------------------------------------------


def _learner(learner_id: str, engine: Engine) -> models.Learner | None:
    with Session(engine) as session:
        return session.get(models.Learner, learner_id)


def _zone(learner: models.Learner | None) -> Any:
    name = (getattr(learner, "timezone", None) or "UTC").strip() or "UTC"
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return UTC


def _quiet_hours(learner: models.Learner | None) -> tuple[int, int] | None:
    raw = (getattr(learner, "preferences", None) or {}).get("quiet_hours")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            return (int(raw[0]), int(raw[1]))
        except (TypeError, ValueError):
            return None
    return None


def _nudges_off(learner: models.Learner | None) -> bool:
    raw = (getattr(learner, "preferences", None) or {}).get("nudges")
    return isinstance(raw, str) and raw.strip().lower() == "off"


def last_nudge(learner_id: str, *, engine: Engine | None = None) -> models.Nudge | None:
    """The most recent recorded nudge, sent or not."""
    engine = _engine(engine)
    statement = (
        select(models.Nudge)
        .where(models.Nudge.learner_id == learner_id)
        .order_by(models.Nudge.scheduled_for.desc(), models.Nudge.id.desc())  # type: ignore[union-attr]
        .limit(1)
    )
    with Session(engine) as session:
        row = session.exec(statement).first()
        if row is not None:
            session.expunge(row)  # the ladder gets values, not a live handle
        return row


def _sent_today(learner_id: str, engine: Engine, moment: datetime, today: date) -> bool:
    statement = select(models.Nudge).where(
        models.Nudge.learner_id == learner_id,
        models.Nudge.state == models.NudgeState.SENT,
        models.Nudge.sent_at >= moment - timedelta(days=2),  # type: ignore[operator]
    )
    with Session(engine) as session:
        rows = list(session.exec(statement))
    tz = moment.tzinfo or UTC
    return any((_aware(row.sent_at) or moment).astimezone(tz).date() == today for row in rows)


def run_due_checks(
    learner_id: str,
    *,
    now: datetime | None = None,
    engine: Engine | None = None,
    send: bool = True,
) -> "NudgeDecision":
    """Compute due work, decide, record, and deliver. The cron entry point.

    Idempotent per calendar day in the learner's own timezone: call it five
    times and at most one nudge goes out. Two guards stand behind that. The
    ladder itself refuses a second nudge the same day, and this function
    refuses to deliver when a ``Nudge`` row was already sent today. Belt and
    braces, because a cron line that fires twice must not shout twice.
    """
    from the_oracle.nudge import NudgeDecision, Rung, decide, get_channel

    engine = _engine(engine)
    moment = _now(now)
    summary = due_summary(learner_id, now=moment, engine=engine)
    learner = _learner(learner_id, engine)
    today = moment.astimezone(_zone(learner)).date()

    if _nudges_off(learner):
        # One-word opt-out. No argument, no row, no event.
        return NudgeDecision(
            rung=Rung.QUIET,
            reason="the learner turned nudges off",
            body="",
            send=False,
            suppressed_reason="nudges off",
        )

    previous = last_nudge(learner_id, engine=engine)
    decision = decide(
        idle_days=summary.idle_days,
        due_count=summary.due_objectives,
        last_nudge=previous,
        now=moment,
        quiet_hours=_quiet_hours(learner),
    )

    if not decision.send or not send:
        return decision

    if _sent_today(learner_id, engine, moment, today):
        return NudgeDecision(
            rung=decision.rung,
            reason=decision.reason,
            body=decision.body,
            send=False,
            suppressed_reason="already nudged today",
        )

    channel = get_channel((getattr(learner, "preferences", None) or {}).get("channel", "terminal"))
    row = models.Nudge(
        learner_id=learner_id,
        rung=int(decision.rung),
        channel=channel.name,
        state=models.NudgeState.SENT,
        body=decision.body,
        scheduled_for=moment,
        sent_at=moment,
    )
    from the_oracle.store.db import session_scope

    with session_scope(engine) as session:
        session.add(row)
        session.flush()
        session.refresh(row)
        # Detach before the commit expires it: the channel and the caller get a
        # plain object with its values already loaded, not a live ORM handle.
        session.expunge(row)

    EventLog(engine).append(
        EventKind.NUDGE_SENT,
        learner_id,
        {
            "nudge_id": row.id,
            "rung": int(decision.rung),
            "reason": decision.reason,
            "channel": channel.name,
            "due_objectives": summary.due_objectives,
            "idle_days": round(summary.idle_days, 3),
        },
        occurred_at=moment,
    )
    channel.send(row)
    return decision


__all__ = [
    "ACTIVITY_KINDS",
    "DueSummary",
    "due_summary",
    "idle_days_for",
    "last_activity_at",
    "last_nudge",
    "run_due_checks",
]
