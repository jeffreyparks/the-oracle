"""The Coach: one weekly report, the only model call in Phase 5.

The design in one line: **the model never invents a number.**

:func:`gather_week` is pure. It folds the append-only event log into
:class:`WeekFacts` and nothing else touches the arithmetic. The Coach is handed
those facts and writes prose over them. Every field it returns is then checked
against the facts; any figure that is not derivable from :class:`WeekFacts` is
rejected and the deterministic sentence takes its place. A hallucinated number
can therefore never reach the learner, even if the model produces one.

Honesty rules (PLAN.md section 8), enforced here rather than requested in a
prompt:

* A bad week gets a bad headline. Spin is replaced, not softened.
* A week with no activity says so plainly, with no scolding.
* ``encouragement`` is specific and earned, or empty. Unearned praise is
  deleted, never rewritten into something warmer.
* What is fading is reported as prominently as what improved.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar

from pydantic import BaseModel, Field
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sqlalchemy import Engine

from the_oracle.agents.base import Agent, Usage
from the_oracle.config import Task
from the_oracle.context import LearnerContext
from the_oracle.mastery import MASTERY_THRESHOLD
from the_oracle.mastery import bkt
from the_oracle.mastery.scheduler import Review, plan as review_plan
from the_oracle.store.events import EventKind, EventLog
from the_oracle.style import BANNED_PHRASES

#: A rise of at least this much in p(mastery) counts as "improved".
IMPROVEMENT_DELTA: float = 0.05
#: A fall of at least this much counts as "fading".
FADE_DELTA: float = 0.05
#: A review this many days overdue counts as fading, whatever the mastery estimate.
OVERDUE_DAYS: float = 1.0
#: Days in one reporting week.
WEEK_DAYS: int = 7
#: A streak must touch today or yesterday to still be a streak.
STREAK_GRACE_DAYS: int = 1

#: Words a headline may not use when the facts earn no praise.
SPIN_WORDS: tuple[str, ...] = (
    "great", "excellent", "amazing", "fantastic", "wonderful", "awesome",
    "brilliant", "outstanding", "superb", "crushed it", "nailed it",
    "well done", "keep it up", "proud", "on fire", "impressive",
)

ACTIVITY_KINDS: tuple[EventKind, ...] = (
    EventKind.RESPONSE_GRADED,
    EventKind.SESSION_STARTED,
    EventKind.SESSION_ENDED,
    EventKind.ITEM_PRESENTED,
)


# --- the facts -------------------------------------------------------------


class WeekFacts(BaseModel):
    """Everything true about one learner's week. Computed, never written.

    Every number the learner sees must come from this object.
    """

    learner_id: str
    sessions: int = 0
    items: int = 0
    correct: int = 0
    minutes: float = 0.0
    objectives_mastered: list[str] = Field(default_factory=list)
    objectives_improved: list[tuple[str, float, float]] = Field(default_factory=list)
    objectives_fading: list[tuple[str, float]] = Field(default_factory=list)
    due_next_week: int = 0
    streak_days: int = 0
    idle_days: float = 0.0
    weeks: int = 1
    """Window length in weeks. Additive to the frozen contract; defaults to 1."""

    @property
    def days(self) -> int:
        """Window length in days."""
        return WEEK_DAYS * self.weeks

    @property
    def wrong(self) -> int:
        return max(0, self.items - self.correct)

    @property
    def accuracy(self) -> float | None:
        """Share correct in [0, 1], or ``None`` when nothing was answered."""
        if self.items <= 0:
            return None
        return self.correct / self.items

    @property
    def has_activity(self) -> bool:
        return self.sessions > 0 or self.items > 0

    @property
    def is_empty_week(self) -> bool:
        """Nothing at all happened. The report must say so, plainly."""
        return not self.has_activity


class WeeklyReport(BaseModel):
    """The prose. One honest sentence, then four short sections."""

    headline: str
    what_moved: str = ""
    what_is_fading: str = ""
    what_is_next: str = ""
    encouragement: str = ""
    """Specific and earned, or empty. Empty beats hollow."""


# --- gather: pure, deterministic, event log only ---------------------------


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


class _Track:
    """Per-objective fold state. Scratch space for :func:`gather_week`."""

    __slots__ = ("p", "p_at_start", "reviews", "touched", "last_at")

    def __init__(self) -> None:
        self.p: float = bkt.DEFAULT_PARAMS.p_init
        self.p_at_start: float | None = None
        self.reviews: list[Review] = []
        self.touched: bool = False
        self.last_at: datetime | None = None


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _session_minutes(events: Sequence[Any], start: datetime, now: datetime) -> float:
    """Wall-clock minutes of sessions that both started and ended in the window.

    An unfinished session contributes nothing. Guessing at its length would be
    inventing a number, which is the one thing this module refuses to do.
    """
    open_by_id: dict[str, datetime] = {}
    anonymous: list[datetime] = []
    total = 0.0
    for event in events:
        at = _aware(event.occurred_at)
        if at < start or at > now:
            continue
        session_id = event.payload.get("session_id")
        if event.kind == EventKind.SESSION_STARTED:
            if isinstance(session_id, str) and session_id:
                open_by_id[session_id] = at
            else:
                anonymous.append(at)
        elif event.kind == EventKind.SESSION_ENDED:
            opened: datetime | None = None
            if isinstance(session_id, str) and session_id:
                opened = open_by_id.pop(session_id, None)
            if opened is None and anonymous:
                opened = anonymous.pop(0)
            if opened is not None and at >= opened:
                total += (at - opened).total_seconds() / 60.0
    return round(total, 1)


def _streak(days_active: set[Any], today: Any) -> int:
    """Consecutive active days ending today or yesterday. Otherwise zero."""
    if not days_active:
        return 0
    last = max(days_active)
    if (today - last).days > STREAK_GRACE_DAYS:
        return 0
    count = 0
    cursor = last
    while cursor in days_active:
        count += 1
        cursor = cursor - timedelta(days=1)
    return count


def gather_week(
    learner_id: str,
    *,
    now: datetime | None = None,
    engine: Engine | None = None,
    weeks: int = 1,
) -> WeekFacts:
    """Fold the event log into :class:`WeekFacts`. Pure, given the log and ``now``.

    Reads events only. Writes nothing. Same log and same ``now`` in, same facts
    out, forever.
    """
    weeks = max(1, int(weeks))
    now = _aware(now or datetime.now(timezone.utc))
    start = now - timedelta(days=WEEK_DAYS * weeks)

    log = EventLog(engine)
    events = [e for e in log.read(learner_id) if _aware(e.occurred_at) <= now]

    tracks: dict[str, _Track] = {}
    items = 0
    correct = 0
    sessions = 0
    active_days: set[Any] = set()
    last_activity: datetime | None = None

    for event in events:
        at = _aware(event.occurred_at)
        in_window = at >= start

        if event.kind in ACTIVITY_KINDS:
            last_activity = at if last_activity is None else max(last_activity, at)
            if in_window:
                active_days.add(at.date())
        if event.kind == EventKind.SESSION_STARTED and in_window:
            sessions += 1
        if event.kind != EventKind.RESPONSE_GRADED:
            continue

        objective_id = event.payload.get("objective_id")
        if not isinstance(objective_id, str) or not objective_id:
            continue
        track = tracks.setdefault(objective_id, _Track())
        if in_window:
            if track.p_at_start is None:
                track.p_at_start = track.p
            track.touched = True
            items += 1
            correct += int(bool(event.payload.get("correct", False)))

        is_correct = bool(event.payload.get("correct", False))
        track.p = bkt.update(track.p, is_correct)
        track.reviews.append(
            Review(correct=is_correct, at=at, difficulty=_int(event.payload.get("difficulty"), 1))
        )
        track.last_at = at

    mastered: list[str] = []
    improved: list[tuple[str, float, float]] = []
    fading: dict[str, float] = {}
    due_next_week = 0
    horizon = now + timedelta(days=WEEK_DAYS)

    for objective_id, track in sorted(tracks.items()):
        before = track.p_at_start
        after = track.p
        if track.touched and before is not None:
            if before < MASTERY_THRESHOLD <= after:
                mastered.append(objective_id)
            elif after - before >= IMPROVEMENT_DELTA:
                improved.append((objective_id, round(before, 2), round(after, 2)))
            elif before - after >= FADE_DELTA:
                fading[objective_id] = round(after, 2)

        schedule = review_plan(track.reviews, track.p)
        if schedule is None:
            continue
        due_at = _aware(schedule.due_at)
        if due_at <= horizon:
            due_next_week += 1
        overdue_days = (now - due_at).total_seconds() / 86400.0
        if overdue_days >= OVERDUE_DAYS and objective_id not in fading:
            fading[objective_id] = round(track.p, 2)

    idle_days = (
        round((now - last_activity).total_seconds() / 86400.0, 2)
        if last_activity is not None
        else float(WEEK_DAYS * weeks)
    )

    return WeekFacts(
        learner_id=learner_id,
        sessions=sessions,
        items=items,
        correct=correct,
        minutes=_session_minutes(events, start, now),
        objectives_mastered=mastered,
        objectives_improved=improved,
        objectives_fading=sorted(fading.items(), key=lambda kv: (kv[1], kv[0])),
        due_next_week=due_next_week,
        streak_days=_streak(active_days, now.date()),
        idle_days=max(0.0, idle_days),
        weeks=weeks,
    )


# --- the no-invented-numbers guard ----------------------------------------

#: A bare number, not glued to a word or an identifier.
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9_])")


def _canon(token: str) -> str | None:
    """Normalise a numeric token so ``3``, ``3.0`` and ``03`` compare equal."""
    try:
        value = Decimal(token.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    return format(value.normalize(), "f")


def _add(bag: set[str], value: float | int | None) -> None:
    """Record one figure in every rendering the report is allowed to use."""
    if value is None:
        return
    for text in (f"{float(value):.4f}", f"{float(value):.2f}", f"{float(value):.1f}",
                 f"{round(float(value))}", f"{int(float(value))}"):
        canon = _canon(text)
        if canon is not None:
            bag.add(canon)


def allowed_numbers(facts: WeekFacts) -> set[str]:
    """Every number derivable from ``facts``, in canonical form.

    Deliberately narrow. Counts, the figures themselves, their percentages, and
    the window length. Nothing else. If a sentence needs a number that is not
    in here, that number was invented.
    """
    bag: set[str] = set()
    for value in (
        facts.sessions, facts.items, facts.correct, facts.wrong, facts.minutes,
        facts.due_next_week, facts.streak_days, facts.idle_days, facts.weeks,
        facts.days, len(facts.objectives_mastered), len(facts.objectives_improved),
        len(facts.objectives_fading), MASTERY_THRESHOLD, MASTERY_THRESHOLD * 100,
        facts.minutes / 60.0 if facts.minutes else None,
    ):
        _add(bag, value)
    if facts.accuracy is not None:
        _add(bag, facts.accuracy)
        _add(bag, facts.accuracy * 100)
    for _oid, before, after in facts.objectives_improved:
        for p in (before, after, after - before):
            _add(bag, p)
            _add(bag, p * 100)
    for _oid, p in facts.objectives_fading:
        _add(bag, p)
        _add(bag, p * 100)
    bag.add("0")
    return bag


def _mask_identifiers(text: str, facts: WeekFacts) -> str:
    """Blank out objective ids, which are facts, before hunting for numbers."""
    ids = [
        *facts.objectives_mastered,
        *(oid for oid, _b, _a in facts.objectives_improved),
        *(oid for oid, _p in facts.objectives_fading),
    ]
    for objective_id in sorted(set(ids), key=len, reverse=True):
        text = text.replace(objective_id, " ")
    return text


def invented_numbers(text: str, facts: WeekFacts) -> list[str]:
    """Numbers in ``text`` that ``facts`` cannot account for. Empty means clean."""
    allowed = allowed_numbers(facts)
    found: list[str] = []
    for token in NUMBER_RE.findall(_mask_identifiers(text, facts)):
        canon = _canon(token)
        if canon is None or canon in allowed:
            continue
        found.append(token)
    return found


def banned_phrases_in(text: str) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in BANNED_PHRASES if phrase in lowered]


def earned_praise(facts: WeekFacts) -> bool:
    """Is there anything concrete to praise? If not, encouragement stays empty."""
    if facts.objectives_mastered or facts.objectives_improved:
        return True
    if facts.streak_days >= 3 and facts.items > 0:
        return True
    accuracy = facts.accuracy
    return bool(facts.items >= 5 and accuracy is not None and accuracy >= 0.8)


# --- the deterministic report ---------------------------------------------


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _pct(value: float) -> str:
    return f"{round(value * 100)}%"


def _join(names: Iterable[str]) -> str:
    names = list(names)
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def deterministic_headline(facts: WeekFacts) -> str:
    """One honest sentence, computed. The floor under the model."""
    window = _plural(facts.days, "day")
    if facts.is_empty_week:
        return f"You did nothing in the last {window}."
    if facts.items == 0:
        return f"You opened {_plural(facts.sessions, 'session')} and answered nothing."
    accuracy = facts.accuracy or 0.0
    if facts.objectives_mastered:
        return (
            f"You cleared {_plural(len(facts.objectives_mastered), 'objective')} "
            f"and answered {_pct(accuracy)} of {_plural(facts.items, 'item')} correctly."
        )
    if facts.objectives_fading and not facts.objectives_improved:
        return (
            f"A thin week: {_plural(facts.items, 'item')} at {_pct(accuracy)}, "
            f"and {_plural(len(facts.objectives_fading), 'objective')} slipping or overdue."
        )
    if facts.objectives_improved:
        return (
            f"{_plural(len(facts.objectives_improved), 'objective')} moved, "
            f"on {_plural(facts.items, 'item')} at {_pct(accuracy)}."
        )
    return (
        f"{_plural(facts.items, 'item')} at {_pct(accuracy)}, "
        f"and nothing crossed the line either way."
    )


def deterministic_report(facts: WeekFacts) -> WeeklyReport:
    """The whole report, from facts alone. No model, no key, no network."""
    window = _plural(facts.days, "day")
    if facts.is_empty_week:
        return WeeklyReport(
            headline=deterministic_headline(facts),
            what_moved=(
                f"Nothing moved. There were no sessions and no answers in the last {window}, "
                "so there is no evidence to read."
            ),
            what_is_fading=(
                f"{_plural(len(facts.objectives_fading), 'objective')} "
                f"{'is' if len(facts.objectives_fading) == 1 else 'are'} past due: "
                f"{_join(oid for oid, _p in facts.objectives_fading)}."
                if facts.objectives_fading
                else "Nothing new is fading, because nothing was measured."
            ),
            what_is_next=(
                f"{_plural(facts.due_next_week, 'objective')} "
                f"{'is' if facts.due_next_week == 1 else 'are'} due in the next "
                f"{_plural(WEEK_DAYS, 'day')}. Start with one short session."
                if facts.due_next_week
                else "Nothing is scheduled. Run a session to give the schedule something to work from."
            ),
            encouragement="",
        )

    accuracy = facts.accuracy or 0.0
    moved: list[str] = []
    if facts.objectives_mastered:
        moved.append(
            f"You cleared {_join(facts.objectives_mastered)} past "
            f"p(mastery) {MASTERY_THRESHOLD:.2f}."
        )
    for objective_id, before, after in facts.objectives_improved:
        moved.append(f"{objective_id} went from {before:.2f} to {after:.2f}.")
    if not moved:
        moved.append(
            f"Nothing crossed a threshold. You answered {_plural(facts.items, 'item')} "
            f"at {_pct(accuracy)}, which held your position rather than changing it."
        )

    if facts.objectives_fading:
        fading = "Losing ground or past due: " + "; ".join(
            f"{objective_id} at p(mastery) {p:.2f}" for objective_id, p in facts.objectives_fading
        ) + ". Review beats new material here."
    else:
        fading = "Nothing is fading. No objective lost ground and no review ran late."

    nxt = (
        f"{_plural(facts.due_next_week, 'objective')} "
        f"{'comes' if facts.due_next_week == 1 else 'come'} due in the next "
        f"{_plural(WEEK_DAYS, 'day')}."
        if facts.due_next_week
        else "Nothing is due in the next " + _plural(WEEK_DAYS, "day") + "."
    )

    encouragement = ""
    if earned_praise(facts):
        if facts.objectives_mastered:
            encouragement = (
                f"You took {_join(facts.objectives_mastered)} to mastery with evidence behind it, "
                "not by skipping the checks."
            )
        elif facts.objectives_improved:
            objective_id, before, after = facts.objectives_improved[0]
            encouragement = (
                f"You moved {objective_id} from {before:.2f} to {after:.2f} by answering, "
                "not by rereading."
            )
        elif facts.streak_days >= 3:
            encouragement = (
                f"You showed up {_plural(facts.streak_days, 'day')} in a row. "
                "The return gap is the variable that matters most, and you kept it short."
            )
        else:
            encouragement = (
                f"You held {_pct(accuracy)} across {_plural(facts.items, 'item')}. "
                "That is a real signal, not a lucky run."
            )

    return WeeklyReport(
        headline=deterministic_headline(facts),
        what_moved=" ".join(moved),
        what_is_fading=fading,
        what_is_next=nxt,
        encouragement=encouragement,
    )


# --- the agent -------------------------------------------------------------

ROLE = """\
You are the Coach. Once a week you write a short report on one learner's week.

You are given FACTS as JSON. They are the complete, verified record.

Hard rules:
- Never state a number that is not in the facts. Do not estimate, round into a
  new figure, or add up two facts into a third. If you need a number, copy it.
- The headline is one sentence and it is honest. A bad week gets a bad
  headline. Never spin a week where little happened.
- Report what is fading as prominently as what improved.
- Encouragement must name the exact thing the learner did. If the facts contain
  nothing specific worth praising, return an empty string for it. Empty is
  correct; hollow praise is a failure.
- Do not scold. State what happened and what to do next.
"""


def prompt_for(facts: WeekFacts) -> str:
    """The one prompt. Facts in JSON, then the job."""
    return (
        "FACTS (the complete record; every number you write must appear here):\n"
        f"{facts.model_dump_json(indent=2)}\n\n"
        "Write the weekly report. Copy numbers from the facts or omit them. "
        "Leave encouragement empty unless the facts name something specific to praise."
    )


class Coach(Agent[WeekFacts, WeeklyReport]):
    """Writes prose over facts it is handed. Never computes, never invents."""

    name: ClassVar[str] = "coach"
    task: ClassVar[Task] = Task.COACH
    role: ClassVar[str] = ROLE
    input_type: ClassVar[type[BaseModel]] = WeekFacts
    output_type: ClassVar[type[BaseModel]] = WeeklyReport

    async def _run(self, ctx: LearnerContext, payload: WeekFacts) -> tuple[WeeklyReport, Usage]:
        if payload.is_empty_week:
            # Nothing happened. There is nothing to write about, so we do not
            # pay a model to write it, and we do not risk it inventing a week.
            return deterministic_report(payload), Usage()
        report, usage = await self._run_llm(prompt_for(payload))
        return enforce(report, payload), usage


def enforce(report: WeeklyReport, facts: WeekFacts) -> WeeklyReport:
    """Replace any field the model got wrong with the computed one.

    Three failure modes, three fixes, all silent to the model and loud in the
    output only by being correct:

    1. An invented number -> the whole field falls back to the computed text.
    2. A banned phrase -> same fallback.
    3. Unearned praise -> ``encouragement`` is emptied, never rewritten.
    """
    floor = deterministic_report(facts)
    fields: dict[str, str] = {}
    for name in ("headline", "what_moved", "what_is_fading", "what_is_next", "encouragement"):
        text = (getattr(report, name) or "").strip()
        fallback = getattr(floor, name)
        if not text:
            fields[name] = "" if name == "encouragement" else fallback
            continue
        if invented_numbers(text, facts) or banned_phrases_in(text):
            fields[name] = fallback
            continue
        fields[name] = text

    if not earned_praise(facts):
        fields["encouragement"] = ""
    if not earned_praise(facts) and _spins(fields["headline"]):
        fields["headline"] = floor.headline
    if facts.objectives_fading and not fields["what_is_fading"].strip():
        fields["what_is_fading"] = floor.what_is_fading
    return WeeklyReport(**fields)


def _spins(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in SPIN_WORDS)


# --- render ----------------------------------------------------------------


def _facts_table(facts: WeekFacts) -> Table:
    """The numbers, straight from ``facts``. Nothing here is computed twice."""
    table = Table(box=None, pad_edge=False, show_header=False)
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("sessions", str(facts.sessions))
    table.add_row("items answered", str(facts.items))
    table.add_row("correct", str(facts.correct))
    if facts.accuracy is not None:
        table.add_row("accuracy", _pct(facts.accuracy))
    table.add_row("minutes", f"{facts.minutes:.1f}")
    table.add_row("streak", _plural(facts.streak_days, "day"))
    table.add_row("idle", f"{facts.idle_days:.1f} days")
    table.add_row("due next week", str(facts.due_next_week))
    return table


def render(report: WeeklyReport, facts: WeekFacts) -> RenderableType:
    """Rich output. Every figure in here comes from ``facts``."""
    window = _plural(facts.days, "day")
    blocks: list[RenderableType] = [
        Panel(
            Text(report.headline, style="bold"),
            title=f"last {window}",
            border_style="cyan" if facts.has_activity else "yellow",
        ),
        _facts_table(facts),
        Text(""),
    ]

    sections: list[tuple[str, str]] = [
        ("what moved", report.what_moved),
        ("what is fading", report.what_is_fading),
        ("what is next", report.what_is_next),
    ]
    if report.encouragement.strip():
        sections.append(("worth saying", report.encouragement))

    for title, body in sections:
        if not body.strip():
            continue
        blocks.append(Text(title, style="bold"))
        blocks.append(Text(body))
        blocks.append(Text(""))
    return Group(*blocks)


__all__ = [
    "Coach",
    "FADE_DELTA",
    "IMPROVEMENT_DELTA",
    "WEEK_DAYS",
    "WeekFacts",
    "WeeklyReport",
    "allowed_numbers",
    "banned_phrases_in",
    "deterministic_headline",
    "deterministic_report",
    "earned_praise",
    "enforce",
    "gather_week",
    "invented_numbers",
    "prompt_for",
    "render",
]
