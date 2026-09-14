"""The study session: four phases, immediate feedback, measured early exit.

This is the loop PLAN.md section 5 describes, and the point of everything the
earlier phases built.

* **Shape is fixed, length adapts.** Warm-up, new material, applied practice,
  consolidation - always in that order. The session ends early only on measured
  fatigue against the learner's own baseline, never on a constant.
* **Review is not a chore.** Due review objectives are the warm-up.
* **Feedback is immediate but never mid-attempt.** Recall and short items get
  the verdict at once; multi-step work gets its critique at the end of *that*
  attempt. The one exception is mandatory: the same misconception twice in one
  session interrupts and re-teaches, because a repeated wrong model strengthens
  with every repetition.
* **The strict gate decides what may be taught.** ``mastery.gate``, not the
  loose diagnostic gate.
* **Mastery is never written here.** The loop appends events and applies them
  through ``mastery.incremental``.

Time is spent in *budget minutes*, an estimate per attempt, not a wall clock.
Replay must be deterministic and a test must not wait 25 minutes to see whether
the practice phase respected its budget.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine

from the_oracle.agents.assessor import (
    Grade,
    GradeRequest,
    Grader,
    Item,
    ItemRequest,
    ItemWriter,
    feedback_is_immediate,
    misconceptions_for,
    next_difficulty,
)
from the_oracle.context import LearnerContext
from the_oracle.mastery import MASTERY_THRESHOLD
from the_oracle.session.fatigue import FatigueSignal, baseline_for, fatigue
from the_oracle.session.gating import blocked_by, prerequisites_met
from the_oracle.session.planner import SessionPlan, plan_session  # noqa: F401  (re-export)
from the_oracle.session.shape import (
    DEFAULT_SHAPE,  # noqa: F401  (re-export)
    Phase,
    PhaseBudget,  # noqa: F401  (re-export)
    budget_for,
)
from the_oracle.store.events import Event, EventKind, EventLog

if TYPE_CHECKING:  # pragma: no cover
    from the_oracle.domains.schema import Domain, Objective
    from the_oracle.mastery import MasteryProfile

#: Budget minutes one attempt costs, by item kind. Multi-step work costs more.
ITEM_MINUTES: dict[str, float] = {"recall": 1.0, "short": 1.5, "multi_step": 3.0}
#: Budget minutes spent putting one new objective in front of the learner.
TEACH_MINUTES: float = 4.0
#: Budget minutes spent re-teaching after a repeated misconception.
RETEACH_MINUTES: float = 2.0
#: Desirable difficulty: steer practice toward roughly this success rate.
TARGET_ACCURACY: float = 0.8

AnswerFn = Callable[[Item], str]
EventFn = Callable[[Event], None]
FeedbackFn = Callable[[Item, str, Grade, Phase], None]


@dataclass(frozen=True, slots=True)
class SessionResult:
    """What one session did, in numbers the learner can check."""

    session_id: str
    asked: int
    correct: int
    objectives_touched: list[str] = field(default_factory=list)
    mastery_before: dict[str, float] = field(default_factory=dict)
    mastery_after: dict[str, float] = field(default_factory=dict)
    ended_early: bool = False
    reason: str = ""

    @property
    def accuracy(self) -> float:
        return self.correct / self.asked if self.asked else 0.0

    def moved(self, *, epsilon: float = 1e-9) -> dict[str, float]:
        """Objectives whose mastery changed, and by how much."""
        out: dict[str, float] = {}
        for objective_id, after in self.mastery_after.items():
            delta = after - self.mastery_before.get(objective_id, after)
            if abs(delta) > epsilon:
                out[objective_id] = delta
        return out


def should_interrupt(seen_misconceptions: Sequence[str], new_id: str | None) -> bool:
    """True when this wrong model has already shown up in this session.

    A repeated wrong model strengthens with every repetition, so the second
    sighting stops the flow and re-teaches. The first sighting does not: it is
    one data point, and interrupting on it would shred the session.
    """
    return bool(new_id) and new_id in tuple(seen_misconceptions)


def _record_response(
    learner_id: str, payload: dict[str, Any], engine: Engine | None
) -> float:
    """Append the graded response and move mastery. Never a direct write.

    ``mastery.incremental.apply_response`` owns both halves: it appends
    ``response.graded`` and updates ``MasteryState``, ``Response`` and
    ``ReviewSchedule`` to exactly what a full replay would produce. Letting one
    function do both is what keeps the log and the derived rows from drifting.
    """
    from the_oracle.mastery.incremental import apply_response

    return apply_response(learner_id, payload, engine=engine)


def _record_misconception(
    learner_id: str, payload: dict[str, Any], engine: Engine | None
) -> None:
    """Append the misconception tag. It records the wrong model; ``p`` stays put."""
    from the_oracle.mastery.incremental import apply_misconception

    apply_misconception(learner_id, payload, engine=engine)


@dataclass(slots=True)
class _Run:
    """Mutable bookkeeping for one session. Not part of the public surface."""

    plan: SessionPlan
    session_id: str
    domain: "Domain"
    log: EventLog
    engine: Engine | None
    on_event: EventFn | None
    on_feedback: FeedbackFn | None
    baseline: float
    profile: "MasteryProfile"
    spent: dict[Phase, float] = field(default_factory=dict)
    asked: int = 0
    correct: int = 0
    responses: list[bool] = field(default_factory=list)
    touched: list[str] = field(default_factory=list)
    seen_misconceptions: list[str] = field(default_factory=list)
    retaught: set[str] = field(default_factory=set)
    focus: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    stopped: FatigueSignal | None = None
    asked_per_objective: dict[str, int] = field(default_factory=dict)

    def stems_for(self, objective: "Objective") -> list[str]:
        """Choose ONE source stem for the next item, rotating per objective.

        A live session asked the same question twice: the teaching item and the
        practice item both received the whole stem list, and the model picked
        the same stem both times. Practice that repeats the teaching question is
        recognition, not retrieval.

        Filtering by the returned text does not work, because the model rewrites
        the stem it was given, so it never matches the source. So the rotation
        has to be decided here, before the call: hand over exactly one stem,
        chosen by how many items this objective has already produced this
        session. Deterministic, and it also varies the idempotency key.
        """
        all_stems = list(objective.assessment_stems)
        if not all_stems:
            return []
        index = self.asked_per_objective.get(objective.id, 0)
        return [all_stems[index % len(all_stems)]]

    def mark_stem(self, objective_id: str, stem: str) -> None:
        self.asked_per_objective[objective_id] = (
            self.asked_per_objective.get(objective_id, 0) + 1
        )

    # -- plumbing ----------------------------------------------------------

    def append(self, kind: EventKind, payload: dict[str, Any]) -> Event:
        event = self.log.append(kind, self.plan.learner_id, payload)
        self.notify(event)
        return event

    def notify(self, event: Event | None) -> None:
        if event is not None and self.on_event is not None:
            self.on_event(event)

    def last(self, kind: EventKind) -> Event | None:
        """The newest event of ``kind`` for this learner.

        Used to hand ``on_event`` the events ``mastery.incremental`` appends on
        our behalf, so a front end sees one stream whoever wrote it.
        """
        events = self.log.read(self.plan.learner_id, kinds=[kind])
        return events[-1] if events else None

    def refresh(self) -> None:
        from the_oracle.mastery import profile_for

        self.profile = profile_for(self.plan.learner_id, self.engine)

    def room(self, phase: Phase, cost: float) -> bool:
        """True when ``phase`` can still afford ``cost`` budget minutes."""
        return self.spent.get(phase, 0.0) + cost <= budget_for(self.plan.shape, phase) + 1e-9

    def charge(self, phase: Phase, cost: float) -> None:
        self.spent[phase] = self.spent.get(phase, 0.0) + cost

    def touch(self, objective_id: str) -> None:
        if objective_id not in self.touched:
            self.touched.append(objective_id)


def _difficulty_for(run: _Run, objective_id: str, *, ease: int = 0) -> int:
    objective = run.domain.objective(objective_id)
    base = next_difficulty(objective.difficulty, run.asked, run.correct, TARGET_ACCURACY)
    return max(1, min(5, base + ease))


def _cost_of(kind: str) -> float:
    return ITEM_MINUTES.get(kind, ITEM_MINUTES["short"])


async def _attempt(
    run: _Run,
    phase: Phase,
    objective_id: str,
    *,
    writer: ItemWriter,
    grader: Grader,
    ctx: LearnerContext,
    ask: AnswerFn,
    ease: int = 0,
) -> Grade | None:
    """Write one item, put it, grade it, log it, fold it into mastery.

    Returns ``None`` when the phase has no budget left for this attempt.
    """
    objective = run.domain.objective(objective_id)
    difficulty = _difficulty_for(run, objective_id, ease=ease)
    request = ItemRequest(
        objective_id=objective_id,
        bloom=objective.bloom,
        difficulty=difficulty,
        stems=run.stems_for(objective),
    )
    item = (await writer.run(ctx, request, objective_id=objective_id)).output
    if not run.room(phase, _cost_of(item.kind)):
        return None
    run.charge(phase, _cost_of(item.kind))
    run.mark_stem(objective_id, item.stem)

    run.append(
        EventKind.ITEM_PRESENTED,
        {
            "session_id": run.session_id,
            "item_id": item.id,
            "objective_id": objective_id,
            "domain_id": run.plan.domain_id,
            "phase": str(phase),
            "kind": item.kind,
            "bloom": item.bloom,
            "difficulty": item.difficulty,
            "immediate_feedback": feedback_is_immediate(item.kind),
        },
    )

    started = time.monotonic()
    answer = ask(item)
    seconds = round(time.monotonic() - started, 3)

    grade = (
        await grader.run(
            ctx,
            GradeRequest(
                item=item,
                answer=answer,
                misconceptions=misconceptions_for(run.domain, objective_id),
            ),
            objective_id=objective_id,
        )
    ).output

    payload = {
        "objective_id": objective_id,
        "item_id": item.id,
        "correct": bool(grade.correct),
        "difficulty": item.difficulty,
        "bloom": item.bloom,
        "seconds": seconds,
        "misconception_id": grade.misconception_id,
        "session_id": run.session_id,
        "phase": str(phase),
    }
    _record_response(run.plan.learner_id, payload, run.engine)
    run.notify(run.last(EventKind.RESPONSE_GRADED))
    if grade.misconception_id:
        _record_misconception(
            run.plan.learner_id,
            {
                "objective_id": objective_id,
                "misconception_id": grade.misconception_id,
                "item_id": item.id,
                "session_id": run.session_id,
                "repeat": should_interrupt(run.seen_misconceptions, grade.misconception_id),
            },
            run.engine,
        )
        run.notify(run.last(EventKind.MISCONCEPTION_DETECTED))
    run.refresh()

    run.asked += 1
    run.correct += int(bool(grade.correct))
    run.responses.append(bool(grade.correct))
    run.touch(objective_id)

    # Feedback lands here: the attempt is over, whatever the item kind.
    if run.on_feedback is not None:
        run.on_feedback(item, answer, grade, phase)

    if grade.misconception_id:
        if should_interrupt(run.seen_misconceptions, grade.misconception_id):
            _interrupt(run, phase, objective_id, grade.misconception_id)
        else:
            run.seen_misconceptions.append(grade.misconception_id)

    signal = fatigue(run.responses, baseline=run.baseline)
    if signal.stop:
        run.stopped = signal
    return grade


def _interrupt(run: _Run, phase: Phase, objective_id: str, misconception_id: str) -> None:
    """Stop the flow and re-teach: this wrong model has shown up twice."""
    if misconception_id in run.retaught:
        return
    run.retaught.add(misconception_id)
    run.charge(phase, RETEACH_MINUTES)
    run.notes.append(
        f"{misconception_id} showed up twice on {objective_id}, so I stopped and "
        "re-taught it. A wrong model gets stronger every time you use it."
    )


def _drifted(run: _Run, objective_id: str) -> bool:
    """True when a focus objective is no longer safe to teach.

    A prerequisite can decay mid-session: the warm-up is real evidence, and a
    run of wrong answers on a prerequisite drops it back under the bar. We drop
    the objective and say so rather than teaching blind.
    """
    if prerequisites_met(run.domain, objective_id, run.profile):
        return False
    missing = blocked_by(run.domain, objective_id, run.profile)
    run.notes.append(
        f"Dropped {objective_id} mid-session: "
        f"{', '.join(missing) or 'a prerequisite'} slipped back under the bar. "
        "Teaching it now would be teaching blind."
    )
    run.focus = [oid for oid in run.focus if oid != objective_id]
    return True


async def _warmup(run: _Run, **calls: Any) -> None:
    """Retrieval practice on what is due. Never a separate chore."""
    for objective_id in run.plan.warmup:
        if run.stopped is not None:
            return
        if not run.room(Phase.WARMUP, _cost_of("short")):
            return
        await _attempt(run, Phase.WARMUP, objective_id, ease=-1, **calls)


async def _new(run: _Run, **calls: Any) -> None:
    """One new objective, rarely two, checked against the strict gate first."""
    for objective_id in list(run.focus):
        if run.stopped is not None:
            return
        if _drifted(run, objective_id):
            continue
        if not run.room(Phase.NEW, TEACH_MINUTES + _cost_of("short")):
            return
        run.charge(Phase.NEW, TEACH_MINUTES)
        run.touch(objective_id)
        while run.room(Phase.NEW, _cost_of("short")) and run.stopped is None:
            if await _attempt(run, Phase.NEW, objective_id, **calls) is None:
                break


async def _practice(run: _Run, **calls: Any) -> None:
    """Use it, do not just read it. Focus first, then interleave the warm-up."""
    order = [*run.focus, *run.plan.warmup]
    if not order:
        return
    index = 0
    while run.stopped is None and run.room(Phase.PRACTICE, _cost_of("short")):
        objective_id = order[index % len(order)]
        index += 1
        if index > len(order) * 8:  # pragma: no cover - belt and braces
            return
        if objective_id in run.focus and _drifted(run, objective_id):
            order = [oid for oid in order if oid != objective_id]
            if not order:
                return
            continue
        if await _attempt(run, Phase.PRACTICE, objective_id, **calls) is None:
            return


def _consolidate(run: _Run) -> None:
    """Summarise and schedule. The phase that makes the next session possible."""
    from the_oracle.session.planner import _graded_by_objective
    from the_oracle.mastery.scheduler import plan as review_plan

    history = _graded_by_objective(run.plan.learner_id, run.engine)
    for objective_id in run.touched:
        schedule = review_plan(history.get(objective_id, []), run.profile.p(objective_id))
        if schedule is None:
            continue
        run.append(
            EventKind.REVIEW_SCHEDULED,
            {
                "objective_id": objective_id,
                "session_id": run.session_id,
                "due_at": schedule.due_at.isoformat(),
                "interval_days": round(schedule.interval_days, 4),
                "p_mastery": round(run.profile.p(objective_id), 6),
            },
        )


def run_session(
    plan: SessionPlan,
    *,
    answer_fn: AnswerFn | None = None,
    on_event: EventFn | None = None,
    engine: Engine | None = None,
    on_feedback: FeedbackFn | None = None,
) -> SessionResult:
    """Run one session start to finish and return what moved.

    ``answer_fn(item) -> str`` supplies the learner's answer, so a synthetic
    learner drives the whole loop in a test. ``on_event`` sees every event as
    it is appended. ``on_feedback`` is optional prose for a human front end; it
    fires once per finished attempt, never during one.
    """
    return asyncio.run(
        run_session_async(
            plan,
            answer_fn=answer_fn,
            on_event=on_event,
            engine=engine,
            on_feedback=on_feedback,
        )
    )


async def run_session_async(
    plan: SessionPlan,
    *,
    answer_fn: AnswerFn | None = None,
    on_event: EventFn | None = None,
    engine: Engine | None = None,
    on_feedback: FeedbackFn | None = None,
) -> SessionResult:
    """Async body of :func:`run_session`."""
    from the_oracle.domains.registry import load_domain
    from the_oracle.mastery import profile_for

    domain = load_domain(plan.domain_id)
    ctx = LearnerContext.for_learner(plan.learner_id)
    session_id = f"ses-{uuid.uuid4().hex[:12]}"
    run = _Run(
        plan=plan,
        session_id=session_id,
        domain=domain,
        log=EventLog(engine),
        engine=engine,
        on_event=on_event,
        on_feedback=on_feedback,
        baseline=baseline_for(plan.learner_id, engine=engine),
        profile=profile_for(plan.learner_id, engine),
        focus=list(plan.focus),
    )
    before = {objective_id: run.profile.p(objective_id) for objective_id in plan.objectives}

    run.append(
        EventKind.SESSION_STARTED,
        {
            "session_id": session_id,
            "domain_id": plan.domain_id,
            "mode": "study",
            "minutes": plan.minutes,
            "warmup": list(plan.warmup),
            "focus": list(plan.focus),
            "baseline": round(run.baseline, 4),
        },
    )

    calls: dict[str, Any] = {
        "writer": ItemWriter(engine),
        "grader": Grader(engine),
        "ctx": ctx,
        "ask": answer_fn or _console_answer,
    }
    await _warmup(run, **calls)
    if run.stopped is None:
        await _new(run, **calls)
    if run.stopped is None:
        await _practice(run, **calls)
    _consolidate(run)

    ended_early = run.stopped is not None
    reason = run.stopped.reason if run.stopped is not None else _closing_line(run)
    if run.notes:
        reason = " ".join([reason, *run.notes])

    after = {
        objective_id: run.profile.p(objective_id)
        for objective_id in [*before, *run.touched]
    }
    run.append(
        EventKind.SESSION_ENDED,
        {
            "session_id": session_id,
            "domain_id": plan.domain_id,
            "mode": "study",
            "asked": run.asked,
            "correct": run.correct,
            "objectives": list(run.touched),
            "ended_early": ended_early,
            "reason": reason,
        },
    )
    return SessionResult(
        session_id=session_id,
        asked=run.asked,
        correct=run.correct,
        objectives_touched=list(run.touched),
        mastery_before=before,
        mastery_after=after,
        ended_early=ended_early,
        reason=reason,
    )


def _closing_line(run: _Run) -> str:
    """The plain close-out for a session that ran its full shape."""
    if run.asked == 0:
        return "Nothing was due and nothing was teachable, so there was no session to run."
    mastered = [
        objective_id
        for objective_id in run.touched
        if run.profile.p(objective_id) >= MASTERY_THRESHOLD
    ]
    count = len(run.touched)
    noun = "objective" if count == 1 else "objectives"
    line = f"You worked {count} {noun} and got {run.correct} of {run.asked} right."
    if mastered:
        line += f" {', '.join(mastered)} cleared the mastery bar."
    return line


def _console_answer(item: Item) -> str:  # pragma: no cover - interactive fallback
    """Default answer source: plain stdin. The CLI supplies a better one."""
    print(item.stem)
    return input("> ")


__all__ = [
    "ITEM_MINUTES",
    "RETEACH_MINUTES",
    "TARGET_ACCURACY",
    "TEACH_MINUTES",
    "AnswerFn",
    "SessionResult",
    "run_session",
    "run_session_async",
    "should_interrupt",
]
