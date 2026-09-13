"""The Assessor: writes diagnostic items, grades answers, runs the diagnostic.

Two agents and one loop.

* :class:`ItemWriter` turns an objective into a single assessment item.
* :class:`Grader` scores a free-text answer and names the misconception the
  answer reveals, if any.
* :func:`run_diagnostic` drives the adaptive loop: pick the objective whose
  ability estimate is least certain, ask, grade, log, repeat.

The loop writes events and nothing else. Mastery is derived by the reducers in
:mod:`the_oracle.mastery`, never written here. See PLAN.md sections 4, 6, 9.

No subject matter lives in this file. Every concrete word comes from the
domain pack.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from the_oracle.agents.base import Agent, Usage
from the_oracle.config import Task
from the_oracle.context import LearnerContext
from the_oracle.domains.registry import load_domain
from the_oracle.store.events import EventKind, EventLog

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy import Engine

    from the_oracle.domains.schema import Domain
    from the_oracle.mastery import MasteryProfile

ItemKind = Literal["recall", "short", "multi_step"]

#: Bloom level -> the item type that actually tests it (PLAN.md section 6).
BLOOM_TO_KIND: dict[str, ItemKind] = {
    "remember": "recall",
    "understand": "short",
    "apply": "short",
    "analyze": "multi_step",
    "evaluate": "multi_step",
    "create": "multi_step",
}

#: Beyond this distance from 0.5 an estimate is settled: the objective reads as
#: clearly mastered or clearly absent, and one more item would barely move it.
CERTAIN_BAND = 0.34

#: Grades at or above this confidence are trusted without a second look.
MIN_CONFIDENCE = 0.5


def kind_for_bloom(bloom: str) -> ItemKind:
    """The item type a Bloom level demands."""
    return BLOOM_TO_KIND.get(bloom, "short")


def feedback_is_immediate(kind: str) -> bool:
    """True when feedback lands the moment the answer is in.

    Recall and short items: instant. Multi-step work gets its critique at the
    end of that attempt, never mid-derivation (PLAN.md section 5).
    """
    return kind in ("recall", "short")


# ---------------------------------------------------------------------------
# Typed agent payloads
# ---------------------------------------------------------------------------


class Item(BaseModel):
    """One assessment item, ready to put in front of a learner."""

    id: str
    objective_id: str
    stem: str
    kind: ItemKind
    difficulty: int
    bloom: str
    expected: str
    misconception_probes: list[str] = Field(default_factory=list)


class ItemRequest(BaseModel):
    """What the :class:`ItemWriter` needs to write one item."""

    objective_id: str
    bloom: str
    difficulty: int
    stems: list[str] = Field(default_factory=list)


class Grade(BaseModel):
    """The verdict on one answer."""

    correct: bool
    confidence: float
    misconception_id: str | None = None
    feedback: str
    reasoning: str


class GradeRequest(BaseModel):
    """What the :class:`Grader` needs to score one answer."""

    item: Item
    answer: str
    misconceptions: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class ItemWriter(Agent[ItemRequest, Item]):
    """Write a single diagnostic item for one objective."""

    name = "item_writer"
    task = Task.ASSESS
    input_type = ItemRequest
    output_type = Item
    role = (
        "You write one assessment item at a time. The item must test the "
        "stated objective at the stated Bloom level and difficulty, and it "
        "must be answerable in free text without multiple choice. Reuse the "
        "supplied assessment stems where they fit. Put the ideal answer in "
        "`expected`, short and checkable."
    )

    async def _run(self, ctx: LearnerContext, payload: ItemRequest) -> tuple[Item, Usage]:
        item, usage = await self._run_llm(self.prompt(payload))
        return self.normalise(item, payload), usage

    def prompt(self, payload: ItemRequest) -> str:
        """The user-side prompt. Deterministic, so cassettes key on it."""
        stems = "\n".join(f"- {s}" for s in payload.stems) or "- (none supplied)"
        return (
            f"objective_id: {payload.objective_id}\n"
            f"bloom: {payload.bloom}\n"
            f"difficulty: {payload.difficulty} (1 easiest, 5 hardest)\n"
            f"item kind: {kind_for_bloom(payload.bloom)}\n"
            f"assessment stems to draw on:\n{stems}\n\n"
            "Write exactly one item. Set objective_id, bloom and difficulty to "
            "the values above. Set kind to the item kind above."
        )

    def normalise(self, item: Item, payload: ItemRequest) -> Item:
        """Force the model's answer back onto the requested facts."""
        return item.model_copy(
            update={
                "id": item.id or f"item-{uuid.uuid4().hex[:12]}",
                "objective_id": payload.objective_id,
                "bloom": payload.bloom,
                "difficulty": payload.difficulty,
                "kind": kind_for_bloom(payload.bloom),
            }
        )


class Grader(Agent[GradeRequest, Grade]):
    """Score a free-text answer and name the wrong model behind it."""

    name = "grader"
    task = Task.ASSESS
    input_type = GradeRequest
    output_type = Grade
    role = (
        "You grade one free-text answer. Decide correct or not, plainly. If "
        "the answer matches one of the listed misconceptions, set "
        "misconception_id to that id and nothing else; otherwise leave it "
        "null. `feedback` is for the learner: what the answer shows and what "
        "to do next. `reasoning` is for the system: why you graded this way."
    )

    async def _run(self, ctx: LearnerContext, payload: GradeRequest) -> tuple[Grade, Usage]:
        grade, usage = await self._run_llm(self.prompt(payload))
        return self.normalise(grade, payload), usage

    def prompt(self, payload: GradeRequest) -> str:
        """The user-side prompt. Deterministic, so cassettes key on it."""
        known = (
            "\n".join(
                f"- {m.get('id')}: {m.get('wrong_model', '')}" for m in payload.misconceptions
            )
            or "- (none known for this objective)"
        )
        return (
            f"item_id: {payload.item.id}\n"
            f"objective_id: {payload.item.objective_id}\n"
            f"kind: {payload.item.kind}\n"
            f"stem: {payload.item.stem}\n"
            f"expected answer: {payload.item.expected}\n"
            f"known misconceptions:\n{known}\n\n"
            f"learner answer:\n{payload.answer}"
        )

    def normalise(self, grade: Grade, payload: GradeRequest) -> Grade:
        """Drop a misconception id the domain pack does not know."""
        allowed = {str(m.get("id")) for m in payload.misconceptions}
        update: dict[str, Any] = {"confidence": max(0.0, min(1.0, grade.confidence))}
        if grade.misconception_id is not None and grade.misconception_id not in allowed:
            update["misconception_id"] = None
        return grade.model_copy(update=update)


# ---------------------------------------------------------------------------
# Adaptive selection
# ---------------------------------------------------------------------------


def prerequisites_ready(
    domain: "Domain",
    profile: "MasteryProfile",
    objective_id: str,
    passed: "frozenset[str] | set[str]" = frozenset(),
) -> bool:
    """May the diagnostic ask about this objective yet?

    Mastery learning blocks teaching past an unmet prerequisite. A diagnostic
    is not teaching: it is looking for the frontier, and it has no mastery
    estimates to start from. So a prerequisite counts as cleared when it is
    mastered *or* when the learner has just answered it correctly. A wrong
    answer closes that branch, which is the protection that matters here.
    """
    if profile.known_prerequisites_met(domain, objective_id):
        return True
    prerequisites = domain.prerequisites(objective_id)
    return bool(prerequisites) and all(pre in passed for pre in prerequisites)


def next_objective(
    domain: "Domain",
    profile: "MasteryProfile",
    asked: set[str],
    *,
    passed: "frozenset[str] | set[str]" = frozenset(),
    band: float = CERTAIN_BAND,
) -> str | None:
    """The most informative objective left to ask about.

    Nearest to ``p = 0.5`` wins, because that is where one observation moves
    the estimate most. Prerequisites must be ready, and an objective is never
    asked twice. Returns ``None`` when nothing is worth asking.
    """
    best: tuple[float, int, str] | None = None
    for position, objective_id in enumerate(domain.teaching_order()):
        if objective_id in asked:
            continue
        if not prerequisites_ready(domain, profile, objective_id, passed):
            continue
        distance = abs(profile.p(objective_id) - 0.5)
        if distance > band:
            continue
        candidate = (distance, position, objective_id)
        if best is None or candidate < best:
            best = candidate
    return None if best is None else best[2]


def next_difficulty(base: int, asked: int, correct: int, target: float) -> int:
    """Nudge difficulty so the learner lands near ``target`` accuracy.

    Below target, ease off. Above it, push. PLAN.md section 6 wants ~70%.
    """
    if asked == 0:
        return max(1, min(5, base))
    accuracy = correct / asked
    step = 1 if accuracy > target + 0.1 else (-1 if accuracy < target - 0.1 else 0)
    return max(1, min(5, base + step))


def misconceptions_for(domain: "Domain", objective_id: str) -> list[dict[str, Any]]:
    """Every misconception the pack ties to this objective, as plain dicts."""
    return [
        {"id": m.id, "wrong_model": m.wrong_model, "diagnostic": m.diagnostic}
        for m in domain.misconceptions
        if objective_id in m.objectives
    ]


# ---------------------------------------------------------------------------
# The diagnostic loop
# ---------------------------------------------------------------------------

AnswerFn = Callable[[Item], str]
GradedFn = Callable[[Item, str, Grade], None]


def run_diagnostic(
    domain_id: str,
    learner_id: str,
    *,
    max_items: int = 12,
    target_accuracy: float = 0.7,
    answer_fn: AnswerFn | None = None,
    engine: "Engine | None" = None,
    on_graded: GradedFn | None = None,
) -> "MasteryProfile":
    """Run the adaptive diagnostic and return the rebuilt mastery profile.

    ``answer_fn(item) -> str`` supplies the learner's answer, so a synthetic
    learner can drive the whole loop in a test. ``on_graded`` is the feedback
    hook: it fires once per item, after that attempt is finished, never
    during it.

    Every item appends ``ITEM_PRESENTED`` and ``RESPONSE_GRADED``. Mastery is
    read back through the reducers; this function never writes it.
    """
    return asyncio.run(
        run_diagnostic_async(
            domain_id,
            learner_id,
            max_items=max_items,
            target_accuracy=target_accuracy,
            answer_fn=answer_fn,
            engine=engine,
            on_graded=on_graded,
        )
    )


async def run_diagnostic_async(
    domain_id: str,
    learner_id: str,
    *,
    max_items: int = 12,
    target_accuracy: float = 0.7,
    answer_fn: AnswerFn | None = None,
    engine: "Engine | None" = None,
    on_graded: GradedFn | None = None,
) -> "MasteryProfile":
    """Async body of :func:`run_diagnostic`."""
    from the_oracle.mastery import profile_for  # lazy: owned by another module
    from the_oracle.store.rebuild import rebuild

    domain = load_domain(domain_id)
    ctx = LearnerContext.for_learner(learner_id)
    log = EventLog(engine)
    writer = ItemWriter(engine)
    grader = Grader(engine)
    ask = answer_fn or _console_answer

    log.append(EventKind.SESSION_STARTED, learner_id, {"domain_id": domain_id, "mode": "diagnostic"})

    asked: set[str] = set()
    passed: set[str] = set()
    correct_count = 0
    for _ in range(max_items):
        # Re-derive mastery from the log before choosing. The loop reads the
        # reducers' output; it never computes or writes mastery itself.
        rebuild(learner_id, engine)
        profile = profile_for(learner_id, engine)
        objective_id = next_objective(domain, profile, asked, passed=passed)
        if objective_id is None:
            break
        asked.add(objective_id)
        objective = domain.objective(objective_id)
        difficulty = next_difficulty(
            objective.difficulty, len(asked) - 1, correct_count, target_accuracy
        )
        request = ItemRequest(
            objective_id=objective_id,
            bloom=objective.bloom,
            difficulty=difficulty,
            stems=list(objective.assessment_stems),
        )
        item = (await writer.run(ctx, request, objective_id=objective_id)).output
        log.append(
            EventKind.ITEM_PRESENTED,
            learner_id,
            {
                "item_id": item.id,
                "objective_id": objective_id,
                "domain_id": domain_id,
                "kind": item.kind,
                "bloom": item.bloom,
                "difficulty": item.difficulty,
            },
        )

        started = time.monotonic()
        answer = ask(item)
        seconds = round(time.monotonic() - started, 3)

        grade_request = GradeRequest(
            item=item, answer=answer, misconceptions=misconceptions_for(domain, objective_id)
        )
        grade = (await grader.run(ctx, grade_request, objective_id=objective_id)).output
        correct_count += int(grade.correct)
        if grade.correct:
            passed.add(objective_id)

        log.append(
            EventKind.RESPONSE_GRADED,
            learner_id,
            {
                "objective_id": objective_id,
                "item_id": item.id,
                "correct": grade.correct,
                "difficulty": item.difficulty,
                "bloom": item.bloom,
                "seconds": seconds,
                "misconception_id": grade.misconception_id,
            },
        )
        if grade.misconception_id:
            log.append(
                EventKind.MISCONCEPTION_DETECTED,
                learner_id,
                {
                    "misconception_id": grade.misconception_id,
                    "objective_id": objective_id,
                    "item_id": item.id,
                },
            )
        if on_graded is not None:
            on_graded(item, answer, grade)

    rebuild(learner_id, engine)
    log.append(
        EventKind.SESSION_ENDED,
        learner_id,
        {"domain_id": domain_id, "mode": "diagnostic", "items": len(asked)},
    )
    return profile_for(learner_id, engine)


def _console_answer(item: Item) -> str:  # pragma: no cover - interactive fallback
    """Default answer source: plain stdin. The CLI supplies a better one."""
    print(item.stem)
    return input("> ")
