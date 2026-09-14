"""The Author: writes a full lesson when the web has nothing good enough.

The most expensive agent in the system. Its output is cached in the shared
corpus and reused by every later learner, so quality per call beats speed
(PLAN.md section 7).

A lesson is not an explainer. It is explainer prose, two or more worked
examples, three to six graded exercises with answer keys and wrong-answer
feedback, runnable code when the objective is computational, and a short
summary for spaced review.

Two rules are enforced in code, not only in the prompt:

* **Cite or abstain.** A citation survives only if its url was supplied in
  ``accepted_sources``. Anything else is stripped and logged. With no accepted
  source at all the lesson says so plainly instead of inventing a reference.
* **Misconceptions land.** Every supplied misconception is reachable from at
  least one exercise's ``wrong_answer_feedback``.

No subject matter lives in this file. Every concrete word comes from the
domain pack and the accepted sources.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from the_oracle.agents.base import Agent, Usage
from the_oracle.config import Task
from the_oracle.context import LearnerContext

log = logging.getLogger(__name__)

ExerciseKind = Literal["recall", "computation", "diagnosis", "build_and_defend"]

#: Bloom level -> the exercise that actually trains it. Same ladder as the
#: Assessor's item kinds (PLAN.md section 6), one rung wider because a lesson
#: exercise may ask the learner to build something.
BLOOM_TO_EXERCISE: dict[str, ExerciseKind] = {
    "remember": "recall",
    "understand": "recall",
    "apply": "computation",
    "analyze": "diagnosis",
    "evaluate": "diagnosis",
    "create": "build_and_defend",
}

BLOOM_LEVELS: tuple[str, ...] = tuple(BLOOM_TO_EXERCISE)

MIN_WORKED_EXAMPLES = 2
MIN_EXERCISES = 3
MAX_EXERCISES = 6
MAX_SUMMARY_WORDS = 80

#: What the lesson says when nothing trustworthy backs it up.
ABSTENTION_NOTE = (
    "No accepted source was available for this objective, so nothing here is "
    "cited. Treat the specifics as unverified and check them against a text "
    "you trust before you rely on them."
)

#: Process words, not subject matter. A computational objective asks the
#: learner to work a number or run something, so the lesson owes them code.
_COMPUTATIONAL_MARKERS: tuple[str, ...] = (
    "calculate",
    "compute",
    "computation",
    "numeric",
    "numerical",
    "formula",
    "equation",
    "algorithm",
    "implement",
    "code",
    "program",
    "simulate",
    "simulation",
    "estimate",
    "derive",
    "plot",
    "solve",
    "quantify",
)


def exercise_kind(bloom: str) -> ExerciseKind:
    """The exercise type a Bloom level demands."""
    return BLOOM_TO_EXERCISE.get(bloom, "recall")


def normalise_url(url: str) -> str:
    """Compare urls the way a human would: no scheme, no trailing slash."""
    cleaned = url.strip().lower()
    cleaned = re.sub(r"^https?://", "", cleaned)
    cleaned = re.sub(r"^www\.", "", cleaned)
    return cleaned.rstrip("/")


def word_count(text: str) -> int:
    return len(text.split())


class LessonRequest(BaseModel):
    """Everything the Author needs to write one lesson."""

    objective_id: str
    title: str
    description: str
    bloom: str
    difficulty: int
    est_minutes: int
    assessment_stems: list[str] = Field(default_factory=list)
    #: ``{id, wrong_model, diagnostic}``
    misconceptions: list[dict[str, Any]] = Field(default_factory=list)
    #: ``{url, title, note}``. May be empty: that is the abstention case.
    accepted_sources: list[dict[str, Any]] = Field(default_factory=list)


class Exercise(BaseModel):
    """One graded exercise with its answer key and its wrong-answer reads."""

    prompt: str
    answer: str
    #: wrong answer -> what holding it reveals about the learner's model.
    wrong_answer_feedback: dict[str, str] = Field(default_factory=dict)
    bloom: str


class Lesson(BaseModel):
    """A full lesson, cached once and reused by every later learner."""

    explainer: str
    worked_examples: list[str] = Field(min_length=MIN_WORKED_EXAMPLES)
    exercises: list[Exercise] = Field(min_length=MIN_EXERCISES, max_length=MAX_EXERCISES)
    code: str | None = None
    review_summary: str
    #: urls actually used. Only urls supplied in ``accepted_sources`` survive.
    citations: list[str] = Field(default_factory=list)

    @field_validator("review_summary")
    @classmethod
    def _summary_fits_a_review_card(cls, value: str) -> str:
        if word_count(value) > MAX_SUMMARY_WORDS:
            raise ValueError(
                f"review_summary is {word_count(value)} words; "
                f"the spaced-review card holds {MAX_SUMMARY_WORDS}"
            )
        return value


def is_computational(payload: LessonRequest) -> bool:
    """Does this objective owe the learner runnable code?"""
    haystack = " ".join(
        [payload.title, payload.description, *payload.assessment_stems]
    ).lower()
    if any(marker in haystack for marker in _COMPUTATIONAL_MARKERS):
        return True
    return False


class Author(Agent[LessonRequest, Lesson]):
    """Write a full lesson for one objective."""

    name = "author"
    task = Task.AUTHOR
    input_type = LessonRequest
    output_type = Lesson
    role = (
        "You write one complete lesson for one objective. A lesson is not an "
        "explainer: it is explainer prose, at least two worked examples that "
        "show the reasoning step by step, three to six graded exercises with "
        "an answer key and wrong-answer feedback, runnable code when the "
        "objective is computational, and a review summary of eighty words or "
        "fewer for spaced repetition.\n"
        "Teach to the stated Bloom level and difficulty, and size the lesson "
        "to the stated minutes. Every exercise carries the Bloom level it "
        "trains and the exercise type required for that level.\n"
        "wrong_answer_feedback maps a plausible wrong answer to what holding "
        "it reveals about the learner's model. It is diagnostic, never "
        "consoling. Where a misconception is supplied, at least one exercise "
        "must be answerable in the wrong way that misconception predicts, and "
        "its wrong_answer_feedback must name what that wrong answer reveals.\n"
        "CITE OR ABSTAIN. You may cite only the urls listed under accepted "
        "sources. Never invent a reference, an author, a title, or a date. "
        "When no accepted source supports a specific claim, say so plainly in "
        "the explainer and mark the claim as unverified. An abstention is a "
        "correct answer; a fabricated citation is not."
    )

    async def _run(self, ctx: LearnerContext, payload: LessonRequest) -> tuple[Lesson, Usage]:
        lesson, usage = await self._run_llm(self.prompt(payload))
        return self.normalise(lesson, payload), usage

    # -- prompt ------------------------------------------------------------

    def prompt(self, payload: LessonRequest) -> str:
        """The user-side prompt. Deterministic, so cassettes key on it."""
        kind = exercise_kind(payload.bloom)
        stems = "\n".join(f"- {s}" for s in payload.assessment_stems) or "- (none supplied)"
        misconceptions = (
            "\n".join(
                f"- {m.get('id')}: wrong model: {m.get('wrong_model', '')}"
                f" | diagnostic: {m.get('diagnostic', '')}"
                for m in payload.misconceptions
            )
            or "- (none known for this objective)"
        )
        sources = (
            "\n".join(
                f"- {s.get('url')} | {s.get('title', '')} | {s.get('note', '')}"
                for s in payload.accepted_sources
            )
            or "- (none accepted; you have no source to cite)"
        )
        computational = is_computational(payload)
        code_rule = (
            "This objective is computational. Include a short runnable Python "
            "snippet in `code` that a learner can paste and run, with the "
            "expected output in a comment."
            if computational
            else "This objective is not computational. Leave `code` null."
        )
        citation_rule = (
            "Cite only from the accepted sources above, by url, in `citations`."
            if payload.accepted_sources
            else (
                "There are no accepted sources. Leave `citations` empty, and "
                "state plainly in the explainer that nothing here is cited and "
                "the specifics are unverified. Do not invent a source."
            )
        )
        return (
            f"objective_id: {payload.objective_id}\n"
            f"title: {payload.title}\n"
            f"description: {payload.description}\n"
            f"bloom: {payload.bloom}\n"
            f"difficulty: {payload.difficulty} (1 easiest, 5 hardest)\n"
            f"est_minutes: {payload.est_minutes}\n"
            f"exercise type required by this bloom level: {kind}\n"
            f"assessment stems to align with:\n{stems}\n"
            f"known misconceptions:\n{misconceptions}\n"
            f"accepted sources:\n{sources}\n\n"
            f"{code_rule}\n"
            f"{citation_rule}\n"
            f"Write {MIN_WORKED_EXAMPLES} or more worked examples and between "
            f"{MIN_EXERCISES} and {MAX_EXERCISES} exercises. Keep "
            f"review_summary to {MAX_SUMMARY_WORDS} words or fewer."
        )

    # -- post-conditions ---------------------------------------------------

    def normalise(self, lesson: Lesson, payload: LessonRequest) -> Lesson:
        """Force the model's answer back onto the rules we can check."""
        citations = self.filter_citations(lesson.citations, payload)
        exercises = self.wire_misconceptions(
            [self.fix_bloom(exercise, payload) for exercise in lesson.exercises[:MAX_EXERCISES]],
            payload,
        )
        explainer = self.enforce_abstention(lesson.explainer, citations, payload)
        return lesson.model_copy(
            update={
                "citations": citations,
                "exercises": exercises,
                "explainer": explainer,
                "code": self.enforce_code(lesson.code, payload),
                "review_summary": self.trim_summary(lesson.review_summary, payload),
            }
        )

    def filter_citations(self, citations: list[str], payload: LessonRequest) -> list[str]:
        """Drop every url that was not supplied. This is cite-or-abstain."""
        allowed = {
            normalise_url(str(s.get("url", ""))): str(s.get("url", "")).strip()
            for s in payload.accepted_sources
            if s.get("url")
        }
        kept: list[str] = []
        for url in citations:
            supplied = allowed.get(normalise_url(url))
            if supplied is None:
                log.warning(
                    "author: stripped a citation not in accepted_sources "
                    "(objective=%s url=%s)",
                    payload.objective_id,
                    url,
                )
                continue
            if supplied not in kept:
                kept.append(supplied)
        return kept

    def enforce_abstention(
        self, explainer: str, citations: list[str], payload: LessonRequest
    ) -> str:
        """With nothing left to cite, the lesson has to say so."""
        if citations:
            return explainer
        if "unverified" in explainer.lower() or "no accepted source" in explainer.lower():
            return explainer
        log.warning(
            "author: no surviving citation; adding the abstention note (objective=%s)",
            payload.objective_id,
        )
        return f"{explainer.rstrip()}\n\n{ABSTENTION_NOTE}"

    def fix_bloom(self, exercise: Exercise, payload: LessonRequest) -> Exercise:
        """An exercise trains a real Bloom level or the objective's own."""
        if exercise.bloom in BLOOM_LEVELS:
            return exercise
        return exercise.model_copy(update={"bloom": payload.bloom})

    def wire_misconceptions(
        self, exercises: list[Exercise], payload: LessonRequest
    ) -> list[Exercise]:
        """Every supplied misconception must be exposable by some exercise."""
        if not exercises:
            return exercises
        wired = list(exercises)
        for index, misconception in enumerate(payload.misconceptions):
            wrong_model = str(misconception.get("wrong_model", "")).strip()
            diagnostic = str(misconception.get("diagnostic", "")).strip()
            identifier = str(misconception.get("id", "")).strip()
            if not wrong_model:
                continue
            if self.covers(wired, misconception):
                continue
            target = index % len(wired)
            feedback = dict(wired[target].wrong_answer_feedback)
            feedback[wrong_model] = (
                diagnostic or f"This answer holds the wrong model recorded as {identifier}."
            )
            wired[target] = wired[target].model_copy(update={"wrong_answer_feedback": feedback})
            log.warning(
                "author: misconception %s was not exposed by any exercise; "
                "wired it into exercise %d (objective=%s)",
                identifier or "(unnamed)",
                target,
                payload.objective_id,
            )
        return wired

    @staticmethod
    def covers(exercises: list[Exercise], misconception: dict[str, Any]) -> bool:
        """Can some exercise already catch this wrong model?"""
        identifier = str(misconception.get("id", "")).strip().lower()
        wrong_model = str(misconception.get("wrong_model", "")).strip().lower()
        for exercise in exercises:
            for key, value in exercise.wrong_answer_feedback.items():
                blob = f"{key}\n{value}".lower()
                if wrong_model and wrong_model in blob:
                    return True
                if identifier and identifier in blob:
                    return True
        return False

    def enforce_code(self, code: str | None, payload: LessonRequest) -> str | None:
        """Code belongs to computational objectives and nowhere else."""
        wanted = is_computational(payload)
        if not wanted:
            if code:
                log.warning(
                    "author: dropped code from a non-computational objective (objective=%s)",
                    payload.objective_id,
                )
            return None
        if not code or not code.strip():
            log.warning(
                "author: computational objective came back without code (objective=%s)",
                payload.objective_id,
            )
            return None
        return code

    def trim_summary(self, summary: str, payload: LessonRequest) -> str:
        """The review card is a card. Anything longer gets cut."""
        words = summary.split()
        if len(words) <= MAX_SUMMARY_WORDS:
            return summary
        log.warning(
            "author: review_summary ran to %d words; trimming to %d (objective=%s)",
            len(words),
            MAX_SUMMARY_WORDS,
            payload.objective_id,
        )
        return " ".join(words[:MAX_SUMMARY_WORDS]).rstrip(",;:") + "."
