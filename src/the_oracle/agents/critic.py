"""The Critic: judges the dedupe middle band, and nothing else.

When two objectives look alike but not alike enough, one question decides
whether mastery data merges or stays separate: *is this the same skill, or the
same words?* Getting it wrong in the merge direction silently corrupts a
learner's profile across every domain that shares the objective. So the
instruction is blunt about depth, audience, and about answering ``False`` when
unsure.

No subject matter lives in this file.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from the_oracle.agents.base import Agent, Usage
from the_oracle.config import Task, get_settings
from the_oracle.context import LearnerContext
from the_oracle.domains.schema import Objective

if TYPE_CHECKING:  # pragma: no cover
    from the_oracle.domains.dedupe import Match


class SamenessRequest(BaseModel):
    """Two objectives that an embedder could not separate on its own."""

    candidate: Objective
    existing: Objective


class SamenessVerdict(BaseModel):
    """The Critic's answer. ``same_skill=False`` is the safe answer."""

    same_skill: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


SAMENESS_ROLE = """\
You decide whether two learning objectives are the same skill.

This is not a wording comparison. Two objectives are the same skill only when a
learner who has mastered one has, by that fact, mastered the other.

Rules you must apply:
- The same words at a different depth are NOT the same skill. A survey-level
  treatment and an expert-level treatment of one named technique are two
  different skills, even when the titles are almost identical.
- The same words for a different audience are NOT the same skill.
- A different Bloom level, or a difficulty gap of two or more, is strong
  evidence of two different skills.
- Recognising a thing is not the same skill as deriving it, and deriving it is
  not the same skill as choosing between it and its alternatives.
- When you are unsure, answer same_skill=false. Splitting costs one duplicate
  objective. Merging wrongly corrupts the learner's mastery record in both
  directions and cannot be detected later.

Set confidence to how sure you are of the answer you gave. State the deciding
difference, or the deciding sameness, in one or two sentences."""


class SamenessJudge(Agent[SamenessRequest, SamenessVerdict]):
    """Judge one middle-band pair. Called only when similarity is ambiguous."""

    name = "sameness_judge"
    task = Task.CRITIC
    input_type = SamenessRequest
    output_type = SamenessVerdict
    role = SAMENESS_ROLE

    async def _run(
        self, ctx: LearnerContext, payload: SamenessRequest
    ) -> tuple[SamenessVerdict, Usage]:
        return await self._run_llm(self.prompt_for(payload))

    @staticmethod
    def prompt_for(payload: SamenessRequest) -> str:
        """The comparison prompt. Stable text: the cassette key depends on it."""
        return (
            "Candidate objective (newly drafted):\n"
            f"{_render(payload.candidate)}\n\n"
            "Existing objective (already in the shared library):\n"
            f"{_render(payload.existing)}\n\n"
            "Are these the same skill? Answer false if unsure."
        )


def _render(obj: Objective) -> str:
    stems = "\n".join(f"  - {s}" for s in obj.assessment_stems)
    return (
        f"title: {obj.title}\n"
        f"description: {obj.description}\n"
        f"bloom: {obj.bloom}\n"
        f"difficulty: {obj.difficulty}/5\n"
        f"estimated minutes: {obj.est_minutes}\n"
        f"tags: {', '.join(obj.tags)}\n"
        f"assessment stems:\n{stems}"
    )


def judge_fn(
    candidates: Sequence[Objective],
    judge: SamenessJudge | None = None,
    ctx: LearnerContext | None = None,
) -> Callable[["Match"], SamenessVerdict]:
    """Adapt the async agent into the sync callable :func:`dedupe.resolve` wants.

    ``candidates`` supplies the drafted objectives by title; the existing side
    is loaded from the shared library by id.
    """
    from the_oracle.domains import library

    agent = judge or SamenessJudge()
    context = ctx or LearnerContext.for_learner(get_settings().learner_id)
    by_title = {c.title: c for c in candidates}

    def run(match: "Match") -> SamenessVerdict:
        candidate = by_title[match.candidate_title]
        if match.best_id is None:
            msg = f"no best match to judge for {match.candidate_title!r}"
            raise ValueError(msg)
        existing = library.get(match.best_id)
        request = SamenessRequest(candidate=candidate, existing=existing)
        result = asyncio.run(agent.run(context, request, objective_id=match.best_id))
        return result.output

    return run
