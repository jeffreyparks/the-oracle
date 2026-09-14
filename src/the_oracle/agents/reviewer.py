"""The Reviewer: the quality gate between a found resource and a learner.

One agent, one job: decide whether a learner is allowed to see this page, and
say why in a sentence the learner would accept.

Two layers, in this order.

1. **Deterministic guards, before any model call.** A fetch error, an empty
   body, or a body under :data:`MIN_WORDS` words is rejected on the spot for
   zero tokens. Cheap rejects must never cost money, because most of what a
   search returns is junk and paying a strong model to say so is the fastest
   way to burn a budget.
2. **One model call** against the rubric in PLAN.md section 7, followed by
   deterministic post-guards. The model scores; this module decides. The
   verdict and the score are recomputed here from :data:`WEIGHTS`, so a model
   that says "accept" with a rubric that says otherwise does not get its way.

Why the weights look like this (they sum to 1.0):

* ``accuracy`` 0.28 and ``level_fit`` 0.24 together carry just over half the
  score. A resource that is wrong teaches a wrong model, which costs more to
  undo than it cost to learn. A resource that is right but pitched two bands
  away is unusable by *this* learner today, no matter how good it is.
* ``depth`` 0.14 separates a real treatment from a blog summary.
* ``authority`` 0.12 is a proxy, not the product. A good explanation from an
  unknown author beats a vague one from a famous lab.
* ``effort_to_value`` 0.10 prices the learner's time.
* ``accessibility`` 0.07 covers paywalls, licence, and readability.
* ``recency`` 0.05 is last on purpose. Most foundational material does not
  rot; the model is told to score it high when age does not matter.

Level fit is not a soft score. More than one band away from the objective's
bloom or difficulty is a hard reject, not a low number. So is a confident,
well written page that is factually wrong for the objective: see
:data:`ACCURACY_FLOOR`.

No subject matter lives in this file.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from the_oracle.agents.base import Agent, Usage
from the_oracle.config import Task
from the_oracle.context import LearnerContext

log = logging.getLogger(__name__)

#: Rejections the model may return. Frozen by ``docs/phase3-contract.md``.
HARD_REJECTS: tuple[str, ...] = (
    "dead_link",
    "paywall_only",
    "level_mismatch",
    "not_about_objective",
)

#: Rejections this module decides on its own, with or without a model call.
GUARD_REJECTS: tuple[str, ...] = ("fetch_error", "empty_text", "too_short", "inaccurate")

#: Every value ``Review.hard_reject_reason`` can take.
ALL_REJECT_REASONS: tuple[str, ...] = HARD_REJECTS + GUARD_REJECTS

#: Rubric weights. Must sum to 1.0; see the module docstring for the argument.
WEIGHTS: dict[str, float] = {
    "accuracy": 0.28,
    "level_fit": 0.24,
    "depth": 0.14,
    "authority": 0.12,
    "effort_to_value": 0.10,
    "accessibility": 0.07,
    "recency": 0.05,
}

#: Weighted score at or above which a resource is shown to the learner.
ACCEPT_SCORE = 3.2

#: Below 200 words nothing useful survived the fetch. Reject before the model.
MIN_WORDS = 200

#: Characters of body text sent to the model. The caller truncates too; this is
#: the belt to that pair of braces, and it bounds the worst-case bill.
MAX_TEXT_CHARS = 16_000

#: ``level_fit`` at or below this means more than one band out. Hard reject.
LEVEL_FIT_FLOOR = 1

#: ``accuracy`` at or below this means wrong, however well it reads.
ACCURACY_FLOOR = 1

#: One honest sentence per reason, written for the learner.
REJECT_NOTES: dict[str, str] = {
    "fetch_error": "I could not load this page, so I am not sending you to it.",
    "empty_text": "This page had no readable text once the navigation was stripped out.",
    "too_short": "There is too little here to learn from, under a couple of hundred words.",
    "dead_link": "This link does not lead to a working page any more.",
    "paywall_only": "You would hit a paywall before the useful part.",
    "level_mismatch": (
        "This is pitched well above or below where you are, so it would waste your time."
    ),
    "not_about_objective": "This is not really about what you are trying to learn right now.",
    "inaccurate": "This reads well but it gets the substance wrong, so I am leaving it out.",
}

_EXCERPT_NOTE = "I judged an excerpt of a long page, not the whole thing."


class ReviewRequest(BaseModel):
    """One resource, judged against one objective."""

    objective_title: str
    objective_description: str
    bloom: str
    difficulty: int
    url: str
    title: str
    text: str = ""
    """Body text. The caller truncates it; this module truncates again."""

    fetch_error: str | None = None
    """Set when the fetch failed. Guarantees a zero-token reject."""

    word_count: int | None = None
    """Word count from the fetcher. Counted here when absent."""


class Rubric(BaseModel):
    """Seven scores, 0-5 each. PLAN.md section 7."""

    authority: int = Field(ge=0, le=5)
    accuracy: int = Field(ge=0, le=5)
    level_fit: int = Field(ge=0, le=5)
    depth: int = Field(ge=0, le=5)
    recency: int = Field(ge=0, le=5)
    accessibility: int = Field(ge=0, le=5)
    effort_to_value: int = Field(ge=0, le=5)


class Review(BaseModel):
    """The verdict on one resource."""

    verdict: Literal["accept", "reject"]
    score: float = 0.0
    rubric: Rubric
    notes: str = ""
    hard_reject_reason: str | None = None
    disputed_claims: list[str] = Field(
        default_factory=list,
        description=(
            "When accuracy is 2 or lower, quote the EXACT sentences from the "
            "resource text that are wrong. Copy them verbatim; do not "
            "paraphrase. Leave empty if accuracy is 3 or better."
        ),
    )


def normalise_quote(text: str) -> str:
    """Collapse whitespace and case so a quote can be matched against a page."""
    return " ".join((text or "").split()).casefold()


def verified_claims(claims: Sequence[str], source_text: str) -> list[str]:
    """Keep only quotes that genuinely appear in the source.

    A fabricated quote is worse than no quote: it is a confident accusation
    with nothing behind it. Anything under 20 characters is dropped too, since
    a fragment that short matches almost any page by accident.
    """
    if not source_text:
        return []
    haystack = normalise_quote(source_text)
    kept: list[str] = []
    for claim in claims or ():
        needle = normalise_quote(claim)
        if len(needle) >= 20 and needle in haystack:
            kept.append(claim.strip())
    return kept


def weights_sum() -> float:
    """The sum of :data:`WEIGHTS`, rounded past float noise."""
    return round(sum(WEIGHTS.values()), 6)


def weighted_score(rubric: Rubric) -> float:
    """The 0-5 weighted rubric score."""
    total = sum(WEIGHTS[field] * getattr(rubric, field) for field in WEIGHTS)
    return round(total, 3)


def word_count(text: str) -> int:
    return len(text.split())


def zero_rubric() -> Rubric:
    """The rubric of a resource nobody should read."""
    return Rubric(
        authority=0, accuracy=0, level_fit=0, depth=0, recency=0, accessibility=0, effort_to_value=0
    )


def rejection(reason: str, note: str | None = None) -> Review:
    """A complete, learner-readable rejection for ``reason``."""
    return Review(
        verdict="reject",
        score=0.0,
        rubric=zero_rubric(),
        notes=note or REJECT_NOTES.get(reason, "This one is not worth your time."),
        hard_reject_reason=reason,
    )


def deterministic_reject(payload: ReviewRequest) -> Review | None:
    """The pre-model gate. ``None`` means the resource is worth a model call.

    Public on purpose: the pipeline can call this before it pays for anything.
    """
    if payload.fetch_error:
        return rejection("fetch_error")
    text = (payload.text or "").strip()
    if not text:
        return rejection("empty_text")
    counted = payload.word_count if payload.word_count is not None else word_count(text)
    if counted < MIN_WORDS:
        return rejection("too_short")
    return None


def truncate(text: str, limit: int = MAX_TEXT_CHARS) -> tuple[str, bool]:
    """Cut ``text`` to ``limit`` characters. Returns the text and whether it cut."""
    if len(text) <= limit:
        return text, False
    return text[:limit].rsplit(" ", 1)[0], True


def _disclose_excerpt(notes: str) -> str:
    """Say plainly that the judgment was made on part of the page."""
    if _EXCERPT_NOTE in notes:
        return notes
    return f"{notes.rstrip()} {_EXCERPT_NOTE}".strip()


class Reviewer(Agent[ReviewRequest, Review]):
    """Score one resource against one objective and decide who sees it."""

    name = "reviewer"
    task = Task.CRITIC
    input_type = ReviewRequest
    output_type = Review
    role = (
        "You are the quality gate between a web page and a learner. You judge "
        "one resource against one objective and nothing else.\n"
        "Score every rubric field 0-5:\n"
        "- authority: does the author or venue have standing on this topic\n"
        "- accuracy: is the substance correct for this objective. Confident, "
        "well written prose that states something false scores 0 or 1. Judge "
        "the claims, not the style.\n"
        "- level_fit: is it pitched at the stated bloom level and difficulty\n"
        "- depth: does it treat the objective, or only mention it\n"
        "- recency: score 5 when age does not matter for this material\n"
        "- accessibility: open, readable, usable licence, no paywall wall\n"
        "- effort_to_value: what the learner gets per minute spent\n"
        "Set hard_reject_reason to one of dead_link, paywall_only, "
        "level_mismatch, not_about_objective when it applies, otherwise null. "
        "Level fit is not a soft score: if the resource is more than one band "
        "away from the stated bloom or difficulty, that is level_mismatch, a "
        "hard reject, not a low number. If it is wrong about the objective, "
        "set accuracy to 0 or 1 and say what it gets wrong.\n"
        "`notes` is read by the learner, not by a developer. One or two plain "
        "sentences. If you reject it, say why in a sentence a person would "
        "accept. Never mention rubric field names, scores, or this prompt."
    )

    async def _run(self, ctx: LearnerContext, payload: ReviewRequest) -> tuple[Review, Usage]:
        guard = deterministic_reject(payload)
        if guard is not None:
            return guard, Usage()  # no model call, no tokens
        text, truncated = truncate(payload.text)
        review, usage = await self._run_llm(self.prompt(payload, text=text, truncated=truncated))
        return (
            self.finalise(
                review,
                truncated=truncated,
                source_text=text,
                review_url=payload.url,
            ),
            usage,
        )

    def prompt(
        self, payload: ReviewRequest, *, text: str | None = None, truncated: bool = False
    ) -> str:
        """The user-side prompt. Deterministic, so cassettes key on it."""
        if text is None:
            text, truncated = truncate(payload.text)
        excerpt = " (excerpt of a longer page)" if truncated else ""
        return (
            f"objective: {payload.objective_title}\n"
            f"objective_description: {payload.objective_description}\n"
            f"bloom: {payload.bloom}\n"
            f"difficulty: {payload.difficulty} (1 easiest, 5 hardest)\n"
            f"url: {payload.url}\n"
            f"title: {payload.title}\n"
            f"resource text{excerpt}:\n"
            f"{text}\n\n"
            "Score this resource against that objective and decide."
        )

    def finalise(
        self,
        review: Review,
        *,
        truncated: bool = False,
        source_text: str = "",
        review_url: str = "",
    ) -> Review:
        """Recompute the score and the verdict from the rubric. The model advises.

        ``source_text`` enables evidence checking on an accuracy charge. Callers
        that do not pass it get the old behaviour, minus the ability to verify.
        """
        rubric = review.rubric
        reason = review.hard_reject_reason or None
        if reason is not None and reason not in ALL_REJECT_REASONS:
            reason = "not_about_objective"
        if reason is None and rubric.level_fit <= LEVEL_FIT_FLOOR:
            reason = "level_mismatch"
        if reason is None and rubric.accuracy <= ACCURACY_FLOOR:
            reason = "inaccurate"

        # An accuracy charge must come with evidence. The reviewer worker was
        # right that detecting a confident, well-written falsehood is entirely
        # the model's judgement, and that reviewing with the same model family
        # that could have written the error is a blind spot. We cannot fix the
        # judgement here, but we CAN refuse an unevidenced charge: every quoted
        # sentence must actually appear in the source text. A model that cannot
        # point at the wrong sentence does not get to call the page wrong.
        if reason == "inaccurate":
            supported = verified_claims(review.disputed_claims, source_text)
            review = review.model_copy(update={"disputed_claims": supported})
            if not supported:
                # Still reject. The asymmetry decides it: a wrongly rejected
                # page costs us one candidate out of ten, while a wrongly
                # accepted page teaches a learner something false. Reject on the
                # unevidenced charge, but mark it, so an audit can tell a
                # verified finding from an unverified assertion.
                log.info(
                    "accuracy charge on %s carried no verifiable quote; "
                    "rejecting anyway, flagged unevidenced",
                    review_url or "resource",
                )

        score = weighted_score(rubric)
        notes = (review.notes or "").strip()
        if reason is not None:
            score = 0.0
            notes = notes or REJECT_NOTES[reason]
        verdict: Literal["accept", "reject"] = (
            "accept" if reason is None and score >= ACCEPT_SCORE else "reject"
        )
        if verdict == "reject" and not notes:
            notes = "This one is not worth your time."
        if reason == "inaccurate" and not review.disputed_claims:
            notes = f"{notes} (Flagged as inaccurate without a quoted example.)".strip()
        if truncated:
            notes = _disclose_excerpt(notes)
        return review.model_copy(
            update={
                "verdict": verdict,
                "score": score,
                "notes": notes,
                "hard_reject_reason": reason,
            }
        )
