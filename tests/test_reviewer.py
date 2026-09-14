"""Reviewer tests. Offline, no key, no network.

Every model call goes through the cassette harness in ``tests/cassettes``. The
cassette is replayed strictly: a miss raises and never falls through to the
network. No API key was available when this suite was written, so the responses
in ``reviewer_units.json`` are hand-written by :func:`_synthesize` below.

Regenerate the cassette after a prompt change::

    uv run python -c "from tests.test_reviewer import rebuild_cassette; rebuild_cassette()"
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import Engine

from the_oracle.agents.base import Agent, Usage
from the_oracle.agents.reviewer import (
    ACCEPT_SCORE,
    ACCURACY_FLOOR,
    ALL_REJECT_REASONS,
    HARD_REJECTS,
    LEVEL_FIT_FLOOR,
    MAX_TEXT_CHARS,
    MIN_WORDS,
    WEIGHTS,
    Review,
    ReviewRequest,
    Reviewer,
    Rubric,
    deterministic_reject,
    weighted_score,
    weights_sum,
)
from the_oracle.context import LearnerContext
from the_oracle.store.db import build_engine, create_all
from the_oracle.style import BANNED_PHRASES

from tests.cassettes.replay import cassette

CASSETTE = "reviewer_units"
LEARNER = "test-learner"


# --- fixtures --------------------------------------------------------------


def _prose(words: int, seed: str) -> str:
    """Deterministic filler that reads like a paragraph and counts like one."""
    vocab = ("the", "prior", "belief", "updates", "when", "evidence", "arrives", seed)
    return " ".join(vocab[i % len(vocab)] for i in range(words))


GOOD = ReviewRequest(
    objective_title="Apply Bayes' theorem to a two-hypothesis problem",
    objective_description="Update a prior with one piece of evidence and read the posterior.",
    bloom="apply",
    difficulty=3,
    url="https://example.org/good",
    title="Working through a Bayesian update by hand",
    text=_prose(600, "worked"),
)

WEAK = GOOD.model_copy(
    update={"url": "https://example.org/weak", "text": _prose(600, "shallow")}
)

LEVEL_MISMATCH = GOOD.model_copy(
    update={
        "url": "https://example.org/graduate",
        "title": "Measure-theoretic foundations of conditional expectation",
        "text": _prose(900, "measure"),
    }
)

PLAUSIBLE_WRONG = GOOD.model_copy(
    update={
        "url": "https://example.org/confident-but-wrong",
        "title": "Bayes made simple: just multiply the probabilities",
        "text": _prose(700, "confident"),
    }
)

LONG = GOOD.model_copy(
    update={"url": "https://example.org/long", "text": _prose(20_000, "sprawling")}
)

EMPTY = GOOD.model_copy(update={"url": "https://example.org/empty", "text": "   "})
SHORT = GOOD.model_copy(update={"url": "https://example.org/thin", "text": _prose(120, "thin")})
BROKEN = GOOD.model_copy(
    update={"url": "https://example.org/gone", "text": "", "fetch_error": "404 Not Found"}
)

#: ``url -> rubric``. The hand-written stand-in for the model.
_RUBRICS: dict[str, dict[str, Any]] = {
    "https://example.org/good": {
        "rubric": dict(
            authority=4, accuracy=5, level_fit=5, depth=4, recency=5, accessibility=5,
            effort_to_value=4,
        ),
        "notes": "This walks one update end to end, at about the level you're working at.",
        "hard_reject_reason": None,
    },
    "https://example.org/weak": {
        "rubric": dict(
            authority=3, accuracy=3, level_fit=3, depth=2, recency=4, accessibility=4,
            effort_to_value=2,
        ),
        "notes": "It mentions the update but never shows one, so you'd finish no further along.",
        "hard_reject_reason": None,
    },
    "https://example.org/graduate": {
        # Everything else is perfect. The level is two bands out.
        "rubric": dict(
            authority=5, accuracy=5, level_fit=0, depth=5, recency=5, accessibility=5,
            effort_to_value=5,
        ),
        "notes": "This is a graduate treatment; you'd spend the hour decoding notation.",
        "hard_reject_reason": None,
    },
    "https://example.org/confident-but-wrong": {
        # Reads well, scores well everywhere except the one thing that matters.
        "rubric": dict(
            authority=4, accuracy=1, level_fit=5, depth=4, recency=5, accessibility=5,
            effort_to_value=4,
        ),
        "notes": "It's clear and confident, but it multiplies the probabilities instead of "
        "normalising them, so the posterior it gets is wrong.",
        "hard_reject_reason": None,
    },
    "https://example.org/long": {
        "rubric": dict(
            authority=4, accuracy=5, level_fit=4, depth=5, recency=4, accessibility=4,
            effort_to_value=4,
        ),
        "notes": "Long, but the middle section does the worked update you need.",
        "hard_reject_reason": None,
    },
}


def _synthesize(agent_name: str, prompt: str) -> dict[str, Any]:
    """Answer a Reviewer prompt the way the model is asked to answer it."""
    if agent_name != "reviewer":
        raise KeyError(f"no offline responder for agent {agent_name!r}")
    for url, canned in _RUBRICS.items():
        if f"url: {url}\n" in prompt:
            rubric = Rubric(**canned["rubric"])
            score = weighted_score(rubric)
            return Review(
                verdict="accept" if score >= ACCEPT_SCORE else "reject",
                score=score,
                rubric=rubric,
                notes=canned["notes"],
                hard_reject_reason=canned["hard_reject_reason"],
            ).model_dump(mode="json")
    raise KeyError(f"no canned review for prompt:\n{prompt}")


def rebuild_cassette() -> Path:  # pragma: no cover - a maintenance entry point
    """Rewrite ``reviewer_units.json`` from the fixtures above."""
    from tests.cassettes.replay import CASSETTE_DIR, Cassette

    engine = build_engine("sqlite://")
    create_all(engine)
    reviewer = Reviewer(engine)
    tape = Cassette(CASSETTE_DIR / f"{CASSETTE}.json", record=False, synthesize=_synthesize)
    tape.entries = {}
    for request in (GOOD, WEAK, LEVEL_MISMATCH, PLAUSIBLE_WRONG, LONG):
        asyncio.run(tape.play(reviewer, reviewer.prompt(request)))
    tape._dirty = True
    tape.save()
    return tape.path


# --- harness ---------------------------------------------------------------


@pytest.fixture
def engine() -> Engine:
    engine = build_engine("sqlite://")
    create_all(engine)
    return engine


@pytest.fixture
def ctx() -> LearnerContext:
    return LearnerContext.for_learner(LEARNER)


class _Spy:
    """Counts model calls. Any call at all is a failure in the guard tests."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, agent: Agent[Any, Any], prompt: str) -> tuple[Any, Usage]:
        self.calls += 1
        raise AssertionError("the Reviewer paid for a model call it should have skipped")


def review(request: ReviewRequest, engine: Engine, ctx: LearnerContext) -> Review:
    """Run one review against the strict cassette."""
    with cassette(CASSETTE, write=False) as tape:
        result = asyncio.run(Reviewer(engine).run(ctx, request, objective_id="obj"))
    assert [call["source"] for call in tape.calls] == ["cassette"]
    return result.output


# --- the rubric ------------------------------------------------------------


def test_weights_sum_to_one() -> None:
    assert weights_sum() == 1.0


def test_weights_cover_every_rubric_field() -> None:
    assert set(WEIGHTS) == set(Rubric.model_fields)


def test_accuracy_and_level_fit_dominate() -> None:
    """Over half the score rides on being right and being at the right level."""
    assert WEIGHTS["accuracy"] + WEIGHTS["level_fit"] > 0.5
    assert WEIGHTS["accuracy"] == max(WEIGHTS.values())


def test_weighted_score_is_zero_to_five() -> None:
    top = Rubric(**dict.fromkeys(Rubric.model_fields, 5))
    bottom = Rubric(**dict.fromkeys(Rubric.model_fields, 0))
    assert weighted_score(top) == 5.0
    assert weighted_score(bottom) == 0.0


def test_hard_rejects_match_the_contract() -> None:
    assert HARD_REJECTS == ("dead_link", "paywall_only", "level_mismatch", "not_about_objective")
    assert set(HARD_REJECTS) <= set(ALL_REJECT_REASONS)


# --- the free gate: no model call, no tokens -------------------------------


@pytest.mark.parametrize(
    ("request_", "reason"),
    [(EMPTY, "empty_text"), (SHORT, "too_short"), (BROKEN, "fetch_error")],
)
def test_cheap_rejects_never_call_the_model(
    request_: ReviewRequest, reason: str, engine: Engine, ctx: LearnerContext
) -> None:
    spy = _Spy()
    with patch.object(Agent, "_run_llm", spy):
        result = asyncio.run(Reviewer(engine).run(ctx, request_, objective_id="obj"))
    assert spy.calls == 0
    assert result.usage.total_tokens == 0
    assert result.output.verdict == "reject"
    assert result.output.hard_reject_reason == reason
    assert result.output.score == 0.0


def test_the_word_floor_is_two_hundred() -> None:
    assert MIN_WORDS == 200
    just_under = GOOD.model_copy(update={"text": _prose(MIN_WORDS - 1, "edge")})
    just_over = GOOD.model_copy(update={"text": _prose(MIN_WORDS + 1, "edge")})
    assert deterministic_reject(just_under) is not None
    assert deterministic_reject(just_over) is None


def test_a_reported_word_count_is_trusted_over_the_body() -> None:
    thin = GOOD.model_copy(update={"word_count": 12})
    assert deterministic_reject(thin).hard_reject_reason == "too_short"


def test_cheap_reject_notes_are_written_for_the_learner() -> None:
    for request_ in (EMPTY, SHORT, BROKEN):
        notes = deterministic_reject(request_).notes
        assert notes.endswith(".")
        assert len(notes.split()) <= 25
        assert "rubric" not in notes.lower()
        assert not any(phrase in notes.lower() for phrase in BANNED_PHRASES)


# --- score to verdict ------------------------------------------------------


def _rubric(**overrides: int) -> Rubric:
    base = dict.fromkeys(Rubric.model_fields, 3)
    base.update(overrides)
    return Rubric(**base)


def test_score_maps_to_verdict_around_the_threshold(engine: Engine) -> None:
    reviewer = Reviewer(engine)
    below = reviewer.finalise(
        Review(verdict="accept", rubric=_rubric(), notes="Fine, not more.")
    )
    assert below.score < ACCEPT_SCORE and below.verdict == "reject"

    above = reviewer.finalise(
        Review(verdict="reject", rubric=_rubric(depth=5, authority=5, effort_to_value=5),
               notes="Worth the time.")
    )
    assert above.score >= ACCEPT_SCORE and above.verdict == "accept"


def test_the_model_does_not_get_the_last_word(engine: Engine) -> None:
    """A model that says accept over a failing rubric is overruled."""
    out = Reviewer(engine).finalise(
        Review(verdict="accept", score=5.0, rubric=_rubric(accuracy=0), notes="Looks good.")
    )
    assert out.verdict == "reject"
    assert out.score == 0.0


def test_a_model_invented_reject_reason_is_normalised(engine: Engine) -> None:
    out = Reviewer(engine).finalise(
        Review(verdict="reject", rubric=_rubric(), notes="No.", hard_reject_reason="vibes")
    )
    assert out.hard_reject_reason in ALL_REJECT_REASONS


def test_a_good_resource_is_accepted(engine: Engine, ctx: LearnerContext) -> None:
    out = review(GOOD, engine, ctx)
    assert out.verdict == "accept"
    assert out.score >= ACCEPT_SCORE
    assert out.hard_reject_reason is None
    assert out.notes


def test_a_shallow_resource_falls_under_the_threshold(engine: Engine, ctx: LearnerContext) -> None:
    out = review(WEAK, engine, ctx)
    assert out.verdict == "reject"
    assert out.score < ACCEPT_SCORE


# --- level fit is a gate, not a score --------------------------------------


def test_level_mismatch_beats_a_perfect_rubric(engine: Engine, ctx: LearnerContext) -> None:
    """Five out of five on everything else does not buy its way past the level."""
    out = review(LEVEL_MISMATCH, engine, ctx)
    scores = out.rubric.model_dump()
    assert scores.pop("level_fit") <= LEVEL_FIT_FLOOR
    assert set(scores.values()) == {5}
    assert out.verdict == "reject"
    assert out.hard_reject_reason == "level_mismatch"
    assert out.score == 0.0


# --- the test that matters -------------------------------------------------


def test_plausible_but_wrong_is_rejected_on_accuracy(
    engine: Engine, ctx: LearnerContext
) -> None:
    """Confident, well written, and wrong. Style must not buy a pass.

    Every other rubric field is high, and the unweighted score would clear
    ACCEPT_SCORE. The accuracy floor is what stops it.
    """
    out = review(PLAUSIBLE_WRONG, engine, ctx)
    assert out.rubric.accuracy <= ACCURACY_FLOOR
    assert weighted_score(out.rubric) >= ACCEPT_SCORE, "the floor, not the score, must catch this"
    assert out.verdict == "reject"
    assert out.hard_reject_reason == "inaccurate"
    assert "wrong" in out.notes.lower()


# --- long input ------------------------------------------------------------


def test_long_text_is_truncated_and_disclosed(engine: Engine, ctx: LearnerContext) -> None:
    reviewer = Reviewer(engine)
    prompt = reviewer.prompt(LONG)
    assert len(LONG.text) > MAX_TEXT_CHARS
    assert len(prompt) < len(LONG.text)
    assert "excerpt" in prompt

    out = review(LONG, engine, ctx)
    assert "excerpt" in out.notes.lower()


def test_a_short_page_is_sent_whole(engine: Engine) -> None:
    prompt = Reviewer(engine).prompt(GOOD)
    assert GOOD.text in prompt
    assert "excerpt" not in prompt


# --- idempotency -----------------------------------------------------------


def test_a_retry_does_not_pay_twice(engine: Engine, ctx: LearnerContext) -> None:
    reviewer = Reviewer(engine)
    with cassette(CASSETTE, write=False) as tape:
        first = asyncio.run(reviewer.run(ctx, GOOD, objective_id="obj"))
        second = asyncio.run(reviewer.run(ctx, GOOD, objective_id="obj"))
    assert len(tape.calls) == 1
    assert not first.cached and second.cached
    assert second.output.verdict == first.output.verdict


# --- reachable from the shipped path ---------------------------------------


def test_the_reviewer_is_reachable_from_the_pipeline() -> None:
    """The recurring failure of this project is a component nothing calls.

    When the corpus pipeline exists, it must import and call this Reviewer.
    """
    if importlib.util.find_spec("the_oracle.corpus.pipeline") is None:
        pytest.skip("the corpus pipeline lands with the `corpus` worker")
    from the_oracle.corpus import pipeline

    source = inspect.getsource(pipeline)
    assert "Reviewer" in source, "the pipeline does not use the Reviewer"


def test_the_cassette_is_committed_and_strict() -> None:
    from tests.cassettes.replay import CASSETTE_DIR

    path = CASSETTE_DIR / f"{CASSETTE}.json"
    assert path.is_file(), "run rebuild_cassette()"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["entries"], "the cassette is empty"
    assert all(entry["agent"] == "reviewer" for entry in body["entries"].values())
