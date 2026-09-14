"""Author tests. Offline, free, deterministic.

Every LLM call runs through the cassette harness in ``tests/cassettes``. The
cassette ``author_units.json`` was written by :func:`synthesize` below, which
answers an Author prompt in the same schema a real model would. Replay is
strict: a miss raises rather than reaching for the network.

Regenerate the cassette with::

    uv run python -m tests.test_author
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import Engine

from the_oracle.agents.author import (
    ABSTENTION_NOTE,
    MAX_EXERCISES,
    MAX_SUMMARY_WORDS,
    MIN_EXERCISES,
    MIN_WORKED_EXAMPLES,
    Author,
    Exercise,
    Lesson,
    LessonRequest,
    exercise_kind,
    is_computational,
    normalise_url,
)
from the_oracle.config import reset_settings_cache
from the_oracle.context import LearnerContext
from the_oracle.store.db import build_engine, create_all

from tests.cassettes.replay import cassette

CASSETTE = "author_units"
LEARNER = "test-learner"
FABRICATED = "https://invented.example.org/paper-that-does-not-exist"


# --- fixtures --------------------------------------------------------------


@pytest.fixture(autouse=True)
def settings_home(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ORACLE_LEARNER_ID", LEARNER)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def engine() -> Engine:
    engine = build_engine("sqlite://")
    create_all(engine)
    return engine


@pytest.fixture
def ctx() -> LearnerContext:
    return LearnerContext.for_learner(LEARNER)


def run(coro: Any) -> Any:
    """Drive one coroutine to completion without a running loop."""
    import asyncio

    return asyncio.run(coro)


# --- the cases the cassette was built from --------------------------------

SOURCES = [
    {
        "url": "https://stat.example.edu/bayes-update",
        "title": "Updating a belief with evidence",
        "note": "worked numeric example, undergraduate level",
    },
    {
        "url": "https://docs.example.org/probability/basics",
        "title": "Probability basics",
        "note": "definitions and notation",
    },
]

BASE_RATE_MISCONCEPTION = {
    "id": "ignores_base_rate",
    "wrong_model": "The test is 95% accurate, so a positive result means a 95% chance of disease.",
    "diagnostic": "You read the likelihood as the posterior and dropped the prior entirely.",
}

PRIOR_MISCONCEPTION = {
    "id": "prior_is_a_guess",
    "wrong_model": "A prior is just a guess, so it does not matter what you pick.",
    "diagnostic": "You treat the prior as decoration rather than as a claim the data has to move.",
}


def unit_cases() -> dict[str, LessonRequest]:
    """Every Author call the tests replay."""
    return {
        "cited": LessonRequest(
            objective_id="bayes_update",
            title="Calculate a posterior from a prior and a likelihood",
            description=(
                "Given a prior probability and a likelihood, compute the posterior "
                "probability using Bayes' theorem."
            ),
            bloom="apply",
            difficulty=3,
            est_minutes=25,
            assessment_stems=["Compute the posterior given a 1% prior and a 95% true positive rate."],
            misconceptions=[BASE_RATE_MISCONCEPTION],
            accepted_sources=SOURCES,
        ),
        "prose": LessonRequest(
            objective_id="prior_meaning",
            title="Explain what a prior represents",
            description="Say in plain words what a prior probability claims before any data arrives.",
            bloom="understand",
            difficulty=2,
            est_minutes=15,
            assessment_stems=["In your own words, what does a prior assert?"],
            misconceptions=[],
            accepted_sources=SOURCES[:1],
        ),
        "uncited": LessonRequest(
            objective_id="evidence_weight",
            title="Judge how much a piece of evidence should move a belief",
            description="Weigh evidence quality and decide how far a belief should shift.",
            bloom="analyze",
            difficulty=4,
            est_minutes=30,
            assessment_stems=["Which of these two studies should move you further, and why?"],
            misconceptions=[],
            accepted_sources=[],
        ),
        "misconception": LessonRequest(
            objective_id="prior_choice",
            title="Choose and defend a prior",
            description="Pick a prior for a stated problem and defend the choice.",
            bloom="create",
            difficulty=4,
            est_minutes=35,
            assessment_stems=["Pick a prior for this problem and defend it."],
            misconceptions=[PRIOR_MISCONCEPTION],
            accepted_sources=SOURCES[:1],
        ),
    }


# --- the stand-in model ----------------------------------------------------


def _objective_id(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("objective_id: "):
            return line.split(": ", 1)[1].strip()
    return ""


def _lesson(
    *,
    explainer: str,
    exercises: list[dict[str, Any]],
    code: str | None,
    citations: list[str],
) -> dict[str, Any]:
    return {
        "explainer": explainer,
        "worked_examples": [
            "Worked example 1. Write the numbers down, then apply the rule step by step.",
            "Worked example 2. Same rule, harder numbers, and one trap on the way through.",
        ],
        "exercises": exercises,
        "code": code,
        "review_summary": (
            "A prior is a belief before data. A likelihood says how well the data fits "
            "each belief. Multiply, then renormalise, and you have the posterior. "
            "Check the base rate before you trust a strong-looking result."
        ),
        "citations": citations,
    }


def synthesize(agent_name: str, prompt: str) -> dict[str, Any]:
    """Answer an Author prompt the way a model would, flaws included."""
    assert agent_name == "author", agent_name
    objective = _objective_id(prompt)

    if objective == "bayes_update":
        return _lesson(
            explainer=(
                "A posterior is what you believe after the evidence lands. You start "
                "from the prior, weigh the likelihood, and renormalise."
            ),
            exercises=[
                {
                    "prompt": "The prior is 1% and the test is 95% accurate. Compute the posterior.",
                    "answer": "About 16%.",
                    "wrong_answer_feedback": {
                        "95%": (
                            "The test is 95% accurate, so a positive result means a 95% chance "
                            "of disease. That reads the likelihood as the posterior and drops "
                            "the prior."
                        )
                    },
                    "bloom": "apply",
                },
                {
                    "prompt": "Raise the prior to 10% and recompute.",
                    "answer": "About 68%.",
                    "wrong_answer_feedback": {"95%": "You are still ignoring the base rate."},
                    "bloom": "apply",
                },
                {
                    "prompt": "State the rule you used in one sentence.",
                    "answer": "Posterior is proportional to prior times likelihood.",
                    "wrong_answer_feedback": {},
                    "bloom": "remember",
                },
            ],
            code=(
                "prior = 0.01\n"
                "sensitivity, false_positive = 0.95, 0.05\n"
                "joint = prior * sensitivity\n"
                "posterior = joint / (joint + (1 - prior) * false_positive)\n"
                "print(round(posterior, 3))  # 0.161\n"
            ),
            citations=[SOURCES[0]["url"], FABRICATED],
        )

    if objective == "prior_meaning":
        return _lesson(
            explainer="A prior is the belief you hold before the data arrives.",
            exercises=[
                {
                    "prompt": "State what a prior asserts, in one sentence.",
                    "answer": "It states how plausible each hypothesis is before the data.",
                    "wrong_answer_feedback": {
                        "It is the answer.": "You have collapsed the prior into the conclusion."
                    },
                    "bloom": "understand",
                },
                {
                    "prompt": "Name one thing a prior is not.",
                    "answer": "It is not a result computed from the data at hand.",
                    "wrong_answer_feedback": {},
                    "bloom": "remember",
                },
                {
                    "prompt": "Give an example of a prior you could defend.",
                    "answer": "A base rate taken from a published population study.",
                    "wrong_answer_feedback": {},
                    "bloom": "understand",
                },
            ],
            code="print('a prior is a belief')  # not needed here",
            citations=[SOURCES[0]["url"]],
        )

    if objective == "evidence_weight":
        return _lesson(
            explainer=(
                "Evidence moves a belief in proportion to how much better it fits one "
                "hypothesis than another."
            ),
            exercises=[
                {
                    "prompt": "Two studies disagree. Which should move you further, and why?",
                    "answer": "The one whose result is far less likely under the rival hypothesis.",
                    "wrong_answer_feedback": {
                        "The bigger sample always wins.": "You are counting rows instead of "
                        "weighing how well each hypothesis predicts the result."
                    },
                    "bloom": "analyze",
                },
                {
                    "prompt": "Name a case where a large study should move you very little.",
                    "answer": "When both hypotheses predict the same result equally well.",
                    "wrong_answer_feedback": {},
                    "bloom": "analyze",
                },
                {
                    "prompt": "Diagnose this claim: 'p < 0.05, so the hypothesis is true.'",
                    "answer": "It confuses a tail probability with a posterior belief.",
                    "wrong_answer_feedback": {},
                    "bloom": "evaluate",
                },
            ],
            code=None,
            citations=[FABRICATED],
        )

    # prior_choice: the model writes a fine lesson and ignores the misconception.
    return _lesson(
        explainer="Choosing a prior is a claim you have to defend, not a formality.",
        exercises=[
            {
                "prompt": "Pick a prior for this problem and defend it in three sentences.",
                "answer": "Any defensible prior with a stated source and a stated sensitivity check.",
                "wrong_answer_feedback": {},
                "bloom": "create",
            },
            {
                "prompt": "Build a sensitivity check for your prior.",
                "answer": "Rerun the update under a second prior and report how far the answer moves.",
                "wrong_answer_feedback": {},
                "bloom": "create",
            },
            {
                "prompt": "Defend your prior against someone who picked a flatter one.",
                "answer": "Point at the evidence behind your prior and how far the data moves both.",
                "wrong_answer_feedback": {},
                "bloom": "evaluate",
            },
        ],
        code=None,
        citations=[SOURCES[0]["url"]],
    )


def build_cassette() -> None:  # pragma: no cover - developer tool
    """Rewrite ``author_units.json`` from :func:`synthesize`."""
    import asyncio

    from tests.cassettes.replay import CASSETTE_DIR, Cassette

    engine = build_engine("sqlite://")
    create_all(engine)
    author = Author(engine)
    tape = Cassette(CASSETTE_DIR / f"{CASSETTE}.json", record=False, synthesize=synthesize)
    for payload in unit_cases().values():
        asyncio.run(tape.play(author, author.prompt(payload)))
    tape.save()
    print(f"{len(tape.entries)} entries in {tape.path}")


# --- pedagogy mapping ------------------------------------------------------


def test_bloom_level_picks_the_exercise_type() -> None:
    assert exercise_kind("remember") == "recall"
    assert exercise_kind("understand") == "recall"
    assert exercise_kind("apply") == "computation"
    assert exercise_kind("analyze") == "diagnosis"
    assert exercise_kind("evaluate") == "diagnosis"
    assert exercise_kind("create") == "build_and_defend"


def test_the_prompt_states_the_required_exercise_type() -> None:
    payload = unit_cases()["misconception"]
    assert "build_and_defend" in Author.prompt(Author.__new__(Author), payload)


def test_the_instruction_carries_cite_or_abstain() -> None:
    role = Author.role.lower()
    assert "cite or abstain" in role
    assert "never invent a reference" in role


def test_computational_objectives_are_detected() -> None:
    cases = unit_cases()
    assert is_computational(cases["cited"])
    assert not is_computational(cases["prose"])


def test_url_comparison_ignores_scheme_and_trailing_slash() -> None:
    assert normalise_url("https://www.a.test/x/") == normalise_url("http://a.test/x")


# --- schema minimums -------------------------------------------------------


def _exercise(n: int = 0) -> Exercise:
    return Exercise(prompt=f"p{n}", answer=f"a{n}", bloom="apply")


def test_a_lesson_needs_two_worked_examples() -> None:
    with pytest.raises(ValidationError):
        Lesson(
            explainer="x",
            worked_examples=["only one"],
            exercises=[_exercise(i) for i in range(3)],
            review_summary="short",
        )


def test_a_lesson_needs_three_to_six_exercises() -> None:
    for count in (MIN_EXERCISES - 1, MAX_EXERCISES + 1):
        with pytest.raises(ValidationError):
            Lesson(
                explainer="x",
                worked_examples=["a", "b"],
                exercises=[_exercise(i) for i in range(count)],
                review_summary="short",
            )


def test_the_review_summary_fits_on_a_card() -> None:
    with pytest.raises(ValidationError):
        Lesson(
            explainer="x",
            worked_examples=["a", "b"],
            exercises=[_exercise(i) for i in range(3)],
            review_summary=" ".join(["word"] * (MAX_SUMMARY_WORDS + 1)),
        )


def test_an_over_long_summary_is_trimmed_not_shipped() -> None:
    author = Author.__new__(Author)
    payload = unit_cases()["cited"]
    trimmed = author.trim_summary(" ".join(["word"] * 200), payload)
    assert len(trimmed.split()) == MAX_SUMMARY_WORDS


# --- replayed lessons ------------------------------------------------------


def _write(engine: Engine, ctx: LearnerContext, case: str) -> Lesson:
    payload = unit_cases()[case]
    with cassette(CASSETTE) as tape:
        result = run(Author(engine).run(ctx, payload, objective_id=payload.objective_id))
    assert [call["source"] for call in tape.calls] == ["cassette"]
    return result.output


def test_a_written_lesson_meets_the_shape(engine: Engine, ctx: LearnerContext) -> None:
    lesson = _write(engine, ctx, "cited")
    assert len(lesson.worked_examples) >= MIN_WORKED_EXAMPLES
    assert MIN_EXERCISES <= len(lesson.exercises) <= MAX_EXERCISES
    assert len(lesson.review_summary.split()) <= MAX_SUMMARY_WORDS
    assert lesson.explainer.strip()
    assert all(e.prompt and e.answer for e in lesson.exercises)
    assert any(e.wrong_answer_feedback for e in lesson.exercises)


def test_a_fabricated_citation_is_stripped(
    engine: Engine, ctx: LearnerContext, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="the_oracle.agents.author"):
        lesson = _write(engine, ctx, "cited")
    assert FABRICATED not in lesson.citations
    assert lesson.citations == [SOURCES[0]["url"]]
    assert any("stripped a citation" in r.getMessage() for r in caplog.records)


def test_no_accepted_sources_still_yields_a_lesson_that_abstains(
    engine: Engine, ctx: LearnerContext
) -> None:
    lesson = _write(engine, ctx, "uncited")
    assert lesson.citations == []
    assert ABSTENTION_NOTE in lesson.explainer
    assert "unverified" in lesson.explainer.lower()
    assert len(lesson.exercises) >= MIN_EXERCISES


def test_a_supplied_misconception_reaches_an_exercise(
    engine: Engine, ctx: LearnerContext
) -> None:
    lesson = _write(engine, ctx, "misconception")
    blob = "\n".join(
        f"{k}\n{v}" for e in lesson.exercises for k, v in e.wrong_answer_feedback.items()
    ).lower()
    assert PRIOR_MISCONCEPTION["wrong_model"].lower() in blob
    assert PRIOR_MISCONCEPTION["diagnostic"].lower() in blob


def test_a_misconception_the_model_already_covered_is_left_alone(
    engine: Engine, ctx: LearnerContext
) -> None:
    lesson = _write(engine, ctx, "cited")
    feedback = [e.wrong_answer_feedback for e in lesson.exercises if e.wrong_answer_feedback]
    assert feedback
    assert all(
        BASE_RATE_MISCONCEPTION["wrong_model"] not in mapping for mapping in feedback[1:]
    )


def test_code_is_present_for_a_computational_objective(
    engine: Engine, ctx: LearnerContext
) -> None:
    lesson = _write(engine, ctx, "cited")
    assert lesson.code is not None
    assert "posterior" in lesson.code
    scope: dict[str, Any] = {}
    exec(compile(lesson.code, "<lesson>", "exec"), scope)  # noqa: S102 - the point is runnable


def test_code_is_dropped_for_a_prose_objective(engine: Engine, ctx: LearnerContext) -> None:
    lesson = _write(engine, ctx, "prose")
    assert lesson.code is None


def test_exercise_bloom_falls_back_to_the_objective() -> None:
    author = Author.__new__(Author)
    payload = unit_cases()["cited"]
    fixed = author.fix_bloom(Exercise(prompt="p", answer="a", bloom="wat"), payload)
    assert fixed.bloom == payload.bloom


def test_the_voice_holds(engine: Engine, ctx: LearnerContext) -> None:
    from the_oracle.style import BANNED_PHRASES

    lesson = _write(engine, ctx, "cited")
    text = " ".join(
        [lesson.explainer, *lesson.worked_examples, lesson.review_summary]
        + [e.prompt + " " + e.answer for e in lesson.exercises]
    )
    assert "!" not in text
    assert not any(phrase in text.lower() for phrase in BANNED_PHRASES)


def test_a_retry_reuses_the_cached_lesson(engine: Engine, ctx: LearnerContext) -> None:
    """The most expensive agent in the system never pays twice."""
    payload = unit_cases()["cited"]
    author = Author(engine)
    with cassette(CASSETTE) as tape:
        first = run(author.run(ctx, payload, objective_id=payload.objective_id))
        second = run(author.run(ctx, payload, objective_id=payload.objective_id))
    assert len(tape.calls) == 1
    assert not first.cached and second.cached
    assert second.output.explainer == first.output.explainer


# --- the shipped path ------------------------------------------------------


def test_an_authored_lesson_lands_in_the_real_corpus(
    engine: Engine, ctx: LearnerContext
) -> None:
    """Reachability, not unit health: the Author's output feeds the real store.

    ``corpus.store.put_lesson`` and ``put_items`` are what the pipeline calls.
    If this breaks, an authored lesson never reaches a learner, however green
    the unit tests are.
    """
    from sqlmodel import Session, select

    from the_oracle.corpus.store import put_items, put_lesson, render_lesson
    from the_oracle.store import models

    payload = unit_cases()["cited"]
    lesson = _write(engine, ctx, "cited")

    resource_id = put_lesson(payload.objective_id, lesson, engine=engine)
    item_ids = put_items(payload.objective_id, lesson.exercises, engine=engine)

    with Session(engine) as session:
        resource = session.get(models.Resource, resource_id)
        items = session.exec(
            select(models.Item).where(models.Item.objective_id == payload.objective_id)
        ).all()

    assert resource is not None
    assert resource.kind == models.ResourceKind.AUTHORED
    assert resource.body == render_lesson(lesson)
    assert lesson.worked_examples[0] in resource.body
    assert resource.meta["citations"] == lesson.citations
    assert FABRICATED not in str(resource.meta)
    assert len(item_ids) == len(lesson.exercises) == len(items)
    assert all(item.answer_key.get("answer") for item in items)


def test_authoring_twice_does_not_pay_or_duplicate(
    engine: Engine, ctx: LearnerContext
) -> None:
    """Idempotency end to end: same objective, one lesson row, one model call."""
    from sqlmodel import Session, select

    from the_oracle.corpus.store import put_lesson
    from the_oracle.store import models

    payload = unit_cases()["cited"]
    author = Author(engine)
    with cassette(CASSETTE) as tape:
        for _ in range(2):
            lesson = run(
                author.run(ctx, payload, objective_id=payload.objective_id)
            ).output
            put_lesson(payload.objective_id, lesson, engine=engine)

    assert len(tape.calls) == 1
    with Session(engine) as session:
        rows = session.exec(
            select(models.Resource).where(models.Resource.objective_id == payload.objective_id)
        ).all()
    assert len(rows) == 1


def test_a_cassette_miss_is_loud(engine: Engine, ctx: LearnerContext) -> None:
    from tests.cassettes.replay import CassetteMiss

    payload = unit_cases()["cited"].model_copy(update={"title": "an objective never recorded"})
    with cassette(CASSETTE), pytest.raises(CassetteMiss):
        run(Author(engine).run(ctx, payload))


if __name__ == "__main__":  # pragma: no cover
    build_cassette()
