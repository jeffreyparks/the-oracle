"""Retrieval-augmented drafting. Offline, free, deterministic, no API key.

The Architect used to draft blind. Four live runs against a deliberately
overlapping topic produced zero reuse: it invented objectives that parallel
existing library ones at a different Bloom level, and dedupe then refused to
merge them, correctly, because merging across a real Bloom gap corrupts
mastery. The library bloats and mastery stops transferring.

So these tests hold the prevention, not the reconciliation:

* the nearest existing objectives are found and ranked,
* they are actually **interpolated into the prompt** the model sees - a block
  that is built and dropped has shipped twice in this project,
* an ``existing_id`` the Architect returns is honoured, reported as
  ``retrieved`` so a human can audit it, and keeps the LIBRARY body,
* and a hallucinated id never breaks a build and never silently reuses the
  wrong objective.

Everything runs on the lexical fallback embedder: no download, no key, no
network. Every objective invented here is filler with no subject matter in it.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from the_oracle.agents.architect import (
    Architect,
    DraftDomain,
    Interview,
    draft_domain,
    plan_domain,
)
from the_oracle.agents.retrieval import (
    CONTEXT_HEADER,
    REUSE_INSTRUCTION,
    query_text,
    render_context,
    retrieve_context,
)
from the_oracle.config import reset_settings_cache
from the_oracle.domains import library
from the_oracle.domains.dedupe import SOURCE_RETRIEVED, SOURCE_SCORED
from the_oracle.domains.embed import LexicalEmbedder
from the_oracle.domains.schema import Objective
from the_oracle.store.db import build_engine, create_all

from tests.cassettes.architect_offline import DRAFT, INTERVIEW, synthesize
from tests.cassettes.replay import cassette

LEARNER = "test-learner"
CASSETTE = "architect_offline"


# --- fixtures --------------------------------------------------------------


@pytest.fixture(autouse=True)
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """A throwaway ORACLE_HOME. The real packs are never touched."""
    root = tmp_path / "home"
    (root / "objectives").mkdir(parents=True)
    (root / "domains").mkdir(parents=True)
    monkeypatch.setenv("ORACLE_HOME", str(root))
    monkeypatch.setenv("ORACLE_LEARNER_ID", LEARNER)
    monkeypatch.setenv("ORACLE_EMBEDDER", "lexical")
    for key in ("OPENAI_API_KEY", "VOYAGE_API_KEY", "ORACLE_EMBED_MODEL"):
        monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield root
    reset_settings_cache()


@pytest.fixture
def engine() -> Engine:
    engine = build_engine("sqlite://")
    create_all(engine)
    return engine


@pytest.fixture
def lexical() -> LexicalEmbedder:
    return LexicalEmbedder()


@pytest.fixture
def interview() -> Interview:
    return Interview.model_validate(INTERVIEW)


def make_objective(
    oid: str,
    title: str,
    description: str,
    *,
    bloom: str = "understand",
    difficulty: int = 3,
    est_minutes: int = 30,
    tags: Sequence[str] = (),
) -> Objective:
    return Objective(
        id=oid,
        version=1,
        title=title,
        description=description,
        bloom=bloom,  # type: ignore[arg-type]
        difficulty=difficulty,
        est_minutes=est_minutes,
        assessment_stems=["State the rule.", "Apply the rule to one case."],
        tags=list(tags),
    )


def seed(obj: Objective) -> Objective:
    library.put(obj)
    return obj


def draft_with(**changes: Any) -> DraftDomain:
    """The cassette draft, with the first objective edited."""
    payload = copy.deepcopy(DRAFT)
    payload["objectives"][0].update(changes)
    return DraftDomain.model_validate(payload)


def responder(payload: dict[str, Any]) -> Any:
    def _synthesize(agent_name: str, prompt: str) -> dict[str, Any]:
        if agent_name == "architect":
            return copy.deepcopy(payload)
        raise KeyError(f"no offline responder for agent {agent_name!r}")

    return _synthesize


# --- 1. retrieval ranks nearest first and respects k ------------------------


def test_retrieve_context_returns_nearest_first_and_respects_k(
    interview: Interview, lexical: LexicalEmbedder
) -> None:
    """The objective that shares the learner's words must come back first."""
    near = make_objective(
        "near_one",
        "Authentication for a Small Production Web Service",
        "Ship a small production web service that can prove who the caller is.",
        tags=["authentication", "production"],
    )
    middle = make_objective(
        "middle_one",
        DRAFT["objectives"][0]["title"],
        DRAFT["objectives"][0]["description"],
        tags=DRAFT["objectives"][0]["tags"],
    )
    far = make_objective(
        "far_one",
        "Tuning a Stringed Instrument",
        "Bring each string to pitch against a reference tone by ear.",
        tags=["orchard", "music"],
    )
    shelf = [far, middle, near]

    ranked = retrieve_context(interview, embedder=lexical, library_objs=shelf)
    assert [o.id for o in ranked] == ["near_one", "middle_one", "far_one"]

    assert [o.id for o in retrieve_context(
        interview, k=2, embedder=lexical, library_objs=shelf
    )] == ["near_one", "middle_one"]
    assert retrieve_context(interview, k=0, embedder=lexical, library_objs=shelf) == []


def test_retrieve_context_reads_the_library_when_it_is_given_nothing(
    interview: Interview, lexical: LexicalEmbedder
) -> None:
    """The shipped call passes no library, so the default must be the library."""
    seed(make_objective("on_the_shelf", "A Skill Already Owned", "Filler body."))
    assert [o.id for o in retrieve_context(interview, embedder=lexical)] == [
        "on_the_shelf"
    ]


def test_the_query_is_the_goal_the_must_covers_and_the_background(
    interview: Interview,
) -> None:
    text = query_text(interview)
    assert INTERVIEW["goal"] in text
    assert "authentication" in text
    assert INTERVIEW["background"] in text


# --- 2. an empty library is normal, not an error ----------------------------


def test_empty_library_returns_nothing_and_does_not_raise(
    interview: Interview, lexical: LexicalEmbedder
) -> None:
    """The first domain anyone builds has nothing to retrieve."""
    assert retrieve_context(interview, embedder=lexical, library_objs=[]) == []
    assert retrieve_context(interview, embedder=lexical) == []
    assert render_context([]) == ""


def test_an_empty_library_leaves_the_prompt_exactly_as_it_was(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder
) -> None:
    architect = Architect(engine, embedder=lexical)
    assert CONTEXT_HEADER not in architect.prompt(interview, [])


# --- 3. a valid existing_id reuses, mints nothing, keeps the library body ----


def test_a_valid_existing_id_reuses_and_keeps_the_library_body(
    interview: Interview, lexical: LexicalEmbedder, home: Path
) -> None:
    """Reuse means the library's body wins, not the draft's."""
    twin = DRAFT["objectives"][0]
    existing = seed(
        make_objective(
            "library_owned_skill",
            twin["title"],
            twin["description"],
            bloom="evaluate",
            difficulty=5,
            est_minutes=75,
            tags=twin["tags"],
        )
    )
    # The draft disagrees about depth and length on purpose.
    draft = draft_with(
        existing_id=existing.id, bloom="understand", difficulty=1, est_minutes=5
    )

    plan = plan_domain(draft, interview, embedder=lexical)

    assert plan.reused[twin["title"]] == existing.id
    assert twin["title"] not in plan.minted
    assert existing.id in [ref.id for ref in plan.domain.objectives]
    # Nothing was minted for it, so no second file will ever be written.
    assert existing.id not in {body.id for body in plan.minted.values()}

    body = plan.resolved[existing.id]
    assert (body.bloom, body.difficulty, body.est_minutes) == ("evaluate", 5, 75)
    assert body.description == existing.description


def test_a_retrieved_reuse_never_asks_the_critic(
    interview: Interview, lexical: LexicalEmbedder
) -> None:
    """The question is already answered, so a judge call would be waste."""
    twin = DRAFT["objectives"][0]
    existing = seed(make_objective("library_owned_skill", twin["title"], twin["description"]))
    seen: list[str] = []

    def judge(match: Any) -> bool:
        seen.append(match.candidate_title)
        return False

    plan = plan_domain(
        draft_with(existing_id=existing.id), interview, embedder=lexical, judge=judge
    )
    assert twin["title"] not in seen
    assert plan.reused[twin["title"]] == existing.id


# --- 4. a hallucinated id is never fatal and never a silent wrong reuse ------


def test_a_hallucinated_existing_id_falls_through_to_dedupe_and_still_builds(
    interview: Interview, lexical: LexicalEmbedder, caplog: pytest.LogCaptureFixture
) -> None:
    """A made-up id must cost a log line, not a build."""
    seed(make_objective("real_but_unrelated", "Pruning A Fruit Tree", "Filler body."))
    draft = draft_with(existing_id="an_id_no_library_ever_had")

    with caplog.at_level(logging.WARNING):
        plan = plan_domain(draft, interview, embedder=lexical)

    title = DRAFT["objectives"][0]["title"]
    assert "an_id_no_library_ever_had" in caplog.text
    # The claim was dropped, not honoured, and dedupe ruled instead.
    assert title not in plan.reused
    assert title in plan.minted
    assert plan.reason_for(title) == "new"
    assert "an_id_no_library_ever_had" not in plan.resolved
    # And the whole pack still assembles.
    assert len(plan.domain.objectives) == len(DRAFT["objectives"])
    assert plan.match_for(title).source == SOURCE_SCORED


def test_a_stale_existing_id_does_not_reuse_whatever_is_nearby(
    interview: Interview, lexical: LexicalEmbedder
) -> None:
    """Dropping a claim must not turn into merging with the closest thing."""
    nearby = seed(
        make_objective("nearby_skill", "Pruning A Fruit Tree", "Decide which branches to cut.")
    )
    plan = plan_domain(draft_with(existing_id="deleted_last_year"), interview, embedder=lexical)
    assert nearby.id not in plan.reused.values()
    assert nearby.id not in [ref.id for ref in plan.domain.objectives]


# --- 5. an odd but real choice is honoured, and reported for audit ----------


def test_an_existing_id_for_a_different_skill_is_honoured_but_reported(
    interview: Interview, lexical: LexicalEmbedder
) -> None:
    """The Architect saw it and chose it. A human gets to see that it did."""
    odd = seed(
        make_objective(
            "unrelated_skill",
            "Tuning a Stringed Instrument",
            "Bring each string to pitch against a reference tone by ear.",
            tags=["music", "ear"],
        )
    )
    title = DRAFT["objectives"][0]["title"]
    plan = plan_domain(draft_with(existing_id=odd.id), interview, embedder=lexical)

    assert plan.reused[title] == odd.id
    assert plan.retrieved[title] == odd.id
    assert plan.reason_for(title) == "retrieved"
    match = plan.match_for(title)
    assert match.source == SOURCE_RETRIEVED
    assert match.best_id == odd.id
    assert str(match.decision) == "reuse"
    # No similarity was computed for this pair, so no number is fabricated.
    assert match.score == 0.0


def test_the_cli_report_names_the_reason_for_every_row(
    interview: Interview, lexical: LexicalEmbedder
) -> None:
    """``domain add`` must distinguish a retrieved reuse from a scored one."""
    from rich.console import Console

    from the_oracle.commands.domain_add import dedupe_table, reuse_summary

    odd = seed(make_objective("unrelated_skill", "Tuning a Stringed Instrument", "Filler."))
    plan = plan_domain(draft_with(existing_id=odd.id), interview, embedder=lexical)

    console = Console(width=200, record=True)
    console.print(dedupe_table(plan))
    rendered = console.export_text()
    assert "reason" in rendered
    assert "retrieved" in rendered
    assert "new" in rendered

    summary = reuse_summary(plan)
    assert summary == (
        f"{len(plan.domain.objectives)} objectives, 1 reused (1 retrieved), "
        f"{len(plan.minted)} new."
    )


# --- 6. the block is interpolated, not merely built -------------------------


def test_the_prompt_contains_the_retrieved_ids(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder
) -> None:
    """Guard against a block that is built and then never used."""
    shelf = [
        seed(make_objective("first_shelf_id", "A Skill Already Owned", "Filler body one.")),
        seed(
            make_objective(
                "second_shelf_id",
                "Authentication and Session Handling",
                "Filler body two.",
                bloom="evaluate",
                difficulty=4,
                est_minutes=75,
            )
        ),
    ]
    architect = Architect(engine, embedder=lexical)
    prompt = architect.prompt(interview, shelf)

    assert CONTEXT_HEADER in prompt
    for obj in shelf:
        assert f"id: {obj.id}" in prompt
        assert obj.title in prompt
        assert obj.description in prompt
    assert "bloom: evaluate | difficulty: 4 | minutes: 75" in prompt
    assert REUSE_INSTRUCTION in prompt
    assert "existing_id" in prompt
    # Still deterministic: cassettes key on this string.
    assert prompt == architect.prompt(interview, shelf)


def test_the_shipped_draft_path_puts_the_library_in_the_prompt(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder
) -> None:
    """The wiring test: ``draft_domain`` itself must retrieve.

    A component built, unit-tested and never wired to the real path has shipped
    twice in this project (a stale ``domain add`` stub, and a Critic nobody
    called). So this asserts against the prompt that actually reached the model
    through ``draft_domain``, not against a hand-assembled one.
    """
    seed(make_objective("on_the_shelf", "A Skill Already Owned", "Filler body."))

    with cassette(CASSETTE, synthesize=synthesize) as tape:
        draft_domain(interview, engine=engine, embedder=lexical)

    prompt = tape.calls[0]["prompt"]
    assert tape.calls[0]["agent"] == "architect"
    assert CONTEXT_HEADER in prompt
    assert "id: on_the_shelf" in prompt


def test_the_draft_carries_existing_id_through_normalisation(
    interview: Interview, engine: Engine
) -> None:
    """The field has to survive the shape-cleaning pass to be worth anything."""
    architect = Architect(engine)
    cleaned = architect.normalise(draft_with(existing_id="  kept_id  "), interview)
    assert cleaned.objectives[0].existing_id == "kept_id"
    blanked = architect.normalise(draft_with(existing_id="   "), interview)
    assert blanked.objectives[0].existing_id is None
    assert DraftDomain.model_validate(copy.deepcopy(DRAFT)).objectives[0].existing_id is None


def test_the_whole_pipeline_reuses_what_the_architect_claimed(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder, home: Path
) -> None:
    """End to end: a claimed id is reused and no duplicate file is written."""
    twin = DRAFT["objectives"][0]
    existing = seed(
        make_objective("library_owned_skill", twin["title"], twin["description"])
    )
    claimed = copy.deepcopy(DRAFT)
    claimed["objectives"][0]["existing_id"] = existing.id

    from the_oracle.agents.architect import build_domain

    with cassette(CASSETTE, synthesize=responder(claimed)):
        domain, matches = build_domain(interview, embedder=lexical, engine=engine)

    assert existing.id in [ref.id for ref in domain.objectives]
    files = {p.stem for p in (home / "objectives").glob("*.yaml")}
    assert len(files) == len(DRAFT["objectives"])
    match = next(m for m in matches if m.candidate_title == twin["title"])
    assert match.source == SOURCE_RETRIEVED
