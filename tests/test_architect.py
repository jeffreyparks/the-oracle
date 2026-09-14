"""Architect tests. Offline, free, deterministic, no API key.

The Architect turns one interview into one curriculum. Two things in that
pipeline can go wrong quietly, so both are tested hard here:

* **A false reuse.** Mastery is keyed by ``(learner_id, objective_id)`` with no
  domain, so reusing the wrong id leaks credit between skills forever.
* **A half-written pack.** A manifest on disk whose objectives are missing
  fails at study time instead of build time. Nothing may be written unless the
  whole pack validates.

Every LLM call runs through the cassette harness in ``tests/cassettes``. The
draft comes from :mod:`tests.cassettes.architect_offline`, which is hand
written because no API key was available. A cassette miss raises; it never
falls through to the network. Dedupe runs on the lexical fallback embedder, so
there is no download and no key.

Every objective invented in this file is filler with no subject matter in it.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import Engine
from typer.testing import CliRunner

from the_oracle.agents.architect import (
    Architect,
    DraftDomain,
    DraftObjective,
    Interview,
    build_domain,
    commit,
    draft_domain,
    plan_domain,
    slugify,
)
from the_oracle.commands import domain_add
from the_oracle.config import reset_settings_cache
from the_oracle.domains import library
from the_oracle.domains.embed import LexicalEmbedder
from the_oracle.domains.errors import PackValidationError
from the_oracle.domains.registry import list_domains, load_domain, oracle_home, validate_domain
from the_oracle.domains.schema import Domain, Objective
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
    """A throwaway in-memory database, so no agent call touches real state."""
    engine = build_engine("sqlite://")
    create_all(engine)
    return engine


@pytest.fixture
def cli_engine(monkeypatch: pytest.MonkeyPatch, engine: Engine) -> Engine:
    """Make the command path use the in-memory database, not a file on disk."""
    monkeypatch.setattr("the_oracle.agents.base.get_engine", lambda *a, **k: engine)
    return engine


@pytest.fixture
def lexical() -> LexicalEmbedder:
    """The offline fallback embedder: no key, no download, deterministic."""
    return LexicalEmbedder()


@pytest.fixture
def interview() -> Interview:
    """The prefilled interview the cassette was recorded for."""
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
    """A syntactically valid library objective."""
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
    """Put one objective in the library and hand it back."""
    library.put(obj)
    return obj


def snapshot(root: Path) -> dict[str, str]:
    """Every file under ORACLE_HOME, by relative path, with its bytes."""
    return {
        str(p.relative_to(root)): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def draft_from(payload: dict[str, Any]) -> DraftDomain:
    return DraftDomain.model_validate(copy.deepcopy(payload))


def responder(payload: dict[str, Any]) -> Any:
    """A cassette responder that always answers with ``payload``."""

    def _synthesize(agent_name: str, prompt: str) -> dict[str, Any]:
        if agent_name == "architect":
            return copy.deepcopy(payload)
        raise KeyError(f"no offline responder for agent {agent_name!r}")

    return _synthesize


# --- 1. a draft becomes a pack that fully validates -------------------------


def test_draft_becomes_a_domain_that_passes_pack_validation(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder
) -> None:
    """The whole point: interview in, loadable pack out."""
    with cassette(CASSETTE, synthesize=synthesize) as tape:
        domain, matches = build_domain(interview, embedder=lexical, engine=engine)

    assert [call["agent"] for call in tape.calls] == ["architect"]
    assert isinstance(domain, Domain)
    assert len(domain.objectives) == len(DRAFT["objectives"])
    assert len(matches) == len(DRAFT["objectives"])

    # The registry, reading from disk, agrees it is valid.
    assert validate_domain(domain.id) == []
    reloaded = load_domain(domain.id)
    assert reloaded.id == domain.id
    assert [ref.id for ref in reloaded.objectives] == [ref.id for ref in domain.objectives]

    # Every objective is on disk, in exactly one module, and the order is legal.
    placed = [oid for module in reloaded.modules for oid in module.objectives]
    assert sorted(placed) == sorted(ref.id for ref in reloaded.objectives)
    for ref in reloaded.objectives:
        assert library.get(ref.id, ref.version).id == ref.id
    position = {oid: i for i, oid in enumerate(placed)}
    for edge in reloaded.edges:
        assert position[edge.from_id] < position[edge.to_id]


def test_teaching_order_is_acyclic_and_complete(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder
) -> None:
    with cassette(CASSETTE, synthesize=synthesize):
        domain, _ = build_domain(interview, embedder=lexical, engine=engine)
    order = load_domain(domain.id).teaching_order()
    assert sorted(order) == sorted(ref.id for ref in domain.objectives)


# --- 2. reuse beats minting a duplicate ------------------------------------


def test_build_domain_reuses_an_existing_library_objective(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder, home: Path
) -> None:
    """A near-identical objective already in the library must be reused."""
    twin = DRAFT["objectives"][0]
    existing = seed(
        make_objective(
            "already_owned_skill",
            twin["title"],
            twin["description"],
            bloom=twin["bloom"],
            difficulty=twin["difficulty"],
            est_minutes=twin["est_minutes"],
            tags=twin["tags"],
        )
    )
    before = {p.name for p in (home / "objectives").glob("*.yaml")}

    with cassette(CASSETTE, synthesize=synthesize):
        domain, matches = build_domain(interview, embedder=lexical, engine=engine)

    ids = [ref.id for ref in domain.objectives]
    assert existing.id in ids
    # The duplicate id that minting would have produced was never created.
    assert slugify(twin["title"]) not in ids
    assert not (home / "objectives" / f"{slugify(twin['title'])}.yaml").exists()

    after = {p.name for p in (home / "objectives").glob("*.yaml")}
    assert after - before == {f"{oid}.yaml" for oid in ids if oid != existing.id}
    assert len(after) == len(DRAFT["objectives"])  # six new plus the reused one

    match = next(m for m in matches if m.candidate_title == twin["title"])
    assert match.best_id == existing.id
    assert match.decision == "reuse"


def test_reused_objective_body_is_not_overwritten(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder, home: Path
) -> None:
    """Reuse means reuse: the library file keeps the body it already had."""
    twin = DRAFT["objectives"][0]
    existing = seed(
        make_objective(
            "already_owned_skill",
            twin["title"],
            twin["description"],
            bloom=twin["bloom"],
            difficulty=twin["difficulty"],
            est_minutes=twin["est_minutes"],
            tags=twin["tags"],
        )
    )
    path = home / "objectives" / f"{existing.id}.yaml"
    before = path.read_text(encoding="utf-8")
    with cassette(CASSETTE, synthesize=synthesize):
        build_domain(interview, embedder=lexical, engine=engine)
    assert path.read_text(encoding="utf-8") == before


# --- 3. a genuinely different candidate is minted ---------------------------


def test_build_domain_mints_when_the_candidate_is_different(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder, home: Path
) -> None:
    """An unrelated library entry must not capture any drafted objective."""
    unrelated = seed(
        make_objective(
            "unrelated_skill",
            "Tuning a Stringed Instrument",
            "Bring each string to pitch against a reference tone by ear.",
            tags=["music", "ear"],
        )
    )
    with cassette(CASSETTE, synthesize=synthesize):
        domain, matches = build_domain(interview, embedder=lexical, engine=engine)

    ids = [ref.id for ref in domain.objectives]
    assert unrelated.id not in ids
    assert len(ids) == len(DRAFT["objectives"])
    assert all(m.decision == "mint" for m in matches)
    for objective in DRAFT["objectives"]:
        assert (home / "objectives" / f"{slugify(objective['title'])}.yaml").is_file()


def test_same_words_at_a_different_depth_are_minted_not_merged(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder
) -> None:
    """Bias to split: a survey treatment is not the practitioner one."""
    twin = DRAFT["objectives"][4]  # bloom analyze, difficulty 4
    shallow = seed(
        make_objective(
            "shallow_twin",
            twin["title"],
            twin["description"],
            bloom="remember",
            difficulty=1,
            tags=twin["tags"],
        )
    )
    with cassette(CASSETTE, synthesize=synthesize):
        domain, _ = build_domain(interview, embedder=lexical, engine=engine)
    assert shallow.id not in [ref.id for ref in domain.objectives]


# --- 4. a broken draft persists nothing at all ------------------------------


BAD_EDGE_DRAFT = copy.deepcopy(DRAFT)
BAD_EDGE_DRAFT["edges"] = [
    *BAD_EDGE_DRAFT["edges"],
    ["Routing and Path Parameters", "Something Nobody Drafted"],
]


def test_unknown_edge_endpoint_is_rejected_and_nothing_is_written(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder, home: Path
) -> None:
    """A half-written pack is worse than no pack. So: write nothing."""
    seed(make_objective("pre_existing", "A Skill Already Owned", "Filler body."))
    before = snapshot(home)

    with cassette(CASSETTE, synthesize=responder(BAD_EDGE_DRAFT)):
        with pytest.raises(PackValidationError) as caught:
            build_domain(interview, embedder=lexical, engine=engine)

    problems = " ".join(caught.value.problems)
    assert "Something Nobody Drafted" in problems

    assert snapshot(home) == before
    assert list_domains() == []
    assert [obj.id for obj in library.all_objectives()] == ["pre_existing"]


def test_unknown_module_title_is_rejected_and_nothing_is_written(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder, home: Path
) -> None:
    bad = copy.deepcopy(DRAFT)
    bad["modules"][0]["objectives"] = [*bad["modules"][0]["objectives"], "Not A Drafted Title"]
    before = snapshot(home)
    with cassette(CASSETTE, synthesize=responder(bad)):
        with pytest.raises(PackValidationError):
            build_domain(interview, embedder=lexical, engine=engine)
    assert snapshot(home) == before


def test_a_cyclic_draft_is_rejected_and_nothing_is_written(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder, home: Path
) -> None:
    bad = copy.deepcopy(DRAFT)
    bad["edges"] = [
        *bad["edges"],
        ["Deploying and Observing a Service", "Request and Response Model"],
    ]
    before = snapshot(home)
    with cassette(CASSETTE, synthesize=responder(bad)):
        with pytest.raises(PackValidationError) as caught:
            build_domain(interview, embedder=lexical, engine=engine)
    assert any("cycle" in p or "rule5" in p for p in caught.value.problems)
    assert snapshot(home) == before


def test_a_draft_too_small_to_be_a_curriculum_is_rejected(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder, home: Path
) -> None:
    bad = copy.deepcopy(DRAFT)
    bad["objectives"] = bad["objectives"][:2]
    bad["edges"] = []
    bad["modules"] = [
        {"id": "only", "title": "Only", "goal": "g", "objectives": [o["title"] for o in bad["objectives"]]}
    ]
    bad["misconceptions"] = []
    before = snapshot(home)
    with cassette(CASSETTE, synthesize=responder(bad)):
        with pytest.raises(PackValidationError):
            build_domain(interview, embedder=lexical, engine=engine)
    assert snapshot(home) == before


def test_commit_rolls_back_when_the_manifest_will_not_reload(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder, home: Path
) -> None:
    """Even a late failure leaves no manifest and no orphan objectives."""
    with cassette(CASSETTE, synthesize=synthesize):
        draft = draft_domain(interview, engine=engine)
    plan = plan_domain(draft, interview, embedder=lexical)
    # Break the pin after validation, the way a racing writer would.
    broken = plan.domain.model_copy(
        update={
            "objectives": [
                ref.model_copy(update={"version": 99}) if i == 0 else ref
                for i, ref in enumerate(plan.domain.objectives)
            ]
        }
    )
    plan.domain = broken
    before = snapshot(home)

    with pytest.raises(PackValidationError):
        commit(plan)

    assert snapshot(home) == before
    assert list_domains() == []


# --- 5. the non-interactive path --------------------------------------------


def test_non_interactive_build_from_a_prefilled_interview(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder
) -> None:
    """A prefilled Interview needs no prompting and no second model call."""
    with cassette(CASSETTE, synthesize=synthesize) as tape:
        draft = draft_domain(interview, engine=engine)
        plan = plan_domain(draft, interview, embedder=lexical)
        domain = commit(plan)
    assert len(tape.calls) == 1
    assert plan.interview == interview
    assert plan.total_minutes == sum(o["est_minutes"] for o in DRAFT["objectives"])
    assert validate_domain(domain.id) == []


def test_domain_add_command_runs_without_any_prompting(
    cli_engine: Engine, home: Path
) -> None:
    """``domain add "<goal>" --yes`` writes the pack and asks nothing."""
    runner = CliRunner()
    with cassette(CASSETTE, synthesize=synthesize):
        result = runner.invoke(
            domain_add.app,
            [
                "--background",
                INTERVIEW["background"],
                "--hours-per-week",
                str(INTERVIEW["hours_per_week"]),
                "--target-weeks",
                str(INTERVIEW["target_weeks"]),
                "--depth",
                INTERVIEW["depth"],
                "--must-cover",
                "authentication",
                "--exclude",
                "front-end frameworks",
                "--yes",
                INTERVIEW["goal"],
            ],
            input="",
        )
    assert result.exit_code == 0, result.output
    written = list_domains()
    assert len(written) == 1
    assert validate_domain(written[0]) == []


def test_domain_add_rejects_a_bad_depth_before_spending_a_token(home: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(domain_add.app, ["--depth", "profound", "a goal"])
    assert result.exit_code == 2
    assert list_domains() == []


def test_domain_add_writes_nothing_when_the_draft_is_invalid(
    cli_engine: Engine, home: Path
) -> None:
    before = snapshot(home)
    runner = CliRunner()
    with cassette(CASSETTE, synthesize=responder(BAD_EDGE_DRAFT)):
        result = runner.invoke(domain_add.app, ["--yes", INTERVIEW["goal"]])
    assert result.exit_code == 1
    assert snapshot(home) == before


# --- 6. minted ids are stable, snake_case, and collision-free ---------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Request and Response Model", "request_and_response_model"),
        ("  spaced   out  ", "spaced_out"),
        ("Caf\u00e9 R\u00e9sum\u00e9 Handling", "cafe_resume_handling"),
        ("MiXeD-Case/Punctuation!", "mixed_case_punctuation"),
        ("...", "untitled"),
        ("one two three four five six seven", "one_two_three_four_five_six"),
    ],
)
def test_slugify_is_stable_snake_case(text: str, expected: str) -> None:
    assert slugify(text) == expected
    assert slugify(text) == slugify(text)
    assert slugify(expected) == expected


def test_minted_ids_are_snake_case_unique_and_derived_from_titles(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder
) -> None:
    with cassette(CASSETTE, synthesize=synthesize):
        domain, _ = build_domain(interview, embedder=lexical, engine=engine)
    ids = [ref.id for ref in domain.objectives]
    assert len(ids) == len(set(ids))
    for oid in ids:
        assert oid == slugify(oid)
        assert oid.islower()
    assert set(ids) == {slugify(o["title"]) for o in DRAFT["objectives"]}


def test_minted_ids_never_collide_with_an_existing_library_id(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder, home: Path
) -> None:
    """A different skill that happens to slug the same must not be clobbered."""
    twin = DRAFT["objectives"][1]
    squatter = seed(
        make_objective(
            slugify(twin["title"]),
            "Pruning A Fruit Tree",
            "Decide which branches to cut this season and where to cut them.",
            tags=["orchard"],
        )
    )
    body = (home / "objectives" / f"{squatter.id}.yaml").read_text(encoding="utf-8")

    with cassette(CASSETTE, synthesize=synthesize):
        domain, _ = build_domain(interview, embedder=lexical, engine=engine)

    ids = [ref.id for ref in domain.objectives]
    assert squatter.id not in ids
    assert f"{squatter.id}_2" in ids
    # The squatter's file is untouched, and the new id has its own file.
    assert (home / "objectives" / f"{squatter.id}.yaml").read_text(encoding="utf-8") == body
    minted = yaml.safe_load(
        (home / "objectives" / f"{squatter.id}_2.yaml").read_text(encoding="utf-8")
    )
    assert minted["title"] == twin["title"]


def test_a_second_build_of_the_same_interview_gets_a_fresh_domain_id(
    interview: Interview, engine: Engine, lexical: LexicalEmbedder
) -> None:
    """Two builds must not overwrite each other's manifests."""
    with cassette(CASSETTE, synthesize=synthesize):
        first, _ = build_domain(interview, embedder=lexical, engine=engine)
        second, _ = build_domain(interview, embedder=lexical, engine=engine)
    assert first.id != second.id
    assert sorted(list_domains()) == sorted([first.id, second.id])
    assert validate_domain(second.id) == []


# --- the agent itself: normalisation and the prompt -------------------------


def test_normalise_cleans_shape_without_touching_pedagogy(engine: Engine) -> None:
    architect = Architect(engine)
    payload = Interview.model_validate(INTERVIEW)
    draft = DraftDomain(
        title="  A   Title  ",
        description="  one   sentence  ",
        objectives=[
            DraftObjective(
                title="  Doubled  Spaces  ",
                description="  a   body ",
                bloom="ANALYZE",
                difficulty=9,
                est_minutes=0,
                assessment_stems=["  keep me ", "   "],
                tags=[" Beta ", "alpha", "ALPHA"],
            ),
            DraftObjective(
                title="doubled spaces",  # a case-folded duplicate: dropped
                description="another body",
                bloom="nonsense",
                difficulty=3,
                est_minutes=10,
            ),
        ],
        edges=[("  Doubled  Spaces  ", "Doubled Spaces")],
    )
    cleaned = architect.normalise(draft, payload)
    assert cleaned.title == "A Title"
    assert cleaned.titles() == ["Doubled Spaces"]
    only = cleaned.objectives[0]
    assert only.bloom == "analyze"
    assert only.difficulty == 5
    assert only.est_minutes == 1
    assert only.assessment_stems == ["keep me"]
    assert only.tags == ["alpha", "beta"]
    assert cleaned.edges == [("Doubled Spaces", "Doubled Spaces")]


def test_the_prompt_is_deterministic_and_carries_the_interview(engine: Engine) -> None:
    """Cassettes key on the prompt, so it must not wobble between runs."""
    architect = Architect(engine)
    payload = Interview.model_validate(INTERVIEW)
    first = architect.prompt(payload)
    assert first == architect.prompt(payload)
    assert INTERVIEW["goal"] in first
    assert "authentication" in first
    assert "front-end frameworks" in first
    assert str(payload.budget_minutes) in first


def test_budget_minutes_discounts_for_real_life() -> None:
    payload = Interview.model_validate({**INTERVIEW, "hours_per_week": 10, "target_weeks": 2})
    assert payload.budget_minutes == int(10 * 60 * 2 * 0.8)


def test_an_absurdly_long_draft_is_rejected_against_the_budget(
    engine: Engine, lexical: LexicalEmbedder, home: Path
) -> None:
    """More than twice the learner's time is not a plan, it is a wish."""
    tiny = Interview.model_validate({**INTERVIEW, "hours_per_week": 0.5, "target_weeks": 1})
    with cassette(CASSETTE, synthesize=synthesize):
        draft = draft_domain(tiny, engine=engine)
    before = snapshot(home)
    with pytest.raises(PackValidationError) as caught:
        plan_domain(draft, tiny, embedder=lexical)
    assert any("budget" in p for p in caught.value.problems)
    assert snapshot(home) == before


def test_no_subject_matter_leaks_into_the_source() -> None:
    """The draft's words come from the model, never from ``src/``."""
    source = Path(__file__).resolve().parents[1] / "src" / "the_oracle"
    text = " ".join(
        p.read_text(encoding="utf-8") for p in source.rglob("*.py")
    ).casefold()
    for word in ("authentication and session handling", "routing and path parameters"):
        assert word not in text


def test_domain_add_accepts_options_after_the_goal(cli_engine: Engine, home: Path) -> None:
    """The natural CLI order a learner will type must work.

    Regression test. ``commands/domain_add.py`` exposes its own Typer group with
    ``invoke_without_command=True``; wiring that group in with ``add_typer``
    makes click stop option parsing at the positional goal, so
    ``domain add "a goal" --yes`` died with "No such command '--yes'".
    ``cli.py`` now registers the callback as a real command instead, which parses
    options in either position. Asserted against the SHIPPED cli app, because
    that is the surface a learner actually types at.
    """
    from the_oracle.cli import app as cli_app

    runner = CliRunner()
    with cassette(CASSETTE, synthesize=synthesize):
        result = runner.invoke(
            cli_app, ["domain", "add", INTERVIEW["goal"], "--yes"], input=""
        )
    assert result.exit_code == 0, result.output
    assert len(list_domains()) == 1
