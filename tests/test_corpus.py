"""The corpus pipeline: lazy, idempotent, and cheap to re-run.

Everything here runs offline. No API key, no network, no model, no cassette -
the pipeline reaches Scout, fetch, the Reviewer and the Author through
:class:`Seams`, so the fakes below are the whole dependency. That is deliberate:
these tests must not need ``agents/scout.py``, ``agents/reviewer.py`` or
``agents/author.py`` to exist in any particular shape beyond the frozen contract.

The three things this file is really guarding:

1. **Laziness.** An objective that already has enough accepted material costs
   nothing on the next run - zero searches, zero model calls.
2. **Idempotency.** A second ``ensure_module`` adds no rows and pays for nothing.
3. **Reachability.** The pipeline is exercised through the shipped
   ``resources`` command, not only through the library API. This project has
   shipped four components that were unit-tested and never wired in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import typer
import yaml
from sqlalchemy import Engine
from sqlmodel import Session, select
from typer.testing import CliRunner

from the_oracle.config import reset_settings_cache
from the_oracle.context import LearnerContext
from the_oracle.corpus import store
from the_oracle.corpus.pipeline import (
    DryRunViolation,
    ObjectiveSpec,
    Seams,
    ensure_module,
    ensure_objective,
)
from the_oracle.domains.registry import load_domain
from the_oracle.store import models
from the_oracle.store.db import build_engine, create_all
from the_oracle.store.events import EventKind, EventLog

DOMAIN_ID = "corpus_domain"
MODULE_ID = "mod_one"
LEARNER = "learner_corpus"
OBJECTIVES = ["obj_one", "obj_two"]


# --- a tiny, subject-neutral pack -------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    objectives_dir = tmp_path / "objectives"
    objectives_dir.mkdir()
    for oid in OBJECTIVES:
        (objectives_dir / f"{oid}.yaml").write_text(
            yaml.safe_dump(
                {
                    "id": oid,
                    "version": 1,
                    "title": f"Skill {oid}",
                    "description": f"A neutral placeholder skill named {oid}.",
                    "bloom": "apply",
                    "difficulty": 2,
                    "est_minutes": 30,
                    "assessment_stems": [f"Do the thing for {oid}."],
                    "tags": ["placeholder"],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    domains_dir = tmp_path / "domains"
    domains_dir.mkdir()
    (domains_dir / f"{DOMAIN_ID}.yaml").write_text(
        yaml.safe_dump(
            {
                "id": DOMAIN_ID,
                "version": 1,
                "title": "Corpus Test Domain",
                "description": "Synthetic pack used only by the corpus tests.",
                "objectives": [{"id": o, "version": 1} for o in OBJECTIVES],
                "edges": [],
                "modules": [
                    {
                        "id": MODULE_ID,
                        "title": "Module One",
                        "goal": "goal",
                        "objectives": list(OBJECTIVES),
                    }
                ],
                "misconceptions": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ORACLE_HOME", str(tmp_path))
    monkeypatch.setenv("ORACLE_LEARNER_ID", LEARNER)
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SERPER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield tmp_path
    reset_settings_cache()


@pytest.fixture
def engine() -> Engine:
    engine = build_engine("sqlite://")
    create_all(engine)
    return engine


@pytest.fixture
def ctx(home: Path) -> LearnerContext:
    return LearnerContext.for_learner(LEARNER)


@pytest.fixture
def domain(home: Path):  # type: ignore[no-untyped-def]
    return load_domain(DOMAIN_ID)


# --- fakes, duck-typed against the frozen contract --------------------------


@dataclass(frozen=True)
class FakeCandidate:
    url: str
    title: str = "A page"
    snippet: str = "snippet"
    source: str = "fake"
    rank: int = 0


@dataclass(frozen=True)
class FakePage:
    url: str
    final_url: str
    status: int = 200
    title: str = "A page"
    text: str = "word " * 400
    word_count: int = 400
    error: str | None = None


@dataclass(frozen=True)
class FakeReview:
    verdict: str
    score: float = 4.0
    rubric: dict[str, int] = field(default_factory=lambda: {"authority": 4})
    notes: str = "fine"
    hard_reject_reason: str | None = None


@dataclass(frozen=True)
class FakeExercise:
    prompt: str
    answer: str = "the answer"
    wrong_answer_feedback: dict[str, str] = field(default_factory=dict)
    bloom: str = "apply"


@dataclass(frozen=True)
class FakeLesson:
    explainer: str = "Here is the idea, stated plainly."
    worked_examples: list[str] = field(default_factory=lambda: ["one", "two"])
    exercises: list[FakeExercise] = field(
        default_factory=lambda: [FakeExercise("Try this."), FakeExercise("Now this.")]
    )
    code: str | None = None
    review_summary: str = "Short recap for spaced review."
    citations: list[str] = field(default_factory=list)


class Spy:
    """Counts every call through every seam. The cost meter for these tests."""

    def __init__(
        self,
        *,
        accept: int = 2,
        found: int = 3,
        search_error: Exception | None = None,
        lesson_verdict: str = "accept",
    ) -> None:
        self.accept = accept
        self.found = found
        self.search_error = search_error
        self.lesson_verdict = lesson_verdict
        self.searches = 0
        self.fetches = 0
        self.reviews = 0
        self.authored = 0
        self._accepted_so_far: dict[str, int] = {}
        self.review_kwargs: list[dict[str, Any]] = []

    @property
    def model_calls(self) -> int:
        return self.reviews + self.authored

    @property
    def calls(self) -> int:
        return self.searches + self.fetches + self.model_calls

    def search(self, spec: ObjectiveSpec, limit: int) -> list[FakeCandidate]:
        self.searches += 1
        if self.search_error is not None:
            raise self.search_error
        return [FakeCandidate(url=f"https://example.test/{spec.id}/{i}") for i in range(self.found)]

    def fetch(self, urls: Any) -> list[FakePage]:
        self.fetches += 1
        return [FakePage(url=u, final_url=u) for u in urls]

    def review(self, spec: ObjectiveSpec, url: str, title: str, text: str, **kwargs: Any) -> FakeReview:
        self.reviews += 1
        self.review_kwargs.append(kwargs)
        if not url:  # the authored lesson
            return FakeReview(verdict=self.lesson_verdict, score=4.5)
        seen = self._accepted_so_far.get(spec.id, 0)
        if seen < self.accept:
            self._accepted_so_far[spec.id] = seen + 1
            return FakeReview(verdict="accept", score=4.2)
        return FakeReview(verdict="reject", score=1.0)

    def author(self, spec: ObjectiveSpec, sources: Any) -> FakeLesson:
        self.authored += 1
        return FakeLesson(citations=[s["url"] for s in sources])

    def seams(self) -> Seams:
        return Seams(search=self.search, fetch=self.fetch, review=self.review, author=self.author)


def resources_in(engine: Engine, objective_id: str | None = None) -> list[models.Resource]:
    statement = select(models.Resource)
    if objective_id:
        statement = statement.where(models.Resource.objective_id == objective_id)
    with Session(engine) as session:
        return list(session.exec(statement).all())


def attached_events(engine: Engine) -> list[Any]:
    return EventLog(engine).read(LEARNER, kinds=[EventKind.RESOURCE_ATTACHED])


# --- store ------------------------------------------------------------------


def test_resource_ids_are_content_derived_so_a_rerun_cannot_duplicate(engine: Engine) -> None:
    """Same objective, same URL - one row, one id, however often it is written."""
    first = store.put_resource("obj_one", url="https://Example.test/a/?utm_source=x", engine=engine)
    second = store.put_resource("obj_one", url="https://example.test/a#frag", engine=engine)
    other = store.put_resource("obj_two", url="https://example.test/a", engine=engine)
    assert first == second
    assert other != first
    assert len(resources_in(engine)) == 2


def test_accepted_for_reads_the_latest_verdict_only(engine: Engine) -> None:
    rid = store.put_resource("obj_one", url="https://example.test/a", engine=engine)
    store.record_review(rid, FakeReview(verdict="reject", score=1.0), engine)
    assert store.accepted_for("obj_one", engine) == []
    store.record_review(rid, FakeReview(verdict="accept", score=4.0), engine)
    assert [r.id for r in store.accepted_for("obj_one", engine)] == [rid]
    # Recording the same verdict twice adds no row.
    store.record_review(rid, FakeReview(verdict="accept", score=4.0), engine)
    with Session(engine) as session:
        assert len(list(session.exec(select(models.ResourceReview)).all())) == 2


def test_put_items_is_idempotent(engine: Engine) -> None:
    exercises = [FakeExercise("Try this."), FakeExercise("Now this.")]
    first = store.put_items("obj_one", exercises, engine)
    second = store.put_items("obj_one", exercises, engine)
    assert first == second
    with Session(engine) as session:
        assert len(list(session.exec(select(models.Item)).all())) == 2


# --- laziness ---------------------------------------------------------------


def test_objective_with_enough_accepted_resources_is_skipped_for_free(
    engine: Engine, domain: Any, ctx: LearnerContext
) -> None:
    """This is what makes the pipeline lazy: enough material, zero spend."""
    for i in range(2):
        rid = store.put_resource("obj_one", url=f"https://example.test/seed/{i}", engine=engine)
        store.record_review(rid, FakeReview(verdict="accept"), engine)

    spy = Spy()
    result = ensure_objective(
        "obj_one", domain, minimum=2, engine=engine, ctx=ctx, seams=spy.seams()
    )

    assert result.skipped is True
    assert result.accepted == 2
    assert spy.calls == 0


def test_one_accepted_lesson_counts_as_enough(
    engine: Engine, domain: Any, ctx: LearnerContext
) -> None:
    """An authored lesson is the expensive fallback. Never buy it twice."""
    rid = store.put_lesson("obj_one", FakeLesson(), engine)
    store.record_review(rid, FakeReview(verdict="accept"), engine)
    spy = Spy()
    result = ensure_objective(
        "obj_one", domain, minimum=2, engine=engine, ctx=ctx, seams=spy.seams()
    )
    assert result.skipped is True
    assert spy.calls == 0


# --- the found path ---------------------------------------------------------


def test_enough_accepted_sources_attaches_and_never_calls_the_author(
    engine: Engine, domain: Any, ctx: LearnerContext
) -> None:
    spy = Spy(accept=2, found=3)
    result = ensure_objective(
        "obj_one", domain, minimum=2, engine=engine, ctx=ctx, seams=spy.seams()
    )

    assert (result.searched, result.fetched) == (3, 3)
    assert (result.accepted, result.rejected) == (2, 1)
    assert result.authored is False
    assert spy.authored == 0
    assert len(attached_events(engine)) == 2
    # The fetcher's evidence reaches the Reviewer, so a dead or thin page is a
    # free reject instead of a paid one.
    assert all("word_count" in kw and "fetch_error" in kw for kw in spy.review_kwargs)


def test_author_fires_only_when_accepted_is_below_the_minimum(
    engine: Engine, domain: Any, ctx: LearnerContext
) -> None:
    spy = Spy(accept=1, found=3)
    result = ensure_objective(
        "obj_one", domain, minimum=2, engine=engine, ctx=ctx, seams=spy.seams()
    )

    assert result.authored is True
    assert spy.authored == 1
    authored = [r for r in resources_in(engine, "obj_one") if r.kind == models.ResourceKind.AUTHORED]
    assert len(authored) == 1
    assert authored[0].body
    assert authored[0].meta["lesson"]["review_summary"]
    # The lesson's exercises became shared items.
    with Session(engine) as session:
        assert len(list(session.exec(select(models.Item)).all())) == 2


def test_scout_failure_degrades_to_the_author_instead_of_crashing(
    engine: Engine, domain: Any, ctx: LearnerContext
) -> None:
    """No Serper key is a Tuesday, not an outage."""
    spy = Spy(search_error=RuntimeError("ScoutUnavailableError: no SERPER_API_KEY"))
    results = ensure_module(
        DOMAIN_ID, MODULE_ID, minimum=2, engine=engine, ctx=ctx, seams=spy.seams(), domain=domain
    )

    assert [r.objective_id for r in results] == OBJECTIVES
    assert all(r.searched == 0 and r.authored for r in results)
    assert spy.fetches == 0
    assert len(attached_events(engine)) == 2


# --- events and idempotency -------------------------------------------------


def test_resource_attached_is_appended_once_per_attachment(
    engine: Engine, domain: Any, ctx: LearnerContext
) -> None:
    spy = Spy(accept=2, found=3)
    ensure_module(
        DOMAIN_ID, MODULE_ID, minimum=2, engine=engine, ctx=ctx, seams=spy.seams(), domain=domain
    )
    events = attached_events(engine)
    assert len(events) == 4  # two objectives, two accepted resources each
    assert {e.payload["objective_id"] for e in events} == set(OBJECTIVES)
    assert all(e.payload["resource_id"] for e in events)

    ensure_module(
        DOMAIN_ID, MODULE_ID, minimum=2, engine=engine, ctx=ctx, seams=spy.seams(), domain=domain
    )
    assert len(attached_events(engine)) == 4


def test_second_run_duplicates_nothing_and_pays_for_nothing(
    engine: Engine, domain: Any, ctx: LearnerContext
) -> None:
    """The cost meter is the assertion: the second run must be exactly free."""
    spy = Spy(accept=2, found=3)
    first = ensure_module(
        DOMAIN_ID, MODULE_ID, minimum=2, engine=engine, ctx=ctx, seams=spy.seams(), domain=domain
    )
    calls_after_first = spy.calls
    rows_after_first = len(resources_in(engine))
    reviews_after_first = _review_count(engine)

    assert calls_after_first > 0
    assert all(not r.skipped for r in first)

    second = ensure_module(
        DOMAIN_ID, MODULE_ID, minimum=2, engine=engine, ctx=ctx, seams=spy.seams(), domain=domain
    )

    assert all(r.skipped for r in second)
    assert spy.calls == calls_after_first
    assert spy.model_calls == spy.reviews + spy.authored
    assert len(resources_in(engine)) == rows_after_first
    assert _review_count(engine) == reviews_after_first


def _review_count(engine: Engine) -> int:
    with Session(engine) as session:
        return len(list(session.exec(select(models.ResourceReview)).all()))


# --- dry run ----------------------------------------------------------------


def test_dry_run_makes_zero_calls_and_writes_nothing(
    engine: Engine, domain: Any, ctx: LearnerContext
) -> None:
    spy = Spy()
    results = ensure_module(
        DOMAIN_ID,
        MODULE_ID,
        minimum=2,
        engine=engine,
        ctx=ctx,
        seams=spy.seams(),
        dry_run=True,
        domain=domain,
    )

    assert [r.searched for r in results] == [10, 10]
    assert spy.calls == 0
    assert resources_in(engine) == []
    assert attached_events(engine) == []


def test_dry_run_seams_raise_if_anything_ever_calls_them() -> None:
    """The promise is enforced, not documented."""
    seams = Seams.dry_run()
    spec = ObjectiveSpec(id="obj_one", title="Skill")
    for call in (
        lambda: seams.search(spec, 10),  # type: ignore[misc]
        lambda: seams.fetch(["https://example.test/a"]),  # type: ignore[misc]
        lambda: seams.review(spec, "u", "t", "x"),  # type: ignore[misc]
        lambda: seams.author(spec, []),  # type: ignore[misc]
    ):
        with pytest.raises(DryRunViolation):
            call()


# --- the shipped path -------------------------------------------------------


def _standalone_app(resources_cmd: Any) -> typer.Typer:
    """Wire the command the way ``cli.py`` does.

    A Typer app holding exactly one command collapses into that command, so the
    ``resources`` name would disappear. The real CLI has several commands; this
    keeps the invocation identical to the shipped one.
    """
    app = typer.Typer()
    app.command("resources", help="Fill the corpus for a module.")(resources_cmd.resources)

    @app.command("noop", hidden=True)
    def _noop() -> None:  # pragma: no cover - keeps the app a multi-command group
        pass

    return app


def test_resources_command_dry_run_runs_end_to_end_without_a_key(home: Path) -> None:
    """The command itself, through Typer, with no key and no network."""
    from the_oracle.commands import resources as resources_cmd

    app = _standalone_app(resources_cmd)
    result = CliRunner().invoke(app, ["resources", DOMAIN_ID, "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "dry run" in result.output.lower()
    assert "tokens" in result.output.lower()
    for oid in OBJECTIVES:
        assert oid in result.output


def test_resources_command_rejects_an_unknown_module(home: Path) -> None:
    from the_oracle.commands import resources as resources_cmd

    app = _standalone_app(resources_cmd)
    result = CliRunner().invoke(app, ["resources", DOMAIN_ID, "--module", "nope", "--dry-run"])
    assert result.exit_code == 1


def test_resources_command_registers_on_the_real_cli(home: Path) -> None:
    """Reachability, not decoration.

    Four components in this project were built, unit-tested, and never wired
    in. This asserts the one line ``cli.py`` needs actually produces a working
    ``the-oracle resources`` command on the real application object.
    """
    from the_oracle import cli
    from the_oracle.commands import resources as resources_cmd

    cli.app.command("resources", help="Fill the corpus for a module.")(resources_cmd.resources)
    result = CliRunner().invoke(cli.app, ["resources", DOMAIN_ID, "--dry-run", "--minimum", "3"])
    assert result.exit_code == 0, result.output
    assert DOMAIN_ID in result.output or "Corpus Test Domain" in result.output
