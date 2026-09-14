"""The lazy resource pipeline.

Per PLAN.md section 7, one objective at a time:

``already has enough -> skip`` -> ``Scout 10`` -> ``fetch`` -> ``Reviewer`` ->
``accepted >= minimum ? attach : Author writes a lesson -> Reviewer checks it -> attach``

The first line is the whole point. Nothing is fetched for a module the learner
has not reached, and nothing is fetched twice for an objective that already has
teachable material. Running :func:`ensure_module` again is close to free: the
skip fires before any seam is touched, and anything that does run goes through
the agent idempotency cache.

**Seams, not monkeypatching.** Scout, fetch, Reviewer and Author reach this
module through :class:`Seams`, a frozen record of four callables. The defaults
import the real agents lazily, inside the call, so this module imports cleanly
even while those files are being written. Tests pass fakes instead, and never
need the real implementations or a network.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine

from the_oracle.corpus import store
from the_oracle.store.db import create_all, get_engine
from the_oracle.store.events import EventKind, EventLog

if TYPE_CHECKING:  # pragma: no cover
    from the_oracle.context import LearnerContext
    from the_oracle.domains.schema import Domain

__all__ = [
    "DEFAULT_CANDIDATES",
    "MAX_REVIEW_CHARS",
    "ObjectiveResult",
    "ObjectiveSpec",
    "Seams",
    "ensure_domain",
    "ensure_module",
    "ensure_objective",
]

DEFAULT_CANDIDATES = 10
"""Ten candidates per objective, per the planning decision in PLAN.md section 7."""

MAX_REVIEW_CHARS = 12_000
"""The caller truncates page text before the Reviewer sees it, per the contract."""


# --------------------------------------------------------------------------
# Value types
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    """What happened for one objective. The unit the CLI renders."""

    objective_id: str
    searched: int = 0
    fetched: int = 0
    accepted: int = 0
    rejected: int = 0
    authored: bool = False
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    """Everything a seam needs about one objective, with no ORM or pack types.

    Built once from the domain pack, then handed to every seam. Fakes in tests
    receive exactly what the real Scout, Reviewer and Author receive.
    """

    id: str
    title: str
    description: str = ""
    bloom: str = "understand"
    difficulty: int = 1
    est_minutes: int = 30
    assessment_stems: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    misconceptions: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_domain(cls, domain: "Domain", objective_id: str) -> "ObjectiveSpec":
        objective = domain.objective(objective_id)
        misconceptions = tuple(
            {"id": m.id, "wrong_model": m.wrong_model, "diagnostic": m.diagnostic}
            for m in getattr(domain, "misconceptions", [])
            if objective_id in getattr(m, "objectives", [])
        )
        return cls(
            id=objective.id,
            title=objective.title,
            description=objective.description,
            bloom=str(objective.bloom),
            difficulty=int(objective.difficulty),
            est_minutes=int(objective.est_minutes),
            assessment_stems=tuple(objective.assessment_stems),
            tags=tuple(objective.tags),
            misconceptions=misconceptions,
        )


SearchFn = Callable[[ObjectiveSpec, int], Sequence[Any]]
FetchFn = Callable[[Sequence[str]], Sequence[Any]]
ReviewFn = Callable[..., Any]
"""``(spec, url, title, text, *, word_count=None, fetch_error=None) -> Review``."""
AuthorFn = Callable[[ObjectiveSpec, Sequence[dict[str, Any]]], Any]


class DryRunViolation(RuntimeError):
    """Raised if anything tries to search, fetch, or call a model in a dry run."""


def _forbidden(name: str) -> Callable[..., Any]:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise DryRunViolation(f"--dry-run must not {name}; this call is a bug, not a warning")

    return _raise


@dataclass(frozen=True, slots=True)
class Seams:
    """The four injection points. Leave one ``None`` to use the real agent.

    ``search(spec, limit) -> candidates``      (``agents.scout.search_candidates``)
    ``fetch(urls) -> fetched pages``           (``corpus.fetch.fetch_many``)
    ``review(spec, url, title, text) -> Review``(``agents.reviewer.Reviewer``)
    ``author(spec, accepted_sources) -> Lesson``(``agents.author.Author``)

    Every return value is duck-typed against the frozen Phase 3 contract, so a
    test fake needs no import from ``agents``.
    """

    search: SearchFn | None = None
    fetch: FetchFn | None = None
    review: ReviewFn | None = None
    author: AuthorFn | None = None

    @classmethod
    def dry_run(cls) -> "Seams":
        """Seams that make a dry run's no-call promise enforceable, not advisory."""
        return cls(
            search=_forbidden("search"),
            fetch=_forbidden("fetch the network"),
            review=_forbidden("call a model"),
            author=_forbidden("call a model"),
        )

    def bound(self, ctx: "LearnerContext", engine: Engine) -> "Seams":
        """Fill unset seams with the real implementations, imported lazily."""
        return replace(
            self,
            search=self.search or _real_search,
            fetch=self.fetch or _real_fetch,
            review=self.review or _real_review(ctx, engine),
            author=self.author or _real_author(ctx, engine),
        )


# --------------------------------------------------------------------------
# Default seams - the shipped path
# --------------------------------------------------------------------------


def _run_sync(coro: Any) -> Any:
    """Await a coroutine from sync code, loop or no loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _real_search(spec: ObjectiveSpec, limit: int) -> Sequence[Any]:
    from the_oracle.agents.scout import ScoutRequest, search_candidates

    request = ScoutRequest(
        objective_id=spec.id,
        title=spec.title,
        description=spec.description,
        bloom=spec.bloom,
        difficulty=spec.difficulty,
        tags=list(spec.tags),
    )
    return search_candidates(request, limit=limit)


def _real_fetch(urls: Sequence[str]) -> Sequence[Any]:
    from the_oracle.corpus.fetch import fetch_many

    return fetch_many(list(urls))


def _real_review(ctx: "LearnerContext", engine: Engine) -> ReviewFn:
    def _review(
        spec: ObjectiveSpec,
        url: str,
        title: str,
        text: str,
        *,
        word_count: int | None = None,
        fetch_error: str | None = None,
    ) -> Any:
        from the_oracle.agents.reviewer import ReviewRequest, Reviewer

        # word_count and fetch_error are what make a dead or thin page a
        # zero-token reject, so they are passed through rather than recomputed.
        request = ReviewRequest(
            objective_title=spec.title,
            objective_description=spec.description,
            bloom=spec.bloom,
            difficulty=spec.difficulty,
            url=url,
            title=title,
            text=text[:MAX_REVIEW_CHARS],
            word_count=word_count,
            fetch_error=fetch_error,
        )
        result = _run_sync(Reviewer(engine).run(ctx, request, objective_id=spec.id))
        return result.output

    return _review


def _real_author(ctx: "LearnerContext", engine: Engine) -> AuthorFn:
    def _author(spec: ObjectiveSpec, accepted_sources: Sequence[dict[str, Any]]) -> Any:
        from the_oracle.agents.author import Author, LessonRequest

        request = LessonRequest(
            objective_id=spec.id,
            title=spec.title,
            description=spec.description,
            bloom=spec.bloom,
            difficulty=spec.difficulty,
            est_minutes=spec.est_minutes,
            assessment_stems=list(spec.assessment_stems),
            misconceptions=[dict(m) for m in spec.misconceptions],
            accepted_sources=[dict(s) for s in accepted_sources],
        )
        result = _run_sync(Author(engine).run(ctx, request, objective_id=spec.id))
        return result.output

    return _author


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


@dataclass(slots=True)
class _Tally:
    searched: int = 0
    fetched: int = 0
    accepted: int = 0
    rejected: int = 0
    authored: bool = False
    notes: list[str] = field(default_factory=list)


def _attach(
    log: EventLog,
    learner_id: str,
    spec: ObjectiveSpec,
    resource_id: str,
    *,
    kind: str,
    url: str | None,
    score: float,
) -> None:
    """One accepted resource, one ``resource.attached`` event. No exceptions."""
    log.append(
        EventKind.RESOURCE_ATTACHED,
        learner_id,
        {
            "objective_id": spec.id,
            "resource_id": resource_id,
            "kind": kind,
            "url": url,
            "score": round(float(score), 3),
        },
    )


def _accepted_sources(objective_id: str, engine: Engine) -> list[dict[str, Any]]:
    return [
        {
            "url": resource.url or "",
            "title": resource.title or "",
            "note": str((resource.meta or {}).get("notes", "") or "")[:400],
        }
        for resource in store.accepted_for(objective_id, engine)
        if resource.url
    ]


def ensure_objective(
    objective_id: str,
    domain: "Domain",
    *,
    minimum: int = 2,
    engine: Engine | None = None,
    ctx: "LearnerContext | None" = None,
    seams: Seams | None = None,
    limit: int = DEFAULT_CANDIDATES,
    dry_run: bool = False,
    log: EventLog | None = None,
) -> ObjectiveResult:
    """Make sure one objective has teachable material. Lazy and idempotent."""
    from the_oracle.context import LearnerContext

    engine = engine or get_engine()
    create_all(engine)
    ctx = ctx or LearnerContext.resolve()
    spec = ObjectiveSpec.from_domain(domain, objective_id)

    already = store.accepted_for(objective_id, engine)
    if store.has_enough(objective_id, minimum=minimum, engine=engine):
        return ObjectiveResult(objective_id, accepted=len(already), skipped=True)

    if dry_run:
        # No seam is bound and no seam is called. The promise is structural.
        return ObjectiveResult(
            objective_id,
            searched=limit,
            fetched=0,
            accepted=len(already),
            rejected=0,
            authored=False,
            skipped=False,
        )

    bound = (seams or Seams()).bound(ctx, engine)
    log = log or EventLog(engine)
    tally = _Tally(accepted=len(already))

    # 1. Scout. A scout failure is not a module failure: degrade to the Author.
    candidates: Sequence[Any] = ()
    try:
        candidates = bound.search(spec, limit) or ()  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001 - any scout failure degrades
        tally.notes.append(f"scout unavailable: {exc}")
    tally.searched = len(candidates)

    # 2. Fetch, then 3. review each page.
    by_url = {str(getattr(c, "url", "") or ""): c for c in candidates}
    urls = [url for url in by_url if url]
    pages: Sequence[Any] = ()
    if urls:
        try:
            pages = bound.fetch(urls) or ()  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - fetch must never kill a module
            tally.notes.append(f"fetch failed: {exc}")

    for page in pages:
        url = str(getattr(page, "final_url", "") or getattr(page, "url", "") or "")
        if not url:
            continue
        tally.fetched += 1
        candidate = by_url.get(str(getattr(page, "url", "") or ""))
        text = str(getattr(page, "text", "") or "")
        title = str(getattr(page, "title", "") or getattr(candidate, "title", "") or "")
        resource_id = store.put_resource(
            spec.id,
            kind="found",
            url=url,
            title=title,
            body=text[:MAX_REVIEW_CHARS],
            meta={
                "source": str(getattr(candidate, "source", "") or ""),
                "rank": int(getattr(candidate, "rank", 0) or 0),
                "snippet": str(getattr(candidate, "snippet", "") or ""),
                "status": int(getattr(page, "status", 0) or 0),
                "word_count": int(getattr(page, "word_count", 0) or 0),
                "fetch_error": getattr(page, "error", None),
            },
            engine=engine,
        )
        try:
            review = bound.review(  # type: ignore[misc]
                spec,
                url,
                title,
                text,
                word_count=int(getattr(page, "word_count", 0) or 0),
                fetch_error=getattr(page, "error", None),
            )
        except Exception as exc:  # noqa: BLE001 - one bad page is not a failure
            tally.notes.append(f"review failed for {url}: {exc}")
            tally.rejected += 1
            continue
        store.record_review(resource_id, review, engine)
        if str(getattr(review, "verdict", "reject")) == "accept":
            tally.accepted += 1
            _attach(
                log,
                ctx.learner_id,
                spec,
                resource_id,
                kind="found",
                url=url,
                score=float(getattr(review, "score", 0.0) or 0.0),
            )
        else:
            tally.rejected += 1

    # 4. Not enough accepted material? The Author writes the lesson.
    if tally.accepted < max(1, int(minimum)):
        _author_lesson(spec, bound, engine, log, ctx, tally)

    return ObjectiveResult(
        objective_id=spec.id,
        searched=tally.searched,
        fetched=tally.fetched,
        accepted=tally.accepted,
        rejected=tally.rejected,
        authored=tally.authored,
        skipped=False,
    )


def _author_lesson(
    spec: ObjectiveSpec,
    bound: Seams,
    engine: Engine,
    log: EventLog,
    ctx: "LearnerContext",
    tally: _Tally,
) -> None:
    """Write, review, and attach one lesson. Failures are recorded, not raised."""
    try:
        lesson = bound.author(spec, _accepted_sources(spec.id, engine))  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001 - the module keeps going
        tally.notes.append(f"author failed: {exc}")
        return
    if lesson is None:
        return

    resource_id = store.put_lesson(spec.id, lesson, engine)
    store.put_items(spec.id, getattr(lesson, "exercises", []) or [], engine)
    tally.authored = True

    body = store.render_lesson(lesson)
    try:
        review = bound.review(spec, "", f"Authored lesson for {spec.id}", body)  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001
        tally.notes.append(f"lesson review failed: {exc}")
        return
    store.record_review(resource_id, review, engine)
    if str(getattr(review, "verdict", "reject")) == "accept":
        tally.accepted += 1
        _attach(
            log,
            ctx.learner_id,
            spec,
            resource_id,
            kind="authored",
            url=None,
            score=float(getattr(review, "score", 0.0) or 0.0),
        )
    else:
        tally.rejected += 1


def _module_objectives(domain: "Domain", module_id: str) -> list[str]:
    for module in domain.modules:
        if module.id == module_id:
            return list(module.objectives)
    raise KeyError(f"{module_id!r} is not a module of domain {domain.id!r}")


def ensure_module(
    domain_id: str,
    module_id: str,
    *,
    minimum: int = 2,
    engine: Engine | None = None,
    ctx: "LearnerContext | None" = None,
    seams: Seams | None = None,
    limit: int = DEFAULT_CANDIDATES,
    dry_run: bool = False,
    domain: "Domain | None" = None,
    log: EventLog | None = None,
) -> list[ObjectiveResult]:
    """Ensure every objective in one module. Safe to run twice."""
    if domain is None:
        from the_oracle.domains.registry import load_domain

        domain = load_domain(domain_id)
    engine = engine or get_engine()
    create_all(engine)
    log = log or EventLog(engine)
    return [
        ensure_objective(
            objective_id,
            domain,
            minimum=minimum,
            engine=engine,
            ctx=ctx,
            seams=seams,
            limit=limit,
            dry_run=dry_run,
            log=log,
        )
        for objective_id in _module_objectives(domain, module_id)
    ]


def ensure_domain(
    domain_id: str,
    *,
    minimum: int = 2,
    engine: Engine | None = None,
    ctx: "LearnerContext | None" = None,
    seams: Seams | None = None,
    limit: int = DEFAULT_CANDIDATES,
    dry_run: bool = False,
    domain: "Domain | None" = None,
) -> dict[str, list[ObjectiveResult]]:
    """Every module of a domain, in teaching order. Used by ``resources`` with no ``--module``."""
    if domain is None:
        from the_oracle.domains.registry import load_domain

        domain = load_domain(domain_id)
    engine = engine or get_engine()
    create_all(engine)
    log = EventLog(engine)
    return {
        module.id: ensure_module(
            domain_id,
            module.id,
            minimum=minimum,
            engine=engine,
            ctx=ctx,
            seams=seams,
            limit=limit,
            dry_run=dry_run,
            domain=domain,
            log=log,
        )
        for module in domain.modules
    }
