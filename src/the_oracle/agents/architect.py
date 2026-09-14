"""The Curriculum Architect: turn a conversation into a curriculum.

One agent and one build pipeline.

* :class:`Architect` drafts a whole domain from an :class:`Interview`.
* :func:`build_domain` drafts, dedupes against the shared objective library,
  mints ids only for genuinely new objectives, writes them, assembles the
  manifest, and validates it before anything is persisted.

The draft speaks in titles, not ids. Ids are minted only after dedupe has
decided what is genuinely new, because an id is a promise: every domain that
pins it, and every learner's mastery keyed on it, depends on it meaning one
skill forever (PLAN.md section 2).

Nothing is written until the whole manifest validates. A half-written pack is
worse than no pack: it fails at study time, not at build time.

No subject matter lives in this file. Every concrete word comes from the
learner's interview and the model's draft.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, Field

from the_oracle.agents.base import Agent, Usage
from the_oracle.config import Task
from the_oracle.context import LearnerContext
from the_oracle.domains import library
from the_oracle.domains.errors import PackValidationError
from the_oracle.domains.registry import oracle_home
from the_oracle.domains.schema import (
    Domain,
    Edge,
    Misconception,
    Module,
    Objective,
    ObjectiveRef,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    from sqlalchemy import Engine

BLOOM_LEVELS: tuple[str, ...] = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)

#: A draft this small is not a curriculum, it is a gesture.
MIN_OBJECTIVES = 3

#: Rough planning constant: how much of an hour survives contact with life.
STUDY_EFFICIENCY = 0.8


# ---------------------------------------------------------------------------
# Typed agent payloads
# ---------------------------------------------------------------------------


class Interview(BaseModel):
    """What the learner told us before anything was generated."""

    goal: str
    background: str
    hours_per_week: float
    target_weeks: int
    depth: Literal["survey", "working", "practitioner", "expert"]
    must_cover: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)

    @property
    def budget_minutes(self) -> int:
        """Realistic teaching minutes the plan has to fit inside."""
        return int(self.hours_per_week * 60 * self.target_weeks * STUDY_EFFICIENCY)


class DraftObjective(BaseModel):
    """An objective as drafted: a body with no id yet."""

    title: str
    description: str
    bloom: str
    difficulty: int
    est_minutes: int
    assessment_stems: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class DraftDomain(BaseModel):
    """A whole curriculum before dedupe. Every reference is by title."""

    title: str
    description: str
    objectives: list[DraftObjective] = Field(default_factory=list)
    edges: list[tuple[str, str]] = Field(default_factory=list)
    modules: list[dict[str, Any]] = Field(default_factory=list)
    misconceptions: list[dict[str, Any]] = Field(default_factory=list)

    def titles(self) -> list[str]:
        return [o.title for o in self.objectives]


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

ARCHITECT_ROLE = """\
You are the Curriculum Architect. You turn one interview into one curriculum:
a set of learning objectives, the prerequisite edges between them, the modules
that group them, and the wrong mental models a learner of this subject
predictably arrives with.

How to build it:

BLOOM LEVEL IS A PROMISE ABOUT THE ITEM TYPE.
- remember and understand get recall and short-answer items.
- apply and analyze get worked problems.
- evaluate and create get multi-step judgement work.
Pick the level the learner actually needs for the stated goal, then write the
objective so that level is the only honest way to answer it. A goal that ends
in doing the work is not satisfied by `understand`.

PREREQUISITES ARE HARD EDGES, NOT SUGGESTIONS.
An edge from A to B means a learner who has not mastered A cannot honestly
attempt B. Do not add an edge for topical flavour or because two things feel
related. The graph must be acyclic. Every edge endpoint must be an objective
title you listed, spelled identically.

DIFFICULTY IS COGNITIVE LOAD. EST_MINUTES IS LENGTH.
difficulty 1-5 is how many things must be held in the head at once and how
much transfer is required. est_minutes is how long the work takes. A long,
mechanical drill is low difficulty and high minutes. A short proof that will
not fit in working memory is high difficulty and low minutes. Never let one
number stand in for the other.

ASSESSMENT STEMS MUST REVEAL MASTERY, NOT RECALL.
Two or three per objective. A good stem forces the learner to choose, apply,
predict, or diagnose, in a situation they have not been shown. A stem whose
answer can be copied from a definition tests nothing. Avoid "what is" and
"list the". Prefer "given this situation, decide", "predict what happens
when", "this result is wrong - find where".

MODULES are ordered groups. Every objective belongs to exactly one module,
and no module teaches an objective before a prerequisite that sits in an
earlier position. Order the modules so the graph flows forwards.

MISCONCEPTIONS name the wrong model in plain words, list the objectives it
damages by title, and give a diagnostic question that forces it into the open.

Respect the interview: the stated background sets where the curriculum starts,
the hours and weeks set how much fits, and anything the learner excluded stays
out. Titles are short, domain-neutral skill names, not sentences. Descriptions
are one sentence.
"""


class Architect(Agent[Interview, DraftDomain]):
    """Draft one whole domain from one interview."""

    name = "architect"
    task = Task.ARCHITECT
    input_type = Interview
    output_type = DraftDomain
    role = ARCHITECT_ROLE

    async def _run(self, ctx: LearnerContext, payload: Interview) -> tuple[DraftDomain, Usage]:
        draft, usage = await self._run_llm(self.prompt(payload))
        return self.normalise(draft, payload), usage

    def prompt(self, payload: Interview) -> str:
        """The user-side prompt. Deterministic, so cassettes key on it."""
        must = "\n".join(f"- {m}" for m in payload.must_cover) or "- (nothing named)"
        skip = "\n".join(f"- {m}" for m in payload.exclude) or "- (nothing excluded)"
        return (
            f"goal: {payload.goal}\n"
            f"background: {payload.background}\n"
            f"hours_per_week: {payload.hours_per_week}\n"
            f"target_weeks: {payload.target_weeks}\n"
            f"depth: {payload.depth}\n"
            f"teaching minutes available: {payload.budget_minutes}\n"
            f"must cover:\n{must}\n"
            f"exclude:\n{skip}\n\n"
            "Draft the curriculum. Reference every objective by its exact "
            "title in edges, modules, and misconceptions. Do not invent ids. "
            "Keep the total of est_minutes inside the teaching minutes above."
        )

    def normalise(self, draft: DraftDomain, payload: Interview) -> DraftDomain:
        """Force the draft back onto the schema the library will accept.

        This cleans shape, never pedagogy. A bad graph stays bad here and is
        rejected later, loudly, with the reason named.
        """
        seen: set[str] = set()
        objectives: list[DraftObjective] = []
        for obj in draft.objectives:
            title = " ".join(obj.title.split())
            if not title or title.casefold() in seen:
                continue
            seen.add(title.casefold())
            bloom = obj.bloom.strip().casefold()
            objectives.append(
                obj.model_copy(
                    update={
                        "title": title,
                        "description": " ".join(obj.description.split()),
                        "bloom": bloom if bloom in BLOOM_LEVELS else "understand",
                        "difficulty": max(1, min(5, obj.difficulty)),
                        "est_minutes": max(1, obj.est_minutes),
                        "assessment_stems": [s.strip() for s in obj.assessment_stems if s.strip()],
                        "tags": sorted({t.strip().casefold() for t in obj.tags if t.strip()}),
                    }
                )
            )
        edges = []
        for pair in draft.edges:
            a, b = (" ".join(str(x).split()) for x in pair)
            if a and b and (a, b) not in edges:
                edges.append((a, b))
        return draft.model_copy(
            update={
                "title": " ".join(draft.title.split()) or payload.goal.strip(),
                "description": " ".join(draft.description.split()),
                "objectives": objectives,
                "edges": edges,
            }
        )


# ---------------------------------------------------------------------------
# Slugs and ids
# ---------------------------------------------------------------------------


def slugify(text: str, *, max_words: int = 6) -> str:
    """A stable snake_case id fragment. ASCII, lowercase, no punctuation."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    words = [w for w in re.split(r"[^a-zA-Z0-9]+", folded.casefold()) if w]
    if not words:
        return "untitled"
    return "_".join(words[:max_words])


def _unique(base: str, taken: "Iterable[str]") -> str:
    """``base``, or ``base_2``, ``base_3`` ... until it collides with nothing."""
    used = set(taken)
    if base not in used:
        return base
    for n in range(2, 1000):
        candidate = f"{base}_{n}"
        if candidate not in used:
            return candidate
    raise ValueError(f"cannot mint a unique id from {base!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# The plan: everything decided, nothing written
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DomainPlan:
    """The full result of drafting and deduping, before any write.

    The CLI shows this to the learner and only then commits, because a reuse
    decision is the one thing in this pipeline a human should get to veto.
    """

    interview: Interview
    draft: DraftDomain
    domain: Domain
    matches: list[Any] = field(default_factory=list)
    reused: dict[str, str] = field(default_factory=dict)
    minted: dict[str, Objective] = field(default_factory=dict)
    resolved: dict[str, Objective] = field(default_factory=dict)

    @property
    def total_minutes(self) -> int:
        return sum(self.resolved[ref.id].est_minutes for ref in self.domain.objectives)

    def score_for(self, title: str) -> float:
        for match in self.matches:
            if getattr(match, "candidate_title", None) == title:
                return float(getattr(match, "score", 0.0))
        return 0.0


def _load_dedupe() -> Any:
    """Import the dedupe module lazily. It is another worker's file."""
    from the_oracle.domains import dedupe

    return dedupe


def draft_domain(
    interview: Interview,
    *,
    engine: "Engine | None" = None,
    learner_id: str | None = None,
) -> DraftDomain:
    """Run the Architect once. One LLM call, and it is not a cheap one."""
    return asyncio.run(draft_domain_async(interview, engine=engine, learner_id=learner_id))


async def draft_domain_async(
    interview: Interview,
    *,
    engine: "Engine | None" = None,
    learner_id: str | None = None,
) -> DraftDomain:
    """Async body of :func:`draft_domain`."""
    ctx = (
        LearnerContext.for_learner(learner_id) if learner_id else LearnerContext.resolve()
    )
    architect = Architect(engine)
    result = await architect.run(ctx, interview)
    return result.output


def plan_domain(
    draft: DraftDomain,
    interview: Interview,
    *,
    embedder: Any = None,
    judge: Any = None,
    domain_id: str | None = None,
) -> DomainPlan:
    """Dedupe the draft against the library and assemble the manifest.

    Pure with respect to the filesystem: it reads the library and writes
    nothing. Raises :class:`PackValidationError` if the draft cannot become a
    valid pack, naming exactly what the Architect got wrong.
    """
    provisional_id = domain_id or _mint_domain_id(draft.title)
    problems = _draft_problems(draft)
    if problems:
        raise PackValidationError(provisional_id, problems)

    dedupe = _load_dedupe()
    existing = list(library.all_objectives())
    candidates = [
        Objective(
            id=f"draft_{slugify(o.title)}",
            version=1,
            title=o.title,
            description=o.description,
            bloom=o.bloom,  # type: ignore[arg-type]
            difficulty=o.difficulty,
            est_minutes=o.est_minutes,
            assessment_stems=o.assessment_stems,
            tags=o.tags,
        )
        for o in draft.objectives
    ]
    matches = list(dedupe.match_all(candidates, existing, embedder=embedder))
    resolution: dict[str, str | None] = dict(dedupe.resolve(matches, judge=judge))

    by_id = {obj.id: obj for obj in existing}
    taken = set(by_id)
    reused: dict[str, str] = {}
    minted: dict[str, Objective] = {}
    resolved: dict[str, Objective] = {}
    ids: dict[str, str] = {}

    for draft_obj, candidate in zip(draft.objectives, candidates, strict=True):
        title = draft_obj.title
        reuse_id = resolution.get(title)
        if reuse_id and reuse_id in by_id:
            ids[title] = reuse_id
            reused[title] = reuse_id
            resolved[reuse_id] = by_id[reuse_id]
            continue
        new_id = _unique(slugify(title), taken)
        taken.add(new_id)
        body = candidate.model_copy(update={"id": new_id})
        ids[title] = new_id
        minted[title] = body
        resolved[new_id] = body

    domain = _assemble(draft, ids, provisional_id)
    _reject_invalid(domain, resolved, interview)
    return DomainPlan(
        interview=interview,
        draft=draft,
        domain=domain.bind(resolved),
        matches=matches,
        reused=reused,
        minted=minted,
        resolved=resolved,
    )


def commit(plan: DomainPlan) -> Domain:
    """Write the new objectives, then the manifest. Roll back on any failure.

    The manifest is written last and verified by a full reload. If that
    reload fails the manifest is removed again, so the registry never sees a
    broken pack.
    """
    from the_oracle.domains.registry import load_domain

    settings_home = oracle_home()
    settings_home.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for body in plan.minted.values():
        library.put(body)
        written.append(body.id)

    path = settings_home / "domains" / f"{plan.domain.id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = plan.domain.model_dump(mode="json", by_alias=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    try:
        return load_domain(plan.domain.id)
    except PackValidationError:
        path.unlink(missing_ok=True)
        for objective_id in written:
            (settings_home / "objectives" / f"{objective_id}.yaml").unlink(missing_ok=True)
        raise


def build_domain(
    interview: Interview,
    *,
    embedder: Any = None,
    judge: Any = None,
    engine: "Engine | None" = None,
) -> tuple[Domain, list[Any]]:
    """Interview in, validated domain pack out.

    Drafts, dedupes against the shared library, mints ids only for genuinely
    new objectives, writes those objectives, assembles the manifest, and
    validates the whole thing. Nothing is persisted unless everything holds.
    """
    draft = draft_domain(interview, engine=engine)
    plan = plan_domain(draft, interview, embedder=embedder, judge=judge)
    domain = commit(plan)
    return domain, plan.matches


# ---------------------------------------------------------------------------
# Assembly and validation
# ---------------------------------------------------------------------------


def _mint_domain_id(title: str) -> str:
    from the_oracle.domains.registry import list_domains

    return _unique(slugify(title), list_domains())


def _draft_problems(draft: DraftDomain) -> list[str]:
    """Everything wrong with the draft that can be seen before dedupe."""
    problems: list[str] = []
    titles = draft.titles()
    known = set(titles)
    if len(titles) < MIN_OBJECTIVES:
        problems.append(
            f"draft: the Architect produced {len(titles)} objectives; "
            f"a curriculum needs at least {MIN_OBJECTIVES}"
        )
    for a, b in draft.edges:
        for role, title in (("from", a), ("to", b)):
            if title not in known:
                problems.append(
                    f"draft: edge {a!r}->{b!r} {role} endpoint {title!r} is not "
                    "one of the drafted objective titles"
                )
        if a == b:
            problems.append(f"draft: objective {a!r} is its own prerequisite")
    for module in draft.modules:
        for title in module.get("objectives", []):
            if title not in known:
                problems.append(
                    f"draft: module {module.get('id', module.get('title'))!r} lists "
                    f"unknown objective title {title!r}"
                )
    for mis in draft.misconceptions:
        for title in mis.get("objectives", []):
            if title not in known:
                problems.append(
                    f"draft: misconception {mis.get('id', mis.get('wrong_model'))!r} "
                    f"references unknown objective title {title!r}"
                )
    return problems


def _assemble(draft: DraftDomain, ids: dict[str, str], domain_id: str) -> Domain:
    """Build the manifest, translating every title into its settled id."""
    modules: list[Module] = []
    placed: set[str] = set()
    for index, module in enumerate(draft.modules, start=1):
        objective_ids = [ids[t] for t in module.get("objectives", []) if t in ids]
        objective_ids = [oid for oid in dict.fromkeys(objective_ids) if oid not in placed]
        placed.update(objective_ids)
        raw_id = str(module.get("id") or module.get("title") or f"module {index}")
        modules.append(
            Module(
                id=f"m{index:02d}_{slugify(raw_id)}",
                title=str(module.get("title") or raw_id),
                goal=str(module.get("goal") or ""),
                objectives=objective_ids,
            )
        )

    misconceptions: list[Misconception] = []
    for index, mis in enumerate(draft.misconceptions, start=1):
        raw_id = str(mis.get("id") or mis.get("wrong_model") or f"misconception {index}")
        misconceptions.append(
            Misconception(
                id=f"mc_{slugify(raw_id)}",
                wrong_model=str(mis.get("wrong_model") or ""),
                objectives=[ids[t] for t in mis.get("objectives", []) if t in ids],
                diagnostic=str(mis.get("diagnostic") or ""),
            )
        )

    edges = [
        Edge(from_id=ids[a], to_id=ids[b])
        for a, b in draft.edges
        if a in ids and b in ids and ids[a] != ids[b]
    ]
    domain = Domain(
        id=domain_id,
        version=1,
        title=draft.title,
        description=draft.description,
        objectives=[
            ObjectiveRef(id=oid, version=1) for oid in dict.fromkeys(ids.values())
        ],
        edges=list(dict.fromkeys(edges)),
        modules=modules,
        misconceptions=misconceptions,
    )
    return _repair_module_order(domain)


def _repair_module_order(domain: Domain) -> Domain:
    """Sort each module's objectives into a prerequisite-respecting order.

    Order *inside* a module is bookkeeping, not pedagogy, so it is safe to
    fix. Order *between* modules is a real design decision, so a violation
    there is reported and the pack is rejected.
    """
    modules: list[Module] = []
    for module in domain.modules:
        members = list(module.objectives)
        inside = {m: i for i, m in enumerate(members)}
        indegree = dict.fromkeys(members, 0)
        successors: dict[str, list[str]] = {m: [] for m in members}
        for edge in domain.edges:
            if edge.from_id in inside and edge.to_id in inside:
                successors[edge.from_id].append(edge.to_id)
                indegree[edge.to_id] += 1
        ready = sorted((m for m in members if indegree[m] == 0), key=inside.__getitem__)
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for nxt in successors[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)
            ready.sort(key=inside.__getitem__)
        if len(order) != len(members):  # a cycle inside the module; leave it to be rejected
            order = members
        modules.append(module.model_copy(update={"objectives": order}))
    return domain.model_copy(update={"modules": modules})


def manifest_problems(domain: Domain, resolved: dict[str, Objective]) -> list[str]:
    """The six pack rules (docs/pack-format.md), checked entirely in memory.

    The registry checks the same rules against files on disk. This runs before
    any file exists, which is the whole point: a pack that cannot pass is never
    written.
    """
    problems: list[str] = []
    declared: set[str] = set()
    for ref in domain.objectives:
        if ref.id in declared:
            problems.append(f"rule1: objective {ref.id!r} referenced twice")
            continue
        declared.add(ref.id)
        body = resolved.get(ref.id)
        if body is None:
            problems.append(f"rule1: objective {ref.id!r} has no body to write")
        elif body.version != ref.version:
            problems.append(
                f"rule1: {ref.id!r} pinned at version {ref.version} but the body "
                f"is version {body.version}"
            )

    for edge in domain.edges:
        for role, oid in (("from", edge.from_id), ("to", edge.to_id)):
            if oid not in declared:
                problems.append(
                    f"rule2: edge {edge.from_id}->{edge.to_id} {role} endpoint "
                    f"{oid!r} is not in objectives"
                )
        if edge.from_id == edge.to_id:
            problems.append(f"rule3: self-loop on {edge.from_id!r}")

    cycle = _find_cycle(declared, domain)
    if cycle:
        problems.append("rule3: cycle detected: " + " -> ".join(cycle))

    placements: dict[str, list[str]] = {oid: [] for oid in declared}
    for module in domain.modules:
        for oid in module.objectives:
            if oid not in declared:
                problems.append(
                    f"rule4: module {module.id!r} lists unknown objective {oid!r}"
                )
                continue
            placements[oid].append(module.id)
    for oid in sorted(declared):
        where = placements[oid]
        if not where:
            problems.append(f"rule4: objective {oid!r} appears in no module")
        elif len(where) > 1:
            problems.append(f"rule4: objective {oid!r} appears in modules {', '.join(where)}")

    position = {
        oid: i for i, oid in enumerate(o for m in domain.modules for o in m.objectives)
    }
    for edge in domain.edges:
        a, b = position.get(edge.from_id), position.get(edge.to_id)
        if a is not None and b is not None and a > b:
            problems.append(
                f"rule5: module order teaches {edge.to_id!r} before its "
                f"prerequisite {edge.from_id!r}"
            )

    for mis in domain.misconceptions:
        for oid in mis.objectives:
            if oid not in declared:
                problems.append(
                    f"rule6: misconception {mis.id!r} references unknown objective {oid!r}"
                )
    return problems


def _find_cycle(declared: set[str], domain: Domain) -> list[str]:
    """The first prerequisite cycle found, as a path. Empty means acyclic."""
    successors: dict[str, list[str]] = {oid: [] for oid in declared}
    for edge in domain.edges:
        if edge.from_id in successors and edge.to_id in successors:
            successors[edge.from_id].append(edge.to_id)
    white, grey, black = 0, 1, 2
    color = dict.fromkeys(successors, white)

    def walk(node: str, stack: list[str]) -> list[str]:
        color[node] = grey
        stack.append(node)
        for nxt in successors[node]:
            if color[nxt] == grey:
                return [*stack[stack.index(nxt) :], nxt]
            if color[nxt] == white:
                found = walk(nxt, stack)
                if found:
                    return found
        stack.pop()
        color[node] = black
        return []

    for node in sorted(successors):
        if color[node] == white:
            found = walk(node, [])
            if found:
                return found
    return []


def _reject_invalid(
    domain: Domain, resolved: dict[str, Objective], interview: Interview
) -> None:
    """Run every load-time rule in memory. Raise before anything is written."""
    problems = manifest_problems(domain, resolved)
    total = sum(resolved[ref.id].est_minutes for ref in domain.objectives)
    if total > interview.budget_minutes * 2:
        problems.append(
            f"budget: the draft needs {total} minutes but the learner has about "
            f"{interview.budget_minutes}; that is more than twice the time available"
        )
    if problems:
        raise PackValidationError(domain.id, problems)
