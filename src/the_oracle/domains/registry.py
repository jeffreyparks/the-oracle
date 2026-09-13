"""Discover, load, and validate domain packs from the Oracle home directory."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from the_oracle.domains import library
from the_oracle.domains.errors import ObjectiveNotFoundError, PackValidationError
from the_oracle.domains.schema import Domain, Objective

DEFAULT_HOME = "~/.the-oracle"


def oracle_home() -> Path:
    """Root directory for packs and state. Overridable by ORACLE_HOME."""
    return Path(os.environ.get("ORACLE_HOME", DEFAULT_HOME)).expanduser()


def _domains_dir() -> Path:
    return oracle_home() / "domains"


def list_domains() -> list[str]:
    """Ids of every domain manifest present, sorted."""
    directory = _domains_dir()
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def _parse(domain_id: str) -> Domain:
    path = _domains_dir() / f"{domain_id}.yaml"
    if not path.is_file():
        raise PackValidationError(domain_id, [f"no manifest at {path}"])
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PackValidationError(domain_id, [f"unreadable YAML: {exc}"]) from exc
    try:
        domain = Domain.model_validate(data)
    except ValidationError as exc:
        problems = [
            f"schema: {'.'.join(str(x) for x in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        ]
        raise PackValidationError(domain_id, problems) from exc
    if domain.id != domain_id:
        raise PackValidationError(
            domain_id, [f"manifest declares id {domain.id!r} but lives at {path.name}"]
        )
    return domain


def _check(domain: Domain) -> tuple[list[str], dict[str, Objective]]:
    """Run all six load-time rules. Returns (problems, resolved objectives)."""
    problems: list[str] = []
    resolved: dict[str, Objective] = {}

    # Rule 1: every pinned reference resolves in the library at that version.
    declared: set[str] = set()
    for ref in domain.objectives:
        if ref.id in declared:
            problems.append(f"rule1: objective {ref.id!r} referenced twice")
            continue
        declared.add(ref.id)
        try:
            resolved[ref.id] = library.get(ref.id, ref.version)
        except (ObjectiveNotFoundError, ValidationError) as exc:
            problems.append(f"rule1: {exc}")

    # Rule 2: every edge endpoint is declared by the domain.
    for edge in domain.edges:
        for role, oid in (("from", edge.from_id), ("to", edge.to_id)):
            if oid not in declared:
                problems.append(
                    f"rule2: edge {edge.from_id}->{edge.to_id} {role} "
                    f"endpoint {oid!r} is not in objectives"
                )
        if edge.from_id == edge.to_id:
            problems.append(f"rule3: self-loop on {edge.from_id!r}")

    # Rule 3: the edge graph is acyclic.
    cycle = _find_cycle(declared, domain)
    if cycle:
        problems.append("rule3: cycle detected: " + " -> ".join(cycle))

    # Rule 4: every objective appears in exactly one module.
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
            problems.append(
                f"rule4: objective {oid!r} appears in modules {', '.join(where)}"
            )

    # Rule 5: module order never violates an edge.
    position = {
        oid: i
        for i, oid in enumerate(o for m in domain.modules for o in m.objectives)
    }
    for edge in domain.edges:
        a, b = position.get(edge.from_id), position.get(edge.to_id)
        if a is not None and b is not None and a > b:
            problems.append(
                f"rule5: module order teaches {edge.to_id!r} before its "
                f"prerequisite {edge.from_id!r}"
            )

    # Rule 6: every misconception objective ref resolves.
    for mis in domain.misconceptions:
        for oid in mis.objectives:
            if oid not in declared:
                problems.append(
                    f"rule6: misconception {mis.id!r} references unknown "
                    f"objective {oid!r}"
                )

    return problems, resolved


def _find_cycle(declared: set[str], domain: Domain) -> list[str]:
    successors: dict[str, list[str]] = {oid: [] for oid in declared}
    for edge in domain.edges:
        if edge.from_id in successors and edge.to_id in successors:
            successors[edge.from_id].append(edge.to_id)

    WHITE, GREY, BLACK = 0, 1, 2
    color = dict.fromkeys(successors, WHITE)

    def walk(node: str, stack: list[str]) -> list[str]:
        color[node] = GREY
        stack.append(node)
        for nxt in successors[node]:
            if color[nxt] == GREY:
                return [*stack[stack.index(nxt) :], nxt]
            if color[nxt] == WHITE:
                found = walk(nxt, stack)
                if found:
                    return found
        stack.pop()
        color[node] = BLACK
        return []

    for node in sorted(successors):
        if color[node] == WHITE:
            found = walk(node, [])
            if found:
                return found
    return []


def load_domain(domain_id: str) -> Domain:
    """Load and fully validate a domain. Raises PackValidationError, never half-loads."""
    domain = _parse(domain_id)
    problems, resolved = _check(domain)
    if problems:
        raise PackValidationError(domain_id, problems)
    return domain.bind(resolved)


def validate_domain(domain_id: str) -> list[str]:
    """Every problem with a domain pack. An empty list means valid."""
    try:
        domain = _parse(domain_id)
    except PackValidationError as exc:
        return exc.problems
    problems, _ = _check(domain)
    return problems
