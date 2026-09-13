"""Pydantic models for the domain pack format (see docs/pack-format.md)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

BloomLevel = Literal[
    "remember", "understand", "apply", "analyze", "evaluate", "create"
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Objective(_Base):
    """A subject-neutral, shared, versioned learning atom."""

    id: str
    version: int = Field(ge=1)
    title: str
    description: str
    bloom: BloomLevel
    difficulty: int = Field(ge=1, le=5)
    est_minutes: int = Field(ge=1)
    assessment_stems: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ObjectiveRef(_Base):
    """A pinned reference from a domain manifest into the library."""

    id: str
    version: int = Field(ge=1)


class Edge(_Base):
    """A prerequisite edge. ``from_id`` must be taught before ``to_id``."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    from_id: str = Field(alias="from")
    to_id: str = Field(alias="to")


class Module(_Base):
    """An ordered group of objectives inside one domain."""

    id: str
    title: str
    goal: str
    objectives: list[str] = Field(default_factory=list)


class Misconception(_Base):
    """A named wrong mental model the Assessor detects."""

    id: str
    wrong_model: str
    objectives: list[str] = Field(default_factory=list)
    diagnostic: str


class Domain(_Base):
    """A manifest: a view over the shared objective library."""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: int = Field(ge=1)
    title: str
    description: str = ""
    objectives: list[ObjectiveRef] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    modules: list[Module] = Field(default_factory=list)
    misconceptions: list[Misconception] = Field(default_factory=list)

    # Populated by the registry once the library resolves cleanly.
    _resolved: dict[str, Objective] = {}

    def bind(self, resolved: dict[str, Objective]) -> "Domain":
        """Attach library-resolved objective bodies. Returns self."""
        object.__setattr__(self, "_resolved", dict(resolved))
        return self

    def objective(self, objective_id: str) -> Objective:
        """Return the library-resolved, version-pinned objective body."""
        try:
            return self._resolved[objective_id]
        except KeyError:
            msg = f"{objective_id!r} is not part of domain {self.id!r}"
            raise KeyError(msg) from None

    def prerequisites(self, objective_id: str) -> list[str]:
        """Direct prerequisites of an objective, in manifest edge order."""
        return [e.from_id for e in self.edges if e.to_id == objective_id]

    def teaching_order(self) -> list[str]:
        """A topological order over the edge graph, stable and module-aware."""
        order = [oid for m in self.modules for oid in m.objectives]
        if not order:
            order = [ref.id for ref in self.objectives]
        rank = {oid: i for i, oid in enumerate(order)}
        indegree = {oid: 0 for oid in order}
        successors: dict[str, list[str]] = {oid: [] for oid in order}
        for edge in self.edges:
            successors[edge.from_id].append(edge.to_id)
            indegree[edge.to_id] += 1

        ready = sorted((o for o in order if indegree[o] == 0), key=rank.__getitem__)
        out: list[str] = []
        while ready:
            current = ready.pop(0)
            out.append(current)
            for nxt in successors[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)
            ready.sort(key=rank.__getitem__)
        return out
