"""Checkpoints: one short assessment at the end of every module.

A checkpoint proves the module actually landed. It is drawn only from that
module's own objectives, and it is sized to fit inside a single session
(PLAN.md section 5: default 25 minutes, fixed four-phase structure). Pure and
deterministic, like the rest of the planner - no LLM chooses these items.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from the_oracle.planning.syllabus import PlannedObjective, Syllabus

if TYPE_CHECKING:  # pragma: no cover
    from the_oracle.domains.schema import Domain

DEFAULT_SESSION_MINUTES: int = 25
"""PLAN.md section 5. Learner-settable to 10 / 25 / 50 elsewhere."""

CONSOLIDATION_MINUTES: int = 2
"""The closing phase of every session. A checkpoint may not spend it."""

MINUTES_PER_ITEM: int = 5
"""One applied checkpoint item: read, answer, get the critique."""

MIN_ITEMS: int = 1
"""A module with planned work always gets at least one checkpoint item."""


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A short assessment placed after one module."""

    module_id: str
    objective_ids: tuple[str, ...]
    est_minutes: int
    after_item_index: int
    """Number of syllabus items that come before this checkpoint."""

    @property
    def item_count(self) -> int:
        return len(self.objective_ids)


def capacity(session_minutes: int = DEFAULT_SESSION_MINUTES) -> int:
    """How many checkpoint items fit in one session."""
    usable = max(0, session_minutes - CONSOLIDATION_MINUTES)
    return max(MIN_ITEMS, usable // MINUTES_PER_ITEM)


def _priority(item: PlannedObjective, difficulty: int, order: int) -> tuple[int, int, float, int]:
    """Rank inside one module: new before review, hard before easy, weak before strong."""
    return (0 if item.status == "learn" else 1, -difficulty, item.p_mastery, order)


def checkpoint_for_module(
    module_id: str,
    items: list[PlannedObjective],
    *,
    domain: "Domain | None" = None,
    session_minutes: int = DEFAULT_SESSION_MINUTES,
    after_item_index: int = 0,
) -> Checkpoint | None:
    """Build one checkpoint from one module's planned items.

    Returns ``None`` when the module has no planned work left - there is
    nothing to check if the learner already knows all of it.
    """
    mine = [item for item in items if item.module_id == module_id]
    if not mine:
        return None

    limit = min(len(mine), capacity(session_minutes))
    order = {item.objective_id: i for i, item in enumerate(mine)}

    def difficulty(item: PlannedObjective) -> int:
        if domain is None:
            return 1
        return domain.objective(item.objective_id).difficulty

    ranked = sorted(mine, key=lambda it: _priority(it, difficulty(it), order[it.objective_id]))
    chosen = {item.objective_id for item in ranked[:limit]}
    picked = tuple(item.objective_id for item in mine if item.objective_id in chosen)

    return Checkpoint(
        module_id=module_id,
        objective_ids=picked,
        est_minutes=len(picked) * MINUTES_PER_ITEM,
        after_item_index=after_item_index,
    )


def build_checkpoints(
    syllabus: Syllabus,
    *,
    domain: "Domain | None" = None,
    session_minutes: int = DEFAULT_SESSION_MINUTES,
) -> list[Checkpoint]:
    """One checkpoint after every module that carries planned work, in plan order."""
    out: list[Checkpoint] = []
    for module_id in syllabus.module_ids():
        last = max(
            i for i, item in enumerate(syllabus.items) if item.module_id == module_id
        )
        checkpoint = checkpoint_for_module(
            module_id,
            syllabus.items,
            domain=domain,
            session_minutes=session_minutes,
            after_item_index=last + 1,
        )
        if checkpoint is not None:
            out.append(checkpoint)
    return out


def checkpoint_minutes(checkpoints: list[Checkpoint]) -> int:
    """Total assessment time the checkpoints add on top of the syllabus."""
    return sum(c.est_minutes for c in checkpoints)


__all__ = [
    "CONSOLIDATION_MINUTES",
    "DEFAULT_SESSION_MINUTES",
    "MINUTES_PER_ITEM",
    "MIN_ITEMS",
    "Checkpoint",
    "build_checkpoints",
    "capacity",
    "checkpoint_for_module",
    "checkpoint_minutes",
]
