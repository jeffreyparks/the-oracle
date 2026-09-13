"""The read model: what a learner knows, right now.

``MasteryState`` is the only source. Nothing here computes mastery - that is
the reducers' job - so a profile can never disagree with a replay.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine
from sqlmodel import Session, select

from the_oracle.mastery.bkt import DEFAULT_PARAMS, MASTERY_THRESHOLD, is_mastered
from the_oracle.store import models

if TYPE_CHECKING:  # pragma: no cover
    from the_oracle.domains.schema import Domain


class MasteryProfile:
    """An immutable snapshot of one learner's mastery over shared objectives.

    An objective with no evidence answers with the BKT prior, not with zero:
    "we have not looked" is not the same claim as "they do not know it".
    """

    __slots__ = ("_confidence", "_default", "_last_seen", "_misconceptions", "_observations", "_scores", "learner_id")

    def __init__(
        self,
        scores: Mapping[str, float] | None = None,
        *,
        learner_id: str = "",
        observations: Mapping[str, int] | None = None,
        confidence: Mapping[str, float] | None = None,
        misconceptions: Mapping[str, Iterable[str]] | None = None,
        last_seen: Mapping[str, datetime | None] | None = None,
        default: float = DEFAULT_PARAMS.p_init,
    ) -> None:
        self.learner_id = learner_id
        self._scores: dict[str, float] = {k: float(v) for k, v in (scores or {}).items()}
        self._observations: dict[str, int] = {k: int(v) for k, v in (observations or {}).items()}
        self._confidence: dict[str, float] = {k: float(v) for k, v in (confidence or {}).items()}
        self._misconceptions: dict[str, tuple[str, ...]] = {
            k: tuple(v) for k, v in (misconceptions or {}).items()
        }
        self._last_seen: dict[str, datetime | None] = dict(last_seen or {})
        self._default = float(default)

    # -- reads -------------------------------------------------------------

    def p(self, objective_id: str) -> float:
        """Posterior probability that the learner knows this objective."""
        return self._scores.get(objective_id, self._default)

    def observations(self, objective_id: str) -> int:
        """How many graded opportunities stand behind :meth:`p`."""
        return self._observations.get(objective_id, 0)

    def confidence(self, objective_id: str) -> float:
        """How much evidence stands behind :meth:`p`, in [0, 1)."""
        return self._confidence.get(objective_id, 0.0)

    def misconceptions(self, objective_id: str) -> tuple[str, ...]:
        """Named wrong models seen on this objective, in first-seen order."""
        return self._misconceptions.get(objective_id, ())

    def last_seen(self, objective_id: str) -> datetime | None:
        return self._last_seen.get(objective_id)

    def seen(self) -> set[str]:
        """Objectives with at least one recorded observation."""
        return {oid for oid, n in self._observations.items() if n > 0}

    def mastered(self, threshold: float = MASTERY_THRESHOLD) -> set[str]:
        """Objectives whose posterior clears the mastery bar."""
        return {oid for oid, p in self._scores.items() if is_mastered(p, threshold)}

    def weakest(self, limit: int = 5) -> list[str]:
        """The lowest-posterior objectives we have evidence for."""
        ranked = sorted(self._scores.items(), key=lambda kv: (kv[1], kv[0]))
        return [oid for oid, _ in ranked[:limit]]

    def known_prerequisites_met(self, domain: "Domain", objective_id: str) -> bool:
        """True when every prerequisite this domain declares is mastered.

        Mastery learning, PLAN.md section 6: no advance until prerequisites
        reach ``p >= 0.85``. An objective with no prerequisites is always ready.
        """
        return all(
            is_mastered(self.p(prerequisite))
            for prerequisite in domain.prerequisites(objective_id)
        )

    def ready(self, domain: "Domain", threshold: float = MASTERY_THRESHOLD) -> list[str]:
        """Teachable objectives: not yet mastered, prerequisites satisfied."""
        return [
            oid
            for oid in domain.teaching_order()
            if not is_mastered(self.p(oid), threshold)
            and self.known_prerequisites_met(domain, oid)
        ]

    def as_dict(self) -> dict[str, float]:
        """Plain ``objective_id -> p`` mapping, sorted for stable output."""
        return {oid: self._scores[oid] for oid in sorted(self._scores)}

    # -- dunders -----------------------------------------------------------

    def __contains__(self, objective_id: object) -> bool:
        return objective_id in self._scores

    def __len__(self) -> int:
        return len(self._scores)

    def __repr__(self) -> str:
        return (
            f"MasteryProfile(learner_id={self.learner_id!r}, "
            f"objectives={len(self._scores)}, mastered={len(self.mastered())})"
        )


def profile_from_rows(
    rows: Iterable[models.MasteryState], learner_id: str = ""
) -> MasteryProfile:
    """Build a profile from ``MasteryState`` rows. Pure; used by tests too."""
    scores: dict[str, float] = {}
    observations: dict[str, int] = {}
    confidence: dict[str, float] = {}
    misconceptions: dict[str, tuple[str, ...]] = {}
    last_seen: dict[str, datetime | None] = {}
    for row in rows:
        scores[row.objective_id] = float(row.p_mastery)
        observations[row.objective_id] = int(row.observations)
        confidence[row.objective_id] = float(row.confidence)
        misconceptions[row.objective_id] = tuple(row.misconception_ids or ())
        last_seen[row.objective_id] = row.last_seen_at
        learner_id = learner_id or row.learner_id
    return MasteryProfile(
        scores,
        learner_id=learner_id,
        observations=observations,
        confidence=confidence,
        misconceptions=misconceptions,
        last_seen=last_seen,
    )


def profile_for(learner_id: str, engine: Engine | None = None) -> MasteryProfile:
    """Read one learner's mastery out of ``MasteryState``."""
    from the_oracle.store.db import get_engine

    engine = engine or get_engine()
    statement = (
        select(models.MasteryState)
        .where(models.MasteryState.learner_id == learner_id)
        .order_by(models.MasteryState.objective_id)  # type: ignore[arg-type]
    )
    with Session(engine) as session:
        rows: list[models.MasteryState] = list(session.exec(statement))
    return profile_from_rows(rows, learner_id)


__all__: list[str] = ["MasteryProfile", "profile_for", "profile_from_rows"]
