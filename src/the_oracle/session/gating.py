"""The strict teaching gate, reached lazily.

Phase 1 left the diagnostic's prerequisite gate deliberately loose: while
probing, a prerequisite counts as cleared if it was mastered *or* just answered
correctly. That relaxation must never reach teaching, where mastery learning
still demands ``p >= 0.85`` on every direct prerequisite (PLAN.md section 6).

``the_oracle.mastery.gate`` owns that strict gate. It is imported lazily here
so the session package builds against the contract signature without a hard
import-time dependency on a module another worker lands. When it is missing,
we fall back to :meth:`MasteryProfile.ready`, which applies the same strict
rule through ``known_prerequisites_met``. The fallback is never looser.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from the_oracle.mastery import MASTERY_THRESHOLD

if TYPE_CHECKING:  # pragma: no cover
    from the_oracle.domains.schema import Domain
    from the_oracle.mastery import MasteryProfile


def _gate() -> Any | None:
    """The strict gate module, or ``None`` when it has not landed yet."""
    try:
        from the_oracle.mastery import gate
    except ImportError:  # pragma: no cover - only before mastery2 lands
        return None
    return gate


def using_strict_gate() -> bool:
    """True when ``mastery.gate`` is the gate actually in use."""
    return _gate() is not None


def teachable(
    domain: "Domain", profile: "MasteryProfile", *, threshold: float = MASTERY_THRESHOLD
) -> list[str]:
    """Objectives this learner may be taught next. Strict prerequisites only."""
    gate = _gate()
    if gate is not None:
        return list(gate.teachable(domain, profile, threshold=threshold))
    return list(profile.ready(domain, threshold))


def prerequisites_met(
    domain: "Domain",
    objective_id: str,
    profile: "MasteryProfile",
    *,
    threshold: float = MASTERY_THRESHOLD,
) -> bool:
    """True when EVERY direct prerequisite is at ``p >= threshold``."""
    gate = _gate()
    if gate is not None:
        return bool(gate.prerequisites_met(domain, objective_id, profile, threshold=threshold))
    return all(profile.p(pre) >= threshold for pre in domain.prerequisites(objective_id))


def blocked_by(
    domain: "Domain",
    objective_id: str,
    profile: "MasteryProfile",
    *,
    threshold: float = MASTERY_THRESHOLD,
) -> list[str]:
    """The prerequisites that are not clear yet, in pack order."""
    gate = _gate()
    if gate is not None:
        return list(gate.blocked_by(domain, objective_id, profile, threshold=threshold))
    return [pre for pre in domain.prerequisites(objective_id) if profile.p(pre) < threshold]


__all__ = ["blocked_by", "prerequisites_met", "teachable", "using_strict_gate"]
