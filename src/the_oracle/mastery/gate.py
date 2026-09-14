"""The strict teaching gate.

Phase 1 debt item 1, paid here.

Two gates exist on purpose, and they must never converge:

* The **diagnostic** gate (:func:`the_oracle.agents.assessor.prerequisites_ready`)
  is loose. A prerequisite counts as cleared when it is mastered *or* when the
  learner just answered it correctly. Strict gating stalls a probe after one
  item, because a pack has a single root and no mastery estimates yet. Probing
  is not teaching, so that relaxation is correct there.
* The **teaching** gate, this module, is strict. Mastery learning (PLAN.md
  section 6) does not advance until EVERY direct prerequisite sits at
  ``p >= 0.85``. No exceptions, no "just answered correctly", no credit for a
  lucky guess.

``tests/test_gate.py`` asserts the two gates disagree on a constructed case, so
a later edit cannot quietly collapse them into one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from the_oracle.mastery.bkt import MASTERY_THRESHOLD, is_mastered

if TYPE_CHECKING:  # pragma: no cover
    from the_oracle.domains.schema import Domain
    from the_oracle.mastery.profile import MasteryProfile


def prerequisites_met(
    domain: "Domain",
    objective_id: str,
    profile: "MasteryProfile",
    *,
    threshold: float = MASTERY_THRESHOLD,
) -> bool:
    """True when every direct prerequisite is at ``p >= threshold``.

    An objective with no prerequisites is always ready: there is nothing to be
    blocked on. An unseen prerequisite answers with the BKT prior (0.2), which
    is well under the bar, so "we have never looked" blocks teaching too.
    """
    return not blocked_by(domain, objective_id, profile, threshold=threshold)


def blocked_by(
    domain: "Domain",
    objective_id: str,
    profile: "MasteryProfile",
    *,
    threshold: float = MASTERY_THRESHOLD,
) -> list[str]:
    """The direct prerequisites that are not yet mastered, in manifest order.

    Empty means teachable. Non-empty names exactly what to work on first, so a
    command can say *why* it will not teach something instead of just refusing.
    """
    return [
        prerequisite
        for prerequisite in domain.prerequisites(objective_id)
        if not is_mastered(profile.p(prerequisite), threshold)
    ]


def teachable(
    domain: "Domain",
    profile: "MasteryProfile",
    *,
    threshold: float = MASTERY_THRESHOLD,
) -> list[str]:
    """Objectives the learner may be taught next, in teaching order.

    Not yet mastered, and every direct prerequisite strictly cleared.
    """
    return [
        objective_id
        for objective_id in domain.teaching_order()
        if not is_mastered(profile.p(objective_id), threshold)
        and prerequisites_met(domain, objective_id, profile, threshold=threshold)
    ]


__all__ = ["blocked_by", "prerequisites_met", "teachable"]
