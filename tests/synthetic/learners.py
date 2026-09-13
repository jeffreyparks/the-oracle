"""Named synthetic cohorts, shared by the mastery tests.

Kept out of ``src/`` on purpose: these are test fixtures, not engine code. The
objective ids are deliberately meaningless (``o_a``, ``o_b``, ...) so no subject
matter leaks anywhere near the engine.
"""

from __future__ import annotations

from the_oracle.mastery.synthetic import SyntheticLearner

#: Confident on two objectives, blank on two. The clean recovery case: the
#: knowledge tracer answers a *binary* question ("do they know it?"), so it can
#: only be scored fairly against mastery that is near 0 or near 1.
POLARISED: dict[str, float] = {"o_a": 0.95, "o_b": 0.90, "o_c": 0.10, "o_d": 0.05}

#: Genuinely half-way. The tracer saturates here - see the docstring of
#: ``test_midpoint_mastery_saturates`` for the honest statement of that limit.
AMBIGUOUS: dict[str, float] = {"o_e": 0.5, "o_f": 0.5}

#: A learner who knows the first two rungs of a chain and not the third.
LADDER: dict[str, float] = {"o_one": 0.95, "o_two": 0.92, "o_three": 0.05}


def polarised(seed: int = 0) -> SyntheticLearner:
    return SyntheticLearner(POLARISED, slip=0.1, guess=0.2, seed=seed)


def ladder(seed: int = 0) -> SyntheticLearner:
    return SyntheticLearner(LADDER, slip=0.1, guess=0.2, seed=seed)


__all__ = ["AMBIGUOUS", "LADDER", "POLARISED", "ladder", "polarised"]
