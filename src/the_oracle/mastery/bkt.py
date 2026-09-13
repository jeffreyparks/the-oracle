"""Bayesian Knowledge Tracing: one latent skill per objective.

The model
---------
Each objective carries a hidden binary state ``K`` for one learner: the learner
either knows the objective or does not. We never see ``K``. We see answers.

Four parameters describe the link between the hidden state and an answer:

* ``p_init``    P(K) before any evidence - the prior.
* ``p_transit`` P(not known -> known) on one practice opportunity - learning.
* ``p_slip``    P(wrong | knows it) - a slip.
* ``p_guess``   P(right | does not know it) - a lucky guess.

The update, derived
-------------------
Let ``p = P(K_t = 1)`` before seeing the answer at step ``t``.

Bayes on a *correct* answer::

    P(correct | K=1) = 1 - p_slip
    P(correct | K=0) = p_guess

    P(K=1 | correct) =            p * (1 - p_slip)
                       -----------------------------------------
                       p * (1 - p_slip) + (1 - p) * p_guess

Bayes on a *wrong* answer::

    P(wrong | K=1) = p_slip
    P(wrong | K=0) = 1 - p_guess

    P(K=1 | wrong) =              p * p_slip
                     -----------------------------------------
                     p * p_slip + (1 - p) * (1 - p_guess)

That conditional is the *posterior about the past*. The practice opportunity
itself may teach the learner, so we then apply the transition::

    P(K_{t+1} = 1) = P(K=1 | obs) + (1 - P(K=1 | obs)) * p_transit

which is the standard Corbett & Anderson (1995) BKT recurrence.

Numerical notes
---------------
* The denominator is zero only for degenerate parameters (``p_slip = 0`` and
  ``p_guess = 0`` with ``p`` at a boundary); we guard it and fall back to the
  prior rather than raising inside a replay.
* The result is clamped to the open interval ``(EPS, 1 - EPS)``. A probability
  of exactly 0 or 1 is an absorbing state that no later evidence can move, and
  an absorbing state is a lie about a human being.
* With ``p_transit = 0`` the recurrence is a pure Bayes filter on a static
  latent. That is the right setting for a diagnostic, where we want to *measure*
  rather than to model learning in progress.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Mastery is claimed at this posterior. PLAN.md section 6: no advance until
#: prerequisites reach p >= 0.85.
MASTERY_THRESHOLD: float = 0.85

#: Probabilities are kept strictly inside (EPS, 1 - EPS).
EPS: float = 1e-6


@dataclass(frozen=True, slots=True)
class BKTParams:
    """Knowledge-tracing parameters for one objective."""

    p_init: float = 0.2      # prior knowledge
    p_transit: float = 0.15  # learning on an opportunity
    p_slip: float = 0.10     # knows it, answers wrong
    p_guess: float = 0.20    # does not know it, answers right

    def __post_init__(self) -> None:
        for name in ("p_init", "p_transit", "p_slip", "p_guess"):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        if self.p_slip + self.p_guess >= 1.0:
            raise ValueError(
                "p_slip + p_guess must be < 1; otherwise a correct answer is "
                "evidence *against* knowing, and the model is inverted"
            )


DEFAULT_PARAMS = BKTParams()


def clamp(p: float) -> float:
    """Keep a probability strictly inside (0, 1)."""
    return min(max(float(p), EPS), 1.0 - EPS)


def conditional(p_prior: float, correct: bool, params: BKTParams | None = None) -> float:
    """P(knows it | this one observation). No learning applied yet."""
    params = params or DEFAULT_PARAMS
    p = clamp(p_prior)
    if correct:
        known = p * (1.0 - params.p_slip)
        unknown = (1.0 - p) * params.p_guess
    else:
        known = p * params.p_slip
        unknown = (1.0 - p) * (1.0 - params.p_guess)
    total = known + unknown
    if total <= 0.0:  # degenerate parameters; keep the prior rather than raise
        return p
    return clamp(known / total)


def posterior(p_prior: float, correct: bool, params: BKTParams) -> float:
    """Bayes update on one observation, then apply the learning rate."""
    params = params or DEFAULT_PARAMS
    p_conditional = conditional(p_prior, correct, params)
    return clamp(p_conditional + (1.0 - p_conditional) * params.p_transit)


def update(p_prior: float, correct: bool, params: BKTParams | None = None) -> float:
    """One practice opportunity folded into the belief. The public entry point."""
    return posterior(p_prior, correct, params or DEFAULT_PARAMS)


def trace(
    observations: list[bool],
    params: BKTParams | None = None,
    *,
    p_start: float | None = None,
) -> list[float]:
    """Fold a whole sequence, returning the belief after each observation."""
    params = params or DEFAULT_PARAMS
    p = clamp(params.p_init if p_start is None else p_start)
    out: list[float] = []
    for correct in observations:
        p = update(p, correct, params)
        out.append(p)
    return out


def confidence(observations: int, *, half_life: int = 3) -> float:
    """How much evidence stands behind a belief, in [0, 1).

    Deliberately simple and monotonic in the number of observations:
    ``n / (n + half_life)``. It says "how much have we seen", not "how sure are
    we that the skill is present" - the posterior already says that.
    """
    n = max(int(observations), 0)
    return n / (n + half_life) if n else 0.0


def is_mastered(p: float, threshold: float = MASTERY_THRESHOLD) -> bool:
    """True when the posterior clears the mastery bar."""
    return float(p) >= threshold


__all__ = [
    "DEFAULT_PARAMS",
    "EPS",
    "MASTERY_THRESHOLD",
    "BKTParams",
    "clamp",
    "conditional",
    "confidence",
    "is_mastered",
    "posterior",
    "trace",
    "update",
]
