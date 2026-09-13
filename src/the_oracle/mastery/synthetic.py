"""Synthetic learners: pedagogy tested in seconds, with no model calls.

PLAN.md section 9: "Simulated learners with known true mastery and a fixed
error rate verify that the Assessor recovers the right profile."

Generative model
----------------
``true_mastery[objective_id]`` is the probability that the learner knows the
objective on any one opportunity. For each answer we draw the hidden state, then
corrupt it with the same slip and guess channel that knowledge tracing assumes::

    knows   ~ Bernoulli(m_eff)
    correct ~ Bernoulli(1 - slip) if knows else Bernoulli(guess)

Item difficulty tilts the hidden draw in log-odds space around difficulty 3, the
middle of the 1-5 pack-format scale::

    logit(m_eff) = logit(m) - DIFFICULTY_TILT * (difficulty - 3)

so difficulty 3 is neutral. The tracer does not model difficulty, so a test that
wants unbiased recovery should present difficulty 3 items.

The learner does not learn. It is a fixed measurement target: if the estimator
cannot find a *static* skill, it has no business tracking a moving one.
Randomness comes from a private ``random.Random(seed)``, so a seed fixes the
whole answer stream.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping

from the_oracle.mastery import bkt

#: Log-odds shift per point of item difficulty away from the neutral point.
DIFFICULTY_TILT: float = 0.4
#: Difficulty at which an item neither helps nor hinders.
NEUTRAL_DIFFICULTY: int = 3


def _logit(p: float) -> float:
    p = bkt.clamp(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class SyntheticLearner:
    """A simulated learner with known true mastery and a fixed error rate."""

    __slots__ = ("_rng", "asked", "guess", "seed", "slip", "true_mastery")

    def __init__(
        self,
        true_mastery: Mapping[str, float],
        slip: float = 0.1,
        guess: float = 0.2,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= slip < 1.0 or not 0.0 <= guess < 1.0:
            raise ValueError("slip and guess must be in [0, 1)")
        self.true_mastery: dict[str, float] = {
            k: bkt.clamp(v) for k, v in true_mastery.items()
        }
        self.slip = float(slip)
        self.guess = float(guess)
        self.seed = int(seed)
        self._rng = random.Random(seed)
        self.asked: list[tuple[str, int, bool]] = []

    def reset(self) -> None:
        """Rewind the answer stream to the seed. The same script replays."""
        self._rng = random.Random(self.seed)
        self.asked.clear()

    def effective_mastery(self, objective_id: str, difficulty: int = NEUTRAL_DIFFICULTY) -> float:
        """True mastery after the difficulty tilt."""
        base = self.true_mastery.get(objective_id, 0.0)
        shift = DIFFICULTY_TILT * (int(difficulty) - NEUTRAL_DIFFICULTY)
        return _sigmoid(_logit(base) - shift)

    def answer(self, objective_id: str, difficulty: int = NEUTRAL_DIFFICULTY) -> bool:
        """Answer one item. ``True`` means correct."""
        knows = self._rng.random() < self.effective_mastery(objective_id, difficulty)
        roll = self._rng.random()
        correct = roll >= self.slip if knows else roll < self.guess
        self.asked.append((objective_id, int(difficulty), correct))
        return correct

    def answers(
        self, objective_id: str, n: int, difficulty: int = NEUTRAL_DIFFICULTY
    ) -> list[bool]:
        """Answer ``n`` items on one objective."""
        return [self.answer(objective_id, difficulty) for _ in range(n)]

    def __repr__(self) -> str:
        return (
            f"SyntheticLearner(objectives={len(self.true_mastery)}, "
            f"slip={self.slip}, guess={self.guess}, seed={self.seed})"
        )


def recovery_error(
    true: Mapping[str, float],
    estimated: Mapping[str, float],
    *,
    default: float = bkt.DEFAULT_PARAMS.p_init,
) -> float:
    """Mean absolute error between true and estimated mastery.

    Scored over the keys of ``true``. An objective the estimator never saw
    scores against ``default``, the prior, because that is what the profile
    would answer for it.
    """
    keys = list(true)
    if not keys:
        return 0.0
    total = sum(abs(float(true[k]) - float(estimated.get(k, default))) for k in keys)
    return total / len(keys)


def estimate(
    learner: SyntheticLearner,
    objective_ids: Iterable[str],
    *,
    opportunities: int = 30,
    params: bkt.BKTParams | None = None,
    difficulty: int = NEUTRAL_DIFFICULTY,
) -> dict[str, float]:
    """Run a pure knowledge-tracing filter over a synthetic answer stream.

    Pass ``p_transit = 0`` for a diagnostic: the synthetic learner does not
    learn, so a non-zero learning rate would bias the estimate upward.
    """
    params = params or bkt.BKTParams(p_transit=0.0, p_slip=learner.slip, p_guess=learner.guess)
    out: dict[str, float] = {}
    for objective_id in objective_ids:
        p = bkt.clamp(params.p_init)
        for _ in range(opportunities):
            p = bkt.update(p, learner.answer(objective_id, difficulty), params)
        out[objective_id] = p
    return out


__all__ = [
    "DIFFICULTY_TILT",
    "NEUTRAL_DIFFICULTY",
    "SyntheticLearner",
    "estimate",
    "recovery_error",
]
