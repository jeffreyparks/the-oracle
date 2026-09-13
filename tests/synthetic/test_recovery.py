"""Does the knowledge tracer recover a learner it has never been told about?

The honest claim, stated up front
---------------------------------
Knowledge tracing estimates a *belief about a binary latent*: "does this learner
know the objective?" Fed a consistent answer stream, that belief saturates
toward 0 or 1. So:

* **Classification recovery is exact.** Above / below the mastery threshold is
  recovered with no errors for a polarised learner, which is what the mastery
  gate in PLAN.md section 6 actually consumes.
* **Point recovery is only meaningful near the ends.** For true mastery near 0
  or 1 the mean absolute error is small (tolerance stated below). For true
  mastery at 0.5 the estimate saturates and MAE is near 0.5. That is a property
  of the model, not a bug, and the test below pins it down instead of hiding it.
"""

from __future__ import annotations


from the_oracle.mastery.bkt import MASTERY_THRESHOLD, BKTParams
from the_oracle.mastery.synthetic import SyntheticLearner, estimate, recovery_error
from tests.synthetic.learners import AMBIGUOUS, POLARISED, polarised

#: Stated tolerance: mean absolute error over a polarised cohort, 40
#: opportunities per objective. Measured worst case over 50 seeds is 0.075.
TOLERANCE = 0.10

#: Opportunities needed before threshold classification is exact on every seed.
#: Measured: 12 items misclassifies 4 seeds in 50; 20 items misclassifies none.
RELIABLE_ITEMS = 20

#: Opportunities per objective. 12 is the diagnostic cap in PLAN.md section 6;
#: 40 is "enough observations" for the point-recovery claim.
DIAGNOSTIC_ITEMS = 12
ENOUGH_ITEMS = 40

#: A diagnostic measures; it must not assume the learner improves mid-test.
DIAGNOSTIC_PARAMS = BKTParams(p_transit=0.0, p_slip=0.1, p_guess=0.2)


def test_point_recovery_within_tolerance_over_many_seeds():
    errors = []
    for seed in range(20):
        learner = polarised(seed)
        estimated = estimate(
            learner, POLARISED, opportunities=ENOUGH_ITEMS, params=DIAGNOSTIC_PARAMS
        )
        errors.append(recovery_error(POLARISED, estimated))
    assert max(errors) <= TOLERANCE, f"worst mean absolute error {max(errors):.3f}"
    assert sum(errors) / len(errors) <= TOLERANCE


def _classified(seed: int, opportunities: int) -> set[str]:
    estimated = estimate(
        polarised(seed), POLARISED, opportunities=opportunities, params=DIAGNOSTIC_PARAMS
    )
    return {oid for oid, p in estimated.items() if p >= MASTERY_THRESHOLD}


def test_classification_recovery_is_exact_with_enough_evidence():
    truth = {oid for oid, m in POLARISED.items() if m >= MASTERY_THRESHOLD}
    for seed in range(20):
        assert _classified(seed, RELIABLE_ITEMS) == truth, f"seed {seed}"


def test_the_diagnostic_cap_is_good_but_not_perfect():
    """12 items is the cap in PLAN.md section 6. Measure what it buys.

    Measured over 50 seeds: 4 seeds misclassify one objective. The near-miss is
    always a true-0.90 learner who slipped early, and it recovers with more
    items. We assert the measured rate so a regression shows up as a number.
    """
    truth = {oid for oid, m in POLARISED.items() if m >= MASTERY_THRESHOLD}
    exact = sum(_classified(seed, DIAGNOSTIC_ITEMS) == truth for seed in range(50))
    assert exact >= 45  # measured 46/50


def test_more_evidence_never_makes_recovery_worse_on_average():
    def mean_error(n: int) -> float:
        return sum(
            recovery_error(
                POLARISED,
                estimate(polarised(seed), POLARISED, opportunities=n, params=DIAGNOSTIC_PARAMS),
            )
            for seed in range(20)
        ) / 20

    assert mean_error(ENOUGH_ITEMS) <= mean_error(4)


def test_midpoint_mastery_saturates_and_we_say_so():
    """A learner who truly knows half of it is not recovered as 0.5.

    The tracer answers a binary question, so it commits. This test exists to
    state the limit, not to hide it: do not read a point estimate near 0.5 as
    "half-mastered", read it as "we have not decided yet".
    """
    learner = SyntheticLearner(AMBIGUOUS, slip=0.1, guess=0.2, seed=7)
    estimated = estimate(learner, AMBIGUOUS, opportunities=ENOUGH_ITEMS, params=DIAGNOSTIC_PARAMS)
    for p in estimated.values():
        assert abs(p - 0.5) > 0.2  # it commits, one way or the other
    assert recovery_error(AMBIGUOUS, estimated) > 0.25  # measured range 0.31 - 0.50


def test_a_noisier_learner_is_harder_to_recover():
    clean = SyntheticLearner(POLARISED, slip=0.05, guess=0.05, seed=3)
    noisy = SyntheticLearner(POLARISED, slip=0.35, guess=0.35, seed=3)
    clean_error = recovery_error(
        POLARISED, estimate(clean, POLARISED, opportunities=8, params=DIAGNOSTIC_PARAMS)
    )
    noisy_error = recovery_error(
        POLARISED, estimate(noisy, POLARISED, opportunities=8, params=DIAGNOSTIC_PARAMS)
    )
    assert clean_error <= noisy_error
