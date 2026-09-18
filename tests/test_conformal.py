"""Label-conditional split conformal (section 8.4).

Required checks (CLAUDE.md section 13):
- the threshold form and the score form agree on a coarse grid (multiples of 0.001)
  full of ties;
- mean coverage over 200 seeded repeats on exchangeable data is >= 1 - alpha - 0.01;
- edge cases: k > n, all-equal probabilities, and an empty calibration class (must raise).
"""

from __future__ import annotations

import numpy as np
import pytest

from triage.policy.decide import decide_many
from triage.uncertainty.conformal import (
    FRAUD,
    LEGIT,
    coverage_report,
    fit_thresholds,
    predict_sets,
    score_form_sets,
    set_labels,
)

ALPHA_PAIRS = [(0.05, 0.01), (0.10, 0.02), (0.30, 0.05), (0.60, 0.01)]


def _coarse_grid_sample(rng: np.random.Generator, n: int) -> np.ndarray:
    """Probabilities on a grid of multiples of 0.001, so ties are everywhere."""
    return rng.integers(0, 1001, size=n) / 1000.0


def _exchangeable_draw(
    rng: np.random.Generator, n: int, prevalence: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scores drawn i.i.d. per class, so calibration and test rows are exchangeable."""
    labels = (rng.random(n) < prevalence).astype(int)
    probs = np.where(labels == 1, rng.beta(5.0, 2.0, size=n), rng.beta(2.0, 6.0, size=n))
    return probs, labels


# --- (a) the two forms agree, even with ties -------------------------------------


@pytest.mark.parametrize(("alpha_fraud", "alpha_legit"), ALPHA_PAIRS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_threshold_form_matches_score_form(
    alpha_fraud: float, alpha_legit: float, seed: int
) -> None:
    """Identical decisions on a coarse grid full of tied probabilities."""
    rng = np.random.default_rng(seed)
    cal_probs = _coarse_grid_sample(rng, 800)
    cal_labels = (rng.random(800) < 0.25).astype(int)
    test_probs = _coarse_grid_sample(rng, 500)

    thresholds = fit_thresholds(
        cal_probs, cal_labels, alpha_fraud=alpha_fraud, alpha_legit=alpha_legit
    )
    threshold_sets = predict_sets(test_probs, thresholds)
    score_sets = score_form_sets(
        cal_probs, cal_labels, test_probs, alpha_fraud=alpha_fraud, alpha_legit=alpha_legit
    )

    assert np.array_equal(threshold_sets, score_sets)
    assert np.array_equal(decide_many(threshold_sets), decide_many(score_sets))


def test_ties_are_everywhere_in_that_sample() -> None:
    """Guard the guard: the grid really does produce heavy ties."""
    rng = np.random.default_rng(0)
    sample = _coarse_grid_sample(rng, 800)
    assert np.unique(sample).size < sample.size, "no ties: the equivalence test would be vacuous"


# --- (b) coverage on exchangeable data -------------------------------------------


@pytest.mark.parametrize(("alpha_fraud", "alpha_legit"), [(0.10, 0.02), (0.30, 0.05)])
def test_mean_coverage_over_200_repeats(alpha_fraud: float, alpha_legit: float) -> None:
    """Both guarantees hold on average over 200 seeded repeats."""
    fraud_coverage = []
    genuine_exclusion = []

    for repeat in range(200):
        rng = np.random.default_rng(10_000 + repeat)
        cal_probs, cal_labels = _exchangeable_draw(rng, 1500, prevalence=0.2)
        test_probs, test_labels = _exchangeable_draw(rng, 1500, prevalence=0.2)

        thresholds = fit_thresholds(
            cal_probs, cal_labels, alpha_fraud=alpha_fraud, alpha_legit=alpha_legit
        )
        report = coverage_report(test_probs, test_labels, thresholds)
        fraud_coverage.append(report["fraud_coverage"])
        genuine_exclusion.append(report["genuine_exclusion_rate"])

    assert float(np.mean(fraud_coverage)) >= 1.0 - alpha_fraud - 0.01
    assert float(np.mean(genuine_exclusion)) <= alpha_legit + 0.01


def test_coverage_is_not_trivially_one() -> None:
    """A loose alpha_fraud must actually leave some fraud uncovered, or the test is empty."""
    rng = np.random.default_rng(7)
    cal_probs, cal_labels = _exchangeable_draw(rng, 4000, prevalence=0.2)
    test_probs, test_labels = _exchangeable_draw(rng, 4000, prevalence=0.2)

    thresholds = fit_thresholds(cal_probs, cal_labels, alpha_fraud=0.40, alpha_legit=0.05)
    report = coverage_report(test_probs, test_labels, thresholds)
    assert 0.5 < report["fraud_coverage"] < 0.75
    assert report["approve_share"] + report["review_share"] + report[
        "verify_share"
    ] == pytest.approx(1.0)
    assert report["empty_share"] <= report["review_share"]


# --- (c) edge cases ---------------------------------------------------------------


def test_k_greater_than_n_covers_everything() -> None:
    """Too few calibration frauds for the requested alpha: tau_fraud falls to -inf."""
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    labels = np.array([0, 0, 0, 0, 1])  # one fraud only

    thresholds = fit_thresholds(probs, labels, alpha_fraud=0.05, alpha_legit=0.30)
    assert thresholds.tau_fraud == -np.inf

    sets = predict_sets(np.array([0.0, 0.5, 1.0]), thresholds)
    assert sets[:, FRAUD].all(), "with tau_fraud = -inf every set must contain fraud"
    assert coverage_report(probs, labels, thresholds)["fraud_coverage"] == 1.0


def test_k0_greater_than_n0_never_excludes_a_genuine_applicant() -> None:
    """Too few calibration genuine rows for the requested alpha: tau_legit rises to +inf."""
    probs = np.array([0.1, 0.9, 0.8, 0.7, 0.6])
    labels = np.array([0, 1, 1, 1, 1])  # one genuine only

    thresholds = fit_thresholds(probs, labels, alpha_fraud=0.30, alpha_legit=0.01)
    assert thresholds.tau_legit == np.inf
    assert predict_sets(np.array([0.0, 0.5, 1.0]), thresholds)[:, LEGIT].all()


def test_all_equal_probabilities_send_everything_to_review() -> None:
    """With no spread at all, every set holds both labels, so nothing is auto-decided."""
    probs = np.full(200, 0.37)
    labels = np.zeros(200, dtype=int)
    labels[:20] = 1

    thresholds = fit_thresholds(probs, labels, alpha_fraud=0.10, alpha_legit=0.05)
    assert thresholds.tau_fraud == 0.37
    assert thresholds.tau_legit == 0.37

    sets = predict_sets(probs, thresholds)
    assert sets.all(), "p >= tau_fraud and p <= tau_legit both hold"
    assert set(decide_many(sets)) == {"review"}
    assert set_labels(sets[:1]) == [["legit", "fraud"]]


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (np.zeros(10, dtype=int), "no fraud"),
        (np.ones(10, dtype=int), "no genuine"),
    ],
)
def test_empty_calibration_class_raises(labels: np.ndarray, message: str) -> None:
    """A missing calibration class is an error, never a silent default."""
    probs = np.linspace(0.01, 0.99, 10)
    with pytest.raises(ValueError, match=message):
        fit_thresholds(probs, labels, alpha_fraud=0.10, alpha_legit=0.05)


@pytest.mark.parametrize(("alpha_fraud", "alpha_legit"), [(0.0, 0.05), (1.0, 0.05), (0.1, 0.0)])
def test_alphas_must_be_strictly_inside_the_unit_interval(
    alpha_fraud: float, alpha_legit: float
) -> None:
    """Alpha 0 or 1 is a configuration error, not a degenerate policy."""
    probs = np.linspace(0.01, 0.99, 20)
    labels = np.tile([0, 1], 10)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        fit_thresholds(probs, labels, alpha_fraud=alpha_fraud, alpha_legit=alpha_legit)


def test_mismatched_lengths_and_bad_labels_raise() -> None:
    """Shape and label checks, so a mis-joined calibration set fails loudly."""
    with pytest.raises(ValueError, match="same length"):
        fit_thresholds(np.array([0.1, 0.2]), np.array([0, 1, 0]), alpha_fraud=0.1, alpha_legit=0.05)
    with pytest.raises(ValueError, match="labels must be 0/1"):
        fit_thresholds(np.array([0.1, 0.2]), np.array([0, 2]), alpha_fraud=0.1, alpha_legit=0.05)


# --- the threshold form is the one that survives isotonic ties ---------------------


def test_thresholds_come_from_the_calibration_probabilities_themselves() -> None:
    """tau_fraud must be an observed calibration probability, not a round-tripped one.

    Computing ``tau_fraud = 1 - qhat`` is what section 8.4 forbids: on isotonic
    output, which is full of tied values, the round trip can land between two
    ties and flip decisions.
    """
    rng = np.random.default_rng(3)
    # Isotonic-style output: a handful of distinct values, heavily repeated.
    plateau = np.array([0.0007, 0.013, 0.081, 0.24, 0.63])
    cal_probs = rng.choice(plateau, size=2000)
    cal_labels = (rng.random(2000) < 0.2).astype(int)

    thresholds = fit_thresholds(cal_probs, cal_labels, alpha_fraud=0.20, alpha_legit=0.05)
    assert thresholds.tau_fraud in set(plateau.tolist())
    assert thresholds.tau_legit in set(plateau.tolist())

    round_tripped = 1.0 - (1.0 - thresholds.tau_fraud)
    sets_direct = predict_sets(plateau, thresholds)
    sets_round_tripped = plateau >= round_tripped
    # Documented, not asserted-away: if this ever diverges, the threshold form is
    # the correct side.
    assert (
        np.array_equal(sets_direct[:, FRAUD], sets_round_tripped)
        or round_tripped != thresholds.tau_fraud
    )


def test_policy_version_string() -> None:
    """The API reports the alphas that produced the thresholds."""
    probs = np.linspace(0.01, 0.99, 100)
    labels = np.tile([0, 1], 50)
    thresholds = fit_thresholds(probs, labels, alpha_fraud=0.3, alpha_legit=0.02)
    assert thresholds.policy_version == "alpha_fraud=0.3,alpha_legit=0.02"
    assert thresholds.to_dict()["n_fraud"] == 50
