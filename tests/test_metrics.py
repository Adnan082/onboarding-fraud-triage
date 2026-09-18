"""Metric definitions (section 8.3).

Every number in the README and the report is produced by these functions, so each
one is checked against a hand-computed case rather than against itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from triage.evaluation.metrics import (
    calibration,
    discrimination,
    rates_at_threshold,
    threshold_at_fpr,
    tpr_at_fpr,
)
from triage.models.calibrate import ece_equal_mass

# 100 genuine applicants scored 0.00, 0.01, ... 0.99 and 10 frauds, five of them
# interleaved with the top five genuine scores. Under a 5% FPR cap the best the
# model can do is catch five of the ten frauds.
HAND_SCORES = np.concatenate(
    [np.arange(100) / 100.0, np.array([0.995, 0.985, 0.975, 0.965, 0.955, 0.5, 0.4, 0.3, 0.2, 0.1])]
)
HAND_LABELS = np.concatenate([np.zeros(100, dtype=int), np.ones(10, dtype=int)])


def test_tpr_at_five_percent_fpr_is_hand_computed() -> None:
    """Five of 100 genuine applicants flagged, and five of ten frauds caught."""
    assert tpr_at_fpr(HAND_LABELS, HAND_SCORES, 0.05) == pytest.approx(0.5)


def test_threshold_at_five_percent_fpr_is_hand_computed() -> None:
    """The threshold that realises it, with flagging defined as score >= threshold.

    Two thresholds reach TPR 0.5 here: 0.95 (five genuine flagged, FPR 0.05) and
    0.955 (four flagged, FPR 0.04). The lower-FPR one wins, which is the right way
    round: the same fraud is caught while one fewer genuine applicant is stopped.
    """
    threshold = threshold_at_fpr(HAND_LABELS, HAND_SCORES, 0.05)
    assert threshold == pytest.approx(0.955)

    realised = rates_at_threshold(HAND_LABELS, HAND_SCORES, threshold)
    assert realised["fpr"] == pytest.approx(0.04)
    assert realised["tpr"] == pytest.approx(0.5)
    assert realised["tpr"] == pytest.approx(tpr_at_fpr(HAND_LABELS, HAND_SCORES, 0.05))


def test_the_fpr_cap_is_never_exceeded() -> None:
    """ "Largest TPR with FPR <= target" -- never the nearest point above it."""
    for target in (0.01, 0.02, 0.05, 0.10):
        threshold = threshold_at_fpr(HAND_LABELS, HAND_SCORES, target)
        assert rates_at_threshold(HAND_LABELS, HAND_SCORES, threshold)["fpr"] <= target + 1e-12


def test_rates_at_threshold_counts_by_hand() -> None:
    """The confusion counts behind the realised rates."""
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.95, 0.3, 0.99])

    rates = rates_at_threshold(labels, scores, 0.9)
    assert rates["n_flagged"] == 3  # 0.9, 0.95, 0.99
    assert rates["fpr"] == pytest.approx(0.5)  # 2 of 4 genuine
    assert rates["tpr"] == pytest.approx(0.5)  # 1 of 2 frauds
    assert rates["precision"] == pytest.approx(1 / 3)
    assert rates["flag_rate"] == pytest.approx(0.5)


def test_discrimination_on_a_perfect_and_a_useless_ranking() -> None:
    """AUC 1.0 when the ranking is perfect, 0.5 when the score says nothing."""
    labels = np.array([0] * 50 + [1] * 50)

    perfect = discrimination(labels, labels.astype(float))
    assert perfect["roc_auc"] == pytest.approx(1.0)
    assert perfect["pr_auc"] == pytest.approx(1.0)
    assert perfect["prevalence"] == pytest.approx(0.5)

    rng = np.random.default_rng(0)
    useless = discrimination(labels, rng.random(100))
    assert useless["roc_auc"] == pytest.approx(0.5, abs=0.15)


def test_discrimination_names_the_target_fpr() -> None:
    """The key says which FPR the TPR was read at, so a table cannot mislead."""
    keys = discrimination(HAND_LABELS, HAND_SCORES, 0.05)
    assert "tpr_at_0.05_fpr" in keys


def test_one_class_only_raises() -> None:
    """A slice with no fraud cannot produce an ROC curve, and must say so."""
    with pytest.raises(ValueError, match="both classes"):
        tpr_at_fpr(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


def test_mismatched_lengths_raise() -> None:
    """A mis-joined score array fails loudly."""
    with pytest.raises(ValueError, match="same length"):
        discrimination(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


# --- calibration ---------------------------------------------------------------------


def test_brier_and_calibration_in_the_large_by_hand() -> None:
    """Brier is the mean squared error; calibration-in-the-large is the mean gap."""
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])

    result = calibration(labels, probs, n_bins=2)
    expected_brier = float(np.mean([0.01, 0.04, 0.04, 0.01]))
    assert result["brier"] == pytest.approx(expected_brier)
    assert result["mean_predicted_rate"] == pytest.approx(0.5)
    assert result["observed_rate"] == pytest.approx(0.5)
    assert result["calibration_in_the_large"] == pytest.approx(0.0)


def test_calibration_in_the_large_catches_a_biased_model() -> None:
    """A model that predicts 10% at 1% prevalence is badly off in the mean."""
    labels = np.zeros(1000, dtype=int)
    labels[:10] = 1
    result = calibration(labels, np.full(1000, 0.10))
    assert result["calibration_in_the_large"] == pytest.approx(0.09)


def test_ece_equal_mass_by_hand() -> None:
    """Two bins of two rows each, worked out by hand."""
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 0, 1, 1])
    # bin 1: mean prob 0.15, observed 0.0 -> gap 0.15; bin 2: 0.35 vs 1.0 -> 0.65
    assert ece_equal_mass(probs, labels, n_bins=2) == pytest.approx((0.15 * 2 + 0.65 * 2) / 4)


def test_ece_is_zero_for_a_perfectly_calibrated_model() -> None:
    """Predicting exactly what happens gives no calibration error."""
    probs = np.array([0.0] * 50 + [1.0] * 50)
    assert ece_equal_mass(probs, probs, n_bins=10) == pytest.approx(0.0)


def test_ece_equal_mass_bins_are_equal_mass_at_one_percent_prevalence() -> None:
    """The reason for rank-based bins: equal-width bins collapse at this prevalence.

    Almost every application scores below 0.05, so equal-width bins put nearly
    everything in the first bin. Equal-mass bins keep all 15 informative.
    """
    rng = np.random.default_rng(1)
    probs = rng.beta(0.3, 30.0, size=3000)
    labels = (rng.random(3000) < probs).astype(int)

    equal_width_counts = np.histogram(probs, bins=15, range=(0.0, 1.0))[0]
    assert equal_width_counts[0] / probs.size > 0.9, "equal width really does collapse here"

    order = np.argsort(probs)
    sizes = [chunk.size for chunk in np.array_split(order, 15)]
    assert max(sizes) - min(sizes) <= 1

    value = ece_equal_mass(probs, labels, n_bins=15)
    assert 0.0 <= value < 0.05


def test_calibration_rejects_scores_outside_the_unit_interval() -> None:
    """Uncalibrated margins are not probabilities, and must not be scored as if they were."""
    with pytest.raises(ValueError, match=r"probabilities in \[0, 1\]"):
        calibration(np.array([0, 1]), np.array([-2.0, 3.0]))


def test_ece_rejects_bad_input() -> None:
    """Shape and bin-count guards."""
    with pytest.raises(ValueError, match="same length"):
        ece_equal_mass(np.array([0.1, 0.2]), np.array([0, 1, 1]))
    with pytest.raises(ValueError, match="at least 1"):
        ece_equal_mass(np.array([0.1, 0.2]), np.array([0, 1]), n_bins=0)
    with pytest.raises(ValueError, match="no rows"):
        ece_equal_mass(np.array([]), np.array([]))
