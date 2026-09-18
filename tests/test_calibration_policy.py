"""Probability calibration and the alpha sweep.

The conformal maths itself is tested in `test_conformal.py`. This file covers the
layer above it: choosing a calibration method on the right part of month 5, and
choosing an alpha pair a review team could actually staff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from triage.models.calibrate import (
    METHODS,
    Calibrator,
    ece_equal_mass,
    fit_all,
    fit_calibrator,
    reliability_curve,
    select,
)
from triage.policy.sweep import coverage_by_age, select_within_capacity, sweep
from triage.uncertainty.conformal import fit_thresholds


@pytest.fixture(scope="module")
def cfg(repo_root):
    """The composed project config."""
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        return compose(config_name="config")


@pytest.fixture(scope="module")
def miscalibrated() -> tuple[np.ndarray, np.ndarray]:
    """Scores that rank well but are far too low in the mean.

    This is what an uncalibrated LightGBM looks like at ~1% prevalence, and the
    reason the policy layer cannot read raw scores.
    """
    rng = np.random.default_rng(20260917)
    labels = (rng.random(20_000) < 0.012).astype(int)
    signal = np.where(labels == 1, rng.beta(6.0, 3.0, 20_000), rng.beta(2.0, 8.0, 20_000))
    return signal * 0.02, labels  # ranks preserved, magnitudes squashed towards zero


def test_uncalibrated_scores_are_badly_off_in_the_mean(miscalibrated) -> None:
    """Guard the guard: the fixture really is miscalibrated."""
    scores, labels = miscalibrated
    assert scores.mean() < labels.mean() / 2


@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_probabilities(miscalibrated, method: str) -> None:
    """Whatever the method, the output is usable as a probability."""
    scores, labels = miscalibrated
    calibrated = fit_calibrator(method, scores, labels).transform(scores)

    assert calibrated.shape == scores.shape
    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0


def test_none_is_the_identity(miscalibrated) -> None:
    """So downstream code never has to special-case "no calibration"."""
    scores, labels = miscalibrated
    assert np.array_equal(fit_calibrator("none", scores, labels).transform(scores), scores)


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_fixes_the_mean(miscalibrated, method: str) -> None:
    """Both real methods bring the predicted rate close to the observed one."""
    scores, labels = miscalibrated
    calibrated = fit_calibrator(method, scores, labels).transform(scores)
    assert abs(calibrated.mean() - labels.mean()) < 0.002


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_preserves_the_ranking(miscalibrated, method: str) -> None:
    """Calibration must not undo the model's discrimination.

    Platt is strictly monotone and isotonic is non-decreasing, so neither can
    reorder applications; only the numbers attached to them change.
    """
    from sklearn.metrics import roc_auc_score

    scores, labels = miscalibrated
    calibrated = fit_calibrator(method, scores, labels).transform(scores)
    assert roc_auc_score(labels, calibrated) == pytest.approx(
        roc_auc_score(labels, scores), abs=0.005
    )


def test_selection_prefers_a_calibrated_method(miscalibrated, cfg) -> None:
    """On badly miscalibrated scores, doing nothing should not win."""
    scores, labels = miscalibrated
    candidates = fit_all(scores, labels)
    best, comparison = select(cfg, candidates, scores, labels)

    assert set(comparison) == set(METHODS)
    assert best != "none"
    assert comparison[best]["brier"] <= comparison["none"]["brier"]


def test_selection_uses_the_configured_criterion(miscalibrated, cfg) -> None:
    """`select_by` decides, and it is Brier by default (section 8.2)."""
    scores, labels = miscalibrated
    candidates = fit_all(scores, labels)
    best, comparison = select(cfg, candidates, scores, labels)

    criterion = str(cfg.calibration.select_by)
    assert criterion == "brier"
    assert comparison[best][criterion] == min(values[criterion] for values in comparison.values())


def test_unknown_method_raises() -> None:
    """A typo in the config is an error, not a silent fallback to no calibration."""
    with pytest.raises(ValueError, match="unknown calibration method"):
        fit_calibrator("sigmoid-ish", np.array([0.1, 0.9]), np.array([0, 1]))
    with pytest.raises(ValueError, match="unknown calibration method"):
        Calibrator("nonsense").transform(np.array([0.5]))


def test_reliability_curve_bins_are_equal_mass(miscalibrated) -> None:
    """The curve the figure will be drawn from, stored with the metrics."""
    scores, labels = miscalibrated
    curve = reliability_curve(scores, labels, n_bins=15)

    assert len(curve) == 15
    sizes = [row["n"] for row in curve]
    assert max(sizes) - min(sizes) <= 1
    assert [row["mean_predicted"] for row in curve] == sorted(
        row["mean_predicted"] for row in curve
    )


def test_ece_improves_after_calibration(miscalibrated) -> None:
    """The headline calibration metric moves in the right direction."""
    scores, labels = miscalibrated
    before = ece_equal_mass(scores, labels, n_bins=15)
    after = ece_equal_mass(fit_calibrator("platt", scores, labels).transform(scores), labels, 15)
    assert after < before


# --- the alpha sweep ------------------------------------------------------------------


@pytest.fixture(scope="module")
def calibration_sample() -> tuple[np.ndarray, np.ndarray]:
    """A calibration part with a realistic number of frauds in it."""
    rng = np.random.default_rng(11)
    labels = (rng.random(40_000) < 0.012).astype(int)
    probs = np.where(labels == 1, rng.beta(6.0, 4.0, 40_000), rng.beta(1.5, 12.0, 40_000))
    return probs, labels


def test_sweep_covers_the_configured_grid(calibration_sample, cfg) -> None:
    """One row per alpha pair, and the thresholds that produced it."""
    probs, labels = calibration_sample
    grid = sweep(probs, labels, cfg)

    expected = len(cfg.policy.sweep.alpha_fraud) * len(cfg.policy.sweep.alpha_legit)
    assert len(grid) == expected
    assert {"alpha_fraud", "alpha_legit", "tau_fraud", "tau_legit"} <= set(grid.columns)


def test_coverage_falls_as_alpha_fraud_rises(calibration_sample, cfg) -> None:
    """The curve has to slope the right way, or the policy means nothing."""
    probs, labels = calibration_sample
    grid = sweep(probs, labels, cfg)
    one_legit = grid[grid["alpha_legit"] == 0.01].sort_values("alpha_fraud")

    coverage = one_legit["fraud_coverage"].to_numpy()
    assert (np.diff(coverage) <= 1e-9).all(), "coverage must not rise with alpha_fraud"
    assert coverage[0] > coverage[-1]


def test_more_coverage_means_fewer_straight_through_approvals(calibration_sample, cfg) -> None:
    """Catching more fraud always costs automation, whatever the score distribution.

    The README's trade-off table shows the cost landing on the review queue, but
    how it splits between review and verify depends on how far apart the two score
    distributions sit. What holds unconditionally is this: lowering `tau_fraud` to
    catch more fraud can only take applications *out* of the approve band.
    """
    probs, labels = calibration_sample
    grid = sweep(probs, labels, cfg)
    one_legit = grid[grid["alpha_legit"] == 0.01].sort_values("fraud_coverage")

    approve = one_legit["approve_share"].to_numpy()
    assert (np.diff(approve) <= 1e-9).all(), "approve share must not rise with coverage"
    # Sorted by ascending coverage, so the first row is the cheapest policy and
    # automates the most.
    assert approve[0] > approve[-1], "the two ends of the curve should differ"

    # And every row still accounts for 100% of applications.
    total = one_legit[["approve_share", "review_share", "verify_share"]].sum(axis=1)
    assert np.allclose(total.to_numpy(), 1.0)


def test_selection_respects_capacity(calibration_sample, cfg) -> None:
    """The chosen pair must be one the review team could actually staff."""
    probs, labels = calibration_sample
    grid = sweep(probs, labels, cfg)
    chosen = select_within_capacity(grid, cfg)

    assert chosen["expected_review_share"] <= float(cfg.policy.capacity.review_share)
    assert chosen["expected_verify_share"] <= float(cfg.policy.capacity.verify_share)

    affordable = grid[
        (grid["review_share"] <= float(cfg.policy.capacity.review_share))
        & (grid["verify_share"] <= float(cfg.policy.capacity.verify_share))
    ]
    assert chosen["expected_fraud_coverage"] == pytest.approx(affordable["fraud_coverage"].max())


def test_an_impossible_capacity_is_reported_not_silently_relaxed(calibration_sample, cfg) -> None:
    """If nothing fits, that is a finding for the owner, not a quiet compromise."""
    from omegaconf import OmegaConf

    probs, labels = calibration_sample
    grid = sweep(probs, labels, cfg)

    tiny = OmegaConf.merge(cfg, {"policy": {"capacity": {"review_share": 1e-9}}})
    with pytest.raises(ValueError, match="no alpha pair fits capacity"):
        select_within_capacity(grid, tiny)


def test_coverage_by_age_reports_every_band(calibration_sample, cfg) -> None:
    """Coverage per band, so a shortfall for older applicants is visible."""
    probs, labels = calibration_sample
    rng = np.random.default_rng(5)
    age = rng.choice([20, 30, 40, 50, 60, 70], size=labels.size)

    thresholds = fit_thresholds(probs, labels, alpha_fraud=0.3, alpha_legit=0.02)
    table = coverage_by_age(probs, labels, age, thresholds, width=10)

    assert list(table["band"]) == ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]
    assert (table["n"] > 0).all()
    assert isinstance(table, pd.DataFrame)
