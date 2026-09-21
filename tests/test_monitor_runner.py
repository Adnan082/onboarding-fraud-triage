"""The one path every monitored window takes.

`detectors_for_window` is the single place a window becomes four numbers, and it
is used for clean windows, test months, shifted variants and injected bugs alike.
That matters more than it sounds: a threshold calibrated on clean windows only
means anything if the windows it is later applied to were measured identically,
so a change here that affects one caller and not another would quietly
invalidate every threshold in `reports/monitoring.json`.

The probabilities are passed in rather than scored, so these tests exercise the
runner's own arithmetic and not LightGBM's.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from triage.features.encode import feature_columns, lgbm_frame, prepare
from triage.monitoring.conformal_rate import rates
from triage.monitoring.domain_clf import reference_sample
from triage.monitoring.runner import (
    THRESHOLDED,
    MonitorContext,
    detectors_for_window,
    thresholded_only,
)

TAU_FRAUD, TAU_LEGIT = 0.40, 0.02


@pytest.fixture(scope="module")
def cfg(repo_root):
    """The composed project config."""
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        return compose(config_name="config")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Big enough to split into a reference and several windows."""
    from tests.fixtures.make_fixture import make_fixture

    return make_fixture(n_rows=20_000, seed=20260917)


@pytest.fixture(scope="module")
def reference(frame: pd.DataFrame) -> pd.DataFrame:
    """The training months, standing in for the monitor's reference population."""
    return frame[frame["month"] <= 4].reset_index(drop=True)


@pytest.fixture(scope="module")
def context(cfg, frame: pd.DataFrame, reference: pd.DataFrame) -> MonitorContext:
    """A reference built the same way `make monitor` builds it."""
    rng = np.random.default_rng(int(cfg.seed))
    reference_scores = rng.beta(1.5, 60.0, size=len(reference))

    return MonitorContext(
        cfg=cfg,
        reference_features=prepare(reference, cfg, use_age=False),
        reference_scores=reference_scores,
        domain_reference=lgbm_frame(
            reference_sample(reference, 2000, seed=20260917), cfg, use_age=False
        ),
        conformal_rates=rates(reference_scores, TAU_FRAUD, TAU_LEGIT),
        tau_fraud=TAU_FRAUD,
        tau_legit=TAU_LEGIT,
        feature_columns=feature_columns(cfg, use_age=False),
    )


def _window(reference: pd.DataFrame, seed: int, n: int = 2000) -> pd.DataFrame:
    """A window drawn from the reference population itself: nothing has moved."""
    return reference.sample(n=n, random_state=seed).reset_index(drop=True)


def test_a_window_from_the_reference_population_looks_quiet(context, reference) -> None:
    """The floor every threshold is calibrated against: no shift, no signal.

    If this drifted, every detector threshold in the project would be measuring
    the runner rather than the data.
    """
    window = _window(reference, seed=1)
    probs = np.random.default_rng(1).beta(1.5, 60.0, size=len(window))

    values = detectors_for_window(window, probs, context)

    assert values["psi_score"] < 0.1, "a same-distribution score PSI should be near zero"
    assert values["psi_feature_max"] < 0.25, "no feature moved, so none should look moved"
    assert 0.40 <= values["domain_auc"] <= 0.60, "the window should be indistinguishable"
    assert values["conformal_flagged"] == 0.0


def test_a_shifted_score_distribution_raises_the_score_psi(context, reference) -> None:
    """The detector that matters most when the model itself starts behaving oddly."""
    window = _window(reference, seed=2)
    quiet = np.random.default_rng(2).beta(1.5, 60.0, size=len(window))
    shifted = np.clip(quiet * 6.0, 0.0, 1.0)

    assert (
        detectors_for_window(window, shifted, context)["psi_score"]
        > detectors_for_window(window, quiet, context)["psi_score"]
    )


def test_a_crossing_rate_far_from_calibration_is_flagged(context, reference) -> None:
    """The conformal-rate test is the detector tied to the policy's own promise."""
    window = _window(reference, seed=3)
    # Every application over tau_legit: the exclusion rate cannot be anything else.
    certain = np.full(len(window), 0.9)

    values = detectors_for_window(window, certain, context, with_domain=False)

    assert values["conformal_excluded_rate"] == 1.0
    assert values["conformal_flagged"] == 1.0
    assert values["conformal_neglogp"] > 3.0, "a total shift should be far past p < 0.001"


def test_an_impossible_p_value_stays_a_finite_number(context, reference) -> None:
    """A p-value of zero would make the detector infinite and the threshold useless.

    `MIN_P_VALUE` floors it. Without that floor, one extreme window would poison
    every percentile computed over a batch of them.
    """
    window = _window(reference, seed=4)
    values = detectors_for_window(window, np.full(len(window), 0.9), context, with_domain=False)

    assert math.isfinite(values["conformal_neglogp"])
    assert values["conformal_neglogp"] <= 300.0


def test_skipping_the_domain_classifier_omits_it_rather_than_faking_it(context, reference) -> None:
    """The expensive detector can be skipped, but never silently defaulted.

    A skipped detector that reported 0.5 would calibrate a threshold against a
    measurement nobody made.
    """
    window = _window(reference, seed=5)
    probs = np.random.default_rng(5).beta(1.5, 60.0, size=len(window))

    values = detectors_for_window(window, probs, context, with_domain=False)

    assert "domain_auc" not in values
    assert set(thresholded_only(values)) == set(THRESHOLDED) - {"domain_auc"}


def test_only_thresholdable_numbers_reach_the_alarm_rules(context, reference) -> None:
    """`psi_feature_max_name` is a string. Compared against a threshold, it raises."""
    window = _window(reference, seed=6)
    probs = np.random.default_rng(6).beta(1.5, 60.0, size=len(window))

    values = detectors_for_window(window, probs, context)
    kept = thresholded_only(values)

    assert set(kept) == set(THRESHOLDED)
    assert all(isinstance(value, float) for value in kept.values())
    assert "psi_feature_max_name" in values, "the name is still reported, just not thresholded"


def test_the_same_window_measures_the_same_twice(context, reference) -> None:
    """Thresholds calibrated on one batch of windows must transfer to the next."""
    window = _window(reference, seed=7)
    probs = np.random.default_rng(7).beta(1.5, 60.0, size=len(window))

    first = detectors_for_window(window, probs, context)
    second = detectors_for_window(window, probs, context)

    assert thresholded_only(first) == thresholded_only(second)
