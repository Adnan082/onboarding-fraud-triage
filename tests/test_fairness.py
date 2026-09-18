"""Fairness metrics and bootstrap CIs.

Required checks (CLAUDE.md section 13):
- FPR ratio on a hand-computed confusion matrix;
- the stratified bootstrap is reproducible with a seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from triage.fairness.bootstrap import (
    bootstrap_table,
    strata_from,
    stratified_bootstrap_ci,
    stratified_resample_indices,
)
from triage.fairness.metrics import (
    NOTABLE_RANGE,
    age_band,
    flag_notable,
    fpr,
    fpr_by_band,
    fpr_by_group,
    fpr_ratio,
    relative_likelihood,
)

# A hand-built slice. Ten genuine applicants aged 60, four of them flagged, and
# ten genuine aged 30, one of them flagged; plus a few frauds, which must not
# enter the FPR at all.
HAND_AGE = np.array([60] * 10 + [30] * 10 + [60, 30])
HAND_LABEL = np.array([0] * 20 + [1, 1])
HAND_FLAG = np.array(
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # older genuine: 4 of 10 flagged -> FPR 0.4
    + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # younger genuine: 1 of 10 flagged -> FPR 0.1
    + [1, 1]  # the two frauds, correctly flagged: irrelevant to FPR
)


def test_fpr_counts_only_genuine_applicants() -> None:
    """A flagged fraud is not a false alarm."""
    assert fpr(HAND_LABEL, HAND_FLAG) == pytest.approx(5 / 20)
    assert fpr(np.array([1, 1]), np.array([1, 1])) != 1.0
    assert np.isnan(fpr(np.array([1, 1]), np.array([1, 1])))


def test_fpr_ratio_on_a_hand_computed_confusion_matrix() -> None:
    """0.1 / 0.4 = 0.25, whichever group happens to be worse off."""
    assert fpr_ratio(HAND_LABEL, HAND_FLAG, HAND_AGE, cut=50) == pytest.approx(0.25)

    # Symmetric: swapping which group is worse off leaves the ratio unchanged.
    swapped_age = np.where(HAND_AGE == 60, 30, 60)
    assert fpr_ratio(HAND_LABEL, HAND_FLAG, swapped_age, cut=50) == pytest.approx(0.25)


def test_fpr_ratio_is_one_when_the_groups_match() -> None:
    """Equal false-alarm rates give exactly 1."""
    age = np.array([60] * 10 + [30] * 10)
    label = np.zeros(20, dtype=int)
    flag = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0] * 2)
    assert fpr_ratio(label, flag, age, cut=50) == pytest.approx(1.0)


def test_fpr_ratio_is_nan_when_it_would_say_nothing() -> None:
    """No false alarms anywhere, or an empty group, gives nan rather than 0 or 1."""
    age = np.array([60] * 5 + [30] * 5)
    label = np.zeros(10, dtype=int)
    assert np.isnan(fpr_ratio(label, np.zeros(10, dtype=int), age, cut=50))
    assert np.isnan(fpr_ratio(label, np.ones(10, dtype=int), np.full(10, 30), cut=50))


def test_fpr_by_group_reports_the_counts_behind_the_rate() -> None:
    """A rate with no denominator is not reviewable."""
    table = fpr_by_group(HAND_LABEL, HAND_FLAG, HAND_AGE, cut=50).set_index("group")

    assert table.loc["age>=50", "n_genuine"] == 10
    assert table.loc["age>=50", "false_alarms"] == 4
    assert table.loc["age>=50", "fpr"] == pytest.approx(0.4)
    assert table.loc["age<50", "fpr"] == pytest.approx(0.1)


def test_age_bands_are_ten_years_wide() -> None:
    """Bands follow the decade the age is rounded to."""
    bands = age_band(np.array([10, 29, 30, 55, 90]), width=10)
    assert list(bands) == ["10-19", "20-29", "30-39", "50-59", "90-99"]


def test_fpr_by_band_is_in_band_order(fixture_frame: pd.DataFrame) -> None:
    """Bands come out in age order, so the report table reads top to bottom."""
    rng = np.random.default_rng(0)
    flags = rng.random(len(fixture_frame)) < 0.05

    table = fpr_by_band(
        fixture_frame["fraud_bool"].to_numpy(), flags, fixture_frame["customer_age"]
    )
    starts = [int(band.split("-")[0]) for band in table["band"]]
    assert starts == sorted(starts)
    assert (table["n_genuine"] >= table["false_alarms"]).all()


# --- relative likelihood -----------------------------------------------------------


def test_relative_likelihood_against_a_reference_group() -> None:
    """Each group's rate divided by the reference group's."""
    rates = pd.Series({"age<50": 0.10, "age>=50": 0.40, "unknown": 0.12})
    ratios = relative_likelihood(rates, reference="age<50")

    assert ratios["age<50"] == pytest.approx(1.0)
    assert ratios["age>=50"] == pytest.approx(4.0)
    assert ratios["unknown"] == pytest.approx(1.2)


def test_notable_flag_follows_the_dwp_range() -> None:
    """Outside 0.80-1.25 is notable; inside it is not."""
    low, high = NOTABLE_RANGE
    ratios = pd.Series({"a": 1.0, "b": low - 0.01, "c": high + 0.01, "d": high})
    flags = flag_notable(ratios)

    assert not flags["a"]
    assert flags["b"]
    assert flags["c"]
    assert not flags["d"], "the range is inclusive at its edges"


def test_relative_likelihood_rejects_a_useless_reference() -> None:
    """A zero or missing reference rate is an error, not an infinity."""
    rates = pd.Series({"a": 0.0, "b": 0.2})
    with pytest.raises(ValueError, match="rate of 0"):
        relative_likelihood(rates, reference="a")
    with pytest.raises(KeyError, match="not in"):
        relative_likelihood(rates, reference="missing")


# --- the bootstrap -----------------------------------------------------------------


def _fpr_ratio_statistic(label: np.ndarray, flag: np.ndarray, age: np.ndarray) -> float:
    return fpr_ratio(label, flag, age, cut=50)


def _wider_sample(seed: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A few thousand rows, so the interval is informative rather than saturated."""
    rng = np.random.default_rng(seed)
    age = rng.choice([30, 60], size=4000, p=[0.8, 0.2])
    label = (rng.random(4000) < 0.01).astype(int)
    base = np.where(age >= 50, 0.20, 0.05)
    flag = rng.random(4000) < np.where(label == 1, 0.8, base)
    return label, flag, age


def test_bootstrap_is_reproducible_with_a_seed() -> None:
    """The same seed gives exactly the same interval, every time."""
    strata = strata_from(HAND_LABEL, HAND_AGE >= 50)
    arrays = (HAND_LABEL, HAND_FLAG, HAND_AGE)

    first = stratified_bootstrap_ci(
        _fpr_ratio_statistic, *arrays, strata=strata, n_resamples=200, seed=7
    )
    again = stratified_bootstrap_ci(
        _fpr_ratio_statistic, *arrays, strata=strata, n_resamples=200, seed=7
    )
    assert first == again


def test_a_different_seed_gives_a_different_interval() -> None:
    """Reproducible, but still a random procedure: the seed is doing real work."""
    label, flag, age = _wider_sample()
    strata = strata_from(label, age >= 50)

    first = stratified_bootstrap_ci(
        _fpr_ratio_statistic, label, flag, age, strata=strata, n_resamples=200, seed=7
    )
    other = stratified_bootstrap_ci(
        _fpr_ratio_statistic, label, flag, age, strata=strata, n_resamples=200, seed=8
    )

    assert first[0] == other[0], "the point estimate does not depend on the seed"
    assert first[1:] != other[1:]


def test_a_tiny_slice_gives_a_useless_interval() -> None:
    """Twenty applicants cannot support a claim about an age gap.

    This is why every fairness number in the report carries its interval: on the
    hand-built slice the point estimate is 0.25 and the interval is the whole
    range, so the honest reading is "we cannot tell".
    """
    strata = strata_from(HAND_LABEL, HAND_AGE >= 50)
    point, low, high = stratified_bootstrap_ci(
        _fpr_ratio_statistic,
        HAND_LABEL,
        HAND_FLAG,
        HAND_AGE,
        strata=strata,
        n_resamples=200,
        seed=7,
    )
    assert point == pytest.approx(0.25)
    assert (low, high) == (0.0, 1.0)


def test_a_wider_sample_gives_a_usable_interval() -> None:
    """With a few thousand applications the interval narrows around the gap."""
    label, flag, age = _wider_sample()
    strata = strata_from(label, age >= 50)

    point, low, high = stratified_bootstrap_ci(
        _fpr_ratio_statistic, label, flag, age, strata=strata, n_resamples=500, seed=2
    )
    assert low < point < high
    assert high - low < 0.25, "a 4,000-row sample should pin the ratio down"
    assert high < 1.0, "the planted gap is real and the interval excludes parity"


def test_bootstrap_brackets_the_point_estimate() -> None:
    """The interval contains the statistic computed on the data itself."""
    strata = strata_from(HAND_LABEL, HAND_AGE >= 50)
    point, low, high = stratified_bootstrap_ci(
        _fpr_ratio_statistic,
        HAND_LABEL,
        HAND_FLAG,
        HAND_AGE,
        strata=strata,
        n_resamples=500,
        seed=1,
    )
    assert point == pytest.approx(0.25)
    assert low <= point <= high
    assert 0.0 <= low <= high <= 1.0


def test_resampling_preserves_stratum_sizes() -> None:
    """Each stratum keeps its size, so label and age composition are held fixed."""
    strata = strata_from(HAND_LABEL, HAND_AGE >= 50)
    rng = np.random.default_rng(0)
    picks = stratified_resample_indices(strata, rng)

    assert picks.size == strata.size
    original = pd.Series(strata).value_counts().sort_index()
    replicate = pd.Series(strata[picks]).value_counts().sort_index()
    assert original.equals(replicate)


def test_resampling_actually_resamples() -> None:
    """A replicate is a draw with replacement, not a copy of the data."""
    strata = strata_from(HAND_LABEL, HAND_AGE >= 50)
    rng = np.random.default_rng(3)
    picks = stratified_resample_indices(strata, rng)
    assert len(set(picks.tolist())) < picks.size


def test_bootstrap_rejects_mismatched_inputs() -> None:
    """A mis-joined array fails loudly rather than silently truncating."""
    with pytest.raises(ValueError, match="same length"):
        stratified_bootstrap_ci(
            _fpr_ratio_statistic,
            HAND_LABEL,
            HAND_FLAG[:-1],
            HAND_AGE,
            strata=strata_from(HAND_LABEL, HAND_AGE >= 50),
            n_resamples=10,
        )
    with pytest.raises(ValueError, match="same length"):
        strata_from(HAND_LABEL, HAND_AGE[:-1])


def test_bootstrap_argument_guards() -> None:
    """Bad settings are configuration errors, not quietly degraded intervals."""
    strata = strata_from(HAND_LABEL, HAND_AGE >= 50)

    with pytest.raises(ValueError, match="at least one array"):
        stratified_bootstrap_ci(_fpr_ratio_statistic, strata=strata)
    with pytest.raises(ValueError, match="level must be"):
        stratified_bootstrap_ci(
            _fpr_ratio_statistic, HAND_LABEL, HAND_FLAG, HAND_AGE, strata=strata, level=1.5
        )
    with pytest.raises(ValueError, match="n_resamples must be"):
        stratified_bootstrap_ci(
            _fpr_ratio_statistic, HAND_LABEL, HAND_FLAG, HAND_AGE, strata=strata, n_resamples=0
        )
    with pytest.raises(ValueError, match="at least one column"):
        strata_from()


def test_an_undefined_statistic_gives_an_undefined_interval() -> None:
    """If every replicate is nan, say nan rather than inventing a number."""
    age = np.full(20, 30)
    label = np.zeros(20, dtype=int)
    flag = np.zeros(20, dtype=bool)  # no false alarms anywhere: the ratio is undefined

    point, low, high = stratified_bootstrap_ci(
        _fpr_ratio_statistic, label, flag, age, strata=strata_from(label, age >= 50), n_resamples=20
    )
    assert np.isnan(point) and np.isnan(low) and np.isnan(high)


def test_bootstrap_table_records_its_own_settings() -> None:
    """The artefact carries the seed and the resample count, so it can be reproduced."""
    label, flag, age = _wider_sample()
    record = bootstrap_table(
        _fpr_ratio_statistic,
        [label, flag, age],
        strata_from(label, age >= 50),
        n_resamples=100,
        seed=5,
    )

    assert set(record) == {"point", "ci_low", "ci_high", "level", "n_resamples", "seed"}
    assert record["ci_low"] <= record["point"] <= record["ci_high"]
    assert record["n_resamples"] == 100
    assert record["seed"] == 5
    assert record["level"] == 0.95
