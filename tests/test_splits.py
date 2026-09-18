"""Time-based splits (section 8.1).

Required checks (CLAUDE.md section 13):
- no month appears in both training and evaluation;
- cal_prob, cal_tune and cal_conf are disjoint and label-stratified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from triage.data.split import (
    CALIBRATION_PARTS,
    deployment_protocol,
    paper_protocol,
    simulated_windows,
    window_frame,
)


@pytest.fixture(scope="module")
def cfg(repo_root):
    """The composed project config."""
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        return compose(config_name="config")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """A larger fixture: month 5 needs enough fraud to stratify three ways, and each
    month needs more rows than one 4,000-application window."""
    from tests.fixtures.make_fixture import make_fixture

    return make_fixture(n_rows=40_000, seed=20260917)


def _months_of(frame: pd.DataFrame, index: pd.Index) -> set[int]:
    return set(frame.loc[index, "month"].unique().tolist())


# --- no month is ever on both sides -----------------------------------------------


def test_paper_protocol_months(frame: pd.DataFrame, cfg) -> None:
    """Train on 0-5, test on 6 and 7, and the threshold is flagged as set on test."""
    splits = paper_protocol(frame, cfg)

    assert _months_of(frame, splits.train) == {0, 1, 2, 3, 4, 5}
    assert set(splits.test_by_month) == {6, 7}
    assert _months_of(frame, splits.test_by_month[6]) == {6}
    assert not _months_of(frame, splits.train) & set(splits.test_by_month)
    assert splits.threshold_on_test is True, "the paper sets its threshold on test data"


def test_deployment_protocol_months(frame: pd.DataFrame, cfg) -> None:
    """Train on 0-4, calibrate on 5, test on 6 and 7 separately."""
    splits = deployment_protocol(frame, cfg)

    train_months = _months_of(frame, splits.train)
    assert train_months == {0, 1, 2, 3, 4}
    assert 5 not in train_months, "month 5 is for calibration only"
    assert not train_months & set(splits.test_by_month)
    assert splits.threshold_on_test is False

    for part in CALIBRATION_PARTS:
        assert _months_of(frame, getattr(splits, part)) == {5}


def test_test_months_are_never_pooled_by_default(frame: pd.DataFrame, cfg) -> None:
    """Months 6 and 7 stay separate; pooling is available but never automatic."""
    splits = deployment_protocol(frame, cfg)
    assert len(splits.test_by_month) == 2
    assert len(splits.test_all) == len(splits.test_by_month[6]) + len(splits.test_by_month[7])


# --- the three calibration parts ---------------------------------------------------


def test_calibration_parts_are_disjoint_and_complete(frame: pd.DataFrame, cfg) -> None:
    """Every month-5 row lands in exactly one part."""
    splits = deployment_protocol(frame, cfg)
    parts = {part: set(getattr(splits, part)) for part in CALIBRATION_PARTS}

    for left in CALIBRATION_PARTS:
        for right in CALIBRATION_PARTS:
            if left < right:
                assert not parts[left] & parts[right], f"{left} and {right} overlap"

    month_five = set(frame.index[frame["month"] == 5])
    assert set().union(*parts.values()) == month_five


def test_calibration_parts_are_stratified_by_label(frame: pd.DataFrame, cfg) -> None:
    """Each part carries roughly the month's fraud prevalence, and none has zero fraud."""
    splits = deployment_protocol(frame, cfg)
    month_five = frame[frame["month"] == 5]
    overall = month_five["fraud_bool"].mean()

    for part in CALIBRATION_PARTS:
        rows = frame.loc[getattr(splits, part)]
        assert len(rows) == pytest.approx(len(month_five) / 3, rel=0.02)
        assert rows["fraud_bool"].sum() > 0, f"{part} has no fraud to calibrate on"
        # Stratified, so the count of frauds per part is within one of an even share.
        assert abs(rows["fraud_bool"].sum() - month_five["fraud_bool"].sum() / 3) <= 1
        assert rows["fraud_bool"].mean() == pytest.approx(overall, rel=0.05)


def test_splits_are_reproducible(frame: pd.DataFrame, cfg) -> None:
    """The same seed gives exactly the same three parts."""
    first = deployment_protocol(frame, cfg)
    second = deployment_protocol(frame, cfg)
    for part in CALIBRATION_PARTS:
        assert getattr(first, part).equals(getattr(second, part))


def test_sizes_reports_every_part(frame: pd.DataFrame, cfg) -> None:
    """The run log records what each part actually contained."""
    sizes = deployment_protocol(frame, cfg).sizes()
    assert set(sizes) == {
        "train",
        "cal_prob",
        "cal_tune",
        "cal_conf",
        "test_month_6",
        "test_month_7",
    }
    assert sizes["train"] > 0


def test_missing_calibration_month_raises(cfg) -> None:
    """A dataset without month 5 is an error, not a silent skip."""
    from tests.fixtures.make_fixture import make_fixture

    frame = make_fixture(n_rows=2000, seed=1)
    with pytest.raises(ValueError, match="cannot calibrate"):
        deployment_protocol(frame[frame["month"] != 5], cfg)


def test_missing_month_column_raises(frame: pd.DataFrame, cfg) -> None:
    """Rule 1 is enforced: no month column, no split."""
    with pytest.raises(KeyError, match="splits are by month only"):
        paper_protocol(frame.drop(columns=["month"]), cfg)


# --- simulated days ----------------------------------------------------------------


def test_simulated_windows_are_fixed_size_and_within_one_month(frame: pd.DataFrame, cfg) -> None:
    """Each window is one month's rows, shuffled, cut to the configured size."""
    windows = simulated_windows(frame, cfg)
    size = int(cfg.data.window_size)

    assert windows, "no windows produced"
    for index in windows:
        assert len(index) == size
        assert frame.loc[index, "month"].nunique() == 1


def test_simulated_windows_do_not_overlap(frame: pd.DataFrame, cfg) -> None:
    """No application appears in two days."""
    windows = simulated_windows(frame, cfg)
    all_rows = np.concatenate([index.to_numpy() for index in windows])
    assert len(all_rows) == len(set(all_rows.tolist()))


def test_window_frame_is_in_month_order(frame: pd.DataFrame, cfg) -> None:
    """Windows are numbered in month order, which the monitor relies on."""
    table = window_frame(frame, cfg)
    assert list(table["window_id"]) == sorted(table["window_id"])
    assert list(table["month"]) == sorted(table["month"])
    assert set(table["n"]) == {int(cfg.data.window_size)}


def test_part_windows_are_dropped(frame: pd.DataFrame, cfg) -> None:
    """A short remainder at the end of a month is not a day, so it is dropped."""
    size = int(cfg.data.window_size)
    per_month = frame.groupby("month").size()
    expected = int((per_month // size).sum())

    windows = simulated_windows(frame, cfg)
    assert len(windows) == expected
    assert len(windows) * size < len(frame), "this fixture should leave a remainder"


def test_windows_are_reproducible(frame: pd.DataFrame, cfg) -> None:
    """The same seed gives the same days."""
    first = simulated_windows(frame, cfg)
    second = simulated_windows(frame, cfg)
    assert all(a.equals(b) for a, b in zip(first, second, strict=True))
