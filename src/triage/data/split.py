"""Time-based splits. Rule 1: split by month only, never at random.

Rule 2 sits on top of this: nothing is ever tuned on months 6-7, and the one
exception -- the paper protocol's threshold, which the BAF paper sets on test
data -- is labelled as such wherever it appears.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig

CALIBRATION_PARTS = ("cal_prob", "cal_tune", "cal_conf")


@dataclass(frozen=True)
class Splits:
    """Row index sets for one protocol. Month sets never overlap."""

    train: pd.Index
    cal_prob: pd.Index | None = None
    cal_tune: pd.Index | None = None
    cal_conf: pd.Index | None = None
    test_by_month: dict[int, pd.Index] | None = None
    protocol: str = "unknown"
    # True only for the paper protocol, whose threshold is set on test data.
    threshold_on_test: bool = False

    @property
    def test_all(self) -> pd.Index:
        """Every test row, pooled. Never report pooled results on their own."""
        if not self.test_by_month:
            return pd.Index([])
        return pd.Index(np.concatenate([index.to_numpy() for index in self.test_by_month.values()]))

    def sizes(self) -> dict[str, int]:
        """Row counts per part, for the run log and the manifest."""
        counts = {"train": len(self.train)}
        for part in CALIBRATION_PARTS:
            index = getattr(self, part)
            if index is not None:
                counts[part] = len(index)
        for month, index in (self.test_by_month or {}).items():
            counts[f"test_month_{month}"] = len(index)
        return counts


def _months(frame: pd.DataFrame, cfg: DictConfig) -> pd.Series:
    column = cfg.data.time_column
    if column not in frame.columns:
        raise KeyError(f"{column!r} is missing: splits are by month only (rule 1)")
    return frame[column]


def _rows_in_months(frame: pd.DataFrame, cfg: DictConfig, months: list[int]) -> pd.Index:
    return frame.index[_months(frame, cfg).isin(list(months))]


def _test_by_month(frame: pd.DataFrame, cfg: DictConfig, months: list[int]) -> dict[int, pd.Index]:
    """Months 6 and 7 stay separate: they are always reported separately."""
    return {int(month): _rows_in_months(frame, cfg, [int(month)]) for month in months}


def paper_protocol(frame: pd.DataFrame, cfg: DictConfig) -> Splits:
    """Train on months 0-5, test on 6-7. Comparison with published results only.

    The operating threshold for this protocol is set on the TEST set at 5% FPR,
    exactly as the BAF paper does. That is why ``threshold_on_test`` is True: any
    number carried out of this protocol must say so.
    """
    protocol = cfg.data.protocol.paper
    return Splits(
        train=_rows_in_months(frame, cfg, list(protocol.train_months)),
        test_by_month=_test_by_month(frame, cfg, list(protocol.test_months)),
        protocol="paper",
        threshold_on_test=str(protocol.threshold_on) == "test",
    )


def _stratified_three_way(
    frame: pd.DataFrame,
    rows: pd.Index,
    label: str,
    fractions: dict[str, float],
    seed: int,
) -> dict[str, pd.Index]:
    """Cut ``rows`` into three disjoint parts, stratified by label, seeded.

    Each label group is shuffled once and cut at the cumulative fractions, so the
    parts are disjoint by construction and every row lands in exactly one of them.
    """
    rng = np.random.default_rng(seed)
    parts: dict[str, list[pd.Index]] = {name: [] for name in fractions}

    labels = frame.loc[rows, label]
    for value in sorted(labels.unique()):
        group = rows[labels.to_numpy() == value]
        shuffled = group.to_numpy()[rng.permutation(len(group))]

        total = len(shuffled)
        names = list(fractions)
        # Cumulative cut points, so rounding never loses or duplicates a row.
        cuts = np.cumsum([fractions[name] for name in names], dtype=float)
        cuts = (cuts / cuts[-1] * total).round().astype(int)
        start = 0
        for name, stop in zip(names, cuts, strict=True):
            parts[name].append(pd.Index(shuffled[start:stop]))
            start = int(stop)

    return {
        name: pd.Index(np.concatenate([p.to_numpy() for p in pieces]))
        for name, pieces in parts.items()
    }


def deployment_protocol(frame: pd.DataFrame, cfg: DictConfig) -> Splits:
    """Train on 0-4; split month 5 into three disjoint, label-stratified, seeded parts.

    ``cal_prob`` fits probability calibration, ``cal_tune`` chooses the alphas and
    the operating threshold, and ``cal_conf`` computes conformal thresholds only --
    it is never used for any choice. Months 6 and 7 are evaluated separately.
    """
    protocol = cfg.data.protocol.deployment
    calibration_month = int(protocol.calibration_month)
    calibration_rows = _rows_in_months(frame, cfg, [calibration_month])
    if calibration_rows.empty:
        raise ValueError(f"month {calibration_month} has no rows: cannot calibrate")

    fractions = {part: float(protocol.calibration_split[part]) for part in CALIBRATION_PARTS}
    parts = _stratified_three_way(
        frame, calibration_rows, str(cfg.data.label), fractions, int(cfg.seed)
    )

    return Splits(
        train=_rows_in_months(frame, cfg, list(protocol.train_months)),
        cal_prob=parts["cal_prob"],
        cal_tune=parts["cal_tune"],
        cal_conf=parts["cal_conf"],
        test_by_month=_test_by_month(frame, cfg, list(protocol.test_months)),
        protocol="deployment",
        threshold_on_test=False,
    )


def simulated_windows(frame: pd.DataFrame, cfg: DictConfig) -> list[pd.Index]:
    """Shuffle within each month with the global seed, then cut into fixed-size windows.

    BAF has no within-month timestamps, so a window of ``data.window_size``
    applications stands in for one day of volume. State that assumption in the
    report. Windows are returned in month order; a short remainder at the end of a
    month is dropped, because a part-window is not a day.
    """
    size = int(cfg.data.window_size)
    if size <= 0:
        raise ValueError(f"window_size must be positive, got {size}")

    rng = np.random.default_rng(int(cfg.seed))
    months = _months(frame, cfg)
    windows: list[pd.Index] = []

    for month in sorted(months.unique()):
        rows = frame.index[months.to_numpy() == month].to_numpy()
        shuffled = rows[rng.permutation(len(rows))]
        for start in range(0, len(shuffled) - size + 1, size):
            windows.append(pd.Index(shuffled[start : start + size]))

    return windows


def window_frame(frame: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """The windows as a table: ``window_id``, ``month`` and ``n``, in month order."""
    records = []
    for window_id, index in enumerate(simulated_windows(frame, cfg)):
        month = frame.loc[index, cfg.data.time_column]
        records.append({"window_id": window_id, "month": int(month.iloc[0]), "n": len(index)})
    return pd.DataFrame.from_records(records, columns=["window_id", "month", "n"])
