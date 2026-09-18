"""PSI per feature and on the score, computed in DuckDB (``psi.sql``).

PSI compares a window of applications with a reference population, bin by bin::

    PSI = sum over bins of (p_window - p_reference) * ln(p_window / p_reference)

Identical distributions give exactly 0. The rule of thumb is 0.10 to watch and
0.25 to alert, but the thresholds this project acts on are the empirical ones
calibrated on clean windows (section 8.5); the rule of thumb is reported beside
them, not instead of them.

This module keeps two paths in step: a small pandas implementation, which is the
definition, and the DuckDB query, which is what runs over the real volumes. The
tests assert they agree.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SQL_PATH = Path(__file__).with_name("psi.sql")

EPSILON = 1e-4
N_BINS = 10  # deciles
OTHER = "other"


def is_numeric(values: pd.Series) -> bool:
    """Whether a column is binned by deciles rather than by category."""
    return pd.api.types.is_numeric_dtype(values)


def numeric_bins(reference: pd.Series, n_bins: int = N_BINS) -> pd.DataFrame:
    """Reference deciles as half-open ``[lo, hi)`` bins, open at both ends.

    Tied quantiles are collapsed, so a column that is mostly one value (a -1
    sentinel, say) produces fewer than ``n_bins`` bins rather than empty ones.
    """
    values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("reference has no numeric values to bin")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    interior = np.unique(np.quantile(values, quantiles))
    edges = np.concatenate([[-np.inf], interior, [np.inf]])

    return pd.DataFrame(
        {
            "bin": [f"[{lo:g},{hi:g})" for lo, hi in zip(edges[:-1], edges[1:], strict=True)],
            "lo": edges[:-1],
            "hi": edges[1:],
        }
    )


def categorical_levels(reference: pd.Series) -> list[str]:
    """Every category the reference saw, plus an ``other`` bucket."""
    levels = sorted({str(value) for value in reference.dropna().unique()})
    return [*levels, OTHER]


def _assign_numeric(values: pd.Series, bins: pd.DataFrame) -> pd.Series:
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    index = np.searchsorted(bins["hi"].to_numpy(), numbers, side="right")
    index = np.clip(index, 0, len(bins) - 1)
    return pd.Series(bins["bin"].to_numpy()[index], index=values.index)


def _assign_categorical(values: pd.Series, levels: list[str]) -> pd.Series:
    known = set(levels) - {OTHER}
    strings = values.astype(str)
    return strings.where(strings.isin(known), OTHER)


def psi_from_counts(
    reference_counts: pd.Series, window_counts: pd.Series, *, epsilon: float = EPSILON
) -> float:
    """PSI from two bin-count series. The definition, in one place."""
    bins = reference_counts.index.union(window_counts.index)
    reference = reference_counts.reindex(bins, fill_value=0).to_numpy(dtype=float)
    window = window_counts.reindex(bins, fill_value=0).to_numpy(dtype=float)

    if reference.sum() == 0 or window.sum() == 0:
        raise ValueError("both the reference and the window need at least one row")

    p_reference = np.maximum(reference / reference.sum(), epsilon)
    p_window = np.maximum(window / window.sum(), epsilon)
    return float(np.sum((p_window - p_reference) * np.log(p_window / p_reference)))


def psi(reference: pd.Series, window: pd.Series, *, epsilon: float = EPSILON) -> float:
    """PSI between a reference column and a window of the same column.

    Numeric columns are binned on the reference deciles, categoricals on the
    reference categories plus ``other``. Identical distributions give 0.
    """
    if is_numeric(reference):
        bins = numeric_bins(reference)
        reference_bins = _assign_numeric(reference, bins)
        window_bins = _assign_numeric(window, bins)
    else:
        levels = categorical_levels(reference)
        reference_bins = _assign_categorical(reference, levels)
        window_bins = _assign_categorical(window, levels)

    return psi_from_counts(
        reference_bins.value_counts(), window_bins.value_counts(), epsilon=epsilon
    )


def _long_frames(
    reference: pd.DataFrame, window: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, ...]:
    """Reshape both frames into the long relations ``psi.sql`` expects."""
    bins: list[pd.DataFrame] = []
    ref_numeric, win_numeric = [], []
    ref_categorical, win_categorical = [], []

    for feature in features:
        if is_numeric(reference[feature]):
            feature_bins = numeric_bins(reference[feature])
            feature_bins.insert(0, "feature", feature)
            bins.append(feature_bins)
            ref_numeric.append(
                pd.DataFrame({"feature": feature, "value": pd.to_numeric(reference[feature])})
            )
            win_numeric.append(
                pd.DataFrame({"feature": feature, "value": pd.to_numeric(window[feature])})
            )
        else:
            levels = categorical_levels(reference[feature])
            ref_categorical.append(
                pd.DataFrame(
                    {"feature": feature, "value": _assign_categorical(reference[feature], levels)}
                )
            )
            win_categorical.append(
                pd.DataFrame(
                    {"feature": feature, "value": _assign_categorical(window[feature], levels)}
                )
            )

    def stack(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)

    return (
        stack(bins, ["feature", "bin", "lo", "hi"]),
        stack(ref_numeric, ["feature", "value"]),
        stack(win_numeric, ["feature", "value"]),
        stack(ref_categorical, ["feature", "value"]),
        stack(win_categorical, ["feature", "value"]),
    )


def psi_table(
    reference: pd.DataFrame,
    window: pd.DataFrame,
    *,
    features: list[str] | None = None,
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    """PSI for every monitored feature, computed in DuckDB.

    Returns one row per feature, sorted by PSI, worst first.
    """
    columns = (
        features
        if features is not None
        else [column for column in reference.columns if column in window.columns]
    )
    if not columns:
        raise ValueError("no shared columns between the reference and the window")

    bins, ref_numeric, win_numeric, ref_categorical, win_categorical = _long_frames(
        reference, window, columns
    )

    connection = duckdb.connect()
    try:
        connection.register("bins", bins)
        connection.register("ref_numeric", ref_numeric)
        connection.register("win_numeric", win_numeric)
        connection.register("ref_categorical", ref_categorical)
        connection.register("win_categorical", win_categorical)
        table = connection.execute(
            SQL_PATH.read_text(encoding="utf-8"), {"epsilon": epsilon}
        ).fetch_df()
    finally:
        connection.close()

    table["watch"] = table["psi"] >= 0.10  # rule of thumb, reported beside the
    table["alert"] = table["psi"] >= 0.25  # empirical thresholds, never instead
    return table.sort_values("psi", ascending=False).reset_index(drop=True)
