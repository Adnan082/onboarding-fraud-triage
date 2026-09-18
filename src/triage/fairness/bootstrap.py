"""Percentile bootstrap CIs: 1,000 resamples, stratified by (label, age group), seeded.

Stratifying matters here. Fraud is about 1% of applications, so an unstratified
resample can draw a replicate with almost no fraud in one age group, and the
interval it produces is then mostly an artefact of the resampling.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

DEFAULT_RESAMPLES = 1000
DEFAULT_LEVEL = 0.95


def strata_from(*columns: np.ndarray) -> np.ndarray:
    """Build a stratum key from several columns, e.g. label and age group."""
    if not columns:
        raise ValueError("at least one column is needed to build strata")
    stacked = [np.asarray(column).ravel().astype(str) for column in columns]
    lengths = {values.size for values in stacked}
    if len(lengths) != 1:
        raise ValueError(f"stratum columns must be the same length, got {sorted(lengths)}")
    return np.array(["|".join(parts) for parts in zip(*stacked, strict=True)])


def stratum_members(strata: np.ndarray) -> list[np.ndarray]:
    """Row positions belonging to each stratum, in sorted stratum order.

    Computed once and reused across replicates: rescanning every row per stratum
    on each of 1,000 replicates dominated the runtime otherwise.
    """
    keys = np.asarray(strata).ravel()
    return [np.flatnonzero(keys == value) for value in np.unique(keys)]


def _resample(members: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One replicate, from strata that were grouped once."""
    if not members:
        return np.array([], dtype=int)
    return np.concatenate([rng.choice(group, size=group.size, replace=True) for group in members])


def stratified_resample_indices(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One bootstrap replicate: draw with replacement WITHIN each stratum.

    Each stratum keeps its original size, so the replicate has the same label and
    age composition as the data.
    """
    return _resample(stratum_members(strata), rng)


def stratified_bootstrap_ci(
    statistic: Callable[..., float],
    *arrays: np.ndarray,
    strata: np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(point, low, high)`` for ``statistic(*arrays)``.

    ``statistic`` is called with the same arrays, resampled row-wise. The interval
    is a percentile interval, and the same seed always gives the same interval.
    Replicates that cannot be computed (an empty class in the replicate, say) are
    dropped, and the result says so only by widening -- so check ``n_resamples``
    is large enough for the slice you are describing.
    """
    if not arrays:
        raise ValueError("at least one array is needed")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be strictly between 0 and 1, got {level}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")

    columns = [np.asarray(array).ravel() for array in arrays]
    keys = np.asarray(strata).ravel()
    lengths = {column.size for column in columns} | {keys.size}
    if len(lengths) != 1:
        raise ValueError(f"arrays and strata must be the same length, got {sorted(lengths)}")

    point = float(statistic(*columns))

    rng = np.random.default_rng(seed)
    members = stratum_members(keys)  # grouped once, reused by every replicate
    replicates: list[float] = []
    for _ in range(n_resamples):
        picks = _resample(members, rng)
        value = float(statistic(*(column[picks] for column in columns)))
        if not np.isnan(value):
            replicates.append(value)

    if not replicates:
        return point, float("nan"), float("nan")

    tail = (1.0 - level) / 2.0
    low, high = np.percentile(replicates, [100 * tail, 100 * (1.0 - tail)])
    return point, float(low), float(high)


def bootstrap_table(
    statistic: Callable[..., float],
    arrays: Sequence[np.ndarray],
    strata: np.ndarray,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 0,
) -> dict[str, float]:
    """The same interval, shaped for ``reports/metrics.json``."""
    point, low, high = stratified_bootstrap_ci(
        statistic, *arrays, strata=strata, n_resamples=n_resamples, level=level, seed=seed
    )
    return {
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "level": level,
        "n_resamples": float(n_resamples),
        "seed": float(seed),
    }
