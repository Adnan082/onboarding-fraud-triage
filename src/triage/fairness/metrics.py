"""Fairness metrics. Age measures outcomes; it never enters a decision (rule 4).

Everything here is computed on genuine applicants, because the question is how
often the model raises a false alarm about someone who has done nothing wrong --
predictive equality. The BAF paper groups age at 50, which splits the population
roughly 80/20.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The DWP fairness assessment treats a relative likelihood outside this range as
# worth explaining. Outside it, we mark the row "notable" -- not "unlawful": that
# is a legal question, and rule 9 says to flag it, never to answer it.
NOTABLE_RANGE = (0.80, 1.25)

OLDER, YOUNGER = "age>=50", "age<50"


def _as_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true).ravel().astype(int)
    flags = np.asarray(y_pred).ravel().astype(bool)
    if labels.shape != flags.shape:
        raise ValueError(
            f"y_true and y_pred must be the same length: {labels.shape} vs {flags.shape}"
        )
    return labels, flags


def fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False positive rate, computed on genuine applicants only.

    Returns ``nan`` for a slice with no genuine applicants in it, so an empty age
    band is visibly empty rather than silently zero.
    """
    labels, flags = _as_arrays(y_true, y_pred)
    genuine = labels == 0
    if not genuine.any():
        return float("nan")
    return float(flags[genuine].mean())


def fpr_ratio(y_true: np.ndarray, y_pred: np.ndarray, age: np.ndarray, cut: int = 50) -> float:
    """Predictive equality: ``min(FPR_older, FPR_younger) / max(...)``.

    1.0 means the two age groups see false alarms at the same rate; smaller is a
    wider gap. The BAF paper reports around 0.3 for its best Base models.
    """
    labels, flags = _as_arrays(y_true, y_pred)
    ages = np.asarray(age).ravel()
    older = ages >= cut

    rate_older = fpr(labels[older], flags[older])
    rate_younger = fpr(labels[~older], flags[~older])
    if np.isnan(rate_older) or np.isnan(rate_younger):
        return float("nan")
    if max(rate_older, rate_younger) == 0.0:
        return float("nan")  # no false alarms anywhere: the ratio says nothing
    return float(min(rate_older, rate_younger) / max(rate_older, rate_younger))


def fpr_by_group(
    y_true: np.ndarray, y_pred: np.ndarray, age: np.ndarray, cut: int = 50
) -> pd.DataFrame:
    """FPR for the two groups either side of the cut, with the counts behind it."""
    labels, flags = _as_arrays(y_true, y_pred)
    ages = np.asarray(age).ravel()

    rows = []
    for name, mask in ((OLDER, ages >= cut), (YOUNGER, ages < cut)):
        genuine = mask & (labels == 0)
        rows.append(
            {
                "group": name,
                "n": int(mask.sum()),
                "n_genuine": int(genuine.sum()),
                "false_alarms": int((genuine & flags).sum()),
                "fpr": fpr(labels[mask], flags[mask]),
            }
        )
    return pd.DataFrame(rows)


def age_band(age: np.ndarray, width: int = 10) -> np.ndarray:
    """Label each application with its 10-year band, e.g. ``"30-39"``."""
    ages = np.asarray(age).ravel().astype(int)
    lower = (ages // width) * width
    return np.array([f"{low}-{low + width - 1}" for low in lower])


def fpr_by_band(
    y_true: np.ndarray, y_pred: np.ndarray, age: np.ndarray, width: int = 10
) -> pd.DataFrame:
    """FPR by 10-year age band, in band order."""
    labels, flags = _as_arrays(y_true, y_pred)
    bands = age_band(age, width)

    rows = []
    for band in sorted(set(bands.tolist()), key=lambda value: int(value.split("-")[0])):
        mask = bands == band
        genuine = mask & (labels == 0)
        rows.append(
            {
                "band": band,
                "n": int(mask.sum()),
                "n_genuine": int(genuine.sum()),
                "false_alarms": int((genuine & flags).sum()),
                "fpr": fpr(labels[mask], flags[mask]),
            }
        )
    return pd.DataFrame(rows)


def relative_likelihood(rates: pd.Series, reference: str) -> pd.Series:
    """Group rate / reference-group rate, as in the DWP assessment."""
    if reference not in rates.index:
        raise KeyError(f"reference group {reference!r} is not in {list(rates.index)}")
    baseline = float(rates[reference])
    if baseline == 0.0:
        raise ValueError(f"reference group {reference!r} has a rate of 0: the ratio is undefined")
    return rates.astype(float) / baseline


def flag_notable(ratios: pd.Series) -> pd.Series:
    """True where a relative likelihood falls outside 0.80-1.25.

    "Notable" means it needs explaining in the report. Whether it is lawful is a
    legal question for the owner, not for this code.
    """
    low, high = NOTABLE_RANGE
    values = ratios.astype(float)
    return (values < low) | (values > high)
