"""Metric definitions (CLAUDE.md section 8.3). One definition, used everywhere.

Rounding lives in prose, never here: artefacts keep full precision.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from triage.models.calibrate import ece_equal_mass

DEFAULT_TARGET_FPR = 0.05


def _as_arrays(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true).ravel().astype(int)
    values = np.asarray(scores, dtype=float).ravel()
    if labels.shape != values.shape:
        raise ValueError(
            f"y_true and scores must be the same length: {labels.shape} vs {values.shape}"
        )
    if labels.size == 0:
        raise ValueError("no rows to score")
    return labels, values


def _best_index_under_fpr(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    labels, values = _as_arrays(y_true, scores)
    if labels.min() == labels.max():
        raise ValueError("both classes are needed to trace an ROC curve")

    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, values)
    allowed = np.flatnonzero(false_positive_rate <= target_fpr)
    # roc_curve always includes FPR = 0, so `allowed` is never empty.
    best = int(allowed[np.argmax(true_positive_rate[allowed])])
    return false_positive_rate, true_positive_rate, thresholds, best


def tpr_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float = DEFAULT_TARGET_FPR
) -> float:
    """Largest TPR with FPR <= ``target_fpr``, read off ``sklearn.metrics.roc_curve``."""
    _, true_positive_rate, _, best = _best_index_under_fpr(y_true, scores, target_fpr)
    return float(true_positive_rate[best])


def threshold_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float = DEFAULT_TARGET_FPR
) -> float:
    """The score threshold that realises :func:`tpr_at_fpr`.

    Applications are flagged when ``score >= threshold``.
    """
    _, _, thresholds, best = _best_index_under_fpr(y_true, scores, target_fpr)
    return float(thresholds[best])


def rates_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Realised confusion rates at a fixed threshold, for one test month.

    This is how a threshold chosen on ``cal_tune`` is reported on months 6 and 7:
    what the model actually did, not what the ROC curve promised.
    """
    labels, values = _as_arrays(y_true, scores)
    flagged = values >= threshold

    positives = labels == 1
    negatives = ~positives
    true_positives = int((flagged & positives).sum())
    false_positives = int((flagged & negatives).sum())

    return {
        "threshold": float(threshold),
        "n": float(labels.size),
        "n_fraud": float(positives.sum()),
        "flag_rate": float(flagged.mean()),
        "tpr": float(true_positives / positives.sum()) if positives.any() else float("nan"),
        "fpr": float(false_positives / negatives.sum()) if negatives.any() else float("nan"),
        "precision": float(true_positives / flagged.sum()) if flagged.any() else float("nan"),
        "n_flagged": float(flagged.sum()),
    }


def discrimination(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float = DEFAULT_TARGET_FPR
) -> dict[str, float]:
    """ROC-AUC, PR-AUC (average precision) and TPR at the target FPR."""
    labels, values = _as_arrays(y_true, scores)
    return {
        "roc_auc": float(roc_auc_score(labels, values)),
        "pr_auc": float(average_precision_score(labels, values)),
        f"tpr_at_{target_fpr:g}_fpr": tpr_at_fpr(labels, values, target_fpr),
        "prevalence": float(labels.mean()),
        "n": float(labels.size),
    }


def calibration(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    """Brier score, equal-mass ECE, and calibration-in-the-large.

    Calibration-in-the-large is the mean predicted rate against the observed rate:
    at ~1% prevalence a model can look sharp and still be badly off in the mean.
    """
    labels, values = _as_arrays(y_true, probs)
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("calibration needs probabilities in [0, 1]")

    mean_predicted = float(values.mean())
    observed = float(labels.mean())
    return {
        "brier": float(np.mean((values - labels) ** 2)),
        "ece_equal_mass": ece_equal_mass(values, labels, n_bins=n_bins),
        "mean_predicted_rate": mean_predicted,
        "observed_rate": observed,
        "calibration_in_the_large": mean_predicted - observed,
        "n_bins": float(n_bins),
        "n": float(labels.size),
    }


def calibration_by_age(
    y_true: np.ndarray,
    probs: np.ndarray,
    age: np.ndarray,
    *,
    cut: int = 50,
    n_bins: int = 15,
) -> dict[str, dict[str, float]]:
    """Calibration overall and for each age group (section 8.3).

    A model can be well calibrated on average and badly calibrated for older
    applicants. Since the conformal thresholds are set on pooled calibration data,
    that is exactly the case the coverage-by-age check has to survive.
    """
    ages = np.asarray(age).ravel()
    labels = np.asarray(y_true).ravel()
    values = np.asarray(probs, dtype=float).ravel()

    older = ages >= cut
    groups = {
        "overall": np.ones(labels.shape, dtype=bool),
        f"age>={cut}": older,
        f"age<{cut}": ~older,
    }
    return {
        name: calibration(labels[mask], values[mask], n_bins=n_bins)
        for name, mask in groups.items()
        if mask.any()
    }
