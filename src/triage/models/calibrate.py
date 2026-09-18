"""Probability calibration: none vs Platt vs isotonic.

Fit on ``cal_prob``, choose by Brier score on ``cal_tune``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import DictConfig
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

METHODS = ("none", "platt", "isotonic")


def ece_equal_mass(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error over equal-MASS bins.

    Equal-width bins are useless at ~1% prevalence: almost every application lands
    in the first bin and the metric stops saying anything. Bins here are cut on the
    ranks of the predictions, so each holds (as near as ties allow) the same number
    of applications, and each contributes in proportion to its size.
    """
    values = np.asarray(probs, dtype=float).ravel()
    targets = np.asarray(labels, dtype=float).ravel()
    if values.shape != targets.shape:
        raise ValueError(
            f"probs and labels must be the same length: {values.shape} vs {targets.shape}"
        )
    if values.size == 0:
        raise ValueError("no rows to score")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")

    order = np.argsort(values, kind="mergesort")  # stable, so ties keep their order
    total = 0.0
    for chunk in np.array_split(order, min(n_bins, values.size)):
        if chunk.size == 0:
            continue
        gap = abs(float(targets[chunk].mean()) - float(values[chunk].mean()))
        total += gap * chunk.size

    return total / values.size


@dataclass
class Calibrator:
    """A fitted probability calibrator, plus the method that produced it.

    ``transform`` maps raw model scores to calibrated probabilities. The ``none``
    method is the identity, so downstream code never has to special-case it.
    """

    method: str
    model: Any = None

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Calibrated fraud probabilities, clipped into [0, 1]."""
        values = np.asarray(scores, dtype=float).ravel()
        if self.method == "none":
            calibrated = values
        elif self.method == "platt":
            calibrated = self.model.predict_proba(_log_odds(values).reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            calibrated = self.model.predict(values)
        else:  # pragma: no cover - guarded at construction
            raise ValueError(f"unknown calibration method {self.method!r}")
        return np.clip(calibrated, 0.0, 1.0)


def _log_odds(scores: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Platt scaling fits a sigmoid, so it takes the log-odds rather than the probability.

    Feeding it a probability that is already near 0 -- and at ~1% prevalence most
    of them are -- leaves almost no range for the sigmoid to work with.
    """
    clipped = np.clip(np.asarray(scores, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def fit_calibrator(method: str, scores: np.ndarray, labels: np.ndarray) -> Calibrator:
    """Fit one calibrator on ``cal_prob``. ``method`` is none, platt or isotonic."""
    if method not in METHODS:
        raise ValueError(f"unknown calibration method {method!r}, expected one of {METHODS}")

    values = np.asarray(scores, dtype=float).ravel()
    targets = np.asarray(labels).ravel().astype(int)

    if method == "none":
        return Calibrator("none")

    if method == "platt":
        model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        model.fit(_log_odds(values).reshape(-1, 1), targets)
        return Calibrator("platt", model)

    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(values, targets)
    return Calibrator("isotonic", model)


def fit_all(scores: np.ndarray, labels: np.ndarray, methods: Iterable[str] = METHODS) -> dict:
    """Fit every candidate calibrator on ``cal_prob``."""
    return {method: fit_calibrator(method, scores, labels) for method in methods}


def select(
    cfg: DictConfig, candidates: dict, scores: np.ndarray, labels: np.ndarray
) -> tuple[str, dict]:
    """Return the method with the best Brier score on ``cal_tune``, and the comparison.

    ``cal_prob`` fitted these; ``cal_tune`` chooses between them. Using the same
    part for both would pick whichever method overfits hardest.
    """
    targets = np.asarray(labels).ravel().astype(float)
    n_bins = int(cfg.calibration.ece_bins)

    comparison = {}
    for method, calibrator in candidates.items():
        probs = calibrator.transform(scores)
        comparison[method] = {
            "brier": float(np.mean((probs - targets) ** 2)),
            "ece_equal_mass": ece_equal_mass(probs, targets, n_bins=n_bins),
            "mean_predicted_rate": float(probs.mean()),
            "observed_rate": float(targets.mean()),
        }

    criterion = str(cfg.calibration.select_by)
    best = min(comparison, key=lambda method: comparison[method][criterion])
    return best, comparison


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> list[dict]:
    """Equal-mass reliability points, stored so the figure can be drawn later.

    One row per bin: how many applications, what the model predicted on average,
    and what actually happened.
    """
    values = np.asarray(probs, dtype=float).ravel()
    targets = np.asarray(labels, dtype=float).ravel()
    order = np.argsort(values, kind="mergesort")

    rows = []
    for index, chunk in enumerate(np.array_split(order, min(n_bins, values.size))):
        if chunk.size == 0:
            continue
        rows.append(
            {
                "bin": index,
                "n": int(chunk.size),
                "mean_predicted": float(values[chunk].mean()),
                "observed": float(targets[chunk].mean()),
                "lower": float(values[chunk].min()),
                "upper": float(values[chunk].max()),
            }
        )
    return rows
