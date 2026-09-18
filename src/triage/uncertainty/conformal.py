"""Label-conditional split conformal prediction, written from scratch (rule 7).

MAPIE is only ever used to cross-check a *marginal* version of this code in tests
or notebooks; it is marginal, not label-conditional, and at ~1% prevalence it
produces many empty sets.

With calibrated fraud probability ``p(x)``, and ``n1`` frauds / ``n0`` genuine
applications in ``cal_conf``::

    k1 = ceil((n1 + 1) * (1 - alpha_fraud))    k0 = ceil((n0 + 1) * (1 - alpha_legit))
    tau_f = (n1 - k1 + 1)-th smallest p among cal_conf frauds   (-inf if k1 > n1)
    tau_l = k0-th smallest p among cal_conf genuine             (+inf if k0 > n0)

    "fraud" in C(x)  <=>  p(x) >= tau_f
    "legit" in C(x)  <=>  p(x) <= tau_l

Guarantees (exchangeability within each class):
``P(fraud in C | fraud) >= 1 - alpha_fraud`` and ``P(legit not in C | genuine) <= alpha_legit``.

Both thresholds are taken directly from the sorted calibration probabilities. We
do NOT compute ``tau_f = 1 - qhat``: isotonic calibration produces many tied
values, and the float round-trip can flip decisions on those ties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

# Column order of the boolean set matrix returned by ``predict_sets``.
LEGIT, FRAUD = 0, 1
CLASS_NAMES = ("legit", "fraud")


@dataclass(frozen=True)
class ConformalThresholds:
    """Thresholds computed on ``cal_conf``, plus the alphas that produced them."""

    tau_fraud: float
    tau_legit: float
    alpha_fraud: float
    alpha_legit: float
    n_fraud: int
    n_legit: int

    @property
    def policy_version(self) -> str:
        """The string the API reports as ``policy_version``."""
        return f"alpha_fraud={self.alpha_fraud:g},alpha_legit={self.alpha_legit:g}"

    def to_dict(self) -> dict[str, float | int | str]:
        """A JSON-safe record for the manifest and for ``reports/metrics.json``."""
        return {
            "tau_fraud": self.tau_fraud,
            "tau_legit": self.tau_legit,
            "alpha_fraud": self.alpha_fraud,
            "alpha_legit": self.alpha_legit,
            "n_fraud": self.n_fraud,
            "n_legit": self.n_legit,
            "policy_version": self.policy_version,
        }

    def with_alphas(self, alpha_fraud: float, alpha_legit: float) -> ConformalThresholds:
        """Return a copy carrying different alphas. Thresholds are NOT recomputed."""
        return replace(self, alpha_fraud=alpha_fraud, alpha_legit=alpha_legit)


def _check_alpha(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value!r}")


def _as_arrays(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels).ravel()
    if probs.shape != labels.shape:
        raise ValueError(
            f"probs and labels must be the same length: {probs.shape} vs {labels.shape}"
        )
    if probs.size and (np.isnan(probs).any() or np.isinf(probs).any()):
        raise ValueError("probs must be finite")
    unique = set(np.unique(labels).tolist())
    if not unique <= {0, 1}:
        raise ValueError(f"labels must be 0/1, found {sorted(unique)}")
    return probs, labels.astype(int)


def fit_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    alpha_fraud: float,
    alpha_legit: float,
) -> ConformalThresholds:
    """Compute label-conditional thresholds from a calibration set.

    ``probs`` are calibrated fraud probabilities and ``labels`` are 0/1, both for
    the SAME calibration rows. Raises ``ValueError`` if either class is empty:
    with no calibration examples of a class there is nothing to guarantee.
    """
    _check_alpha("alpha_fraud", alpha_fraud)
    _check_alpha("alpha_legit", alpha_legit)
    probs, labels = _as_arrays(probs, labels)

    fraud_probs = np.sort(probs[labels == 1])
    legit_probs = np.sort(probs[labels == 0])
    n1, n0 = fraud_probs.size, legit_probs.size
    if n1 == 0:
        raise ValueError("calibration set contains no fraud: cannot set tau_fraud")
    if n0 == 0:
        raise ValueError("calibration set contains no genuine applications: cannot set tau_legit")

    # Order statistics straight off the sorted calibration probabilities.
    k1 = math.ceil((n1 + 1) * (1.0 - alpha_fraud))
    tau_fraud = -np.inf if k1 > n1 else float(fraud_probs[n1 - k1])

    k0 = math.ceil((n0 + 1) * (1.0 - alpha_legit))
    tau_legit = np.inf if k0 > n0 else float(legit_probs[k0 - 1])

    return ConformalThresholds(
        tau_fraud=tau_fraud,
        tau_legit=tau_legit,
        alpha_fraud=float(alpha_fraud),
        alpha_legit=float(alpha_legit),
        n_fraud=int(n1),
        n_legit=int(n0),
    )


def predict_sets(probs: np.ndarray, thresholds: ConformalThresholds) -> np.ndarray:
    """Return a boolean ``(n, 2)`` array of set membership, columns ``[legit, fraud]``."""
    values = np.asarray(probs, dtype=float).ravel()
    sets = np.empty((values.size, 2), dtype=bool)
    sets[:, LEGIT] = values <= thresholds.tau_legit
    sets[:, FRAUD] = values >= thresholds.tau_fraud
    return sets


def set_labels(sets: np.ndarray) -> list[list[str]]:
    """Render the boolean set matrix as label lists, for the API and the demo."""
    return [
        [name for name, present in zip(CLASS_NAMES, row, strict=True) if present] for row in sets
    ]


def coverage_report(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: ConformalThresholds,
) -> dict[str, float]:
    """Coverage and band shares for one evaluation slice (a month, or an age band).

    ``fraud_coverage`` is the share of fraud whose set contains "fraud", and
    ``genuine_exclusion_rate`` the share of genuine applicants whose set lacks
    "legit". The three band shares sum to 1; ``empty_share`` is a diagnostic
    *inside* ``review_share``, because an empty set routes to review (rule 5).
    """
    probs, labels = _as_arrays(probs, labels)
    sets = predict_sets(probs, thresholds)
    has_legit, has_fraud = sets[:, LEGIT], sets[:, FRAUD]

    is_fraud = labels == 1
    is_legit = ~is_fraud
    empty = ~has_legit & ~has_fraud

    return {
        "n": float(labels.size),
        "n_fraud": float(is_fraud.sum()),
        "fraud_coverage": float(has_fraud[is_fraud].mean()) if is_fraud.any() else float("nan"),
        "genuine_exclusion_rate": float((~has_legit[is_legit]).mean())
        if is_legit.any()
        else float("nan"),
        "approve_share": float((has_legit & ~has_fraud).mean()),
        "review_share": float(((has_legit & has_fraud) | empty).mean()),
        "verify_share": float((has_fraud & ~has_legit).mean()),
        "empty_share": float(empty.mean()),
        "tau_fraud": thresholds.tau_fraud,
        "tau_legit": thresholds.tau_legit,
    }


def score_form_sets(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    *,
    alpha_fraud: float,
    alpha_legit: float,
) -> np.ndarray:
    """Textbook score form, with nonconformity ``1 - p_class``.

    This is the reference implementation ``test_conformal.py`` compares against:
    it computes the quantile of the nonconformity scores, rather than reading a
    threshold off the sorted probabilities. Production code uses the threshold
    form in :func:`fit_thresholds`.
    """
    _check_alpha("alpha_fraud", alpha_fraud)
    _check_alpha("alpha_legit", alpha_legit)
    cal_probs, cal_labels = _as_arrays(cal_probs, cal_labels)
    values = np.asarray(test_probs, dtype=float).ravel()

    fraud_scores = np.sort(1.0 - cal_probs[cal_labels == 1])  # nonconformity for "fraud"
    legit_scores = np.sort(cal_probs[cal_labels == 0])  # 1 - (1 - p) for "legit"
    n1, n0 = fraud_scores.size, legit_scores.size
    if n1 == 0 or n0 == 0:
        raise ValueError("calibration set must contain both classes")

    k1 = math.ceil((n1 + 1) * (1.0 - alpha_fraud))
    q_fraud = np.inf if k1 > n1 else float(fraud_scores[k1 - 1])
    k0 = math.ceil((n0 + 1) * (1.0 - alpha_legit))
    q_legit = np.inf if k0 > n0 else float(legit_scores[k0 - 1])

    sets = np.empty((values.size, 2), dtype=bool)
    sets[:, LEGIT] = values <= q_legit
    sets[:, FRAUD] = (1.0 - values) <= q_fraud
    return sets
