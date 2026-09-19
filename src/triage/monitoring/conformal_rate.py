"""Conformal-rate test: do the threshold-crossing rates still match ``cal_conf``?

This is the detector closest to the thing that actually matters. PSI and the
domain classifier ask whether the *inputs* have moved; this asks whether the
*policy* is still behaving as calibrated.

Two rates per window: the share of applications above ``tau_legit`` (heading for
verify or an empty set) and the share at or above ``tau_fraud`` (carrying "fraud"
in their set). Both are known exactly on ``cal_conf``, because that is where the
thresholds came from, so each window is a binomial sample against a known rate.

No scipy: the exact test needs the regularized incomplete beta, and scipy is not
a declared dependency of this project. A normal approximation with a continuity
correction is used instead, which is accurate at the sizes involved -- a 4,000
application window at a 1-5% rate gives np well above the usual rule of thumb.
The approximation is documented rather than hidden, and ``exact_small_n`` covers
the case where it would not hold.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_P_VALUE = 1e-3


def rates(probs: np.ndarray, tau_fraud: float, tau_legit: float) -> tuple[float, float]:
    """Share of applications with ``p > tau_legit`` and with ``p >= tau_fraud``.

    The comparisons match :func:`triage.uncertainty.conformal.predict_sets`
    exactly: "legit" is in the set when ``p <= tau_legit``, so the rate of
    *exclusion* is ``p > tau_legit``.
    """
    values = np.asarray(probs, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("no applications in the window")
    return float((values > tau_legit).mean()), float((values >= tau_fraud).mean())


def _normal_two_sided_p(observed: int, n: int, expected_rate: float) -> float:
    """Two-sided binomial p-value, normal approximation with continuity correction."""
    mean = n * expected_rate
    variance = n * expected_rate * (1.0 - expected_rate)
    if variance <= 0.0:
        # A degenerate expected rate: any deviation at all is decisive.
        return 0.0 if observed != mean else 1.0

    deviation = abs(observed - mean) - 0.5  # continuity correction
    if deviation <= 0.0:
        return 1.0
    z = deviation / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2.0))  # 2 * (1 - Phi(z)), without scipy


def exact_small_n(observed: int, n: int, expected_rate: float) -> float:
    """Exact two-sided binomial p-value, by summing outcomes at most as likely.

    Only for small ``n`` -- it sums ``n + 1`` terms. Used to check the
    approximation in tests rather than in the monitoring loop.
    """
    if n > 2000:
        raise ValueError("exact_small_n is for small samples; use the approximation")

    def pmf(k: int) -> float:
        return math.comb(n, k) * expected_rate**k * (1.0 - expected_rate) ** (n - k)

    observed_pmf = pmf(observed)
    total = sum(p for k in range(n + 1) if (p := pmf(k)) <= observed_pmf * (1 + 1e-12))
    return min(1.0, total)


def binomial_p_value(observed: int, n: int, expected_rate: float) -> float:
    """Two-sided p-value for ``observed`` successes in ``n`` against ``expected_rate``."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= expected_rate <= 1.0:
        raise ValueError(f"expected_rate must be a probability, got {expected_rate}")
    if not 0 <= observed <= n:
        raise ValueError(f"observed must be between 0 and {n}, got {observed}")
    return _normal_two_sided_p(observed, n, expected_rate)


def binomial_flag(
    observed: int, n: int, expected_rate: float, *, p_value: float = DEFAULT_P_VALUE
) -> bool:
    """Two-sided binomial test against the ``cal_conf`` rate; flag below ``p_value``."""
    return binomial_p_value(observed, n, expected_rate) < p_value


def window_test(
    probs: np.ndarray,
    tau_fraud: float,
    tau_legit: float,
    reference_rates: tuple[float, float],
    *,
    p_value: float = DEFAULT_P_VALUE,
) -> dict[str, float | bool]:
    """Both rates for one window, with their p-values and flags."""
    values = np.asarray(probs, dtype=float).ravel()
    n = values.size
    excluded_rate, fraud_rate = rates(values, tau_fraud, tau_legit)
    expected_excluded, expected_fraud = reference_rates

    return {
        "n": float(n),
        "excluded_rate": excluded_rate,
        "expected_excluded_rate": expected_excluded,
        "excluded_p_value": binomial_p_value(round(excluded_rate * n), n, expected_excluded),
        "fraud_rate": fraud_rate,
        "expected_fraud_rate": expected_fraud,
        "fraud_p_value": binomial_p_value(round(fraud_rate * n), n, expected_fraud),
        "flagged": bool(
            binomial_flag(round(excluded_rate * n), n, expected_excluded, p_value=p_value)
            or binomial_flag(round(fraud_rate * n), n, expected_fraud, p_value=p_value)
        ),
    }
