"""Conformal-rate test: do the threshold-crossing rates still match ``cal_conf``?"""

from __future__ import annotations

import numpy as np


def rates(probs: np.ndarray, tau_fraud: float, tau_legit: float) -> tuple[float, float]:
    """Share of applications with ``p > tau_legit`` and with ``p >= tau_fraud``."""
    raise NotImplementedError("TODO(week 2): two rates per window")


def binomial_flag(observed: int, n: int, expected_rate: float, *, p_value: float = 1e-3) -> bool:
    """Two-sided binomial test against the ``cal_conf`` rate; flag below ``p_value``."""
    raise NotImplementedError("TODO(week 2): scipy-free exact binomial or normal approximation")
