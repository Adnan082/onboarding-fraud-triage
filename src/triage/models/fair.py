"""Fairness mitigations M1-M4.

M1 drops age (usually not enough: other fields proxy for it).
M2 FairGBM, ``constraint_type="FPR"``, group ``customer_age >= 50``, training only.
M3 fairlearn ExponentiatedGradient with FalsePositiveRateParity.
M4 policy only: rely on the review band.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig


def fairgbm_available() -> bool:
    """FairGBM ships a Linux ``.so``; import it lazily and skip elsewhere."""
    try:
        import fairgbm  # noqa: F401
    except Exception:
        return False
    return True


def fit_m2_fairgbm(train: pd.DataFrame, cfg: DictConfig):
    """FairGBM with an FPR constraint on the age group. Linux only."""
    raise NotImplementedError("TODO(week 2): M2")


def fit_m3_fairlearn(train: pd.DataFrame, cfg: DictConfig):
    """ExponentiatedGradient + FalsePositiveRateParity.

    Returns a randomised classifier at ONE operating point, so evaluate it at that
    point rather than with threshold-sweep metrics. ``predict(X)`` needs no
    sensitive feature.
    """
    raise NotImplementedError("TODO(week 2): M3")
