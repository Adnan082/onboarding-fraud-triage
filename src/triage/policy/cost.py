"""Illustrative cost frame. Never state a "£ saved" figure (rule 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def cost_per_10k(
    decisions: np.ndarray,
    labels: np.ndarray,
    loss_proxy: np.ndarray,
    cfg: DictConfig,
) -> float:
    """``review_cost*n_review + verify_cost*n_verify + sum(loss on missed fraud)
    + friction_cost*n_genuine_verified``, scaled to 10,000 applications."""
    raise NotImplementedError("TODO(week 2): cost accounting from policy.costs")


def sensitivity_table(*args, cfg: DictConfig) -> pd.DataFrame:
    """Cost over ``policy.costs.sensitivity_ratios``.

    Always published alongside the point estimate.
    """
    raise NotImplementedError("TODO(week 2): sweep the cost ratios")
