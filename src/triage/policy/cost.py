"""Illustrative cost frame. Never state a "£ saved" figure (rule 9).

Every parameter here is an assumption the owner sets, not a fact about the data
(section 3). The numbers exist to compare *policies against each other* under one
stated set of assumptions, which is a question the data can answer. What a policy
would save a real bank is a question it cannot, so this module never claims it.

The accounting, per section 8.3:

    review_cost   x applications sent to review
  + verify_cost   x applications sent to verify
  + loss_proxy    summed over fraud that was auto-approved
  + friction_cost x genuine applicants sent to verify

scaled to a fixed number of applications so policies with different volumes are
comparable. The sensitivity table is published alongside it, always, because the
ranking of two policies can flip when the assumed cost ratios move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

APPROVE, REVIEW, VERIFY = "approve", "review", "verify"


def cost_breakdown(
    decisions: np.ndarray,
    labels: np.ndarray,
    loss_proxy: np.ndarray,
    cfg: DictConfig,
    *,
    review_cost: float | None = None,
    verify_cost: float | None = None,
    friction_cost: float | None = None,
    loss_multiplier: float | None = None,
) -> dict[str, float]:
    """Each component of the cost, and the total, scaled per N applications.

    Overriding a cost is how the sensitivity table is built; left alone, the
    configured assumptions apply.
    """
    costs = cfg.policy.costs
    review_cost = float(costs.review_cost if review_cost is None else review_cost)
    verify_cost = float(costs.verify_cost if verify_cost is None else verify_cost)
    friction_cost = float(costs.friction_cost if friction_cost is None else friction_cost)
    loss_multiplier = float(
        costs.loss_proxy_multiplier if loss_multiplier is None else loss_multiplier
    )
    per_n = float(costs.per_n_applications)

    bands = np.asarray(decisions)
    outcomes = np.asarray(labels).ravel().astype(int)
    proxy = np.asarray(loss_proxy, dtype=float).ravel()
    if not (bands.shape[0] == outcomes.shape[0] == proxy.shape[0]):
        raise ValueError("decisions, labels and loss_proxy must be the same length")

    n = bands.shape[0]
    if n == 0:
        raise ValueError("no applications to cost")

    is_fraud = outcomes == 1
    review = bands == REVIEW
    verify = bands == VERIFY
    approve = bands == APPROVE

    # Fraud that was waved through is the expensive failure.
    missed_fraud = approve & is_fraud
    genuine_verified = verify & ~is_fraud

    components = {
        "review": review_cost * int(review.sum()),
        "verify": verify_cost * int(verify.sum()),
        "missed_fraud": loss_multiplier * float(proxy[missed_fraud].sum()),
        "friction": friction_cost * int(genuine_verified.sum()),
    }
    scale = per_n / n

    breakdown = {name: value * scale for name, value in components.items()}
    breakdown["total"] = sum(breakdown.values())
    breakdown["per_n_applications"] = per_n
    breakdown["n_review"] = float(review.sum()) * scale
    breakdown["n_verify"] = float(verify.sum()) * scale
    breakdown["n_missed_fraud"] = float(missed_fraud.sum()) * scale
    breakdown["n_genuine_verified"] = float(genuine_verified.sum()) * scale
    return breakdown


def cost_per_10k(
    decisions: np.ndarray,
    labels: np.ndarray,
    loss_proxy: np.ndarray,
    cfg: DictConfig,
) -> float:
    """The total cost, scaled to ``policy.costs.per_n_applications``."""
    return cost_breakdown(decisions, labels, loss_proxy, cfg)["total"]


def sensitivity_table(
    decisions: np.ndarray,
    labels: np.ndarray,
    loss_proxy: np.ndarray,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Cost over ``policy.costs.sensitivity_ratios``, published beside the point estimate.

    The ratio scales how expensive a missed fraud is relative to everything else.
    If the ranking of two policies survives the whole range, the comparison means
    something; if it flips, the comparison was really a statement about the
    assumption.
    """
    rows = []
    for ratio in cfg.policy.costs.sensitivity_ratios:
        breakdown = cost_breakdown(
            decisions,
            labels,
            loss_proxy,
            cfg,
            loss_multiplier=float(cfg.policy.costs.loss_proxy_multiplier) * float(ratio),
        )
        rows.append({"loss_ratio": float(ratio), **breakdown})
    return pd.DataFrame(rows)
