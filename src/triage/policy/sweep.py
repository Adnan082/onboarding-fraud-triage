"""Alpha sweep on ``cal_tune``, then final thresholds on ``cal_conf`` (section 8.4).

1. Sweep ``alpha_fraud`` x ``alpha_legit`` with thresholds computed on ``cal_tune``.
2. Pick the pair with the highest fraud coverage whose review share fits capacity.
3. Only then compute the final thresholds on ``cal_conf``.

The order matters. ``cal_tune`` is where choices are made and ``cal_conf`` is where
the guarantee is earned. Selecting on ``cal_conf`` would tune the thresholds
against the very data meant to validate them, and the coverage guarantee would be
worth nothing.

A 90% catch guarantee will probably need a very large review share. The whole
trade-off curve is published, not just the chosen point.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from triage.policy.decide import band_shares, decide_many
from triage.uncertainty.conformal import (
    ConformalThresholds,
    coverage_report,
    fit_thresholds,
    predict_sets,
)

log = logging.getLogger("triage")

REPORTED = (
    "fraud_coverage",
    "genuine_exclusion_rate",
    "approve_share",
    "review_share",
    "verify_share",
    "empty_share",
)


def sweep(probs: np.ndarray, labels: np.ndarray, cfg: DictConfig) -> pd.DataFrame:
    """One row per alpha pair: coverage, exclusion and band shares.

    ``probs`` and ``labels`` are ``cal_tune``. Thresholds are fitted and evaluated
    on the same part here, which is acceptable because nothing is being guaranteed
    yet: this is the map used to choose, not the measurement that validates.
    """
    rows = []
    for alpha_fraud in cfg.policy.sweep.alpha_fraud:
        for alpha_legit in cfg.policy.sweep.alpha_legit:
            thresholds = fit_thresholds(
                probs, labels, alpha_fraud=float(alpha_fraud), alpha_legit=float(alpha_legit)
            )
            report = coverage_report(probs, labels, thresholds)
            rows.append(
                {
                    "alpha_fraud": float(alpha_fraud),
                    "alpha_legit": float(alpha_legit),
                    "tau_fraud": thresholds.tau_fraud,
                    "tau_legit": thresholds.tau_legit,
                    **{key: report[key] for key in REPORTED},
                }
            )
    return pd.DataFrame(rows)


def select_within_capacity(grid: pd.DataFrame, cfg: DictConfig) -> dict[str, float]:
    """Best fraud coverage whose review and verify shares fit ``policy.capacity``.

    Capacity is an owner-set assumption (section 3), not a property of the data.
    If nothing fits, that is itself a finding -- the team cannot run this policy at
    this capacity -- so it raises rather than quietly relaxing the constraint.
    """
    capacity = cfg.policy.capacity
    review_cap = float(capacity.review_share)
    verify_cap = float(capacity.verify_share)

    affordable = grid[(grid["review_share"] <= review_cap) & (grid["verify_share"] <= verify_cap)]
    if affordable.empty:
        cheapest = grid.loc[grid["review_share"].idxmin()]
        raise ValueError(
            f"no alpha pair fits capacity (review <= {review_cap:.1%}, "
            f"verify <= {verify_cap:.1%}). The smallest review share on the grid is "
            f"{cheapest['review_share']:.1%}, at alpha_fraud={cheapest['alpha_fraud']}, "
            f"alpha_legit={cheapest['alpha_legit']}. Raise the capacity or accept less coverage."
        )

    best = affordable.loc[affordable["fraud_coverage"].idxmax()]
    return {
        "alpha_fraud": float(best["alpha_fraud"]),
        "alpha_legit": float(best["alpha_legit"]),
        "expected_fraud_coverage": float(best["fraud_coverage"]),
        "expected_review_share": float(best["review_share"]),
        "expected_verify_share": float(best["verify_share"]),
    }


def apply_policy(
    probs: np.ndarray, labels: np.ndarray, thresholds: ConformalThresholds
) -> dict[str, float]:
    """Coverage and band shares for one slice, under thresholds fitted elsewhere."""
    decisions = decide_many(predict_sets(probs, thresholds))
    report = coverage_report(probs, labels, thresholds)
    return {**report, **{f"band_{name}": share for name, share in band_shares(decisions).items()}}


def coverage_by_age(
    probs: np.ndarray,
    labels: np.ndarray,
    age: np.ndarray,
    thresholds: ConformalThresholds,
    *,
    width: int = 10,
) -> pd.DataFrame:
    """Coverage per 10-year band (section 8.4).

    If coverage falls short for older applicants that is recorded as a finding.
    It is **not** fixed with age-specific thresholds: that would use age at
    decision time, which rule 4 forbids.
    """
    from triage.fairness.metrics import age_band

    bands = age_band(age, width)
    rows = []
    for band in sorted(set(bands.tolist()), key=lambda value: int(value.split("-")[0])):
        mask = bands == band
        report = coverage_report(probs[mask], labels[mask], thresholds)
        rows.append({"band": band, **{key: report[key] for key in ("n", "n_fraud", *REPORTED)}})
    return pd.DataFrame(rows)
