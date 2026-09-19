"""One window in, one set of detector values out.

Everything the monitor measures goes through here, so a clean window, a test
month, a shifted variant and an injected bug are all scored exactly the same way.
That is the point: a threshold calibrated on clean windows is only meaningful if
the windows it is later applied to were measured identically.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from triage.features.encode import lgbm_frame, prepare
from triage.monitoring.conformal_rate import window_test
from triage.monitoring.domain_clf import domain_auc
from triage.monitoring.psi import psi, psi_table

log = logging.getLogger("triage")

# A p-value of 0 would be an infinite detector value, so it is floored here.
MIN_P_VALUE = 1e-300


@dataclass
class MonitorContext:
    """Everything a window is compared against. Built once, before any window."""

    cfg: DictConfig
    reference_features: pd.DataFrame  # training months, for PSI
    reference_scores: np.ndarray  # training-month probabilities, for score PSI
    domain_reference: pd.DataFrame  # fixed sample from cal_conf, model-ready
    conformal_rates: tuple[float, float]  # (excluded, fraud) rates on cal_conf
    tau_fraud: float
    tau_legit: float
    feature_columns: list[str]


def detectors_for_window(
    window: pd.DataFrame,
    probs: np.ndarray,
    context: MonitorContext,
    *,
    with_domain: bool = True,
) -> dict[str, Any]:
    """The four detector values for one window. All label-free.

    ``with_domain`` exists because the domain classifier is by far the most
    expensive detector; the 200 calibration windows can skip it when a run needs
    to be quick, at the cost of not calibrating its threshold.
    """
    cfg = context.cfg
    seed = int(cfg.seed)

    # 1 and 2: PSI on the score, and the worst feature PSI. The window is prepared
    # exactly as the model sees it, so a shift in a *_missing flag counts as drift
    # in its own right -- an upstream field going blank is a real failure mode.
    score_psi = psi(pd.Series(context.reference_scores), pd.Series(probs))
    feature_psi = psi_table(
        context.reference_features,
        prepare(window, cfg, use_age=False),
        features=context.feature_columns,
    )
    worst = feature_psi.iloc[0]

    values = {
        "psi_score": float(score_psi),
        "psi_feature_max": float(worst["psi"]),
        "psi_feature_max_name": str(worst["feature"]),  # kept for the report, not thresholded
    }

    # 3: the conformal-rate test, as a magnitude rather than a yes/no.
    rate_result = window_test(
        probs,
        context.tau_fraud,
        context.tau_legit,
        context.conformal_rates,
        p_value=float(cfg.monitor.detectors.conformal_rate.p_value),
    )
    smallest_p = min(float(rate_result["excluded_p_value"]), float(rate_result["fraud_p_value"]))
    values["conformal_neglogp"] = -math.log10(max(smallest_p, MIN_P_VALUE))
    values["conformal_excluded_rate"] = float(rate_result["excluded_rate"])
    values["conformal_fraud_rate"] = float(rate_result["fraud_rate"])
    values["conformal_flagged"] = float(bool(rate_result["flagged"]))

    # 4: the domain classifier.
    if with_domain:
        window_features = lgbm_frame(window, cfg, use_age=False)
        values["domain_auc"] = domain_auc(
            window_features[context.domain_reference.columns],
            context.domain_reference,
            seed=seed,
            n_estimators=int(cfg.monitor.detectors.domain_clf.n_estimators),
            cv_folds=int(cfg.monitor.detectors.domain_clf.cv_folds),
        )

    return values


# Only these are compared against a calibrated threshold. The rest are context.
THRESHOLDED = ("psi_score", "psi_feature_max", "conformal_neglogp", "domain_auc")


def thresholded_only(values: dict[str, Any]) -> dict[str, float]:
    """Drop the descriptive fields, keeping what the alarm rules act on."""
    return {name: values[name] for name in THRESHOLDED if name in values}
