"""The two evaluation protocols (CLAUDE.md section 8.1).

**Paper protocol** exists only to compare with published results. It trains on
months 0-5 and sets its operating threshold on the *test* set at 5% FPR, exactly
as the BAF paper does. Every number it produces carries ``threshold_set_on:
"test"``, because a threshold fitted on the data it is then measured against is
optimistic by construction and must never be read as a deployment figure.

**Deployment protocol** is everything else. It trains on months 0-4, chooses the
threshold on ``cal_tune`` (a third of month 5), and reports months 6 and 7
**separately**. Pooled figures are produced too, but never stand alone.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from triage.data.split import deployment_protocol, paper_protocol
from triage.evaluation.metrics import discrimination, rates_at_threshold, threshold_at_fpr
from triage.fairness.bootstrap import bootstrap_table, strata_from
from triage.fairness.metrics import fpr_by_band, fpr_by_group, fpr_ratio
from triage.models.baselines import FittedModel, fit

log = logging.getLogger("triage")


def _tpr_statistic(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    from triage.evaluation.metrics import tpr_at_fpr

    if labels.min() == labels.max():
        return float("nan")  # a replicate with one class says nothing
    return tpr_at_fpr(labels, scores, target_fpr)


def evaluate_slice(
    frame: pd.DataFrame,
    scores: np.ndarray,
    cfg: DictConfig,
    *,
    threshold: float,
    with_ci: bool = True,
) -> dict:
    """Everything reported about one month: detection, realised rates, fairness.

    ``threshold`` was chosen elsewhere -- on the test set under the paper protocol,
    on ``cal_tune`` under the deployment one -- and is applied here unchanged.
    """
    labels = frame[cfg.data.label].to_numpy()
    age = frame[cfg.features.protected.age].to_numpy()
    cut = int(cfg.features.protected.age_cut)
    target_fpr = float(cfg.data.protocol.paper.threshold_fpr)

    flagged = scores >= threshold
    result: dict = {
        "discrimination": discrimination(labels, scores, target_fpr),
        "realised": rates_at_threshold(labels, scores, threshold),
        "fairness": {
            "age_cut": cut,
            "fpr_ratio": fpr_ratio(labels, flagged, age, cut=cut),
            "by_group": fpr_by_group(labels, flagged, age, cut=cut).to_dict("records"),
            "by_band": fpr_by_band(
                labels, flagged, age, width=int(cfg.features.protected.age_band_width)
            ).to_dict("records"),
        },
    }

    if with_ci:
        n_resamples = int(cfg.bootstrap.n_resamples)
        seed = int(cfg.seed)
        older = (age >= cut).astype(int)

        result["discrimination"]["tpr_ci"] = bootstrap_table(
            lambda y, s: _tpr_statistic(y, s, target_fpr),
            [labels, scores],
            strata_from(labels),
            n_resamples=n_resamples,
            seed=seed,
        )
        result["fairness"]["fpr_ratio_ci"] = bootstrap_table(
            lambda y, f, a: fpr_ratio(y, f, a, cut=cut),
            [labels, flagged, age],
            strata_from(labels, older),
            n_resamples=n_resamples,
            seed=seed,
        )

    return result


def _evaluate_months(
    frame: pd.DataFrame,
    model: FittedModel,
    cfg: DictConfig,
    months: dict[int, pd.Index],
    threshold: float,
    *,
    with_ci: bool,
) -> dict:
    months_out = {}
    for month, index in sorted(months.items()):
        part = frame.loc[index]
        months_out[str(month)] = evaluate_slice(
            part, model.score(part), cfg, threshold=threshold, with_ci=with_ci
        )
        realised = months_out[str(month)]["realised"]
        log.info(
            "  month %s: TPR %.4f at FPR %.4f (n=%s)",
            month,
            realised["tpr"],
            realised["fpr"],
            f"{int(realised['n']):,}",
        )
    return months_out


def run_paper_protocol(frame: pd.DataFrame, cfg: DictConfig, *, with_ci: bool = True) -> dict:
    """Train on months 0-5; threshold at 5% FPR **on the test set**, as the paper does."""
    splits = paper_protocol(frame, cfg)
    target_fpr = float(cfg.data.protocol.paper.threshold_fpr)

    model = fit(frame.loc[splits.train], cfg)
    log.info(
        "paper protocol: %s trained on %s rows (%s fraud)",
        model.name,
        f"{model.n_train:,}",
        f"{model.n_train_fraud:,}",
    )

    pooled = frame.loc[splits.test_all]
    pooled_scores = model.score(pooled)
    # The one place a threshold may touch test data: rule 2's stated exception.
    threshold = threshold_at_fpr(pooled[cfg.data.label].to_numpy(), pooled_scores, target_fpr)

    return {
        "model": model.name,
        "protocol": "paper",
        "threshold": threshold,
        "threshold_set_on": "test",
        "threshold_caveat": (
            "Set on the test set at 5% FPR, as the BAF paper does. Optimistic by "
            "construction: never read as a deployment figure."
        ),
        "train_months": list(cfg.data.protocol.paper.train_months),
        "n_train": model.n_train,
        "n_train_fraud": model.n_train_fraud,
        "months": _evaluate_months(
            frame, model, cfg, splits.test_by_month or {}, threshold, with_ci=with_ci
        ),
        "pooled": evaluate_slice(pooled, pooled_scores, cfg, threshold=threshold, with_ci=with_ci),
    }


def run_deployment_protocol(frame: pd.DataFrame, cfg: DictConfig, *, with_ci: bool = True) -> dict:
    """Train on months 0-4; threshold on ``cal_tune``; months 6 and 7 reported separately."""
    splits = deployment_protocol(frame, cfg)
    target_fpr = float(cfg.data.protocol.paper.threshold_fpr)

    model = fit(frame.loc[splits.train], cfg)
    log.info(
        "deployment protocol: %s trained on %s rows (%s fraud)",
        model.name,
        f"{model.n_train:,}",
        f"{model.n_train_fraud:,}",
    )

    if splits.cal_tune is None:
        raise ValueError("the deployment protocol needs cal_tune to set its threshold")
    tune = frame.loc[splits.cal_tune]
    threshold = threshold_at_fpr(tune[cfg.data.label].to_numpy(), model.score(tune), target_fpr)
    log.info("  threshold %.6f chosen on cal_tune (%s rows)", threshold, f"{len(tune):,}")

    pooled_frame = frame.loc[splits.test_all]
    return {
        "model": model.name,
        "protocol": "deployment",
        "threshold": threshold,
        "threshold_set_on": "cal_tune",
        "train_months": list(cfg.data.protocol.deployment.train_months),
        "n_train": model.n_train,
        "n_train_fraud": model.n_train_fraud,
        "split_sizes": splits.sizes(),
        "months": _evaluate_months(
            frame, model, cfg, splits.test_by_month or {}, threshold, with_ci=with_ci
        ),
        "pooled": evaluate_slice(
            pooled_frame, model.score(pooled_frame), cfg, threshold=threshold, with_ci=with_ci
        ),
    }
