"""Train the champion plus probability calibration -> models/ and models/manifest.json.

The deployment protocol, in order:

1. fit the champion on months 0-4, without ``customer_age`` (rule 4);
2. fit every calibration candidate on ``cal_prob``;
3. choose between them by Brier score on ``cal_tune`` -- a different part of month
   5, so the choice is not made on the data that fitted them;
4. report calibration and detection on months 6 and 7, separately, and by age;
5. write the artefacts and a manifest the API can verify.

``cal_conf`` is not touched here. It exists to set conformal thresholds, and using
it for anything else would spend the guarantee before it is made.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

from triage.data.load import load_interim
from triage.data.split import deployment_protocol
from triage.evaluation.artefacts import update_section
from triage.evaluation.metrics import calibration_by_age, discrimination, threshold_at_fpr
from triage.evaluation.protocols import evaluate_slice
from triage.explain.reasons import training_medians
from triage.features.encode import lgbm_frame
from triage.models.calibrate import fit_all, reliability_curve, select
from triage.models.champion import fit, save
from triage.stages._base import CONFIG_PATH, start

log = logging.getLogger("triage")


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for ``make train``."""
    context = start(cfg, "train")
    started = time.perf_counter()

    frame = load_interim(cfg, "base")
    splits = deployment_protocol(frame, cfg)
    label = str(cfg.data.label)
    age_column = str(cfg.features.protected.age)
    cut = int(cfg.features.protected.age_cut)
    n_bins = int(cfg.calibration.ece_bins)

    # 1. the champion, on months 0-4 only
    model = fit(frame.loc[splits.train], cfg)
    log.info(
        "champion trained on %s rows (%s fraud), %d features, age excluded",
        f"{model.n_train:,}",
        f"{model.n_train_fraud:,}",
        len(model.estimator.feature_name_),
    )

    # 2. candidates fitted on cal_prob
    if splits.cal_prob is None or splits.cal_tune is None:
        raise ValueError("the deployment protocol must provide cal_prob and cal_tune")
    prob_part = frame.loc[splits.cal_prob]
    candidates = fit_all(model.score(prob_part), prob_part[label].to_numpy())
    log.info("fitted calibration candidates on cal_prob (%s rows)", f"{len(prob_part):,}")

    # 3. chosen on cal_tune
    tune_part = frame.loc[splits.cal_tune]
    tune_scores = model.score(tune_part)
    tune_labels = tune_part[label].to_numpy()
    best, comparison = select(cfg, candidates, tune_scores, tune_labels)
    for method, scores in sorted(comparison.items()):
        marker = " <- chosen" if method == best else ""
        log.info(
            "  %-9s brier %.6f  ECE %.5f  mean predicted %.5f vs observed %.5f%s",
            method,
            scores["brier"],
            scores["ece_equal_mass"],
            scores["mean_predicted_rate"],
            scores["observed_rate"],
            marker,
        )
    calibrator = candidates[best]

    # The operating threshold comes from cal_tune too, on calibrated probabilities.
    tune_probs = calibrator.transform(tune_scores)
    threshold = threshold_at_fpr(
        tune_labels, tune_probs, float(cfg.data.protocol.paper.threshold_fpr)
    )
    log.info("threshold %.6f chosen on cal_tune", threshold)

    # 4. what it does on the test months, separately
    months: dict[str, dict] = {}
    for month, index in sorted((splits.test_by_month or {}).items()):
        part = frame.loc[index]
        probs = calibrator.transform(model.score(part))
        labels = part[label].to_numpy()

        summary = evaluate_slice(part, probs, cfg, threshold=threshold)
        summary["calibration"] = calibration_by_age(
            labels, probs, part[age_column], cut=cut, n_bins=n_bins
        )
        summary["reliability"] = reliability_curve(probs, labels, n_bins=n_bins)
        months[str(month)] = summary

        log.info(
            "  month %s: ROC-AUC %.4f, TPR %.4f at FPR %.4f, Brier %.6f",
            month,
            summary["discrimination"]["roc_auc"],
            summary["realised"]["tpr"],
            summary["realised"]["fpr"],
            summary["calibration"]["overall"]["brier"],
        )

    # 5. artefacts, with the hashes the API checks at startup.
    # The training medians travel with the model: reason codes say "high" or "low"
    # relative to the population the model learned from, and the API has no access
    # to the training data to work that out for itself.
    manifest = save(
        model,
        calibrator,
        cfg,
        extra={
            "threshold": threshold,
            "threshold_set_on": "cal_tune",
            "split_sizes": splits.sizes(),
            "training_medians": training_medians(
                lgbm_frame(frame.loc[splits.train], cfg, use_age=False)
            ),
        },
    )

    payload = {
        "model_version": manifest["model_version"],
        "calibration": {
            "chosen": best,
            "select_by": str(cfg.calibration.select_by),
            "fit_on": "cal_prob",
            "selected_on": "cal_tune",
            "comparison": comparison,
            "cal_tune": calibration_by_age(
                tune_labels, tune_probs, tune_part[age_column], cut=cut, n_bins=n_bins
            ),
        },
        "threshold": threshold,
        "threshold_set_on": "cal_tune",
        "discrimination_cal_tune": discrimination(tune_labels, tune_probs),
        "split_sizes": splits.sizes(),
        "months": months,
        "seconds": time.perf_counter() - started,
    }
    update_section(Path(cfg.paths.metrics), "champion", payload, context)
    log.info("wrote %s -> champion (%.1fs)", cfg.paths.metrics, payload["seconds"])


if __name__ == "__main__":
    main()
