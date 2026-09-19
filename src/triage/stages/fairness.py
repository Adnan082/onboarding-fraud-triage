"""Mitigation experiments M1-M4, bootstrap CIs and the trade-off plot.

Four ways to narrow the false-alarm gap between age groups, and what each costs
in fraud caught:

- **M1** drop age. The champion already does this.
- **M2** FairGBM with an FPR constraint on the age group, applied while training.
- **M3** fairlearn ExponentiatedGradient with FalsePositiveRateParity.
- **M4** change no model at all: let the conformal policy's review band absorb it.

M3 has no threshold to sweep -- it is a randomised classifier at one operating
point -- so it is evaluated where it sits, and the trade-off plot marks it
differently for that reason.

M2 needs FairGBM, which is Linux-only. When it is unavailable the experiment is
skipped with a message and recorded as skipped, never silently dropped.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from triage.config import with_model
from triage.data.load import load_interim
from triage.data.split import deployment_protocol
from triage.evaluation.artefacts import read_metrics, update_section
from triage.evaluation.metrics import discrimination, rates_at_threshold, threshold_at_fpr
from triage.fairness.bootstrap import bootstrap_table, strata_from
from triage.fairness.metrics import fpr_by_band, fpr_by_group, fpr_ratio
from triage.models.champion import load_fitted
from triage.models.fair import SKIP_MESSAGE, fairgbm_available, fit_m2_fairgbm, fit_m3_fairlearn
from triage.policy.decide import decide_many
from triage.stages._base import CONFIG_PATH, start
from triage.uncertainty.conformal import ConformalThresholds, predict_sets

log = logging.getLogger("triage")


def _fairness_of(frame: pd.DataFrame, flagged: np.ndarray, cfg: DictConfig) -> dict:
    """FPR ratio with its interval, plus the group and band tables."""
    labels = frame[cfg.data.label].to_numpy()
    age = frame[str(cfg.features.protected.age)].to_numpy()
    cut = int(cfg.features.protected.age_cut)

    return {
        "fpr_ratio": fpr_ratio(labels, flagged, age, cut=cut),
        "fpr_ratio_ci": bootstrap_table(
            lambda y, f, a: fpr_ratio(y, f, a, cut=cut),
            [labels, flagged, age],
            strata_from(labels, (age >= cut).astype(int)),
            n_resamples=int(cfg.bootstrap.n_resamples),
            seed=int(cfg.seed),
        ),
        "by_group": fpr_by_group(labels, flagged, age, cut=cut).to_dict("records"),
        "by_band": fpr_by_band(
            labels, flagged, age, width=int(cfg.features.protected.age_band_width)
        ).to_dict("records"),
    }


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for ``make fairness``."""
    context = start(cfg, "fairness")
    started = time.perf_counter()

    frame = load_interim(cfg, "base")
    splits = deployment_protocol(frame, cfg)
    label = str(cfg.data.label)
    target_fpr = float(cfg.data.protocol.paper.threshold_fpr)

    train = frame.loc[splits.train]
    if splits.cal_tune is None:
        raise ValueError("the deployment protocol must provide cal_tune")
    tune = frame.loc[splits.cal_tune]
    months = {int(m): frame.loc[i] for m, i in sorted((splits.test_by_month or {}).items())}

    champion, calibrator, manifest = load_fitted(cfg)
    experiments: dict[str, dict] = {}

    def evaluate_scored(name: str, description: str, score_fn) -> None:
        """A mitigation that produces scores: threshold it on cal_tune, then measure."""
        threshold = threshold_at_fpr(tune[label].to_numpy(), score_fn(tune), target_fpr)
        per_month = {}
        for month, part in months.items():
            scores = score_fn(part)
            flagged = scores >= threshold
            per_month[str(month)] = {
                "discrimination": discrimination(part[label].to_numpy(), scores, target_fpr),
                "realised": rates_at_threshold(part[label].to_numpy(), scores, threshold),
                "fairness": _fairness_of(part, flagged, cfg),
            }
            log.info(
                "  %-18s month %s: TPR %.4f at FPR %.4f, FPR ratio %.3f",
                name,
                month,
                per_month[str(month)]["realised"]["tpr"],
                per_month[str(month)]["realised"]["fpr"],
                per_month[str(month)]["fairness"]["fpr_ratio"],
            )
        experiments[name] = {
            "description": description,
            "kind": "scored",
            "threshold": threshold,
            "threshold_set_on": "cal_tune",
            "months": per_month,
        }

    def evaluate_point(name: str, description: str, predict_fn) -> None:
        """A mitigation with a single operating point: measure it where it stands."""
        per_month = {}
        for month, part in months.items():
            flagged = predict_fn(part).astype(bool)
            labels = part[label].to_numpy()
            positives, negatives = labels == 1, labels == 0
            per_month[str(month)] = {
                "realised": {
                    "tpr": float(flagged[positives].mean()) if positives.any() else float("nan"),
                    "fpr": float(flagged[negatives].mean()) if negatives.any() else float("nan"),
                    "flag_rate": float(flagged.mean()),
                    "n": float(labels.size),
                },
                "fairness": _fairness_of(part, flagged, cfg),
            }
            log.info(
                "  %-18s month %s: TPR %.4f at FPR %.4f, FPR ratio %.3f",
                name,
                month,
                per_month[str(month)]["realised"]["tpr"],
                per_month[str(month)]["realised"]["fpr"],
                per_month[str(month)]["fairness"]["fpr_ratio"],
            )
        experiments[name] = {
            "description": description,
            "kind": "point",
            "note": "a randomised classifier at one operating point: no threshold to sweep",
            "months": per_month,
        }

    # --- M1: the champion, which never sees age -----------------------------------
    log.info("=== M1: drop age ===")
    evaluate_scored(
        "m1_drop_age",
        "The champion: LightGBM trained without customer_age.",
        lambda part: calibrator.transform(champion.score(part)),
    )

    # --- M2: FairGBM ---------------------------------------------------------------
    log.info("=== M2: FairGBM (FPR constraint) ===")
    if fairgbm_available():
        m2 = fit_m2_fairgbm(train, with_model(cfg, "fairgbm"))
        evaluate_scored(
            "m2_fairgbm",
            "FairGBM with constraint_type=FPR on the age group, applied during training.",
            m2.score,
        )
    else:
        experiments["m2_fairgbm"] = {"skipped": True, "reason": SKIP_MESSAGE}
        log.warning(SKIP_MESSAGE)

    # --- M3: fairlearn -------------------------------------------------------------
    log.info("=== M3: fairlearn ExponentiatedGradient ===")
    m3 = fit_m3_fairlearn(train, with_model(cfg, "fairlearn_eg"))
    evaluate_point(
        "m3_fairlearn_eg",
        "ExponentiatedGradient with FalsePositiveRateParity, age as a training constraint.",
        m3.predict,
    )

    # --- M4: change nothing; let the policy absorb it ------------------------------
    log.info("=== M4: policy only ===")
    policy = read_metrics(Path(cfg.paths.metrics)).get("policy")
    if policy:
        stored = policy["thresholds"]
        thresholds = ConformalThresholds(
            tau_fraud=float(stored["tau_fraud"]),
            tau_legit=float(stored["tau_legit"]),
            alpha_fraud=float(stored["alpha_fraud"]),
            alpha_legit=float(stored["alpha_legit"]),
            n_fraud=int(stored["n_fraud"]),
            n_legit=int(stored["n_legit"]),
        )

        def not_approved(part: pd.DataFrame) -> np.ndarray:
            probs = calibrator.transform(champion.score(part))
            return decide_many(predict_sets(probs, thresholds)) != "approve"

        evaluate_point(
            "m4_policy_only",
            "No model change: an application counts as stopped if the conformal policy "
            "sends it to review or verify rather than approving it.",
            not_approved,
        )
    else:
        experiments["m4_policy_only"] = {
            "skipped": True,
            "reason": "run `make conformal` first: no policy thresholds yet",
        }
        log.warning("M4 skipped: no policy thresholds in metrics.json")

    # --- the trade-off table ---------------------------------------------------------
    rows = []
    for name, experiment in experiments.items():
        if experiment.get("skipped"):
            continue
        for month, values in experiment["months"].items():
            rows.append(
                {
                    "experiment": name,
                    "kind": experiment["kind"],
                    "month": int(month),
                    "tpr": values["realised"]["tpr"],
                    "fpr": values["realised"]["fpr"],
                    "fpr_ratio": values["fairness"]["fpr_ratio"],
                    "fpr_ratio_low": values["fairness"]["fpr_ratio_ci"]["ci_low"],
                    "fpr_ratio_high": values["fairness"]["fpr_ratio_ci"]["ci_high"],
                }
            )
    table = pd.DataFrame(rows)
    table_path = Path(cfg.paths.tables) / "fairness_tradeoff.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    log.info("wrote %s", table_path)

    payload = {
        "model_version": manifest["model_version"],
        "target_fpr": target_fpr,
        "age_cut": int(cfg.features.protected.age_cut),
        "experiments": experiments,
        "tradeoff_csv": str(table_path.relative_to(Path(cfg.paths.root))),
        "tradeoff": rows,
        "seconds": time.perf_counter() - started,
    }
    update_section(Path(cfg.paths.metrics), "fairness", payload, context)
    log.info("wrote %s -> fairness (%.1fs)", cfg.paths.metrics, payload["seconds"])


if __name__ == "__main__":
    main()
