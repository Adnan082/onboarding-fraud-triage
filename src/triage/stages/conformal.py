"""Alpha sweep on cal_tune, final thresholds on cal_conf -> reports/tables/policy_grid.csv.

This is where the project's headline claim is made, so the order is strict:

1. sweep the alpha grid on ``cal_tune`` and publish the whole trade-off curve;
2. choose the pair with the best fraud coverage that the review team can absorb;
3. **only then** fit the final thresholds on ``cal_conf``, which nothing else has
   touched;
4. measure coverage on months 6 and 7 separately, and by age band.

Step 3 is what makes the guarantee mean anything. Coverage measured on the data
that chose the thresholds would be a description of the past, not a guarantee
about the future.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from triage.data.load import load_interim
from triage.data.split import deployment_protocol
from triage.evaluation.artefacts import update_section
from triage.fairness.metrics import fpr_by_group, fpr_ratio
from triage.models.champion import load_fitted
from triage.policy.cost import cost_breakdown
from triage.policy.decide import decide_many
from triage.policy.sweep import apply_policy, coverage_by_age, select_within_capacity, sweep
from triage.stages._base import CONFIG_PATH, start
from triage.uncertainty.conformal import coverage_report, fit_thresholds, predict_sets

log = logging.getLogger("triage")


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for ``make conformal``."""
    context = start(cfg, "conformal")
    started = time.perf_counter()

    frame = load_interim(cfg, "base")
    splits = deployment_protocol(frame, cfg)
    label = str(cfg.data.label)
    age_column = str(cfg.features.protected.age)
    loss_column = str(cfg.policy.costs.loss_proxy)

    model, calibrator, manifest = load_fitted(cfg)
    log.info("loaded %s (calibration: %s)", manifest["model_version"], calibrator.method)

    def probabilities(part: pd.DataFrame):
        return calibrator.transform(model.score(part))

    # 1. the trade-off curve, on cal_tune
    if splits.cal_tune is None or splits.cal_conf is None:
        raise ValueError("the deployment protocol must provide cal_tune and cal_conf")
    tune = frame.loc[splits.cal_tune]
    grid = sweep(probabilities(tune), tune[label].to_numpy(), cfg)

    grid_path = Path(cfg.paths.tables) / "policy_grid.csv"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(grid_path, index=False)
    log.info("swept %d alpha pairs on cal_tune -> %s", len(grid), grid_path)

    # 2. the choice, under the owner's capacity assumption
    capacity = cfg.policy.capacity
    try:
        chosen = select_within_capacity(grid, cfg)
        infeasible = None
        log.info(
            "chose alpha_fraud=%.2f, alpha_legit=%.2f: coverage %.3f at review %.3f, verify %.3f",
            chosen["alpha_fraud"],
            chosen["alpha_legit"],
            chosen["expected_fraud_coverage"],
            chosen["expected_review_share"],
            chosen["expected_verify_share"],
        )
    except ValueError as error:
        # A capacity nothing can satisfy is a finding for the owner, not a crash.
        infeasible = str(error)
        chosen = {
            "alpha_fraud": float(cfg.policy.alpha_fraud),
            "alpha_legit": float(cfg.policy.alpha_legit),
        }
        log.warning("%s", infeasible)
        log.warning(
            "falling back to the configured alphas (%.2f, %.2f), which are placeholders",
            chosen["alpha_fraud"],
            chosen["alpha_legit"],
        )

    # 3. the guarantee, fitted on cal_conf and nothing else
    conf = frame.loc[splits.cal_conf]
    thresholds = fit_thresholds(
        probabilities(conf),
        conf[label].to_numpy(),
        alpha_fraud=chosen["alpha_fraud"],
        alpha_legit=chosen["alpha_legit"],
    )
    log.info(
        "thresholds on cal_conf (%s rows, %s fraud): tau_fraud %.6f, tau_legit %.6f",
        f"{thresholds.n_legit + thresholds.n_fraud:,}",
        f"{thresholds.n_fraud:,}",
        thresholds.tau_fraud,
        thresholds.tau_legit,
    )

    # 4. what actually happens on the test months
    months: dict[str, dict] = {}
    for month, index in sorted((splits.test_by_month or {}).items()):
        part = frame.loc[index]
        probs = probabilities(part)
        labels = part[label].to_numpy()

        result = apply_policy(probs, labels, thresholds)
        result["by_age_band"] = coverage_by_age(
            probs,
            labels,
            part[age_column].to_numpy(),
            thresholds,
            width=int(cfg.features.protected.age_band_width),
        ).to_dict("records")
        months[str(month)] = result

        log.info(
            "  month %s: fraud coverage %.4f (target %.2f), genuine excluded %.4f "
            "(target %.2f), approve %.3f / review %.3f / verify %.3f",
            month,
            result["fraud_coverage"],
            1 - chosen["alpha_fraud"],
            result["genuine_exclusion_rate"],
            chosen["alpha_legit"],
            result["band_approve"],
            result["band_review"],
            result["band_verify"],
        )

    # 5. every alpha pair, evaluated on the test months, so the demo's capacity
    # slider shows what a policy would actually have done rather than what
    # cal_tune predicted it would do. Cheap: the scores are already computed.
    cut = int(cfg.features.protected.age_cut)
    conf_probs = probabilities(conf)
    conf_labels = conf[label].to_numpy()
    # Score each test month once, not once per alpha pair.
    scored_months = {
        int(month): (
            probabilities(frame.loc[index]),
            frame.loc[index, label].to_numpy(),
            frame.loc[index, age_column].to_numpy(),
            frame.loc[index, loss_column].to_numpy(),
        )
        for month, index in sorted((splits.test_by_month or {}).items())
    }

    outcomes = []
    for row in grid.itertuples():
        pair_thresholds = fit_thresholds(
            conf_probs,
            conf_labels,
            alpha_fraud=float(row.alpha_fraud),
            alpha_legit=float(row.alpha_legit),
        )
        for month, (probs, labels, age, loss) in scored_months.items():
            decisions = decide_many(predict_sets(probs, pair_thresholds))
            report = coverage_report(probs, labels, pair_thresholds)
            stopped = decisions != "approve"
            groups = {
                record["group"]: record["fpr"]
                for record in fpr_by_group(labels, stopped, age, cut=cut).to_dict("records")
            }
            costs = cost_breakdown(decisions, labels, loss, cfg)

            outcomes.append(
                {
                    "alpha_fraud": float(row.alpha_fraud),
                    "alpha_legit": float(row.alpha_legit),
                    "month": int(month),
                    "fraud_coverage": report["fraud_coverage"],
                    "genuine_exclusion_rate": report["genuine_exclusion_rate"],
                    "approve_share": report["approve_share"],
                    "review_share": report["review_share"],
                    "verify_share": report["verify_share"],
                    "n": report["n"],
                    "n_fraud": report["n_fraud"],
                    "fraud_caught": report["fraud_coverage"] * report["n_fraud"],
                    "fpr_ratio": fpr_ratio(labels, stopped, age, cut=cut),
                    "fpr_under_cut": groups.get(f"age<{cut}"),
                    "fpr_over_cut": groups.get(f"age>={cut}"),
                    "cost_total": costs["total"],
                    "cost_per_n": costs["per_n_applications"],
                }
            )

    outcomes_frame = pd.DataFrame(outcomes)
    outcomes_path = Path(cfg.paths.tables) / "policy_outcomes.csv"
    outcomes_frame.to_csv(outcomes_path, index=False)
    log.info(
        "wrote %s (%d rows: every alpha pair on both test months)", outcomes_path, len(outcomes)
    )

    payload = {
        "model_version": manifest["model_version"],
        "calibration": calibrator.method,
        "chosen": chosen,
        "outcomes_csv": str(outcomes_path.relative_to(Path(cfg.paths.root))),
        "capacity": {
            "review_share": float(capacity.review_share),
            "verify_share": float(capacity.verify_share),
            "note": "ASSUMPTION: set by the owner, not derived from the data (section 3)",
        },
        "infeasible": infeasible,
        "thresholds": thresholds.to_dict(),
        "thresholds_fitted_on": "cal_conf",
        "grid_csv": str(grid_path.relative_to(Path(cfg.paths.root))),
        "grid": grid.to_dict("records"),
        "months": months,
        "seconds": time.perf_counter() - started,
    }
    update_section(Path(cfg.paths.metrics), "policy", payload, context)
    log.info("wrote %s -> policy (%.1fs)", cfg.paths.metrics, payload["seconds"])


if __name__ == "__main__":
    main()
