"""Clean windows, variant stress tests and injected bugs -> reports/monitoring.json.

The run, in order:

1. build the reference: training months for PSI, a fixed sample of ``cal_conf``
   for the domain classifier, and the ``cal_conf`` threshold-crossing rates;
2. calibrate each detector on bootstrap windows drawn from ``cal_prob + cal_tune``,
   which are clean by construction, and take the 99th percentile;
3. score the test months as they actually arrived. Anything that fires here is
   **natural drift**, reported separately -- calling it a false alarm would be
   wrong, because the population really did move;
4. score Variant IV and Variant V, which are the same applications under a
   different population mix;
5. inject each bug from window 5 of month 6 and measure how long it takes to
   reach ALERT, and what it costs in detection while undetected.

Nothing here uses a label to decide anything. Labels appear only to measure what
the drift cost, after the fact.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from experiments.inject_bugs import apply_bug
from triage.data.contract import check as contract_check
from triage.data.load import load_interim
from triage.data.split import deployment_protocol, simulated_windows
from triage.evaluation.artefacts import update_section
from triage.evaluation.metrics import rates_at_threshold
from triage.features.encode import feature_columns, lgbm_frame, prepare
from triage.models.champion import load_fitted
from triage.monitoring.alarms import ALERT, AlarmRun, calibrate_thresholds
from triage.monitoring.conformal_rate import rates
from triage.monitoring.domain_clf import reference_sample
from triage.monitoring.fallback import activate
from triage.monitoring.runner import MonitorContext, detectors_for_window, thresholded_only
from triage.stages._base import CONFIG_PATH, start
from triage.uncertainty.conformal import ConformalThresholds

log = logging.getLogger("triage")


def _policy_thresholds(cfg: DictConfig) -> ConformalThresholds:
    """The thresholds `make conformal` fitted on cal_conf."""
    from triage.evaluation.artefacts import read_metrics

    metrics = read_metrics(Path(cfg.paths.metrics))
    if "policy" not in metrics:
        raise ValueError("run `make conformal` before `make monitor`: no policy thresholds yet")

    stored = metrics["policy"]["thresholds"]
    return ConformalThresholds(
        tau_fraud=float(stored["tau_fraud"]),
        tau_legit=float(stored["tau_legit"]),
        alpha_fraud=float(stored["alpha_fraud"]),
        alpha_legit=float(stored["alpha_legit"]),
        n_fraud=int(stored["n_fraud"]),
        n_legit=int(stored["n_legit"]),
    )


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for ``make monitor``."""
    context_stamp = start(cfg, "monitor")
    started = time.perf_counter()

    frame = load_interim(cfg, "base")
    splits = deployment_protocol(frame, cfg)
    model, calibrator, manifest = load_fitted(cfg)
    thresholds = _policy_thresholds(cfg)
    columns = feature_columns(cfg, use_age=False)
    window_size = int(cfg.data.window_size)
    seed = int(cfg.seed)

    def probabilities(part: pd.DataFrame) -> np.ndarray:
        return calibrator.transform(model.score(part))

    # 1. the reference
    train = frame.loc[splits.train]
    if splits.cal_conf is None or splits.cal_prob is None or splits.cal_tune is None:
        raise ValueError("the deployment protocol must provide all three calibration parts")
    conf = frame.loc[splits.cal_conf]
    conf_probs = probabilities(conf)

    context = MonitorContext(
        cfg=cfg,
        reference_features=prepare(train, cfg, use_age=False),
        reference_scores=probabilities(train),
        domain_reference=lgbm_frame(
            reference_sample(conf, int(cfg.monitor.reference.domain_clf_rows), seed=seed),
            cfg,
            use_age=False,
        ),
        conformal_rates=rates(conf_probs, thresholds.tau_fraud, thresholds.tau_legit),
        tau_fraud=thresholds.tau_fraud,
        tau_legit=thresholds.tau_legit,
        feature_columns=columns,
    )
    log.info(
        "reference built: %s training rows, %s cal_conf rows, crossing rates %.4f / %.4f",
        f"{len(train):,}",
        f"{len(conf):,}",
        *context.conformal_rates,
    )

    # 2. calibrate on windows that are clean by construction
    clean_pool = pd.concat([frame.loc[splits.cal_prob], frame.loc[splits.cal_tune]])
    n_clean = int(cfg.monitor.threshold_calibration.clean_windows)
    percentile = float(cfg.monitor.threshold_calibration.percentile)
    rng = np.random.default_rng(seed)

    clean_values = []
    for index in range(n_clean):
        sample = clean_pool.sample(n=window_size, random_state=int(rng.integers(1 << 31)))
        values = thresholded_only(detectors_for_window(sample, probabilities(sample), context))
        clean_values.append(values)
        if (index + 1) % 50 == 0:
            log.info("  calibrated on %d/%d clean windows", index + 1, n_clean)

    detector_thresholds = calibrate_thresholds(clean_values, percentile=percentile)
    log.info("thresholds (%gth percentile of %d clean windows):", percentile, n_clean)
    for name, value in sorted(detector_thresholds.items()):
        log.info("  %-20s %.5f", name, value)

    # The false-alarm rate the thresholds imply, measured on the clean windows.
    clean_run = AlarmRun(detector_thresholds)
    for values in clean_values:
        clean_run.add(values)
    clean_counts = clean_run.counts()
    log.info("clean windows: %s", clean_counts)

    def score_windows(source: pd.DataFrame, label: str, bug=None, bug_from: int | None = None):
        """Score every window of a frame, optionally injecting a bug part-way.

        A bugged window is validated against the contract *first*. If the contract
        rejects it, nothing downstream ever sees it, so no detector is run on it:
        the schema caught it at the door, which is the whole point of having one.
        """
        run = AlarmRun(detector_thresholds)
        rows = []
        rejected_from = None

        for window_id, index in enumerate(simulated_windows(source, cfg)):
            window = source.loc[index]
            month = int(source.loc[index, cfg.data.time_column].iloc[0])
            bugged = bug is not None and bug_from is not None and window_id >= bug_from
            if bugged:
                window = apply_bug(window, bug)
                verdict = contract_check(window, name=f"{label} w{window_id}")
                if not verdict.passed:
                    rejected_from = window_id if rejected_from is None else rejected_from
                    rows.append(
                        {
                            "window_id": window_id,
                            "month": month,
                            "bugged": True,
                            "level": "contract_rejected",
                            "contract_failures": verdict.failures,
                        }
                    )
                    continue  # never scored, so never monitored

            values = detectors_for_window(window, probabilities(window), context)
            values["window_id"] = window_id
            values["month"] = month
            result = run.add({**thresholded_only(values), "window_id": window_id, "month": month})
            rows.append(
                {**values, "level": result.level, "bugged": bugged, "reasons": list(result.reasons)}
            )

        log.info("  %-22s %s", label, run.counts())
        return run, rows, rejected_from

    # 3. the test months, as they arrived
    test_frame = frame.loc[splits.test_all]
    natural_run, natural_rows, _ = score_windows(test_frame, "months 6-7 (natural)")

    # 4. the variants
    variants = {}
    for variant in cfg.monitor.stress_variants:
        other = load_interim(cfg, str(variant))
        other = other[
            other[cfg.data.time_column].isin(list(cfg.data.protocol.deployment.test_months))
        ]
        run, rows, _ = score_windows(other, f"{variant} (months 6-7)")
        variants[str(variant)] = {
            "counts": run.counts(),
            "first_alert_window": run.first_alert().window_id if run.first_alert() else None,
            "windows": rows,
        }

    # 5. the injected bugs
    bug_month = int(cfg.monitor.bugs.start_month)
    bug_window = int(cfg.monitor.bugs.start_window)
    month_frame = frame[frame[cfg.data.time_column] == bug_month]
    baseline_tpr = rates_at_threshold(
        month_frame[cfg.data.label].to_numpy(),
        probabilities(month_frame),
        float(manifest["threshold"]),
    )["tpr"]

    # How month 6 behaves with no bug at all, so an alarm during a bug run can be
    # attributed rather than assumed. Month 6 drifts on its own (see natural_drift),
    # and counting that as "detection" would flatter every bug.
    baseline_run, _, _ = score_windows(month_frame, "month 6 (no bug)")
    baseline_alert = baseline_run.first_alert()
    already_alerting = baseline_alert is not None and baseline_alert.window_id < bug_window

    # Section 8.5 injects each bug into month 6. Empirically month 6 is already in
    # ALERT before the bug starts -- it has drifted from month 5 on its own -- so a
    # detection delay measured there cannot be attributed to the bug. The clean
    # pool is month 5, quiet by construction, so a delay measured on it means what
    # it says. Both experiments are reported.
    quiet_run, _, _ = score_windows(clean_pool, "clean stream (no bug)")
    quiet_baseline_alert = quiet_run.first_alert()

    bugs = {}
    for spec in cfg.monitor.bugs.specs:
        run, rows, rejected_from = score_windows(
            month_frame, f"bug: {spec.id}", bug=spec, bug_from=bug_window
        )

        # Only an alert at or after the first bugged window can be the bug's doing.
        after = next(
            (r for r in run.results if r.level == ALERT and r.window_id >= bug_window), None
        )
        delay_windows = (after.window_id - bug_window) if after else None
        if rejected_from is not None:
            delay_windows = 0  # stopped at the door, before any application was scored

        bugged_frame = apply_bug(month_frame, spec)
        degraded_tpr = (
            None
            if rejected_from is not None
            else rates_at_threshold(
                bugged_frame[cfg.data.label].to_numpy(),
                probabilities(bugged_frame),
                float(manifest["threshold"]),
            )["tpr"]
        )

        # The attributable version: the same bug, injected into a stream that is
        # quiet without it.
        quiet_bug_run, _, quiet_rejected = score_windows(
            clean_pool, f"bug on clean: {spec.id}", bug=spec, bug_from=bug_window
        )
        quiet_after = next(
            (r for r in quiet_bug_run.results if r.level == ALERT and r.window_id >= bug_window),
            None,
        )
        quiet_delay = (
            0
            if quiet_rejected is not None
            else ((quiet_after.window_id - bug_window) if quiet_after else None)
        )

        bugs[str(spec.id)] = {
            "in_contract": bool(spec.in_contract),
            "rejected_by_contract": rejected_from is not None,
            "on_clean_stream": {
                "baseline_quiet": quiet_baseline_alert is None,
                "detected": quiet_rejected is not None or quiet_after is not None,
                "detected_by": (
                    "contract"
                    if quiet_rejected is not None
                    else ("monitor" if quiet_after else None)
                ),
                "detection_delay_windows": quiet_delay,
                "detection_delay_applications": (
                    quiet_delay * window_size if quiet_delay is not None else None
                ),
                "counts": quiet_bug_run.counts(),
            },
            "first_rejected_window": rejected_from,
            "detected": rejected_from is not None or after is not None,
            "detected_by": (
                "contract" if rejected_from is not None else ("monitor" if after else None)
            ),
            "first_alert_window": after.window_id if after else None,
            "detection_delay_windows": delay_windows,
            "detection_delay_applications": (
                delay_windows * window_size if delay_windows is not None else None
            ),
            "already_alerting_before_bug": already_alerting,
            "tpr_before": baseline_tpr,
            "tpr_after": degraded_tpr,
            "tpr_change": (degraded_tpr - baseline_tpr) if degraded_tpr is not None else None,
            "counts": run.counts(),
            "windows": rows,
        }
        log.info(
            "  %-18s month6: by=%-8s delay=%s | clean stream: by=%-8s delay=%s | TPR %s",
            spec.id,
            bugs[str(spec.id)]["detected_by"],
            delay_windows,
            bugs[str(spec.id)]["on_clean_stream"]["detected_by"],
            quiet_delay,
            "n/a (never scored)"
            if degraded_tpr is None
            else f"{baseline_tpr:.4f} -> {degraded_tpr:.4f}",
        )

    # The fallback the API reads, driven by the worst thing we saw.
    state = None
    if natural_run.first_alert() is not None:
        alert = natural_run.first_alert()
        state = activate(
            f"natural drift: {'; '.join(alert.reasons)}",
            Path(cfg.monitor.fallback.state_file),
            Path(cfg.monitor.fallback.event_log),
            {
                "alpha_fraud": float(cfg.policy.fallback.alpha_fraud),
                "alpha_legit": float(cfg.policy.fallback.alpha_legit),
            },
            window_id=alert.window_id,
            month=alert.month,
        )

    payload = {
        "model_version": manifest["model_version"],
        "window_size": window_size,
        "thresholds": detector_thresholds,
        "threshold_calibration": {
            "clean_windows": n_clean,
            "percentile": percentile,
            "source": "cal_prob + cal_tune",
            "counts": clean_counts,
            "false_alarm_rate": clean_counts["watch"] + clean_counts["alert"],
        },
        "conformal_reference_rates": {
            "excluded": context.conformal_rates[0],
            "fraud": context.conformal_rates[1],
        },
        "natural_drift": {
            "counts": natural_run.counts(),
            "first_alert_window": (
                natural_run.first_alert().window_id if natural_run.first_alert() else None
            ),
            "note": "months 6-7 as published: real population movement, not false alarms",
            "windows": natural_rows,
        },
        "variants": variants,
        "bugs": bugs,
        "fallback_state": state,
        "seconds": time.perf_counter() - started,
    }
    update_section(
        Path(cfg.paths.reports) / "monitoring.json", "monitoring", payload, context_stamp
    )
    update_section(
        Path(cfg.paths.metrics),
        "monitoring",
        {k: v for k, v in payload.items() if k not in ("natural_drift", "variants", "bugs")}
        | {
            "natural_drift": {k: v for k, v in payload["natural_drift"].items() if k != "windows"},
            "variants": {
                k: {n: m for n, m in v.items() if n != "windows"} for k, v in variants.items()
            },
            "bugs": {k: {n: m for n, m in v.items() if n != "windows"} for k, v in bugs.items()},
        },
        context_stamp,
    )
    log.info(
        "wrote reports/monitoring.json and metrics.json -> monitoring (%.1fs)", payload["seconds"]
    )


if __name__ == "__main__":
    main()
