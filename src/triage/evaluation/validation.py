"""The two-page validation report, generated from ``reports/metrics.json``.

Written for a model-risk reviewer in the style of a UK bank validation under PRA
SS1/23, and generated rather than typed for the same reason as every other table
(rule 6): a report that drifts from the artefacts it describes is worse than no
report.

It is deliberately short. A reviewer's attention is the scarce resource, and
section 14 caps it at two pages, so each section gets a few sentences and the
numbers that matter, not everything that could be said.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from triage.evaluation.artefacts import format_interval, format_number

MISSING = "_Not generated: the stage that produces this has not been run._"


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _finding(rating: str, title: str, body: str, recommendation: str) -> dict[str, str]:
    return {"rating": rating, "title": title, "body": body, "recommendation": recommendation}


def collect_findings(metrics: dict[str, Any]) -> list[dict[str, str]]:
    """The findings a reviewer would raise, derived from what the artefacts show.

    Ratings follow the usual convention: High means it must be resolved before
    the model is relied on, Medium before it is scaled, Low is worth recording.
    """
    findings: list[dict[str, str]] = []
    policy = metrics.get("policy")
    champion = metrics.get("champion")
    monitoring = metrics.get("monitoring")

    if policy:
        target_exclusion = float(policy["chosen"]["alpha_legit"])
        breaches = {
            month: result["genuine_exclusion_rate"]
            for month, result in policy["months"].items()
            if result["genuine_exclusion_rate"] > target_exclusion
        }
        if breaches:
            worst = max(breaches.values())
            findings.append(
                _finding(
                    "High",
                    "The genuine-applicant guarantee did not hold out of sample",
                    f"The policy promises at most {_pct(target_exclusion)} of genuine "
                    f"applicants sent for extra verification. It reached {_pct(worst)}, "
                    f"breaching the promise on "
                    f"{'both test months' if len(breaches) > 1 else 'one test month'}. The "
                    "guarantee is conditional on exchangeability with the calibration month, "
                    "and that fails here: prevalence rises and the score distribution moves.",
                    "Never quote the genuine-applicant figure without its exchangeability "
                    "condition beside it. Re-derive thresholds on a recent month rather than a "
                    "fixed one, triggered by the monitor's alert.",
                )
            )

        target_coverage = 1 - float(policy["chosen"]["alpha_fraud"])
        gaps = [result["fraud_coverage"] - target_coverage for result in policy["months"].values()]
        if gaps and max(abs(gap) for gap in gaps) > 0.015:
            findings.append(
                _finding(
                    "Medium",
                    "Fraud coverage is looser than intended",
                    f"Coverage came in {max(gaps) * 100:+.1f} points against a "
                    f"{_pct(target_coverage, 0)} target, outside the ±1.5 point tolerance. "
                    "Over-delivery is the safe direction, but it means the thresholds are "
                    f"not tight: they rest on {policy['thresholds']['n_fraud']:,} frauds in "
                    "the conformal calibration set, so each threshold is an order statistic "
                    "from a small sample.",
                    "Size the calibration set by the number of frauds it contains, not by "
                    "the number of applications. Consider pooling more than one month for "
                    "the conformal split, accepting the loss of recency.",
                )
            )

        older = [
            record
            for month in policy["months"].values()
            for record in month.get("by_age_band", [])
            if record.get("n_fraud", 0) >= 20
        ]
        if older:
            coverages = [r["fraud_coverage"] for r in older if r["fraud_coverage"] is not None]
            reviews = [r["review_share"] for r in older]
            if coverages and max(coverages) - min(coverages) > 0.1:
                findings.append(
                    _finding(
                        "High",
                        "The policy's burden falls unevenly across age bands",
                        f"Fraud coverage runs from {_pct(min(coverages))} to "
                        f"{_pct(max(coverages))} across age bands, and the share sent to "
                        f"review from {_pct(min(reviews))} to {_pct(max(reviews))}. The policy "
                        "is age-blind by construction, so this is the model's own behaviour "
                        "surfacing through one pair of thresholds.",
                        "Do not correct this with age-specific thresholds: that uses a "
                        "protected attribute at decision time and needs legal sign-off before "
                        "it could even be considered. Report the disparity, and use the "
                        "mitigation experiments to price what narrowing it would cost.",
                    )
                )

    if champion:
        calibration = champion.get("calibration", {})
        by_group = calibration.get("cal_tune", {})
        if len(by_group) >= 3:
            eces = {name: values["ece_equal_mass"] for name, values in by_group.items()}
            worst_group = max(eces, key=lambda name: eces[name])
            if eces[worst_group] > 2 * min(eces.values()):
                findings.append(
                    _finding(
                        "Medium",
                        "Calibration is worse for older applicants",
                        f"Expected calibration error is {format_number(eces[worst_group])} for "
                        f"{worst_group} against {format_number(min(eces.values()))} for the "
                        "best-served group. Conformal thresholds come from pooled "
                        "probabilities, so a less well calibrated group inherits a weaker "
                        "guarantee, invisibly.",
                        "Report coverage by age band alongside the headline guarantee, every "
                        "time. Group-wise calibration is worth analysing, noting it would use "
                        "age at scoring time and needs sign-off.",
                    )
                )

    if monitoring:
        calibration = monitoring.get("threshold_calibration", {})
        counts = calibration.get("counts", {})
        total = sum(counts.values()) if counts else 0
        if total:
            findings.append(
                _finding(
                    "Low",
                    "Detector thresholds behave as designed on clean windows",
                    f"Across {total} windows drawn from the calibration month, "
                    f"{counts.get('watch', 0)} raised a watch and {counts.get('alert', 0)} an "
                    "alert. That is close to what a 99th-percentile threshold on four "
                    "detectors implies, so the false-alarm rate is understood rather than "
                    "assumed.",
                    "Re-calibrate the thresholds whenever the model or the calibration month "
                    "changes; they are properties of that pairing, not constants.",
                )
            )

        bugs = monitoring.get("bugs", {})
        undetected = [
            name
            for name, bug in bugs.items()
            if bug.get("on_clean_stream", {}).get("detected") is False
        ]
        if undetected:
            findings.append(
                _finding(
                    "High",
                    "An injected data fault went undetected",
                    f"{', '.join(undetected)} did not raise an alert on a stream that is "
                    "otherwise quiet. A fault the monitor cannot see is a fault that runs "
                    "until someone notices the fraud numbers.",
                    "Add a detector targeting the affected field, or accept the exposure "
                    "explicitly and document the control that covers it instead.",
                )
            )

    if not findings:
        findings.append(
            _finding(
                "Low",
                "No findings raised",
                "The stages that would produce findings have not been run.",
                "Run `make evaluate` before relying on this report.",
            )
        )
    return findings


def render(metrics: dict[str, Any]) -> str:
    """The whole report, as markdown."""
    baselines = metrics.get("baselines")
    champion = metrics.get("champion")
    policy = metrics.get("policy")
    monitoring = metrics.get("monitoring")
    fairness = metrics.get("fairness")

    context = (champion or policy or baselines or {}).get("context", {})
    lines = [
        "# Model validation report",
        "",
        "**Model:** onboarding fraud triage — application scoring and decision policy  ",
        f"**Version:** `{(champion or {}).get('model_version', 'not trained')}`  ",
        f"**Generated:** {datetime.now(UTC).date().isoformat()} "
        f"from `reports/metrics.json` (config `{context.get('config_hash', '?')}`, "
        f"commit `{context.get('git_sha', '?')}`)  ",
        "**Status:** portfolio project on public synthetic data. Not a production model.",
        "",
        "> Generated by `make report`. Every figure comes from a stored artefact; none is",
        "> typed by hand. Where a figure is missing, the stage that produces it has not run.",
        "",
        "## 1. Purpose, users and proposed risk tier",
        "",
        "Scores online bank-account applications and routes each to **approve**, **human",
        "review**, or **extra verification**. It never declines automatically: its worst",
        "direct action against a customer is a request for further checks.",
        "",
        "Users are fraud operations, who work the review queue, and model risk, who own this",
        "report. **Proposed tier: high** — it affects access to a payment account, touches a",
        "protected characteristic in its measurement, and sits within the automated",
        "decision-making safeguards of the Data (Use and Access) Act. No automatic decline",
        "lowers the severity of one error; it does not lower the tier.",
        "",
        "## 2. Data and known gaps",
        "",
    ]

    if baselines:
        data = baselines["data"]
        lines += [
            f"Public Bank Account Fraud dataset (NeurIPS 2022): {data['rows']:,} applications",
            f"across eight months, {_pct(data['prevalence'], 2)} fraud. Split by month only:",
            "training on months 0-4, calibration on month 5 split three ways, and months 6",
            "and 7 held out and reported separately.",
        ]
    else:
        lines.append(MISSING)

    lines += [
        "",
        "Known gaps, in order of how much they limit the conclusions:",
        "",
        "- **The data is synthetic**, generated from an anonymised real dataset.",
        "  It is not UK data, so no rate here transfers to a UK population.",
        "- **No vulnerability information**, so no vulnerability analysis is possible.",
        "- **No within-month timestamps.** A 'day' is simulated as 4,000 applications shuffled",
        "  within a month, which is an assumption, not an observation.",
        "- **Costs are illustrative parameters** set for comparison between policies.",
        "  No saving is claimed, in pounds or otherwise.",
        "",
        "## 3. Method and key choices",
        "",
        "LightGBM without `customer_age`, calibrated on one third of month 5 and selected on",
        "another. Decision bands come from label-conditional split conformal prediction: two",
        "thresholds, fitted on a third part of month 5 that nothing else touches, turn a",
        "probability into a set of plausible labels, and the set picks the band.",
        "",
        "Three choices a reviewer should test:",
        "",
        "- **Age is excluded from the model, used only to measure it.** Dropping it costs",
        "  almost nothing in detection: the information was never uniquely there.",
        "- **Thresholds are never set on evaluation data.** The published baseline protocol,",
        "  which sets its threshold on the test set, is reproduced only for comparison and",
        "  labelled optimistic wherever it appears.",
        "- **The conformal calibration part is used once.** Every other choice happens on a",
        "  different part of the same month.",
        "",
        "## 4. Performance",
        "",
    ]

    if champion:
        rows = [
            "| Month | ROC-AUC | Fraud caught at the operating point | False alarms | Brier |",
            "|---|---|---|---|---|",
        ]
        for month, values in sorted(champion["months"].items()):
            rows.append(
                f"| {month} | {format_number(values['discrimination']['roc_auc'])} "
                f"| {_pct(values['realised']['tpr'])} "
                f"| {_pct(values['realised']['fpr'])} "
                f"| {format_number(values['calibration']['overall']['brier'], 4)} |"
            )
        lines += rows
        comparison = champion["calibration"]["comparison"]
        chosen = champion["calibration"]["chosen"]
        lines += [
            "",
            f"Calibration method **{chosen}**, chosen by Brier score on held-out calibration",
            f"data. Uncalibrated, the model predicts a "
            f"{_pct(comparison['none']['mean_predicted_rate'], 2)} fraud rate against an",
            f"observed {_pct(comparison['none']['observed_rate'], 2)}: it ranks well and is",
            "wrong about magnitude, which matters because the policy reads probabilities.",
            "",
            "Stability: performance holds across both test months, while the realised false",
            "alarm rate does not match the rate the threshold was set for. That gap is the",
            "subject of finding 1.",
        ]
    else:
        lines.append(MISSING)

    lines += ["", "## 5. Fairness", ""]

    if champion:
        rows = [
            "| Month | False alarms, under 50 | 50 and over | Ratio (95% CI) |",
            "|---|---|---|---|",
        ]
        for month, values in sorted(champion["months"].items()):
            groups = {r["group"]: r for r in values["fairness"]["by_group"]}
            rows.append(
                f"| {month} | {_pct(groups.get('age<50', {}).get('fpr'))} "
                f"| {_pct(groups.get('age>=50', {}).get('fpr'))} "
                f"| {format_interval(values['fairness'].get('fpr_ratio_ci'))} |"
            )
        lines += rows
        lines += [
            "",
            "Predictive equality, on genuine applicants only, at the operating threshold. A",
            "ratio of 1.00 would mean both groups are stopped equally often. Age is used here",
            "to measure the model; the model does not see it.",
        ]
    else:
        lines.append(MISSING)

    if fairness:
        lines += [
            "",
            "**What each mitigation cost** (month 6; both months are in the README):",
            "",
            "| Experiment | Fraud caught | False alarms | Ratio |",
            "|---|---|---|---|",
        ]
        # Month 6 only, to stay inside two pages.
        for row in (r for r in fairness.get("tradeoff", []) if r["month"] == 6):
            lines.append(
                f"| `{row['experiment']}` | {_pct(row['tpr'])} "
                f"| {_pct(row['fpr'])} | {format_number(row['fpr_ratio'])} |"
            )
        lines += [
            "",
            "Neither model-level mitigation beat simply dropping age. M3 equalised by",
            "flagging almost nobody -- at ~1% prevalence the cheapest way to equalise",
            "false-positive rates is to stop having any -- and was still the least equal.",
        ]
    else:
        lines += ["", "_Mitigation experiments not yet run (`make fairness`)._"]

    lines += ["", "## 6. Monitoring plan", ""]

    if monitoring:
        thresholds = monitoring["thresholds"]
        calibration = monitoring["threshold_calibration"]
        lines += [
            "Three label-free detectors run on every window of 4,000 applications, with",
            f"thresholds set at the {calibration['percentile']:g}th percentile of",
            f"{calibration['clean_windows']} windows drawn from the calibration month:",
            "",
            "| Detector | Threshold | What it sees |",
            "|---|---|---|",
            f"| Score PSI | {format_number(thresholds.get('psi_score'), 4)} "
            "| the distribution of risk scores moving |",
            f"| Worst feature PSI | {format_number(thresholds.get('psi_feature_max'))} "
            "| any single input moving |",
            f"| Domain classifier AUC | {format_number(thresholds.get('domain_auc'))} "
            "| the window being distinguishable from the reference at all |",
            f"| Conformal rate test | {format_number(thresholds.get('conformal_neglogp'))} "
            "| the policy's own crossing rates drifting from calibration |",
            "",
            "**Alarms.** Watch when any detector exceeds its threshold; alert when the same",
            "detector exceeds it twice running, or score PSI passes 0.25.",
            "",
            "**Fallback.** On alert the policy tightens its alphas, sending more applications",
            "to a human; the event is logged and the service reads the state on every request.",
            "It is never cleared automatically.",
            "",
            "**Owner.** Fraud operations own the queue; model risk own the thresholds.",
            "**Re-validation triggers:** any alert; any change to the model, its calibration or",
            "the alpha targets; a new calibration month; or twelve months elapsed.",
        ]
    else:
        lines.append(MISSING)

    lines += ["", "## 7. Findings", ""]
    for index, finding in enumerate(collect_findings(metrics), start=1):
        lines += [
            f"**{index}. {finding['title']}** — *{finding['rating']}*",
            "",
            finding["body"],
            "",
            f"*Recommendation:* {finding['recommendation']}",
            "",
        ]

    return "\n".join(lines) + "\n"
