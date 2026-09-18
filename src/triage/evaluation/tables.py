"""Markdown tables built from ``reports/metrics.json``, and nothing else (rule 6).

Each renderer takes the artefact and returns markdown. If a section is missing,
the table says so instead of inventing a row -- a missing number must look missing.
"""

from __future__ import annotations

from typing import Any

from triage.evaluation.artefacts import format_interval, format_number

MISSING = "_Not generated yet._"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [_row(header), _row(["---"] * len(header))]
    lines += [_row(cells) for cells in rows]
    return "\n".join(lines)


def baselines_table(metrics: dict[str, Any]) -> str:
    """Detection for B0 and B1, per test month, under both protocols."""
    section = metrics.get("baselines")
    if not section:
        return MISSING

    target = format_number(section.get("target_fpr", 0.05), 2)
    rows = []
    for label, model in sorted(section["models"].items()):
        for protocol in ("paper", "deployment"):
            run = model.get(protocol)
            if not run:
                continue
            for month, slice_ in sorted(run["months"].items()):
                discrimination = slice_["discrimination"]
                realised = slice_["realised"]
                rows.append(
                    [
                        f"`{label}`",
                        protocol,
                        month,
                        format_number(discrimination["roc_auc"]),
                        format_number(discrimination["pr_auc"]),
                        format_interval(discrimination.get("tpr_ci")),
                        format_number(realised["tpr"]),
                        format_number(realised["fpr"]),
                    ]
                )

    header = [
        "Model",
        "Protocol",
        "Month",
        "ROC-AUC",
        "PR-AUC",
        f"TPR@{target} FPR (95% CI)",
        "Realised TPR",
        "Realised FPR",
    ]
    level_pct = format_number(section["bootstrap"]["level"] * 100, 2)
    note = (
        "\n\nThe paper protocol sets its threshold **on the test set**, as the BAF paper "
        "does; it is optimistic by construction and exists only for comparison with "
        "published results. The deployment protocol sets its threshold on `cal_tune`, "
        "which is part of month 5, and never touches months 6 or 7.\n\n"
        f"Training data: {section['data']['rows']:,} applications, "
        f"{format_number(section['data']['prevalence'] * 100)}% fraud. "
        f"Confidence intervals: {section['bootstrap']['n_resamples']:,} stratified "
        f"bootstrap resamples at the {level_pct}% level."
    )
    return _table(header, rows) + note


def fairness_table(metrics: dict[str, Any]) -> str:
    """False-alarm rates by age group, at each model's deployment threshold."""
    section = metrics.get("baselines")
    if not section:
        return MISSING

    rows = []
    for label, model in sorted(section["models"].items()):
        run = model.get("deployment")
        if not run:
            continue
        for month, slice_ in sorted(run["months"].items()):
            fairness = slice_["fairness"]
            groups = {record["group"]: record for record in fairness["by_group"]}
            older = groups.get("age>=50", {})
            younger = groups.get("age<50", {})
            rows.append(
                [
                    f"`{label}`",
                    month,
                    format_number(younger.get("fpr")),
                    format_number(older.get("fpr")),
                    format_interval(fairness.get("fpr_ratio_ci")),
                ]
            )

    header = ["Model", "Month", "FPR, under 50", "FPR, 50 and over", "FPR ratio (95% CI)"]
    note = (
        "\n\nPredictive equality: `min(FPR) / max(FPR)` across the two age groups, "
        "computed on genuine applicants only, at the threshold chosen on `cal_tune`. "
        "1.0 is parity; lower is a wider gap. Age is used here to **measure** the "
        "model, never to decide anything.\n\n"
        "Both baselines see `customer_age` as a feature. The champion does not, and "
        "the mitigation experiments M1-M4 test whether that is enough."
    )
    return _table(header, rows) + note


def age_band_table(metrics: dict[str, Any], model: str = "b1_lgbm", month: str = "6") -> str:
    """False-alarm rate by 10-year age band, for one model and one month."""
    section = metrics.get("baselines")
    if not section or model not in section.get("models", {}):
        return MISSING

    run = section["models"][model].get("deployment")
    if not run or month not in run["months"]:
        return MISSING

    rows = []
    for record in run["months"][month]["fairness"]["by_band"]:
        genuine = int(record["n_genuine"])
        # A band this small cannot support a claim; say so rather than print a rate.
        reliable = genuine >= 1000
        rows.append(
            [
                record["band"],
                f"{genuine:,}",
                f"{int(record['false_alarms']):,}",
                format_number(record["fpr"]) if reliable else f"({format_number(record['fpr'])})",
                "" if reliable else "too few to read",
            ]
        )

    header = ["Age band", "Genuine applicants", "False alarms", "FPR", "Note"]
    note = (
        f"\n\n`{model}`, month {month}, at the threshold chosen on `cal_tune`. The rate "
        "climbs steadily with age. A bracketed rate comes from fewer than 1,000 genuine "
        "applicants and is noise, not a finding: with a handful of people in a band, the "
        "confidence interval covers almost the whole range.\n\n"
        "This is the gap the mitigation experiments (M1-M4) have to close, and the reason "
        "the champion never sees age."
    )
    return _table(header, rows) + note


def calibration_table(metrics: dict[str, Any]) -> str:
    """Which calibration method won, and what it did to the probabilities."""
    section = metrics.get("champion")
    if not section:
        return MISSING

    calibration = section["calibration"]
    rows = []
    for method, values in sorted(calibration["comparison"].items()):
        chosen = " **chosen**" if method == calibration["chosen"] else ""
        rows.append(
            [
                f"`{method}`{chosen}",
                format_number(values["brier"], 4),
                format_number(values["ece_equal_mass"], 3),
                format_number(values["mean_predicted_rate"] * 100) + "%",
                format_number(values["observed_rate"] * 100) + "%",
            ]
        )

    header = ["Method", "Brier", "ECE (15 equal-mass bins)", "Mean predicted rate", "Observed rate"]
    note = (
        "\n\nFitted on `cal_prob` and chosen on `cal_tune`, two disjoint thirds of month 5, "
        f"by {calibration['select_by']} score. The last two columns are "
        "calibration-in-the-large: an uncalibrated model can rank well and still be badly "
        "wrong about *how likely* fraud is, which matters here because the decision policy "
        "reads probabilities, not ranks."
    )
    return _table(header, rows) + note


def policy_table(metrics: dict[str, Any]) -> str:
    """The trade-off curve, and whether the two guarantees held on the test months."""
    section = metrics.get("policy")
    if not section:
        return MISSING

    chosen = section["chosen"]
    target_coverage = 1 - chosen["alpha_fraud"]
    target_exclusion = chosen["alpha_legit"]

    rows = []
    for month, result in sorted(section["months"].items()):
        rows.append(
            [
                month,
                format_number(result["fraud_coverage"]),
                format_number(result["genuine_exclusion_rate"]),
                format_number(result["band_approve"]),
                format_number(result["band_review"]),
                format_number(result["band_verify"]),
            ]
        )

    header = [
        "Month",
        "Fraud caught",
        "Genuine sent to verify",
        "Approve",
        "Review",
        "Verify",
    ]
    note = (
        f"\n\nPolicy: `alpha_fraud = {chosen['alpha_fraud']:g}`, "
        f"`alpha_legit = {chosen['alpha_legit']:g}`, chosen on `cal_tune` as the best "
        "fraud coverage the review team could absorb, then turned into thresholds on "
        "`cal_conf` — a third of month 5 that nothing else touches.\n\n"
        f"The promise is: catch at least **{target_coverage:.0%}** of fraud, and send at "
        f"most **{target_exclusion:.0%}** of genuine applicants for extra verification. "
        "The first half held on both test months. The second did not — see *What didn't "
        "work*. The capacity figures behind the choice are the owner's assumptions, not "
        "facts about the data."
    )
    return _table(header, rows) + note


def tradeoff_table(metrics: dict[str, Any]) -> str:
    """What each level of fraud coverage costs in review workload."""
    section = metrics.get("policy")
    if not section:
        return MISSING

    alpha_legit = section["chosen"]["alpha_legit"]
    rows = []
    for row in section["grid"]:
        if row["alpha_legit"] != alpha_legit:
            continue
        chosen = " **chosen**" if row["alpha_fraud"] == section["chosen"]["alpha_fraud"] else ""
        rows.append(
            [
                f"{row['fraud_coverage']:.0%}{chosen}",
                format_number(row["review_share"]),
                format_number(row["verify_share"]),
                f"{row['review_share'] * 100_000:,.0f}",
            ]
        )
    rows.reverse()  # most coverage first: the interesting end of the curve

    header = [
        "Fraud caught",
        "Review share",
        "Verify share",
        "Reviews per 100,000 applications",
    ]
    note = (
        f"\n\nMeasured on `cal_tune`, at `alpha_legit = {alpha_legit:g}`. This is the whole "
        "curve, not just the point that was chosen, because the shape is the argument: "
        "the last few points of fraud coverage cost far more review capacity than the "
        "first. A 90% catch rate is not a modelling problem, it is a staffing one."
    )
    return _table(header, rows) + note


def coverage_by_age_table(metrics: dict[str, Any], month: str = "6") -> str:
    """Who carries the cost of the policy, by age band."""
    section = metrics.get("policy")
    if not section or month not in section.get("months", {}):
        return MISSING

    rows = []
    for record in section["months"][month]["by_age_band"]:
        fraud = int(record["n_fraud"])
        reliable = fraud >= 20
        coverage = format_number(record["fraud_coverage"])
        rows.append(
            [
                record["band"],
                f"{int(record['n']):,}",
                f"{fraud:,}",
                coverage if reliable else f"({coverage})",
                format_number(record["review_share"]),
                format_number(record["verify_share"]),
            ]
        )

    header = [
        "Age band",
        "Applications",
        "Frauds",
        "Fraud caught",
        "Sent to review",
        "Sent to verify",
    ]
    note = (
        f"\n\nMonth {month}, under one age-blind pair of thresholds: the policy does not know "
        "anyone's age. A bracketed figure rests on fewer than 20 frauds and is noise.\n\n"
        "Coverage rises with age and so does the workload: an older applicant is far more "
        "likely to be stopped for review or verification. Fixing this with age-specific "
        "thresholds would mean using age at decision time, which this project does not do "
        "(rule 4). It is recorded as a finding for the mitigation experiments instead."
    )
    return _table(header, rows) + note


def data_table(metrics: dict[str, Any]) -> str:
    """What the models were trained and evaluated on."""
    section = metrics.get("baselines")
    if not section:
        return MISSING

    deployment = next(
        (
            model["deployment"]
            for model in section["models"].values()
            if model.get("deployment", {}).get("split_sizes")
        ),
        None,
    )
    if not deployment:
        return MISSING

    sizes = deployment["split_sizes"]
    rows = [
        ["Training", "0-4", f"{sizes['train']:,}", "fits the model"],
        ["`cal_prob`", "5", f"{sizes['cal_prob']:,}", "fits probability calibration"],
        ["`cal_tune`", "5", f"{sizes['cal_tune']:,}", "chooses the alphas and the threshold"],
        ["`cal_conf`", "5", f"{sizes['cal_conf']:,}", "conformal thresholds only, no choices"],
        ["Test", "6", f"{sizes['test_month_6']:,}", "reported on its own"],
        ["Test", "7", f"{sizes['test_month_7']:,}", "reported on its own"],
    ]
    return _table(["Split", "Months", "Applications", "What it is for"], rows)


RENDERERS = {
    "baselines": baselines_table,
    "calibration": calibration_table,
    "policy": policy_table,
    "tradeoff": tradeoff_table,
    "coverage_by_age": coverage_by_age_table,
    "fairness": fairness_table,
    "age_bands": age_band_table,
    "data": data_table,
}


def render_all(metrics: dict[str, Any]) -> dict[str, str]:
    """Every table the README knows about, keyed by its marker name."""
    return {name: renderer(metrics) for name, renderer in RENDERERS.items()}
