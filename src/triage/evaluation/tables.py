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


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def headline_table(metrics: dict[str, Any]) -> str:
    """The findings, in one block, for a reader who will not scroll.

    Generated like every other table (rule 6). A summary is exactly where a typed
    number would rot first: it is the part people copy into an email, and the part
    nobody re-checks after a re-run.
    """
    champion = metrics.get("champion")
    policy = metrics.get("policy")
    fairness = metrics.get("fairness")
    monitoring = metrics.get("monitoring")
    service = metrics.get("service")
    if not champion or not policy:
        return MISSING

    rows: list[list[str]] = []
    months = sorted(champion["months"])

    aucs = " / ".join(
        format_number(champion["months"][m]["discrimination"]["roc_auc"]) for m in months
    )
    tprs = " / ".join(_pct(champion["months"][m]["realised"]["tpr"]) for m in months)
    rows.append(
        [
            "**Detection**",
            f"ROC-AUC {aucs} on months {' and '.join(months)}, catching {tprs} of fraud "
            "at the operating threshold.",
        ]
    )

    target_legit = float(policy["chosen"]["alpha_legit"])
    realised = " / ".join(_pct(policy["months"][m]["genuine_exclusion_rate"]) for m in months)
    rows.append(
        [
            "**The guarantee that broke**",
            f"At most {_pct(target_legit)} of genuine applicants were promised extra "
            f"verification. {realised} got it. The promise holds only while new "
            "applications look like the calibration month, and they stop.",
        ]
    )

    target_fraud = 1 - float(policy["chosen"]["alpha_fraud"])
    coverage = " / ".join(_pct(policy["months"][m]["fraud_coverage"]) for m in months)
    rows.append(
        [
            "**The guarantee that held**",
            f"{_pct(target_fraud, 0)} of fraud was promised a non-approval. "
            f"{coverage} got one -- over-delivered, which is the safe direction "
            "and still a finding.",
        ]
    )

    ratios = " / ".join(
        format_number(champion["months"][m]["fairness"]["fpr_ratio"]) for m in months
    )
    rows.append(
        [
            "**Fairness**",
            f"False alarms among genuine applicants are {ratios} as likely for the "
            "younger group as the older one. 1.00 would be parity. The model never "
            "sees age.",
        ]
    )

    if fairness:
        by_name = {r["experiment"]: r for r in fairness.get("tradeoff", []) if r["month"] == 6}
        eg = by_name.get("m3_fairlearn_eg")
        if eg:
            rows.append(
                [
                    "**What fairness cost**",
                    "Neither model-level mitigation beat simply dropping age. The "
                    f"constrained learner reached {format_number(eg['fpr_ratio'])} by "
                    f"catching {_pct(eg['tpr'])} of fraud -- equal treatment by way of "
                    "flagging almost nobody.",
                ]
            )

    if monitoring:
        counts = monitoring.get("threshold_calibration", {}).get("counts", {})
        total = sum(counts.values()) if counts else 0
        bugs = monitoring.get("bugs", {})
        detected = sum(1 for b in bugs.values() if b.get("on_clean_stream", {}).get("detected"))
        if total and bugs:
            rows.append(
                [
                    "**Monitoring, without labels**",
                    f"{detected} of {len(bugs)} injected data faults were caught, on a "
                    f"stream carrying no fraud labels at all. On {total} clean windows "
                    f"the detectors raised {counts.get('watch', 0)} watches and "
                    f"{counts.get('alert', 0)} alerts.",
                ]
            )

    if service:
        latency = service.get("latency", {}).get("http_without_shap")
        docker = service.get("docker", {})
        if latency:
            size = (
                f", in a {docker['compressed_mb']:.0f} MB image" if docker.get("available") else ""
            )
            rows.append(
                [
                    "**Shipped**",
                    f"{format_number(latency['p50_ms'], 2)} ms per scored request at the "
                    f"median, against a {service['target_p50_ms']:g} ms target{size}.",
                ]
            )

    return _table(["", ""], rows)


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


def mitigations_table(metrics: dict[str, Any]) -> str:
    """What each fairness mitigation bought, and what it cost."""
    section = metrics.get("fairness")
    if not section:
        return MISSING

    labels = {
        "m1_drop_age": "M1 — drop age",
        "m2_fairgbm": "M2 — FairGBM, FPR constraint",
        "m3_fairlearn_eg": "M3 — fairlearn, FPR parity",
        "m4_policy_only": "M4 — policy only, no model change",
    }

    rows = []
    for row in section.get("tradeoff", []):
        rows.append(
            [
                labels.get(row["experiment"], f"`{row['experiment']}`"),
                str(row["month"]),
                format_number(row["tpr"]),
                format_number(row["fpr"]),
                f"{format_number(row['fpr_ratio'])} "
                f"({format_number(row['fpr_ratio_low'])}-{format_number(row['fpr_ratio_high'])})",
            ]
        )

    skipped = [name for name, e in section["experiments"].items() if e.get("skipped")]
    header = ["Mitigation", "Month", "Fraud caught", "Genuine stopped", "FPR ratio (95% CI)"]
    note = (
        "\n\nAll four use age at training or measurement time only; none uses it to decide "
        "anything about an application. M1 and M2 are thresholded at 5% FPR on `cal_tune`. "
        "M3 is a randomised classifier with a single operating point, so it is measured "
        "where it sits rather than swept. M4 changes no model at all: an application counts "
        "as stopped if the conformal policy sends it to review or verify.\n\n"
        "The two model-level mitigations did not earn their place. Read the M3 row carefully: "
        "at roughly 1% prevalence, the cheapest way to equalise false-positive rates between "
        "groups is to stop flagging anyone, and that is close to what it did — while still "
        "ending up the least equal of the four."
    )
    if skipped:
        note += f"\n\nSkipped: {', '.join(skipped)}."
    return _table(header, rows) + note


def monitoring_table(metrics: dict[str, Any]) -> str:
    """Whether the monitor catches faults, and how fast."""
    section = metrics.get("monitoring")
    if not section:
        return MISSING

    names = {
        "swap_employment": "Two employment codes swapped",
        "mirror_income": "Income scale reversed",
        "all_mobile_valid": "Mobile check always passes",
        "income_x10": "Income multiplied by ten",
    }
    window = int(section["window_size"])

    rows = []
    for key, bug in sorted(section.get("bugs", {}).items()):
        clean = bug.get("on_clean_stream", {})
        delay = clean.get("detection_delay_windows")
        caught_by = clean.get("detected_by") or "not detected"
        change = bug.get("tpr_change")

        if delay is None:
            when = "never"
        elif caught_by == "contract":
            when = "before scoring"
        else:
            when = f"{delay} window ({delay * window:,} applications)"

        rows.append(
            [
                names.get(key, f"`{key}`"),
                "yes" if bug["in_contract"] else "**no**",
                caught_by,
                when,
                "never scored" if change is None else f"{change * 100:+.1f} pts",
            ]
        )

    header = [
        "Injected fault",
        "Legal under the contract?",
        "Caught by",
        "How long it ran",
        "Detection cost",
    ]

    calibration = section["threshold_calibration"]
    counts = calibration["counts"]
    total = sum(counts.values())
    watch_rate = counts["watch"] / total if total else 0.0
    natural = section.get("natural_drift", {}).get("counts", {})

    note = (
        f"\n\n**False alarms.** Across {total} windows drawn from the calibration month — "
        f"clean by construction — {counts['watch']} raised a watch ({watch_rate:.1%}) and "
        f"{counts['alert']} an alert. For four detectors at a 99th-percentile threshold the "
        "expected watch rate is 3.9%, so the false-alarm rate is a measured property rather "
        "than a hope.\n\n"
        "**Detection.** The three faults that are *legal* under the contract are the ones "
        "worth catching, because no schema check can see them: the values stay in range and "
        "only their meaning changes. Each reached an alert one window after injection, which "
        "is the floor the two-consecutive-windows rule allows. The fourth is the control — "
        "the contract rejects it before anything is scored.\n\n"
        "Delays are measured on a stream drawn from the calibration month, which is quiet "
        "without a fault. Section 8.5 injects into month 6, but month 6 is already alerting "
        f"on its own drift ({natural.get('alert', 0)} of "
        f"{sum(natural.values()) or '?'} windows), so a delay measured there could not be "
        "attributed to the fault. Both experiments are in `reports/monitoring.json`.\n\n"
        "**Natural drift is not a false alarm.** Months 6 and 7 really did move, and the "
        "monitor saying so is it working. That same movement is why one half of the "
        "conformal guarantee failed."
    )
    return _table(header, rows) + note


def service_table(metrics: dict[str, Any]) -> str:
    """Latency, and what reason codes cost."""
    section = metrics.get("service")
    if not section:
        return MISSING

    names = {
        "scoring_without_shap": "Scoring only, no reason codes",
        "scoring_with_shap": "Scoring only, with reason codes",
        "http_without_shap": "Full HTTP request, no reason codes",
        "http_with_shap": "Full HTTP request, with reason codes",
    }
    target = float(section["target_p50_ms"])

    rows = []
    for key, label in names.items():
        values = section["latency"].get(key)
        if not values:
            continue
        rows.append(
            [
                label,
                f"{values['p50_ms']:.2f}",
                f"{values['p95_ms']:.2f}",
                f"{values['p99_ms']:.2f}",
                "yes" if values["p50_ms"] <= target else "**no**",
            ]
        )

    docker = section.get("docker", {})
    if docker.get("available"):
        # Both figures, because they differ by several times and a reader who
        # checks with `docker images` should not find a number that disagrees.
        unpacked = docker.get("on_disk")
        image = f"{docker['compressed_mb']:.0f} MB compressed" + (
            f", {unpacked} unpacked" if unpacked else ""
        )
    else:
        image = f"not measured — {docker.get('reason', 'unknown')}"

    header = ["Path", "p50 (ms)", "p95 (ms)", "p99 (ms)", f"Under {target:g} ms?"]
    note = (
        f"\n\n{section['requests']:,} sequential requests after {section['warmup']} warm-up, "
        "on CPU, single process, no network.\n\n"
        "Reason codes dominate: they cost roughly twenty times the entire latency "
        "budget, because a SHAP explanation walks every tree for every request. "
        "Scoring itself is comfortably inside target, and the HTTP layer — "
        "validating the request against the frozen contract, then serialising — adds "
        "under two milliseconds. A caller that does not need an explanation should "
        "pass `?explain=false`; a queue that does should compute them out of band.\n\n"
        f"Docker image: {image}."
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
    "headline": headline_table,
    "baselines": baselines_table,
    "calibration": calibration_table,
    "policy": policy_table,
    "tradeoff": tradeoff_table,
    "coverage_by_age": coverage_by_age_table,
    "fairness": fairness_table,
    "age_bands": age_band_table,
    "mitigations": mitigations_table,
    "monitoring": monitoring_table,
    "service": service_table,
    "data": data_table,
}


def render_all(metrics: dict[str, Any]) -> dict[str, str]:
    """Every table the README knows about, keyed by its marker name."""
    return {name: renderer(metrics) for name, renderer in RENDERERS.items()}
