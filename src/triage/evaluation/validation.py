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

import re
from datetime import UTC, datetime
from typing import Any

from triage.evaluation.artefacts import format_interval, format_number

MISSING = "_Not generated: the stage that produces this has not been run._"

# For the page estimate below: A4, 11pt, single spaced, 2.5cm margins. That wraps
# body text at roughly 95 characters and fits about 52 lines to a page.
CHARS_PER_LINE = 95
LINES_PER_PAGE = 52


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def estimated_pages(markdown: str) -> float:
    """Roughly how many pages this renders to, so the section 14 cap can be tested.

    A word count is the wrong measure: a table row costs one line whether it holds
    three words or twelve, and a heading costs its own line plus the space around
    it. This counts lines instead -- prose paragraphs re-wrapped, tables and
    headings as they are -- which is what actually fills a page.

    It is an estimate and it is meant to be. The alternative is rendering a PDF in
    a test, which would make the cap depend on a toolchain nobody else has.
    """
    total = 0
    for block in re.split(r"\n\s*\n", markdown):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        if lines[0].startswith("#"):
            total += len(lines) + 1  # a heading, and the space it sits in
        elif lines[0].strip().startswith("|"):
            total += len(lines) + 1  # one line per row, whatever the row holds
        else:
            # A paragraph re-wrapped, plus the blank line that ends it. Each
            # paragraph ends on a partial line, which is why they are counted
            # one at a time rather than in bulk.
            text = " ".join(" ".join(lines).split())
            total += -(-len(text) // CHARS_PER_LINE) + 1
    return total / LINES_PER_PAGE


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
                    f"At most {_pct(target_exclusion)} of genuine applicants should be "
                    f"verified; {_pct(worst)} were, on "
                    f"{'both test months' if len(breaches) > 1 else 'one test month'}. "
                    "Exchangeability with the calibration month fails: prevalence rises, "
                    "scores move.",
                    "re-derive thresholds on a recent month, on the alert; never quote the "
                    "figure without its condition.",
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
                    "Over-delivery is safe, but the thresholds are loose: they rest on "
                    f"{policy['thresholds']['n_fraud']:,} frauds, so each is an order "
                    "statistic from a small sample.",
                    "size the calibration set by the frauds in it, not the applications.",
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
                        "Burden falls unevenly across age bands",
                        f"Fraud coverage runs {_pct(min(coverages))} to "
                        f"{_pct(max(coverages))} across age bands, the review share "
                        f"{_pct(min(reviews))} to {_pct(max(reviews))}. The policy is "
                        "age-blind, so this is the model showing through one pair of "
                        "thresholds.",
                        "do not correct this with age-specific thresholds -- that needs "
                        "legal sign-off. Report it; price the alternatives with the "
                        "mitigations.",
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
                        "best-served group. Thresholds come from pooled probabilities, so a "
                        "less well calibrated group inherits a weaker guarantee, invisibly.",
                        "report coverage by age band beside the headline guarantee, every time.",
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
                    f"Across {total} windows from the calibration month, "
                    f"{counts.get('watch', 0)} raised a watch and {counts.get('alert', 0)} an "
                    "alert -- close to what a 99th-percentile threshold on four detectors "
                    "implies, so the false-alarm rate is understood, not assumed.",
                    "re-calibrate them whenever the model or the calibration month changes.",
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
                    f"{', '.join(undetected)} did not raise an alert on an otherwise quiet "
                    "stream. A fault the monitor cannot see runs until someone notices the "
                    "fraud numbers.",
                    "add a detector for the affected field, or accept the exposure and "
                    "document the control that covers it.",
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
        "**Status:** portfolio project on public synthetic data, not a production model.",
        "",
        "## 1. Purpose, users and proposed risk tier",
        "",
        "Scores online bank-account applications and routes each to **approve**, **human",
        "review** or **extra verification**; the worst it can do to a customer is ask for",
        "checks. Users are fraud operations and model risk. **Proposed tier: high** -- it",
        "gates access to a payment account, measures a protected characteristic, and falls",
        "under the Data (Use and Access) Act. No automatic decline lowers the severity of",
        "one error, not the tier.",
        "",
        "## 2. Data and known gaps",
        "",
    ]

    if baselines:
        data = baselines["data"]
        lines += [
            f"Bank Account Fraud (NeurIPS 2022): {data['rows']:,} applications over eight",
            f"months, {_pct(data['prevalence'], 2)} fraud. Split by month only -- train 0-4,",
            "calibrate on month 5 in three parts, report 6 and 7 separately.",
        ]
    else:
        lines.append(MISSING)

    lines += [
        "",
        "Known gaps, worst first. The data is **synthetic and not UK data**, so no rate",
        "here transfers to a UK population. There is **no vulnerability information**, so",
        "no vulnerability analysis is possible. There are **no within-month timestamps**,",
        "so a 'day' is 4,000 shuffled applications: an assumption. The **costs are",
        "illustrative**. No saving is claimed, in pounds or otherwise.",
        "",
        "## 3. Method and key choices",
        "",
        "LightGBM without `customer_age`, calibrated on one third of month 5 and selected",
        "on another. Bands come from label-conditional split conformal prediction: two",
        "thresholds, fitted on a third part of month 5 that nothing else touches, turn a",
        "probability into a set of plausible labels, and the set picks the band. Three",
        "choices worth testing: **age is excluded and used only to measure**, at almost no",
        "cost in detection; **thresholds are never set on evaluation data**; and **the",
        "conformal calibration part is used once**.",
        "",
        "## 4. Performance",
        "",
    ]

    if champion:
        # Performance and fairness in one table, a row per month. Two tables of the
        # same two rows cost five lines each against the two-page cap, and a
        # reviewer comparing months wants them side by side anyway.
        rows = [
            "| Month | ROC-AUC | Fraud caught | False alarms | Brier "
            "| FA under 50 | FA 50+ | Ratio (95% CI) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for month, values in sorted(champion["months"].items()):
            groups = {r["group"]: r for r in values["fairness"]["by_group"]}
            rows.append(
                f"| {month} | {format_number(values['discrimination']['roc_auc'])} "
                f"| {_pct(values['realised']['tpr'])} "
                f"| {_pct(values['realised']['fpr'])} "
                f"| {format_number(values['calibration']['overall']['brier'], 4)} "
                f"| {_pct(groups.get('age<50', {}).get('fpr'))} "
                f"| {_pct(groups.get('age>=50', {}).get('fpr'))} "
                f"| {format_interval(values['fairness'].get('fpr_ratio_ci'))} |"
            )
        lines += rows
        comparison = champion["calibration"]["comparison"]
        chosen = champion["calibration"]["chosen"]
        lines += [
            "",
            f"Calibration **{chosen}**, on held-out data. Uncalibrated it predicts a "
            f"{_pct(comparison['none']['mean_predicted_rate'], 2)} fraud rate against "
            f"{_pct(comparison['none']['observed_rate'], 2)} observed: it ranks well and is",
            "wrong about magnitude, which matters: the policy reads probabilities.",
            "Discrimination holds on both months; the false alarm rate does not (finding 1).",
        ]
    else:
        lines.append(MISSING)

    lines += ["", "## 5. Fairness", ""]

    if champion:
        lines += [
            "The last three columns are predictive equality, on genuine applicants only; a",
            "ratio of 1.00 would mean both age groups stopped equally often. Age measures",
            "the model, which never sees it.",
        ]
    else:
        lines.append(MISSING)

    if fairness:
        lines += [
            "",
            "| Mitigation, month 6 | Fraud caught | False alarms | Ratio |",
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
            "Both months are in the README. Neither model-level mitigation beat dropping",
            "age, and M3 equalised by flagging almost nobody yet was still least equal.",
        ]
    else:
        lines += ["", "_Mitigation experiments not yet run (`make fairness`)._"]

    lines += ["", "## 6. Monitoring plan", ""]

    if monitoring:
        thresholds = monitoring["thresholds"]
        calibration = monitoring["threshold_calibration"]
        lines += [
            f"Label-free detectors on 4,000-application windows, at the "
            f"{calibration['percentile']:g}th percentile of "
            f"{calibration['clean_windows']} clean ones:",
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
            "**Alarms.** Watch when any detector exceeds its threshold; alert when one does",
            "so twice running, or score PSI passes 0.25. **Fallback.** On alert the policy",
            "tightens its alphas, sending more work to humans; the event is logged, read",
            "on every request, and never cleared automatically.",
            "**Owner.** Fraud operations own the queue, model risk the thresholds.",
            "**Re-validation:** any alert; any change to the model, its calibration or the",
            "alphas; a new calibration month; or a year.",
        ]
    else:
        lines.append(MISSING)

    lines += ["", "## 7. Findings", ""]
    for index, finding in enumerate(collect_findings(metrics), start=1):
        # One paragraph per finding: title, rating, what it is, what to do. At two
        # pages the blank lines between those cost a sentence each, and a reviewer
        # wants the recommendation in the same breath as the finding anyway.
        lines += [
            f"**{index}. {finding['title']}** — *{finding['rating']}*. "
            f"{finding['body']} *Recommendation:* {finding['recommendation']}",
            "",
        ]

    return "\n".join(lines) + "\n"
