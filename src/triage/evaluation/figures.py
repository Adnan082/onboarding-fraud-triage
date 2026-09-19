"""Figures for the report, drawn from ``reports/metrics.json`` and nothing else.

Section 14's rules, applied here: axes carry units, confidence intervals are shown
wherever they exist, the month is in the title, and the palette is colour-blind
safe. Every figure also has a table behind it in the README or the validation
report, so nothing depends on colour alone.

The palette is four hues validated for colour-vision deficiency (worst adjacent
pair: protan delta-E 9.1, normal-vision 22.9). Series keep a fixed slot: a figure
with three series uses slots 1-3, never a different three.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display in CI or WSL
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("triage")

# Validated categorical slots, in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d9d8d3"
SURFACE = "#fcfcfb"

FIGSIZE = (7.2, 4.4)
DPI = 150


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive axes, ink-coloured text, a light grid behind the marks."""
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    log.info("  wrote %s", path.name)
    return path


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: sane at small counts, where normal approximation is not.

    An age band with 200 genuine applicants needs an interval that does not fall
    off the end of the scale, which is exactly where the textbook interval fails.
    """
    if total == 0:
        return (float("nan"), float("nan"))

    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    spread = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def reliability_figure(metrics: dict[str, Any], out_dir: Path) -> Path | None:
    """Predicted against observed fraud rate, per test month."""
    champion = metrics.get("champion")
    if not champion:
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE)
    limit = 0.0

    for index, (month, values) in enumerate(sorted(champion["months"].items())):
        curve = values.get("reliability", [])
        if not curve:
            continue
        predicted = [row["mean_predicted"] * 100 for row in curve]
        observed = [row["observed"] * 100 for row in curve]
        limit = max(limit, max(predicted + observed))

        ax.plot(
            predicted,
            observed,
            color=SERIES[index],
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"Month {month}",
        )
        ax.annotate(
            f"Month {month}",
            (predicted[-1], observed[-1]),
            textcoords="offset points",
            xytext=(6, -2),
            color=INK_SECONDARY,
            fontsize=8.5,
        )

    edge = limit * 1.05 or 1.0
    ax.plot([0, edge], [0, edge], color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(
        "perfect calibration",
        (edge * 0.62, edge * 0.68),
        color=INK_SECONDARY,
        fontsize=8,
        rotation=38,
    )

    _style(
        ax,
        f"Are the probabilities honest? Months 6 and 7, {champion['calibration']['chosen']} "
        "calibration",
        "Predicted fraud rate (%)",
        "Observed fraud rate (%)",
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, loc="upper left")
    return _save(fig, out_dir / "reliability_by_month.png")


def fairness_tradeoff_figure(metrics: dict[str, Any], out_dir: Path) -> Path | None:
    """What each mitigation costs in detection, against what it buys in equality."""
    fairness = metrics.get("fairness")
    if not fairness or not fairness.get("tradeoff"):
        return None

    labels = {
        "m1_drop_age": "M1 drop age",
        "m2_fairgbm": "M2 FairGBM",
        "m3_fairlearn_eg": "M3 fairlearn",
        "m4_policy_only": "M4 policy only",
    }

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for index, (key, label) in enumerate(labels.items()):
        rows = [row for row in fairness["tradeoff"] if row["experiment"] == key]
        if not rows:
            continue

        x = [row["fpr_ratio"] for row in rows]
        y = [row["tpr"] * 100 for row in rows]
        low = [row["fpr_ratio"] - row["fpr_ratio_low"] for row in rows]
        high = [row["fpr_ratio_high"] - row["fpr_ratio"] for row in rows]

        ax.errorbar(
            x,
            y,
            xerr=[low, high],
            fmt="o",
            markersize=9,
            color=SERIES[index],
            ecolor=SERIES[index],
            elinewidth=1.5,
            capsize=3,
            alpha=0.95,
            label=label,
        )
        # Only M3 is labelled directly. M1, M2 and M4 sit on top of one another,
        # and three colliding labels would imply a precision the data does not
        # have; the legend identifies them and the note below says what the
        # cluster means.
        if key == "m3_fairlearn_eg":
            ax.annotate(
                label,
                (x[0], y[0]),
                textcoords="offset points",
                xytext=(12, 8),
                color=INK_SECONDARY,
                fontsize=8.5,
            )

    ax.axvline(1.0, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(
        "equal false-alarm rates",
        (1.0, ax.get_ylim()[0]),
        textcoords="offset points",
        xytext=(-6, 14),
        rotation=90,
        color=INK_SECONDARY,
        fontsize=8,
        ha="right",
    )

    _style(
        ax,
        "What each fairness mitigation costs. Months 6 and 7, with 95% intervals",
        "FPR ratio between age groups (1.00 = equal)",
        "Fraud caught (%)",
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, loc="center left")
    fig.text(
        0.5,
        -0.04,
        "M1, M2 and M4 overlap: neither model-level mitigation separates from simply "
        "dropping age.\nM3 equalises by flagging almost nobody, and is still the least "
        "equal of the four. Nothing reaches parity.",
        ha="center",
        color=INK_SECONDARY,
        fontsize=8,
    )
    return _save(fig, out_dir / "fairness_tradeoff.png")


def workload_figure(metrics: dict[str, Any], out_dir: Path) -> Path | None:
    """The trade-off that makes 90% coverage a staffing decision."""
    policy = metrics.get("policy")
    if not policy or not policy.get("grid"):
        return None

    alpha_legit = policy["chosen"]["alpha_legit"]
    rows = sorted(
        (row for row in policy["grid"] if row["alpha_legit"] == alpha_legit),
        key=lambda row: row["fraud_coverage"],
    )
    coverage = [row["fraud_coverage"] * 100 for row in rows]
    reviews = [row["review_share"] * 100_000 for row in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(coverage, reviews, color=SERIES[0], linewidth=2, marker="o", markersize=5)

    chosen = policy["chosen"]["expected_fraud_coverage"] * 100
    chosen_reviews = policy["chosen"]["expected_review_share"] * 100_000
    ax.plot([chosen], [chosen_reviews], marker="o", markersize=11, color=SERIES[1], zorder=5)
    ax.annotate(
        f"chosen: {chosen:.0f}% caught,\n{chosen_reviews:,.0f} reviews",
        (chosen, chosen_reviews),
        textcoords="offset points",
        xytext=(14, -26),
        color=INK_SECONDARY,
        fontsize=8.5,
    )

    top = rows[-1]
    ax.annotate(
        f"{top['fraud_coverage']:.0%} caught costs\n{top['review_share'] * 100_000:,.0f} reviews",
        (coverage[-1], reviews[-1]),
        textcoords="offset points",
        xytext=(-118, -6),
        color=INK_SECONDARY,
        fontsize=8.5,
    )

    _style(
        ax,
        "Catching more fraud is a staffing decision. Measured on month 5",
        "Fraud caught (%)",
        "Applications sent to review, per 100,000",
    )
    return _save(fig, out_dir / "coverage_vs_workload.png")


def age_band_figure(metrics: dict[str, Any], out_dir: Path, month: str = "6") -> Path | None:
    """False-alarm rate by age band, with intervals that show where it is noise."""
    champion = metrics.get("champion")
    if not champion or month not in champion.get("months", {}):
        return None

    bands = champion["months"][month]["fairness"]["by_band"]
    labels, rates, lows, highs, faded = [], [], [], [], []
    for record in bands:
        total = int(record["n_genuine"])
        if total == 0:
            continue
        rate = record["fpr"] * 100
        low, high = wilson_interval(int(record["false_alarms"]), total)

        labels.append(record["band"])
        rates.append(rate)
        lows.append(max(0.0, rate - low * 100))
        highs.append(high * 100 - rate)
        faded.append(total < 1000)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    positions = range(len(labels))
    colors = [SERIES[3] if thin else SERIES[0] for thin in faded]

    ax.bar(positions, rates, color=colors, width=0.62, zorder=3)
    ax.errorbar(
        positions,
        rates,
        yerr=[lows, highs],
        fmt="none",
        ecolor=INK_SECONDARY,
        elinewidth=1.4,
        capsize=4,
        zorder=4,
    )

    for position, rate, high, thin in zip(positions, rates, highs, faded, strict=True):
        # Sit the value above the interval's top, not the bar's, or the two collide.
        ax.annotate(
            f"{rate:.1f}%",
            (position, rate + high),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            color=INK_SECONDARY,
            fontsize=8,
        )
        if thin:
            ax.annotate(
                "few",
                (position, 0),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                color=INK,
                fontsize=7.5,
            )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=0)
    _style(
        ax,
        f"Genuine applicants stopped, by age. Month {month}, with 95% intervals",
        "Age band (years)",
        "False-alarm rate (%)",
    )
    # A caption below the axes, so it reads after the chart rather than above the title.
    fig.text(
        0.5,
        -0.04,
        'Bars marked "few" rest on fewer than 1,000 genuine applicants: their intervals are '
        "wide\nand the rate should not be read as a finding.",
        ha="center",
        color=INK_SECONDARY,
        fontsize=8,
    )
    return _save(fig, out_dir / f"fpr_by_age_band_month_{month}.png")


FIGURES = (reliability_figure, fairness_tradeoff_figure, workload_figure, age_band_figure)


def render_all(metrics: dict[str, Any], out_dir: Path) -> list[Path]:
    """Draw every figure the artefacts support. Missing sections are skipped."""
    drawn = []
    for figure in FIGURES:
        path = figure(metrics, out_dir)
        if path is not None:
            drawn.append(path)
    return drawn
