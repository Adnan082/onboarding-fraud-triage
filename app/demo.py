"""One-screen Streamlit demo (CLAUDE.md section 12).

Reads ONLY precomputed aggregates: ``reports/tables/*.csv`` and
``reports/metrics.json``. No training, no scoring, no heavy computation -- so it
starts instantly and can never disagree with the report, because both read the
same artefacts.

The screen answers one question for a non-technical viewer: *how much fraud do we
catch, and what does that cost the people reviewing it?* Everything else is in
service of that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics.json"
POLICY_GRID = ROOT / "reports" / "tables" / "policy_grid.csv"
POLICY_OUTCOMES = ROOT / "reports" / "tables" / "policy_outcomes.csv"

SYNTHETIC_NOTE = (
    "**These are not real customers.** The figures come from a public, synthetic "
    "research dataset of bank-account applications (BAF, NeurIPS 2022). It is not "
    "UK data, and nothing here is a claim about any bank."
)

BANDS = {
    "approve": "Approved straight through",
    "review": "Sent to an analyst",
    "verify": "Asked for extra verification",
}


@st.cache_data
def load_artefacts() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Everything the screen shows, read once."""
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    return metrics, pd.read_csv(POLICY_GRID), pd.read_csv(POLICY_OUTCOMES)


def main() -> None:
    """Render the demo."""
    st.set_page_config(page_title="Application fraud triage", layout="wide")
    st.title("Application fraud triage")
    st.caption(
        "Every application is approved, sent to an analyst, or asked for extra "
        "verification. Nothing is ever declined automatically."
    )
    st.info(SYNTHETIC_NOTE)

    missing = [p.name for p in (METRICS, POLICY_GRID, POLICY_OUTCOMES) if not p.exists()]
    if missing:
        st.warning(
            f"No results yet ({', '.join(missing)} missing). Run `make evaluate` and "
            "`make report` first: this screen only reads artefacts, it never computes "
            "anything itself."
        )
        return

    metrics, grid, outcomes = load_artefacts()
    policy = metrics.get("policy")
    if not policy:
        st.warning("`reports/metrics.json` has no policy section yet. Run `make conformal`.")
        return

    alpha_legit = float(policy["chosen"]["alpha_legit"])
    affordable_grid = grid[grid["alpha_legit"] == alpha_legit].sort_values("review_share")

    # --- the one control on the screen -------------------------------------------
    st.subheader("How much review capacity does the team have?")
    capacity = st.slider(
        "Share of applications an analyst can look at",
        min_value=float(affordable_grid["review_share"].min().round(3)),
        max_value=float(affordable_grid["review_share"].max().round(3)),
        value=float(policy["capacity"]["review_share"]),
        step=0.005,
        format="%.1f%%",
        help=(
            "Chosen by the bank, not by the data. The policy below is the one that "
            "catches the most fraud without exceeding this."
        ),
    )

    affordable = affordable_grid[affordable_grid["review_share"] <= capacity]
    if affordable.empty:
        st.error(
            "No policy fits that capacity. The smallest review share available is "
            f"{affordable_grid['review_share'].min():.1%}."
        )
        return

    best = affordable.loc[affordable["fraud_coverage"].idxmax()]
    alpha_fraud = float(best["alpha_fraud"])

    st.success(
        f"With **{capacity:.1%}** of applications going to review, the best policy "
        f"catches **{best['fraud_coverage']:.0%}** of fraud "
        f"(`alpha_fraud = {alpha_fraud:g}`, `alpha_legit = {alpha_legit:g}`)."
    )

    # --- what that policy actually did, on months it never saw --------------------
    chosen_outcomes = outcomes[
        (outcomes["alpha_fraud"] == alpha_fraud) & (outcomes["alpha_legit"] == alpha_legit)
    ].sort_values("month")

    if chosen_outcomes.empty:
        st.warning("No test-month outcomes stored for that policy. Re-run `make conformal`.")
        return

    st.subheader("What it would have done, on two months it never saw")
    for _, row in chosen_outcomes.iterrows():
        st.markdown(f"**Month {int(row['month'])}** — {int(row['n']):,} applications")
        columns = st.columns(5)
        columns[0].metric(
            "Fraud caught",
            f"{row['fraud_coverage']:.0%}",
            help=f"{row['fraud_caught']:.0f} of {int(row['n_fraud']):,} frauds",
        )
        columns[1].metric(BANDS["approve"], f"{row['approve_share']:.1%}")
        columns[2].metric(BANDS["review"], f"{row['review_share']:.1%}")
        columns[3].metric(BANDS["verify"], f"{row['verify_share']:.1%}")
        columns[4].metric(
            "Cost per 10,000",
            f"{row['cost_total']:,.0f}",
            help="Illustrative units, from assumed costs. Not a saving, and not pounds.",
        )

        gap = st.columns(3)
        gap[0].metric(
            "False alarms, under 50",
            f"{row['fpr_under_cut']:.1%}",
            help="Genuine applicants who were stopped.",
        )
        gap[1].metric("False alarms, 50 and over", f"{row['fpr_over_cut']:.1%}")
        gap[2].metric(
            "Equality of false alarms",
            f"{row['fpr_ratio']:.2f}",
            help="1.00 would mean both age groups are stopped equally often.",
        )
        st.divider()

    # --- the trade-off, which is the point ----------------------------------------
    st.subheader("What more fraud costs in review workload")
    curve = affordable_grid.set_index("fraud_coverage")[["review_share"]]
    curve.columns = ["Share of applications sent to review"]
    st.line_chart(curve)
    st.caption(
        "Measured on the calibration month. The last few points of fraud coverage "
        "cost far more review capacity than the first: catching 90% is a staffing "
        "decision, not a modelling one."
    )

    with st.expander("What this does not show"):
        st.markdown(
            "- The data is synthetic and not from the UK.\n"
            "- Costs are illustrative parameters chosen for comparison, never a saving.\n"
            "- Age is used to measure the system, never to decide anything about "
            "anyone. The model does not see it.\n"
            "- The guarantee holds while new applications look like the calibration "
            "month. On these two test months one half of it did not hold, which is "
            "why the system is monitored.\n"
            f"- Model version `{policy.get('model_version', 'unknown')}`, "
            f"calibration `{policy.get('calibration', 'unknown')}`."
        )


if __name__ == "__main__":
    main()
