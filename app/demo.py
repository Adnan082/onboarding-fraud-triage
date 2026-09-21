"""Two-tab Streamlit demo (CLAUDE.md section 12).

**The policy** reads ONLY precomputed aggregates: ``reports/tables/*.csv`` and
``reports/metrics.json``. No training, no scoring, no heavy computation -- so it
starts instantly and can never disagree with the report, because both read the
same artefacts.

**Score an application** posts to the running service rather than scoring in
process. That keeps section 12's promise -- this app still computes nothing --
and it means the screen shows what the API actually returns, contract validation
and all, instead of a second implementation that could drift from it. The form is
built from the frozen contract, the same source the request model is generated
from, so a value the API would reject cannot be entered here either.

The screen answers one question for a non-technical viewer: *how much fraud do we
catch, and what does that cost the people reviewing it?* Everything else is in
service of that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

from triage.api.schemas import example_application, request_fields

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics.json"
POLICY_GRID = ROOT / "reports" / "tables" / "policy_grid.csv"
POLICY_OUTCOMES = ROOT / "reports" / "tables" / "policy_outcomes.csv"

API_URL = "http://127.0.0.1:8000"

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

# What each band means on screen. There is no fourth entry, because there is no
# automatic decline (rule 5).
BAND_STYLE = {
    "approve": ("✅", "Approved straight through", "No further checks."),
    "review": ("🔍", "Sent to an analyst", "The model cannot separate this case."),
    "verify": ("📋", "Asked for extra verification", "Step-up checks, never a decline."),
}

# The fields worth putting in front of someone in a two-minute demo. The rest are
# real inputs too and are editable below; these are the ones that move the answer.
HEADLINE_FIELDS = (
    "income",
    "customer_age",
    "employment_status",
    "housing_status",
    "credit_risk_score",
    "proposed_credit_limit",
    "name_email_similarity",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
    "keep_alive_session",
)

# Starting points, named for what they contain rather than what they score.
# Calling one "risky" would claim an outcome before the model has given one.
#
# As it happens the three currently land on the three different bands, which
# makes for a tidy two-minute demo. That is an observation about this model, not
# a property of the presets, and it is not asserted anywhere: a retrain is
# allowed to move them and the screen should show whatever it shows.
PRESETS: dict[str, dict[str, Any]] = {
    "Contract midpoint": {},
    "Established, long settled": {
        "income": 0.8,
        "customer_age": 40,
        "employment_status": "CA",
        "housing_status": "BA",
        "prev_address_months_count": 120,
        "current_address_months_count": 180,
        "bank_months_count": 30,
        "credit_risk_score": 220,
        "has_other_cards": 1,
        "phone_home_valid": 1,
        "phone_mobile_valid": 1,
        "email_is_free": 0,
        "name_email_similarity": 0.85,
        "device_distinct_emails_8w": 1,
        "session_length_in_minutes": 12.0,
    },
    "Thin file, shared device": {
        "income": 0.2,
        "customer_age": 20,
        "prev_address_months_count": -1,
        "bank_months_count": -1,
        "credit_risk_score": 40,
        "has_other_cards": 0,
        "phone_home_valid": 0,
        "email_is_free": 1,
        "name_email_similarity": 0.05,
        "device_distinct_emails_8w": 6,
        "session_length_in_minutes": 2.0,
        "keep_alive_session": 0,
        "foreign_request": 1,
    },
}


@st.cache_data
def load_artefacts() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Everything the policy tab shows, read once."""
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    return metrics, pd.read_csv(POLICY_GRID), pd.read_csv(POLICY_OUTCOMES)


def api_health() -> dict[str, Any] | None:
    """Whether the service is answering. Returns None rather than raising."""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=2)
        return dict(response.json()) if response.status_code == 200 else None
    except httpx.HTTPError:
        return None


def score(features: dict[str, Any], explain: bool) -> tuple[int, dict[str, Any]]:
    """Post one application. Returns the status code and the parsed body."""
    response = httpx.post(
        f"{API_URL}/score",
        params={"explain": str(explain).lower()},
        json={"application_id": "demo", "features": features},
        timeout=30,
    )
    return response.status_code, response.json()


def _input_for(name: str, spec: Any, value: Any) -> Any:
    """One form control, shaped by the contract rather than written out by hand."""
    label = name.replace("_", " ")
    help_text = spec.note or None

    if spec.levels and spec.kind == "category":
        levels = [str(level) for level in spec.levels]
        return st.selectbox(label, levels, index=levels.index(str(value)), help=help_text)
    if spec.levels:
        numbers = [int(level) for level in spec.levels]
        return st.selectbox(label, numbers, index=numbers.index(int(value)), help=help_text)
    if spec.kind == "int":
        return st.number_input(
            label,
            min_value=int(spec.lo),
            max_value=int(spec.hi),
            value=int(value),
            step=1,
            help=help_text,
        )
    return st.number_input(
        label,
        min_value=float(spec.lo),
        max_value=float(spec.hi),
        value=float(value),
        help=help_text,
    )


def render_scoring_tab() -> None:
    """One live application, scored by the service exactly as a caller would."""
    st.subheader("Score one application")
    st.caption(
        "This posts to the running service and shows what it returns. Every field "
        "is bounded by the frozen data contract, so a value the API would reject "
        "cannot be entered here either."
    )

    if api_health() is None:
        st.error(
            f"The scoring service is not answering on `{API_URL}`. Start it with "
            "`make serve` in another terminal — or use `make demo`, which starts "
            "both together."
        )
        return

    fields = request_fields()
    preset_name = st.selectbox(
        "Start from", list(PRESETS), help="A starting point, not a prediction."
    )
    features = {**example_application(), **PRESETS[preset_name]}

    edited = dict(features)
    columns = st.columns(2)
    for index, name in enumerate(HEADLINE_FIELDS):
        with columns[index % 2]:
            edited[name] = _input_for(name, fields[name], features[name])

    with st.expander(f"The other {len(fields) - len(HEADLINE_FIELDS)} fields"):
        rest = [name for name in fields if name not in HEADLINE_FIELDS]
        more = st.columns(3)
        for index, name in enumerate(rest):
            with more[index % 3]:
                edited[name] = _input_for(name, fields[name], features[name])

    explain = st.checkbox(
        "Ask for reason codes",
        value=True,
        help=(
            "SHAP, computed per request. It costs roughly twenty times the rest of "
            "the request, which is why it is optional."
        ),
    )

    if not st.button("Score it", type="primary"):
        return

    status, body = score(edited, explain)
    if status == 422:
        st.error("The contract rejected this application, so it was never scored.")
        st.json(body)
        return
    if status != 200:
        st.error(f"The service returned {status}.")
        st.json(body)
        return

    icon, headline, subtitle = BAND_STYLE[body["decision"]]
    st.markdown(f"## {icon} {headline}")
    st.caption(subtitle)

    top = st.columns(3)
    top[0].metric("Risk score", f"{body['risk_score']:.3f}")
    top[1].metric(
        "Labels not ruled out",
        " + ".join(body["conformal_set"]) if body["conformal_set"] else "none",
        help=(
            "An empty set means the application looks unlike anything in the "
            "calibration month, which sends it to a human rather than deciding."
        ),
    )
    top[2].metric(
        "Drift status",
        body["drift_status"],
        help="Read from the monitor on every request. Never cleared automatically.",
    )

    if body.get("reasons"):
        st.markdown("**Why — written for an analyst, never shown to an applicant:**")
        for reason in body["reasons"]:
            st.markdown(f"- {reason['text']}  \n  _{reason['feature']} is {reason['direction']}_")

    if body.get("fallback_active"):
        st.warning(
            "The monitor has raised an alert, so the policy is running its tightened "
            "alphas: more applications go to a human than usual."
        )

    st.caption(
        f"model `{body['model_version']}` · policy `{body['policy_version']}` · "
        "no application is ever declined automatically"
    )


def render_policy_tab(metrics: dict, grid: pd.DataFrame, outcomes: pd.DataFrame) -> None:
    """The capacity trade-off, from precomputed artefacts only."""
    policy = metrics["policy"]
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

    policy_tab, scoring_tab = st.tabs(["The policy", "Score an application"])
    with policy_tab:
        render_policy_tab(metrics, grid, outcomes)
    with scoring_tab:
        render_scoring_tab()

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
