"""One-screen Streamlit demo (CLAUDE.md section 12).

Reads ONLY precomputed aggregates: ``reports/tables/policy_grid.csv`` and
``reports/metrics.json``. No training, no heavy computation, plain-English labels,
and a visible note that the data is synthetic.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics.json"
POLICY_GRID = ROOT / "reports" / "tables" / "policy_grid.csv"

SYNTHETIC_NOTE = (
    "These figures come from a public, synthetic research dataset of bank-account "
    "applications. They are not real customers, and not UK data."
)


def main() -> None:
    """Render the demo."""
    st.set_page_config(page_title="Application fraud triage", layout="wide")
    st.title("Application fraud triage")
    st.info(SYNTHETIC_NOTE)

    if not METRICS.exists() or not POLICY_GRID.exists():
        st.warning(
            "No results yet. Run `make evaluate` and `make report` first: this screen "
            "only reads artefacts, it never computes anything."
        )
        return

    # TODO(week 3): capacity slider -> best-coverage alpha pair on cal_tune, then
    # month 6 and month 7 outcomes: band shares, fraud caught, genuine reviewed,
    # false-alarm rate by age group with CIs, and expected cost.
    st.warning("The demo screen is built in week 3.")


if __name__ == "__main__":
    main()
