"""The drift monitor: detectors, alarm rules, fallback, and the injected bugs.

PSI has its own file (`test_psi.py`). This covers the other two detectors, the
WATCH/ALERT logic, the state the API reads, and the bug injectors — including the
property that makes three of the four bugs interesting: they stay legal.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from experiments.inject_bugs import mirror_lookup, multiply, set_constant, swap_top_categories
from triage.data.contract import check
from triage.monitoring.alarms import (
    ALERT,
    OK,
    WATCH,
    AlarmRun,
    calibrate_thresholds,
    evaluate_window,
    exceeded_detectors,
)
from triage.monitoring.conformal_rate import (
    binomial_flag,
    binomial_p_value,
    exact_small_n,
    rates,
    window_test,
)
from triage.monitoring.domain_clf import domain_auc, reference_sample
from triage.monitoring.fallback import activate, clear, read_events, read_state


@pytest.fixture(scope="module")
def cfg(repo_root):
    """The composed project config."""
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        return compose(config_name="config")


# --- domain classifier ----------------------------------------------------------------


def _pool(n: int, seed: int) -> pd.DataFrame:
    """Model-ready numeric features from one distribution."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "velocity_6h": rng.normal(5000, 2000, n),
            "income": rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], n),
            "credit_risk_score": rng.normal(120, 60, n),
        }
    )


@pytest.fixture(scope="module")
def disjoint_halves() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A reference and a window: same distribution, **different rows**.

    Drawing the window out of the reference instead would put identical rows on
    both sides of the classifier with opposite labels, which makes the AUC
    meaningless. In the real monitor the reference comes from `cal_conf` and the
    windows from months 6 and 7, so they are always disjoint.
    """
    pool = _pool(2100, seed=20260918)
    return pool.iloc[:1500].reset_index(drop=True), pool.iloc[1500:].reset_index(drop=True)


def test_an_identical_window_is_indistinguishable(disjoint_halves) -> None:
    """Same distribution, different rows: the classifier should be near chance."""
    reference, window = disjoint_halves
    assert domain_auc(window, reference, seed=7) == pytest.approx(0.5, abs=0.08)


def test_a_shifted_window_is_obvious(disjoint_halves) -> None:
    """A column set to a constant should be trivially separable."""
    reference, window = disjoint_halves
    shifted = window.copy()
    shifted["income"] = 0.9
    assert domain_auc(shifted, reference, seed=7) > 0.7


def test_the_domain_classifier_is_seeded(disjoint_halves) -> None:
    """Two runs with the same seed give the same number."""
    reference, window = disjoint_halves
    assert domain_auc(window, reference, seed=11) == domain_auc(window, reference, seed=11)


def test_mismatched_columns_raise(disjoint_halves) -> None:
    """Comparing different feature sets would measure the schema, not the drift."""
    reference, window = disjoint_halves
    with pytest.raises(ValueError, match="same columns"):
        domain_auc(window.drop(columns=["income"]), reference, seed=1)


def test_reference_sample_is_fixed(disjoint_halves) -> None:
    """The reference is drawn once: re-drawing per window would add noise."""
    reference, _ = disjoint_halves
    first = reference_sample(reference, 500, seed=42)
    second = reference_sample(reference, 500, seed=42)
    assert first.equals(second)
    assert len(first) == 500


# --- conformal-rate test ----------------------------------------------------------------


def test_rates_match_the_set_construction() -> None:
    """The comparisons must mirror predict_sets exactly, or the detector drifts from the policy."""
    probs = np.array([0.01, 0.05, 0.10, 0.20, 0.50])
    excluded, fraud = rates(probs, tau_fraud=0.10, tau_legit=0.20)

    assert excluded == pytest.approx(1 / 5)  # only 0.50 is above tau_legit
    assert fraud == pytest.approx(3 / 5)  # 0.10, 0.20, 0.50 are at or above tau_fraud


def test_a_rate_on_target_is_not_flagged() -> None:
    """The expected rate itself must never raise an alarm."""
    assert not binomial_flag(200, 4000, 0.05)
    assert binomial_p_value(200, 4000, 0.05) == pytest.approx(1.0, abs=0.05)


def test_a_clearly_shifted_rate_is_flagged() -> None:
    """Double the expected rate in a 4,000-application window is decisive."""
    assert binomial_flag(400, 4000, 0.05)
    assert binomial_p_value(400, 4000, 0.05) < 1e-3


def test_the_normal_approximation_matches_an_exact_test() -> None:
    """The approximation is documented, so it had better be right at these sizes."""
    for observed, n, rate in [(20, 200, 0.05), (5, 100, 0.10), (30, 300, 0.08)]:
        approximate = binomial_p_value(observed, n, rate)
        exact = exact_small_n(observed, n, rate)
        assert approximate == pytest.approx(exact, abs=0.05)


def test_binomial_guards() -> None:
    """Nonsense input is an error rather than a silent zero."""
    with pytest.raises(ValueError, match="n must be positive"):
        binomial_p_value(1, 0, 0.5)
    with pytest.raises(ValueError, match="between 0 and"):
        binomial_p_value(11, 10, 0.5)
    with pytest.raises(ValueError, match="probability"):
        binomial_p_value(1, 10, 1.5)


def test_window_test_reports_both_rates() -> None:
    """One window, both rates, both p-values, one flag."""
    rng = np.random.default_rng(2)
    probs = rng.beta(1.2, 60, 4000)
    result = window_test(probs, tau_fraud=0.02, tau_legit=0.08, reference_rates=(0.02, 0.10))

    assert {"excluded_rate", "fraud_rate", "excluded_p_value", "fraud_p_value"} <= set(result)
    assert result["n"] == 4000


# --- alarm rules --------------------------------------------------------------------------


def test_thresholds_are_the_99th_percentile_of_clean_windows() -> None:
    """The empirical threshold fixes the false-alarm rate by construction."""
    clean = [{"psi_score": value / 100} for value in range(100)]
    thresholds = calibrate_thresholds(clean, percentile=99)
    assert thresholds["psi_score"] == pytest.approx(0.99, abs=0.01)


def test_calibration_needs_clean_windows() -> None:
    with pytest.raises(ValueError, match="no clean windows"):
        calibrate_thresholds([])


def test_one_detector_over_threshold_is_a_watch() -> None:
    """A single window over threshold is common. It is worth noting, not acting on."""
    result = evaluate_window(
        {"psi_score": 0.2, "domain_auc": 0.5}, {"psi_score": 0.1, "domain_auc": 0.6}
    )
    assert result.level == WATCH
    assert result.exceeded == ("psi_score",)


def test_nothing_over_threshold_is_ok() -> None:
    result = evaluate_window({"psi_score": 0.05}, {"psi_score": 0.1})
    assert result.level == OK
    assert result.exceeded == ()


def test_the_same_detector_twice_running_is_an_alert() -> None:
    """Two in a row is what separates a break from a blip."""
    thresholds = {"psi_score": 0.1}
    run = AlarmRun(thresholds)

    first = run.add({"psi_score": 0.15, "window_id": 1, "month": 6})
    second = run.add({"psi_score": 0.16, "window_id": 2, "month": 6})

    assert first.level == WATCH
    assert second.level == ALERT
    assert "two consecutive windows" in " ".join(second.reasons)


def test_two_different_detectors_in_a_row_is_not_an_alert() -> None:
    """The rule is the *same* detector twice, not any two watches."""
    thresholds = {"psi_score": 0.1, "domain_auc": 0.6}
    run = AlarmRun(thresholds)

    run.add({"psi_score": 0.15, "domain_auc": 0.5})
    second = run.add({"psi_score": 0.05, "domain_auc": 0.7})

    assert second.level == WATCH


def test_a_large_score_psi_alerts_immediately() -> None:
    """Above the 0.25 rule of thumb the score distribution has plainly moved."""
    result = evaluate_window({"psi_score": 0.30}, {"psi_score": 10.0})
    assert result.level == ALERT
    assert "rule of thumb" in " ".join(result.reasons)


def test_alarm_run_counts_and_first_alert() -> None:
    """The summary the monitoring report reads."""
    run = AlarmRun({"psi_score": 0.1})
    run.add({"psi_score": 0.01})
    run.add({"psi_score": 0.15})
    run.add({"psi_score": 0.15})

    assert run.counts() == {OK: 1, WATCH: 1, ALERT: 1}
    assert run.first_alert() is not None
    assert run.first_alert().window_id == -1  # not supplied in these windows


def test_exceeded_ignores_detectors_without_a_threshold() -> None:
    """A detector that was never calibrated cannot raise an alarm."""
    assert exceeded_detectors({"new_detector": 99.0}, {"psi_score": 0.1}) == ()


# --- fallback and the state the API reads --------------------------------------------------


def test_missing_state_reads_as_ok(tmp_path) -> None:
    """Before the monitor has ever run, the service is not in fallback."""
    assert read_state(tmp_path / "monitor_state.json") == {
        "drift_status": "ok",
        "fallback_active": False,
    }


def test_activate_writes_state_and_logs_an_event(tmp_path) -> None:
    """The API reads the state; a reviewer reads the log."""
    state_path = tmp_path / "monitor_state.json"
    event_log = tmp_path / "monitor_events.jsonl"

    state = activate(
        "psi_score over threshold twice",
        state_path,
        event_log,
        {"alpha_fraud": 0.15, "alpha_legit": 0.05},
        window_id=12,
        month=6,
    )

    assert state["fallback_active"] is True
    assert state["drift_status"] == "alert"
    assert state["alpha_fraud"] == 0.15

    written = read_state(state_path)
    assert written["fallback_active"] is True
    assert written["last_window"] == 12

    events = read_events(event_log)
    assert len(events) == 1
    assert events[0]["event"] == "fallback_activated"
    assert "at" in events[0]


def test_the_event_log_is_appended_never_rewritten(tmp_path) -> None:
    """The history of what the monitor did is evidence; it does not get tidied."""
    state_path = tmp_path / "monitor_state.json"
    event_log = tmp_path / "monitor_events.jsonl"

    activate("first", state_path, event_log, {"alpha_fraud": 0.15, "alpha_legit": 0.05})
    activate("second", state_path, event_log, {"alpha_fraud": 0.15, "alpha_legit": 0.05})
    clear(state_path, event_log)

    events = read_events(event_log)
    assert [event["event"] for event in events] == [
        "fallback_activated",
        "fallback_activated",
        "fallback_cleared",
    ]
    assert read_state(state_path)["fallback_active"] is False


def test_a_corrupt_state_file_does_not_take_the_service_down(tmp_path) -> None:
    """A monitor that cannot be read is a monitoring problem, not a scoring one."""
    state_path = tmp_path / "monitor_state.json"
    state_path.write_text("{ this is not json", encoding="utf-8")

    state = read_state(state_path)
    assert state["fallback_active"] is False
    assert "unreadable" in state["reason"]


def test_state_is_written_atomically(tmp_path) -> None:
    """No half-written state: the API may read at any moment."""
    state_path = tmp_path / "monitor_state.json"
    event_log = tmp_path / "events.jsonl"
    activate("x", state_path, event_log, {"alpha_fraud": 0.1, "alpha_legit": 0.02})

    assert not list(tmp_path.glob("*.tmp"))
    json.loads(state_path.read_text(encoding="utf-8"))  # parses, so it is complete


# --- the injected bugs -----------------------------------------------------------------------


def test_swapping_categories_keeps_the_totals(fixture_frame: pd.DataFrame) -> None:
    """Bug (a) moves meaning, not counts. No schema check can see it."""
    broken = swap_top_categories(fixture_frame, "employment_status")
    before = fixture_frame["employment_status"].value_counts()
    after = broken["employment_status"].value_counts()

    top_two = before.index[:2].tolist()
    assert after[top_two[0]] == before[top_two[1]]
    assert after[top_two[1]] == before[top_two[0]]
    assert set(after.index) == set(before.index)
    assert after.sum() == before.sum()


def test_mirrored_income_stays_inside_the_contract(fixture_frame: pd.DataFrame) -> None:
    """Bug (b) uses a lookup, not 1 - x, precisely so every value stays legal."""
    broken = mirror_lookup(fixture_frame, "income")

    assert set(broken["income"].unique()) <= set(fixture_frame["income"].unique())
    assert check(broken, name="mirrored income").passed
    assert not broken["income"].equals(fixture_frame["income"])


def test_the_constant_bug_stays_inside_the_contract(fixture_frame: pd.DataFrame) -> None:
    """Bug (c): every value is legal; only the distribution collapses."""
    broken = set_constant(fixture_frame, "phone_mobile_valid", 1)
    assert broken["phone_mobile_valid"].nunique() == 1
    assert check(broken, name="all mobile valid").passed


def test_the_scaling_bug_is_rejected_by_the_contract(fixture_frame: pd.DataFrame) -> None:
    """Bug (d) is the control: it must never reach a detector at all."""
    broken = multiply(fixture_frame, "income", 10)
    result = check(broken, name="income x 10")

    assert not result.passed
    assert any("income" in failure for failure in result.failures)


def test_the_in_contract_bugs_are_invisible_to_the_schema(fixture_frame: pd.DataFrame) -> None:
    """The whole case for a label-free monitor, in one assertion."""
    for broken in (
        swap_top_categories(fixture_frame, "employment_status"),
        mirror_lookup(fixture_frame, "income"),
        set_constant(fixture_frame, "phone_mobile_valid", 1),
    ):
        assert check(broken, name="in-contract bug").passed
