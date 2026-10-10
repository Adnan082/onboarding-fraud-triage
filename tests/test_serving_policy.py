"""Exercise real fixture-model scoring, bundle checks and live policy changes."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from triage.api import service
from triage.api.app import app
from triage.api.schemas import example_application
from triage.models.calibrate import Calibrator
from triage.models.champion import fit, save
from triage.monitoring.fallback import activate, clear, read_state
from triage.policy.bundle import POLICY_FILE, publish
from triage.uncertainty.conformal import ConformalThresholds


@pytest.fixture
def serving(tmp_path, fixture_frame, monkeypatch):
    cfg = service.load_config()
    cfg.paths.root = str(tmp_path)
    cfg.model.params.n_estimators = 8
    cfg.model.params.min_child_samples = 10
    cfg.model.params.n_jobs = 1
    model = fit(fixture_frame, cfg)
    save(model, Calibrator("none"), cfg)
    # Deliberately separated test thresholds make a fixed input change band.
    # They are test policy fixtures, not fitted or published performance claims.
    normal = ConformalThresholds(1.0, 1.0, 0.45, 0.01, 20, 200)
    fallback = ConformalThresholds(0.0, 1.0, 0.15, 0.05, 20, 200)
    publish(cfg, normal, fallback)
    loaded = service.ScoringService.load(cfg)
    state = tmp_path / "monitor_state.json"
    events = tmp_path / "events.jsonl"
    clear(state, events)
    monkeypatch.setattr(service, "MONITOR_STATE_PATH", state)
    return cfg, loaded, state, events


def test_alert_changes_decision_and_reset_restores_it(serving):
    _, loaded, state, events = serving
    features = example_application()
    before = loaded.score(features, explain=False)
    activate("test drift", state, events, {"alpha_fraud": 0.15, "alpha_legit": 0.05})
    after = loaded.score(features, explain=False)
    assert before["decision"] == "approve"
    assert after["decision"] == "review"
    assert before["risk_score"] == after["risk_score"]
    assert before["policy_version"] != after["policy_version"]
    assert after["fallback_active"] is True
    clear(state, events)
    assert loaded.score(features, explain=False)["decision"] == "approve"


def test_concurrent_requests_use_the_same_active_policy(serving):
    _, loaded, state, events = serving
    activate("test drift", state, events, {"alpha_fraud": 0.15, "alpha_legit": 0.05})
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(lambda _: loaded.score(example_application(), explain=False), range(6))
        )
    assert {r["decision"] for r in results} == {"review"}
    assert {r["policy_version"] for r in results} == {loaded.fallback_thresholds.policy_version}


def test_changed_policy_is_rejected(serving):
    cfg, _, _, _ = serving
    from pathlib import Path

    path = Path(cfg.paths.models) / POLICY_FILE
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="manifest"):
        service.ScoringService.load(cfg)


def test_changed_feature_config_is_rejected(serving):
    cfg, _, _, _ = serving
    cfg.features.use_age = True
    with pytest.raises(ValueError, match="context|contract"):
        service.ScoringService.load(cfg)


def test_http_readiness_and_scoring_fail_closed(serving, monkeypatch):
    _, loaded, state, _ = serving
    monkeypatch.setattr(service.ScoringService, "load", classmethod(lambda cls: loaded))
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        assert (
            client.post(
                "/score?explain=false", json={"features": example_application()}
            ).status_code
            == 200
        )
        state.unlink()
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert (
            client.post(
                "/score?explain=false", json={"features": example_application()}
            ).status_code
            == 503
        )


def test_readiness_without_model(monkeypatch):
    def broken(cls):
        raise ValueError("artifact mismatch")

    monkeypatch.setattr(service.ScoringService, "load", classmethod(broken))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


@pytest.mark.parametrize("payload", ["[]", '{"drift_status": [], "fallback_active": false}'])
def test_malformed_monitor_shapes_are_unknown(tmp_path, payload):
    path = tmp_path / "state.json"
    path.write_text(payload)
    assert read_state(path)["drift_status"] == "unknown"


def test_stale_monitor_is_unknown_but_alert_remains_latched(tmp_path, monkeypatch):
    import json

    path = tmp_path / "state.json"
    state = {
        "drift_status": "ok",
        "fallback_active": False,
        "updated_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
    }
    path.write_text(json.dumps(state))
    monkeypatch.setenv("TRIAGE_MONITOR_MAX_AGE_SECONDS", "60")
    assert read_state(path)["drift_status"] == "unknown"
    state.update(drift_status="alert", fallback_active=True)
    path.write_text(json.dumps(state))
    assert read_state(path)["fallback_active"] is True
