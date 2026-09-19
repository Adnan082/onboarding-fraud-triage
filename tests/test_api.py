"""The scoring service (section 10).

Required checks (CLAUDE.md section 13):
- 200 for valid input, 422 for invalid;
- changing customer_age does not change the score;
- the response carries drift status.

The valid payload is built from the frozen contract rather than hand-written, so
these tests cannot quietly pass against a stale idea of what an application looks
like. Tests that need trained artefacts are marked `data` and skip without them.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from triage.api.app import app
from triage.api.schemas import EXCLUDED_FROM_REQUEST, ApplicationFeatures
from triage.data.contract import CONTRACT


def valid_features() -> dict[str, Any]:
    """One legal application, generated from the contract."""
    payload: dict[str, Any] = {}
    for name, spec in CONTRACT.items():
        if name in EXCLUDED_FROM_REQUEST:
            continue
        if spec.levels:
            payload[name] = spec.levels[0]
        else:
            low = float(spec.lo if spec.lo is not None else 0.0)
            high = float(spec.hi if spec.hi is not None else 1.0)
            middle = low + (high - low) / 2
            payload[name] = int(middle) if spec.kind == "int" else middle
    return payload


@pytest.fixture(scope="module")
def client():
    """A client with the app's lifespan run, so artefacts load if they exist."""
    with TestClient(app) as test_client:
        yield test_client


def has_model(client) -> bool:
    """Whether trained artefacts were loaded at startup."""
    return client.get("/version").status_code == 200


# --- validation, which needs no model at all -------------------------------------


def test_the_request_schema_covers_the_contract() -> None:
    """Generated from the contract, minus the label and the split key."""
    fields = set(ApplicationFeatures.model_fields)
    assert fields == set(CONTRACT) - set(EXCLUDED_FROM_REQUEST)
    assert "fraud_bool" not in fields, "a live application has no outcome"
    assert "month" not in fields
    assert "customer_age" in fields, "accepted for fairness logging, dropped before scoring"


def test_valid_payload_passes_validation() -> None:
    """The generated example is itself legal, or every test below is vacuous."""
    ApplicationFeatures(**valid_features())


@pytest.mark.parametrize(
    ("field", "value", "why"),
    [
        ("income", 9.0, "the income x 10 bug: outside the 0.1-0.9 decile grid"),
        ("customer_age", 35, "not one of the nine decades"),
        ("name_email_similarity", 1.5, "outside [0, 1]"),
        ("device_os", "haiku_os", "a category the contract never saw"),
        ("email_is_free", 7, "not a binary"),
    ],
)
def test_invalid_input_is_rejected(client, field: str, value: Any, why: str) -> None:
    """Invalid input never reaches the model: it is a 422 at the door."""
    features = valid_features()
    features[field] = value

    response = client.post("/score", json={"application_id": "t", "features": features})
    assert response.status_code == 422, why
    assert field in response.text


def test_unknown_fields_are_rejected(client) -> None:
    """extra="forbid": an unexpected field is a breach, not something to ignore."""
    features = valid_features()
    features["surprise_feature"] = 1

    response = client.post("/score", json={"application_id": "t", "features": features})
    assert response.status_code == 422


def test_a_missing_field_is_rejected(client) -> None:
    """A partial application cannot be scored."""
    features = valid_features()
    del features["velocity_6h"]

    response = client.post("/score", json={"application_id": "t", "features": features})
    assert response.status_code == 422


# --- endpoints that answer without a model ----------------------------------------


def test_health_is_always_available(client) -> None:
    """A health check must distinguish "no model" from "process down"."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_monitor_status_is_available(client) -> None:
    """The service always knows whether it is in fallback."""
    response = client.get("/monitor/status")
    assert response.status_code == 200
    assert "fallback_active" in response.json()


# --- scoring, which needs the trained artefacts -----------------------------------


@pytest.mark.data
def test_valid_input_scores(client) -> None:
    """200, with everything section 10 says the response carries."""
    if not has_model(client):
        pytest.skip("no trained artefacts: run `make train` and `make conformal`")

    response = client.post(
        "/score", json={"application_id": "demo-001", "features": valid_features()}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["application_id"] == "demo-001"
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["decision"] in {"approve", "review", "verify"}
    assert set(body["conformal_set"]) <= {"legit", "fraud"}
    assert body["model_version"]
    assert body["policy_version"].startswith("alpha_fraud=")
    assert "drift_status" in body
    assert isinstance(body["fallback_active"], bool)


@pytest.mark.data
def test_changing_age_does_not_change_the_score(client) -> None:
    """Rule 4, end to end: age is logged, never scored."""
    if not has_model(client):
        pytest.skip("no trained artefacts")

    scores = []
    for age in (20, 50, 90):
        features = valid_features() | {"customer_age": age}
        response = client.post("/score", json={"features": features})
        assert response.status_code == 200
        scores.append(response.json()["risk_score"])

    assert len(set(scores)) == 1, f"age changed the score: {scores}"


@pytest.mark.data
def test_reasons_are_analyst_facing_and_optional(client) -> None:
    """Reason codes are for analysts, and `explain=false` skips them for latency."""
    if not has_model(client):
        pytest.skip("no trained artefacts")

    payload = {"features": valid_features()}
    with_reasons = client.post("/score?explain=true", json=payload).json()
    without = client.post("/score?explain=false", json=payload).json()

    assert without["reasons"] == []
    assert without["risk_score"] == with_reasons["risk_score"], "explaining must not move the score"
    for reason in with_reasons["reasons"]:
        assert reason["direction"] in {"high", "low"}
        assert reason["text"]


@pytest.mark.data
def test_version_returns_the_manifest(client) -> None:
    """A reviewer must be able to ask the running service what it is."""
    if not has_model(client):
        pytest.skip("no trained artefacts")

    manifest = client.get("/version").json()
    assert manifest["model_version"]
    assert manifest["model"]["use_age"] is False
    assert "customer_age" not in manifest["features"]
    assert manifest["artefacts"]
