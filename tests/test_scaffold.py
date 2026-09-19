"""The scaffold itself: layout, config composition and the endpoints that work today.

These run without the Kaggle data and without any trained model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hydra import compose, initialize_config_dir

from triage.api.app import app
from triage.config import config_hash


@pytest.mark.parametrize(
    "relative",
    [
        "Makefile",
        "pyproject.toml",
        "CLAUDE.md",
        ".pre-commit-config.yaml",
        ".github/workflows/ci.yml",
        "configs/config.yaml",
        "configs/reasons.yaml",
        "docker/Dockerfile",
        "docs/PROGRESS.md",
        "docs/DECISIONS.md",
        "app/demo.py",
        "src/triage/uncertainty/conformal.py",
        "src/triage/monitoring/psi.sql",
    ],
)
def test_layout(repo_root: Path, relative: str) -> None:
    """Every path CLAUDE.md section 6 names is present."""
    assert (repo_root / relative).exists(), f"missing: {relative}"


def test_config_composes(repo_root: Path) -> None:
    """The Hydra defaults list resolves, and the rules encoded in it hold."""
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        cfg = compose(config_name="config")

    assert cfg.seed > 0
    assert cfg.features.use_age is False, "rule 4: the champion must not see customer_age"
    assert cfg.model.use_age is False, "the champion config must keep age out"
    assert cfg.data.window_size == 4000
    assert cfg.data.protocol.deployment.train_months == [0, 1, 2, 3, 4]
    assert cfg.data.protocol.deployment.test_months == [6, 7]
    assert "decline" not in str(cfg.policy.decisions).lower(), "rule 5: never decline"


def test_train_and_test_months_never_overlap(repo_root: Path) -> None:
    """Rule 1: split by time only, with no month in both sides of a protocol."""
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        cfg = compose(config_name="config")

    for name in ("paper", "deployment"):
        protocol = cfg.data.protocol[name]
        train = set(protocol.train_months)
        test = set(protocol.test_months)
        assert not train & test, f"{name}: months overlap"
        if name == "deployment":
            assert protocol.calibration_month not in train | test


def test_data_is_git_ignored(repo_root: Path) -> None:
    """Rule 3: row-level data must never reach git."""
    ignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/*", "mlruns/", "kaggle.json", ".env", "*.parquet"):
        assert pattern in ignore, f"missing .gitignore entry: {pattern}"


def test_health_endpoint() -> None:
    """``/health`` works before any model exists."""
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_monitor_status_answers() -> None:
    """``/monitor/status`` always answers, whatever the monitor has or has not done.

    It deliberately does not assert *which* state: once the monitor has run and
    found drift, `fallback_active` is true, and the service reporting that
    faithfully is the point. The default-when-missing behaviour is covered in
    `test_monitoring.py`, where the state file can be controlled.
    """
    with TestClient(app) as client:
        response = client.get("/monitor/status")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["fallback_active"], bool)
    assert body["drift_status"] in {"ok", "watch", "alert"}


def test_config_hash_survives_moving_the_checkout(repo_root: Path) -> None:
    """The provenance key must not change just because the repo moved.

    It changed between Windows and WSL on the same machine before `paths` and
    `mlflow` were excluded, which would have made every artefact look like it came
    from different settings.
    """
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        cfg = compose(config_name="config")

    original = config_hash(cfg)
    cfg.paths.root = "/somewhere/else/entirely"
    cfg.mlflow.tracking_uri = "file:/somewhere/else/entirely/mlruns"
    assert config_hash(cfg) == original


def test_config_hash_still_notices_a_real_change(repo_root: Path) -> None:
    """It must still change when a setting that affects results changes."""
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        cfg = compose(config_name="config")

    original = config_hash(cfg)
    cfg.seed = int(cfg.seed) + 1
    assert config_hash(cfg) != original
