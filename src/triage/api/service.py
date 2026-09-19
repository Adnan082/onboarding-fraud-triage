"""Artefact loading and single-application scoring, kept out of the web layer.

Startup loads the artefacts once and verifies their hashes against
``models/manifest.json``. A mismatch stops the service from starting: a model that
is not the model the manifest describes must never score anything, because every
claim in the validation report is attached to that manifest.

Scoring one application follows exactly the batch path -- sentinel flags, the same
feature columns, the same calibrator, the same conformal thresholds -- so the API
and the report cannot disagree about what the policy does.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from triage.config import CONFIG_DIR
from triage.explain.reasons import Reason, load_templates, top_reasons
from triage.features.encode import lgbm_frame
from triage.models.champion import MANIFEST_FILE, load
from triage.monitoring.fallback import read_state
from triage.policy.decide import decide
from triage.uncertainty.conformal import ConformalThresholds, predict_sets, set_labels

log = logging.getLogger("triage")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = MODELS_DIR / MANIFEST_FILE
MONITOR_STATE_PATH = MODELS_DIR / "monitor_state.json"
REASONS_PATH = CONFIG_DIR / "reasons.yaml"


def load_config() -> DictConfig:
    """The project config, composed without a Hydra run.

    ``paths.root`` defaults to ``${hydra:runtime.cwd}``, which only resolves inside
    a ``hydra.main`` app. The service is not one -- it is started by uvicorn -- so
    the root is pinned explicitly to the installed project directory.
    """
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=[f"paths.root={PROJECT_ROOT}"])


def _thresholds_from(cfg: DictConfig) -> ConformalThresholds:
    """The conformal thresholds the policy stage fitted on ``cal_conf``."""
    metrics_path = Path(cfg.paths.metrics)
    stored = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        stored = metrics.get("policy", {}).get("thresholds")

    if not stored:
        raise ValueError(
            "no conformal thresholds in reports/metrics.json. Run `make conformal` "
            "before serving: without them there is no policy to apply."
        )

    return ConformalThresholds(
        tau_fraud=float(stored["tau_fraud"]),
        tau_legit=float(stored["tau_legit"]),
        alpha_fraud=float(stored["alpha_fraud"]),
        alpha_legit=float(stored["alpha_legit"]),
        n_fraud=int(stored["n_fraud"]),
        n_legit=int(stored["n_legit"]),
    )


@dataclass
class ScoringService:
    """Holds the loaded model, calibrator, conformal thresholds and manifest."""

    cfg: DictConfig
    estimator: Any
    calibrator: Any
    manifest: dict[str, Any]
    thresholds: ConformalThresholds
    medians: dict[str, float] = field(default_factory=dict)
    templates: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, cfg: DictConfig | None = None) -> ScoringService:
        """Load artefacts and verify every hash against the manifest."""
        cfg = cfg or load_config()
        estimator, calibrator, manifest = load(cfg)
        thresholds = _thresholds_from(cfg)

        log.info(
            "loaded %s (calibration %s, policy %s)",
            manifest["model_version"],
            manifest["calibration"]["method"],
            thresholds.policy_version,
        )
        return cls(
            cfg=cfg,
            estimator=estimator,
            calibrator=calibrator,
            manifest=manifest,
            thresholds=thresholds,
            medians=manifest.get("training_medians", {}),
            templates=load_templates(REASONS_PATH),
        )

    def score(self, features: dict[str, Any], *, explain: bool = True) -> dict[str, Any]:
        """Score one application.

        ``explain=False`` skips SHAP, which is what the latency benchmark measures.
        ``customer_age`` is logged for fairness monitoring and dropped before
        scoring, so changing it cannot change the score.
        """
        frame = pd.DataFrame([features])
        prepared = lgbm_frame(frame, self.cfg, use_age=False)

        raw = float(self.estimator.predict_proba(prepared)[0, 1])
        probability = float(self.calibrator.transform(np.array([raw]))[0])

        sets = predict_sets(np.array([probability]), self.thresholds)
        has_legit, has_fraud = bool(sets[0, 0]), bool(sets[0, 1])

        reasons: list[Reason] = []
        if explain:
            reasons = top_reasons(self.estimator, prepared, self.medians, self.templates)

        state = read_state(MONITOR_STATE_PATH)
        return {
            "risk_score": probability,
            "decision": decide(has_legit, has_fraud),
            "conformal_set": set_labels(sets)[0],
            "reasons": [reason.to_dict() for reason in reasons],
            "model_version": self.manifest["model_version"],
            "policy_version": self.thresholds.policy_version,
            "drift_status": str(state.get("drift_status", "ok")),
            "fallback_active": bool(state.get("fallback_active", False)),
        }


def read_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any] | None:
    """Return the manifest, or ``None`` before ``make train`` has produced one."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_monitor_state(path: Path = MONITOR_STATE_PATH) -> dict[str, Any]:
    """Return the monitor state, defaulting to 'ok' before the monitor has run."""
    return read_state(path)
