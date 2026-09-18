"""Artefact loading and single-application scoring, kept out of the web layer.

Startup loads the artefacts once and verifies their hashes against
``models/manifest.json``; a mismatch must stop the service from starting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
MONITOR_STATE_PATH = MODELS_DIR / "monitor_state.json"


@dataclass
class ScoringService:
    """Holds the loaded model, calibrator, conformal thresholds and manifest."""

    manifest: dict

    @classmethod
    def load(cls, manifest_path: Path = MANIFEST_PATH) -> ScoringService:
        """Load artefacts and verify every hash against the manifest."""
        raise NotImplementedError("TODO(week 3): load artefacts, verify hashes, refuse on mismatch")

    def score(self, features: dict, *, explain: bool = True) -> dict:
        """Score one application. ``explain=False`` skips SHAP for latency tests.

        ``customer_age`` is logged for fairness monitoring and dropped before scoring:
        changing it must never change the score.
        """
        raise NotImplementedError("TODO(week 3): drop age, encode, score, calibrate, decide")


def read_manifest(path: Path = MANIFEST_PATH) -> dict | None:
    """Return the manifest, or ``None`` before ``make train`` has produced one."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_monitor_state(path: Path = MONITOR_STATE_PATH) -> dict:
    """Return the monitor state, defaulting to 'ok' before the monitor has run."""
    if not path.exists():
        return {"drift_status": "ok", "fallback_active": False}
    return json.loads(path.read_text(encoding="utf-8"))
