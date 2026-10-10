"""Version the complete serving policy alongside the fitted model.

Only the calibration stage publishes this bundle. Serving never fits thresholds
or reads decision policy from mutable reporting artifacts.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from triage.config import CONFIG_DIR, file_checksum
from triage.data.contract import CONTRACT
from triage.uncertainty.conformal import ConformalThresholds

POLICY_FILE = "serving_policy.json"


def serving_context(cfg: DictConfig) -> dict[str, Any]:
    """Capture the contract, feature config and explanation templates in use."""
    return {
        "contract": json.loads(json.dumps({k: asdict(v) for k, v in CONTRACT.items()})),
        "features": OmegaConf.to_container(cfg.features, resolve=True),
        "reasons_sha256": file_checksum(CONFIG_DIR / "reasons.yaml"),
    }


def publish(
    cfg: DictConfig,
    primary: ConformalThresholds,
    fallback: ConformalThresholds,
) -> None:
    """Publish both predeclared policies and bind them to the model manifest."""
    folder = Path(cfg.paths.models)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    policy = {
        "schema_version": 1,
        "model_version": manifest["model_version"],
        "primary": primary.to_dict(),
        "fallback": fallback.to_dict(),
        "fitted_on": "cal_conf",
        "context": serving_context(cfg),
    }
    path = folder / POLICY_FILE
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(policy, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)
    manifest["artefacts"][POLICY_FILE] = file_checksum(path)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, manifest_path)


def load(
    cfg: DictConfig, manifest: dict[str, Any]
) -> tuple[ConformalThresholds, ConformalThresholds]:
    """Reject incomplete, modified, or context-incompatible serving bundles."""
    path = Path(cfg.paths.models) / POLICY_FILE
    expected = manifest.get("artefacts", {}).get(POLICY_FILE)
    if not expected or not path.exists() or file_checksum(path) != expected:
        raise ValueError("No verified serving policy. Run make conformal before serving.")
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1 or policy.get("fitted_on") != "cal_conf":
        raise ValueError("Unsupported serving policy schema or calibration split.")
    if policy["model_version"] != manifest["model_version"]:
        raise ValueError("Serving policy belongs to a different model.")
    if policy["context"] != serving_context(cfg):
        raise ValueError("Serving contract, feature config or explanation templates changed.")

    def thresholds(name: str) -> ConformalThresholds:
        fields = dict(policy[name])
        fields.pop("policy_version", None)
        return ConformalThresholds(**fields)

    return thresholds("primary"), thresholds("fallback")
