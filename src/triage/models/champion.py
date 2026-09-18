"""Champion: LightGBM without customer_age, lightly tuned (rule 4).

The fitting is the same code path as B1 -- the difference is entirely in the
config, which sets ``use_age: false``. That is deliberate: if the champion had its
own fitting code, "does it see age?" would be a question about two implementations
instead of one line of YAML.

Saving writes ``models/manifest.json``, recording a SHA-256 of every artefact
alongside the config hash, git SHA and the exact feature list. The API verifies
those hashes at startup and refuses to serve on a mismatch.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from triage.config import file_checksum, git_sha, run_context
from triage.features.encode import feature_columns
from triage.models.baselines import FittedModel
from triage.models.baselines import fit as fit_baseline
from triage.models.calibrate import Calibrator

log = logging.getLogger("triage")

MODEL_FILE = "champion.joblib"
CALIBRATOR_FILE = "calibrator.joblib"
MANIFEST_FILE = "manifest.json"


def fit(train: pd.DataFrame, cfg: DictConfig) -> FittedModel:
    """Fit the champion on the training months.

    Seed, ``deterministic`` and ``force_row_wise`` come from the config, so two
    runs of the same commit produce the same model.
    """
    if bool(cfg.model.use_age):
        raise ValueError(
            "the champion must not see customer_age (rule 4): this config has use_age: true"
        )
    return fit_baseline(train, cfg)


def tune(train: pd.DataFrame, valid: pd.DataFrame, cfg: DictConfig) -> dict:
    """Tune on months 0-3 against month 4, at most 30 trials, then refit on 0-4."""
    raise NotImplementedError("TODO(week 2): <= 30 trials, month 4 as validation")


def model_version() -> str:
    """What the API reports as ``model_version``."""
    return f"champion-{git_sha()}"


def save(
    model: FittedModel,
    calibrator: Calibrator,
    cfg: DictConfig,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the model, the calibrator and the manifest. Returns the manifest."""
    models_dir = Path(cfg.paths.models)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / MODEL_FILE
    calibrator_path = models_dir / CALIBRATOR_FILE
    joblib.dump(model.estimator, model_path)
    joblib.dump(calibrator, calibrator_path)

    manifest: dict[str, Any] = {
        "model_version": model_version(),
        "created_at": datetime.now(UTC).isoformat(),
        **run_context(cfg),
        "model": {
            "name": str(cfg.model.name),
            "kind": str(cfg.model.kind),
            "use_age": bool(cfg.model.use_age),
            "params": OmegaConf.to_container(cfg.model.params, resolve=True),
            "n_train": model.n_train,
            "n_train_fraud": model.n_train_fraud,
        },
        "calibration": {"method": calibrator.method},
        "features": feature_columns(cfg, use_age=bool(cfg.model.use_age)),
        "artefacts": {
            MODEL_FILE: file_checksum(model_path),
            CALIBRATOR_FILE: file_checksum(calibrator_path),
        },
        **(extra or {}),
    }

    manifest_path = models_dir / MANIFEST_FILE
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    log.info("wrote %s, %s and %s", model_path.name, calibrator_path.name, manifest_path.name)
    return manifest


def load_fitted(cfg: DictConfig) -> tuple[FittedModel, Calibrator, dict[str, Any]]:
    """Load the champion as a :class:`FittedModel`, ready to score a raw frame.

    Every stage after training goes through here, so they all score with the same
    artefacts the API serves -- and all get the hash check for free.
    """
    estimator, calibrator, manifest = load(cfg)
    model = FittedModel(
        name=str(manifest["model"]["name"]),
        kind="lgbm",
        estimator=estimator,
        use_age=bool(manifest["model"]["use_age"]),
        cfg=cfg,
        n_train=int(manifest["model"]["n_train"]),
        n_train_fraud=int(manifest["model"]["n_train_fraud"]),
    )

    expected = feature_columns(cfg, use_age=model.use_age)
    if list(manifest["features"]) != expected:
        raise ValueError(
            "the manifest's feature list does not match the current config. "
            "Re-run `make train` before using this model."
        )
    return model, calibrator, manifest


def load(cfg: DictConfig) -> tuple[Any, Calibrator, dict[str, Any]]:
    """Load the artefacts and verify them against the manifest.

    Raises if a hash does not match: a model that is not the one the manifest
    describes must not score anything.
    """
    models_dir = Path(cfg.paths.models)
    manifest = json.loads((models_dir / MANIFEST_FILE).read_text(encoding="utf-8"))

    for name, expected in manifest["artefacts"].items():
        actual = file_checksum(models_dir / name)
        if actual != expected:
            raise ValueError(
                f"{name} does not match the manifest "
                f"(expected {expected[:12]}..., found {actual[:12]}...). "
                "Re-run `make train` rather than serving an unknown model."
            )

    estimator = joblib.load(models_dir / MODEL_FILE)
    calibrator = joblib.load(models_dir / CALIBRATOR_FILE)
    return estimator, calibrator, manifest
