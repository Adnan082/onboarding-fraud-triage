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

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from omegaconf import DictConfig, OmegaConf

from triage.config import file_checksum, git_sha, run_context
from triage.features.encode import feature_columns, lgbm_frame
from triage.models.baselines import FittedModel, model_params
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


def tune(train: pd.DataFrame, valid: pd.DataFrame, cfg: DictConfig) -> dict[str, Any]:
    """Tune on months 0-3, score on month 4, at most 30 trials (section 8.1).

    Random search over a small grid, seeded. Random search rather than a smarter
    optimiser because with 30 trials the search budget is the binding constraint,
    not the search strategy, and a seeded random search is trivially reproducible.

    Month 4 is the validation month and months 5 to 7 are untouched: the tuning
    never sees the calibration month, let alone the test months. The caller refits
    on months 0-4 with whatever this returns.

    Returns the best parameters and every trial, so a reviewer can see the spread
    rather than take the winner on trust.
    """
    from triage.evaluation.metrics import tpr_at_fpr

    tuning = cfg.model.tuning
    max_trials = int(tuning.max_trials)
    target_fpr = float(cfg.data.protocol.paper.threshold_fpr)
    rng = np.random.default_rng(int(cfg.seed))

    base = model_params(cfg)
    grid: dict[str, list[Any]] = {
        "n_estimators": [200, 350, 500, 700, 1000],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.12],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [20, 50, 100, 200, 400],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    }

    valid_labels = valid[cfg.data.label].to_numpy()
    train_features = lgbm_frame(train, cfg, use_age=False)
    valid_features = lgbm_frame(valid, cfg, use_age=False)
    train_labels = train[cfg.data.label].to_numpy()

    def score_of(params: dict[str, Any]) -> float:
        model = LGBMClassifier(**params)
        model.fit(train_features, train_labels)
        scores = np.asarray(model.predict_proba(valid_features))[:, 1]
        return float(tpr_at_fpr(valid_labels, scores, target_fpr))

    seen: set[tuple] = set()
    trials: list[dict[str, Any]] = []

    # Trial -1 is the configured parameters. Without it "best of 30" says nothing
    # about whether the search was worth running: a search that loses to the
    # defaults is a finding, and this is how it becomes visible.
    incumbent = {key: base[key] for key in grid if key in base}
    incumbent_score = score_of(base)
    seen.add(tuple(sorted(incumbent.items())))
    trials.append({"trial": -1, "params": incumbent, str(tuning.metric): incumbent_score})
    log.info("  incumbent (configured params): %s = %.4f", tuning.metric, incumbent_score)

    for trial in range(max_trials):
        candidate = {name: rng.choice(values).item() for name, values in grid.items()}
        key = tuple(sorted(candidate.items()))
        if key in seen:
            continue  # a repeat would spend a trial saying nothing
        seen.add(key)

        score = score_of({**base, **candidate})
        trials.append({"trial": trial, "params": candidate, str(tuning.metric): score})
        log.info("  trial %2d: %s = %.4f", trial, tuning.metric, score)

    if not trials:
        raise ValueError("no tuning trials completed")

    best = max(trials, key=lambda row: row[str(tuning.metric)])
    if best["trial"] == -1:
        log.warning(
            "%d trials of random search did not beat the configured parameters "
            "(%s = %.4f): keeping them",
            len(trials) - 1,
            tuning.metric,
            incumbent_score,
        )
    log.info(
        "best of %d trials (and the incumbent): %s = %.4f with %s",
        len(trials) - 1,
        tuning.metric,
        best[str(tuning.metric)],
        best["params"],
    )
    return {
        "best_params": {**incumbent, **best["params"]},
        "best_score": best[str(tuning.metric)],
        "beat_incumbent": bool(best["trial"] != -1),
        "incumbent_score": incumbent_score,
        "metric": str(tuning.metric),
        "n_trials": len(trials) - 1,
        "validation_month": int(cfg.data.protocol.champion_tuning.valid_month),
        "trials": trials,
    }


def write_tuning_trials(path: Path, tuning: dict[str, Any]) -> None:
    """Write every trial to CSV, one row each, the incumbent as trial -1.

    Separate from ``reports/metrics.json`` on purpose. That file is the current
    state of the project, and the current state of an ordinary run is "not
    tuned", so a search recorded only there disappears the next time anyone
    trains. This is the durable record of what the configured parameters beat.
    """
    metric = str(tuning["metric"])
    trials = sorted(tuning["trials"], key=lambda row: int(row["trial"]))
    names = sorted({name for row in trials for name in row["params"]})

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "validation_month", metric, *names])
        for row in trials:
            writer.writerow(
                [
                    row["trial"],
                    tuning["validation_month"],
                    row[metric],
                    *(row["params"].get(name) for name in names),
                ]
            )
    log.info("wrote %s (%d trials, incumbent as trial -1)", path.name, len(trials) - 1)


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
