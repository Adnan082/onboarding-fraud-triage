"""MLflow logging, to a local file store in ``mlruns/`` (CLAUDE.md section 4).

Every stage logs its params, metrics, config hash, data checksums and git SHA.
This sits *alongside* ``reports/metrics.json`` rather than replacing it, and the
two answer different questions:

- ``metrics.json`` is the **current** state of the project. It is committed, it is
  what the README and the validation report read, and there is exactly one of it.
- MLflow is the **history**. Every run ever made, including the ones that were
  wrong, the sampled development runs, and the experiments that lost. It is
  git-ignored, because that history is local working state rather than a claim.

Keeping both means a reviewer can see what is true now *and* how many attempts it
took, which is a fairer picture than either alone.

Logging never fails a stage. A tracking server that is unavailable is an
inconvenience; losing a completed training run because the logger threw is not.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger("triage")

# Values MLflow cannot store as a metric.
SKIP_KEYS = ("context", "note", "grid", "windows", "months", "experiments")


def _flatten(payload: Any, prefix: str = "", depth: int = 0) -> dict[str, float]:
    """Pull scalar metrics out of a nested payload, up to a sane depth."""
    if depth > 3:
        return {}

    flat: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in SKIP_KEYS:
                continue
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (bool, int, float)):
                flat[name] = float(value)
            elif isinstance(value, dict):
                flat.update(_flatten(value, name, depth + 1))
    return flat


def data_checksums(cfg: DictConfig) -> dict[str, str]:
    """The recorded checksums of the raw files, so a run is tied to its data."""
    path = Path(cfg.paths.raw).parent / "checksums.sha256"
    if not path.exists():
        return {}

    from triage.data.load import read_checksums

    return {name: digest[:12] for name, digest in read_checksums(path).items()}


@contextmanager
def track(cfg: DictConfig, stage: str, context: dict[str, Any]) -> Iterator[Any]:
    """Log one stage run to ``mlruns/``. Yields the run, or ``None`` if unavailable.

    Wrap a stage body in this. Anything it raises is logged and swallowed: the
    stage's own artefacts are the deliverable, and tracking is a convenience on
    top of them.
    """
    try:
        import mlflow
    except ImportError:  # pragma: no cover - mlflow is a declared dependency
        log.warning("mlflow is not installed: skipping run tracking")
        yield None
        return

    try:
        mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
        mlflow.set_experiment(str(cfg.mlflow.experiment))
        run = mlflow.start_run(run_name=stage)
    except Exception as error:  # noqa: BLE001 - never fail a stage for logging
        log.warning("could not start mlflow run (%s): continuing untracked", error)
        yield None
        return

    try:
        mlflow.set_tags(
            {
                "stage": stage,
                "git_sha": context.get("git_sha", "unknown"),
                "config_hash": context.get("config_hash", "unknown"),
                # A sampled run is development: tag it so it can never be mistaken
                # for a reportable one when browsing history.
                "sampled": str(float(context.get("sample_frac", 1.0)) < 1.0),
            }
        )
        params = OmegaConf.to_container(cfg, resolve=False)
        if isinstance(params, dict):
            for section in ("data", "features", "model", "calibration", "policy", "monitor"):
                if section in params:
                    mlflow.log_params(
                        {f"{section}.{k}": v for k, v in _params_of(params[section]).items()}
                    )
        mlflow.log_params({f"checksum.{k}": v for k, v in data_checksums(cfg).items()})
        yield run
    finally:
        try:
            mlflow.end_run()
        except Exception as error:  # noqa: BLE001
            log.warning("could not close the mlflow run: %s", error)


def _params_of(section: Any) -> dict[str, Any]:
    """Top-level scalars of a config section, which is what a param should be."""
    if not isinstance(section, dict):
        return {}
    return {
        key: value
        for key, value in section.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def log_metrics(payload: dict[str, Any], prefix: str = "") -> None:
    """Log whatever scalars a stage produced. Never raises."""
    try:
        import mlflow

        if mlflow.active_run() is None:
            return
        metrics = _flatten(payload, prefix)
        if metrics:
            mlflow.log_metrics(metrics)
    except Exception as error:  # noqa: BLE001
        log.warning("could not log metrics to mlflow: %s", error)


def log_artefact(path: Path) -> None:
    """Attach a generated file to the run. Never raises."""
    try:
        import mlflow

        if mlflow.active_run() is None or not path.exists():
            return
        mlflow.log_artifact(str(path))
    except Exception as error:  # noqa: BLE001
        log.warning("could not log %s to mlflow: %s", path.name, error)
