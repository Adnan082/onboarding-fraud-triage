"""Config and run-context helpers shared by every stage."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# Excluded from the config hash: these hold absolute paths, which differ between
# machines -- and between Windows and WSL on the same machine. A provenance key
# that changes when the checkout moves cannot tell you whether two runs used the
# same settings, which is the only thing it is for.
MACHINE_SPECIFIC = ("paths", "mlflow")


def config_hash(cfg: DictConfig) -> str:
    """Return a stable 8-character hash of the experiment settings.

    The config is hashed **as authored**, not as resolved: interpolations such as
    ``${paths.models}`` are left as text. That has two consequences, both wanted:

    - the hash is identical on any machine and from any checkout location, so it
      can actually serve as the provenance key stamped on every artefact;
    - it can be computed outside a Hydra run. Resolving instead would need
      ``HydraConfig`` to be set, which it is not under a plain ``compose()``, in a
      test, or in a notebook.

    A change that affects results -- a different seed, alpha, or override -- still
    changes the hash, because those are stored values rather than interpolations.
    """
    container = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(container, dict):
        container = {key: value for key, value in container.items() if key not in MACHINE_SPECIFIC}
    blob = json.dumps(container, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def git_sha(short: bool = True) -> str:
    """Return the current git SHA, or ``"unknown"`` outside a repository."""
    args = ["git", "rev-parse", "--short=8", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def file_checksum(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_context(cfg: DictConfig) -> dict[str, Any]:
    """Provenance stamped onto every artefact: config hash, git SHA, seed, sampling.

    ``sample_frac`` travels with the context so that artefact writing can refuse a
    development run outright, rather than trusting everyone to remember.
    """
    return {
        "config_hash": config_hash(cfg),
        "git_sha": git_sha(),
        "seed": int(cfg.seed),
        "sample_frac": float(cfg.data.sample_frac),
    }


def with_model(cfg: DictConfig, name: str) -> DictConfig:
    """Return ``cfg`` with a different model config composed in.

    A stage that compares models -- the baselines, the fairness experiments --
    needs several of them in one run, which Hydra's defaults list cannot give.
    Merging keeps interpolations such as ``${seed}`` working.
    """
    path = CONFIG_DIR / "model" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no model config named {name!r} at {path}")

    # Replace the node rather than merging into it: the models have different
    # parameter sets, so merging logreg's class_weight into lgbm's keys fails.
    merged = copy.deepcopy(cfg)
    OmegaConf.set_struct(merged, False)
    merged["model"] = OmegaConf.load(path)
    OmegaConf.set_struct(merged, True)
    return cast(DictConfig, merged)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with full precision; rounding happens in prose, not in artefacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
