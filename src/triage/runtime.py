"""Shared plumbing for anything runnable as a Hydra app.

Every stage runs as ``uv run python -m triage.stages.<name> [overrides]``, and
``triage.data.load`` / ``triage.data.contract`` run the same way. They all seed
from one place, log the same provenance line, warn loudly on a sampled run, and
track themselves to MLflow.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from omegaconf import DictConfig, OmegaConf

from triage.config import CONFIG_DIR, run_context
from triage.seeding import seed_everything
from triage.tracking import track

CONFIG_PATH = str(CONFIG_DIR)
log = logging.getLogger("triage")


def start(cfg: DictConfig, stage: str) -> dict[str, Any]:
    """Seed everything, log the resolved config, and return the run context."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_everything(int(cfg.seed))
    context = run_context(cfg)
    log.info(
        "stage=%s config_hash=%s git_sha=%s seed=%s",
        stage,
        context["config_hash"],
        context["git_sha"],
        context["seed"],
    )
    if float(cfg.data.sample_frac) < 1.0:
        log.warning(
            "data.sample_frac=%s: development run. Writing to reports/ will be refused.",
            cfg.data.sample_frac,
        )
    log.debug("resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    return context


@contextmanager
def stage_run(cfg: DictConfig, stage: str) -> Iterator[dict[str, Any]]:
    """Everything a stage needs around its body: seeding, provenance, MLflow.

    Using this rather than calling :func:`start` directly means a stage cannot
    forget to log itself. Section 4 asks for every stage to be tracked, and
    "remember to add it" is not a mechanism.
    """
    context = start(cfg, stage)
    with track(cfg, stage, context):
        yield context
