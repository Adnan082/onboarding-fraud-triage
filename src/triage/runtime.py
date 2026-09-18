"""Shared plumbing for anything runnable as a Hydra app.

Every stage runs as ``uv run python -m triage.stages.<name> [overrides]``, and
``triage.data.load`` / ``triage.data.contract`` run the same way. They all seed
from one place, log the same provenance line, and warn loudly on a sampled run.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig, OmegaConf

from triage.config import CONFIG_DIR, run_context
from triage.seeding import seed_everything

CONFIG_PATH = str(CONFIG_DIR)
log = logging.getLogger("triage")


def start(cfg: DictConfig, stage: str) -> dict[str, Any]:
    """Seed everything, log the resolved config, and return the run context."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_everything(int(cfg.seed))
    context = run_context(cfg)
    log.info("stage=%s config_hash=%s git_sha=%s seed=%s", stage, *context.values())
    if float(cfg.data.sample_frac) < 1.0:
        log.warning(
            "data.sample_frac=%s: development run, never report these numbers.",
            cfg.data.sample_frac,
        )
    log.debug("resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    return context
