"""Mitigation experiments M1-M4, bootstrap CIs and the trade-off plot."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from triage.stages._base import CONFIG_PATH, start


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for ``make fairness``."""
    start(cfg, "fairness")
    raise NotImplementedError("TODO(week 2): run M1-M4 and plot TPR@5%FPR against the FPR ratio")


if __name__ == "__main__":
    main()
