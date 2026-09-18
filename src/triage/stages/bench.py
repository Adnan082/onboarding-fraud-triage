"""Latency benchmark -> reports/metrics.json:service."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from triage.stages._base import CONFIG_PATH, start


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for ``make bench``."""
    start(cfg, "bench")
    raise NotImplementedError(
        "TODO(week 3): 100 warm-up then 2,000 sequential requests, p50/p99, image size"
    )


if __name__ == "__main__":
    main()
