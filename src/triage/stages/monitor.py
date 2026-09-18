"""Clean windows, variant stress tests and injected bugs -> reports/monitoring.json."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from triage.stages._base import CONFIG_PATH, start


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for ``make monitor``."""
    start(cfg, "monitor")
    raise NotImplementedError(
        "TODO(week 2): calibrate detectors, run windows, record detection delay"
    )


if __name__ == "__main__":
    main()
