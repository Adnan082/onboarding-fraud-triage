"""B0 and B1 under both protocols, with bootstrap CIs -> reports/metrics.json:baselines.

B0 is logistic regression; B1 is LightGBM on all features including
``customer_age``, which is what makes it comparable with the published BAF
results. Both run under the paper protocol (for that comparison) and the
deployment protocol (for everything else).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

from triage.config import with_model
from triage.data.load import load_interim
from triage.evaluation.artefacts import update_section
from triage.evaluation.protocols import run_deployment_protocol, run_paper_protocol
from triage.stages._base import CONFIG_PATH, start

log = logging.getLogger("triage")

# B0 and B1 from section 8.2. The champion is trained by `make train`, not here.
BASELINES = {"b0_logreg": "logreg", "b1_lgbm": "lgbm"}


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for ``make baseline``."""
    context = start(cfg, "baseline")

    frame = load_interim(cfg, "base")
    log.info("loaded base: %s rows, %s columns", f"{len(frame):,}", frame.shape[1])

    models: dict[str, dict] = {}
    for label, model_name in BASELINES.items():
        model_cfg = with_model(cfg, model_name)
        log.info("=== %s (%s) ===", label, model_name)

        started = time.perf_counter()
        models[label] = {
            "config": {"name": model_name, "use_age": bool(model_cfg.model.use_age)},
            "paper": run_paper_protocol(frame, model_cfg),
            "deployment": run_deployment_protocol(frame, model_cfg),
        }
        models[label]["seconds"] = time.perf_counter() - started
        log.info("%s done in %.1fs", label, models[label]["seconds"])

    payload = {
        "data": {
            "variant": "base",
            "rows": len(frame),
            "sample_frac": float(cfg.data.sample_frac),
            "prevalence": float(frame[cfg.data.label].mean()),
        },
        "target_fpr": float(cfg.data.protocol.paper.threshold_fpr),
        "bootstrap": {
            "n_resamples": int(cfg.bootstrap.n_resamples),
            "level": float(cfg.bootstrap.level),
        },
        "models": models,
    }
    update_section(Path(cfg.paths.metrics), "baselines", payload, context)
    log.info("wrote %s -> baselines", cfg.paths.metrics)


if __name__ == "__main__":
    main()
