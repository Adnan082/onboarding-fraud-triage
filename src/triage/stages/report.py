"""Regenerate README tables and reports/figures/ from artefacts only.

Rule 6 in one place: this stage reads ``reports/metrics.json`` and writes the
README's marker-delimited tables. It never computes a metric, so a number can only
appear in the README if some other stage generated it first.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from triage.evaluation.artefacts import read_metrics, render_readme_tables
from triage.evaluation.tables import render_all
from triage.evaluation.validation import collect_findings
from triage.evaluation.validation import render as render_validation
from triage.stages._base import CONFIG_PATH, start

log = logging.getLogger("triage")


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for ``make report``."""
    start(cfg, "report")

    metrics_path = Path(cfg.paths.metrics)
    metrics = read_metrics(metrics_path)
    log.info("read %s with sections: %s", metrics_path, sorted(metrics))

    tables = render_all(metrics)
    readme = Path(cfg.paths.root) / "README.md"
    replaced = render_readme_tables(readme, tables)

    for name in sorted(tables):
        status = "rendered" if name in replaced else "no marker in README"
        log.info("  %-16s %s", name, status)
    log.info("updated %s (%d table(s))", readme, len(replaced))

    # The validation report is generated from the same artefacts, for the same
    # reason: a report that drifts from the numbers it describes is worse than none.
    report_path = Path(cfg.paths.reports) / "validation_report.md"
    report_path.write_text(render_validation(metrics), encoding="utf-8")
    findings = collect_findings(metrics)
    log.info(
        "wrote %s (%d finding(s): %s)",
        report_path,
        len(findings),
        ", ".join(sorted({f["rating"] for f in findings})),
    )


if __name__ == "__main__":
    main()
