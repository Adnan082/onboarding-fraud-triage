"""Reason codes for analysts.

SHAP ``TreeExplainer`` on the UNCALIBRATED LightGBM margin (say so in the docs).
Top 3 features pushing the score towards fraud, rendered from
``configs/reasons.yaml`` by feature and direction ("high" or "low" relative to
the training median).

Analysts only. No demo text ever shows these to an applicant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Reason:
    """One rendered reason code."""

    feature: str
    direction: str
    text: str


def load_templates(path: Path) -> dict[str, dict[str, str]]:
    """Load the plain-English templates from ``configs/reasons.yaml``."""
    raise NotImplementedError("TODO(week 3): read yaml, validate keys against the feature list")


def top_reasons(model, row, medians: dict[str, float], templates: dict, k: int = 3) -> list[Reason]:
    """Top ``k`` features pushing this application towards fraud."""
    raise NotImplementedError("TODO(week 3): TreeExplainer on the raw margin, take the top k")
