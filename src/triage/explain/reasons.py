"""Reason codes for analysts.

SHAP ``TreeExplainer`` on the **uncalibrated** LightGBM margin. That distinction
matters and is stated wherever these appear: calibration is a monotone transform
of the score, so it cannot change the *ranking* of contributions, but the
attributions themselves are in margin units, not probability units. A reason code
says which features pushed this application towards fraud, not by how many
percentage points.

Top 3 features pushing towards fraud, rendered from ``configs/reasons.yaml`` by
feature and direction -- "high" or "low" relative to the **training** median, so
the comparison is against the population the model learned from rather than
against whatever else happens to be in today's batch.

Analysts only. No demo text ever shows these to an applicant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("triage")

TOP_K = 3
HIGH, LOW = "high", "low"


@dataclass(frozen=True)
class Reason:
    """One rendered reason code."""

    feature: str
    direction: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"feature": self.feature, "direction": self.direction, "text": self.text}


def load_templates(path: Path) -> dict[str, dict[str, str]]:
    """Load the plain-English templates from ``configs/reasons.yaml``."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    templates = (loaded or {}).get("reasons", {})
    if not templates:
        raise ValueError(f"{path} has no `reasons:` section")
    return templates


def missing_templates(templates: dict[str, dict[str, str]], features: list[str]) -> list[str]:
    """Features a model uses that have no analyst-facing wording.

    Not fatal -- a feature without a template simply cannot be reported, and the
    next one is used instead -- but worth knowing, because it silently limits what
    an analyst can be told.
    """
    return sorted(
        feature
        for feature in features
        if feature not in templates and not feature.endswith(("_missing", "_negative"))
    )


def direction_for(feature: str, value: Any, medians: dict[str, float]) -> str:
    """Whether this application sits above or below the training median."""
    median = medians.get(feature)
    if median is None or value is None or (isinstance(value, float) and np.isnan(value)):
        return HIGH  # no basis to say otherwise; the template must still make sense
    try:
        return HIGH if float(value) >= float(median) else LOW
    except (TypeError, ValueError):
        return HIGH  # a category has no median: the template is keyed on "high"


def top_reasons(
    model: Any,
    row: pd.DataFrame,
    medians: dict[str, float],
    templates: dict[str, dict[str, str]],
    k: int = TOP_K,
) -> list[Reason]:
    """The ``k`` features pushing this application hardest towards fraud.

    ``row`` is a one-row model-ready frame. Only positive contributions are
    reported: an analyst wants to know why this case was flagged, not why it
    nearly was not.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(row)

    # LightGBM binary returns either (n, features) or a list per class.
    contributions = np.asarray(values[1] if isinstance(values, list) else values).reshape(-1)
    features = list(row.columns)

    order = np.argsort(contributions)[::-1]  # most fraud-ward first
    reasons: list[Reason] = []
    for index in order:
        if len(reasons) >= k or contributions[index] <= 0:
            break
        feature = features[index]
        if feature not in templates:
            continue
        direction = direction_for(feature, row.iloc[0, index], medians)
        text = templates[feature].get(direction)
        if text:
            reasons.append(Reason(feature=feature, direction=direction, text=text))

    return reasons


def training_medians(frame: pd.DataFrame) -> dict[str, float]:
    """Median of each numeric feature on the training months, for direction."""
    numeric = frame.select_dtypes("number")
    return {column: float(numeric[column].median()) for column in numeric.columns}
