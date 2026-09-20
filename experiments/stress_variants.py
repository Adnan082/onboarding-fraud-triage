"""Stress test: score the Base-trained champion on months 6-7 of Variant IV and Variant V.

The variants are the same fraud problem under a different population mix. Base is
18.3% aged 50 or over; both variants are 50.6%. A model trained on one and scored
on the other is the cleanest available test of whether the monitor notices a
population it was not calibrated for -- without anyone having to invent a fault.

Alarms on the untouched months 6-7 of Base are **natural drift**, reported
separately and never called false alarms: that population really did move.

This module holds the variant selection so it is importable and testable on its
own. The scoring loop lives in ``triage.stages.monitor``, which already owns the
detectors, the calibrated thresholds and the alarm state -- duplicating it here
would create a second implementation that could silently disagree with the one
that produces the published numbers.
"""

from __future__ import annotations

import logging

import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger("triage")


def variant_names(cfg: DictConfig) -> list[str]:
    """The variants to stress test, from ``monitor.stress_variants``."""
    return [str(name) for name in cfg.monitor.stress_variants]


def held_out_months(cfg: DictConfig) -> list[int]:
    """The months to score. The same held-out months used everywhere else."""
    return [int(month) for month in cfg.data.protocol.deployment.test_months]


def select_test_window(frame: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """The rows of a variant that the stress test scores.

    Only the held-out months: scoring a variant's training months would compare a
    population the model learned from against one it did not, and confuse a
    difference in *population* with a difference in *time*.
    """
    months = held_out_months(cfg)
    selected = frame[frame[cfg.data.time_column].isin(months)]
    if selected.empty:
        raise ValueError(f"no rows for months {months}: cannot stress test this variant")
    return selected


def population_summary(frame: pd.DataFrame, cfg: DictConfig) -> dict[str, float]:
    """How this variant's population differs, for the report.

    Age appears here purely as measurement -- it is what makes Variants IV and V
    interesting -- and never reaches the model.
    """
    age_column = str(cfg.features.protected.age)
    cut = int(cfg.features.protected.age_cut)
    older = frame[age_column] >= cut
    label = str(cfg.data.label)

    return {
        "n": float(len(frame)),
        "share_older": float(older.mean()),
        "prevalence": float(frame[label].mean()),
        "prevalence_older": float(frame.loc[older, label].mean()) if older.any() else float("nan"),
        "prevalence_younger": (
            float(frame.loc[~older, label].mean()) if (~older).any() else float("nan")
        ),
    }
