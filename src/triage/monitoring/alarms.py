"""WATCH / ALERT logic over detector outputs.

WATCH: any detector exceeds its threshold in a window.
ALERT: the same detector exceeds its threshold in 2 consecutive windows, or the
score PSI is above 0.25.

Thresholds come from 200 bootstrap clean windows drawn from ``cal_prob + cal_tune``,
at the 99th percentile. PSI is also reported against the 0.10 / 0.25 rule of thumb.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Level = Literal["ok", "watch", "alert"]


@dataclass(frozen=True)
class WindowResult:
    """One window's detector values and the level they imply."""

    window_id: int
    month: int
    detectors: dict[str, float]
    level: Level


def calibrate_thresholds(
    clean_windows: list[dict[str, float]], percentile: float = 99
) -> dict[str, float]:
    """99th percentile of each detector over the clean windows."""
    raise NotImplementedError("TODO(week 2): per-detector percentile")


def evaluate_window(
    values: dict[str, float], thresholds: dict[str, float], history: list[dict]
) -> Level:
    """Apply the WATCH/ALERT rules to one window."""
    raise NotImplementedError("TODO(week 2): watch/alert rules incl. 2-in-a-row and score PSI")
