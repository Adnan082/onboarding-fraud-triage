"""WATCH / ALERT logic over detector outputs.

WATCH: any detector exceeds its threshold in a window.
ALERT: the same detector exceeds its threshold in 2 consecutive windows, or the
score PSI is above 0.25.

Thresholds come from 200 bootstrap clean windows drawn from ``cal_prob + cal_tune``,
at the 99th percentile. PSI is also reported against the 0.10 / 0.25 rule of thumb.

Why an empirical threshold rather than the rule of thumb alone: a PSI of 0.1 means
different things for a stable column and a noisy one, and at a window size of
4,000 some columns wander on their own. Calibrating on windows that are known to
be clean says what "normal" looks like *for this data at this window size*, and
the 99th percentile fixes the false-alarm rate at roughly one window in a hundred
by construction.

The two-in-a-row rule is what separates a blip from a break. A single window over
threshold is common; the same detector twice running is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Level = Literal["ok", "watch", "alert"]

OK: Level = "ok"
WATCH: Level = "watch"
ALERT: Level = "alert"

# The detector whose rule-of-thumb breach alerts on its own.
SCORE_PSI = "psi_score"
SCORE_PSI_ALERT = 0.25


@dataclass(frozen=True)
class WindowResult:
    """One window's detector values and the level they imply."""

    window_id: int
    month: int
    detectors: dict[str, float]
    level: Level
    exceeded: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """JSON-safe, for ``reports/monitoring.json``."""
        return {
            "window_id": self.window_id,
            "month": self.month,
            "detectors": self.detectors,
            "level": self.level,
            "exceeded": list(self.exceeded),
            "reasons": list(self.reasons),
        }


def calibrate_thresholds(
    clean_windows: list[dict[str, float]], percentile: float = 99
) -> dict[str, float]:
    """The given percentile of each detector over windows known to be clean.

    Detectors not present in every window are still calibrated, on whatever
    windows do report them.
    """
    if not clean_windows:
        raise ValueError("no clean windows to calibrate on")

    names = sorted({name for window in clean_windows for name in window})
    thresholds = {}
    for name in names:
        values = [window[name] for window in clean_windows if name in window]
        finite = [value for value in values if np.isfinite(value)]
        if finite:
            thresholds[name] = float(np.percentile(finite, percentile))
    return thresholds


def exceeded_detectors(values: dict[str, float], thresholds: dict[str, float]) -> tuple[str, ...]:
    """Which detectors are over their calibrated threshold in this window."""
    return tuple(
        name
        for name, value in sorted(values.items())
        if name in thresholds and np.isfinite(value) and value > thresholds[name]
    )


def evaluate_window(
    values: dict[str, float],
    thresholds: dict[str, float],
    history: list[WindowResult] | None = None,
) -> WindowResult:
    """Apply the WATCH/ALERT rules to one window.

    ``history`` is the windows already seen, most recent last. Only the previous
    one matters, but the whole list is accepted so callers can just pass what they
    have.
    """
    exceeded = exceeded_detectors(values, thresholds)
    reasons: list[str] = []
    level: Level = OK

    if exceeded:
        level = WATCH
        reasons.append(f"over threshold: {', '.join(exceeded)}")

    # The same detector twice running is a break, not a blip.
    previous = history[-1] if history else None
    if previous is not None:
        repeated = sorted(set(exceeded) & set(previous.exceeded))
        if repeated:
            level = ALERT
            reasons.append(f"two consecutive windows: {', '.join(repeated)}")

    # A large enough score PSI alerts on its own, without waiting for a second window.
    score_psi = values.get(SCORE_PSI)
    if score_psi is not None and np.isfinite(score_psi) and score_psi > SCORE_PSI_ALERT:
        level = ALERT
        reasons.append(f"{SCORE_PSI} {score_psi:.3f} above the {SCORE_PSI_ALERT} rule of thumb")

    return WindowResult(
        window_id=int(values.get("window_id", -1)),
        month=int(values.get("month", -1)),
        detectors={k: v for k, v in values.items() if k not in ("window_id", "month")},
        level=level,
        exceeded=exceeded,
        reasons=tuple(reasons),
    )


@dataclass
class AlarmRun:
    """A sequence of windows scored against one set of thresholds."""

    thresholds: dict[str, float]
    results: list[WindowResult] = field(default_factory=list)

    def add(self, values: dict[str, float]) -> WindowResult:
        """Score the next window, given everything before it."""
        result = evaluate_window(values, self.thresholds, self.results)
        self.results.append(result)
        return result

    def first_alert(self) -> WindowResult | None:
        """The first window that reached ALERT, if any."""
        return next((result for result in self.results if result.level == ALERT), None)

    def counts(self) -> dict[str, int]:
        """How many windows landed at each level."""
        return {
            level: sum(1 for result in self.results if result.level == level)
            for level in (OK, WATCH, ALERT)
        }
