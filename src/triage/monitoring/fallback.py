"""On ALERT: switch to the fallback alpha preset and record it.

Appends to ``reports/monitor_events.jsonl`` and updates ``models/monitor_state.json``,
which the API reads on every request.

The fallback is a separately configured policy, fitted on the calibration split.
Its workload must be measured: the configured alphas do not uniformly tighten
both classes. Drift invalidates the exchangeability assumption; switching policy
does not restore a coverage guarantee. There is no automatic decline (rule 5).

The state file is written atomically, via a temporary file and a rename, because
the API may read it at any moment and a half-written state is worse than a stale
one.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("triage")

OK_STATE: dict[str, Any] = {"drift_status": "ok", "fallback_active": False}


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)  # atomic on POSIX and Windows


def append_event(event_log: Path, event: dict[str, Any]) -> None:
    """Append one line to the monitor's event log. The log is never rewritten."""
    event_log.parent.mkdir(parents=True, exist_ok=True)
    with event_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**event, "at": datetime.now(UTC).isoformat()}) + "\n")


def activate(
    reason: str,
    state_path: Path,
    event_log: Path,
    preset: dict[str, Any],
    *,
    window_id: int | None = None,
    month: int | None = None,
) -> dict[str, Any]:
    """Turn the fallback on, log the event, and write the state the API serves."""
    state = {
        "drift_status": "alert",
        "fallback_active": True,
        "reason": reason,
        "alpha_fraud": float(preset["alpha_fraud"]),
        "alpha_legit": float(preset["alpha_legit"]),
        "last_window": window_id,
        "month": month,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _write_atomic(state_path, state)
    append_event(
        event_log,
        {"event": "fallback_activated", "reason": reason, "window_id": window_id, "month": month},
    )
    log.warning("fallback activated: %s", reason)
    return state


def clear(state_path: Path, event_log: Path, *, reason: str = "manual reset") -> dict[str, Any]:
    """Return to normal operation. Only ever done deliberately, never automatically.

    An alarm that clears itself the moment a window looks calm would flap, and a
    reviewer would never see what happened.
    """
    state = {**OK_STATE, "reason": reason, "updated_at": datetime.now(UTC).isoformat()}
    _write_atomic(state_path, state)
    append_event(event_log, {"event": "fallback_cleared", "reason": reason})
    log.info("fallback cleared: %s", reason)
    return state


def read_state(state_path: Path) -> dict[str, Any]:
    """Report unavailable monitoring explicitly; never silently default to healthy.

    Optional TRIAGE_MONITOR_MAX_AGE_SECONDS applies an operator-chosen wall-clock
    freshness budget. It is unset by default because BAF windows are application
    counts rather than real elapsed days. A latched alert never expires to OK.
    """

    def unknown(reason: str) -> dict[str, Any]:
        log.error("monitor unavailable: %s", reason)
        return {"drift_status": "unknown", "fallback_active": False, "reason": reason}

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        return unknown(f"unreadable monitor state: {error}")
    if (
        not isinstance(state, dict)
        or type(state.get("fallback_active")) is not bool
        or state.get("drift_status") not in ("ok", "watch", "alert")
        or (state["drift_status"] == "alert" and not state["fallback_active"])
    ):
        return unknown("invalid monitor state")
    max_age = os.environ.get("TRIAGE_MONITOR_MAX_AGE_SECONDS")
    if max_age is not None and not state["fallback_active"]:
        try:
            budget = float(max_age)
            updated = datetime.fromisoformat(state["updated_at"])
            age = (datetime.now(UTC) - updated).total_seconds()
            if not 0 < budget < float("inf") or age < 0 or age > budget:
                return unknown("stale monitor state or invalid freshness budget")
        except (ValueError, TypeError, KeyError):
            return unknown("invalid monitor timestamp or freshness budget")
    return state


def read_events(event_log: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read the event log, newest last. Malformed lines are skipped, not fatal."""
    if not event_log.exists():
        return []

    events = []
    for line in event_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("skipping a malformed line in %s", event_log)
    return events[-limit:] if limit else events
