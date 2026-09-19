"""On ALERT: switch to the fallback alpha preset and record it.

Appends to ``reports/monitor_events.jsonl`` and updates ``models/monitor_state.json``,
which the API reads on every request.

The fallback is deliberately blunt: tighten the alphas so more applications go to
a human. Drift means the coverage guarantee no longer holds, and the safe response
to "I no longer know how wrong I am" is to decide less automatically -- never to
decline automatically, which this system does not do at all (rule 5).

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
    """Read ``monitor_state.json``; return an 'ok' default when it is missing.

    A corrupt state file also reads as 'ok' with a note, rather than taking the
    service down: the API's job is to keep scoring, and a monitor that cannot be
    read is a monitoring problem, not a scoring one. It is logged loudly.
    """
    if not state_path.exists():
        return dict(OK_STATE)

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        log.error("could not read %s (%s): serving 'ok'", state_path, error)
        return {**OK_STATE, "reason": f"unreadable monitor state: {error}"}

    if not isinstance(state, dict) or "fallback_active" not in state:
        log.error("%s is not a valid monitor state: serving 'ok'", state_path)
        return {**OK_STATE, "reason": "invalid monitor state"}

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
