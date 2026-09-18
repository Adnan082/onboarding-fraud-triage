"""On ALERT: switch to the fallback alpha preset and record it.

Appends to ``reports/monitor_events.jsonl`` and updates ``models/monitor_state.json``,
which the API reads.
"""

from __future__ import annotations

from pathlib import Path


def activate(reason: str, state_path: Path, event_log: Path, preset: dict) -> None:
    """Turn the fallback on, log the event, and write the state the API serves."""
    raise NotImplementedError("TODO(week 2): append event, write state atomically")


def read_state(state_path: Path) -> dict:
    """Read ``monitor_state.json``; return an 'ok' default when it is missing."""
    if not state_path.exists():
        return {"drift_status": "ok", "fallback_active": False}
    raise NotImplementedError("TODO(week 2): parse and validate the state file")
