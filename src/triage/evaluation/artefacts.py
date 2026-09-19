"""Every reported number comes from here (rule 6).

README tables, the validation report and the demo read only ``reports/metrics.json``
and ``reports/tables/``. Nothing is ever typed, estimated or recalled.

Each section carries its own provenance -- config hash, git SHA, seed, the data it
was computed from, and when -- so a stale section is visible rather than silently
mixed with a fresh one.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MARKER_START = "<!-- metrics:{name} -->"
MARKER_END = "<!-- /metrics:{name} -->"


def read_metrics(path: Path) -> dict[str, Any]:
    """Read ``reports/metrics.json``. Raises ``FileNotFoundError`` when it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the make target that generates it")
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    """NaN and infinity are not JSON. Keep them as null rather than inventing a number."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _json_safe(value.item())
        except (AttributeError, ValueError):
            return value
    return value


def update_section(
    path: Path,
    section: str,
    payload: dict[str, Any],
    context: dict[str, Any],
    *,
    allow_sampled: bool = False,
) -> dict[str, Any]:
    """Merge one top-level section into ``metrics.json``, stamped with run context.

    Other sections are left exactly as they were, so one stage never overwrites
    another's results.

    **A sampled run cannot write here.** ``data.sample_frac`` is for development,
    and rule 6 says no reported number may come from one -- but an artefact on
    disk looks identical either way, and a stage run with ``sample_frac=0.05``
    will happily overwrite real results with development ones. So it is refused
    rather than trusted. Pass ``allow_sampled=True`` only to write somewhere the
    report never reads.
    """
    fraction = float(context.get("sample_frac", 1.0))
    if fraction < 1.0 and not allow_sampled:
        raise ValueError(
            f"refusing to write '{section}' to {path.name} from a sampled run "
            f"(data.sample_frac={fraction}). Development runs must not overwrite "
            "reported results (rule 6). Re-run without the override, or pass "
            "allow_sampled=True to write somewhere the report does not read."
        )

    metrics = read_metrics(path) if path.exists() else {}
    metrics[section] = _json_safe(
        {**payload, "context": {**context, "generated_at": datetime.now(UTC).isoformat()}}
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def render_readme_tables(readme: Path, tables: dict[str, str]) -> list[str]:
    """Replace the content between ``<!-- metrics:NAME -->`` markers in the README.

    Returns the names actually replaced. A marker with no table, or a table with no
    marker, is reported rather than guessed at.
    """
    text = readme.read_text(encoding="utf-8")
    replaced = []

    for name, body in tables.items():
        start, end = MARKER_START.format(name=name), MARKER_END.format(name=name)
        if start not in text or end not in text:
            continue
        head, _, rest = text.partition(start)
        _, _, tail = rest.partition(end)
        text = f"{head}{start}\n{body.strip()}\n{end}{tail}"
        replaced.append(name)

    readme.write_text(text, encoding="utf-8")
    return replaced


def format_number(value: float | None, digits: int = 3) -> str:
    """Three significant figures in prose; artefacts keep full precision (section 8.3)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}g}"


def format_interval(record: dict[str, Any] | None, digits: int = 3) -> str:
    """``0.535 (0.512-0.558)`` from a bootstrap record."""
    if not record:
        return "n/a"
    point = format_number(record.get("point"), digits)
    low, high = record.get("ci_low"), record.get("ci_high")
    if low is None or high is None:
        return point
    return f"{point} ({format_number(low, digits)}-{format_number(high, digits)})"
