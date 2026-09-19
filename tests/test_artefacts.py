"""Rule 6: every reported number comes from an artefact.

These tests are about the plumbing that makes that true -- the metrics file, and
the marker-delimited README tables that read from it.
"""

from __future__ import annotations

import json

import pytest

from triage.evaluation.artefacts import (
    format_interval,
    format_number,
    read_metrics,
    render_readme_tables,
    update_section,
)
from triage.evaluation.tables import MISSING, baselines_table, fairness_table

CONTEXT = {"config_hash": "abc12345", "git_sha": "deadbeef", "seed": 20260917}


def test_update_section_stamps_provenance(tmp_path) -> None:
    """A section records the config, commit and seed it came from."""
    path = tmp_path / "metrics.json"
    update_section(path, "baselines", {"models": {}}, CONTEXT)

    metrics = read_metrics(path)
    assert metrics["baselines"]["context"]["config_hash"] == "abc12345"
    assert metrics["baselines"]["context"]["git_sha"] == "deadbeef"
    assert "generated_at" in metrics["baselines"]["context"]


def test_update_section_leaves_other_sections_alone(tmp_path) -> None:
    """One stage must never wipe another's results."""
    path = tmp_path / "metrics.json"
    update_section(path, "baselines", {"value": 1}, CONTEXT)
    update_section(path, "monitoring", {"value": 2}, CONTEXT)
    update_section(path, "baselines", {"value": 3}, CONTEXT)

    metrics = read_metrics(path)
    assert metrics["baselines"]["value"] == 3
    assert metrics["monitoring"]["value"] == 2


def test_nan_is_written_as_null_not_as_a_number(tmp_path) -> None:
    """NaN is not JSON. An undefined metric stays undefined rather than becoming 0."""
    path = tmp_path / "metrics.json"
    update_section(path, "baselines", {"fpr_ratio": float("nan"), "ok": 0.5}, CONTEXT)

    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert json.loads(raw)["baselines"]["fpr_ratio"] is None
    assert json.loads(raw)["baselines"]["ok"] == 0.5


def test_a_sampled_run_cannot_overwrite_reported_results(tmp_path) -> None:
    """Rule 6, enforced rather than trusted.

    A stage run with `data.sample_frac=0.05` writes artefacts that look exactly
    like real ones. This happened during development: a smoke test silently
    replaced a full monitoring run with a 5% sample. The context carries the
    fraction, so the write is refused.
    """
    path = tmp_path / "metrics.json"
    update_section(path, "baselines", {"value": "real"}, CONTEXT)

    sampled = {**CONTEXT, "sample_frac": 0.05}
    with pytest.raises(ValueError, match="sampled run"):
        update_section(path, "baselines", {"value": "development"}, sampled)

    assert read_metrics(path)["baselines"]["value"] == "real", "the real result survived"


def test_a_sampled_run_may_write_somewhere_the_report_ignores(tmp_path) -> None:
    """The escape hatch is explicit, not the default."""
    path = tmp_path / "scratch.json"
    sampled = {**CONTEXT, "sample_frac": 0.1}

    update_section(path, "baselines", {"value": "dev"}, sampled, allow_sampled=True)
    assert read_metrics(path)["baselines"]["context"]["sample_frac"] == 0.1


def test_a_full_run_writes_normally(tmp_path) -> None:
    """The guard must not obstruct the ordinary case."""
    path = tmp_path / "metrics.json"
    update_section(path, "baselines", {"value": 1}, {**CONTEXT, "sample_frac": 1.0})
    assert read_metrics(path)["baselines"]["value"] == 1


def test_missing_metrics_file_raises_rather_than_returning_empty(tmp_path) -> None:
    """A missing artefact is reported, never quietly treated as "no results"."""
    with pytest.raises(FileNotFoundError, match="run the make target"):
        read_metrics(tmp_path / "nothing.json")


# --- README tables ---------------------------------------------------------------


def test_render_replaces_only_between_the_markers(tmp_path) -> None:
    """Prose around a table survives regeneration."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\nBefore.\n\n"
        "<!-- metrics:baselines -->\nold\n<!-- /metrics:baselines -->\n\n"
        "After.\n",
        encoding="utf-8",
    )

    replaced = render_readme_tables(readme, {"baselines": "| new |\n| --- |"})
    text = readme.read_text(encoding="utf-8")

    assert replaced == ["baselines"]
    assert "Before." in text and "After." in text
    assert "old" not in text
    assert "| new |" in text


def test_rendering_is_idempotent(tmp_path) -> None:
    """Running `make report` twice gives the same file, so it is safe to re-run."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- metrics:baselines -->\nx\n<!-- /metrics:baselines -->\n", encoding="utf-8"
    )

    render_readme_tables(readme, {"baselines": "| a |"})
    once = readme.read_text(encoding="utf-8")
    render_readme_tables(readme, {"baselines": "| a |"})
    assert readme.read_text(encoding="utf-8") == once


def test_a_table_with_no_marker_is_skipped_not_appended(tmp_path) -> None:
    """No marker, no table: never dump content at the end of the file."""
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n", encoding="utf-8")

    assert render_readme_tables(readme, {"baselines": "| a |"}) == []
    assert readme.read_text(encoding="utf-8") == "# Title\n"


def test_tables_say_so_when_a_section_is_missing() -> None:
    """A missing result must look missing, not like a zero."""
    assert baselines_table({}) == MISSING
    assert fairness_table({}) == MISSING


# --- number formatting ------------------------------------------------------------


def test_three_significant_figures_in_prose() -> None:
    """Section 8.3: three significant figures in prose, full precision in artefacts."""
    assert format_number(0.53512345) == "0.535"
    assert format_number(0.000123456) == "0.000123"
    assert format_number(1234.5678) == "1.23e+03"


def test_undefined_numbers_read_as_not_available() -> None:
    """An undefined statistic prints as n/a rather than as a number."""
    assert format_number(float("nan")) == "n/a"
    assert format_number(None) == "n/a"
    assert format_interval(None) == "n/a"


def test_interval_formatting() -> None:
    """Point estimate with its interval, the way the README shows it."""
    record = {"point": 0.535123, "ci_low": 0.512456, "ci_high": 0.558789}
    assert format_interval(record) == "0.535 (0.512-0.559)"
