"""The variant stress test's selection rules.

The scoring loop lives in the monitor stage, which owns the detectors and the
calibrated thresholds. What lives here is the part that decides *what* gets
scored, and that is worth testing on its own: pick the wrong rows and the
experiment measures time rather than population.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from experiments.stress_variants import (
    held_out_months,
    population_summary,
    select_test_window,
    variant_names,
)


@pytest.fixture(scope="module")
def cfg(repo_root):
    """The composed project config."""
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        return compose(config_name="config")


def test_the_configured_variants_are_the_ones_section_7_names(cfg) -> None:
    """Base trains the model; IV and V are the shifted populations it is tested on."""
    assert variant_names(cfg) == ["variant_iv", "variant_v"]
    assert "base" not in variant_names(cfg), "Base is the reference, not a stress test"


def test_only_the_held_out_months_are_scored(fixture_frame: pd.DataFrame, cfg) -> None:
    """Scoring a variant's training months would confuse population with time."""
    assert held_out_months(cfg) == [6, 7]

    selected = select_test_window(fixture_frame, cfg)
    assert set(selected["month"].unique()) == {6, 7}
    assert len(selected) < len(fixture_frame)


def test_a_variant_without_the_test_months_raises(fixture_frame: pd.DataFrame, cfg) -> None:
    """An empty stress test is an error, not a pass."""
    early = fixture_frame[fixture_frame["month"] <= 4]
    with pytest.raises(ValueError, match="cannot stress test"):
        select_test_window(early, cfg)


def test_population_summary_describes_the_shift(fixture_frame: pd.DataFrame, cfg) -> None:
    """The report needs to say *why* a variant is a stress test."""
    summary = population_summary(fixture_frame, cfg)

    assert summary["n"] == float(len(fixture_frame))
    assert 0.0 <= summary["share_older"] <= 1.0
    assert 0.0 <= summary["prevalence"] <= 1.0
    assert {"prevalence_older", "prevalence_younger"} <= set(summary)


def test_population_summary_uses_age_only_to_measure(fixture_frame: pd.DataFrame, cfg) -> None:
    """Age describes the population here; it never reaches a model (rule 4)."""
    changed = fixture_frame.copy()
    changed["customer_age"] = 90

    summary = population_summary(changed, cfg)
    assert summary["share_older"] == 1.0
    # The label distribution is untouched: only the description of who they are moved.
    assert summary["prevalence"] == pytest.approx(
        population_summary(fixture_frame, cfg)["prevalence"]
    )
