"""The champion's hyper-parameter search (CLAUDE.md section 8.1).

The thing worth testing here is not that the search finds good parameters -- on a
4,000-row fixture it will not -- but that it can be *shown* to have found them.
A search reports its winner; this one has to report what the winner beat, and has
to leave a file behind saying so, because `reports/metrics.json` records the
current state of the project and an ordinary run is not a tuning run.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from triage.models.champion import tune, write_tuning_trials


@pytest.fixture(scope="module")
def cfg(repo_root):
    """The project config, with a search small enough to run in a test."""
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        composed = compose(config_name="config")
    OmegaConf.update(composed, "model.tuning.max_trials", 2)
    return composed


@pytest.fixture(scope="module")
def result(fixture_frame: pd.DataFrame, cfg) -> dict:
    """One small search, reused by every assertion below."""
    time_column = str(cfg.data.time_column)
    train = fixture_frame[fixture_frame[time_column].isin([0, 1, 2, 3])]
    valid = fixture_frame[fixture_frame[time_column] == 4]
    return tune(train, valid, cfg)


def test_the_configured_parameters_are_scored_alongside_the_search(result: dict) -> None:
    """Without an incumbent, "best of N" cannot be judged worth the compute."""
    trials = {row["trial"]: row for row in result["trials"]}

    assert -1 in trials, "the configured parameters must run as a trial of their own"
    assert result["incumbent_score"] == trials[-1][result["metric"]]
    assert result["n_trials"] == len(trials) - 1, "the incumbent is not one of the search's trials"


def test_the_winner_is_reported_as_beating_the_incumbent_or_not(result: dict) -> None:
    """`beat_incumbent` must agree with the scores, in either direction."""
    best = result["best_score"]
    assert best >= result["incumbent_score"], "the best trial cannot score below the incumbent"
    assert result["beat_incumbent"] == (best > result["incumbent_score"])


def test_a_search_that_loses_keeps_the_configured_parameters(result: dict) -> None:
    """Losing is an allowed outcome, and it must not silently change the model."""
    if result["beat_incumbent"]:
        pytest.skip("this fixture's search won; the losing path is covered by the assertion below")

    incumbent = next(row for row in result["trials"] if row["trial"] == -1)
    assert result["best_params"] == incumbent["params"]


def test_the_search_never_sees_the_calibration_or_test_months(result: dict, cfg) -> None:
    """Tuning on month 5 would spend the guarantee before it is made."""
    assert result["validation_month"] == int(cfg.data.protocol.champion_tuning.valid_month)
    assert result["validation_month"] < min(cfg.data.protocol.deployment.test_months)


def test_the_trials_survive_the_next_ordinary_train(result: dict, tmp_path: Path) -> None:
    """The CSV is the durable record: metrics.json will not keep this."""
    path = tmp_path / "tuning_trials.csv"
    write_tuning_trials(path, result)

    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == len(result["trials"])
    assert rows[0]["trial"] == "-1", "the incumbent sorts first, so it is impossible to miss"
    assert result["metric"] in rows[0]
    assert all(row["validation_month"] == str(result["validation_month"]) for row in rows)
