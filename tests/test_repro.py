"""Reproducibility.

Required check (CLAUDE.md section 13):
- the same seed gives an identical hash of the scores.

Rule 10 says a clean clone plus `make all` regenerates every artefact. That is a
claim about the whole pipeline, and it can only hold if each step is
deterministic. These tests check the steps a reviewer would doubt: the model fit,
the calibration, the conformal thresholds, and the split.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from triage.config import with_model
from triage.data.split import deployment_protocol
from triage.models.baselines import fit
from triage.models.calibrate import fit_calibrator
from triage.seeding import seed_everything
from triage.uncertainty.conformal import fit_thresholds


def score_hash(scores: np.ndarray) -> str:
    """A stable digest of a score vector, to 12 decimal places.

    Full float equality would make this a test of the hardware's last bit. Twelve
    places is far tighter than any decision the scores feed.
    """
    rounded = np.round(np.asarray(scores, dtype=float), 12)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


@pytest.fixture(scope="module")
def cfg(repo_root):
    """The composed project config."""
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        return compose(config_name="config")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """A fixture big enough to split and fit."""
    from tests.fixtures.make_fixture import make_fixture

    return make_fixture(n_rows=20_000, seed=20260917)


def test_the_same_seed_gives_an_identical_score_hash(frame: pd.DataFrame, cfg) -> None:
    """Two runs of the same commit produce the same scores, bit for bit."""
    model_cfg = with_model(cfg, "champion")
    train = frame[frame["month"] <= 4]
    test = frame[frame["month"] == 6]

    hashes = []
    for _ in range(2):
        seed_everything(int(cfg.seed))
        hashes.append(score_hash(fit(train, model_cfg).score(test)))

    assert hashes[0] == hashes[1], "the same seed produced different scores"


def test_a_different_seed_gives_different_scores(frame: pd.DataFrame, cfg) -> None:
    """Guard the guard: if the seed did nothing, the test above would be vacuous."""
    from omegaconf import OmegaConf

    train = frame[frame["month"] <= 4]
    test = frame[frame["month"] == 6]

    base = with_model(cfg, "champion")
    other = OmegaConf.merge(base, {"seed": int(cfg.seed) + 1})

    assert score_hash(fit(train, base).score(test)) != score_hash(fit(train, other).score(test))


def test_the_calibration_split_is_reproducible(frame: pd.DataFrame, cfg) -> None:
    """The three parts of month 5 must land identically every run."""
    first = deployment_protocol(frame, cfg)
    second = deployment_protocol(frame, cfg)

    for part in ("cal_prob", "cal_tune", "cal_conf"):
        assert getattr(first, part).equals(getattr(second, part))


def test_calibration_is_reproducible(frame: pd.DataFrame, cfg) -> None:
    """The calibrator is fitted from scores, so it inherits their determinism."""
    model_cfg = with_model(cfg, "champion")
    splits = deployment_protocol(frame, cfg)
    train = frame.loc[splits.train]
    prob_part = frame.loc[splits.cal_prob]
    test = frame[frame["month"] == 6]

    hashes = []
    for _ in range(2):
        model = fit(train, model_cfg)
        calibrator = fit_calibrator(
            "platt", model.score(prob_part), prob_part["fraud_bool"].to_numpy()
        )
        hashes.append(score_hash(calibrator.transform(model.score(test))))

    assert hashes[0] == hashes[1]


def test_conformal_thresholds_are_reproducible(frame: pd.DataFrame, cfg) -> None:
    """The guarantee would mean nothing if the thresholds moved between runs."""
    model_cfg = with_model(cfg, "champion")
    splits = deployment_protocol(frame, cfg)
    train = frame.loc[splits.train]
    conf = frame.loc[splits.cal_conf]

    results = []
    for _ in range(2):
        scores = fit(train, model_cfg).score(conf)
        results.append(
            fit_thresholds(scores, conf["fraud_bool"].to_numpy(), alpha_fraud=0.3, alpha_legit=0.02)
        )

    assert results[0] == results[1]


@pytest.mark.data
def test_the_saved_model_reproduces_its_manifest(repo_root) -> None:
    """The artefact on disk is the artefact the manifest describes.

    This is the check that would catch a model quietly replaced without the
    pipeline being re-run: the API refuses to start in exactly this case.
    """
    from triage.api.service import load_config
    from triage.models.champion import load

    if not (repo_root / "models" / "manifest.json").exists():
        pytest.skip("no trained artefacts: run `make train`")

    # load() raises if any hash disagrees with the manifest.
    _, _, manifest = load(load_config())
    assert manifest["artefacts"]
    assert manifest["config_hash"]
