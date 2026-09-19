"""Pinned metrics on the fixture.

Required check (CLAUDE.md section 13):
- headline metrics stay within tolerance of their recorded values.

This is the canary for silent behaviour changes: a library upgrade, a refactor of
the feature pipeline, a change to how sentinels are handled. None of those would
fail a unit test, and all of them would move these numbers.

The values are pinned on the **fixture**, never on the real data, so the test runs
anywhere and is reproducible from a clean clone. They are therefore not results:
they are a fingerprint of the pipeline's behaviour.

If a change moves these legitimately, re-pin them **and say so in
docs/PROGRESS.md** with the reason. A silently re-pinned regression test is worse
than no regression test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from triage.config import with_model
from triage.data.split import deployment_protocol
from triage.evaluation.metrics import calibration, discrimination
from triage.models.baselines import fit
from triage.models.calibrate import fit_calibrator
from triage.uncertainty.conformal import coverage_report, fit_thresholds

# Recorded 2026-09-19, champion on a 20,000-row fixture (seed 20260917):
# trained on 12,500 rows with 121 frauds, evaluated on month 6 (2,500 rows, 39 frauds).
PINNED = {
    "roc_auc": 0.917451,
    "pr_auc": 0.225396,
    "tpr_at_5pct_fpr": 0.589744,
    "brier": 0.013823,
    "ece": 0.008401,
    "fraud_coverage": 0.769231,
    "approve_share": 0.856000,
}

# Tight enough to catch a real change, loose enough to survive a patch release of
# a numerical library on a different CPU.
TOLERANCE = 0.02

ALPHA_FRAUD, ALPHA_LEGIT = 0.3, 0.05


@pytest.fixture(scope="module")
def cfg(repo_root):
    """The composed project config."""
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        return compose(config_name="config")


@pytest.fixture(scope="module")
def measured(cfg) -> dict[str, float]:
    """Run the pipeline end to end on the fixture, exactly as the pins were made."""
    from tests.fixtures.make_fixture import make_fixture

    frame = make_fixture(n_rows=20_000, seed=20260917)
    splits = deployment_protocol(frame, cfg)
    model = fit(frame.loc[splits.train], with_model(cfg, "champion"))

    prob_part = frame.loc[splits.cal_prob]
    calibrator = fit_calibrator("platt", model.score(prob_part), prob_part["fraud_bool"].to_numpy())

    test = frame[frame["month"] == 6]
    labels = test["fraud_bool"].to_numpy()
    probs = calibrator.transform(model.score(test))

    conf = frame.loc[splits.cal_conf]
    thresholds = fit_thresholds(
        calibrator.transform(model.score(conf)),
        conf["fraud_bool"].to_numpy(),
        alpha_fraud=ALPHA_FRAUD,
        alpha_legit=ALPHA_LEGIT,
    )
    coverage = coverage_report(probs, labels, thresholds)

    detection = discrimination(labels, probs)
    calibrated = calibration(labels, probs)
    return {
        "roc_auc": detection["roc_auc"],
        "pr_auc": detection["pr_auc"],
        "tpr_at_5pct_fpr": detection["tpr_at_0.05_fpr"],
        "brier": calibrated["brier"],
        "ece": calibrated["ece_equal_mass"],
        "fraud_coverage": coverage["fraud_coverage"],
        "approve_share": coverage["approve_share"],
    }


@pytest.mark.parametrize("metric", sorted(PINNED))
def test_pinned_metric(measured: dict[str, float], metric: str) -> None:
    """Each headline metric is where it was when it was recorded."""
    assert measured[metric] == pytest.approx(PINNED[metric], abs=TOLERANCE), (
        f"{metric} moved from {PINNED[metric]:.6f} to {measured[metric]:.6f}. "
        "If that is intended, re-pin it and record why in docs/PROGRESS.md."
    )


def test_the_model_still_finds_the_planted_signal(measured: dict[str, float]) -> None:
    """A sanity floor that does not depend on the pins being right."""
    assert measured["roc_auc"] > 0.8
    assert measured["fraud_coverage"] >= 1 - ALPHA_FRAUD - 0.05


def test_the_pins_cover_every_stage(measured: dict[str, float]) -> None:
    """Detection, calibration and the policy: a regression in any of them shows up."""
    assert set(measured) == set(PINNED)
    assert {"roc_auc", "brier", "fraud_coverage"} <= set(PINNED)


@pytest.mark.data
def test_the_real_pipeline_matches_its_published_reference(repo_root) -> None:
    """The real run is checked against the public reference, not against a pin.

    Section 9 cites a third-party time-based split at 0.535 recall at 5% FPR and
    ROC-AUC 0.89. Pinning real-data numbers here would duplicate metrics.json and
    drift from it; comparing against the external reference is the check that
    actually means something.
    """
    import json

    metrics_path = repo_root / "reports" / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("no metrics.json: run `make baseline`")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "baselines" not in metrics:
        pytest.skip("no baselines section yet")

    pooled = metrics["baselines"]["models"]["b1_lgbm"]["paper"]["pooled"]["discrimination"]
    assert pooled["roc_auc"] == pytest.approx(0.89, abs=0.03)
    assert pooled["tpr_at_0.05_fpr"] == pytest.approx(0.535, abs=0.05)


def test_the_fixture_itself_is_stable() -> None:
    """If the fixture drifts, every pin above is measuring something else."""
    from tests.fixtures.make_fixture import make_fixture

    frame = make_fixture(n_rows=20_000, seed=20260917)
    assert len(frame) == 20_000
    assert int(frame["fraud_bool"].sum()) == 213
    assert list(frame.columns)[:3] == ["fraud_bool", "income", "name_email_similarity"]


def test_pins_are_documented() -> None:
    """The recording date and setup must be written down, or the pins are folklore."""
    source = Path(__file__).read_text(encoding="utf-8")
    assert "Recorded 2026-09-19" in source
    assert "docs/PROGRESS.md" in source


def test_the_fixture_frame_is_not_the_real_data(cfg) -> None:
    """These are a fingerprint of behaviour, never a reportable result."""
    from tests.fixtures.make_fixture import make_fixture

    frame = make_fixture(n_rows=20_000, seed=20260917)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) < 1_000_000, "the real data has a million rows; pins are fixture-only"
