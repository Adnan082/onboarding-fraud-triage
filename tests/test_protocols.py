"""The two evaluation protocols, and the baselines that run under them.

All on the seeded fixture: no Kaggle data, no trained artefacts. Confidence
intervals are switched off here because they are tested in `test_fairness.py`;
what matters in this file is that the protocols cannot leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from triage.config import with_model
from triage.evaluation.protocols import run_deployment_protocol, run_paper_protocol
from triage.models.baselines import fit, fit_b0, fit_b1


@pytest.fixture(scope="module")
def cfg(repo_root):
    """The composed project config."""
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        return compose(config_name="config")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Large enough that month 5 can be split three ways and still hold fraud."""
    from tests.fixtures.make_fixture import make_fixture

    return make_fixture(n_rows=40_000, seed=20260917)


@pytest.fixture(scope="module")
def b1(frame: pd.DataFrame, cfg):
    """B1 fitted on the training months, for the cheap assertions."""
    model_cfg = with_model(cfg, "lgbm")
    train = frame[frame["month"] <= 4]
    return fit_b1(train, model_cfg), model_cfg


# --- the models ---------------------------------------------------------------------


def test_b0_and_b1_both_find_the_planted_signal(frame: pd.DataFrame, cfg) -> None:
    """A baseline that cannot beat chance would make every later comparison meaningless."""
    from sklearn.metrics import roc_auc_score

    train = frame[frame["month"] <= 4]
    test = frame[frame["month"] == 6]
    labels = test["fraud_bool"].to_numpy()

    for name, fitter in (("logreg", fit_b0), ("lgbm", fit_b1)):
        model = fitter(train, with_model(cfg, name))
        assert roc_auc_score(labels, model.score(test)) > 0.7, f"{name} found nothing"


def test_scores_are_probabilities(b1, frame: pd.DataFrame) -> None:
    """The policy layer needs calibrated-scale probabilities, not raw margins."""
    model, _ = b1
    scores = model.score(frame[frame["month"] == 7])

    assert scores.shape == (int((frame["month"] == 7).sum()),)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_the_same_seed_gives_the_same_scores(frame: pd.DataFrame, cfg) -> None:
    """Rule 10: reproducible from a clean clone."""
    model_cfg = with_model(cfg, "lgbm")
    train = frame[frame["month"] <= 4]
    test = frame[frame["month"] == 6]

    first = fit_b1(train, model_cfg).score(test)
    second = fit_b1(train, model_cfg).score(test)
    assert np.array_equal(first, second)


def test_b1_sees_age_and_the_champion_does_not(frame: pd.DataFrame, cfg) -> None:
    """B1 is the paper-comparable baseline; the champion is the one bound by rule 4."""
    train = frame[frame["month"] <= 4]

    b1_model = fit(train, with_model(cfg, "lgbm"))
    champion = fit(train, with_model(cfg, "champion"))

    assert b1_model.use_age is True
    assert "customer_age" in b1_model.estimator.feature_name_

    assert champion.use_age is False
    assert "customer_age" not in champion.estimator.feature_name_


def test_changing_age_cannot_change_a_champion_score(frame: pd.DataFrame, cfg) -> None:
    """The end-to-end version of rule 4, at the model rather than the feature level."""
    train = frame[frame["month"] <= 4]
    test = frame[frame["month"] == 6]
    champion = fit(train, with_model(cfg, "champion"))

    older = test.copy()
    older["customer_age"] = 90
    assert np.array_equal(champion.score(test), champion.score(older))


# --- the protocols -------------------------------------------------------------------


@pytest.fixture(scope="module")
def paper_run(frame: pd.DataFrame, cfg) -> dict:
    return run_paper_protocol(frame, with_model(cfg, "lgbm"), with_ci=False)


@pytest.fixture(scope="module")
def deployment_run(frame: pd.DataFrame, cfg) -> dict:
    return run_deployment_protocol(frame, with_model(cfg, "lgbm"), with_ci=False)


def test_paper_protocol_declares_its_threshold_came_from_test(paper_run: dict) -> None:
    """Rule 2's exception must be labelled wherever it appears."""
    assert paper_run["threshold_set_on"] == "test"
    assert "optimistic" in paper_run["threshold_caveat"].lower()
    assert paper_run["train_months"] == [0, 1, 2, 3, 4, 5]


def test_deployment_protocol_takes_its_threshold_from_cal_tune(deployment_run: dict) -> None:
    """Nothing is ever tuned on months 6 or 7."""
    assert deployment_run["threshold_set_on"] == "cal_tune"
    assert deployment_run["train_months"] == [0, 1, 2, 3, 4]
    assert 5 not in deployment_run["train_months"]


def test_both_protocols_report_the_test_months_separately(
    paper_run: dict, deployment_run: dict
) -> None:
    """Pooling is allowed as an extra, never on its own (section 8.1)."""
    for run in (paper_run, deployment_run):
        assert set(run["months"]) == {"6", "7"}
        assert "pooled" in run


def test_the_paper_threshold_holds_its_own_fpr_cap(paper_run: dict) -> None:
    """It was fitted on exactly this data, so it must hit the cap it was fitted to."""
    assert paper_run["pooled"]["realised"]["fpr"] <= 0.05 + 1e-9


def test_the_deployment_threshold_transfers_imperfectly(deployment_run: dict) -> None:
    """The honest reason the deployment protocol exists.

    A threshold chosen on month 5 will not realise exactly 5% FPR on months 6 and
    7. Reporting the realised rate per month is the point; asserting it lands
    exactly on target would be asserting the leak the protocol exists to avoid.
    """
    for month in ("6", "7"):
        realised = deployment_run["months"][month]["realised"]
        assert 0.0 < realised["fpr"] < 0.25
        assert realised["threshold"] == deployment_run["threshold"]


def test_every_slice_carries_detection_and_fairness(deployment_run: dict) -> None:
    """One slice, one complete picture: no metric reported without its context."""
    slice_ = deployment_run["months"]["6"]

    assert {"roc_auc", "pr_auc", "prevalence", "n"} <= set(slice_["discrimination"])
    assert {"tpr", "fpr", "precision", "n_flagged"} <= set(slice_["realised"])
    assert "fpr_ratio" in slice_["fairness"]
    assert {record["group"] for record in slice_["fairness"]["by_group"]} == {
        "age>=50",
        "age<50",
    }
    assert slice_["fairness"]["by_band"], "the 10-year band table must not be empty"


def test_split_sizes_are_recorded(deployment_run: dict) -> None:
    """The README's data table reads these, so they have to be in the artefact."""
    sizes = deployment_run["split_sizes"]
    assert sizes["train"] > 0
    assert sizes["cal_prob"] > 0 and sizes["cal_tune"] > 0 and sizes["cal_conf"] > 0
    assert sizes["test_month_6"] > 0 and sizes["test_month_7"] > 0
