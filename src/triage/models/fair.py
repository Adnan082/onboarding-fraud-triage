"""Fairness mitigations M1-M4.

M1 drops age. Usually not enough: other fields proxy for it.
M2 FairGBM, ``constraint_type="FPR"``, group ``customer_age >= 50``, training only.
M3 fairlearn ExponentiatedGradient with FalsePositiveRateParity.
M4 policy only: rely on the review band rather than changing the model.

All four use age at **training or measurement** time only. None of them uses age
to decide anything about an individual application, which is the line rule 4
draws. M2 and M3 both take the age group as a constraint while fitting and then
never see it again: ``predict`` needs no sensitive feature.

The four are not interchangeable. M1 and M2 produce scores, so they can be
compared on a threshold sweep. M3 returns a randomised classifier at a single
operating point -- there is no threshold to sweep, so it is evaluated where it
sits. M4 changes no model at all; it asks what the decision policy already does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from triage.features.encode import lgbm_frame
from triage.models.baselines import LGBM, FittedModel, model_params
from triage.models.baselines import fit as fit_baseline

log = logging.getLogger("triage")

SKIP_MESSAGE = (
    "FairGBM is unavailable: it ships a Linux .so, so M2 is skipped on Windows and macOS. "
    "Install the `fairgbm` extra on Linux (or use the Docker image) to run it."
)


def fairgbm_available() -> bool:
    """FairGBM ships a Linux ``.so``; import it lazily and skip elsewhere."""
    try:
        import fairgbm  # noqa: F401
    except Exception:
        return False
    return True


def age_group(frame: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    """The training-time constraint group: 1 where age is at or above the cut."""
    column = str(cfg.features.protected.age)
    cut = int(cfg.features.protected.age_cut)
    return (frame[column] >= cut).astype(int).to_numpy()


@dataclass
class PointClassifier:
    """A classifier that only has one operating point.

    fairlearn's ``ExponentiatedGradient`` returns a randomised mixture of
    classifiers. It has no meaningful score to threshold, so it exposes
    ``predict`` and nothing else, and everything downstream must evaluate it where
    it stands.
    """

    name: str
    estimator: Any
    cfg: DictConfig
    n_train: int
    n_train_fraud: int

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Flag or do not flag, for each application."""
        features = lgbm_frame(frame, self.cfg, use_age=False)
        return np.asarray(self.estimator.predict(features)).astype(int)


def fit_m1_drop_age(train: pd.DataFrame, cfg: DictConfig) -> FittedModel:
    """M1: the champion itself, which never sees age. The cheapest mitigation there is."""
    return fit_baseline(train, cfg)


def fit_m2_fairgbm(train: pd.DataFrame, cfg: DictConfig) -> FittedModel:
    """FairGBM with an FPR constraint on the age group. Linux only.

    The constraint enters the training objective: the model is penalised for
    unequal false-positive rates between the groups while it fits. At prediction
    time it is an ordinary gradient-boosted model that has never seen age.
    """
    if not fairgbm_available():
        raise RuntimeError(SKIP_MESSAGE)

    from fairgbm import FairGBMClassifier

    params = model_params(cfg)

    features = lgbm_frame(train, cfg, use_age=False)
    labels = train[cfg.data.label].to_numpy()

    model = FairGBMClassifier(**params)
    model.fit(features, labels, constraint_group=age_group(train, cfg))

    return FittedModel(
        name="m2_fairgbm",
        kind=LGBM,
        estimator=model,
        use_age=False,
        cfg=cfg,
        n_train=len(train),
        n_train_fraud=int(labels.sum()),
    )


def fit_m3_fairlearn(train: pd.DataFrame, cfg: DictConfig) -> PointClassifier:
    """ExponentiatedGradient + FalsePositiveRateParity.

    Returns a randomised classifier at ONE operating point, so it is evaluated
    there rather than with threshold-sweep metrics. ``predict(X)`` needs no
    sensitive feature.
    """
    from fairlearn.reductions import ExponentiatedGradient, FalsePositiveRateParity
    from lightgbm import LGBMClassifier

    settings = model_params(cfg)

    features = lgbm_frame(train, cfg, use_age=False)
    labels = train[cfg.data.label].to_numpy()

    base = LGBMClassifier(
        n_estimators=int(settings.get("n_estimators", 200)),
        learning_rate=float(settings.get("learning_rate", 0.05)),
        num_leaves=int(settings.get("num_leaves", 31)),
        seed=int(cfg.seed),
        deterministic=True,
        force_row_wise=True,
        n_jobs=int(settings.get("n_jobs", 4)),
        verbosity=-1,
    )
    mitigator = ExponentiatedGradient(
        estimator=base,
        constraints=FalsePositiveRateParity(),
        eps=float(cfg.model.get("eps", 0.01)),
        max_iter=int(cfg.model.get("max_iter", 50)),
    )
    mitigator.fit(features, labels, sensitive_features=age_group(train, cfg))

    return PointClassifier(
        name="m3_fairlearn_eg",
        estimator=mitigator,
        cfg=cfg,
        n_train=len(train),
        n_train_fraud=int(labels.sum()),
    )
