"""B0 (logistic regression) and B1 (LightGBM defaults, including customer_age).

B1 is the paper-comparable baseline: LightGBM on **all** features, age included.
B0 is the interpretable one. Neither is the champion -- the champion drops age
(rule 4) and lives in ``champion.py``.

Both are returned wrapped in :class:`FittedModel`, which owns its own
preprocessing. A caller scores a raw frame and never has to remember which model
wanted one-hot columns and which wanted pandas categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from triage.features.encode import build_encoder, lgbm_frame, prepare

SKLEARN, LGBM = "sklearn", "lgbm"


@dataclass
class FittedModel:
    """A fitted model plus the preprocessing it expects.

    ``score`` takes a raw frame (as loaded, before sentinels) and returns fraud
    probabilities, so every caller treats every model the same way.
    """

    name: str
    kind: str
    estimator: Any
    use_age: bool
    cfg: DictConfig
    n_train: int
    n_train_fraud: int

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.kind == LGBM:
            return lgbm_frame(frame, self.cfg, use_age=self.use_age)
        return prepare(frame, self.cfg, use_age=self.use_age)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Fraud probability for each row."""
        return np.asarray(self.estimator.predict_proba(self._prepare(frame))[:, 1], dtype=float)


def _params(cfg: DictConfig) -> dict[str, Any]:
    """The model's constructor arguments, with ``${seed}`` and friends resolved."""
    from omegaconf import OmegaConf

    params = OmegaConf.to_container(cfg.model.params, resolve=True)
    if not isinstance(params, dict):
        return {}
    return {str(key): value for key, value in params.items()}


def fit_b0(train: pd.DataFrame, cfg: DictConfig) -> FittedModel:
    """Logistic regression: one-hot categoricals, standardised numerics, balanced classes.

    The encoder is fit inside the pipeline, on the training months only.
    """
    use_age = bool(cfg.model.use_age)
    labels = train[cfg.data.label].to_numpy()

    pipeline = Pipeline(
        [
            ("encode", build_encoder(cfg, use_age=use_age)),
            ("model", LogisticRegression(**_params(cfg))),
        ]
    )
    pipeline.fit(prepare(train, cfg, use_age=use_age), labels)

    return FittedModel(
        name=str(cfg.model.name),
        kind=SKLEARN,
        estimator=pipeline,
        use_age=use_age,
        cfg=cfg,
        n_train=len(train),
        n_train_fraud=int(labels.sum()),
    )


def fit_b1(train: pd.DataFrame, cfg: DictConfig) -> FittedModel:
    """LightGBM on all features, ``customer_age`` included. The paper-comparable baseline."""
    use_age = bool(cfg.model.use_age)
    labels = train[cfg.data.label].to_numpy()

    model = LGBMClassifier(**_params(cfg))
    model.fit(lgbm_frame(train, cfg, use_age=use_age), labels)

    return FittedModel(
        name=str(cfg.model.name),
        kind=LGBM,
        estimator=model,
        use_age=use_age,
        cfg=cfg,
        n_train=len(train),
        n_train_fraud=int(labels.sum()),
    )


def fit(train: pd.DataFrame, cfg: DictConfig) -> FittedModel:
    """Fit whichever model ``cfg.model`` names."""
    kind = str(cfg.model.kind)
    if kind.startswith("sklearn"):
        return fit_b0(train, cfg)
    if kind.startswith("lightgbm"):
        return fit_b1(train, cfg)
    raise ValueError(f"no baseline fitter for model kind {kind!r}")
