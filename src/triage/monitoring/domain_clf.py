"""Domain classifier: can a small model tell a window from the reference rows?

The idea is label-free. Take a window of applications and a sample of reference
rows, label them by *where they came from* rather than by outcome, and try to tell
them apart. If a small model can, the two populations differ; if it cannot, they
are interchangeable -- which is exactly the condition the conformal guarantee
needs.

The score is cross-validated ROC-AUC: 0.5 means indistinguishable, 1.0 trivially
separable. It catches shifts PSI misses, because it sees *combinations* of
features rather than one marginal at a time. An upstream bug that swaps two
category codes can leave every marginal distribution intact.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

log = logging.getLogger("triage")

N_ESTIMATORS = 50
CV_FOLDS = 5


def domain_auc(
    window: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int = N_ESTIMATORS,
    cv_folds: int = CV_FOLDS,
) -> float:
    """5-fold CV ROC-AUC of a 50-tree LightGBM separating window from reference.

    Both frames must already be model-ready: the same columns, in the same order,
    with categories aligned. 0.5 means the window is indistinguishable from the
    reference.
    """
    if window.empty or reference.empty:
        raise ValueError("both the window and the reference need rows")
    if list(window.columns) != list(reference.columns):
        raise ValueError("window and reference must have the same columns, in the same order")

    features = pd.concat([reference, window], ignore_index=True)
    origin = np.concatenate([np.zeros(len(reference), dtype=int), np.ones(len(window), dtype=int)])

    # With too few rows on either side, cross-validation says nothing.
    smallest = min(int(origin.sum()), int((origin == 0).sum()))
    folds = min(cv_folds, smallest)
    if folds < 2:
        raise ValueError(f"need at least 2 rows on each side, got {smallest}")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for train_index, test_index in splitter.split(features, origin):
        model = LGBMClassifier(
            n_estimators=n_estimators,
            seed=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(features.iloc[train_index], origin[train_index])
        predicted = np.asarray(model.predict_proba(features.iloc[test_index]))[:, 1]
        scores.append(roc_auc_score(origin[test_index], predicted))

    return float(np.mean(scores))


def reference_sample(frame: pd.DataFrame, n_rows: int, *, seed: int) -> pd.DataFrame:
    """A fixed, seeded reference sample, drawn once and reused for every window.

    Re-drawing it per window would add sampling noise to the detector, and the
    threshold calibrated on clean windows would stop meaning anything.
    """
    if len(frame) <= n_rows:
        return frame
    return frame.sample(n=n_rows, random_state=seed)
