"""Domain classifier: can a small model tell a window from the reference rows?"""

from __future__ import annotations

import pandas as pd


def domain_auc(window: pd.DataFrame, reference: pd.DataFrame, *, seed: int) -> float:
    """5-fold CV ROC-AUC of a 50-tree LightGBM separating window from reference.

    The reference is 4,000 rows sampled from ``cal_conf``. 0.5 means indistinguishable.
    """
    raise NotImplementedError("TODO(week 2): 50 trees, 5-fold CV, seeded")
