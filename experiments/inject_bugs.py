"""Injected upstream data bugs, each starting at window 5 of month 6.

(a) swap the two most frequent ``employment_status`` codes;
(b) mirror ``income`` through a lookup over its observed values (lowest <-> highest),
    never ``1 - x``, so the values stay exactly within the contract;
(c) set ``phone_mobile_valid`` to 1 for every row;
(d) multiply ``income`` by 10 -- out of contract, so batch validation AND the API
    must reject it.

For each bug report the detection delay (applications from the first bugged row
to ALERT), false alarms on clean windows, and model degradation (TPR at the
deployment threshold, before vs after).
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig


def swap_top_categories(frame: pd.DataFrame, column: str, n: int = 2) -> pd.DataFrame:
    """Bug (a): swap the ``n`` most frequent codes in ``column``."""
    raise NotImplementedError("TODO(week 2): value_counts, then swap the top n")


def mirror_lookup(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Bug (b): map the sorted observed values onto their reverse."""
    raise NotImplementedError("TODO(week 2): sorted unique -> reversed lookup")


def set_constant(frame: pd.DataFrame, column: str, value: object) -> pd.DataFrame:
    """Bug (c): overwrite a column with one value."""
    raise NotImplementedError("TODO(week 2): assign the constant")


def multiply(frame: pd.DataFrame, column: str, factor: float) -> pd.DataFrame:
    """Bug (d): scale a column out of its contracted range."""
    raise NotImplementedError("TODO(week 2): scale, and check the contract rejects it")


def run(cfg: DictConfig) -> dict:
    """Apply each configured bug from its start window and measure the response."""
    raise NotImplementedError("TODO(week 2): per-bug detection delay and degradation")
