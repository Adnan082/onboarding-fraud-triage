"""Negative values are missing-value sentinels. Turn them into explicit flags.

Verified against the first load (CLAUDE.md section 7, DECISIONS D10):

- exactly ``-1`` means missing in five count columns;
- any negative value means missing in ``intended_balcon_amount``;
- ``credit_risk_score`` is negative for 1.44% of Base including 488 rows at exactly
  ``-1``, and those are **real scores**. Flagging them would corrupt a feature the
  model leans on heavily;
- ``velocity_6h`` is negative for 44 rows in a million, which is not physically
  meaningful. It gets a flag but keeps its value: a quirk to report, not a hole.

A sentinel is blanked to ``NaN`` once flagged, so no model can read ``-1`` as a
small count. LightGBM handles the ``NaN`` natively; B0 imputes it, and the flag
carries the information that it was missing at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

MISSING_SUFFIX = "_missing"
NEGATIVE_SUFFIX = "_negative"


def flag_columns(cfg: DictConfig) -> list[str]:
    """Every flag column :func:`add_missing_flags` will add, in a stable order."""
    sentinels = cfg.features.sentinels
    names = [f"{column}{MISSING_SUFFIX}" for column in sentinels.minus_one]
    names += [f"{column}{MISSING_SUFFIX}" for column in sentinels.any_negative]
    names += [f"{column}{NEGATIVE_SUFFIX}" for column in sentinels.get("quirk_negative", [])]
    return names


def add_missing_flags(frame: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Add a flag per sentinel column and blank the sentinel itself.

    Returns a copy; the input is never modified. A configured column that is not
    in the frame raises, because a silently skipped sentinel would quietly change
    what the model sees.
    """
    sentinels = cfg.features.sentinels
    out = frame.copy()

    for column in sentinels.minus_one:
        if column not in out.columns:
            raise KeyError(f"sentinel column {column!r} is not in the frame")
        missing = out[column] == -1
        out[f"{column}{MISSING_SUFFIX}"] = missing.astype(int)
        out[column] = out[column].astype(float).mask(missing, np.nan)

    for column in sentinels.any_negative:
        if column not in out.columns:
            raise KeyError(f"sentinel column {column!r} is not in the frame")
        missing = out[column] < 0
        out[f"{column}{MISSING_SUFFIX}"] = missing.astype(int)
        out[column] = out[column].astype(float).mask(missing, np.nan)

    # Flagged, but the value stays: these are real numbers, however odd.
    for column in sentinels.get("quirk_negative", []):
        if column not in out.columns:
            raise KeyError(f"quirk column {column!r} is not in the frame")
        out[f"{column}{NEGATIVE_SUFFIX}"] = (out[column] < 0).astype(int)

    return out


def zero_variance_columns(frame: pd.DataFrame) -> list[str]:
    """Columns with a single distinct value. Record any drop in docs/DECISIONS.md."""
    return [column for column in frame.columns if frame[column].nunique(dropna=False) <= 1]


def sentinel_report(frame: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Share of rows flagged per sentinel column, for the EDA and the report."""
    flagged = add_missing_flags(frame, cfg)
    rows = [
        {
            "column": name,
            "n_flagged": int(flagged[name].sum()),
            "share_flagged": float(flagged[name].mean()),
        }
        for name in flag_columns(cfg)
    ]
    return pd.DataFrame(rows)
