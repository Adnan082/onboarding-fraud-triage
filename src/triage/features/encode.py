"""Feature encoding. Every encoder is fit on TRAINING months only.

Rule 4 lives here: with ``use_age=False`` the column is dropped before any
transform touches it, so the champion cannot see age even by accident.

Categorical levels come from the **frozen contract**, not from whatever happens to
appear in a slice. That way month 7, Variant V and a single API request all encode
a category to the same place, and a level that is missing from one window does not
silently shift the columns.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from triage.data.contract import CONTRACT
from triage.features.sentinels import add_missing_flags, flag_columns

# Never a model input: the label, the split key, and the protected attribute
# unless a model is explicitly allowed it.
ALWAYS_EXCLUDED = ("fraud_bool", "month")


def categorical_levels(column: str) -> list[str]:
    """The contract's level set for one categorical column."""
    spec = CONTRACT[column]
    return [str(level) for level in spec.levels]


def feature_columns(cfg: DictConfig, *, use_age: bool) -> list[str]:
    """Model inputs, in a stable order: numerics, binaries, flags, then categoricals."""
    features = cfg.features
    dropped = set(ALWAYS_EXCLUDED)
    if features.encoding.drop_zero_variance:
        dropped |= set(features.get("zero_variance", []))
    if not use_age:
        dropped.add(str(features.protected.age))

    ordered = [
        *[c for c in features.numerics if c not in dropped],
        *[c for c in features.binaries if c not in dropped],
        *flag_columns(cfg),
        *[c for c in features.categoricals if c not in dropped],
    ]
    if use_age:
        ordered.insert(0, str(features.protected.age))
    return ordered


def prepare(frame: pd.DataFrame, cfg: DictConfig, *, use_age: bool) -> pd.DataFrame:
    """Sentinel flags, then the model's column selection. Shared by every model."""
    flagged = add_missing_flags(frame, cfg)
    return flagged[feature_columns(cfg, use_age=use_age)]


def build_encoder(cfg: DictConfig, *, use_age: bool) -> Pipeline:
    """One-hot categoricals and standardised numerics for B0.

    Fit this on the training months only. An unseen category encodes as all-zeros
    (``handle_unknown="ignore"``), which is the one-hot equivalent of the
    configured ``other`` bucket.
    """
    features = cfg.features
    columns = feature_columns(cfg, use_age=use_age)
    categoricals = [c for c in features.categoricals if c in columns]
    numerics = [c for c in columns if c not in categoricals]

    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy=str(features.encoding.impute_numerics)))
    ]
    if features.encoding.standardise_numerics:
        numeric_steps.append(("scale", StandardScaler()))

    return Pipeline(
        [
            (
                "columns",
                ColumnTransformer(
                    [
                        ("numeric", Pipeline(numeric_steps), numerics),
                        (
                            "categorical",
                            OneHotEncoder(
                                categories=[categorical_levels(c) for c in categoricals],
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                            categoricals,
                        ),
                    ],
                    remainder="drop",
                ),
            )
        ]
    )


def lgbm_frame(frame: pd.DataFrame, cfg: DictConfig, *, use_age: bool) -> pd.DataFrame:
    """Return a frame with native pandas ``category`` dtypes for LightGBM.

    The categories are the contract's, so the encoding is identical across months,
    variants and single requests.
    """
    out = prepare(frame, cfg, use_age=use_age).copy()
    for column in cfg.features.categoricals:
        if column in out.columns:
            out[column] = pd.Categorical(
                out[column].astype(str), categories=categorical_levels(column)
            )
    return out
