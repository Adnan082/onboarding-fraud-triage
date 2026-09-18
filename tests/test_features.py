"""Sentinel flags and feature encoding.

The rule under test throughout is rule 4: the champion must not see
``customer_age``, by any route, including one-hot columns and LightGBM categories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from triage.features.encode import build_encoder, feature_columns, lgbm_frame, prepare
from triage.features.sentinels import (
    MISSING_SUFFIX,
    add_missing_flags,
    flag_columns,
    sentinel_report,
    zero_variance_columns,
)


@pytest.fixture(scope="module")
def cfg(repo_root):
    """The composed project config."""
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        return compose(config_name="config")


# --- sentinels ---------------------------------------------------------------------


def test_minus_one_becomes_a_flag_and_a_blank(fixture_frame: pd.DataFrame, cfg) -> None:
    """A -1 count is missing: flagged, then blanked so nothing reads it as a number."""
    flagged = add_missing_flags(fixture_frame, cfg)
    column = "prev_address_months_count"

    was_missing = fixture_frame[column] == -1
    assert was_missing.any(), "the fixture should contain sentinels to flag"
    assert (flagged[f"{column}{MISSING_SUFFIX}"] == was_missing.astype(int)).all()
    assert flagged.loc[was_missing, column].isna().all()
    assert (flagged.loc[~was_missing, column] == fixture_frame.loc[~was_missing, column]).all()


def test_any_negative_balcon_is_missing(fixture_frame: pd.DataFrame, cfg) -> None:
    """intended_balcon_amount is missing whenever it is negative, not only at -1."""
    flagged = add_missing_flags(fixture_frame, cfg)
    negative = fixture_frame["intended_balcon_amount"] < 0

    assert negative.any()
    assert (flagged[f"intended_balcon_amount{MISSING_SUFFIX}"] == negative.astype(int)).all()
    assert flagged.loc[negative, "intended_balcon_amount"].isna().all()


def test_credit_risk_score_is_never_treated_as_missing(fixture_frame: pd.DataFrame, cfg) -> None:
    """DECISIONS D10: negative scores there are real, including exactly -1.

    Getting this wrong would blank 1.44% of the real column, one of the strongest
    features the model has.
    """
    frame = fixture_frame.copy()
    frame.loc[frame.index[:5], "credit_risk_score"] = -1
    flagged = add_missing_flags(frame, cfg)

    assert f"credit_risk_score{MISSING_SUFFIX}" not in flagged.columns
    assert (flagged["credit_risk_score"] == frame["credit_risk_score"]).all()
    assert flagged["credit_risk_score"].notna().all()


def test_velocity_6h_is_flagged_but_keeps_its_value(fixture_frame: pd.DataFrame, cfg) -> None:
    """A published quirk is reported, not filled in."""
    frame = fixture_frame.copy()
    frame.loc[frame.index[:3], "velocity_6h"] = -170.6
    flagged = add_missing_flags(frame, cfg)

    assert flagged["velocity_6h_negative"].sum() == 3
    assert (flagged["velocity_6h"] == frame["velocity_6h"]).all()
    assert flagged["velocity_6h"].notna().all()


def test_flagging_does_not_modify_the_input(fixture_frame: pd.DataFrame, cfg) -> None:
    """Callers keep their frame; the sentinel pass returns a copy."""
    before = fixture_frame.copy()
    add_missing_flags(fixture_frame, cfg)
    pd.testing.assert_frame_equal(fixture_frame, before)


def test_a_missing_sentinel_column_raises(fixture_frame: pd.DataFrame, cfg) -> None:
    """Silently skipping a sentinel would change what the model sees."""
    with pytest.raises(KeyError, match="bank_months_count"):
        add_missing_flags(fixture_frame.drop(columns=["bank_months_count"]), cfg)


def test_flag_columns_matches_what_is_added(fixture_frame: pd.DataFrame, cfg) -> None:
    """The declared flag list is the one the encoder relies on."""
    flagged = add_missing_flags(fixture_frame, cfg)
    assert set(flag_columns(cfg)) <= set(flagged.columns)
    assert len(flag_columns(cfg)) == 7  # five -1 columns, balcon, and the velocity quirk


def test_zero_variance_finds_device_fraud_count(fixture_frame: pd.DataFrame) -> None:
    """It is 0 for all million rows of Base, and all rows of the fixture."""
    assert zero_variance_columns(fixture_frame) == ["device_fraud_count"]


def test_sentinel_report_shares(fixture_frame: pd.DataFrame, cfg) -> None:
    """The EDA table reports a share per flag, between 0 and 1."""
    report = sentinel_report(fixture_frame, cfg)
    assert list(report["column"]) == flag_columns(cfg)
    assert ((report["share_flagged"] >= 0) & (report["share_flagged"] <= 1)).all()


# --- rule 4: age must not reach the model --------------------------------------------


def test_age_is_absent_when_use_age_is_false(cfg) -> None:
    """The champion's column list has no age in it."""
    assert "customer_age" not in feature_columns(cfg, use_age=False)
    assert "customer_age" in feature_columns(cfg, use_age=True)


def test_prepare_drops_age_and_the_label(fixture_frame: pd.DataFrame, cfg) -> None:
    """Neither the label, the split key, nor age survives into the model input."""
    prepared = prepare(fixture_frame, cfg, use_age=False)
    for column in ("customer_age", "fraud_bool", "month", "device_fraud_count"):
        assert column not in prepared.columns


def test_changing_age_cannot_change_the_champion_input(fixture_frame: pd.DataFrame, cfg) -> None:
    """The API test asserts the same thing about scores; this is the feature-level check."""
    other = fixture_frame.copy()
    other["customer_age"] = 90

    pd.testing.assert_frame_equal(
        prepare(fixture_frame, cfg, use_age=False), prepare(other, cfg, use_age=False)
    )


def test_one_hot_encoding_has_no_age_column(fixture_frame: pd.DataFrame, cfg) -> None:
    """Not even as a one-hot column name."""
    encoder = build_encoder(cfg, use_age=False)
    encoder.fit(prepare(fixture_frame, cfg, use_age=False))
    names = encoder.get_feature_names_out()
    assert not any("customer_age" in str(name) for name in names)


# --- encoding ------------------------------------------------------------------------


def test_encoder_is_fit_on_training_months_only(fixture_frame: pd.DataFrame, cfg) -> None:
    """Fitting on train and transforming test must give the same shape, not refit.

    The scaler's centre comes from the training months alone: if it moved with the
    evaluation data, every reported number would be contaminated.
    """
    train = fixture_frame[fixture_frame["month"] <= 4]
    test = fixture_frame[fixture_frame["month"] == 6]

    encoder = build_encoder(cfg, use_age=False)
    transformed_train = encoder.fit_transform(prepare(train, cfg, use_age=False))
    transformed_test = encoder.transform(prepare(test, cfg, use_age=False))

    assert transformed_train.shape[1] == transformed_test.shape[1]
    assert np.isfinite(transformed_train).all()
    assert np.isfinite(transformed_test).all(), "imputation should leave no NaN for B0"
    # Standardised on train: the training mean is ~0, the test mean is not forced to be.
    numeric_count = len(
        [c for c in feature_columns(cfg, use_age=False) if c not in cfg.features.categoricals]
    )
    assert abs(transformed_train[:, :numeric_count].mean()) < 1e-6


def test_unseen_category_encodes_as_all_zeros(fixture_frame: pd.DataFrame, cfg) -> None:
    """An upstream code the training months never saw must not shift the columns."""
    train = fixture_frame[fixture_frame["month"] <= 4]
    encoder = build_encoder(cfg, use_age=False)
    encoder.fit(prepare(train, cfg, use_age=False))

    unseen = fixture_frame.head(5).copy()
    unseen["device_os"] = "an_os_nobody_has_seen"
    transformed = encoder.transform(prepare(unseen, cfg, use_age=False))

    names = list(encoder.get_feature_names_out())
    device_columns = [i for i, name in enumerate(names) if "device_os" in str(name)]
    assert device_columns
    assert transformed[:, device_columns].sum() == 0


def test_lgbm_frame_uses_contract_categories(fixture_frame: pd.DataFrame, cfg) -> None:
    """Category levels come from the frozen contract, not from the slice."""
    one_month = fixture_frame[fixture_frame["month"] == 7]
    frame = lgbm_frame(one_month, cfg, use_age=False)

    assert list(frame["device_os"].cat.categories) == [
        "windows",
        "macintosh",
        "linux",
        "x11",
        "other",
    ]
    assert list(frame["source"].cat.categories) == ["INTERNET", "TELEAPP"]


def test_lgbm_encoding_is_identical_across_slices(fixture_frame: pd.DataFrame, cfg) -> None:
    """Two different months encode a category to the same code."""
    first = lgbm_frame(fixture_frame[fixture_frame["month"] == 0], cfg, use_age=False)
    second = lgbm_frame(fixture_frame[fixture_frame["month"] == 7], cfg, use_age=False)

    for column in cfg.features.categoricals:
        assert list(first[column].cat.categories) == list(second[column].cat.categories)


def test_lgbm_frame_keeps_nan_for_sentinels(fixture_frame: pd.DataFrame, cfg) -> None:
    """LightGBM handles missing natively, so the blanks stay blank."""
    frame = lgbm_frame(fixture_frame, cfg, use_age=False)
    assert frame["prev_address_months_count"].isna().any()
    assert frame["prev_address_months_count_missing"].notna().all()
