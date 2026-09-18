"""The frozen pandera contract.

Required checks (CLAUDE.md section 13):
- rejects out-of-range values;
- rejects unseen categories;
- rejects the income x 10 bug, in batch validation and at the API.

The API half of the last one lives in `test_api.py`; this file covers batch
validation. Everything here runs on the seeded fixture, so it needs no Kaggle data.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from triage.data.contract import (
    CONTRACT,
    CONTRACT_VERSION,
    SENTINEL_COLUMNS,
    ZERO_VARIANCE_COLUMNS,
    build_schema,
    check,
    validate,
    write_contract_doc,
)


def test_the_fixture_obeys_the_frozen_contract(fixture_frame: pd.DataFrame) -> None:
    """The stand-in for the real data must satisfy the same contract it does.

    This is what keeps the fixture honest: if it drifts away from the shape of
    BAF, the whole data-free test suite stops meaning anything.
    """
    result = check(fixture_frame, name="fixture")
    assert result.passed, result.failures


def test_contract_covers_every_fixture_column(fixture_frame: pd.DataFrame) -> None:
    """No column is silently outside the contract."""
    assert set(fixture_frame.columns) == set(CONTRACT)


# --- the bug that must never get through ---------------------------------------------


def test_income_times_ten_is_rejected(fixture_frame: pd.DataFrame) -> None:
    """Injected bug (d): out of contract, so batch validation must refuse it."""
    broken = fixture_frame.copy()
    broken["income"] = broken["income"] * 10

    result = check(broken, name="income x 10")
    assert not result.passed
    assert any("income" in failure for failure in result.failures)

    with pytest.raises(pa.errors.SchemaErrors):
        validate(broken)


def test_the_in_contract_bugs_are_accepted(fixture_frame: pd.DataFrame) -> None:
    """Bugs (a) to (c) are designed to pass the contract: that is the point of them.

    They are the ones the label-free monitor has to catch, because no schema check
    ever will.
    """
    swapped = fixture_frame.copy()
    top_two = swapped["employment_status"].value_counts().index[:2]
    swapped["employment_status"] = swapped["employment_status"].replace(
        {top_two[0]: top_two[1], top_two[1]: top_two[0]}
    )
    assert check(swapped, name="swap_employment").passed

    mirrored = fixture_frame.copy()
    levels = sorted(mirrored["income"].unique())
    mirrored["income"] = mirrored["income"].map(dict(zip(levels, reversed(levels), strict=True)))
    assert check(mirrored, name="mirror_income").passed

    constant = fixture_frame.copy()
    constant["phone_mobile_valid"] = 1
    assert check(constant, name="all_mobile_valid").passed


# --- ranges, categories, shape --------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("customer_age", 35),  # not a decade
        ("month", 12),  # outside 0-7
        ("name_email_similarity", 1.5),  # outside [0, 1]
        ("velocity_24h", 999_999.0),  # far outside the observed band
        ("credit_risk_score", -9_999),  # far below anything published
        ("email_is_free", 7),  # not a binary
    ],
)
def test_out_of_range_values_are_rejected(
    fixture_frame: pd.DataFrame, column: str, value: object
) -> None:
    """One bad value in one row is enough to fail the batch."""
    broken = fixture_frame.copy()
    broken.loc[broken.index[0], column] = value

    result = check(broken, name=f"{column}={value}")
    assert not result.passed
    assert any(column in failure for failure in result.failures)


@pytest.mark.parametrize(
    "column", ["payment_type", "employment_status", "housing_status", "source", "device_os"]
)
def test_unseen_categories_are_rejected(fixture_frame: pd.DataFrame, column: str) -> None:
    """An upstream code nobody has seen before is a breach, not a new level."""
    broken = fixture_frame.copy()
    broken.loc[broken.index[0], column] = "ZZ_NEW_CODE"

    result = check(broken, name=column)
    assert not result.passed
    assert any(column in failure for failure in result.failures)


def test_a_missing_column_is_rejected(fixture_frame: pd.DataFrame) -> None:
    """A dropped column is a breach."""
    assert not check(fixture_frame.drop(columns=["velocity_6h"]), name="dropped").passed


def test_an_unexpected_column_is_rejected(fixture_frame: pd.DataFrame) -> None:
    """Strict mode: the batch equivalent of the API's extra="forbid"."""
    extra = fixture_frame.copy()
    extra["surprise_feature"] = 1.0
    assert not check(extra, name="extra column").passed


def test_nulls_are_rejected(fixture_frame: pd.DataFrame) -> None:
    """BAF has no nulls: missing values arrive as negative sentinels."""
    broken = fixture_frame.copy()
    broken.loc[broken.index[0], "velocity_4w"] = None
    assert not check(broken, name="null").passed


# --- what the contract deliberately allows ---------------------------------------------


def test_negative_sentinels_are_allowed(fixture_frame: pd.DataFrame) -> None:
    """A -1 in a sentinel column is data, not a breach."""
    assert set(SENTINEL_COLUMNS) >= {
        "prev_address_months_count",
        "current_address_months_count",
        "bank_months_count",
        "session_length_in_minutes",
        "device_distinct_emails_8w",
        "intended_balcon_amount",
    }

    sentinel_frame = fixture_frame.copy()
    for column in SENTINEL_COLUMNS:
        if column != "intended_balcon_amount":
            sentinel_frame[column] = -1
    assert check(sentinel_frame, name="all sentinels").passed


def test_credit_risk_score_may_be_negative(fixture_frame: pd.DataFrame) -> None:
    """Negative scores are real values there, not missing ones.

    1.44% of the real Base data is negative, and 488 rows are exactly -1. Treating
    that -1 as a sentinel would corrupt a feature the model leans on heavily.
    """
    assert "credit_risk_score" not in SENTINEL_COLUMNS

    negative = fixture_frame.copy()
    negative["credit_risk_score"] = -150
    assert check(negative, name="negative credit risk").passed


def test_device_fraud_count_is_recorded_as_zero_variance() -> None:
    """It is all zeros in Base, so the feature layer drops it."""
    assert ZERO_VARIANCE_COLUMNS == ("device_fraud_count",)


# --- the artefact ------------------------------------------------------------------------


def test_schema_is_named_with_its_version() -> None:
    """A frozen contract has to be identifiable in a validation report."""
    assert build_schema().name == f"baf-contract-v{CONTRACT_VERSION}"
    assert len(build_schema().columns) == len(CONTRACT) == 32


def test_contract_doc_is_generated_not_typed(tmp_path) -> None:
    """reports/data_contract.md comes from the schema itself (rule 6)."""
    out_path = tmp_path / "data_contract.md"
    write_contract_doc(out_path)
    text = out_path.read_text(encoding="utf-8")

    assert f"Version **{CONTRACT_VERSION}**" in text
    for column in CONTRACT:
        assert f"`{column}`" in text


# --- against the real data, when it is present -------------------------------------------


@pytest.mark.data
@pytest.mark.parametrize("variant", ["base", "variant_iv", "variant_v"])
def test_real_variants_pass_the_contract(repo_root, variant: str) -> None:
    """The contract was frozen from Base; both variants must still satisfy it."""
    path = repo_root / "data" / "interim" / f"{variant}.parquet"
    if not path.exists():
        pytest.skip(f"{path} is missing: run the loader first")

    result = check(pd.read_parquet(path), name=variant)
    assert result.passed, result.failures
