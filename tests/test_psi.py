"""PSI in DuckDB and in pandas.

Required checks (CLAUDE.md section 13):
- known-answer PSI on hand-made bins;
- identical distributions give exactly 0.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from triage.monitoring.psi import (
    EPSILON,
    OTHER,
    categorical_levels,
    numeric_bins,
    psi,
    psi_from_counts,
    psi_table,
)


def _hand_psi(reference: list[float], window: list[float]) -> float:
    """The textbook sum, written out by hand for the known-answer test."""
    return sum((w - r) * math.log(w / r) for r, w in zip(reference, window, strict=True))


# --- known answers ------------------------------------------------------------------


def test_known_answer_on_hand_made_bins() -> None:
    """Two bins, 50/50 against 60/40, worked out by hand."""
    reference = pd.Series({"a": 500, "b": 500})
    window = pd.Series({"a": 600, "b": 400})

    expected = 0.1 * math.log(0.6 / 0.5) + (-0.1) * math.log(0.4 / 0.5)
    assert psi_from_counts(reference, window) == pytest.approx(expected)
    assert psi_from_counts(reference, window) == pytest.approx(0.040_546, abs=1e-6)


def test_known_answer_on_four_bins() -> None:
    """A second hand-computed case, with a bin that shrinks to nothing."""
    reference = pd.Series({"a": 250, "b": 250, "c": 250, "d": 250})
    window = pd.Series({"a": 400, "b": 300, "c": 300, "d": 0})

    expected = _hand_psi([0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.3, EPSILON])
    assert psi_from_counts(reference, window) == pytest.approx(expected)


def test_identical_distributions_give_exactly_zero() -> None:
    """Not "close to zero": every term is (p - p) * ln(p / p)."""
    counts = pd.Series({"a": 17, "b": 250, "c": 1})
    assert psi_from_counts(counts, counts) == 0.0

    rng = np.random.default_rng(0)
    column = pd.Series(rng.normal(size=5000))
    assert psi(column, column) == 0.0

    categorical = pd.Series(rng.choice(["AA", "AB", "AC"], size=5000))
    assert psi(categorical, categorical) == 0.0


def test_psi_is_symmetric() -> None:
    """Swapping reference and window leaves the value unchanged."""
    left = pd.Series({"a": 500, "b": 500})
    right = pd.Series({"a": 300, "b": 700})
    assert psi_from_counts(left, right) == pytest.approx(psi_from_counts(right, left))


def test_psi_grows_with_the_size_of_the_shift() -> None:
    """A bigger move has to score higher, or the detector is useless."""
    reference = pd.Series({"a": 500, "b": 500})
    small = psi_from_counts(reference, pd.Series({"a": 520, "b": 480}))
    large = psi_from_counts(reference, pd.Series({"a": 700, "b": 300}))
    assert 0 < small < large


def test_empty_side_raises() -> None:
    """An empty window is a pipeline failure, not a PSI of zero."""
    with pytest.raises(ValueError, match="at least one row"):
        psi_from_counts(pd.Series({"a": 10}), pd.Series(dtype=float))


# --- binning -------------------------------------------------------------------------


def test_numeric_bins_are_reference_deciles_open_at_both_ends() -> None:
    """Ten bins, running from -inf to +inf, so nothing falls outside."""
    reference = pd.Series(np.arange(1000, dtype=float))
    bins = numeric_bins(reference)

    assert len(bins) == 10
    assert bins["lo"].iloc[0] == -np.inf
    assert bins["hi"].iloc[-1] == np.inf
    assert list(bins["lo"][1:]) == list(bins["hi"][:-1]), "bins must tile without gaps"


def test_tied_quantiles_collapse_instead_of_making_empty_bins() -> None:
    """A column that is mostly one sentinel value gets fewer, real bins."""
    reference = pd.Series([-1.0] * 900 + list(np.arange(100, dtype=float)))
    bins = numeric_bins(reference)

    assert len(bins) < 10
    assert bins["bin"].is_unique


def test_a_value_beyond_the_reference_range_still_lands_in_a_bin() -> None:
    """A window can hold values the reference never saw; PSI must still compute."""
    reference = pd.Series(np.arange(100, dtype=float))
    window = pd.Series([-5000.0, 5000.0] * 50)
    assert psi(reference, window) > 0.25


def test_unseen_categories_go_to_other() -> None:
    """An upstream code that never appeared in training is not silently dropped."""
    reference = pd.Series(["AA"] * 60 + ["AB"] * 40)
    window = pd.Series(["AA"] * 50 + ["ZZ"] * 50)

    assert categorical_levels(reference) == ["AA", "AB", OTHER]
    assert psi(reference, window) > 0.25


# --- the DuckDB path agrees with the pandas path --------------------------------------


@pytest.fixture(scope="module")
def two_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A reference month and a shifted window, from the seeded fixture."""
    from tests.fixtures.make_fixture import make_fixture

    frame = make_fixture(n_rows=20_000, seed=20260917)
    return frame[frame["month"] <= 4], frame[frame["month"] == 7]


def test_duckdb_matches_pandas(two_windows: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The SQL that runs over real volumes agrees with the definition."""
    reference, window = two_windows
    features = ["velocity_6h", "income", "payment_type", "device_os", "credit_risk_score"]

    table = psi_table(reference, window, features=features).set_index("feature")
    for feature in features:
        assert table.loc[feature, "psi"] == pytest.approx(
            psi(reference[feature], window[feature]), abs=1e-12
        )


def test_duckdb_gives_zero_for_a_window_of_itself(
    two_windows: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Comparing a slice with itself must be exactly 0 in SQL too."""
    reference, _ = two_windows
    table = psi_table(reference, reference, features=["velocity_6h", "device_os"])
    assert (table["psi"] == 0.0).all()
    assert not table["watch"].any()


def test_psi_table_flags_a_planted_shift(two_windows: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """A column set to one constant is an obvious break, and must rank first."""
    reference, window = two_windows
    broken = window.copy()
    broken["phone_mobile_valid"] = 1  # injected bug (c)

    table = psi_table(reference, broken, features=["phone_mobile_valid", "velocity_24h"])
    assert table.iloc[0]["feature"] == "phone_mobile_valid"
    assert table.iloc[0]["alert"]


def test_psi_table_needs_shared_columns() -> None:
    """Comparing frames with nothing in common is a mistake, not an empty table."""
    with pytest.raises(ValueError, match="no shared columns"):
        psi_table(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"b": [1, 2]}), features=[])
