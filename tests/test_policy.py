"""Conformal set -> decision band.

Required checks (CLAUDE.md section 13):
- {legit} approves, {fraud} verifies, {legit, fraud} and {} both go to review;
- no input ever produces an automatic decline.
"""

from __future__ import annotations

import numpy as np
import pytest

from triage.policy.decide import band_shares, decide, decide_many


@pytest.mark.parametrize(
    ("has_legit", "has_fraud", "expected"),
    [
        (True, False, "approve"),
        (True, True, "review"),
        (False, False, "review"),  # the empty set goes to review, never to decline
        (False, True, "verify"),
    ],
)
def test_set_to_decision(has_legit: bool, has_fraud: bool, expected: str) -> None:
    """Every conformal set maps to its documented band."""
    assert decide(has_legit, has_fraud) == expected


def test_never_declines() -> None:
    """Rule 5: the riskiest band is verify. Nothing is ever declined automatically."""
    bands = {decide(a, b) for a in (True, False) for b in (True, False)}
    assert bands == {"approve", "review", "verify"}


def test_decide_many_matches_decide() -> None:
    """The vectorised mapping agrees with the scalar one, row by row."""
    rng = np.random.default_rng(11)
    sets = rng.random((500, 2)) < 0.5

    vectorised = decide_many(sets)
    scalar = [decide(bool(row[0]), bool(row[1])) for row in sets]
    assert list(vectorised) == scalar


def test_band_shares_sum_to_one() -> None:
    """Every application lands in exactly one band."""
    rng = np.random.default_rng(12)
    shares = band_shares(decide_many(rng.random((1000, 2)) < 0.5))
    assert sum(shares.values()) == pytest.approx(1.0)
    assert set(shares) == {"approve", "review", "verify"}


def test_band_shares_of_nothing() -> None:
    """An empty slice reports zeros rather than dividing by zero."""
    assert band_shares(np.array([])) == {"approve": 0.0, "review": 0.0, "verify": 0.0}
