"""Shared fixtures. Tests must run without the Kaggle data."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root."""
    return ROOT


@pytest.fixture(scope="session")
def fixture_frame():
    """A small, seeded, BAF-shaped frame with a known signal."""
    from tests.fixtures.make_fixture import make_fixture

    return make_fixture(n_rows=4000, seed=20260917)
