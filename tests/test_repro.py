"""Reproducibility.

Required checks (CLAUDE.md section 13):
- the same seed gives an identical hash of the scores.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="scaffold: implement alongside the module under test")


def test_placeholder() -> None:
    """Replace with the checks listed in this module's docstring."""
    raise AssertionError("not implemented")
