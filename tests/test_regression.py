"""Pinned metrics on the fixture.

Required checks (CLAUDE.md section 13):
- headline metrics stay within tolerance of their recorded values.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="scaffold: implement alongside the module under test")


def test_placeholder() -> None:
    """Replace with the checks listed in this module's docstring."""
    raise AssertionError("not implemented")
