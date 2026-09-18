"""The scoring service (section 10).

Required checks (CLAUDE.md section 13):
- 200 for valid input, 422 for invalid;
- changing customer_age does not change the score;
- the response carries drift status.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="scaffold: implement alongside the module under test")


def test_placeholder() -> None:
    """Replace with the checks listed in this module's docstring."""
    raise AssertionError("not implemented")
