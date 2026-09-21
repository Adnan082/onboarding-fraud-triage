"""The demo's inputs, checked against the same contract the API enforces.

The demo is the one screen a non-technical assessor sees, so its failure mode is
the worst kind: a preset that the API rejects looks like a broken model rather
than a rejected field. These tests fail at build time instead.

Nothing here starts Streamlit. The pieces worth testing are the payloads, and
they are plain dictionaries.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.demo import HEADLINE_FIELDS, PRESETS
from triage.api.schemas import ApplicationFeatures, example_application, request_fields


def test_the_starting_payload_is_accepted_by_the_api_model() -> None:
    """`example_application` is the form's default state and the benchmark's body."""
    ApplicationFeatures(**example_application())


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_is_contract_legal(name: str) -> None:
    """A preset the contract rejects would read as a broken model on screen."""
    features = {**example_application(), **PRESETS[name]}
    try:
        ApplicationFeatures(**features)
    except ValidationError as error:  # pragma: no cover - the message is the point
        pytest.fail(f"preset {name!r} is not a legal application: {error}")


@pytest.mark.parametrize("name", list(PRESETS))
def test_no_preset_invents_a_field(name: str) -> None:
    """`extra="forbid"` would reject it, but the error is clearer here."""
    unknown = set(PRESETS[name]) - set(request_fields())
    assert not unknown, f"preset {name!r} sets fields the contract does not have: {unknown}"


def test_the_headline_fields_exist_and_are_editable() -> None:
    """The form would raise a KeyError building a control for a field that is gone."""
    missing = [name for name in HEADLINE_FIELDS if name not in request_fields()]
    assert not missing, f"the demo asks for fields the contract does not have: {missing}"


def test_the_demo_never_offers_to_decline() -> None:
    """Rule 5: the riskiest band is verify, and the screen must not imply otherwise.

    Only the band names are checked. The explanatory line under a band is allowed
    to say "never a decline", which is the point rather than a breach of it.
    """
    from app.demo import BAND_STYLE

    assert set(BAND_STYLE) == {"approve", "review", "verify"}
    names = [headline for _, headline, _ in BAND_STYLE.values()]
    assert not any("declin" in name.lower() for name in names), names
    assert not any("reject" in name.lower() for name in names), names
