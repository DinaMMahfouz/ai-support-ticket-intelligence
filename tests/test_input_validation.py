"""
Input validation at the API boundary.

The bug these lock down: `environment="Klingon"` used to be accepted,
handed to a OneHotEncoder built with handle_unknown="ignore", encoded
as all-zeros, and returned as a confidently different prediction with
no signal to the caller. On the sample ticket that moved the answer
from High 0.83 to Medium 0.57. Silently.

An unknown value must now be a 422 at the boundary. These tests assert
that, and they assert the enums are derived from the fitted model
rather than hand-copied - because a hand-copied list is what put "QA",
"Portal" and "Basic" in the web form for months.
"""

import pytest
from pydantic import ValidationError

from ticket_intelligence.api_models import (
    MAX_DESCRIPTION_CHARS,
    MAX_SUBJECT_CHARS,
    MODEL_VOCABULARY,
    TriageRequest,
)


def _valid_payload(**overrides) -> dict:
    payload = {
        "subject": "Production authentication failure",
        "description": "Multiple users cannot authenticate after a config change.",
        "category": MODEL_VOCABULARY["category"][0],
        "environment": MODEL_VOCABULARY["environment"][0],
        "channel": MODEL_VOCABULARY["channel"][0],
        "region": MODEL_VOCABULARY["region"][0],
        "customer_tier": MODEL_VOCABULARY["customer_tier"][0],
    }
    payload.update(overrides)
    return payload


def test_valid_payload_is_accepted():
    request = TriageRequest(**_valid_payload())
    assert request.escalation_provider == "none"


@pytest.mark.parametrize("field", list(MODEL_VOCABULARY))
def test_every_trained_value_is_accepted(field):
    """Whatever the model was trained on, the API must accept."""
    for value in MODEL_VOCABULARY[field]:
        TriageRequest(**_valid_payload(**{field: value}))


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("environment", "Klingon"),
        ("environment", "QA"),          # was in the web form, never in the model
        ("environment", "Test"),        # ditto - the model knows "Testing"
        ("channel", "Portal"),          # ditto
        ("customer_tier", "Basic"),     # ditto
        ("category", "Nonsense"),
        ("region", "Atlantis"),
    ],
)
def test_unknown_values_are_rejected(field, bad_value):
    with pytest.raises(ValidationError) as excinfo:
        TriageRequest(**_valid_payload(**{field: bad_value}))
    assert field in str(excinfo.value)


def test_enums_match_the_fitted_model_exactly():
    """
    Guards against the failure mode that started all this: a hand-written
    option list drifting from what the model was actually trained on.
    """
    from ticket_intelligence.api_models import (
        CategoryEnum, ChannelEnum, CustomerTierEnum, EnvironmentEnum, RegionEnum
    )

    pairs = [
        ("category", CategoryEnum), ("environment", EnvironmentEnum),
        ("channel", ChannelEnum), ("region", RegionEnum),
        ("customer_tier", CustomerTierEnum),
    ]
    for field, enum_cls in pairs:
        assert sorted(m.value for m in enum_cls) == sorted(MODEL_VOCABULARY[field])


# --- free text boundaries --------------------------------------------------


@pytest.mark.parametrize("field", ["subject", "description"])
def test_empty_free_text_is_rejected(field):
    with pytest.raises(ValidationError):
        TriageRequest(**_valid_payload(**{field: ""}))


def test_oversized_description_is_rejected():
    """A 500KB body should be a defined refusal, not silent work."""
    with pytest.raises(ValidationError):
        TriageRequest(**_valid_payload(description="x" * 500_000))


def test_description_at_the_cap_is_accepted():
    TriageRequest(**_valid_payload(description="x" * MAX_DESCRIPTION_CHARS))


def test_subject_over_the_cap_is_rejected():
    with pytest.raises(ValidationError):
        TriageRequest(**_valid_payload(subject="x" * (MAX_SUBJECT_CHARS + 1)))


def test_unknown_escalation_provider_is_rejected():
    with pytest.raises(ValidationError):
        TriageRequest(**_valid_payload(escalation_provider="gpt"))
