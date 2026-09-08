"""
The retry-and-fail-closed contract.

These are the tests that turn docstring claims into facts. The most
important one is test_network_failure_is_not_retried: the distinction
between "the answer was malformed" (retry, often transient) and "the
provider is unreachable" (do not retry, nothing will change) is the
single most deliberate decision in the LLM path, and until now it
existed only as prose.
"""

import pytest
from pydantic import ValidationError

from ticket_intelligence.llm.priority_validation import (
    MAX_ATTEMPTS,
    Priority,
    PriorityResponse,
    PriorityValidationError,
    ProviderParseError,
    parse_stats,
    reset_parse_stats,
    validate_priority,
    validate_priority_with_retry,
)


@pytest.fixture(autouse=True)
def _clean_stats():
    reset_parse_stats()
    yield
    reset_parse_stats()


# --- the schema itself -----------------------------------------------------


def test_valid_response_passes(valid_response):
    result = validate_priority(valid_response)
    assert result["priority"] == "High"
    assert result["confidence"] == 0.82


@pytest.mark.parametrize("label", ["Critical", "High", "Medium", "Low"])
def test_all_four_labels_accepted(valid_response, label):
    valid_response["priority"] = label
    assert validate_priority(valid_response)["priority"] == label


@pytest.mark.parametrize("label", ["Urgent", "Sev1", "high", "MEDIUM", "P1", ""])
def test_labels_outside_the_four_are_rejected(valid_response, label):
    """A near-miss label is the failure mode that would silently corrupt
    a confusion matrix, so casing counts as wrong."""
    valid_response["priority"] = label
    with pytest.raises(ValidationError):
        validate_priority(valid_response)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, 2, -5])
def test_confidence_must_be_a_probability(valid_response, confidence):
    valid_response["confidence"] = confidence
    with pytest.raises(ValidationError):
        validate_priority(valid_response)


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_confidence_bounds_are_inclusive(valid_response, confidence):
    valid_response["confidence"] = confidence
    assert validate_priority(valid_response)["confidence"] == confidence


def test_empty_reasoning_is_rejected(valid_response):
    """Passes a JSON schema, tells a human reviewer nothing."""
    valid_response["reasoning"] = ""
    with pytest.raises(ValidationError):
        validate_priority(valid_response)


def test_missing_key_is_rejected(valid_response):
    del valid_response["confidence"]
    with pytest.raises(ValidationError):
        validate_priority(valid_response)


def test_priority_enum_covers_exactly_four_labels():
    assert [p.value for p in Priority] == ["Critical", "High", "Medium", "Low"]


# --- the retry contract ----------------------------------------------------


def test_succeeds_on_first_attempt_without_retrying(valid_response):
    calls = []

    def call_raw(correction):
        calls.append(correction)
        return valid_response

    result = validate_priority_with_retry(call_raw)

    assert result["priority"] == "High"
    assert calls == [None], "a valid first answer must not trigger a retry"
    assert parse_stats()["parse_failures"] == 0


def test_retries_once_then_succeeds(valid_response):
    calls = []

    def call_raw(correction):
        calls.append(correction)
        if len(calls) == 1:
            return {"priority": "Urgent", "confidence": 0.9, "reasoning": "bad label here"}
        return valid_response

    result = validate_priority_with_retry(call_raw)

    assert result["priority"] == "High"
    assert len(calls) == 2
    assert parse_stats()["parse_failures"] == 1


def test_retry_feeds_the_error_back():
    """A retry that does not say what was wrong is just re-rolling dice."""
    corrections = []

    def call_raw(correction):
        corrections.append(correction)
        return {"priority": "Urgent", "confidence": 0.9, "reasoning": "still bad here"}

    with pytest.raises(PriorityValidationError):
        validate_priority_with_retry(call_raw)

    assert corrections[0] is None
    assert corrections[1] is not None
    assert "Critical" in corrections[1]
    assert "rejected" in corrections[1].lower()


def test_fails_closed_never_defaults():
    """
    The caller must get an exception, not a label.

    Note the assertion is on the *return path*, not on the text of the
    error. An earlier version of this test asserted that "Medium" did
    not appear in the message, which failed for an innocent reason:
    Pydantic lists the permitted values in its own error text. Asserting
    on a message is a proxy; asserting that nothing was returned is the
    actual contract.
    """
    result = "sentinel"

    def call_raw(correction):
        return {"priority": "Sev1", "confidence": 5, "reasoning": "x"}

    with pytest.raises(PriorityValidationError):
        result = validate_priority_with_retry(call_raw)

    assert result == "sentinel", "a failed validation must not return a value"
    assert parse_stats()["hard_failures"] == 1


def test_stops_after_max_attempts():
    calls = []

    def call_raw(correction):
        calls.append(correction)
        raise ProviderParseError("not JSON")

    with pytest.raises(PriorityValidationError):
        validate_priority_with_retry(call_raw)

    assert len(calls) == MAX_ATTEMPTS


def test_network_failure_is_not_retried():
    """
    The load-bearing test. A parse failure is often transient, so we try
    again. An unreachable provider is not, so we fail immediately -
    retrying a connection that is down wastes time and hides the real
    error behind a validation message.
    """
    calls = []

    def call_raw(correction):
        calls.append(correction)
        raise RuntimeError("Could not reach Ollama at http://localhost:11434")

    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
        validate_priority_with_retry(call_raw)

    assert len(calls) == 1, "a network failure must not be retried"


def test_parse_failure_rate_is_per_call(valid_response):
    def good(correction):
        return valid_response

    def bad_then_good(correction):
        if correction is None:
            return {"priority": "Nope", "confidence": 0.5, "reasoning": "bad answer here"}
        return valid_response

    validate_priority_with_retry(good)
    validate_priority_with_retry(bad_then_good)

    stats = parse_stats()
    assert stats["calls"] == 2
    assert stats["parse_failures"] == 1
    assert stats["parse_failure_rate"] == 0.5
