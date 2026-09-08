"""
Pydantic validation and retry for LLM provider responses, used by
escalation_llm.py and escalation_llm_anthropic.py.

Ollama's `format` field and Anthropic's forced tool_choice already
constrain each response to match a JSON schema at the API level, but
that's not the same guarantee as validating the *parsed* result in
Python: a provider can still return a schema-conformant but degenerate
response (an empty reasoning string), or the raw HTTP response can fail
to parse as JSON at all if a provider has a bad day. EscalationResponse
is the one place that decides what "a valid escalation response"
actually means, instead of trusting each provider's raw dict.

Retry here is scoped specifically to parse/validation failures - the
response came back but its content was wrong shape, missing a key, or
failed schema validation. A network-level failure (the provider is
unreachable, or returns an HTTP error) is a different problem and is
not retried here: both provider modules raise plain RuntimeError for
that, which propagates immediately rather than being retried with no
backoff against a connection that's actually down.
"""

from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError

MAX_ATTEMPTS = 3


class EscalationResponse(BaseModel):
    escalate: Literal["yes", "no"]
    confidence: Literal["low", "medium", "high"]
    reasoning: str = Field(min_length=10)


class ProviderParseError(Exception):
    """
    Raised by a provider's raw-call function when the response came
    back but its shape was wrong - missing keys, no tool_use block,
    body that wasn't valid JSON. Distinct from RuntimeError (used for
    network-level failures), which validate_escalation_with_retry
    deliberately does not catch.
    """


class EscalationValidationError(RuntimeError):
    """Raised when a provider's response never validates after every retry attempt."""


def validate_escalation_with_retry(call_raw: Callable[[], dict]) -> dict:
    """
    Calls call_raw() (one provider request, returning a raw parsed
    dict with no validation) up to MAX_ATTEMPTS times, validating each
    result against EscalationResponse. Returns the validated response
    as a plain dict on the first attempt that validates.

    Raises EscalationValidationError, with every attempt's failure
    reason attached, if no attempt ever validates. Does NOT catch
    RuntimeError - a network-level failure from call_raw propagates
    immediately, on the first attempt, since retrying an unreachable
    provider isn't a parse-failure retry.
    """

    failures = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = call_raw()
            validated = EscalationResponse(**raw)
            return validated.model_dump()
        except (ProviderParseError, ValidationError, TypeError) as exc:
            failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")

    raise EscalationValidationError(
        f"Provider response never validated after {MAX_ATTEMPTS} attempts:\n    "
        + "\n    ".join(failures)
    )
