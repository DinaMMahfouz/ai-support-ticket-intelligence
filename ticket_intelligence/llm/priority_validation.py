"""
Pydantic validation and retry for LLM *priority* predictions.

Why this exists when each provider already constrains its output:
Ollama's `format` field and Anthropic's forced `tool_choice` both
constrain generation at the API level, but they are two different
mechanisms with two different failure modes, and neither is a promise
about the parsed Python object. A provider can return a
schema-conformant but useless response (confidence 1.0 with empty
reasoning), a label with different casing, or - on a bad day - a body
that is not JSON at all. PriorityResponse is the single place that
decides what "a valid priority prediction" means, for every provider.

Two rules that matter more than the code:

1. Retry feeds the error back. A bare retry re-rolls the same dice. A
   retry that tells the model what was wrong with its last answer is
   the only kind worth the extra call, so `call_raw` takes an optional
   `correction` string and providers put it in the conversation.

2. Failure is closed, never defaulted. If validation never succeeds
   this raises. It does not fall back to "Medium", and it does not
   return a low-confidence guess. A silently defaulted prediction is
   worse than a refusal because it enters the evaluation as a real
   answer and quietly moves every metric - a refusal is at least
   visible and countable.

Parse failures are counted, because the parse-failure rate is itself a
result worth reporting: a model that needs two attempts on 15% of
tickets costs 1.15x and has a reliability problem that accuracy alone
will not show.
"""

from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel, Field, ValidationError

# One initial attempt plus one retry with the error fed back. Two is a
# deliberate choice rather than a tunable: if a model cannot produce a
# schema-conformant answer when told exactly what it got wrong, more
# attempts are unlikely to help and the latency is real.
MAX_ATTEMPTS = 2


class Priority(str, Enum):
    """The only four labels that exist. Anything else is a parse failure."""

    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class PriorityResponse(BaseModel):
    """
    A validated priority prediction.

    confidence is a float in [0, 1] rather than a low/medium/high enum
    because the escalation threshold needs to be swept continuously -
    three buckets cannot produce a tradeoff curve. Worth stating the
    caveat plainly: a self-reported float from a language model is not
    a calibrated probability, and evaluation/threshold_sweep.py exists
    to measure how far off it is rather than to assume it is fine.

    reasoning has a minimum length because an empty string passes a
    JSON schema while telling a human reviewer nothing, and the whole
    point of the escalation path is that a human reads this.
    """

    priority: Priority
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=10, max_length=2000)


class ProviderParseError(Exception):
    """
    Raised by a provider's raw-call function when a response came back
    but its shape was wrong - not valid JSON, no tool_use block,
    missing keys. Distinct from RuntimeError, which providers raise for
    network-level failures and which is deliberately never retried
    here.
    """


class PriorityValidationError(RuntimeError):
    """Raised when no attempt produced a valid PriorityResponse."""


# Module-level counters. Deliberately simple: this is a portfolio
# project with one process, not a service that needs a metrics backend.
# reset_parse_stats() exists so tests and separate eval runs start
# from zero.
_STATS = {"attempts": 0, "parse_failures": 0, "calls": 0, "hard_failures": 0}


def reset_parse_stats() -> None:
    for key in _STATS:
        _STATS[key] = 0


def parse_stats() -> dict:
    """
    Returns counts plus the derived rate.

    parse_failure_rate is per *call*, not per attempt: "how often did a
    ticket need at least one retry" is the number that maps to cost and
    to reliability, which is what a reader of the eval output wants.
    """

    calls = _STATS["calls"]

    return {
        **_STATS,
        "parse_failure_rate": (_STATS["parse_failures"] / calls) if calls else 0.0,
        "hard_failure_rate": (_STATS["hard_failures"] / calls) if calls else 0.0
    }


def validate_priority(raw: dict) -> dict:
    """
    Validate one raw provider dict. Raises ValidationError on failure.
    Split out from the retry loop so tests can exercise the schema
    without a provider.
    """

    return PriorityResponse(**raw).model_dump(mode="json")


def validate_priority_with_retry(
    call_raw: Callable[[Optional[str]], dict]
) -> dict:
    """
    Call a provider, validate its answer, and retry once with the
    validation error fed back before failing closed.

    `call_raw(correction)` makes exactly one provider request and
    returns the parsed dict with no validation. On the first attempt
    `correction` is None; on the retry it is a plain-English
    description of what was wrong with the previous answer, which the
    provider is expected to include in the conversation.

    Returns the validated response as a plain dict.

    Raises PriorityValidationError if no attempt validates - never a
    default label. Does NOT catch RuntimeError: a network failure
    propagates on the first attempt, because retrying an unreachable
    provider is not a parse retry.
    """

    _STATS["calls"] += 1

    failures = []
    correction = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        _STATS["attempts"] += 1
        try:
            raw = call_raw(correction)
            return validate_priority(raw)
        except (ProviderParseError, ValidationError, TypeError, KeyError) as exc:
            _STATS["parse_failures"] += 1
            message = f"{type(exc).__name__}: {exc}"
            failures.append(f"attempt {attempt}: {message}")
            correction = (
                "Your previous response was rejected. "
                f"{message}. "
                "Respond again with valid structured output: priority must be "
                "exactly one of Critical, High, Medium, Low; confidence must "
                "be a number between 0 and 1; reasoning must be at least 10 "
                "characters."
            )

    _STATS["hard_failures"] += 1

    raise PriorityValidationError(
        f"No valid priority response after {MAX_ATTEMPTS} attempts:\n    "
        + "\n    ".join(failures)
    )
