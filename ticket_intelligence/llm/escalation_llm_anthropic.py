"""
LLM path for escalation judgment using the Anthropic API (Claude
Haiku) - the hosted counterpart to escalation_llm.py's local Ollama
version. Same task, same zero-shot rationale, same forced-structured-
output approach as priority_llm_anthropic.py (tool use instead of
Ollama's `format` field).

See escalation_llm.py's module docstring for why this task is zero-shot
(hard_eval_set.csv is both the only labeled escalation data and the
eval set for it) and for the honesty caveat about the triage principles
below having been written after reading that file's hard_because notes.

Requires ANTHROPIC_API_KEY to be set in the environment (loaded from a
local .env file by priority_features).
"""

import json
import os
import urllib.error
import urllib.request

from ticket_intelligence.llm.llm_validation import ProviderParseError, validate_escalation_with_retry
from ticket_intelligence.priority_features import load_env

load_env()

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
MODEL_NAME = "claude-haiku-4-5-20251001"
REQUEST_TIMEOUT_SECONDS = 60
MAX_TOKENS = 512

TOOL_NAME = "assess_escalation"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "escalate": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string"}
    },
    "required": ["escalate", "confidence", "reasoning"]
}

TOOLS = [
    {
        "name": TOOL_NAME,
        "description": "Record whether this ticket needs human escalation before automated triage proceeds.",
        "input_schema": RESPONSE_SCHEMA
    }
]

SYSTEM_PROMPT = (
    "You triage IT support tickets for an identity and access management "
    "product and decide whether a ticket needs a human to review it "
    "before automated handling proceeds. Escalate to a human when: the "
    "ticket lacks enough technical detail to act on (no affected system, "
    "error text, or user count) even if the language sounds urgent; the "
    "customer states a likely cause with explicit uncertainty that could "
    "misdirect triage if it's wrong; the description suggests a possible "
    "security concern (an unrecognized access pattern, an unexpected "
    "privilege change) regardless of how calmly it's phrased; or a "
    "ticket reports an unexplained production incident that was never "
    "actually investigated, even if the customer says no action is "
    "needed now. Do not escalate purely because the language sounds "
    "urgent, and do not decline to escalate purely because the customer "
    "downplays the issue or says it's already resolved - judge the "
    "underlying facts stated, not the tone they're stated in. Base your "
    "decision only on what the ticket actually says - never assume "
    "facts, systems, or history that aren't stated."
)


def _build_ticket_text(subject: str, description: str, category: str, stated_priority: str) -> str:
    return (
        f"Subject: {subject}\n"
        f"Description: {description}\n"
        f"Category: {category}\n"
        f"Priority as recorded in the ticket system: {stated_priority}"
    )


def _call_anthropic_raw(
    subject: str,
    description: str,
    category: str,
    stated_priority: str
) -> dict:
    """
    Makes one Anthropic API request and returns the tool_use input as
    a raw dict - no validation of its shape or values.

    Raises RuntimeError if the API is unreachable or returns an HTTP
    error (a network failure, not a parse failure). Raises
    ProviderParseError if the response came back but had no tool_use
    block or the block was malformed - validate_escalation_with_retry
    retries on the latter, never the former.
    """

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Put it in a .env file at the "
            "repo root (ANTHROPIC_API_KEY=...) or export it in your shell."
        )

    payload = {
        "model": MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": _build_ticket_text(subject, description, category, stated_priority)
            }
        ],
        "tools": TOOLS,
        "tool_choice": {"type": "tool", "name": TOOL_NAME}
    }

    request = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API returned {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach the Anthropic API. Original error: {exc}"
        ) from exc

    tool_use_blocks = [block for block in body.get("content", []) if block.get("type") == "tool_use"]
    if not tool_use_blocks:
        raise ProviderParseError(f"No tool_use block in Anthropic response: {body}")

    try:
        return tool_use_blocks[0]["input"]
    except KeyError as exc:
        raise ProviderParseError(f"Malformed tool_use block: {tool_use_blocks[0]}") from exc


def assess_escalation_anthropic(
    subject: str,
    description: str,
    category: str,
    stated_priority: str
) -> dict:
    """
    Decide whether a ticket needs human escalation before automated
    triage continues, using the Anthropic API. Validated against
    EscalationResponse, with retry on parse/validation failure - see
    llm_validation.py.

    Returns {"escalate": "yes"|"no", "confidence": ..., "reasoning": ...}.
    Raises llm_validation.EscalationValidationError if the response
    never validates after every retry attempt.
    """

    return validate_escalation_with_retry(
        lambda: _call_anthropic_raw(subject, description, category, stated_priority)
    )
