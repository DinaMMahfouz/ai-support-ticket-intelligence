"""
LLM path for escalation judgment: does this ticket need a human before
automated triage proceeds? Local Ollama model, mirroring priority_llm.py's
approach (structured output via Ollama's `format` field, stdlib-only
HTTP client).

This is a deliberately different task from priority_llm.py's priority
classification - see docs/INTERVIEW_NOTES.md for why the LLM path was
repositioned away from competing with the classical model on priority
(it lost by a wide margin, on identical data, even with a stronger
model and more examples) and toward this instead. Escalation judgment -
recognizing sarcasm, a customer downplaying their own outage, a
customer's likely-wrong stated cause - is closer to what LLM reasoning
is actually suited for, and is exactly what data/hard_eval_set.csv was
built to test.

No few-shot examples: hard_eval_set.csv (18 rows) is the only labeled
escalation data that exists, and it's the eval set for this task -
using any of its rows as examples would mean scoring the model partly
on cases it had already seen the answer to. Zero-shot with general
triage principles is the only option that doesn't contaminate the only
eval data available here.

Caveat worth being honest about: the principles below, while meant as
general triage judgment rather than anything specific to
hard_eval_set.csv, were written after reading all 18 of that file's
`hard_because` notes earlier in this project. That makes this a
best-effort blind eval, not a provably blind one - a genuinely clean
test would need principles written before ever seeing the file, or a
fresh batch of hard cases for later validation.
"""

import json
import urllib.error
import urllib.request

from ticket_intelligence.llm.llm_validation import ProviderParseError, validate_escalation_with_retry

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.2:latest"
REQUEST_TIMEOUT_SECONDS = 60

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "escalate": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string"}
    },
    "required": ["escalate", "confidence", "reasoning"]
}

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
    "facts, systems, or history that aren't stated. Respond only with "
    "the structured output."
)


def _build_user_message(subject: str, description: str, category: str, stated_priority: str) -> str:
    return (
        f"Subject: {subject}\n"
        f"Description: {description}\n"
        f"Category: {category}\n"
        f"Priority as recorded in the ticket system: {stated_priority}"
    )


def _call_ollama_raw(
    subject: str,
    description: str,
    category: str,
    stated_priority: str
) -> dict:
    """
    Makes one Ollama request and returns the parsed response content
    as a raw dict - no validation of its shape or values.

    Raises RuntimeError if Ollama is unreachable (a network failure,
    not a parse failure). Raises ProviderParseError if the response
    came back but its content wasn't valid JSON or was missing the
    expected key - llm_validation.validate_escalation_with_retry
    retries on the latter, never the former.
    """

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_message(subject, description, category, stated_priority)
            }
        ],
        "stream": False,
        "format": RESPONSE_SCHEMA
    }

    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_HOST}. Is it running? "
            f"Original error: {exc}"
        ) from exc

    try:
        content = body["message"]["content"]
    except KeyError as exc:
        raise ProviderParseError(f"Unexpected Ollama response shape: {body}") from exc

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ProviderParseError(f"Ollama response content was not valid JSON: {content!r}") from exc


def assess_escalation(
    subject: str,
    description: str,
    category: str,
    stated_priority: str
) -> dict:
    """
    Decide whether a ticket needs human escalation before automated
    triage continues. Validated against EscalationResponse, with retry
    on parse/validation failure - see llm_validation.py.

    Returns {"escalate": "yes"|"no", "confidence": ..., "reasoning": ...}.
    Raises llm_validation.EscalationValidationError if the response
    never validates after every retry attempt.
    """

    return validate_escalation_with_retry(
        lambda: _call_ollama_raw(subject, description, category, stated_priority)
    )
