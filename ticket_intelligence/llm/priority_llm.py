"""
LLM path for ticket-priority prediction, using a local Ollama model
instead of a hosted API - no API key, no per-call cost, no external
network dependency.

Uses Ollama's structured-output mode (a JSON schema passed as `format`
on the /api/chat request) so the model is constrained to return one of
the four real training priority labels, never free text that would
need to be parsed out of a sentence.

Talks to Ollama's local HTTP API with the standard library only
(urllib + json) - no `requests` or `ollama` package. The call surface
is one POST of a JSON body to http://localhost:11434; a full HTTP
client dependency isn't justified for that.

A zero-shot version of this prompt scored 38.3% accuracy on the 240-row
LLM benchmark sample, against the classical baseline's 74.2% on the
same rows - the model had no way to know where THIS dataset draws the
line between, say, Medium and Low, so it defaulted to generic severity
intuition and was badly miscalibrated (see docs/INTERVIEW_NOTES.md). To fix
that, every request now includes a handful of labeled example tickets,
sampled once from data/train_pool_v1.csv - never from an eval file, so
the model gets a concrete anchor for this dataset's actual boundary
without any risk of leaking eval rows into the prompt.

Requires Ollama running locally with the model below already pulled:

    ollama pull llama3.2
"""

import json
import urllib.error
import urllib.request

from ticket_intelligence.priority_features import PRIORITY_LABELS, TARGET_COLUMN, sample_few_shot_rows
from ticket_intelligence.llm.priority_validation import (
    ProviderParseError,
    validate_priority_with_retry
)

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.2:latest"
REQUEST_TIMEOUT_SECONDS = 60

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "priority": {"type": "string", "enum": PRIORITY_LABELS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"}
    },
    "required": ["priority", "confidence", "reasoning"]
}

SYSTEM_PROMPT = (
    "You triage IT support tickets for an identity and access management "
    "product. Given a ticket's subject, description, and metadata, decide "
    "its priority: Critical, High, Medium, or Low. Base the decision on "
    "business impact - how many users are affected, whether it's "
    "production, whether a workaround exists - not on urgent-sounding "
    "language alone. The labeled examples that follow show how these "
    "four levels are actually used in this system - weigh them over your "
    "own assumptions about what 'sounds' urgent. Respond only with the "
    "structured output."
)


def _build_user_message(
    subject: str,
    description: str,
    category: str,
    environment: str,
    channel: str,
    region: str,
    customer_tier: str
) -> str:
    return (
        f"Subject: {subject}\n"
        f"Description: {description}\n"
        f"Category: {category}\n"
        f"Environment: {environment}\n"
        f"Channel: {channel}\n"
        f"Region: {region}\n"
        f"Customer tier: {customer_tier}"
    )


def _build_few_shot_messages() -> list:
    """
    Turns priority_features.sample_few_shot_rows() into alternating
    user/assistant messages - a user turn shaped exactly like a real
    prediction request, followed by an assistant turn giving the
    historically correct label. The reasoning text states only the
    known fact (this row's training label), never an invented detail
    about the ticket, consistent with the "don't invent case details"
    guardrail this project is heading toward for the real predictor.

    Uses the shared sampling function (not a local copy) so this
    provider and priority_llm_anthropic.py see identical examples -
    otherwise a difference in benchmark results could mean "different
    examples," not "different model."
    """

    messages = []

    for _, row in sample_few_shot_rows().iterrows():
        user_content = _build_user_message(
            subject=row["subject"],
            description=row["description"],
            category=row["category"],
            environment=row["environment"],
            channel=row["channel"],
            region=row["region"],
            customer_tier=row["customer_tier"]
        )
        assistant_content = json.dumps({
            "priority": row[TARGET_COLUMN],
            "confidence": 0.9,
            "reasoning": f"Historical training example labeled {row[TARGET_COLUMN]}."
        })

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

    return messages


# Built once at import time from the training pool - the examples don't
# change per request, so there's no reason to re-sample them on every
# call.
FEW_SHOT_MESSAGES = _build_few_shot_messages()


def predict_priority_llm(
    subject: str,
    description: str,
    category: str,
    environment: str,
    channel: str,
    region: str,
    customer_tier: str
) -> dict:
    """
    Predict priority for a ticket using a local Ollama model.

    Returns the same {"priority": ...} key as
    priority_service.predict_priority, plus a "confidence" float in
    [0, 1] and a "reasoning" string the classical model has no
    equivalent of.

    Every response is validated against PriorityResponse with one
    error-feedback retry, and fails closed - see priority_validation.py.
    """

    return validate_priority_with_retry(
        lambda correction: _call_ollama_raw(
            subject, description, category, environment,
            channel, region, customer_tier, correction
        )
    )


def _call_ollama_raw(
    subject: str,
    description: str,
    category: str,
    environment: str,
    channel: str,
    region: str,
    customer_tier: str,
    correction: str = None
) -> dict:
    """
    One Ollama request, returning the parsed content with no validation.

    `correction` carries the validation error from a rejected previous
    answer. It is appended as an extra user turn rather than edited
    into the system prompt, so the model sees its own bad answer's
    problem in conversational position - which is what makes a retry
    different from re-rolling the same dice.

    Raises RuntimeError if Ollama is unreachable (network, never
    retried) and ProviderParseError if a response came back in the
    wrong shape (retried once).
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_MESSAGES,
        {
            "role": "user",
            "content": _build_user_message(
                subject, description, category,
                environment, channel, region, customer_tier
            )
        }
    ]

    if correction:
        messages.append({"role": "user", "content": correction})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
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
    except (KeyError, TypeError) as exc:
        raise ProviderParseError(f"Unexpected Ollama response shape: {body}") from exc

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ProviderParseError(
            f"Ollama response content was not valid JSON: {content!r}"
        ) from exc
