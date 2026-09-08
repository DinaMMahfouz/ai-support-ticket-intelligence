"""
LLM path for ticket-priority prediction using the Anthropic API (Claude
Haiku) - a second, hosted provider alongside priority_llm.py's local
Ollama path, so the benchmark can compare classical baseline vs. small
local LLM vs. frontier hosted LLM on identical data.

Requires ANTHROPIC_API_KEY to be set in the environment (loaded from a
local .env file by priority_features - see _load_dotenv there). This
module never logs, prints, or embeds the key anywhere except the one
request header it authenticates.

Talks to the Messages API with the standard library only (urllib +
json) - no `anthropic` SDK. The call surface is one POST with two
static headers and a JSON body; that doesn't justify a dependency, same
reasoning as priority_llm.py's Ollama client.

Forces structured output via tool use: the request requires the model
to call a single tool (extract_priority) whose input schema matches
RESPONSE_SCHEMA, so an invalid label is structurally impossible to
receive, not just unlikely - the same guarantee priority_llm.py gets
from Ollama's `format` field, achieved the way this API supports it.

Few-shot examples are the *same* rows priority_llm.py uses
(priority_features.sample_few_shot_rows) so a difference in benchmark
results reflects the model, not different examples. They're formatted
as a flat text block in the system prompt rather than simulated
multi-turn tool calls - the simpler, idiomatic pattern for this API,
and it avoids fabricating synthetic tool_use turns just to mimic the
Ollama module's message shape.
"""

import json
import os
import urllib.error
import urllib.request

from ticket_intelligence.priority_features import PRIORITY_LABELS, TARGET_COLUMN, load_env, sample_few_shot_rows

load_env()

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
MODEL_NAME = "claude-haiku-4-5-20251001"
REQUEST_TIMEOUT_SECONDS = 60
MAX_TOKENS = 512

TOOL_NAME = "extract_priority"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "priority": {"type": "string", "enum": PRIORITY_LABELS},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string"}
    },
    "required": ["priority", "confidence", "reasoning"]
}

TOOLS = [
    {
        "name": TOOL_NAME,
        "description": "Record the predicted priority for this support ticket.",
        "input_schema": RESPONSE_SCHEMA
    }
]

BASE_SYSTEM_PROMPT = (
    "You triage IT support tickets for an identity and access management "
    "product. Given a ticket's subject, description, and metadata, decide "
    "its priority: Critical, High, Medium, or Low. Base the decision on "
    "business impact - how many users are affected, whether it's "
    "production, whether a workaround exists - not on urgent-sounding "
    "language alone. The labeled examples below show how these four "
    "levels are actually used in this system - weigh them over your own "
    "assumptions about what 'sounds' urgent."
)


def _build_ticket_text(
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


def _build_few_shot_block() -> str:
    lines = ["Labeled examples from this system's own history:"]

    for _, row in sample_few_shot_rows().iterrows():
        ticket_text = _build_ticket_text(
            subject=row["subject"],
            description=row["description"],
            category=row["category"],
            environment=row["environment"],
            channel=row["channel"],
            region=row["region"],
            customer_tier=row["customer_tier"]
        )
        lines.append(f"\n{ticket_text}\n-> Priority: {row[TARGET_COLUMN]}")

    return "\n".join(lines)


# Built once at import time from the training pool - the examples don't
# change per request, so there's no reason to re-sample them on every
# call.
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n\n" + _build_few_shot_block()


def predict_priority_llm_anthropic(
    subject: str,
    description: str,
    category: str,
    environment: str,
    channel: str,
    region: str,
    customer_tier: str
) -> dict:
    """
    Predict priority for a ticket using the Anthropic API.

    Returns the same {"priority": ...} key as
    priority_service.predict_priority and priority_llm.predict_priority_llm,
    plus "confidence" and "reasoning".
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
                "content": _build_ticket_text(
                    subject,
                    description,
                    category,
                    environment,
                    channel,
                    region,
                    customer_tier
                )
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

    tool_use_blocks = [block for block in body["content"] if block.get("type") == "tool_use"]
    if not tool_use_blocks:
        raise RuntimeError(f"No tool_use block in Anthropic response: {body}")

    parsed = tool_use_blocks[0]["input"]

    return {
        "priority": parsed["priority"],
        "confidence": parsed["confidence"],
        "reasoning": parsed["reasoning"]
    }
