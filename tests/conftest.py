"""
Shared fixtures.

The design rule for this suite: anything that needs a live model is
marked `llm` and deselected by default. Everything else - schema
validation, the retry contract, the fabrication detector, request
validation - is pure Python and runs on every commit.

That split is deliberate. A guardrail test that only runs when someone
remembers to start Ollama is not a guardrail, it is a script.
"""

import os
import urllib.request

import pytest


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    return _ollama_up()


@pytest.fixture(scope="session")
def anthropic_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.fixture
def valid_response() -> dict:
    return {
        "priority": "High",
        "confidence": 0.82,
        "reasoning": "Production authentication failure affecting multiple users."
    }
