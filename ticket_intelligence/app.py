"""
FastAPI application: HTTP surface for the ticket-intelligence services.

This module contains routing only - no model logic, no scoring, no
prompt text. Every endpoint delegates to a service module
(priority_service, sentiment_service, summarization_service) or to the
escalation policy. Keeping it that way means the API can be re-shaped
without touching anything that was benchmarked.

Three families of endpoint:

  POST /triage                the main entry point. Takes a JSON body
                              via a Pydantic model, so a long ticket
                              description never travels in a URL (where
                              it would hit length limits and be written
                              to every access log in plaintext).

  GET /health, GET /schema    operational endpoints. /health reports
                              what this instance is serving; /schema
                              reports the exact field values the
                              trained model accepts.

  GET /predict-priority       legacy endpoints kept for backwards
  GET /analyze-sentiment      compatibility. index.html now uses
  GET /summarize-ticket       POST /triage.
"""

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ticket_intelligence.api_models import (
    MODEL_VOCABULARY,
    EscalationProvider,
    TriageRequest,
    TriageResponse
)
from ticket_intelligence.priority_features import METRICS_PATH, MODEL_PATH
from ticket_intelligence.priority_service import model, predict_priority
from ticket_intelligence.sentiment_service import analyze_sentiment
from ticket_intelligence.summarization_service import summarize_ticket

logger = logging.getLogger("ticket_intelligence")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# app.py lives in ticket_intelligence/; the page it serves lives in
# static/ at the repo root, one level up.
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="AI Support Ticket Intelligence Platform",
    version="1.2.0"
)

# Only the local dev origins the bundled index.html is served from.
# allow_origins=["*"] together with allow_credentials=True is rejected
# by browsers anyway, so the wildcard was never actually doing anything
# except looking permissive.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


def _assess_fn(provider: str):
    """
    Returns the escalation assess function for a provider name, or None
    for "none". Imported lazily so that starting the app - or calling
    /triage without escalation - never requires Ollama to be running or
    ANTHROPIC_API_KEY to be set.
    """

    if provider == "ollama":
        from ticket_intelligence.llm.escalation_llm import assess_escalation
        return assess_escalation

    if provider == "anthropic":
        from ticket_intelligence.llm.escalation_llm_anthropic import assess_escalation_anthropic
        return assess_escalation_anthropic

    return None


@app.get("/")
def home():
    """
    Open the HTML user interface.

    Served with no-store so an edit to index.html shows up on the next
    reload. Without it a browser will happily keep serving a cached
    copy, which during development looks exactly like "my change did
    nothing" - a slow and confusing way to lose an hour.
    """

    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store"}
    )


@app.get("/schema")
def schema_endpoint():
    """
    The accepted values for every categorical field, straight from the
    fitted encoder.

    index.html builds its dropdowns from this response instead of
    hard-coding option lists, so the form cannot drift out of sync with
    the model. Before this existed, the page offered "QA", "Test",
    "Portal" and "Basic" - none of which the model was trained on - and
    silently produced degraded predictions for any ticket using them.
    """

    return MODEL_VOCABULARY


@app.get("/health")
def health_endpoint():
    """
    Report what this instance is actually serving: whether the model
    artifacts are present, when they were trained, and how the model
    scored at training time.

    Takes no input. A health check reports the deployed state - a
    caller asking "are you healthy" does not already know the answer.

    Returns 503 if the model artifacts are missing, because an instance
    that cannot predict is not healthy, whatever else is running.
    """

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact missing at {MODEL_PATH.name}. Run 'python -m scripts.priority_train'."
        )

    payload = {
        "status": "ok",
        "api_version": app.version,
        "model_file": MODEL_PATH.name
    }

    # metrics.json is written by priority_train.py. Its absence isn't
    # fatal - the model still serves - so report what's known instead
    # of failing the whole health check.
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        payload.update(
            trained_at=metrics.get("trained_at"),
            label_classes=metrics.get("label_classes"),
            test_accuracy=metrics.get("test_accuracy"),
            n_train_rows=metrics.get("n_train_rows")
        )
    else:
        payload["metrics"] = "not available"

    return payload


@app.post("/triage", response_model=TriageResponse)
def triage_endpoint(request: TriageRequest):
    """
    Triage one ticket: predict its priority with the classical model,
    and optionally ask an LLM whether a human should look at it first.

    The two steps are deliberately independent. Priority comes from the
    classical model, which won the benchmark; escalation comes from an
    LLM, which is better suited to judging tone and intent. A provider
    being unreachable degrades the escalation answer, never the
    priority prediction.

    If escalation is requested and the provider fails, this fails safe
    and escalates - the same rule escalation_policy applies to a model
    that never returns a valid judgment.
    """

    result = predict_priority(
        subject=request.subject,
        description=request.description,
        category=request.category,
        environment=request.environment,
        channel=request.channel,
        region=request.region,
        customer_tier=request.customer_tier
    )

    response = TriageResponse(
        priority=result["priority"],
        confidence=result["confidence"]
    )

    assess_fn = _assess_fn(request.escalation_provider)
    if assess_fn is None:
        logger.info(
            "triage priority=%s confidence=%.4f escalation=skipped",
            response.priority, response.confidence
        )
        return response

    from ticket_intelligence.llm.escalation_policy import decide

    try:
        decision = decide(
            subject=request.subject,
            description=request.description,
            category=request.category,
            stated_priority=result["priority"],
            assess_fn=assess_fn
        )
        response.escalate = decision["escalate"]
        response.escalation_reason = decision["reason"]
        response.escalation_detail = decision.get("detail")
    except RuntimeError as exc:
        # Provider unreachable, or no API key. Not a parse failure, so
        # llm_validation deliberately doesn't retry it - and an
        # unanswered escalation question means a human should look.
        response.escalate = True
        response.escalation_reason = "provider_unavailable_fail_safe"
        response.escalation_detail = str(exc)

    logger.info(
        "triage priority=%s confidence=%.4f escalate=%s reason=%s provider=%s",
        response.priority, response.confidence, response.escalate,
        response.escalation_reason, request.escalation_provider
    )

    return response


# --- legacy endpoints, superseded by POST /triage ---------------------------


@app.get("/analyze-sentiment")
def analyze_sentiment_endpoint(text: str):
    """Analyze the sentiment of a support ticket."""

    return analyze_sentiment(text)


@app.get("/summarize-ticket")
def summarize_ticket_endpoint(text: str):
    """Summarize a support ticket description."""

    return {"summary": summarize_ticket(text)}


@app.get("/predict-priority")
def predict_priority_endpoint(
    subject: str,
    description: str,
    category: str,
    environment: str,
    channel: str,
    region: str,
    customer_tier: str
):
    """
    Predict priority for a new support ticket.

    Superseded by POST /triage. A long description in a query string can
    exceed URL length limits and is written to access logs in
    plaintext, which is why the new endpoint takes a body instead.
    """

    return predict_priority(
        subject=subject,
        description=description,
        category=category,
        environment=environment,
        channel=channel,
        region=region,
        customer_tier=customer_tier
    )
