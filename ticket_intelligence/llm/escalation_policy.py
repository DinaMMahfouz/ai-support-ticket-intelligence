"""
Escalation policy: turns a validated model judgment into a final
escalate/don't-escalate decision, with a confidence-threshold safety
net and a fail-safe default when the model can't produce a validated
judgment at all.

Kept separate from escalation_llm.py/escalation_llm_anthropic.py on
purpose: those modules answer "what does the model think"; this module
answers "what does the system do about it." A low-confidence "no" is
exactly the case a guardrail exists to catch - the model wasn't sure,
so a human should look, even though the model didn't say to escalate.
That's a policy question, not the model's job to decide.
"""

from ticket_intelligence.llm.llm_validation import EscalationValidationError

LOW_CONFIDENCE_ESCALATES = True


def apply_policy(response: dict) -> dict:
    """
    Turns a validated {"escalate", "confidence", "reasoning"} dict
    (from escalation_llm.assess_escalation or
    escalation_llm_anthropic.assess_escalation_anthropic) into a final
    decision.

    Returns {"escalate": bool, "reason": str, "model_response": dict}.
    "reason" is "model_said_yes", "low_confidence_safety_net", or
    "model_said_no".
    """

    if response["escalate"] == "yes":
        return {"escalate": True, "reason": "model_said_yes", "model_response": response}

    if LOW_CONFIDENCE_ESCALATES and response["confidence"] == "low":
        return {"escalate": True, "reason": "low_confidence_safety_net", "model_response": response}

    return {"escalate": False, "reason": "model_said_no", "model_response": response}


def decide(subject: str, description: str, category: str, stated_priority: str, assess_fn) -> dict:
    """
    Calls assess_fn(subject, description, category, stated_priority) -
    assess_escalation or assess_escalation_anthropic - and applies the
    policy to its result.

    If the model never produces a validated response
    (EscalationValidationError, every retry exhausted), fails safe:
    escalates anyway rather than silently proceeding with no judgment
    at all or guessing one on the model's behalf.

    Returns {"escalate": bool, "reason": str, "model_response": dict | None}.
    "reason" additionally includes "validation_failed_fail_safe" for
    the failure path, in which case "model_response" is None and
    "detail" carries the validation failure text.
    """

    try:
        response = assess_fn(subject, description, category, stated_priority)
    except EscalationValidationError as exc:
        return {
            "escalate": True,
            "reason": "validation_failed_fail_safe",
            "model_response": None,
            "detail": str(exc)
        }

    return apply_policy(response)
