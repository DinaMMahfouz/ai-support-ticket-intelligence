"""
Request and response models for the API boundary.

The problem this fixes: every endpoint used to take bare `str`
parameters, so `environment="Klingon"` was accepted, passed to a
OneHotEncoder built with handle_unknown="ignore", encoded as all-zeros,
and returned a confidently different prediction with no signal to the
caller. Measured on the sample ticket, changing environment from
"Production" to an unknown value moved the answer from High 0.83 to
Medium 0.57. Silently.

That is a guardrail failure, not a typing preference. An unknown value
should be a 422 at the boundary, not a degraded answer downstream.

The enums are built at import time from the fitted encoder rather than
written out by hand. Hand-written copies are how the web form ended up
offering "QA", "Portal" and "Basic" - four values the model had never
seen - while looking perfectly reasonable in the source. Deriving them
means a retrain that changes the label space changes the API contract
in the same move, and FastAPI's generated OpenAPI schema stays true
without anyone remembering to update it.
"""

from enum import Enum

from pydantic import BaseModel, Field

from ticket_intelligence.priority_features import CATEGORICAL_COLUMNS
from ticket_intelligence.priority_service import model

# Free-text caps. Real ticket descriptions are long but not unbounded;
# without a cap, a multi-megabyte body is accepted, vectorised, and
# turned into work. The numbers are generous rather than tight - the
# point is a defined refusal, not a tuned limit.
MAX_SUBJECT_CHARS = 500
MAX_DESCRIPTION_CHARS = 20000


def _vocabulary() -> dict:
    """The exact values the fitted OneHotEncoder was trained on."""

    encoder = model.named_steps["preprocessor"].named_transformers_["categorical"]

    return {
        column: list(values)
        for column, values in zip(CATEGORICAL_COLUMNS, encoder.categories_)
    }


MODEL_VOCABULARY = _vocabulary()


def _enum_for(column: str) -> type:
    """
    Build a str-valued Enum whose members are the trained categories.

    Member names are sanitised ("Web Form" -> WEB_FORM) but the values
    are the exact strings the encoder expects, which is what Pydantic
    validates against and what reaches the model.
    """

    members = {
        value.upper().replace(" ", "_").replace("-", "_"): value
        for value in MODEL_VOCABULARY[column]
    }

    return Enum(column.title().replace("_", "") + "Enum", members, type=str)


CategoryEnum = _enum_for("category")
EnvironmentEnum = _enum_for("environment")
ChannelEnum = _enum_for("channel")
RegionEnum = _enum_for("region")
CustomerTierEnum = _enum_for("customer_tier")


class EscalationProvider(str, Enum):
    NONE = "none"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


class TriageRequest(BaseModel):
    """
    One ticket to triage, as a JSON body.

    Closed-set fields are enums, so an unrecognised value is rejected
    with a 422 naming the field and listing what is allowed - instead
    of being silently discarded by the encoder.
    """

    model_config = {"use_enum_values": True}

    subject: str = Field(min_length=1, max_length=MAX_SUBJECT_CHARS)
    description: str = Field(min_length=1, max_length=MAX_DESCRIPTION_CHARS)
    category: CategoryEnum
    environment: EnvironmentEnum
    channel: ChannelEnum
    region: RegionEnum
    customer_tier: CustomerTierEnum
    escalation_provider: EscalationProvider = EscalationProvider.NONE


class TriageResponse(BaseModel):
    priority: str
    confidence: float
    escalate: bool | None = None
    escalation_reason: str | None = None
    escalation_detail: str | None = None
