"""
Serves ticket-priority predictions from artifacts saved by priority_train.py.

This module only loads pre-trained files - it never trains anything. If
the artifacts are missing, run:

    python -m scripts.priority_train
"""

import joblib
import pandas as pd

from ticket_intelligence.priority_features import (
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    TEXT_COLUMN
)

if not MODEL_PATH.exists() or not LABEL_ENCODER_PATH.exists():
    raise FileNotFoundError(
        f"Priority model artifacts not found at {MODEL_PATH} and "
        f"{LABEL_ENCODER_PATH}. Run 'python -m scripts.priority_train' first."
    )

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)


def predict_priority(
    subject: str,
    description: str,
    category: str,
    environment: str,
    channel: str,
    region: str,
    customer_tier: str
) -> dict:
    """
    Predict priority for a new support ticket.
    """

    subject = subject or ""
    description = description or ""

    combined_text = f"{subject} {description}".strip()

    new_ticket = pd.DataFrame(
        [
            {
                TEXT_COLUMN: combined_text,
                "category": category,
                "environment": environment,
                "channel": channel,
                "region": region,
                "customer_tier": customer_tier
            }
        ]
    )

    encoded_prediction = model.predict(new_ticket)[0]

    predicted_priority = label_encoder.inverse_transform(
        [encoded_prediction]
    )[0]

    probabilities = model.predict_proba(new_ticket)[0]
    confidence = float(probabilities.max())

    return {
        "priority": predicted_priority,
        "confidence": round(confidence, 4)
    }
