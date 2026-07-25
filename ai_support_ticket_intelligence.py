import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ai_support_ticket_training_data_realistic_v2.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Handle missing text safely
df["subject"] = df["subject"].fillna("")
df["description"] = df["description"].fillna("")

# Combine subject and description
df["text"] = df["subject"] + " " + df["description"]

# Features used to predict priority
feature_columns = [
    "text",
    "category",
    "environment",
    "channel",
    "region",
    "customer_tier"
]

X = df[feature_columns]

# Target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["priority"])

print("Priority mapping:")
for number, label in enumerate(label_encoder.classes_):
    print(number, "=", label)

# Split before fitting preprocessors
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=32,
    stratify=y
)

# Define column types
text_column = "text"

categorical_columns = [
    "category",
    "environment",
    "channel",
    "region",
    "customer_tier"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                stop_words="english",
                max_features=10000,
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 2),
                sublinear_tf=True
            ),
            text_column
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ]
)

# Complete ML pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]
)

# Hyperparameter search
params = {
    "classifier__C": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        20,
        50
    ],
    "classifier__class_weight": [
        None,
        "balanced"
    ]
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=params,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nBest parameters:", grid.best_params_)
print("Best cross-validation score:", grid.best_score_)

# Best complete pipeline
model = grid.best_estimator_

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

model = grid.best_estimator_

# Predict on training data
y_train_pred = model.predict(X_train)

# Training accuracy
train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

print("\nTraining Accuracy:", train_accuracy)

# Training Classification Report
print("\nTraining Classification Report:")
print(
    classification_report(
        y_train,
        y_train_pred,
        target_names=label_encoder.classes_
    )
)

# Training predictions
y_train_pred = model.predict(X_train)

# Testing predictions
y_test_pred = model.predict(X_test)

print(
    f"Training Accuracy: "
    f"{accuracy_score(y_train, y_train_pred):.4f}"
)

print(
    f"Testing Accuracy : "
    f"{accuracy_score(y_test, y_test_pred):.4f}"
)


# ============================================================
# SAVE THE TRAINED MODEL
# ============================================================

# Models folder
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Saved model paths
MODEL_PATH = MODELS_DIR / "priority_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# Save the complete pipeline
joblib.dump(
    model,
    MODEL_PATH
)

# Save the LabelEncoder
joblib.dump(
    label_encoder,
    LABEL_ENCODER_PATH
)

print("\nModel files saved successfully:")
print(f"Priority model: {MODEL_PATH}")
print(f"Label encoder: {LABEL_ENCODER_PATH}")


# ============================================================
# REUSABLE PRIORITY PREDICTION FUNCTION
# ============================================================

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

    combined_text = (
        f"{subject} {description}"
    ).strip()

    new_ticket = pd.DataFrame(
        [
            {
                "text": combined_text,
                "category": category,
                "environment": environment,
                "channel": channel,
                "region": region,
                "customer_tier": customer_tier
            }
        ]
    )

    encoded_prediction = model.predict(
        new_ticket
    )[0]

    predicted_priority = (
        label_encoder.inverse_transform(
            [encoded_prediction]
        )[0]
    )

    probabilities = model.predict_proba(
        new_ticket
    )[0]

    confidence = float(
        probabilities.max()
    )

    return {
        "priority": predicted_priority,
        "confidence": round(
            confidence,
            4
        )
    }


# ============================================================
# TEST THE MODEL WITH ONE NEW TICKET
# ============================================================

test_ticket_result = predict_priority(
    subject="Production authentication failure",
    description=(
        "Multiple users cannot authenticate in the production "
        "environment. A temporary workaround exists, but normal "
        "business operations are significantly affected."
    ),
    category="Authentication",
    environment="Production",
    channel="Email",
    region="Middle East",
    customer_tier="Enterprise"
)

print("\nNew Ticket Priority Prediction:")
print(test_ticket_result)
