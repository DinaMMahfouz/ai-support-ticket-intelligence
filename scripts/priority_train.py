"""
Trains the ticket-priority classifier and saves the artifacts that
priority_service.py loads at inference time.

Run explicitly:

    python -m scripts.priority_train

This must never run as a side effect of importing another module. That
was the bug in the file this replaced (ai_support_ticket_intelligence.py):
importing it just to get predict_priority() also re-ran a 70-fit grid
search and retrained the model every time.

Same model, same hyperparameter grid, same cross-validation as before -
this is a restructuring, not a retuning.

Trains only on data/train_pool_v1.csv (created by make_eval_split.py),
never on the raw CSV or on data/eval_holdout_v1.csv - those rows must
stay unseen so the held-out eval set is a fair, unbiased measurement,
not a number partly explained by the model having trained on the exact
rows it's being scored on. Run make_eval_split.py once before this.
"""

import json
from datetime import datetime, timezone

import joblib

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

from ticket_intelligence.priority_features import (
    TRAIN_POOL_PATH,
    MODELS_DIR,
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    METRICS_PATH,
    TEXT_COLUMN,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_ticket_csv
)


def build_pipeline() -> Pipeline:
    """Build the untrained TF-IDF + one-hot + logistic-regression pipeline."""

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
                TEXT_COLUMN
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_COLUMNS
            )
        ]
    )

    return Pipeline(
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


def main():
    if not TRAIN_POOL_PATH.exists():
        raise FileNotFoundError(
            f"{TRAIN_POOL_PATH} not found. Run 'python -m scripts.make_eval_split' "
            f"first - training must not read the raw CSV directly, or it "
            f"could include rows that data/eval_holdout_v1.csv also uses."
        )

    df = load_ticket_csv(TRAIN_POOL_PATH)
    X = df[FEATURE_COLUMNS]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COLUMN])

    print("Priority mapping:")
    for number, label in enumerate(label_encoder.classes_):
        print(number, "=", label)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=32,
        stratify=y
    )

    pipeline = build_pipeline()

    params = {
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 20, 50],
        "classifier__class_weight": [None, "balanced"]
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

    model = grid.best_estimator_

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_report = classification_report(
        y_train,
        y_train_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    test_report = classification_report(
        y_test,
        y_test_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    test_confusion = confusion_matrix(y_test, y_test_pred)

    print("\nTraining Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)

    print("\nTest Confusion Matrix:")
    print(test_confusion)

    print("\nTest Classification Report:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=label_encoder.classes_
        )
    )

    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "label_classes": label_encoder.classes_.tolist(),
        "best_params": grid.best_params_,
        "best_cv_f1_macro": grid.best_score_,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_confusion_matrix": test_confusion.tolist(),
        "train_classification_report": train_report,
        "test_classification_report": test_report
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved label encoder: {LABEL_ENCODER_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
