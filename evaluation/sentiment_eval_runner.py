"""
Scores a sentiment model against data/sentiment_eval_set.csv.

    python -m evaluation.sentiment_eval_runner --model current
    python -m evaluation.sentiment_eval_runner --model sst2
    python -m evaluation.sentiment_eval_runner --model both

Exists because swapping the sentiment model on a hunch would be
inconsistent with how every other model choice in this project was
made. The priority classifier beat two LLMs on frozen data before it
was kept; the sentiment model should have to clear the same bar.

"sst2" is the previous model (distilbert-base-uncased-finetuned-sst-2-
english), loaded here only so the change can be reported as a delta
rather than asserted. It has no neutral class at all, so every neutral
row is necessarily wrong - that is the finding, not a bug in this
runner.

A note on the labels: they are hand-written judgments about tone, and
two of them (SENT-018, SENT-019) are genuinely arguable - both are
factual defect reports with an undercurrent of fatigue. The `note`
column says so per row. Read them before quoting any number from this
file; a 24-row set built by one person is a sanity check, not a
benchmark, and the second decimal place means nothing at this size.
"""

import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ticket_intelligence.priority_features import DATA_DIR

SENTIMENT_EVAL_PATH = DATA_DIR / "sentiment_eval_set.csv"
LABELS = ["negative", "neutral", "positive"]


def _predict_current(texts):
    """The three-class model this project now uses."""

    from ticket_intelligence.sentiment_service import analyze_sentiment

    return [analyze_sentiment(text)["label"] for text in texts]


def _predict_sst2(texts):
    """
    The previous two-class model, loaded directly rather than through
    sentiment_service so both can be scored in one run.
    """

    from transformers import pipeline

    model = pipeline(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    return [model(text, truncation=True)[0]["label"].lower() for text in texts]


PREDICTORS = {
    "current": _predict_current,
    "sst2": _predict_sst2
}


def run_eval(name: str) -> None:
    df = pd.read_csv(SENTIMENT_EVAL_PATH)
    y_true = df["sentiment"]

    print(f"\n=== sentiment: {name} ({SENTIMENT_EVAL_PATH.name}, n={len(df)}) ===")

    y_pred = PREDICTORS[name](df["text"].tolist())

    print("\nAccuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    print(f"Confusion matrix (rows=true, cols=pred), labels = {LABELS}")
    print(confusion_matrix(y_true, y_pred, labels=LABELS))

    misses = [
        (row.ticket_id, row.sentiment, pred, row.text)
        for row, pred in zip(df.itertuples(), y_pred)
        if row.sentiment != pred
    ]
    if misses:
        print(f"\nMisclassified ({len(misses)}):")
        for ticket_id, true, pred, text in misses:
            print(f"  {ticket_id}  true={true:8s} predicted={pred:8s}  {text[:70]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=list(PREDICTORS) + ["both"],
        required=True,
        help="Which sentiment model to score."
    )
    args = parser.parse_args()

    names = list(PREDICTORS) if args.model == "both" else [args.model]
    for name in names:
        run_eval(name)


if __name__ == "__main__":
    main()
