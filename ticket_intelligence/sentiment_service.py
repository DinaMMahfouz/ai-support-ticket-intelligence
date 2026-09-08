"""
Sentiment analysis for support tickets, using a three-class model.

Why not distilbert-base-uncased-finetuned-sst-2-english (the previous
model): it is fine-tuned on SST-2, a movie-review dataset, and has
exactly two labels - POSITIVE and NEGATIVE. Support tickets are mostly
neither. A plain informational question ("what is the current version
of Authentication Manager?") has no bucket to land in, so the model
forces it into NEGATIVE and, because a softmax over two classes must
sum to 1, reports high confidence while doing so. Measured on
data/sentiment_eval_set.csv, that failure is not rare - it is the
common case, because most tickets are neutral in tone.

cardiffnlp/twitter-roberta-base-sentiment-latest has a real neutral
class (id2label = negative / neutral / positive) and is trained on
short informal text rather than film criticism, which is closer to how
tickets are actually written.

This module only loads a pre-trained model - it never trains anything.
Run sentiment_eval_runner.py to score it against the labeled eval set.
"""

from transformers import pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# top_k=None returns every class score, not just the winner. The extra
# two numbers are what make a neutral prediction interpretable: "neutral
# 0.71 / negative 0.22 / positive 0.07" says something a bare label
# does not, and it is what lets the UI show a genuine distribution
# instead of a single number that always reads as certainty.
sentiment_model = pipeline(
    task="text-classification",
    model=MODEL_NAME,
    top_k=None
)

# Some checkpoints ship generic LABEL_0/1/2 names instead of real ones.
# Normalising here means the rest of the app, and the eval runner, only
# ever see "negative" / "neutral" / "positive" whatever the checkpoint
# happens to call them.
LABEL_ALIASES = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
    "POSITIVE": "positive"
}


def _normalise(label: str) -> str:
    return LABEL_ALIASES.get(label, label.strip().lower())


def analyze_sentiment(text: str) -> dict:
    """
    Classify the tone of a support ticket.

    Returns {"label", "confidence", "scores"} where scores maps every
    class to its probability. label is the highest-scoring class.
    """

    if not text or not text.strip():
        return {"label": "unknown", "confidence": 0.0, "scores": {}}

    raw = sentiment_model(text.strip(), truncation=True)

    # With top_k=None the pipeline returns a list of score dicts for a
    # single input, but wraps it in an outer list in some versions.
    # Unwrap defensively rather than pinning behaviour to one release -
    # this is exactly the kind of thing that broke summarization when
    # transformers went to v5.
    if raw and isinstance(raw[0], list):
        raw = raw[0]

    scores = {
        _normalise(entry["label"]): round(float(entry["score"]), 4)
        for entry in raw
    }

    top_label = max(scores, key=scores.get)

    return {
        "label": top_label,
        "confidence": scores[top_label],
        "scores": scores
    }
