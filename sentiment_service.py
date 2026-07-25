from transformers import pipeline


sentiment_model = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a support ticket.
    """

    if not text or not text.strip():
        return {
            "label": "UNKNOWN",
            "confidence": 0.0
        }

    result = sentiment_model(
        text.strip(),
        truncation=True
    )[0]

    return {
        "label": result["label"],
        "confidence": round(float(result["score"]), 4)
    }