from transformers import pipeline


summarizer = pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6"
)


def summarize_ticket(text: str) -> str:
    """
    Summarize a support ticket description.
    """

    if not text or not text.strip():
        return "No description provided."

    cleaned_text = text.strip()
    word_count = len(cleaned_text.split())

    if word_count < 40:
        return cleaned_text

    result = summarizer(
        cleaned_text,
        max_length=80,
        min_length=20,
        do_sample=False,
        truncation=True
    )

    return result[0]["summary_text"]