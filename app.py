from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ai_support_ticket_intelligence import predict_priority
from sentiment_service import analyze_sentiment
from summarization_service import summarize_ticket


app = FastAPI(
    title="AI Support Ticket Intelligence Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = Path(__file__).resolve().parent


@app.get("/")
def home():
    """
    Open the HTML user interface.
    """

    return FileResponse(
        BASE_DIR / "index.html"
    )


@app.get("/analyze-sentiment")
def analyze_sentiment_endpoint(text: str):
    """
    Analyze the sentiment of a support ticket.
    """

    return analyze_sentiment(text)


@app.get("/summarize-ticket")
def summarize_ticket_endpoint(text: str):
    """
    Summarize a support ticket description.
    """

    return {
        "summary": summarize_ticket(text)
    }


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