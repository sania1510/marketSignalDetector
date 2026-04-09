# models/run_nlp.py
import os
import sys
import pandas as pd

# Always run relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.stdout.reconfigure(encoding='utf-8')

from models.nlp_sentiment import SentimentModel

def run():
    model = SentimentModel()

    # --- Load saved model if it exists, otherwise fine-tune ---
    saved_path = os.path.join(BASE_DIR, "models", "saved", "finbert_finetuned")

    if os.path.exists(saved_path):
        print("Fine-tuned model found — loading saved model.")
        model.load()
    else:
        phrasebank_path = os.path.join(BASE_DIR, "data", "raw", "news", "phrasebank.csv")
        if not os.path.exists(phrasebank_path):
            raise FileNotFoundError(
                f"No saved model AND no phrasebank.csv found.\n"
                f"Expected at: {phrasebank_path}\n"
                f"Download it from: https://huggingface.co/datasets/financial_phrasebank"
            )
        phrasebank = pd.read_csv(phrasebank_path)
        print(f"Training on {len(phrasebank)} labelled sentences...")
        model.fine_tune(phrasebank, epochs=3)

    # --- Score all headlines ---
    headlines_path = os.path.join(BASE_DIR, "data", "raw", "news", "headlines.csv")
    if not os.path.exists(headlines_path):
        raise FileNotFoundError(
            f"headlines.csv not found at {headlines_path}\n"
            f"Run ingestion pipeline first: python ingestion/run_pipeline.py"
        )

    headlines = pd.read_csv(headlines_path, parse_dates=["date"])
    print(f"Scoring {len(headlines)} headlines...")
    daily_sentiment = model.score_headlines(headlines)

    print("\nSample output (last 5 days):")
    print(daily_sentiment.tail(5).to_string())

    # --- Sanity check ---
    test_sentences = [
        "Company crushes earnings, raises guidance significantly",
        "Mass layoffs announced as revenue collapses",
        "Markets remain flat amid mixed economic signals",
    ]
    print("\nSanity check:")
    for text, result in zip(test_sentences, model.predict(test_sentences)):
        print(f"  '{text[:55]}' → {result['label']} ({result['confidence']:.2f})")

if __name__ == "__main__":
    run()