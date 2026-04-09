# ingestion/news_fetcher.py

import sys
import pandas as pd
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import NEWS_API_KEY, NEWS_DIR
from transformers import pipeline

# ============================================================
# LOAD MODEL (only once)
# ============================================================

print("[NewsFetcher] Loading FinBERT...")
sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
print("[NewsFetcher] FinBERT loaded")

SEARCH_QUERIES = [
    "stock market crash", "Federal Reserve interest rate",
    "inflation CPI", "earnings report", "recession",
    "S&P 500", "market volatility"
]

# ============================================================
# SENTIMENT FUNCTION
# ============================================================

def compute_sentiment(headlines):
    if not headlines:
        return 0.0

    # limit for speed + API sanity
    headlines = headlines[:25]

    results = sentiment_model(headlines)

    scores = []
    for r in results:
        if r["label"] == "positive":
            scores.append(r["score"])
        elif r["label"] == "negative":
            scores.append(-r["score"])
        else:
            scores.append(0)

    return sum(scores) / len(scores)


# ============================================================
# MAIN CLASS
# ============================================================

class NewsFetcher:
    """Fetches financial headlines and computes AI sentiment."""

    def __init__(self):
        os.makedirs(NEWS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # FETCH NEWS
    # --------------------------------------------------------

    def fetch_live_headlines(self, force_refresh: bool = False) -> pd.DataFrame:
        path = f"{NEWS_DIR}/headlines.csv"

        # FIX 1: Only use cache if it contains today's data — otherwise re-fetch
        if os.path.exists(path) and not force_refresh:
            df = pd.read_csv(path, parse_dates=["date"])
            if not df.empty:
                latest_cached = pd.to_datetime(df["date"].max()).date()
                today = pd.Timestamp.today().date()
                if latest_cached >= today:
                    print(f"[NewsFetcher] Loaded {len(df)} cached headlines (fresh for today)")
                    return df
            print("[NewsFetcher] Cache is stale — re-fetching from API")

        from newsapi import NewsApiClient
        api = NewsApiClient(api_key=NEWS_API_KEY)

        records = []

        for query in SEARCH_QUERIES:
            print(f"[NewsFetcher] Querying: '{query}'")
            try:
                resp = api.get_everything(
                    q=query,
                    language="en",
                    sort_by="publishedAt",
                    page_size=100
                )

                for a in resp.get("articles", []):
                    records.append({
                        "date":        a["publishedAt"][:10],
                        "title":       a.get("title", ""),
                        "description": a.get("description", ""),
                        "source":      a["source"]["name"],
                        "query":       query,
                    })

            except Exception as e:
                print(f"  Warning: {e}")

        df = (pd.DataFrame(records)
                .drop_duplicates(subset="title")
                .dropna(subset=["title"])
                .sort_values("date"))

        df.to_csv(path, index=False)
        print(f"[NewsFetcher] Saved {len(df)} headlines")

        return df

    # --------------------------------------------------------
    # BUILD DAILY SENTIMENT
    # --------------------------------------------------------

    def build_sentiment(self) -> pd.DataFrame:

        # FIX 2: Always force a fresh fetch so daily updates get real headlines
        df = self.fetch_live_headlines(force_refresh=True)

        if df.empty:
            print("[NewsFetcher] No news found → fallback neutral sentiment")
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])

        grouped = df.groupby(df["date"].dt.date)

        rows = []

        for dt, group in grouped:
            headlines = group["title"].tolist()

            sentiment = compute_sentiment(headlines)

            rows.append({
                "date": pd.to_datetime(dt),
                "sentiment_mean": sentiment,
                "sentiment_3d_ma": sentiment,
                "bullish_ratio": 0.5 + sentiment * 0.3
            })

        df_sent = pd.DataFrame(rows).set_index("date").sort_index()

        # smooth (important for stability)
        df_sent["sentiment_3d_ma"] = df_sent["sentiment_mean"].rolling(3).mean().fillna(0)

        out_path = "data/processed/sentiment_daily.csv"

        # FIX 3: Merge with existing history instead of overwriting it
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path, index_col=0, parse_dates=True)
            df_sent = pd.concat([existing, df_sent])
            df_sent = df_sent[~df_sent.index.duplicated(keep="last")]
            df_sent.sort_index(inplace=True)
            print(f"[NewsFetcher] Merged with existing sentiment — total rows: {len(df_sent)}")

        df_sent.to_csv(out_path)

        print(f"[NewsFetcher] Sentiment saved → {out_path}")

        return df_sent

    # --------------------------------------------------------
    # PHRASEBANK (unchanged)
    # --------------------------------------------------------

    def load_phrasebank(self) -> pd.DataFrame:
        path = f"{NEWS_DIR}/phrasebank.csv"

        if os.path.exists(path):
            return pd.read_csv(path)

        print("[NewsFetcher] Downloading Financial PhraseBank...")
        from datasets import load_dataset

        ds = load_dataset("financial_phrasebank", "sentences_allagree",
                          trust_remote_code=True)

        df = pd.DataFrame(ds["train"])
        df.columns = ["text", "label"]

        df.to_csv(path, index=False)

        print(f"[NewsFetcher] Saved {len(df)} labelled sentences")

        return df