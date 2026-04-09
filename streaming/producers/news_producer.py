# kafka/producers/news_producer.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json, time
import pandas as pd
from datetime import datetime
from kafka import KafkaProducer
from streaming.topics import KAFKA_BOOTSTRAP, TOPIC_NEWS
from config import NEWS_API_KEY

class NewsProducer:

    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def stream_from_csv(self, path: str = "data/raw/news/headlines.csv"):
        """Replay headlines from the downloaded CSV."""
        print(f"[NewsProducer] Streaming from {path}...")
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            msg = {
                "date":        str(row.get("date", "")),
                "title":       str(row.get("title", "")),
                "description": str(row.get("description", "")),
                "source":      str(row.get("source", "")),
                "timestamp":   datetime.utcnow().isoformat(),
            }
            self.producer.send(TOPIC_NEWS, value=msg)
        self.producer.flush()
        print(f"[NewsProducer] Sent {len(df)} headlines")

    def fetch_live(self):
        """Fetch fresh headlines from NewsAPI."""
        from newsapi import NewsApiClient
        api = NewsApiClient(api_key=NEWS_API_KEY)
        resp = api.get_top_headlines(category="business", language="en",
                                      page_size=20)
        for a in resp.get("articles", []):
            msg = {
                "date":        datetime.utcnow().strftime("%Y-%m-%d"),
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "source":      a["source"]["name"],
                "timestamp":   datetime.utcnow().isoformat(),
                "live":        True,
            }
            self.producer.send(TOPIC_NEWS, value=msg)
        self.producer.flush()
        print(f"[NewsProducer] Sent {len(resp['articles'])} live headlines")

    def run(self, interval: int = 300):
        """Stream live headlines every N seconds."""
        while True:
            try:
                self.fetch_live()
            except Exception as e:
                print(f"[NewsProducer] Error: {e}")
            time.sleep(interval)

if __name__ == "__main__":
    p = NewsProducer()
    p.stream_from_csv()
