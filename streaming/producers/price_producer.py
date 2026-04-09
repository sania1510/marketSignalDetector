# kafka/producers/price_producer.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json, time
import yfinance as yf
import pandas as pd
from datetime import datetime
from kafka import KafkaProducer
from streaming.topics import KAFKA_BOOTSTRAP, TOPIC_PRICES
from config import PRICE_TICKERS

class PriceProducer:
    """
    Streams latest price data to Kafka every 60 seconds.
    In production this would use a real-time feed.
    For development, it replays historical data.
    """

    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
        )
        print(f"[PriceProducer] Connected to Kafka")

    def fetch_latest(self, ticker: str) -> dict:
        """Fetch the most recent OHLCV bar."""
        df = yf.download(ticker, period="2d",
                         interval="1d", progress=False)
        if df.empty:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        row = df.iloc[-1]
        return {
            "ticker":    ticker,
            "timestamp": datetime.utcnow().isoformat(),
            "open":      round(float(row["Open"]),  2),
            "high":      round(float(row["High"]),  2),
            "low":       round(float(row["Low"]),   2),
            "close":     round(float(row["Close"]), 2),
            "volume":    int(row["Volume"]),
            "date":      str(df.index[-1].date()),
        }

    def stream_once(self):
        """Send latest price for all tickers."""
        for ticker in PRICE_TICKERS:
            try:
                data = self.fetch_latest(ticker)
                if data:
                    self.producer.send(
                        TOPIC_PRICES,
                        key=ticker,
                        value=data
                    )
                    print(f"[PriceProducer] Sent {ticker}: ${data['close']}")
            except Exception as e:
                print(f"[PriceProducer] Error for {ticker}: {e}")
        self.producer.flush()

    def run(self, interval_seconds: int = 60):
        """Stream continuously every N seconds."""
        print(f"[PriceProducer] Streaming every {interval_seconds}s...")
        while True:
            self.stream_once()
            time.sleep(interval_seconds)

    def replay_historical(self, start: str = "2018-01-01",
                          end: str = "2024-01-01",
                          delay: float = 0.05):
        """
        Replay all historical data day by day.
        Useful for testing the full pipeline.
        delay = seconds between each message (0.05 = fast replay)
        """
        print(f"[PriceProducer] Replaying history {start} ? {end}...")
        for ticker in PRICE_TICKERS:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            for date, row in df.iterrows():
                msg = {
                    "ticker":    ticker,
                    "timestamp": date.isoformat(),
                    "date":      str(date.date()),
                    "open":      round(float(row["Open"]),  2),
                    "high":      round(float(row["High"]),  2),
                    "low":       round(float(row["Low"]),   2),
                    "close":     round(float(row["Close"]), 2),
                    "volume":    int(row["Volume"]),
                    "replay":    True,
                }
                self.producer.send(TOPIC_PRICES, key=ticker, value=msg)
            self.producer.flush()
            print(f"  Replayed {len(df)} rows for {ticker}")
            time.sleep(delay)

if __name__ == "__main__":
    p = PriceProducer()
    p.replay_historical()   # for testing
    # p.run(60)             # for live streaming
