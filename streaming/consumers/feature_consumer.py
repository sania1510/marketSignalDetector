# kafka/consumers/feature_consumer.py
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
from collections import defaultdict
import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from streaming.topics import (KAFKA_BOOTSTRAP, TOPIC_PRICES,
                           TOPIC_FEATURES)

class FeatureConsumer:
    """
    Consumes raw prices ? computes technical features ? publishes to features topic.
    Maintains a rolling window of prices in memory per ticker.
    """

    WINDOW = 50   # need at least 50 days to compute all indicators

    def __init__(self):
        self.consumer = KafkaConsumer(
            TOPIC_PRICES,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="feature-consumer-group",
            auto_offset_reset="earliest",
        )
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        # Rolling price buffer per ticker
        self.buffers = defaultdict(list)
        print("[FeatureConsumer] Ready")

    def compute_features(self, ticker: str) -> dict | None:
        buf = self.buffers[ticker]
        if len(buf) < self.WINDOW:
            return None

        df = pd.DataFrame(buf[-self.WINDOW:])
        df = df.sort_values("date")
        close = df["close"].astype(float)

        # Returns
        r1  = float(close.pct_change(1).iloc[-1])
        r5  = float(close.pct_change(5).iloc[-1])  if len(close) >= 5  else 0.0
        r20 = float(close.pct_change(20).iloc[-1]) if len(close) >= 20 else 0.0

        # SMA
        sma10 = float(close.rolling(10).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma10

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-9)
        rsi   = float(100 - (100 / (1 + rs.iloc[-1])))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9,  adjust=False).mean()
        macd_hist = float((macd - sig).iloc[-1])

        # Bollinger
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        bb_pos = float((close.iloc[-1] - lower.iloc[-1]) /
                       (upper.iloc[-1] - lower.iloc[-1] + 1e-9))

        # Volatility
        vol10 = float(close.pct_change().rolling(10).std().iloc[-1] * np.sqrt(252))

        return {
            "ticker":        ticker,
            "date":          buf[-1]["date"],
            "timestamp":     buf[-1]["timestamp"],
            "close":         float(close.iloc[-1]),
            "return_1d":     round(r1,  4),
            "return_5d":     round(r5,  4),
            "return_20d":    round(r20, 4),
            "rsi":           round(rsi, 2),
            "rsi_normalized":round((rsi - 50) / 50, 4),
            "macd_norm":     round(macd_hist / (float(close.iloc[-1]) + 1e-9), 6),
            "bb_position":   round(bb_pos, 4),
            "price_vs_sma10":round((float(close.iloc[-1]) - sma10) / sma10, 4),
            "ma_cross":      int(sma10 > sma50),
            "volatility_10d":round(vol10, 4),
            "high_vol_regime": int(vol10 > 0.20),
        }

    def run(self):
        print("[FeatureConsumer] Listening to market.prices...")
        for message in self.consumer:
            msg = message.value
            ticker = msg.get("ticker", "")
            if not ticker:
                continue

            self.buffers[ticker].append(msg)
            # Keep buffer bounded
            if len(self.buffers[ticker]) > 200:
                self.buffers[ticker] = self.buffers[ticker][-200:]

            features = self.compute_features(ticker)
            if features:
                self.producer.send(TOPIC_FEATURES, value=features)
                if ticker == "SPY":
                    print(f"  [Features] SPY {features['date']}: "
                          f"RSI={features['rsi']:.1f} "
                          f"vol={features['volatility_10d']:.3f}")

if __name__ == "__main__":
    FeatureConsumer().run()
