# kafka/consumers/signal_consumer.py
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
from collections import defaultdict
from datetime import datetime
from kafka import KafkaConsumer, KafkaProducer
from streaming.topics import (KAFKA_BOOTSTRAP, TOPIC_FEATURES,
                           TOPIC_SENTIMENT, TOPIC_SIGNALS)

class SignalConsumer:
    """
    Consumes features + sentiment ? fuses them ? publishes signals.
    Uses two consumer instances listening to different topics.
    """

    def __init__(self):
        # Listen to both features and sentiment
        self.consumer = KafkaConsumer(
            TOPIC_FEATURES, TOPIC_SENTIMENT,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="signal-consumer-group",
            auto_offset_reset="earliest",
        )
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        # Latest state per date
        self.latest_features  = {}   # date ? feature dict
        self.daily_sentiments = defaultdict(list)  # date ? [scores]
        print("[SignalConsumer] Ready")

    def fuse_and_signal(self, date: str):
        """Combine latest features + sentiment into a signal."""
        from signals.engine import SignalEngine, SignalConfig

        feat = self.latest_features.get(date, {})
        if not feat:
            return

        sents = self.daily_sentiments.get(date, [])
        if sents:
            avg_sent     = sum(s["score"]    for s in sents) / len(sents)
            bullish_ratio= sum(s["prob_pos"] for s in sents) / len(sents)
        else:
            avg_sent      = 0.0
            bullish_ratio = 0.33

        # Use LSTM prediction if available
        pred_path = "data/processed/lstm_predictions.csv"
        prob_up = 0.5
        if os.path.exists(pred_path):
            import pandas as pd
            preds = pd.read_csv(pred_path, index_col=0, parse_dates=True)
            if date in preds.index.strftime("%Y-%m-%d").tolist():
                prob_up = float(preds.loc[preds.index.strftime("%Y-%m-%d") == date,
                                          "prob_up"].iloc[0])

        engine = SignalEngine(SignalConfig())
        result = engine.generate(
            date            = date,
            prob_up         = prob_up,
            sentiment_mean  = avg_sent,
            sentiment_trend = 0.0,
            bullish_ratio   = bullish_ratio,
            vix             = feat.get("vix", 20.0),
            vol_10d         = feat.get("volatility_10d", 0.15),
            vol_mean        = 0.15,
            yield_curve     = feat.get("yield_curve", 0.5),
        )

        signal_msg = {
            "date":            date,
            "timestamp":       datetime.utcnow().isoformat(),
            "signal":          result.signal,
            "composite_score": result.composite_score,
            "lstm_score":      result.lstm_score,
            "sentiment_score": result.sentiment_score,
            "volatility_score":result.volatility_score,
            "strength":        result.strength,
            "confidence":      result.confidence,
            "prob_up":         prob_up,
            "vix":             feat.get("vix", 0),
            "rationale":       " | ".join(result.rationale),
        }

        self.producer.send(TOPIC_SIGNALS, value=signal_msg)
        print(f"  [Signal] {date}: {result.signal} "
              f"(score={result.composite_score:+.4f})")
        return signal_msg

    def run(self):
        print("[SignalConsumer] Listening to features + sentiment...")
        for message in self.consumer:
            msg   = message.value
            topic = message.topic
            date  = msg.get("date", "")

            if topic == TOPIC_FEATURES and msg.get("ticker") == "SPY":
                self.latest_features[date] = msg
                self.fuse_and_signal(date)

            elif topic == TOPIC_SENTIMENT:
                self.daily_sentiments[date].append(msg)

if __name__ == "__main__":
    SignalConsumer().run()
