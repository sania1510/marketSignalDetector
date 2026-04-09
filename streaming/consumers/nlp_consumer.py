# kafka/consumers/nlp_consumer.py
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
from kafka import KafkaConsumer, KafkaProducer
from streaming.topics import (KAFKA_BOOTSTRAP, TOPIC_NEWS, TOPIC_SENTIMENT)

class NLPConsumer:
    """
    Consumes raw news ? runs FinBERT ? publishes sentiment scores.
    """

    def __init__(self):
        self.consumer = KafkaConsumer(
            TOPIC_NEWS,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="nlp-consumer-group",
            auto_offset_reset="earliest",
        )
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self.model = None
        print("[NLPConsumer] Ready  model loads on first message")

    def _load_model(self):
        """Lazy load FinBERT so startup is fast."""
        from models.nlp_sentiment import SentimentModel
        self.model = SentimentModel()
        saved = "models/saved/finbert_finetuned"
        if os.path.exists(saved):
            self.model.load()
            print("[NLPConsumer] Loaded saved FinBERT model")
        else:
            print("[NLPConsumer] No saved model  using base FinBERT")

    def score(self, title: str, description: str) -> dict:
        text = f"{title}. {description or ''}".strip()
        results = self.model.predict([text])
        r = results[0]
        return {
            "label":      r["label"],
            "score":      r["score"],
            "prob_pos":   r["prob_pos"],
            "prob_neg":   r["prob_neg"],
            "prob_neu":   r["prob_neu"],
            "confidence": r["confidence"],
        }

    def run(self):
        print("[NLPConsumer] Listening to market.news...")
        for message in self.consumer:
            if self.model is None:
                self._load_model()

            msg = message.value
            try:
                sentiment = self.score(
                    msg.get("title", ""),
                    msg.get("description", "")
                )
                out = {
                    "date":        msg.get("date", ""),
                    "timestamp":   msg.get("timestamp", ""),
                    "title":       msg.get("title", "")[:100],
                    "source":      msg.get("source", ""),
                    **sentiment,
                }
                self.producer.send(TOPIC_SENTIMENT, value=out)
                print(f"  [NLP] {out['title'][:50]}... ? "
                      f"{out['label']} ({out['confidence']:.2f})")
            except Exception as e:
                print(f"  [NLP] Error: {e}")

if __name__ == "__main__":
    NLPConsumer().run()

