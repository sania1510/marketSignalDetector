# kafka/consumers/db_writer.py
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import pandas as pd
from kafka import KafkaConsumer
from streaming.topics import KAFKA_BOOTSTRAP, TOPIC_SIGNALS, TOPIC_SENTIMENT

class DBWriter:
    """
    Consumes signals + sentiment topics ? appends to CSV files.
    This is what the FastAPI reads from.
    """

    SIGNALS_PATH   = "data/processed/signals_live.csv"
    SENTIMENT_PATH = "data/processed/sentiment_live.csv"

    def __init__(self):
        self.consumer = KafkaConsumer(
            TOPIC_SIGNALS, TOPIC_SENTIMENT,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="db-writer-group",
            auto_offset_reset="earliest",
        )
        os.makedirs("data/processed", exist_ok=True)
        print("[DBWriter] Ready writing to CSV")

    def append_row(self, path: str, row: dict, index_col: str = "date"):
        """Append a single row to CSV, creating file if needed."""
        df_new = pd.DataFrame([row]).set_index(index_col)
        if os.path.exists(path):
            df_existing = pd.read_csv(path, index_col=0)
            # Update or append
            df_existing = pd.concat(
                [df_existing, df_new]
            ).groupby(level=0).last()
        else:
            df_existing = df_new
        df_existing.to_csv(path)

    def run(self):
        print("[DBWriter] Listening to signals + sentiment...")
        count = 0
        for message in self.consumer:
            msg   = message.value
            topic = message.topic

            if topic == TOPIC_SIGNALS:
                self.append_row(self.SIGNALS_PATH, msg)
                count += 1
                if count % 10 == 0:
                    print(f"  [DBWriter] Written {count} signal rows")

            elif topic == TOPIC_SENTIMENT:
                # Aggregate sentiment by date then write
                self.append_row(self.SENTIMENT_PATH, {
                    "date":       msg.get("date",""),
                    "score":      msg.get("score", 0),
                    "label":      msg.get("label", "neutral"),
                    "prob_pos":   msg.get("prob_pos", 0),
                    "prob_neg":   msg.get("prob_neg", 0),
                    "confidence": msg.get("confidence", 0),
                })

if __name__ == "__main__":
    DBWriter().run()
