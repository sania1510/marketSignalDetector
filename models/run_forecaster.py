# models/run_forecaster.py
import sys
import os

# Always resolve paths relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import torch
from models.ts_forecaster import ForecastModel, FEATURE_COLS, SEQUENCE_LEN

def run():
    master_path    = os.path.join(BASE_DIR, "data", "processed", "master.csv")
    sentiment_path = os.path.join(BASE_DIR, "data", "processed", "sentiment_daily.csv")

    if not os.path.exists(master_path):
        raise FileNotFoundError(
            f"master.csv not found at {master_path}\n"
            f"Run ingestion pipeline first:\n"
            f"  cd {BASE_DIR}\n"
            f"  python ingestion/run_pipeline.py"
        )
    if not os.path.exists(sentiment_path):
        raise FileNotFoundError(
            f"sentiment_daily.csv not found at {sentiment_path}\n"
            f"Run NLP pipeline first:\n"
            f"  python models/run_nlp.py"
        )

    master    = pd.read_csv(master_path,    index_col=0, parse_dates=True)
    sentiment = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)

    model = ForecastModel()
    df    = model.prepare_data(master, sentiment)

    print(f"\nFeatures used : {df.shape[1]} columns")
    print(f"Date range    : {df.index[0].date()} → {df.index[-1].date()}")

    # Train
    model.train(df, epochs=40)

    # Generate predictions for all dates
    print("\n[ForecastModel] Generating predictions for all dates...")
    available = [c for c in FEATURE_COLS if c in df.columns]
    X_scaled  = model.scaler.transform(df[available].values)

    records = []
    model.model.eval()
    with torch.no_grad():
        for i in range(SEQUENCE_LEN, len(X_scaled)):
            seq     = X_scaled[i - SEQUENCE_LEN:i]
            prob_up = model.predict_proba(seq)
            records.append({
                "date":      df.index[i],
                "prob_up":   round(prob_up, 4),
                "direction": "UP" if prob_up > 0.5 else "DOWN",
            })

    predictions_df = pd.DataFrame(records).set_index("date")
    out_path = os.path.join(BASE_DIR, "data", "processed", "lstm_predictions.csv")
    predictions_df.to_csv(out_path)
    print(f"[ForecastModel] Saved {len(predictions_df)} predictions → {out_path}")

    # Latest prediction
    result = model.predict_latest(df)
    print(f"\nLatest prediction:")
    print(f"  Direction  : {result['direction']}")
    print(f"  P(up)      : {result['prob_up']:.3f}")
    print(f"  Confidence : {result['confidence']:.3f}")

if __name__ == "__main__":
    run()