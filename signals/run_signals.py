# signals/run_signals.py
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from engine import SignalEngine, SignalConfig


def apply_cooldown(signals_df: pd.DataFrame,
                   cooldown_days: int = 15) -> pd.DataFrame:
    """
    After a BUY signal fires, suppress BUY signals for N days.
    After a SELL/RISK_ALERT fires, suppress SELL/RISK_ALERT for N days.
    BUY cooldown does NOT block SELL signals and vice versa.
    """
    df = signals_df.copy()
    last_buy_date  = None
    last_sell_date = None

    for date in df.index:
        sig = df.at[date, "signal"]

        if sig == "BUY":
            if last_buy_date is not None:
                if (date - last_buy_date).days < cooldown_days:
                    df.at[date, "signal"] = "HOLD"
                    continue
            last_buy_date = date

        elif sig in ("SELL", "RISK_ALERT"):
            if last_sell_date is not None:
                if (date - last_sell_date).days < cooldown_days:
                    df.at[date, "signal"] = "HOLD"
                    continue
            last_sell_date = date

    return df

def run():
    master    = pd.read_csv("data/processed/master.csv",
                            index_col=0, parse_dates=True)
    sentiment = pd.read_csv("data/processed/sentiment_daily.csv",
                            index_col=0, parse_dates=True)

    pred_path = "data/processed/lstm_predictions.csv"
    if os.path.exists(pred_path):
        forecasts = pd.read_csv(pred_path, index_col=0, parse_dates=True)
        print(f"[SignalEngine] Loaded {len(forecasts)} LSTM predictions")
    else:
        print("[SignalEngine] No predictions found — using neutral 0.5")
        forecasts = pd.DataFrame({"prob_up": 0.5}, index=master.index)

    config = SignalConfig(
        weight_lstm       = 0.50,
        weight_sentiment  = 0.30,
        weight_volatility = 0.20,
        buy_threshold     =  0.062,
        sell_threshold    = -0.062,
        vix_danger        = 30.0,
        strong_threshold  = 0.068,
        weak_threshold    = 0.058,
    )

    engine  = SignalEngine(config)
    signals = engine.run_history(master, forecasts, sentiment)

    print("\n[SignalEngine] Before cooldown:")
    print(signals["signal"].value_counts().to_string())

    # Apply 15-day cooldown
    signals = apply_cooldown(signals, cooldown_days=15)

    print("\n[SignalEngine] After 15-day cooldown:")
    print(signals["signal"].value_counts().to_string())

    # Save cleaned signals — this overwrites the old file
    signals.to_csv("data/processed/signals.csv")
    print("\n[SignalEngine] Saved cleaned signals → data/processed/signals.csv")

    # Verify stress windows look clean
    events = {
        "COVID_crash":    ("2020-02-01", "2020-04-30"),
        "inflation_2022": ("2022-01-01", "2022-10-31"),
        "rate_hike_2018": ("2018-10-01", "2018-12-31"),
    }
    print("\n[Verify] Signals per stress window after cooldown:")
    for name, (start, end) in events.items():
        w = signals[start:end]
        print(f"  {name}: {w['signal'].value_counts().to_dict()}")

if __name__ == "__main__":
    run()