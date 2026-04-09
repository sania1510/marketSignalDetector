# generate_historical_sentiment.py
import pandas as pd
import numpy as np
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
print("Generating historical sentiment aligned with price data...")

master = pd.read_csv("data/processed/master.csv",
                     index_col=0, parse_dates=True)

# Load any real headlines we have
real_sent_path = "data/processed/sentiment_daily.csv"
real_sent = pd.read_csv(real_sent_path, index_col=0, parse_dates=True) \
            if os.path.exists(real_sent_path) else pd.DataFrame()

print(f"Price data range: {master.index.min().date()} → {master.index.max().date()}")
print(f"Real sentiment rows: {len(real_sent)}")

rows = []

for date, row in master.iterrows():
    vix        = float(row.get("VIX", 20)) if not pd.isna(row.get("VIX", np.nan)) else 20.0
    spy_price  = float(row.get("SPY", 300))

    # Compute momentum from price data
    spy_series = master["SPY"].dropna()
    loc = spy_series.index.get_loc(date)

    ret_1d  = float(spy_series.pct_change(1).iloc[loc])  if loc >= 1  else 0.0
    ret_5d  = float(spy_series.pct_change(5).iloc[loc])  if loc >= 5  else 0.0
    ret_20d = float(spy_series.pct_change(20).iloc[loc]) if loc >= 20 else 0.0

    # Sentiment proxy formula:
    # - VIX above 25 = negative sentiment
    # - Strong positive returns = positive sentiment
    # - Add small random noise for realism
    np.random.seed(int(date.timestamp()) % 10000)

    vix_component  = -(vix - 20) / 30          # -1 to +1
    mom_component  = np.clip(ret_5d * 5, -1, 1) # amplify 5d return
    noise          = np.random.normal(0, 0.05)

    raw_sentiment  = 0.5 * vix_component + 0.5 * mom_component + noise
    sentiment_mean = float(np.clip(raw_sentiment, -1, 1))

    # Bullish/bearish ratios derived from sentiment
    if sentiment_mean > 0.1:
        bullish = 0.45 + sentiment_mean * 0.3
        bearish = 0.25 - sentiment_mean * 0.1
    elif sentiment_mean < -0.1:
        bullish = 0.25 + sentiment_mean * 0.1
        bearish = 0.45 - sentiment_mean * 0.3
    else:
        bullish = 0.35
        bearish = 0.35

    bullish = float(np.clip(bullish + np.random.normal(0, 0.02), 0, 1))
    bearish = float(np.clip(bearish + np.random.normal(0, 0.02), 0, 1))

    rows.append({
        "date":            date,
        "sentiment_mean":  round(sentiment_mean, 4),
        "sentiment_std":   round(abs(noise) * 2, 4),
        "bullish_ratio":   round(bullish, 4),
        "bearish_ratio":   round(bearish, 4),
        "headline_count":  int(np.random.randint(3, 20)),
        "avg_confidence":  round(0.7 + abs(sentiment_mean) * 0.2, 4),
    })

df = pd.DataFrame(rows).set_index("date")

# Compute 3-day rolling MA
df["sentiment_3d_ma"] = df["sentiment_mean"].rolling(3, min_periods=1).mean().round(4)

# Overlay real sentiment where we have it
if not real_sent.empty:
    real_sent.index = pd.to_datetime(real_sent.index)
    for col in ["sentiment_mean", "bullish_ratio", "bearish_ratio",
                "headline_count", "avg_confidence", "sentiment_3d_ma"]:
        if col in real_sent.columns:
            overlap = real_sent.index.intersection(df.index)
            if not overlap.empty:
                df.loc[overlap, col] = real_sent.loc[overlap, col]
    print(f"Overlaid {len(overlap)} real sentiment rows on top of synthetic data")

out = "data/processed/sentiment_daily.csv"
df.to_csv(out)
print(f"\nSaved {len(df)} sentiment rows → {out}")
print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
print(f"\nSample (COVID crash period):")
print(df["2020-02-20":"2020-03-05"][
    ["sentiment_mean","bullish_ratio","bearish_ratio"]
].to_string())