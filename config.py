# config.py
import os
from datetime import date

# --- API Keys (set as environment variables, never hardcode) ---
NEWS_API_KEY  = os.getenv("NEWS_API_KEY", "32d10e6c013741b4a0d6a66f31a6fb15")
FRED_API_KEY  = os.getenv("FRED_API_KEY", "d53ff20c7998022d5c81e76b9cb08af9")

PRICE_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "GLD", "TLT"]
BENCHMARK     = "SPY"

START_DATE = "2018-01-01"
END_DATE   = date.today().isoformat()   # ← always today's date

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
PRICE_DIR     = f"{RAW_DIR}/prices"
MACRO_DIR     = f"{RAW_DIR}/macro"
NEWS_DIR      = f"{RAW_DIR}/news"

STRESS_EVENTS = {
    "COVID_crash":    ("2020-02-01", "2020-04-30"),
    "inflation_2022": ("2022-01-01", "2022-10-31"),
    "rate_hike_2018": ("2018-10-01", "2018-12-31"),
    "covid_recovery": ("2020-04-01", "2020-08-31"),
}