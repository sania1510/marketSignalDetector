# config.py
import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
NEWS_API_KEY  = os.getenv("NEWS_API_KEY")
FRED_API_KEY  = os.getenv("FRED_API_KEY")
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PRICE_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "GLD", "TLT"]
BENCHMARK     = "SPY"

START_DATE = "2018-01-01"
END_DATE   = date.today().isoformat()

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
