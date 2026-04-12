# scheduler/daily_update.py

import sys
import os
import logging
import traceback
from datetime import datetime, date

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

REQUIRED_DIRS = ["data/raw/prices", "data/raw/macro", "data/processed", "scheduler"]
for d in REQUIRED_DIRS:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

# ============================================================
# LOGGING
# ============================================================

LOG_PATH = os.path.join(BASE_DIR, "scheduler", "daily_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
from supabase import create_client

from config import PRICE_TICKERS, FRED_API_KEY

PRICE_DIR = os.path.join(BASE_DIR, "data", "raw", "prices")
MACRO_DIR  = os.path.join(BASE_DIR, "data", "raw", "macro")

# ============================================================
# SUPABASE CLIENT
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set as environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
# SUPABASE HELPERS
# ============================================================

def sb_read(table: str) -> pd.DataFrame:
    """Read all rows from a Supabase table into a DataFrame."""
    try:
        res = supabase.table(table).select("*").execute()
        if res.data:
            df = pd.DataFrame(res.data)
            if "date" in df.columns:
                df = df.set_index("date")
                df.index = pd.to_datetime(df.index)
            elif "Date" in df.columns:
                df = df.set_index("Date")
                df.index = pd.to_datetime(df.index)
            return df
        return pd.DataFrame()
    except Exception as e:
        log.error(f"sb_read({table}) failed: {e}")
        return pd.DataFrame()


def sb_upsert(table: str, df: pd.DataFrame, date_col: str = "date"):
    """Upsert a DataFrame into a Supabase table row by row."""
    try:
        df_reset = df.copy()
        if df_reset.index.name in ("date", "Date") or pd.api.types.is_datetime64_any_dtype(df_reset.index):
            df_reset = df_reset.reset_index()
            df_reset = df_reset.rename(columns={df_reset.columns[0]: date_col})

        # Convert datetime columns to string
        for col in df_reset.columns:
            if pd.api.types.is_datetime64_any_dtype(df_reset[col]):
                df_reset[col] = df_reset[col].astype(str)

        # Replace NaN with None
        df_reset = df_reset.where(pd.notnull(df_reset), None)

        records = df_reset.to_dict(orient="records")
        # Upsert in batches of 500
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            supabase.table(table).upsert(batch).execute()

        log.info(f"sb_upsert({table}): {len(records)} rows saved")
    except Exception as e:
        log.error(f"sb_upsert({table}) failed: {e}")
        log.error(traceback.format_exc())


# ============================================================
# PATH HELPER
# ============================================================

def p(*parts):
    return os.path.join(BASE_DIR, *parts)


# ============================================================
# MAIN CLASS
# ============================================================

class DailyUpdater:

    def run(self):
        log.info("=" * 60)
        log.info(f"DAILY UPDATE STARTED  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        log.info(f"BASE_DIR = {BASE_DIR}")
        log.info("=" * 60)

        if self.already_updated_today():
            log.info("Already updated today — exiting safely")
            return True

        steps = [
            ("Fetch new prices",               self.fetch_new_prices),
            ("Fetch new macro",                self.fetch_new_macro),
            ("Rebuild master",                 self.rebuild_master),
            ("Update sentiment",               self.update_sentiment),
            ("Generate signals (append-only)", self.generate_signals),
        ]

        success = True
        for name, fn in steps:
            try:
                log.info(f"--- {name} ---")
                fn()
                log.info("    OK")
            except Exception as e:
                log.error(f"    FAILED: {e}")
                log.error(traceback.format_exc())
                success = False

        log.info("=" * 60)
        log.info("DAILY UPDATE COMPLETE")
        log.info("=" * 60)
        return success

    # ============================================================
    # SAFETY — skip if today already has a signal
    # ============================================================

    def already_updated_today(self):
        try:
            today_str = date.today().strftime("%Y-%m-%d")
            res = supabase.table("signals").select("date").eq("date", today_str).execute()
            if res.data:
                log.info(f"Signal for {today_str} already exists in Supabase — skipping")
                return True
            log.info(f"No signal for {today_str} yet — running pipeline")
            return False
        except Exception as e:
            log.warning(f"already_updated_today check failed: {e} — running pipeline anyway")
            return False

    # ============================================================
    # STEP 1 — PRICES (still uses local raw files as cache)
    # ============================================================

    def fetch_new_prices(self):
        today = date.today().strftime("%Y-%m-%d")

        for ticker in PRICE_TICKERS:
            path = os.path.join(PRICE_DIR, f"{ticker}.csv")

            if os.path.exists(path):
                existing = pd.read_csv(path, index_col=0, parse_dates=True)
                start    = str(existing.index.max().date())
            else:
                start = "2018-01-01"

            df_new = yf.download(ticker, start=start, end=today,
                                 auto_adjust=True, progress=False)
            if df_new.empty:
                log.warning(f"{ticker}: no new data")
                continue

            df_new.index.name = "Date"

            if os.path.exists(path):
                combined = pd.concat([existing, df_new])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined.sort_index(inplace=True)
                combined.to_csv(path)
            else:
                df_new.to_csv(path)

            log.info(f"{ticker}: updated to {df_new.index.max().date()}")

    # ============================================================
    # STEP 2 — MACRO (still uses local raw files as cache)
    # ============================================================

    def fetch_new_macro(self):
        try:
            from fredapi import Fred
            fred = Fred(api_key=FRED_API_KEY)

            SERIES = {"VIX": "VIXCLS", "YieldCurve": "T10Y2Y"}

            for name, sid in SERIES.items():
                path = os.path.join(MACRO_DIR, f"{name}.csv")
                s    = fred.get_series(sid, observation_start="2024-01-01")
                s.name = "value"

                if os.path.exists(path):
                    existing = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
                    combined = pd.concat([existing, s])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.to_csv(path)
                else:
                    s.to_csv(path)

                log.info(f"{name}: updated")

        except Exception as e:
            log.warning(f"Macro update failed: {e}")

    # ============================================================
    # STEP 3 — REBUILD MASTER → save to Supabase
    # ============================================================

    def rebuild_master(self):
        from ingestion.price_fetcher import PriceFetcher
        from ingestion.macro_fetcher import MacroFetcher
        from ingestion.data_merger   import DataMerger

        prices = PriceFetcher().get_close_prices()
        macro  = MacroFetcher().fetch_all()
        master = DataMerger().merge(prices, macro, None)

        # Also save to local CSV (other parts of pipeline read it)
        master_path = p("data", "processed", "master.csv")
        master.to_csv(master_path)
        log.info(f"master.csv saved locally: {len(master)} rows")

        # Save to Supabase
        sb_upsert("master", master, date_col="Date")

    # ============================================================
    # STEP 4 — SENTIMENT → save to Supabase
    # ============================================================

    def update_sentiment(self):
        try:
            from ingestion.news_fetcher import NewsFetcher
            df = NewsFetcher().build_sentiment()
            if df.empty:
                log.warning("Sentiment empty — keeping existing data")
                return

            # Save to local CSV
            sentiment_path = p("data", "processed", "sentiment_daily.csv")
            df.to_csv(sentiment_path)
            log.info(f"sentiment_daily.csv saved locally: {df.shape}")

            # Save to Supabase
            sb_upsert("sentiment_daily", df)

        except Exception as e:
            log.error(f"Sentiment update failed: {e}")
            log.error(traceback.format_exc())

    # ============================================================
    # STEP 5 — GENERATE SIGNAL → save to Supabase
    # ============================================================

    def generate_signals(self):
        from signals.engine import SignalEngine, SignalConfig

        master_path    = p("data", "processed", "master.csv")
        sentiment_path = p("data", "processed", "sentiment_daily.csv")

        # --- load master (from local CSV built in step 3) ---
        master = pd.read_csv(master_path, index_col=0, parse_dates=True)
        log.info(f"master: {len(master)} rows  latest={str(master.index.max())[:10]}")

        # --- load sentiment ---
        sentiment = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
        log.info(f"sentiment: {len(sentiment)} rows  latest={str(sentiment.index.max())[:10]}")

        # --- load existing signals from Supabase ---
        existing = sb_read("signals")
        if not existing.empty:
            log.info(f"signals from Supabase: {len(existing)} rows  latest={str(existing.index.max())[:10]}")
        else:
            log.info("No signals in Supabase yet — will create fresh")

        latest_date     = master.index.max()
        latest_date_str = str(latest_date)[:10]

        # Skip if signal already exists
        already_exists = (
            not existing.empty and
            any(str(idx)[:10] == latest_date_str for idx in existing.index)
        )
        if already_exists:
            log.info(f"Signal already exists for {latest_date_str} — skipping")
            return

        log.info(f"Generating signal for {latest_date_str} ...")

        engine = SignalEngine(SignalConfig())

        m_row = master.loc[latest_date]

        if latest_date in sentiment.index:
            s_row = sentiment.loc[latest_date]
            log.info(f"Sentiment: exact row for {latest_date_str}")
        else:
            prior = sentiment[sentiment.index <= latest_date]
            if not prior.empty:
                s_row = prior.iloc[-1]
                log.warning(f"No sentiment for {latest_date_str} — forward-filling")
            else:
                s_row = None
                log.warning("No sentiment at all — running without it")

        f_row = {"prob_up": 0.5}

        new_signal = engine.run_single(m_row, f_row, s_row)
        log.info(f"Generated: {new_signal.to_dict(orient='records')}")

        # Save new signal to Supabase (upsert = safe, no duplicates)
        sb_upsert("signals", new_signal)

        # Also update local CSV for any local tools that need it
        signals_path = p("data", "processed", "signals.csv")
        if not existing.empty:
            combined = pd.concat([existing, new_signal])
            combined = combined[~combined.index.duplicated(keep="first")]
            combined.sort_index(inplace=True)
        else:
            combined = new_signal
        combined.to_csv(signals_path)

        log.info(f"✅ Signal written for {latest_date_str}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    DailyUpdater().run()
