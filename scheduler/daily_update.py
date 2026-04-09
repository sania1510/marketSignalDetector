# scheduler/daily_update.py

import sys
import os
import logging
import traceback
from datetime import datetime, date

# ============================================================
# PATH SETUP  — must happen before ANY other imports
# ============================================================

# BASE_DIR = the project root  (one level above scheduler/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

# ✅ Set CWD to project root so every relative path inside
#    PriceFetcher / MacroFetcher / DataMerger / engine etc. resolves correctly
os.chdir(BASE_DIR)

REQUIRED_DIRS = [
    "data/raw/prices",
    "data/raw/macro",
    "data/processed",
    "scheduler"
]
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
# IMPORTS  (after chdir so config relative paths resolve)
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf

from config import PRICE_TICKERS, FRED_API_KEY

# ✅ Build absolute versions of the data dirs so they work
#    regardless of where the script is launched from.
PRICE_DIR = os.path.join(BASE_DIR, "data", "raw", "prices")
MACRO_DIR = os.path.join(BASE_DIR, "data", "raw", "macro")

# ============================================================
# PATH HELPER
# ============================================================

def p(*parts):
    """Absolute path joined from BASE_DIR."""
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
            ("Rebuild master.csv",             self.rebuild_master),
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
                log.error(traceback.format_exc())   # full traceback in log
                success = False

        log.info("=" * 60)
        log.info("DAILY UPDATE COMPLETE")
        log.info("=" * 60)
        return success

    # ============================================================
    # SAFETY — skip if today already has a signal
    # ============================================================

    def already_updated_today(self):
        path = p("data", "processed", "signals.csv")
        if not os.path.exists(path):
            log.info("signals.csv not found — will run full pipeline")
            return False

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return False

        today_str = date.today().strftime("%Y-%m-%d")
        already   = any(str(idx)[:10] == today_str for idx in df.index)

        if already:
            log.info(f"Signal for {today_str} already exists — skipping")
        else:
            log.info(f"No signal for {today_str} yet — running pipeline")
            log.info(f"Latest entry in signals.csv: {str(df.index.max())[:10]}")

        return already

    # ============================================================
    # STEP 1 — PRICES
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
    # STEP 2 — MACRO
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
    # STEP 3 — REBUILD MASTER
    # ============================================================

    def rebuild_master(self):
        from ingestion.price_fetcher import PriceFetcher
        from ingestion.macro_fetcher import MacroFetcher
        from ingestion.data_merger   import DataMerger

        prices = PriceFetcher().get_close_prices()
        macro  = MacroFetcher().fetch_all()
        DataMerger().merge(prices, macro, None)

    # ============================================================
    # STEP 4 — SENTIMENT
    # ============================================================

    def update_sentiment(self):
        try:
            from ingestion.news_fetcher import NewsFetcher
            df = NewsFetcher().build_sentiment()
            if df.empty:
                log.warning("Sentiment empty — keeping existing data")
            else:
                log.info(f"Sentiment updated: {df.shape}")
        except Exception as e:
            log.error(f"Sentiment update failed: {e}")
            log.error(traceback.format_exc())

    # ============================================================
    # STEP 5 — GENERATE SIGNAL FOR TODAY (append-only)
    # ============================================================

    def generate_signals(self):
        from signals.engine import SignalEngine, SignalConfig

        master_path    = p("data", "processed", "master.csv")
        sentiment_path = p("data", "processed", "sentiment_daily.csv")
        signals_path   = p("data", "processed", "signals.csv")

        # --- load master ---
        master = pd.read_csv(master_path, index_col=0, parse_dates=True)
        log.info(f"master.csv  : {len(master)} rows  latest={str(master.index.max())[:10]}")

        # --- load sentiment ---
        sentiment = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
        log.info(f"sentiment   : {len(sentiment)} rows  latest={str(sentiment.index.max())[:10]}")

        # --- load existing signals ---
        if os.path.exists(signals_path):
            existing = pd.read_csv(signals_path, index_col=0, parse_dates=True)
            log.info(f"signals.csv : {len(existing)} rows  latest={str(existing.index.max())[:10]}")
        else:
            existing = pd.DataFrame()
            log.info("signals.csv : not found — will create fresh")

        latest_date     = master.index.max()
        latest_date_str = str(latest_date)[:10]

        # skip if signal already exists for this date
        already_exists = (
            not existing.empty and
            any(str(idx)[:10] == latest_date_str for idx in existing.index)
        )
        if already_exists:
            log.info(f"Signal already exists for {latest_date_str} — skipping")
            return

        log.info(f"Generating signal for {latest_date_str} ...")

        engine = SignalEngine(SignalConfig())

        # ================================================================
        # KEY FIX — engine.run_single() calls:
        #
        #   master_row.name          → needs Series  (.name = the date)
        #   master_row["VIX"]        → needs Series or dict
        #   master_row.get("VIX",20) → needs Series  (Series has .get())
        #   forecast_row["prob_up"]  → needs Series or dict
        #   sentiment_row["sentiment_3d_ma"] → needs Series or dict
        #
        # pandas .loc[label]   (no list) → returns a Series  ✅
        # pandas .loc[[label]] (list)    → returns a DataFrame ✗
        # ================================================================

        # master_row — Series
        m_row = master.loc[latest_date]          # Series; .name == latest_date

        # sentiment_row — Series (exact match or forward-fill)
        if latest_date in sentiment.index:
            s_row = sentiment.loc[latest_date]   # Series
            log.info(f"Sentiment: exact row for {latest_date_str}")
        else:
            prior = sentiment[sentiment.index <= latest_date]
            if not prior.empty:
                s_row = prior.iloc[-1]           # Series
                log.warning(
                    f"No sentiment for {latest_date_str} — "
                    f"forward-filling from {str(prior.index[-1])[:10]}"
                )
            else:
                s_row = None
                log.warning("No sentiment at all — running without it")

        # forecast_row — plain dict is fine; engine only does forecast_row["prob_up"]
        f_row = {"prob_up": 0.5}

        new_signal = engine.run_single(m_row, f_row, s_row)
        log.info(f"Generated  : {new_signal.to_dict(orient='records')}")

        # append and save
        combined = pd.concat([existing, new_signal])
        combined = combined[~combined.index.duplicated(keep="first")]
        combined.sort_index(inplace=True)
        combined.to_csv(signals_path)

        log.info(f"✅ Signal written for {latest_date_str} "
                 f"(signals.csv now has {len(combined)} rows)")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    DailyUpdater().run()