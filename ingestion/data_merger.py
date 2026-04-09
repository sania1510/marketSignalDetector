# ingestion/data_merger.py
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import PROCESSED_DIR, BENCHMARK


class DataMerger:
    """
    Aligns price, macro, and news data on a common trading-day index.
    Handles the key challenge: macro data is monthly/weekly,
    price data is daily — we forward-fill the gaps.
    """

    def __init__(self):
        os.makedirs(PROCESSED_DIR, exist_ok=True)

    def merge(self,
              prices: pd.DataFrame,
              macro: pd.DataFrame,
              news_daily: pd.DataFrame = None) -> pd.DataFrame:

        print("[DataMerger] Starting merge...")

        # Step 1: Use benchmark (SPY) as date spine
        date_spine = prices[[BENCHMARK]].copy()
        date_spine.index = pd.to_datetime(date_spine.index)

        # Step 2: Clean column names (fix weird encoding like 'VI X')
        prices.columns = prices.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True)
        macro.columns = macro.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True)

        # Step 3: Forward-fill macro to daily frequency
        macro.index = pd.to_datetime(macro.index)
        macro_daily = macro.reindex(date_spine.index, method="ffill")

        # Step 4: Merge prices
        prices.index = pd.to_datetime(prices.index)
        master = date_spine.join(
            prices.drop(columns=[BENCHMARK], errors="ignore"),
            how="left"
        )

        # Step 5: Merge macro
        master = master.join(macro_daily, how="left")

        # IMPORTANT: forward-fill AFTER merge (fixes 96% null issue)
        master = master.sort_index().ffill()

        # Step 6: Add news data
        if news_daily is not None:
            news_daily.index = pd.to_datetime(news_daily.index)
            master = master.join(news_daily, how="left")
            master["daily_headline_count"] = master["daily_headline_count"].fillna(0)

        # Step 7: Validation
        self._validate(master)

        # Step 8: Save
        out_path = f"{PROCESSED_DIR}/master.csv"
        master = master.round(2)
        master.to_csv(out_path)

        # FIX arrow issue
        print(f"[DataMerger] Saved master dataset - {master.shape} -> {out_path}")

        return master

    def _validate(self, df: pd.DataFrame):
        print("\n[DataMerger] Validation report:")
        print(f"  Shape      : {df.shape}")
        print(f"  Date range : {df.index.min().date()} -> {df.index.max().date()}")

        null_pct = (df.isnull().sum() / len(df) * 100).round(1)
        problematic = null_pct[null_pct > 5]

        if not problematic.empty:
            print(f"  Columns with >5% nulls:\n{problematic.to_string()}")
        else:
            print("  No columns with >5% nulls - data looks clean")

        # Check fully empty rows
        all_null_rows = df[df.isnull().all(axis=1)]
        if not all_null_rows.empty:
            print(f"  Warning: {len(all_null_rows)} fully empty rows detected")