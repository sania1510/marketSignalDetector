# ingestion/macro_fetcher.py
import sys
import pandas as pd
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import FRED_API_KEY, MACRO_DIR, START_DATE

MACRO_SERIES = {
    "VIX":          "VIXCLS",
    "FedFunds":     "FEDFUNDS",
    "CPI":          "CPIAUCSL",
    "Unemployment": "UNRATE",
    "YieldCurve":   "T10Y2Y",   # 10Y minus 2Y Treasury — recession indicator
}

class MacroFetcher:
    """Fetches macroeconomic indicators from FRED."""

    def __init__(self):
        os.makedirs(MACRO_DIR, exist_ok=True)
        # Lazy import so the rest of the project works without fredapi installed
        from fredapi import Fred
        self.fred = Fred(api_key=FRED_API_KEY)

    def fetch(self, name: str, force_refresh: bool = False) -> pd.Series:
        path = f"{MACRO_DIR}/{name}.csv"

        if os.path.exists(path) and not force_refresh:
            s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
            print(f"[MacroFetcher] Loaded {name} from cache")
            return s

        series_id = MACRO_SERIES[name]
        print(f"[MacroFetcher] Fetching {name} ({series_id}) from FRED...")
        s = self.fred.get_series(series_id, observation_start=START_DATE)
        s.name = name
        s.to_csv(path, header=True)
        return s

    def fetch_all(self) -> pd.DataFrame:
        """Returns all macro series aligned into one DataFrame."""
        series = {name: self.fetch(name) for name in MACRO_SERIES}
        df = pd.DataFrame(series)
        df.index.name = "Date"
        return df