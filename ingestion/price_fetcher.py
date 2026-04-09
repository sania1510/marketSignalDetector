# ingestion/price_fetcher.py
import sys
import yfinance as yf
import pandas as pd
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import PRICE_TICKERS, START_DATE, END_DATE, PRICE_DIR

class PriceFetcher:
    """Downloads and caches OHLCV data for all configured tickers."""

    def __init__(self):
        os.makedirs(PRICE_DIR, exist_ok=True)

    def fetch(self, ticker: str, force_refresh: bool = False) -> pd.DataFrame:
        path = f"{PRICE_DIR}/{ticker}.csv"

        # Use cached file if it exists and refresh not requested
        if os.path.exists(path) and not force_refresh:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            print(f"[PriceFetcher] Loaded {ticker} from cache ({len(df)} rows)")
            return df

        print(f"[PriceFetcher] Downloading {ticker} from yfinance...")
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)

        if df.empty:
            raise ValueError(f"No data returned for ticker: {ticker}")

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "Date"
        df.to_csv(path)
        print(f"[PriceFetcher] Saved {ticker}   {len(df)} rows")
        return df

    def fetch_all(self) -> dict[str, pd.DataFrame]:
        """Returns a dict of {ticker: DataFrame} for all configured tickers."""
        return {ticker: self.fetch(ticker) for ticker in PRICE_TICKERS}

    def get_close_prices(self) -> pd.DataFrame:
        """Returns a single DataFrame of Close prices, one column per ticker."""
        dfs = self.fetch_all()
        close = pd.DataFrame({
            ticker: df["Close"] for ticker, df in dfs.items()
        })
        close.index.name = "Date"
        return close