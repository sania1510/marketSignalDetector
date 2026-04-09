# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TechnicalFeatures:
    """
    Computes technical indicators + ML-ready features.
    """

    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = self.add_returns(df)
        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_volume_features(df)
        df = self.add_volatility(df)

        # NEW (important upgrades)
        df = self.add_lag_features(df)
        df = self.add_targets(df)

        return df

    # -------------------------------------------------
    # RETURNS
    # -------------------------------------------------
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["return_1d"]  = df["Close"].pct_change(1)
        df["return_5d"]  = df["Close"].pct_change(5)
        df["return_20d"] = df["Close"].pct_change(20)
        return df

    # -------------------------------------------------
    # MOVING AVERAGES
    # -------------------------------------------------
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sma_10"]  = df["Close"].rolling(10).mean()
        df["sma_50"]  = df["Close"].rolling(50).mean()
        df["sma_200"] = df["Close"].rolling(200).mean()

        df["price_vs_sma10"] = (df["Close"] - df["sma_10"]) / df["sma_10"]
        df["price_vs_sma50"] = (df["Close"] - df["sma_50"]) / df["sma_50"]

        df["ma_cross"] = (df["sma_10"] > df["sma_50"]).astype(int)
        return df

    # -------------------------------------------------
    # RSI
    # -------------------------------------------------
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = df["Close"].diff()

        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()

        rs = gain / loss.replace(0, np.nan)

        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_normalized"] = (df["rsi"] - 50) / 50

        return df

    # -------------------------------------------------
    # MACD
    # -------------------------------------------------
    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()

        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        df["macd_norm"] = df["macd_hist"] / df["Close"]
        return df

    # -------------------------------------------------
    # BOLLINGER BANDS
    # -------------------------------------------------
    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        sma = df["Close"].rolling(period).mean()
        std = df["Close"].rolling(period).std()

        upper = sma + 2 * std
        lower = sma - 2 * std

        df["bb_position"] = (df["Close"] - lower) / (upper - lower + 1e-9)
        df["bb_width"] = (upper - lower) / sma

        return df

    # -------------------------------------------------
    # VOLUME FEATURES
    # -------------------------------------------------
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["volume_ma20"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / (df["volume_ma20"] + 1)

        df["vol_price_corr"] = df["volume_ratio"] * np.sign(df["return_1d"])
        return df

    # -------------------------------------------------
    # VOLATILITY
    # -------------------------------------------------
    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        df["volatility_10d"] = df["return_1d"].rolling(10).std() * np.sqrt(252)
        df["volatility_30d"] = df["return_1d"].rolling(30).std() * np.sqrt(252)

        df["high_vol_regime"] = (df["volatility_10d"] > 0.20).astype(int)
        return df

    # -------------------------------------------------
    # NEW: LAG FEATURES (VERY IMPORTANT)
    # -------------------------------------------------
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["return_1d_lag1"] = df["return_1d"].shift(1)
        df["return_1d_lag2"] = df["return_1d"].shift(2)
        df["return_1d_lag3"] = df["return_1d"].shift(3)

        df["rsi_lag1"] = df["rsi"].shift(1)
        df["macd_lag1"] = df["macd"].shift(1)

        return df

    # -------------------------------------------------
    # NEW: BETTER TARGETS
    # -------------------------------------------------
    def add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        # Next-day (original)
        df["target_1d"] = (df["return_1d"].shift(-1) > 0).astype(int)

        #  Better: 3-day future return
        df["future_return_3d"] = df["Close"].pct_change(3).shift(-3)
        df["target_3d"] = (df["future_return_3d"] > 0).astype(int)

        return df