import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Literal

SignalType = Literal["BUY", "SELL", "HOLD", "RISK_ALERT"]

# ============================================================
# CONFIG
# ============================================================

@dataclass
class SignalConfig:
    weight_lstm:       float = 0.60
    weight_sentiment:  float = 0.25
    weight_volatility: float = 0.15

    buy_threshold:     float =  0.03
    sell_threshold:    float = -0.025

    strong_threshold:  float = 0.06
    weak_threshold:    float = 0.025

    vix_danger:        float = 26.0
    sentiment_crash:   float = -0.38
    vol_spike_mult:    float = 1.8

    recovery_momentum_block: float = 0.04


@dataclass
class Signal:
    date: pd.Timestamp
    signal: SignalType
    composite_score: float
    strength: str
    confidence: float

    lstm_score: float
    sentiment_score: float
    volatility_score: float

    prob_up: float
    sentiment_mean: float
    vix: float
    yield_curve: float

    rationale: list[str] = field(default_factory=list)


# ============================================================
# ENGINE
# ============================================================

class SignalEngine:

    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()

    # --------------------------------------------------------
    # REGIME DETECTION
    # --------------------------------------------------------
    def _detect_regime(self, vix, yield_curve, price_momentum=0.0):
        if vix > 28 or yield_curve < -0.3:
            if price_momentum > 0.05:
                return "RISK_OFF"
            return "CRISIS"
        elif vix > 22:
            return "RISK_OFF"
        elif vix < 15 and yield_curve > 0:
            return "BULL"
        else:
            return "NORMAL"

    # --------------------------------------------------------
    # SCORERS
    # --------------------------------------------------------

    def _score_lstm(self, prob_up):
        score = (prob_up - 0.5) * 2.0
        return float(np.clip(score, -1, 1)), [f"LSTM P_up={prob_up:.2f}"]

    def _score_sentiment(self, sentiment_mean, sentiment_trend, bullish_ratio):
        level = np.clip(sentiment_mean, -1, 1)
        trend = np.clip(sentiment_trend * 3, -1, 1)
        score = 0.6 * level + 0.4 * trend

        reasons = []
        if sentiment_mean < -0.3:
            reasons.append("Negative sentiment")
        if sentiment_trend < -0.1:
            reasons.append("Sentiment falling")

        return float(score), reasons

    def _score_volatility(self, vix, vol_10d, vol_mean, yield_curve):
        vix_score   = np.clip(-(vix - 18) / 20, -1, 1)
        spike       = vol_10d / (vol_mean + 1e-9)
        spike_score = np.clip(-(spike - 1), -1, 0)
        yc_score    = np.clip(yield_curve / 1.5, -1, 1)
        score       = 0.5 * vix_score + 0.3 * spike_score + 0.2 * yc_score

        reasons = []
        if vix > 25:
            reasons.append("High VIX")
        if yield_curve < 0:
            reasons.append("Yield inversion")

        return float(score), reasons

    # --------------------------------------------------------
    # RISK ALERT
    # --------------------------------------------------------
    def _check_risk_alert(self, vix, sentiment_mean, vol_10d, vol_mean):
        cfg = self.config
        triggers = []

        if vix > cfg.vix_danger:
            triggers.append("VIX spike")
        if sentiment_mean < cfg.sentiment_crash:
            triggers.append("Sentiment crash")
        if vol_10d > vol_mean * cfg.vol_spike_mult:
            triggers.append("Volatility spike")

        return len(triggers) >= 2, triggers

    # --------------------------------------------------------
    # MAIN SIGNAL GENERATION
    # --------------------------------------------------------
    def generate(self, date, prob_up, sentiment_mean,
                 sentiment_trend, bullish_ratio,
                 vix, vol_10d, vol_mean, yield_curve,
                 price_momentum=0.0):

        cfg  = self.config
        date = pd.to_datetime(date)

        regime = self._detect_regime(vix, yield_curve, price_momentum)

        lstm_score, r1 = self._score_lstm(prob_up)
        sent_score, r2 = self._score_sentiment(sentiment_mean, sentiment_trend, bullish_ratio)
        vol_score,  r3 = self._score_volatility(vix, vol_10d, vol_mean, yield_curve)

        if regime == "CRISIS":
            w_l, w_s, w_v = 0.4, 0.2, 0.4
        elif regime == "BULL":
            w_l, w_s, w_v = 0.7, 0.2, 0.1
        else:
            w_l, w_s, w_v = cfg.weight_lstm, cfg.weight_sentiment, cfg.weight_volatility

        composite = w_l*lstm_score + w_s*sent_score + w_v*vol_score
        composite = float(np.clip(composite, -1, 1))

        is_alert, alert_reasons = self._check_risk_alert(
            vix, sentiment_mean, vol_10d, vol_mean)

        if price_momentum > cfg.recovery_momentum_block:
            is_alert  = False
            composite = max(composite, 0.0)

        reasons = r1 + r2 + r3

        if regime in ("CRISIS", "RISK_OFF"):
            buy_threshold  = cfg.buy_threshold  * 1.5
            sell_threshold = cfg.sell_threshold * 0.4
        else:
            buy_threshold  = cfg.buy_threshold
            sell_threshold = cfg.sell_threshold

        if is_alert:
            signal_type = "RISK_ALERT"
            reasons     = alert_reasons + reasons
        elif composite > buy_threshold:
            signal_type = "BUY"
        elif composite < sell_threshold:
            signal_type = "SELL"
        else:
            signal_type = "HOLD"

        abs_score = abs(composite)
        if abs_score > cfg.strong_threshold:
            strength = "strong"
        elif abs_score > cfg.weak_threshold:
            strength = "moderate"
        else:
            strength = "weak"

        return Signal(
            date=date,
            signal=signal_type,
            composite_score=round(composite, 4),
            strength=strength,
            confidence=round(abs_score, 4),
            lstm_score=round(lstm_score, 4),
            sentiment_score=round(sent_score, 4),
            volatility_score=round(vol_score, 4),
            prob_up=round(prob_up, 4),
            sentiment_mean=round(sentiment_mean, 4),
            vix=round(vix, 2),
            yield_curve=round(yield_curve, 4),
            rationale=reasons + [f"Regime={regime}"]
        )

    # --------------------------------------------------------
    # RUN OVER HISTORY
    # --------------------------------------------------------
    def run_history(self, df, forecast_df, sentiment_df):

        combined = df.join(forecast_df[["prob_up"]], how="left")
        combined = combined.join(
            sentiment_df[["sentiment_3d_ma", "bullish_ratio"]], how="left"
        )

        combined["prob_up"]          = combined["prob_up"].fillna(0.5)
        combined["sentiment_3d_ma"]  = combined["sentiment_3d_ma"].fillna(0)
        combined["bullish_ratio"]    = combined["bullish_ratio"].fillna(0.33)
        combined["vol_mean_30d"]     = combined["VIX"].rolling(30).mean().fillna(20)
        combined["sentiment_trend"]  = combined["sentiment_3d_ma"].diff(5).fillna(0)
        combined["price_momentum"]   = combined["SPY"].pct_change(10).fillna(0)

        records = []

        for date, row in combined.iterrows():
            sig = self.generate(
                date,
                row["prob_up"],
                row["sentiment_3d_ma"],
                row["sentiment_trend"],
                row["bullish_ratio"],
                row["VIX"],
                row.get("volatility_10d", 0.15),
                row["vol_mean_30d"],
                row["YieldCurve"],
                row.get("price_momentum", 0.0)
            )

            records.append({
                "date":       sig.date,
                "signal":     sig.signal,
                "composite_score": sig.composite_score,
                "confidence": sig.confidence,
                "rationale":  " | ".join(sig.rationale)
            })

        df_out = pd.DataFrame(records).set_index("date")
        df_out.index = pd.to_datetime(df_out.index)
        df_out.to_csv("data/processed/signals.csv")

        print("[SignalEngine] Signals generated ")
        print(df_out["signal"].value_counts())

        return df_out

    # --------------------------------------------------------
    # ✅ NEW: RUN ONLY ONE ROW (TODAY)
    # --------------------------------------------------------
    def run_single(self, master_row, forecast_row=None, sentiment_row=None):

        prob_up = forecast_row["prob_up"] if forecast_row is not None else 0.5
        sentiment_mean = sentiment_row["sentiment_3d_ma"] if sentiment_row is not None else 0
        bullish_ratio = sentiment_row["bullish_ratio"] if sentiment_row is not None else 0.33

        sentiment_trend = 0  # cannot compute from single row
        vol_mean_30d = master_row.get("VIX", 20)
        price_momentum = 0

        sig = self.generate(
            master_row.name,
            prob_up,
            sentiment_mean,
            sentiment_trend,
            bullish_ratio,
            master_row["VIX"],
            master_row.get("volatility_10d", 0.15),
            vol_mean_30d,
            master_row["YieldCurve"],
            price_momentum
        )

        return pd.DataFrame([{
            "date": sig.date,
            "signal": sig.signal,
            "composite_score": sig.composite_score,
            "confidence": sig.confidence,
            "rationale": " | ".join(sig.rationale)
        }]).set_index("date")