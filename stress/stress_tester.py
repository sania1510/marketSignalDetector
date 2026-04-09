# stress/stress_tester.py  — full file replacement
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from config import STRESS_EVENTS, BENCHMARK

@dataclass
class StressResult:
    event_name:          str
    start:               str
    end:                 str
    market_return:       float
    max_drawdown:        float
    first_alert_date:    str
    lead_time_days:      int
    alert_count:         int
    false_alarm_rate:    float
    signal_accuracy:     float
    composite_at_bottom: float
    vix_at_bottom:       float
    notes:               str


class StressTester:

    def __init__(self,
                 signals_df: pd.DataFrame,
                 prices_df:  pd.DataFrame,
                 macro_df:   pd.DataFrame):
        self.signals = signals_df
        self.prices  = prices_df
        self.macro   = macro_df

    def analyse_event(self, event_name: str,
                      start: str, end: str) -> StressResult:

        sig_window   = self.signals[start:end].copy()
        price_window = self.prices[BENCHMARK][start:end].copy()

        vix_series   = self.macro["VIX"] \
                       if "VIX" in self.macro.columns else pd.Series(dtype=float)
        macro_window = vix_series[start:end] if not vix_series.empty else pd.Series(dtype=float)

        if price_window.empty:
            raise ValueError(
                f"No price data for event: {event_name} ({start}→{end})")

        # 1. Market return
        market_return = float(
            (price_window.iloc[-1] / price_window.iloc[0]) - 1)

        # 2. Max drawdown + bottom date
        rolling_max  = price_window.cummax()
        drawdown     = (price_window - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())
        bottom_date  = drawdown.idxmin()

        # 3. Warning signals — SELL or RISK_ALERT fired before the bottom
        pre_bottom = sig_window[:bottom_date]
        warning_signals = pre_bottom[
            pre_bottom["signal"].isin(["RISK_ALERT", "SELL"])
        ]

        if not warning_signals.empty:
            first_alert      = warning_signals.index[0]
            first_alert_date = str(first_alert.date())
            lead_time_days   = max(0, (bottom_date - first_alert).days)
        else:
            first_alert_date = "No alert fired"
            lead_time_days   = 0

        # 4. Alert count
        alert_count = len(warning_signals)

        # 5. False alarm rate
        if alert_count > 0:
            false_alarms = 0
            for alert_date in warning_signals.index:
                lookahead = price_window[alert_date:]
                if len(lookahead) > 1:
                    fwd = (lookahead.iloc[min(10, len(lookahead)-1)]
                           / lookahead.iloc[0]) - 1
                    if fwd > -0.02:
                        false_alarms += 1
            false_alarm_rate = false_alarms / alert_count
        else:
            false_alarm_rate = 0.0

        # 6. Signal accuracy — did price fall after each warning?
        if alert_count > 0:
            correct = 0
            for alert_date in warning_signals.index:
                lookahead = price_window[alert_date:]
                if len(lookahead) > 5:
                    fwd = (lookahead.iloc[5] / lookahead.iloc[0]) - 1
                    if fwd < 0:
                        correct += 1
            signal_accuracy = correct / alert_count
        else:
            signal_accuracy = 0.0

        # 7. Composite score at market bottom
        try:
            if not sig_window.empty:
                idx = sig_window.index.get_indexer(
                    [bottom_date], method="nearest")[0]
                composite_at_bottom = float(
                    sig_window.iloc[idx]["composite_score"]) \
                    if idx >= 0 else 0.0
            else:
                composite_at_bottom = 0.0
        except Exception:
            composite_at_bottom = 0.0

        # 8. VIX at bottom
        try:
            if not macro_window.empty:
                idx = macro_window.index.get_indexer(
                    [bottom_date], method="nearest")[0]
                vix_at_bottom = float(macro_window.iloc[idx]) \
                                if idx >= 0 else 0.0
            else:
                vix_at_bottom = 0.0
        except Exception:
            vix_at_bottom = 0.0

        # 9. Notes
        if lead_time_days > 5:
            notes = (f"System warned {lead_time_days} days before bottom. "
                     f"Signal accuracy: {signal_accuracy:.0%}. "
                     f"False alarm rate: {false_alarm_rate:.0%}.")
        elif lead_time_days > 0:
            notes = (f"Late warning — only {lead_time_days} days before bottom. "
                     f"Signal accuracy: {signal_accuracy:.0%}.")
        else:
            notes = (f"No SELL/RISK_ALERT fired before the bottom on "
                     f"{bottom_date.date()}. "
                     f"Market fell {max_drawdown:.1%} max. "
                     f"Consider lowering thresholds.")

        return StressResult(
            event_name=event_name,
            start=start, end=end,
            market_return=round(market_return, 4),
            max_drawdown=round(max_drawdown, 4),
            first_alert_date=first_alert_date,
            lead_time_days=lead_time_days,
            alert_count=alert_count,
            false_alarm_rate=round(false_alarm_rate, 3),
            signal_accuracy=round(signal_accuracy, 3),
            composite_at_bottom=round(composite_at_bottom, 4),
            vix_at_bottom=round(vix_at_bottom, 1),
            notes=notes,
        )

    def run_all(self) -> pd.DataFrame:
        results = []
        for name, (start, end) in STRESS_EVENTS.items():
            print(f"\n[StressTester] {name} ({start} → {end})")
            try:
                r = self.analyse_event(name, start, end)
                results.append(r.__dict__)
                print(f"  Market return    : {r.market_return:.1%}")
                print(f"  Max drawdown     : {r.max_drawdown:.1%}")
                print(f"  Market bottom    : (see notes)")
                print(f"  First alert      : {r.first_alert_date}")
                print(f"  Lead time        : {r.lead_time_days} days")
                print(f"  Alert count      : {r.alert_count}")
                print(f"  Signal accuracy  : {r.signal_accuracy:.0%}")
                print(f"  False alarm rate : {r.false_alarm_rate:.0%}")
                print(f"  Notes            : {r.notes}")
            except Exception as e:
                print(f"  Skipped: {e}")

        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv("data/processed/stress_results.csv", index=False)
            print(f"\n[StressTester] Saved → data/processed/stress_results.csv")
        return df

    def simulate_scenario(self,
                          base_row: pd.Series,
                          vix_shock: float = 40.0,
                          sentiment_shock: float = -0.8,
                          label: str = "custom_scenario") -> dict:
        from signals.engine import SignalEngine, SignalConfig
        engine = SignalEngine(SignalConfig())
        shocked = engine.generate(
            date            = pd.Timestamp.today().normalize(),
            prob_up         = max(0.1, float(base_row.get("prob_up", 0.5)) - 0.2),
            sentiment_mean  = sentiment_shock,
            sentiment_trend = -0.3,
            bullish_ratio   = 0.1,
            vix             = vix_shock,
            vol_10d         = float(base_row.get("volatility_10d", 0.15)) * 2.5,
            vol_mean        = float(base_row.get("vol_mean_30d", 20.0)),
            yield_curve     = float(base_row.get("YieldCurve", 0.5)) - 0.3,
        )
        return {
            "scenario":        label,
            "vix_shock":       vix_shock,
            "sentiment_shock": sentiment_shock,
            "signal":          shocked.signal,
            "composite_score": shocked.composite_score,
            "rationale":       shocked.rationale,
        }