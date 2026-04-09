import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

TRADING_DAYS = 252
FORWARD_WINDOWS = [1, 5, 10, 20]


class SignalEvaluator:

    def __init__(self, signals_df: pd.DataFrame,
                 prices_df: pd.DataFrame,
                 macro_df: pd.DataFrame):
        self.signals = signals_df.copy()
        self.prices  = prices_df["SPY"].dropna()
        self.macro   = macro_df
        self._attach_outcomes()

    # ------------------------------------------------------------------
    # Attach forward returns
    # ------------------------------------------------------------------

    def _attach_outcomes(self):
        for window in FORWARD_WINDOWS:
            col = f"fwd_return_{window}d"
            self.signals[col] = (
                self.prices
                    .pct_change(window)
                    .shift(-window)
                    .reindex(self.signals.index)
            )

        self.signals["outcome_5d"] = (
            self.signals["fwd_return_5d"] > 0).astype(int)

        self.signals["signal_binary"] = self.signals["signal"].map(
            {"BUY": 1, "SELL": 0, "RISK_ALERT": 0, "HOLD": np.nan}
        )

    # ------------------------------------------------------------------
    # 1. Precision / recall
    # ------------------------------------------------------------------

    def precision_recall(self) -> pd.DataFrame:
        rows = []

        for sig_type in ["BUY", "SELL", "RISK_ALERT"]:
            mask   = self.signals["signal"] == sig_type
            subset = self.signals[mask].dropna(subset=["fwd_return_5d"])

            if len(subset) == 0:
                print(f"  [Evaluator] No data for signal type: {sig_type} — skipping")
                continue

            if sig_type == "BUY":
                correct = (subset["fwd_return_5d"] > 0).sum()
            else:
                correct = (subset["fwd_return_5d"] < 0).sum()

            n         = len(subset)
            precision = correct / n

            avg_fwd = {}
            for w in FORWARD_WINDOWS:
                col = f"fwd_return_{w}d"
                if col in subset.columns:
                    avg_fwd[f"avg_{w}d_return"] = subset[col].mean()
                else:
                    avg_fwd[f"avg_{w}d_return"] = np.nan

            rows.append({
                "signal":    sig_type,
                "count":     n,
                "precision": round(precision, 3),
                **{k: round(v, 4) for k, v in avg_fwd.items()}
            })

        # ── Guard: return empty DataFrame if no rows ──────────────────
        if not rows:
            print("  [Evaluator] WARNING: No signal types had sufficient data.")
            print("  This usually means signals.csv has very few BUY/SELL/RISK_ALERT rows.")
            empty = pd.DataFrame(columns=[
                "signal", "count", "precision",
                "avg_1d_return", "avg_5d_return",
                "avg_10d_return", "avg_20d_return"
            ]).set_index("signal")
            return empty

        df = pd.DataFrame(rows).set_index("signal")
        print("\n[Evaluator] Precision by signal type:")
        print(df.to_string())
        return df

    # ------------------------------------------------------------------
    # 2. Regime analysis
    # ------------------------------------------------------------------

    def regime_analysis(self) -> pd.DataFrame:
        spy_60d = self.prices.pct_change(60)
        vix     = self.macro.get("VIX", pd.Series(dtype=float)) \
                  if hasattr(self.macro, "get") else pd.Series(dtype=float)

        def classify_regime(row):
            date   = row.name
            ret_60 = spy_60d.get(date, 0) if date in spy_60d.index else 0
            try:
                vix_v = float(vix.loc[date]) if date in vix.index else 20.0
            except Exception:
                vix_v = 20.0
            if pd.isna(vix_v):
                vix_v = 20.0
            if vix_v > 28 or ret_60 < -0.10:
                return "bear"
            elif ret_60 > 0.08 and vix_v < 20:
                return "bull"
            else:
                return "sideways"

        self.signals["regime"] = self.signals.apply(classify_regime, axis=1)

        rows = []
        for regime in ["bull", "bear", "sideways"]:
            mask = self.signals["regime"] == regime
            sub  = self.signals[mask].dropna(
                subset=["fwd_return_5d", "signal_binary"])

            if len(sub) < 5:
                continue

            buy_mask  = sub["signal"] == "BUY"
            sell_mask = sub["signal"].isin(["SELL", "RISK_ALERT"])

            buy_prec  = (sub[buy_mask]["fwd_return_5d"] > 0).mean() \
                        if buy_mask.any() else np.nan
            sell_prec = (sub[sell_mask]["fwd_return_5d"] < 0).mean() \
                        if sell_mask.any() else np.nan

            buy_returns = sub[buy_mask]["fwd_return_5d"].dropna()
            sharpe = float(
                buy_returns.mean() / (buy_returns.std() + 1e-9) * np.sqrt(52)
            ) if len(buy_returns) > 2 else np.nan

            rows.append({
                "regime":         regime,
                "n_days":         int(mask.sum()),
                "buy_precision":  round(buy_prec,  3) if not pd.isna(buy_prec)  else None,
                "sell_precision": round(sell_prec, 3) if not pd.isna(sell_prec) else None,
                "buy_sharpe":     round(sharpe,    2) if not pd.isna(sharpe)    else None,
            })

        if not rows:
            print("  [Evaluator] WARNING: Not enough data for regime analysis.")
            return pd.DataFrame(columns=[
                "regime","n_days","buy_precision","sell_precision","buy_sharpe"
            ]).set_index("regime")

        df = pd.DataFrame(rows).set_index("regime")
        print("\n[Evaluator] Performance by market regime:")
        print(df.to_string())
        return df

    # ------------------------------------------------------------------
    # 3. Calibration
    # ------------------------------------------------------------------

    def calibration(self) -> pd.DataFrame:
        sub = self.signals.dropna(
            subset=["fwd_return_5d", "signal_binary", "confidence"])
        sub = sub[sub["signal"] != "HOLD"]

        if len(sub) < 10:
            print("  [Evaluator] WARNING: Not enough data for calibration.")
            return pd.DataFrame()

        bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        sub = sub.copy()
        sub["conf_bin"] = pd.cut(sub["confidence"], bins=bins, labels=labels)

        rows = []
        for label in labels:
            group    = sub[sub["conf_bin"] == label]
            if len(group) < 3:
                continue
            buy_grp  = group[group["signal"] == "BUY"]
            sell_grp = group[group["signal"].isin(["SELL", "RISK_ALERT"])]

            buy_acc  = (buy_grp["fwd_return_5d"] > 0).mean() \
                       if len(buy_grp) > 0 else np.nan
            sell_acc = (sell_grp["fwd_return_5d"] < 0).mean() \
                       if len(sell_grp) > 0 else np.nan
            overall  = np.nanmean([buy_acc, sell_acc])

            rows.append({
                "confidence_bin":   label,
                "n":                len(group),
                "buy_accuracy":     round(buy_acc,  3) if not pd.isna(buy_acc)  else None,
                "sell_accuracy":    round(sell_acc, 3) if not pd.isna(sell_acc) else None,
                "overall_accuracy": round(overall,  3) if not pd.isna(overall)  else None,
            })

        if not rows:
            print("  [Evaluator] WARNING: No calibration bins had enough data.")
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("confidence_bin")
        print("\n[Evaluator] Calibration (confidence vs accuracy):")
        print(df.to_string())
        return df

    # ------------------------------------------------------------------
    # 4. Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        feature_cols = [
            "lstm_score", "sentiment_score", "volatility_score",
            "composite_score", "prob_up", "vix",
        ]
        available = [c for c in feature_cols if c in self.signals.columns]

        if not available:
            print("  [Evaluator] WARNING: No feature columns found.")
            return pd.DataFrame()

        sub = self.signals.dropna(subset=["fwd_return_5d"] + available)

        if len(sub) < 10:
            print("  [Evaluator] WARNING: Not enough rows for feature importance.")
            return pd.DataFrame()

        correlations = {}
        for col in available:
            correlations[col] = round(sub[col].corr(sub["fwd_return_5d"]), 4)

        df = (pd.Series(correlations, name="correlation_with_5d_return")
                .sort_values(key=abs, ascending=False)
                .to_frame())
        print("\n[Evaluator] Feature importance (correlation with 5d return):")
        print(df.to_string())
        return df

    # ------------------------------------------------------------------
    # 5. Full report
    # ------------------------------------------------------------------

    def full_report(self) -> dict:
        print("\n" + "=" * 58)
        print("SIGNAL RELIABILITY REPORT")
        print("=" * 58)

        pr   = self.precision_recall()
        reg  = self.regime_analysis()
        cal  = self.calibration()
        feat = self.feature_importance()

        total  = len(self.signals)
        alerts = (self.signals["signal"] == "RISK_ALERT").sum()
        buys   = (self.signals["signal"] == "BUY").sum()
        sells  = (self.signals["signal"] == "SELL").sum()
        holds  = total - buys - sells - alerts

        print(f"\nSignal counts ({total} total trading days):")
        print(f"  BUY        : {buys}  ({buys/total:.0%})")
        print(f"  SELL       : {sells} ({sells/total:.0%})")
        print(f"  RISK_ALERT : {alerts} ({alerts/total:.0%})")
        print(f"  HOLD       : {holds} ({holds/total:.0%})")

        # Save outputs — only if non-empty
        os.makedirs("data/processed", exist_ok=True)
        if not pr.empty:
            pr.to_csv("data/processed/eval_precision.csv")
        if not reg.empty:
            reg.to_csv("data/processed/eval_regime.csv")
        if not cal.empty:
            cal.to_csv("data/processed/eval_calibration.csv")
        if not feat.empty:
            feat.to_csv("data/processed/eval_features.csv")

        # ── Key findings summary ──────────────────────────────────────
        print("\n" + "=" * 58)
        print("KEY FINDINGS")
        print("=" * 58)

        if not pr.empty:
            for sig in ["BUY", "SELL", "RISK_ALERT"]:
                if sig in pr.index:
                    print(f"{sig} precision  : {pr.loc[sig,'precision']:.0%}  "
                          f"(n={int(pr.loc[sig,'count'])})")

        if not reg.empty:
            for regime in ["bull", "bear", "sideways"]:
                if regime in reg.index:
                    bp = reg.loc[regime, "buy_precision"]
                    sp = reg.loc[regime, "sell_precision"]
                    print(f"{regime:>10} regime — "
                          f"buy precision: {bp if bp else 'n/a'}, "
                          f"sell precision: {sp if sp else 'n/a'}")

        if not cal.empty and "overall_accuracy" in cal.columns:
            valid = cal[cal["overall_accuracy"].notna()]
            if not valid.empty:
                best = valid.iloc[-1]
                print(f"High-confidence accuracy : "
                      f"{best['overall_accuracy']:.0%} (n={int(best['n'])})")

        return {"precision": pr, "regime": reg,
                "calibration": cal, "features": feat}