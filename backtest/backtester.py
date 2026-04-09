# backtest/backtester.py
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
from dataclasses import dataclass

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE        = 0.04     # 4% annual (approximate T-bill rate)
TRANSACTION_COST      = 0.001    # 0.1% per trade (realistic for ETFs)


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    position_size:   float = 1.0       # fraction of capital to deploy on BUY
    stop_loss:       float = 0.07      # exit if position drops 7% from entry
    take_profit:     float = 0.15      # exit if position gains 15%
    risk_alert_cash: float = 1.0       # go 100% cash on RISK_ALERT
    transaction_cost:float = TRANSACTION_COST


@dataclass
class BacktestResult:
    total_return:       float
    annualised_return:  float
    sharpe_ratio:       float
    sortino_ratio:      float
    max_drawdown:       float
    calmar_ratio:       float      # annualised return / abs(max drawdown)
    win_rate:           float
    total_trades:       int
    benchmark_return:   float
    alpha:              float      # strategy return − benchmark return
    equity_curve:       pd.Series
    trades_log:         pd.DataFrame


class Backtester:
    """
    Event-driven backtester.
    Simulates a portfolio that follows generated signals on SPY.
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    # ------------------------------------------------------------------
    # Main backtest loop
    # ------------------------------------------------------------------

    def run(self, signals_df: pd.DataFrame,
            prices_df: pd.DataFrame) -> BacktestResult:

        cfg     = self.config
        prices  = prices_df["SPY"].dropna()

        # Align signals to price dates
        df = pd.DataFrame({"price": prices})
        df = df.join(signals_df[["signal", "composite_score",
                                  "confidence"]], how="left")
        df["signal"] = df["signal"].fillna("HOLD")
        df.dropna(subset=["price"], inplace=True)

        # Portfolio state
        cash     = cfg.initial_capital
        shares   = 0.0
        entry_px = 0.0

        equity_curve = []
        trades_log   = []

        for date, row in df.iterrows():
            price  = row["price"]
            signal = row["signal"]

            # --- Check stop-loss / take-profit on open position ---
            if shares > 0:
                pnl_pct = (price - entry_px) / entry_px
                if pnl_pct <= -cfg.stop_loss or pnl_pct >= cfg.take_profit:
                    # Force exit
                    proceeds = shares * price * (1 - cfg.transaction_cost)
                    trades_log.append({
                        "date": date, "action": "FORCED_EXIT",
                        "price": price, "shares": shares,
                        "pnl_pct": round(pnl_pct, 4),
                        "reason": "stop_loss" if pnl_pct < 0 else "take_profit"
                    })
                    cash   += proceeds
                    shares  = 0.0
                    entry_px = 0.0

            # --- Execute signal ---
            if signal == "BUY" and shares == 0:
                invest  = cash * cfg.position_size
                cost    = invest * cfg.transaction_cost
                shares  = (invest - cost) / price
                cash   -= invest
                entry_px = price
                trades_log.append({
                    "date": date, "action": "BUY",
                    "price": price, "shares": round(shares, 4),
                    "pnl_pct": 0.0, "reason": "signal"
                })

            elif signal in ("SELL", "RISK_ALERT") and shares > 0:
                proceeds = shares * price * (1 - cfg.transaction_cost)
                pnl_pct  = (price - entry_px) / entry_px if entry_px else 0
                trades_log.append({
                    "date": date, "action": signal,
                    "price": price, "shares": round(shares, 4),
                    "pnl_pct": round(pnl_pct, 4), "reason": "signal"
                })
                cash    += proceeds
                shares   = 0.0
                entry_px = 0.0

            # Record equity (mark-to-market)
            equity = cash + shares * price
            equity_curve.append({"date": date, "equity": equity})

        equity_series = (pd.DataFrame(equity_curve)
                           .set_index("date")["equity"])
        trades_df = pd.DataFrame(trades_log)

        # --- Benchmark: buy-and-hold SPY from day 1 ---
        bh_shares      = cfg.initial_capital / df["price"].iloc[0]
        benchmark_eq   = bh_shares * df["price"]
        benchmark_ret  = float((benchmark_eq.iloc[-1] / cfg.initial_capital) - 1)

        metrics = self._compute_metrics(equity_series, trades_df, benchmark_ret)
        return metrics

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_metrics(self, equity: pd.Series,
                         trades: pd.DataFrame,
                         benchmark_return: float) -> BacktestResult:
        cfg = self.config

        # Daily returns
        daily_ret   = equity.pct_change().dropna()
        total_days  = len(daily_ret)
        years       = total_days / TRADING_DAYS_PER_YEAR

        # Returns
        total_return      = float((equity.iloc[-1] / equity.iloc[0]) - 1)
        annualised_return = float((1 + total_return) ** (1 / max(years, 0.1)) - 1)

        # Sharpe ratio
        daily_rf   = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        excess_ret = daily_ret - daily_rf
        sharpe     = float(
            excess_ret.mean() / (excess_ret.std() + 1e-9) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

        # Sortino ratio (downside deviation only)
        downside = daily_ret[daily_ret < daily_rf]
        sortino  = float(
            excess_ret.mean() / (downside.std() + 1e-9) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

        # Max drawdown
        rolling_max  = equity.cummax()
        drawdown     = (equity - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())
        calmar       = annualised_return / abs(max_drawdown + 1e-9)

        # Win rate from trades
        if not trades.empty:
            closed = trades[trades["pnl_pct"] != 0.0]
            win_rate    = float((closed["pnl_pct"] > 0).mean()) if len(closed) else 0.0
            total_trades = len(closed)
        else:
            win_rate, total_trades = 0.0, 0

        alpha = total_return - benchmark_return

        return BacktestResult(
            total_return=round(total_return, 4),
            annualised_return=round(annualised_return, 4),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            max_drawdown=round(max_drawdown, 4),
            calmar_ratio=round(calmar, 3),
            win_rate=round(win_rate, 3),
            total_trades=total_trades,
            benchmark_return=round(benchmark_return, 4),
            alpha=round(alpha, 4),
            equity_curve=equity,
            trades_log=trades,
        )

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward(self, signals_df: pd.DataFrame,
                     prices_df: pd.DataFrame,
                     n_splits: int = 4) -> pd.DataFrame:
        """
        Splits the history into n equal windows and backtests each independently.
        This reveals whether performance is consistent or concentrated in one period.
        """
        dates  = prices_df.index
        size   = len(dates) // n_splits
        rows   = []

        for i in range(n_splits):
            start = dates[i * size]
            end   = dates[min((i + 1) * size, len(dates) - 1)]
            label = f"{start.year}-{end.year} (fold {i+1})"

            sig_fold   = signals_df[start:end]
            price_fold = prices_df[start:end]

            result = self.run(sig_fold, price_fold)
            rows.append({
                "period":             label,
                "total_return":       f"{result.total_return:.1%}",
                "annualised_return":  f"{result.annualised_return:.1%}",
                "sharpe_ratio":       result.sharpe_ratio,
                "max_drawdown":       f"{result.max_drawdown:.1%}",
                "win_rate":           f"{result.win_rate:.0%}",
                "alpha":              f"{result.alpha:.1%}",
                "trades":             result.total_trades,
            })
            print(f"  {label}: return={result.total_return:.1%}  "
                  f"sharpe={result.sharpe_ratio:.2f}  "
                  f"alpha={result.alpha:.1%}")

        df = pd.DataFrame(rows)
        df.to_csv("data/processed/walk_forward.csv", index=False)
        return df

    # ------------------------------------------------------------------
    # Pretty report
    # ------------------------------------------------------------------

    def print_report(self, result: BacktestResult):
        print("\n" + "=" * 52)
        print("BACKTEST RESULTS")
        print("=" * 52)
        print(f"  Total return        : {result.total_return:.1%}")
        print(f"  Annualised return   : {result.annualised_return:.1%}")
        print(f"  Benchmark (SPY B&H) : {result.benchmark_return:.1%}")
        print(f"  Alpha               : {result.alpha:+.1%}")
        print(f"  ─────────────────────────────────")
        print(f"  Sharpe ratio        : {result.sharpe_ratio:.2f}")
        print(f"  Sortino ratio       : {result.sortino_ratio:.2f}")
        print(f"  Calmar ratio        : {result.calmar_ratio:.2f}")
        print(f"  Max drawdown        : {result.max_drawdown:.1%}")
        print(f"  ─────────────────────────────────")
        print(f"  Total trades        : {result.total_trades}")
        print(f"  Win rate            : {result.win_rate:.0%}")
        print("=" * 52)