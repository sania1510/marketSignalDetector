# backtest/run_backtest.py
import pandas as pd
from backtester import Backtester, BacktestConfig

def run():
    signals = pd.read_csv("data/processed/signals.csv",
                           index_col=0, parse_dates=True)
    prices  = pd.read_csv("data/processed/master.csv",
                           index_col=0, parse_dates=True)

    cfg = BacktestConfig(
        initial_capital = 10_000,
        position_size   = 1.0,
        stop_loss       = 0.07,
        take_profit     = 0.15,
    )
    bt = Backtester(cfg)

    # Full backtest
    print("Running full backtest...")
    result = bt.run(signals, prices)
    bt.print_report(result)

    # Save equity curve for dashboard
    result.equity_curve.to_csv("data/processed/equity_curve.csv", header=["equity"])
    result.trades_log.to_csv("data/processed/trades_log.csv", index=False)

    # Walk-forward (checks consistency across time periods)
    print("\nWalk-forward validation (4 folds):")
    wf = bt.walk_forward(signals, prices, n_splits=4)
    print(wf.to_string(index=False))

if __name__ == "__main__":
    run()
