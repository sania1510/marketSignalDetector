# backtest/run_evaluation.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from evaluator import SignalEvaluator

def run():
    signals = pd.read_csv("data/processed/signals.csv",
                           index_col=0, parse_dates=True)
    master  = pd.read_csv("data/processed/master.csv",
                           index_col=0, parse_dates=True)

    evaluator = SignalEvaluator(signals, master, master)
    results   = evaluator.full_report()

    # Highlight key findings
    pr  = results["precision"]
    reg = results["regime"]
    cal = results["calibration"]

    print("\n" + "=" * 58)
    print("KEY FINDINGS")
    print("=" * 58)

    if "BUY" in pr.index:
        print(f"\nBUY signal precision  : {pr.loc['BUY','precision']:.0%}")
    if "RISK_ALERT" in pr.index:
        print(f"RISK_ALERT precision  : {pr.loc['RISK_ALERT','precision']:.0%}")

    if "bear" in reg.index and reg.loc["bear", "sell_precision"]:
        print(f"SELL precision in bear: {reg.loc['bear','sell_precision']:.0%}")
    if "bull" in reg.index and reg.loc["bull", "buy_precision"]:
        print(f"BUY precision in bull : {reg.loc['bull','buy_precision']:.0%}")

    if not cal.empty and "overall_accuracy" in cal.columns:
        high_conf = cal[cal["overall_accuracy"].notna()].iloc[-1]
        print(f"High-confidence accuracy: {high_conf['overall_accuracy']:.0%} "
              f"(n={int(high_conf['n'])})")

if __name__ == "__main__":
    run()
