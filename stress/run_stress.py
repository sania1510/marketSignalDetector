# stress/run_stress.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from stress_tester import StressTester

def run():
    signals = pd.read_csv("data/processed/signals.csv",
                           index_col=0, parse_dates=True)
    master  = pd.read_csv("data/processed/master.csv",
                           index_col=0, parse_dates=True)

    print("=" * 55)
    print("STRESS TEST REPORT")
    print("=" * 55)

    tester  = StressTester(signals, master, master)
    results = tester.run_all()

    # Print clean summary
    if not results.empty:
        print("\n" + "=" * 55)
        print("SUMMARY")
        print("=" * 55)
        for _, row in results.iterrows():
            print(f"\n{row['event_name']}")
            print(f"  Market fell      : {float(row['market_return']):.1%}")
            print(f"  Max drawdown     : {float(row['max_drawdown']):.1%}")
            print(f"  First warning    : {row['first_alert_date']}")
            print(f"  Lead time        : {int(row['lead_time_days'])} days before bottom")
            print(f"  Signal accuracy  : {float(row['signal_accuracy']):.0%}")

    # Scenario simulation
    print("\n" + "=" * 55)
    print("SCENARIO: hypothetical VIX spike to 45")
    print("=" * 55)
    latest   = master.iloc[-1]
    scenario = tester.simulate_scenario(
        base_row       = latest,
        vix_shock      = 45.0,
        sentiment_shock= -0.75,
        label          = "vix45_scenario"
    )
    print(f"  Signal    : {scenario['signal']}")
    print(f"  Score     : {scenario['composite_score']:.3f}")
    for r in scenario['rationale'][:3]:
        print(f"  Rationale : {r}")

if __name__ == "__main__":
    run()