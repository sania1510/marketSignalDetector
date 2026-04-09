# kafka/consumers/run_all_consumers.py
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import threading
from feature_consumer import FeatureConsumer
from nlp_consumer     import NLPConsumer
from signal_consumer  import SignalConsumer
from db_writer        import DBWriter

def run():
    print("Starting all consumers...")
    consumers = [
        ("FeatureConsumer", FeatureConsumer),
        ("NLPConsumer",     NLPConsumer),
        ("SignalConsumer",  SignalConsumer),
        ("DBWriter",        DBWriter),
    ]
    threads = []
    for name, cls in consumers:
        t = threading.Thread(target=cls().run, name=name, daemon=True)
        threads.append(t)
        t.start()
        print(f"  Started {name}")

    print("\nAll consumers running. Press Ctrl+C to stop.")
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nShutting down.")

if __name__ == "__main__":
    run()
