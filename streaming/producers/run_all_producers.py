# kafka/producers/run_all_producers.py
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import threading
from price_producer import PriceProducer
from news_producer  import NewsProducer

def run():
    print("Starting all producers...")

    # Create topics first
    from streaming.topics import create_topics
    create_topics()

    # Run each producer in its own thread
    threads = []

    def run_prices():
        p = PriceProducer()
        p.replay_historical()   # swap for p.run(60) in live mode

    def run_news():
        p = NewsProducer()
        p.stream_from_csv()     # swap for p.run(300) in live mode

    threads.append(threading.Thread(target=run_prices, daemon=True))
    threads.append(threading.Thread(target=run_news,   daemon=True))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("All producers finished.")

if __name__ == "__main__":
    run()
