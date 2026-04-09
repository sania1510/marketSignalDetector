# ingestion/run_pipeline.py

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from price_fetcher import PriceFetcher
from macro_fetcher import MacroFetcher
from news_fetcher import NewsFetcher
from data_merger import DataMerger

def run():
    print("=" * 50)
    print("MARKET SIGNAL DETECTOR:Data Ingestion Pipeline")
    print("=" * 50)

    # 1. Fetch all data sources
    prices = PriceFetcher().get_close_prices()
    macro  = MacroFetcher().fetch_all()
    news   = NewsFetcher().fetch_live_headlines()

    # 2. Aggregate news: count headlines per day
    # (sentiment scores will be added in Step 4 after NLP model is built)
    news_daily = (news.groupby("date")
                      .size()
                      .rename("daily_headline_count")
                      .to_frame())

    # 3. Merge everything into master.csv
    merger = DataMerger()
    master = merger.merge(prices, macro, news_daily)

    print("\nPipeline complete. Sample output:")
    print(master.tail(3).to_string())

if __name__ == "__main__":
    run()