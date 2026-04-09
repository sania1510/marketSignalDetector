# download_all_data.py
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import urllib.request
import pandas as pd
import yfinance as yf
import time

FRED_KEY    = "d53ff20c7998022d5c81e76b9cb08af9"     # fred.stlouisfed.org
NEWS_KEY    = "32d10e6c013741b4a0d6a66f31a6fb15"  # newsapi.org
START_DATE  = "2018-01-01"
END_DATE    = "2024-01-01"

# ── Create folders ─────────────────────────────────────────────────────────────
for folder in ["data/raw/prices", "data/raw/macro",
               "data/raw/news",   "data/processed",
               "models/saved"]:
    os.makedirs(folder, exist_ok=True)

print("=" * 50)
print("STEP 1 — Price data (yfinance)")
print("=" * 50)
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "GLD", "TLT"]
for ticker in TICKERS:
    path = f"data/raw/prices/{ticker}.csv"
    if os.path.exists(path):
        print(f"  {ticker} already exists — skipping")
        continue
    print(f"  Downloading {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    df.to_csv(path)
    print(f"  Saved {len(df)} rows → {path}")

print("\n" + "=" * 50)
print("STEP 2 — Macro data (FRED)")
print("=" * 50)
SERIES = {
    "VIX":          "VIXCLS",
    "FedFunds":     "FEDFUNDS",
    "CPI":          "CPIAUCSL",
    "Unemployment": "UNRATE",
    "T10Y2Y":       "T10Y2Y",
}
try:
    from fredapi import Fred
    fred = Fred(api_key=FRED_KEY)
    for name, series_id in SERIES.items():
        path = f"data/raw/macro/{name}.csv"
        if os.path.exists(path):
            print(f"  {name} already exists — skipping")
            continue
        print(f"  Downloading {name}...")
        s = fred.get_series(series_id, observation_start=START_DATE)
        s.name = "value"
        s.index.name = "Date"
        s.to_csv(path, header=True)
        print(f"  Saved {len(s)} rows → {path}")
except Exception as e:
    print(f"  FRED error: {e}")
    print("  Get free key at: fred.stlouisfed.org/docs/api/api_key.html")

print("\n" + "=" * 50)
print("STEP 3 — PhraseBank (HuggingFace)")
print("=" * 50)
path = "data/raw/news/phrasebank.csv"
if os.path.exists(path):
    print("  phrasebank.csv already exists — skipping")
else:
    try:
        url = ("https://huggingface.co/datasets/takala/financial_phrasebank"
               "/resolve/main/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt")
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("latin-1")
        label_map = {"negative":0,"neutral":1,"positive":2}
        rows = []
        for line in content.splitlines():
            line = line.strip()
            if "@" not in line:
                continue
            parts = line.rsplit("@", 1)
            if len(parts) == 2:
                text  = parts[0].strip()
                label = parts[1].strip().lower()
                if label in label_map:
                    rows.append({"text":text,"label":label_map[label]})
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(f"  Saved {len(df)} rows → {path}")
    except Exception as e:
        print(f"  Error: {e} — creating synthetic fallback")
        rows = []
        for text, label in [
            ("Company beats earnings expectations significantly", 2),
            ("Revenue grew 20 percent driven by strong demand", 2),
            ("Operating profit surged on cost cutting measures", 2),
            ("Stock hits record high after blowout results", 2),
            ("Dividend raised reflecting strong cash generation", 2),
            ("Company misses earnings estimates badly", 0),
            ("Profit warning issued as demand collapses", 0),
            ("Mass layoffs announced cutting 10 percent of workforce", 0),
            ("Debt levels raise serious liquidity concerns", 0),
            ("Credit rating downgraded on deteriorating outlook", 0),
            ("Results in line with analyst consensus estimates", 1),
            ("Company will report earnings next Tuesday", 1),
            ("Board meeting scheduled for next month", 1),
            ("Trading volume in line with 30 day average", 1),
            ("Management reaffirms existing guidance range", 1),
        ] * 20:
            rows.append({"text":text,"label":label})
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Saved synthetic {len(rows)} rows → {path}")

print("\n" + "=" * 50)
print("STEP 4 — News headlines (NewsAPI)")
print("=" * 50)
path = "data/raw/news/headlines.csv"
if os.path.exists(path):
    print("  headlines.csv already exists — skipping")
else:
    try:
        from newsapi import NewsApiClient
        api  = NewsApiClient(api_key=NEWS_KEY)
        rows = []
        for query in ["S&P 500", "Federal Reserve", "US inflation",
                      "Wall Street", "stock market crash"]:
            resp = api.get_everything(q=query, language="en",
                                       sort_by="publishedAt", page_size=100)
            for a in resp.get("articles", []):
                title = a.get("title","")
                if not title or title == "[Removed]":
                    continue
                rows.append({
                    "date":        a["publishedAt"][:10],
                    "title":       title,
                    "description": a.get("description",""),
                    "source":      a["source"]["name"],
                })
            time.sleep(0.3)
        df = pd.DataFrame(rows).drop_duplicates(subset="title")
        df.to_csv(path, index=False)
        print(f"  Saved {len(df)} headlines → {path}")
    except Exception as e:
        print(f"  NewsAPI error: {e}")
        print("  Creating placeholder headlines.csv")
        pd.DataFrame([
            {"date":"2023-06-01","title":"Fed holds rates steady",
             "description":"Federal Reserve keeps rates unchanged",
             "source":"Reuters"},
            {"date":"2023-06-02","title":"S&P 500 rises on strong jobs data",
             "description":"Markets rallied after better than expected payrolls",
             "source":"Bloomberg"},
            {"date":"2023-06-05","title":"Inflation cools to 3 year low",
             "description":"CPI data came in below expectations",
             "source":"CNBC"},
        ]).to_csv(path, index=False)
        print(f"  Saved placeholder → {path}")

print("\n" + "=" * 50)
print("VERIFICATION")
print("=" * 50)
checks = [
    ("data/raw/prices/SPY.csv",          "prices"),
    ("data/raw/prices/QQQ.csv",          "prices"),
    ("data/raw/prices/AAPL.csv",         "prices"),
    ("data/raw/prices/MSFT.csv",         "prices"),
    ("data/raw/prices/TSLA.csv",         "prices"),
    ("data/raw/prices/GLD.csv",          "prices"),
    ("data/raw/prices/TLT.csv",          "prices"),
    ("data/raw/macro/VIX.csv",           "macro"),
    ("data/raw/macro/FedFunds.csv",      "macro"),
    ("data/raw/macro/CPI.csv",           "macro"),
    ("data/raw/macro/Unemployment.csv",  "macro"),
    ("data/raw/macro/T10Y2Y.csv",        "macro"),
    ("data/raw/news/phrasebank.csv",     "news"),
    ("data/raw/news/headlines.csv",      "news"),
]
all_ok = True
for path, category in checks:
    if os.path.exists(path):
        df   = pd.read_csv(path)
        print(f"  OK  {os.path.basename(path):<25} {len(df):>5} rows")
    else:
        print(f"  MISSING  {path}")
        all_ok = False

print()
if all_ok:
    print("All 14 raw files present.")
    print("\nNow run the pipeline in this order:")
    print("  python ingestion/run_pipeline.py")
    print("  python generate_historical_sentiment.py")
    print("  python models/run_nlp.py")
    print("  python models/run_forecaster.py")
    print("  python signals/run_signals.py")
    print("  python stress/run_stress.py")
    print("  python backtest/run_backtest.py")
    print("  uvicorn api.main:app --reload --port 8000")
else:
    print("Some files missing — check errors above.")