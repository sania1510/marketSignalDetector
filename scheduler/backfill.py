# scheduler/backfill.py
#
# Run this ONCE to fill in any missing signal dates.
# It finds every date in master.csv that has no entry in signals.csv
# and generates a signal for each one.
#
# Usage (from your project root):
#   python scheduler/backfill.py

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

from signals.engine import SignalEngine, SignalConfig

# ── paths ────────────────────────────────────────────────────
master_path    = os.path.join(BASE_DIR, "data", "processed", "master.csv")
sentiment_path = os.path.join(BASE_DIR, "data", "processed", "sentiment_daily.csv")
signals_path   = os.path.join(BASE_DIR, "data", "processed", "signals.csv")

# ── load ─────────────────────────────────────────────────────
master    = pd.read_csv(master_path,    index_col=0, parse_dates=True)
sentiment = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
existing  = pd.read_csv(signals_path,   index_col=0, parse_dates=True)

log.info(f"master.csv    : {len(master)} rows  latest={str(master.index.max())[:10]}")
log.info(f"sentiment     : {len(sentiment)} rows  latest={str(sentiment.index.max())[:10]}")
log.info(f"signals.csv   : {len(existing)} rows  latest={str(existing.index.max())[:10]}")

# ── find missing dates ────────────────────────────────────────
existing_dates = set(str(idx)[:10] for idx in existing.index)
master_dates   = set(str(idx)[:10] for idx in master.index)
missing_dates  = sorted(master_dates - existing_dates)

log.info(f"Missing signal dates: {len(missing_dates)}")
for d in missing_dates:
    log.info(f"  {d}")

if not missing_dates:
    log.info("Nothing to backfill signals.csv is already up to date!")
    sys.exit(0)

# ── generate signals for each missing date ────────────────────
engine    = SignalEngine(SignalConfig())
new_rows  = []

for date_str in missing_dates:
    # find the matching timestamp in master index
    matching = [idx for idx in master.index if str(idx)[:10] == date_str]
    if not matching:
        continue
    ts = matching[0]

    # master row as Series
    m_row = master.loc[ts]

    # sentiment row as Series — exact match or forward-fill
    sent_matching = [idx for idx in sentiment.index if str(idx)[:10] == date_str]
    if sent_matching:
        s_row = sentiment.loc[sent_matching[0]]
    else:
        prior = sentiment[sentiment.index <= ts]
        if not prior.empty:
            s_row = prior.iloc[-1]
            log.warning(f"  {date_str}: no exact sentiment forward-filled from {str(prior.index[-1])[:10]}")
        else:
            s_row = None
            log.warning(f"  {date_str}: no sentiment at all running without it")

    # forecast row — plain dict
    f_row = {"prob_up": 0.5}

    try:
        sig = engine.run_single(m_row, f_row, s_row)
        new_rows.append(sig)
        log.info(f"  {date_str}:{sig['signal'].iloc[0]}  score={sig['composite_score'].iloc[0]}")
    except Exception as e:
        log.error(f"  {date_str}: FAILED - {e}")

# ── merge and save ────────────────────────────────────────────
if new_rows:
    combined = pd.concat([existing] + new_rows)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)
    combined.to_csv(signals_path)
    log.info(f"\n Backfill complete signals.csv now has {len(combined)} rows")
    log.info(f"   Latest date: {str(combined.index.max())[:10]}")
else:
    log.info("No new rows were generated.")