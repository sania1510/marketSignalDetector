# scheduler/run_scheduler.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding='utf-8')
import schedule
import time
import logging
from datetime import datetime
from daily_update import DailyUpdater

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SCHEDULER] %(message)s"
)
log = logging.getLogger(__name__)

def run_daily():
    log.info("Scheduled daily update triggered")
    updater = DailyUpdater()
    updater.run()

# ── Schedule configuration ────────────────────────────────────────────────────
# US stock market opens at 9:30 AM EST
# We run at 6:30 AM EST to prepare predictions before market open
# Adjust this time to your timezone

schedule.every().monday.at("06:30").do(run_daily)
schedule.every().tuesday.at("06:30").do(run_daily)
schedule.every().wednesday.at("06:30").do(run_daily)
schedule.every().thursday.at("06:30").do(run_daily)
schedule.every().friday.at("06:30").do(run_daily)
# Weekends: full retrain (daily_update handles this internally)
schedule.every().saturday.at("07:00").do(run_daily)

log.info("Scheduler started. Daily update runs at 06:30 on weekdays.")
log.info("Press Ctrl+C to stop.")

# Run once immediately on startup so you have fresh data now
log.info("Running initial update on startup...")
run_daily()

while True:
    schedule.run_pending()
    time.sleep(60)   # check every minute