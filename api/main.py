# api/main.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import asyncio
import threading
from typing import List
from datetime import datetime

import pandas as pd
import numpy as np
import google.generativeai as genai
from supabase import create_client

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------------- APP ----------------
app = FastAPI(title="Market Signal Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="frontend/static", check_dir=False), name="static")

# ---------------- BASE DIR ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------- SUPABASE CLIENT ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- SUPABASE HELPER ----------------
def sb_load(table: str, date_col: str = "date") -> pd.DataFrame:
    """Load a full table from Supabase into a DataFrame."""
    try:
        res = supabase.table(table).select("*").execute()
        if not res.data:
            print(f"⚠️  Supabase table '{table}' is empty")
            return pd.DataFrame()
        df = pd.DataFrame(res.data)
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
        elif "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
        return df
    except Exception as e:
        print(f"❌ sb_load({table}) failed: {e}")
        return pd.DataFrame()

# ---------------- INDEX ----------------
@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

# ---------------- OVERVIEW ----------------
@app.get("/api/overview")
def overview():
    signals = sb_load("signals")
    sent    = sb_load("sentiment_daily")

    if signals.empty:
        return {"error": "No data"}

    latest = signals.iloc[-1]

    sent["sentiment_3d_ma"] = sent["sentiment_3d_ma"].ffill().fillna(0) if not sent.empty else pd.Series([0])
    sent["bullish_ratio"]   = sent["bullish_ratio"].ffill().fillna(0.5) if not sent.empty else pd.Series([0.5])

    return {
        "latest_signal":    latest.get("signal", "HOLD"),
        "composite_score":  float(latest.get("composite_score", 0)),
        "vix":              float(latest.get("vix", 20)),
        "sentiment_3d_ma":  float(sent.iloc[-1].get("sentiment_3d_ma", 0)) if not sent.empty else 0,
        "bullish_ratio":    float(sent.iloc[-1].get("bullish_ratio", 0))   if not sent.empty else 0,
        "total_days":       len(signals),
        "signal_counts":    signals["signal"].value_counts().to_dict()
    }

# ---------------- PRICES ----------------
@app.get("/api/prices")
def prices():
    df = sb_load("master", date_col="Date")

    if df.empty or "SPY" not in df.columns:
        return {"dates": [], "prices": []}

    return {
        "dates":  df.index.astype(str).tolist(),
        "prices": df["SPY"].fillna(0).tolist()
    }

# ---------------- SIGNALS ----------------
@app.get("/api/signals")
def signals_api():
    df = sb_load("signals")

    if df.empty:
        return {"signals": []}

    df = df.reset_index().rename(columns={"index": "date", "date": "date"})

    return {"signals": df.to_dict(orient="records")}

# ---------------- SENTIMENT ----------------
@app.get("/api/sentiment")
def sentiment():
    df = sb_load("sentiment_daily")

    if df.empty:
        return {"dates": [], "sentiment": [], "bullish": []}

    df["sentiment_3d_ma"] = df["sentiment_3d_ma"].ffill().fillna(0)
    df["bullish_ratio"]   = df["bullish_ratio"].ffill().fillna(0.5)

    return {
        "dates":     df.index.astype(str).tolist(),
        "sentiment": df["sentiment_3d_ma"].round(4).tolist(),
        "bullish":   df["bullish_ratio"].round(4).tolist()
    }

# ---------------- BACKTEST ----------------
@app.get("/api/backtest")
def backtest():
    equity  = sb_load("equity_curve")
    master  = sb_load("master", date_col="Date")
    trades  = sb_load("trades_log")

    if equity.empty:
        return {"error": "Run backtest first: python backtest/run_backtest.py"}

    if "equity" not in equity.columns:
        equity.columns = ["equity"]

    equity_series = equity["equity"]
    spy = master["SPY"].dropna() if not master.empty and "SPY" in master.columns else pd.Series()

    initial_capital = equity_series.iloc[0]

    if not spy.empty:
        spy_aligned   = spy.reindex(equity_series.index, method="ffill").dropna()
        bh_shares     = initial_capital / spy_aligned.iloc[0]
        benchmark_eq  = (bh_shares * spy_aligned).reindex(equity_series.index).ffill()
        benchmark_ret = float((benchmark_eq.iloc[-1] / initial_capital) - 1)
    else:
        benchmark_eq  = equity_series * 0
        benchmark_ret = 0.0

    total_return = float((equity_series.iloc[-1] / initial_capital) - 1)

    rolling_max  = equity_series.cummax()
    drawdown_pct = ((equity_series - rolling_max) / rolling_max * 100).round(2)
    max_drawdown = float(drawdown_pct.min()) / 100

    daily_ret  = equity_series.pct_change().dropna()
    excess_ret = daily_ret - 0.04 / 252
    sharpe     = float(excess_ret.mean() / (excess_ret.std() + 1e-9) * np.sqrt(252))

    win_rate     = 0.0
    total_trades = 0
    if not trades.empty and "pnl_pct" in trades.columns:
        closed = trades[trades["pnl_pct"] != 0.0]
        if len(closed):
            win_rate     = float((closed["pnl_pct"] > 0).mean())
            total_trades = len(closed)

    return {
        "metrics": {
            "total_return":     round(total_return, 4),
            "benchmark_return": round(benchmark_ret, 4),
            "sharpe_ratio":     round(sharpe, 3),
            "max_drawdown":     round(max_drawdown, 4),
            "win_rate":         round(win_rate, 3),
            "total_trades":     total_trades,
            "alpha":            round(total_return - benchmark_ret, 4),
        },
        "equity_curve": {
            "dates":     equity_series.index.astype(str).tolist(),
            "equity":    equity_series.round(2).tolist(),
            "benchmark": benchmark_eq.round(2).tolist() if not spy.empty else [],
        },
        "drawdown": {
            "dates":  equity_series.index.astype(str).tolist(),
            "values": drawdown_pct.tolist(),
        }
    }

# ---------------- STRESS ----------------
@app.get("/api/stress")
def stress():
    df = sb_load("stress_results")

    if df.empty:
        return {"events": []}

    df = df.reset_index(drop=True)
    df = df.fillna({
        "signal_accuracy":  0.0,
        "false_alarm_rate": 0.0,
        "max_drawdown":     0.0,
        "market_return":    0.0,
        "lead_time_days":   0,
        "first_alert_date": "No alert fired",
        "notes":            "",
    })

    return {"events": df.to_dict(orient="records")}

# ---------------- SIMULATE ----------------
@app.post("/api/simulate")
def simulate(body: dict):
    score = (
        body.get("prob_up", 0) * 0.4 +
        body.get("sentiment", 0) * 0.2 -
        body.get("vix", 20) * 0.01
    )

    signal = "BUY" if score > 0.1 else "SELL" if score < -0.1 else "HOLD"

    return {
        "signal":           signal,
        "composite_score":  round(score, 3),
        "strength":         "strong" if abs(score) > 0.3 else "medium",
        "confidence":       str(round(abs(score), 2)),
        "lstm_score":       body.get("prob_up", 0),
        "sentiment_score":  body.get("sentiment", 0),
        "volatility_score": -body.get("vix", 20) / 50,
        "rationale": [
            "Based on probability and sentiment",
            "Higher VIX reduces confidence"
        ]
    }

# ---------------- LIVE PRICES ----------------
@app.get("/api/live_prices")
def live_prices():
    df = sb_load("master", date_col="Date")

    if df.empty:
        return [
            {"symbol": "SPY",  "price": 520, "change": 0.5},
            {"symbol": "QQQ",  "price": 430, "change": -0.2},
            {"symbol": "AAPL", "price": 190, "change": 1.1},
        ]

    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]
    result  = []

    for s in symbols:
        if s not in df.columns:
            continue

        col = df[s].dropna()

        if len(col) < 2:
            price  = round(float(col.iloc[-1]), 2) if len(col) == 1 else 0.0
            change = 0.0
        else:
            prev_close = float(col.iloc[-2])
            last_close = float(col.iloc[-1])
            price      = round(last_close, 2)
            change     = round(((last_close - prev_close) / prev_close) * 100, 2) if prev_close != 0 else 0.0

        result.append({"symbol": s, "price": price, "change": change})

    return result

# ---------------- WEBSOCKET ----------------
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

# ---------------- CHAT API ----------------
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_KEY)

@app.post("/api/chat")
async def chat(body: dict):
    user_message = body.get("message", "")
    if not user_message:
        return {"error": "No message"}

    try:
        signals = sb_load("signals")
        sent    = sb_load("sentiment_daily")
        lstm    = sb_load("lstm_predictions")
        master  = sb_load("master", date_col="Date")

        latest_sig  = signals.iloc[-1]
        latest_sent = sent.iloc[-1]   if not sent.empty   else {}
        latest_lstm = lstm.iloc[-1]   if not lstm.empty   else {}
        latest_price= master.iloc[-1] if not master.empty else {}

        actual_signal   = latest_sig.get("signal", "HOLD")
        composite_score = float(latest_sig.get("composite_score", 0))
        confidence      = float(latest_sig.get("confidence", 0))
        rationale       = latest_sig.get("rationale", "")
        vix             = float(latest_sig.get("vix", 20))
        prob_up         = float(latest_lstm.get("prob_up", 0.5)) if hasattr(latest_lstm, 'get') else 0.5
        sentiment_3d    = float(latest_sent.get("sentiment_3d_ma", 0)) if hasattr(latest_sent, 'get') else 0
        bullish_ratio   = float(latest_sent.get("bullish_ratio", 0.5)) if hasattr(latest_sent, 'get') else 0.5
        spy_price       = float(latest_price.get("SPY", 0)) if hasattr(latest_price, 'get') else 0
        signal_date     = str(signals.index[-1].date()) if not signals.empty else "unknown"

        recent = signals.tail(5)[["signal", "composite_score"]].to_string()

        system_prompt = f"""You are the AI assistant for Market Signal Detector, a quantitative trading system.

CURRENT SYSTEM STATE (as of {signal_date}):
- Signal:          {actual_signal}
- Composite Score: {composite_score:.4f}
- Confidence:      {confidence:.4f}
- Rationale:       {rationale}
- VIX:             {vix:.2f}
- LSTM P(up):      {prob_up:.3f}
- Sentiment 3d MA: {sentiment_3d:.3f}
- Bullish Ratio:   {bullish_ratio:.3f}
- SPY Price:       {spy_price:.2f}

RECENT SIGNALS (last 5 days):
{recent}

INSTRUCTIONS:
- Always base your answers on the actual system data above
- If asked about today's signal, say it is {actual_signal} with score {composite_score:.4f}
- Do not contradict the system's actual signal
- Be concise (2-4 lines max)
- You may explain why the signal was generated using the rationale
- If asked general market questions, still reference the system state
"""

        model    = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(f"{system_prompt}\n\nUser: {user_message}")

        return {
            "response":  response.text,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------- SCHEDULER ----------------
import schedule
import time
from scheduler.daily_update import DailyUpdater

def run_scheduler():
    updater = DailyUpdater()
    schedule.every().day.at("07:00").do(updater.run)

    while True:
        schedule.run_pending()
        time.sleep(60)

threading.Thread(target=run_scheduler, daemon=True).start()
