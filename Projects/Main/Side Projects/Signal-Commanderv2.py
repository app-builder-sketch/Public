# app.py
# 👁️ DarkPool Commander v3 (The Architect's Final Build)
# ASYNC Execution | Instant Backtesting | AI Intelligence | Fortress Security
#
# SETUP:
# pip install streamlit ccxt pandas numpy scipy plotly openai aiohttp
#
# SECRETS (.streamlit/secrets.toml):
# [general]
# APP_PASSWORD = "..."
# OPENAI_API_KEY = "sk-..."
# TELEGRAM_BOT_TOKEN = "..."
# TELEGRAM_CHAT_ID = "..."

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict

import aiohttp
import ccxt.async_support as ccxt  # Async CCXT
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import AsyncOpenAI  # Async OpenAI
from scipy.stats import linregress

# =========================
# ⚙️ CONFIG
# =========================
APP_TITLE = "👁️ GOD MODE COMMANDER v3"
DB_PATH = "godmode_v3.db"

# Default Universe (High Liquidity)
DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", 
    "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "ADA/USDT", "TRX/USDT"
]
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h", "1d"]

# =========================
# 🎨 DPC ARCHITECTURE (CSS)
# =========================
def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
        
        :root{
            --bg: #050505;
            --card-bg: #111111;
            --border: #333;
            --neon-green: #00ff41;
            --neon-red: #ff0055;
            --neon-gold: #ffd700;
        }
        .stApp { background-color: var(--bg); font-family: 'Roboto Mono', monospace; color: #e0e0e0; }
        
        /* HEADERS */
        .glitch {
            color: white;
            font-size: 2.5em;
            font-weight: bold;
            text-transform: uppercase;
            text-shadow: 2px 2px 0px #ff0055, -2px -2px 0px #00ff41;
            letter-spacing: 2px;
            margin-bottom: 20px;
        }
        
        /* METRICS */
        div[data-testid="stMetric"] {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 10px;
            transition: 0.3s;
        }
        div[data-testid="stMetric"]:hover {
            border-color: var(--neon-green);
            box-shadow: 0 0 10px rgba(0, 255, 65, 0.2);
        }
        
        /* SIGNAL CARDS */
        .signal-card {
            background: rgba(20, 20, 20, 0.9);
            border: 1px solid var(--border);
            border-left: 3px solid #555;
            border-radius: 0px;
            padding: 15px;
            margin-bottom: 10px;
            position: relative;
        }
        .signal-card.bull { border-left-color: var(--neon-green); }
        .signal-card.bear { border-left-color: var(--neon-red); }
        
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .card-ticker { font-size: 1.2em; font-weight: bold; color: white; }
        .card-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; font-size: 0.85em; color: #aaa; }
        .stat-val { color: white; font-weight: bold; }
        
        .win-badge {
            padding: 2px 6px;
            border-radius: 2px;
            font-size: 0.7em;
            font-weight: bold;
            border: 1px solid #555;
        }
        .win-high { border-color: var(--neon-green); color: var(--neon-green); background: rgba(0,255,65,0.1); }
        .win-mid { border-color: var(--neon-gold); color: var(--neon-gold); background: rgba(255,215,0,0.1); }
        .win-low { border-color: var(--neon-red); color: var(--neon-red); background: rgba(255,0,85,0.1); }
        
        .ai-analysis {
            margin-top: 10px;
            padding: 10px;
            background: rgba(255, 215, 0, 0.05);
            border-left: 2px solid var(--neon-gold);
            font-size: 0.8em;
            color: #ddd;
        }
        
        /* LOGS */
        .log-line { font-family: 'Courier New', monospace; font-size: 0.8em; color: #666; }
    </style>
    """, unsafe_allow_html=True)

# =========================
# 🔐 FORTRESS GATE
# =========================
def check_password():
    if "APP_PASSWORD" not in st.secrets: return True
    if "password_correct" not in st.session_state: st.session_state.password_correct = False
    
    if not st.session_state.password_correct:
        pwd = st.text_input("ENTER CLEARANCE CODE:", type="password")
        if pwd == st.secrets["APP_PASSWORD"]:
            st.session_state.password_correct = True
            st.rerun()
        elif pwd:
            st.error("ACCESS DENIED")
        return False
    return True

# =========================
# 🧮 QUANT ENGINE (VECTORIZED)
# =========================
def calc_hma(series, length):
    """Hull Moving Average"""
    wma = lambda s, l: s.rolling(l).apply(lambda x: np.dot(x, np.arange(1, l+1))/(l*(l+1)/2), raw=True)
    half = int(length/2)
    sqrt = int(np.sqrt(length))
    wma1 = wma(series, half)
    wma2 = wma(series, length)
    diff = 2 * wma1 - wma2
    return wma(diff, sqrt)

def calc_indicators(df, hma_len=55, mult=1.5):
    """Calculate all technicals in one pass"""
    # ATR
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(span=14, adjust=False).mean()
    
    # Apex Trend
    df['HMA'] = calc_hma(c, hma_len)
    df['Apex_Upper'] = df['HMA'] + (df['ATR'] * mult)
    df['Apex_Lower'] = df['HMA'] - (df['ATR'] * mult)
    
    # Squeeze Momentum (Vectorized LinReg Slope proxy)
    # Slope of (Close - Avg) over 20 bars
    length = 20
    sma = c.rolling(length).mean()
    avg = (h.rolling(length).max() + l.rolling(length).min() + sma) / 3
    src = c - avg
    
    # Vectorized slope calculation using covariance
    x = np.arange(length)
    x_mean = x.mean()
    df['Sqz_Mom'] = src.rolling(length).apply(
        lambda y: ((x - x_mean) * (y - y.mean())).sum() / ((x - x_mean)**2).sum(), 
        raw=True
    )
    
    return df

# =========================
# 🧬 SIGNAL & BACKTEST
# =========================
@dataclass
class Signal:
    symbol: str
    timeframe: str
    bias: str
    setup: str
    entry: float
    stop: float
    tp: float
    win_rate: float # Historical Win Rate
    ev: float       # Expected Value
    sqz_mom: float
    ai_msg: str = ""

def backtest_signal(df, signal_type, hma_len, mult):
    """Instant Backtest: Checks last 100 bars for similar setups"""
    wins = 0
    total = 0
    
    # Logic matches generate_signal
    # Shifted df to simulate historical execution
    for i in range(max(hma_len, 50), len(df)-1):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        bias = "Bull" if prev['close'] > prev['Apex_Upper'] else "Bear" if prev['close'] < prev['Apex_Lower'] else "Neutral"
        trigger = False
        
        if signal_type == "Apex Reclaim":
            if bias == "Bull" and row['close'] > row['HMA'] and row['Sqz_Mom'] > 0 and prev['Sqz_Mom'] < 0:
                trigger = True
                target = row['close'] + (row['close'] - (row['HMA'] - row['ATR'])) * 2.0
                stop = row['HMA'] - row['ATR']
            elif bias == "Bear" and row['close'] < row['HMA'] and row['Sqz_Mom'] < 0 and prev['Sqz_Mom'] > 0:
                trigger = True
                target = row['close'] - ((row['HMA'] + row['ATR']) - row['close']) * 2.0
                stop = row['HMA'] + row['ATR']
        
        if trigger:
            total += 1
            # Check outcome in next 10 candles
            outcome = "Loss"
            for j in range(1, 11):
                if i+j >= len(df): break
                fut = df.iloc[i+j]
                if bias == "Bull":
                    if fut['high'] >= target: outcome = "Win"; break
                    if fut['low'] <= stop: outcome = "Loss"; break
                else:
                    if fut['low'] <= target: outcome = "Win"; break
                    if fut['high'] >= stop: outcome = "Loss"; break
            if outcome == "Win": wins += 1
            
    return (wins / total * 100) if total > 0 else 0.0

def analyze_market(df, symbol, tf, hma_len, mult):
    df = calc_indicators(df, hma_len, mult)
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Apex State
    bias = "Bull" if prev['close'] > prev['Apex_Upper'] else "Bear" if prev['close'] < prev['Apex_Lower'] else "Neutral"
    
    sig = None
    
    # Setup: Apex Reclaim
    if bias == "Bull":
        if curr['close'] > curr['HMA'] and curr['Sqz_Mom'] > 0 and prev['Sqz_Mom'] < 0:
            stop = curr['HMA'] - curr['ATR']
            tp = curr['close'] + (curr['close'] - stop) * 2.0
            wr = backtest_signal(df, "Apex Reclaim", hma_len, mult)
            sig = Signal(symbol, tf, "BULL", "Apex Reclaim", curr['close'], stop, tp, wr, 0, curr['Sqz_Mom'])
            
    elif bias == "Bear":
        if curr['close'] < curr['HMA'] and curr['Sqz_Mom'] < 0 and prev['Sqz_Mom'] > 0:
            stop = curr['HMA'] + curr['ATR']
            tp = curr['close'] - (stop - curr['close']) * 2.0
            wr = backtest_signal(df, "Apex Reclaim", hma_len, mult)
            sig = Signal(symbol, tf, "BEAR", "Apex Reclaim", curr['close'], stop, tp, wr, 0, curr['Sqz_Mom'])
            
    return sig

# =========================
# ⚡ ASYNC CORE
# =========================
async def fetch_and_analyze(ex_id, symbol, tf, hma_len, mult, openai_key):
    try:
        ex_class = getattr(ccxt, ex_id)
        async with ex_class() as ex:
            ohlcv = await ex.fetch_ohlcv(symbol, tf, limit=200)
            df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            
            # CPU-bound math run in executor to avoid blocking event loop
            sig = await asyncio.to_thread(analyze_market, df, symbol, tf, hma_len, mult)
            
            if sig and openai_key:
                # Async AI Call
                prompt = f"Analyze {symbol} {tf}. Trend: {sig.bias}. Signal: {sig.setup}. WinRate (Last 200): {sig.win_rate:.1f}%. Momentum: {sig.sqz_mom:.2f}. Give 1 sentence on key risk."
                client = AsyncOpenAI(api_key=openai_key)
                res = await client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}], max_tokens=60)
                sig.ai_msg = res.choices[0].message.content
                
            return sig
    except Exception as e:
        return None

async def scan_universe(ex_id, symbols, tf, hma_len, mult, openai_key):
    tasks = [fetch_and_analyze(ex_id, s, tf, hma_len, mult, openai_key) for s in symbols]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

# =========================
# 🖥️ DASHBOARD
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="👁️")
    inject_css()
    
    if not check_password(): st.stop()
    
    # DB Init
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS signals (timestamp TEXT, symbol TEXT, bias TEXT, setup TEXT, entry REAL, wr REAL)")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ SYSTEM PARAMETERS")
        ex_id = st.selectbox("Exchange", ["binance", "bybit", "mexc"], index=0)
        tf = st.selectbox("Timeframe", DEFAULT_TIMEFRAMES, index=1)
        universe = st.text_area("Symbols", ", ".join(DEFAULT_SYMBOLS)).split(",")
        universe = [s.strip() for s in universe]
        
        st.markdown("---")
        st.markdown("### 🧠 GOD MODE LOGIC")
        hma_len = st.number_input("Apex HMA", value=55)
        mult = st.number_input("ATR Mult", value=1.5)
        
        st.markdown("---")
        openai_key = st.secrets.get("OPENAI_API_KEY", "")
        if not openai_key: st.error("NO AI KEY")

    # Header
    st.markdown('<div class="glitch">👁️ GOD MODE COMMANDER v3</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🚀 INITIATE ASYNC SCAN", type="primary"):
            if not openai_key:
                st.warning("AI Key Missing - Scanning without Intelligence.")
                
            status = st.empty()
            status.text("⚡ Spinning up AsyncIO Event Loop...")
            
            # Run Async Scan
            start = time.time()
            signals = asyncio.run(scan_universe(ex_id, universe, tf, hma_len, mult, openai_key))
            duration = time.time() - start
            
            status.success(f"SCAN COMPLETE: {len(universe)} symbols in {duration:.2f}s")
            
            # Display
            if not signals:
                st.info("No Signals Detected. Market is sleeping.")
            else:
                for sig in signals:
                    # Save
                    conn.execute("INSERT INTO signals VALUES (?,?,?,?,?,?)", 
                                 (datetime.now().isoformat(), sig.symbol, sig.bias, sig.setup, sig.entry, sig.win_rate))
                    conn.commit()
                    
                    # Style
                    css = "bull" if sig.bias == "BULL" else "bear"
                    wr_css = "win-high" if sig.win_rate > 60 else "win-mid" if sig.win_rate > 45 else "win-low"
                    
                    st.markdown(f"""
                    <div class="signal-card {css}">
                        <div class="card-header">
                            <span class="card-ticker">{sig.symbol} <span style="font-size:0.6em; color:#888;">{sig.timeframe}</span></span>
                            <span class="win-badge {wr_css}">WR: {sig.win_rate:.1f}%</span>
                        </div>
                        <div class="card-stats">
                            <div>
                                <div style="color:#888;">SETUP</div>
                                <div class="stat-val">{sig.setup}</div>
                            </div>
                            <div>
                                <div style="color:#888;">ENTRY</div>
                                <div class="stat-val">{sig.entry}</div>
                            </div>
                            <div>
                                <div style="color:#888;">TARGET</div>
                                <div class="stat-val" style="color:var(--neon-green)">{sig.tp:.4f}</div>
                            </div>
                        </div>
                        <div class="ai-analysis">
                            🤖 <b>AI INTELLIGENCE:</b> {sig.ai_msg}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📜 LIVE FEED")
        try:
            hist = pd.read_sql("SELECT symbol, bias, entry, wr FROM signals ORDER BY rowid DESC LIMIT 10", conn)
            st.dataframe(hist, hide_index=True, use_container_width=True)
        except:
            st.caption("No History")

if __name__ == "__main__":
    main()
