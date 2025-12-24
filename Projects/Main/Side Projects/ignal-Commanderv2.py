# app.py
# 👁️ DarkPool Signal Commander (God Mode Edition)
# Institutional-grade signal engine fusing CCXT data with Apex/Squeeze math.
#
# ARCHITECTURE:
# - Data: CCXT (Direct Exchange Feed)
# - Math: Apex Trend (HMA+ATR) + Squeeze Momentum (LinReg)
# - State: SQLite3 (Local persistence)
# - Comms: Telegram (Rich HTML alerts)
#
# SETUP:
# pip install streamlit ccxt pandas numpy scipy plotly requests
#
# SECRETS (.streamlit/secrets.toml):
# TELEGRAM_BOT_TOKEN = "..."
# TELEGRAM_CHAT_ID = "..."

import os
import time
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import linregress

# Optional dependencies check
try:
    import ccxt
    CCXT_OK = True
except ImportError:
    CCXT_OK = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# =========================
# ⚙️ CONFIG & CONSTANTS
# =========================
APP_TITLE = "👁️ DarkPool Signal Commander"
DB_PATH = "godmode_signals.db"

DEFAULT_TIMEFRAMES = ["15m", "1h", "4h", "1d"]
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "DOGE/USDT", "AVAX/USDT"]

# God Mode Defaults
DEFAULT_HMA_LEN = 55
DEFAULT_APEX_MULT = 1.5
DEFAULT_SQZ_LEN = 20
DEFAULT_SQZ_MULT = 2.0
DEFAULT_RISK_REWARD = 2.0

# Throttle
DEFAULT_MAX_SIGNALS = 5
DEFAULT_COOLDOWN = 60  # Minutes

# =========================
# 🎨 UI THEME (DPC ARCHITECTURE)
# =========================
def inject_css():
    st.markdown("""
    <style>
        :root{
            --bg: #0e1117;
            --card: #161b22;
            --border: #30363d;
            --text: #e6edf3;
            --bull: #00f5d4;
            --bear: #ff2e63;
            --neon: #00ff00;
        }
        .stApp { background-color: var(--bg); color: var(--text); font-family: 'Roboto Mono', monospace; }
        
        /* Neon Glow Text */
        .glow { 
            color: #ffffff; 
            text-shadow: 0 0 10px rgba(0, 255, 0, 0.4); 
        }
        
        /* Metrics */
        div[data-testid="stMetric"] {
            background-color: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        div[data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.8rem; }
        div[data-testid="stMetricValue"] { color: var(--text); font-weight: 700; }
        
        /* Custom Cards */
        .signal-card {
            background: linear-gradient(180deg, rgba(22, 27, 34, 0.9), rgba(14, 17, 23, 0.95));
            border: 1px solid var(--border);
            border-left: 4px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .signal-card.bull { border-left-color: var(--bull); }
        .signal-card.bear { border-left-color: var(--bear); }
        
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            margin-right: 6px;
        }
        .bg-bull { background: rgba(0, 245, 212, 0.15); color: var(--bull); border: 1px solid rgba(0, 245, 212, 0.3); }
        .bg-bear { background: rgba(255, 46, 99, 0.15); color: var(--bear); border: 1px solid rgba(255, 46, 99, 0.3); }
        .bg-neutral { background: rgba(139, 148, 158, 0.15); color: #8b949e; border: 1px solid rgba(139, 148, 158, 0.3); }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            background-color: var(--card);
            border: 1px solid var(--border);
            color: #8b949e;
            border-radius: 4px 4px 0 0;
        }
        .stTabs [aria-selected="true"] {
            border-bottom-color: var(--neon);
            color: var(--neon);
        }
    </style>
    """, unsafe_allow_html=True)

# =========================
# 🧮 GOD MODE MATH ENGINE
# =========================
def calculate_wma(series: pd.Series, length: int) -> pd.Series:
    """Vectorized Weighted Moving Average."""
    return series.rolling(length).apply(
        lambda x: np.dot(x, np.arange(1, length + 1)) / (length * (length + 1) / 2), 
        raw=True
    )

def calculate_hma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average (Responsive Trend)."""
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    
    wma_half = calculate_wma(series, half_length)
    wma_full = calculate_wma(series, length)
    
    diff = 2 * wma_half - wma_full
    return calculate_wma(diff, sqrt_length)

def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """True Range & ATR."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()

def calculate_linreg_slope(series: pd.Series, length: int = 20) -> pd.Series:
    """Rolling Linear Regression Slope (Vectorized-ish)."""
    # For speed in loops, we use a simplified momentum proxy if strictly necessary,
    # but scipy is robust for standard timeframes.
    def get_slope(y):
        x = np.arange(len(y))
        slope, _, _, _, _ = linregress(x, y)
        return slope
    
    return series.rolling(window=length).apply(get_slope, raw=True)

def calculate_apex_trend(df: pd.DataFrame, len_hma: int, mult_atr: float) -> pd.DataFrame:
    """
    Apex Trend Logic:
    - Base: HMA(Close)
    - Bands: Base +/- (ATR * Mult)
    - Bias: Price > Upper (Bull), Price < Lower (Bear)
    """
    df["HMA"] = calculate_hma(df["close"], len_hma)
    df["ATR"] = calculate_atr(df, len_hma) # Using same length for smooth bands
    
    df["Apex_Upper"] = df["HMA"] + (df["ATR"] * mult_atr)
    df["Apex_Lower"] = df["HMA"] - (df["ATR"] * mult_atr)
    
    # Determine Bias
    # 1 = Bull, -1 = Bear, 0 = Chop/Neutral (inside cloud)
    # Logic: Hold previous state if inside cloud
    
    conditions = [
        df["close"] > df["Apex_Upper"],
        df["close"] < df["Apex_Lower"]
    ]
    choices = [1, -1]
    
    # Initialize with 0
    df["Apex_State_Raw"] = np.select(conditions, choices, default=0)
    
    # Forward fill logic for state persistence (avoiding 0s if we want strict trend)
    # If 0, keep previous non-zero.
    df["Apex_State"] = df["Apex_State_Raw"].replace(to_replace=0, method='ffill')
    
    return df

def calculate_squeeze(df: pd.DataFrame, length: int, mult: float) -> pd.DataFrame:
    """
    TTM Squeeze Logic:
    - BB: SMA +/- StdDev
    - KC: SMA +/- ATR
    - Squeeze ON: BB inside KC
    - Momentum: LinReg of (Close - Avg(High, Low))
    """
    # Bollinger Bands
    df["SMA"] = df["close"].rolling(length).mean()
    df["StdDev"] = df["close"].rolling(length).std()
    df["BB_Upper"] = df["SMA"] + (df["StdDev"] * mult)
    df["BB_Lower"] = df["SMA"] - (df["StdDev"] * mult)
    
    # Keltner Channels
    df["KC_ATR"] = calculate_atr(df, length)
    df["KC_Upper"] = df["SMA"] + (df["KC_ATR"] * 1.5)
    df["KC_Lower"] = df["SMA"] - (df["KC_ATR"] * 1.5)
    
    # Squeeze Condition
    df["Squeeze_On"] = (df["BB_Upper"] < df["KC_Upper"]) & (df["BB_Lower"] > df["KC_Lower"])
    
    # Momentum (Linear Regression of deviation from mean)
    # Source: Close - (Highest + Lowest + SMA)/3
    highest = df["high"].rolling(length).max()
    lowest = df["low"].rolling(length).min()
    avg_val = (highest + lowest + df["SMA"]) / 3.0
    
    source = df["close"] - avg_val
    df["Sqz_Mom"] = calculate_linreg_slope(source, length)
    
    return df

# =========================
# 🧠 INTELLIGENCE LAYER
# =========================
@dataclass
class Signal:
    exchange: str
    symbol: str
    timeframe: str
    bias: str          # BULL / BEAR
    setup: str         # Apex Reclaim / Squeeze Firing
    entry: float
    stop: float
    tp1: float
    tp2: float
    rr: float
    notes: str
    timestamp: str

def generate_god_mode_signal(
    df: pd.DataFrame, 
    exchange: str, 
    symbol: str, 
    timeframe: str,
    hma_len: int,
    apex_mult: float,
    min_rr: float
) -> Optional[Signal]:
    
    # Ensure sufficient data
    if len(df) < max(hma_len, 50) + 10:
        return None
        
    # Run Math
    df = calculate_apex_trend(df, hma_len, apex_mult)
    df = calculate_squeeze(df, 20, 2.0)
    
    # Analysis on CLOSED candle (index -2), confirming with current (-1)
    # actually, for signals, we usually look at the last completed candle (-1 if data excludes current forming, -2 if includes)
    # CCXT fetch_ohlcv usually includes forming candle as last row.
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    atr_now = curr["ATR"]
    if atr_now == 0: return None
    
    bias = "Bull" if prev["Apex_State"] == 1 else "Bear" if prev["Apex_State"] == -1 else "Neutral"
    
    signal_obj = None
    
    # --- SETUP 1: Apex Trend Reclaim (Pullback Entry) ---
    # Price dipped into cloud (between Base and Band) and closed back outside/strong
    
    if bias == "Bull":
        # Check for pullback: Low was near HMA recently
        # Trigger: Momentum positive + Close > HMA
        
        is_uptrend = curr["close"] > curr["HMA"]
        mom_rising = curr["Sqz_Mom"] > 0 and curr["Sqz_Mom"] > prev["Sqz_Mom"]
        
        # Simple Apex entry: Trend is Bull, Price Closed above HMA, Mom Rising
        if is_uptrend and mom_rising:
            entry = curr["close"]
            stop = curr["HMA"] - (atr_now * 1.0) # Tight stop below HMA
            tp1 = entry + (entry - stop) * min_rr
            tp2 = entry + (entry - stop) * (min_rr * 2)
            rr = (tp1 - entry) / (entry - stop)
            
            # Filter: Check if we just flipped or if mom is accelerating
            if prev["Sqz_Mom"] < 0 or (prev["close"] < prev["HMA"]):
                signal_obj = Signal(
                    exchange, symbol, timeframe, "BULL", "Apex Trend Reclaim",
                    entry, stop, tp1, tp2, rr, 
                    f"Price reclaimed Apex Baseline. Sqz Mom: {curr['Sqz_Mom']:.4f}",
                    datetime.now(timezone.utc).isoformat()
                )

    elif bias == "Bear":
        is_downtrend = curr["close"] < curr["HMA"]
        mom_falling = curr["Sqz_Mom"] < 0 and curr["Sqz_Mom"] < prev["Sqz_Mom"]
        
        if is_downtrend and mom_falling:
            entry = curr["close"]
            stop = curr["HMA"] + (atr_now * 1.0)
            tp1 = entry - (stop - entry) * min_rr
            tp2 = entry - (stop - entry) * (min_rr * 2)
            rr = (entry - tp1) / (stop - entry)
            
            if prev["Sqz_Mom"] > 0 or (prev["close"] > prev["HMA"]):
                signal_obj = Signal(
                    exchange, symbol, timeframe, "BEAR", "Apex Trend Rejection",
                    entry, stop, tp1, tp2, rr,
                    f"Price rejected at Apex Baseline. Sqz Mom: {curr['Sqz_Mom']:.4f}",
                    datetime.now(timezone.utc).isoformat()
                )

    # --- SETUP 2: Squeeze Firing (Volatility Expansion) ---
    # Squeeze was ON, now OFF
    if prev2["Squeeze_On"] and not prev["Squeeze_On"]:
        # Squeeze Fired
        direction = "BULL" if curr["Sqz_Mom"] > 0 else "BEAR"
        
        if direction == "BULL" and bias == "Bull":
             entry = curr["close"]
             stop = curr["low"] - atr_now
             tp1 = entry + (entry - stop) * 2.0
             tp2 = entry + (entry - stop) * 3.0
             signal_obj = Signal(
                exchange, symbol, timeframe, "BULL", "Squeeze Fire (Long)",
                entry, stop, tp1, tp2, 2.0,
                f"Volatility expansion aligned with Apex Trend.",
                datetime.now(timezone.utc).isoformat()
             )
        elif direction == "BEAR" and bias == "Bear":
             entry = curr["close"]
             stop = curr["high"] + atr_now
             tp1 = entry - (stop - entry) * 2.0
             tp2 = entry - (stop - entry) * 3.0
             signal_obj = Signal(
                exchange, symbol, timeframe, "BEAR", "Squeeze Fire (Short)",
                entry, stop, tp1, tp2, 2.0,
                f"Volatility expansion aligned with Apex Trend.",
                datetime.now(timezone.utc).isoformat()
             )

    return signal_obj

# =========================
# 💾 DB LAYER
# =========================
def db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS signals 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                     timestamp TEXT, exchange TEXT, symbol TEXT, timeframe TEXT, 
                     bias TEXT, setup TEXT, entry REAL, stop REAL, tp1 REAL, tp2 REAL, 
                     sent INTEGER DEFAULT 0)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS cooldowns 
                    (key TEXT PRIMARY KEY, last_ts TEXT)''')
    conn.commit()
    return conn

def is_cooldown(conn, key, cooldown_mins):
    cur = conn.cursor()
    cur.execute("SELECT last_ts FROM cooldowns WHERE key=?", (key,))
    row = cur.fetchone()
    if row:
        last_dt = datetime.fromisoformat(row[0])
        diff = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
        if diff < cooldown_mins:
            return True
    return False

def set_cooldown(conn, key):
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute("INSERT OR REPLACE INTO cooldowns (key, last_ts) VALUES (?, ?)", (key, ts))
    conn.commit()

def save_signal(conn, sig: Signal):
    conn.execute("INSERT INTO signals (timestamp, exchange, symbol, timeframe, bias, setup, entry, stop, tp1, tp2) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (sig.timestamp, sig.exchange, sig.symbol, sig.timeframe, sig.bias, sig.setup, sig.entry, sig.stop, sig.tp1, sig.tp2))
    conn.commit()

def get_recent_signals(conn, limit=20):
    return pd.read_sql(f"SELECT * FROM signals ORDER BY id DESC LIMIT {limit}", conn)

# =========================
# 📡 TELEGRAM BROADCASTER
# =========================
def send_telegram(token, chat_id, sig: Signal):
    emoji = "🐂" if sig.bias == "BULL" else "🐻"
    color_emoji = "🟢" if sig.bias == "BULL" else "🔴"
    
    html = f"""
<b>{color_emoji} GOD MODE SIGNAL: {sig.symbol}</b>
    
<b>Bias:</b> {sig.bias} {emoji}
<b>Setup:</b> {sig.setup}
<b>Timeframe:</b> {sig.timeframe}
    
🎯 <b>Entry:</b> <code>{sig.entry:.4f}</code>
🛑 <b>Stop:</b> <code>{sig.stop:.4f}</code>
💰 <b>Target 1:</b> <code>{sig.tp1:.4f}</code>
💰 <b>Target 2:</b> <code>{sig.tp2:.4f}</code>
    
<i>RR: {sig.rr:.2f}R | {sig.notes}</i>
    """
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": html, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Telegram Error: {e}")

# =========================
# 🖥️ MAIN APP
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="👁️")
    inject_css()
    
    st.markdown(f"<h1 class='glow'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.caption("Fusing CCXT Execution Speed with God Mode Math (Apex + Squeeze)")
    
    # Init DB
    conn = db_connect()
    
    # Sidebar
    st.sidebar.header("🕹️ Command Deck")
    
    # Secrets
    tg_token = st.sidebar.text_input("Telegram Token", type="password", value=st.secrets.get("TELEGRAM_BOT_TOKEN", ""))
    tg_chat = st.sidebar.text_input("Chat ID", value=st.secrets.get("TELEGRAM_CHAT_ID", ""))
    
    # Scanner Settings
    exchange_id = st.sidebar.selectbox("Exchange", ["binance", "mexc", "bybit", "kraken"], index=1)
    timeframe = st.sidebar.selectbox("Timeframe", DEFAULT_TIMEFRAMES, index=1)
    symbols_raw = st.sidebar.text_area("Symbols (comma sep)", ", ".join(DEFAULT_SYMBOLS))
    symbols = [s.strip() for s in symbols_raw.split(",")]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧮 God Mode Params")
    hma_len = st.sidebar.number_input("Apex HMA Length", 10, 200, DEFAULT_HMA_LEN)
    apex_mult = st.sidebar.number_input("Apex ATR Mult", 0.5, 5.0, DEFAULT_APEX_MULT)
    min_rr = st.sidebar.number_input("Min R:R", 1.0, 5.0, 2.0)
    
    # Scan Action
    if st.button("🚀 INITIATE SCAN SEQUENCE", type="primary", use_container_width=True):
        if not CCXT_OK:
            st.error("❌ CCXT Library Missing. Install via `pip install ccxt`")
            return
            
        st.write(f"📡 Connecting to {exchange_id.upper()} node...")
        progress = st.progress(0)
        logs = st.empty()
        
        try:
            ex = getattr(ccxt, exchange_id)()
            found_signals = []
            
            for i, sym in enumerate(symbols):
                progress.progress((i + 1) / len(symbols))
                logs.text(f"Scanning {sym}...")
                
                # Check Cooldown
                cd_key = f"{exchange_id}_{sym}_{timeframe}"
                if is_cooldown(conn, cd_key, DEFAULT_COOLDOWN):
                    continue
                
                try:
                    # Fetch Data
                    ohlcv = ex.fetch_ohlcv(sym, timeframe, limit=100)
                    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Generate Signal
                    sig = generate_god_mode_signal(df, exchange_id, sym, timeframe, hma_len, apex_mult, min_rr)
                    
                    if sig:
                        found_signals.append(sig)
                        save_signal(conn, sig)
                        set_cooldown(conn, cd_key)
                        
                        # Broadcast
                        if tg_token and tg_chat:
                            send_telegram(tg_token, tg_chat, sig)
                            
                except Exception as e:
                    st.error(f"Error scanning {sym}: {e}")
                    
            if found_signals:
                st.success(f"⚠️ SCAN COMPLETE. {len(found_signals)} THREATS DETECTED.")
                for sig in found_signals:
                    css_class = "bull" if sig.bias == "BULL" else "bear"
                    st.markdown(f"""
                    <div class="signal-card {css_class}">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span class="badge bg-{css_class.lower()}">{sig.bias}</span>
                                <span class="glow" style="font-size:1.2em; font-weight:bold;">{sig.symbol}</span>
                            </div>
                            <span style="color:#8b949e;">{sig.timestamp.split('T')[1][:8]}</span>
                        </div>
                        <div style="margin-top:10px; font-size:0.9em; color:#e6edf3;">
                            ⚡ <b>Setup:</b> {sig.setup}<br>
                            🎯 <b>Entry:</b> {sig.entry}<br>
                            💰 <b>Targets:</b> {sig.tp1} | {sig.tp2}<br>
                            🛑 <b>Stop:</b> {sig.stop}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("✅ Sector Clear. No signals triggered.")
                
        except Exception as e:
            st.error(f"Critical Engine Failure: {e}")

    # --- HISTORY TAB ---
    st.markdown("---")
    st.subheader("📜 Signal Registry")
    
    hist_df = get_recent_signals(conn)
    if not hist_df.empty:
        st.dataframe(
            hist_df, 
            column_config={
                "timestamp": "Time (UTC)",
                "rr": st.column_config.NumberColumn("R:R", format="%.2fR"),
                "entry": st.column_config.NumberColumn("Entry", format="%.4f"),
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption("Registry Empty.")

    # --- CHART VISUALIZER ---
    if PLOTLY_OK and not hist_df.empty:
        with st.expander("📊 Visual Forensics"):
            sel_sig = st.selectbox("Select Signal", hist_df['id'].tolist(), format_func=lambda x: f"Signal #{x}")
            if st.button("Reconstruct Chart"):
                sig_row = hist_df[hist_df['id'] == sel_sig].iloc[0]
                ex = getattr(ccxt, sig_row['exchange'])()
                ohlcv = ex.fetch_ohlcv(sig_row['symbol'], sig_row['timeframe'], limit=100)
                df_viz = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                
                # Re-calc Math
                df_viz = calculate_apex_trend(df_viz, hma_len, apex_mult)
                df_viz["dt"] = pd.to_datetime(df_viz["time"], unit='ms')
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_viz["dt"], open=df_viz["open"], high=df_viz["high"], low=df_viz["low"], close=df_viz["close"], name="Price"))
                fig.add_trace(go.Scatter(x=df_viz["dt"], y=df_viz["HMA"], line=dict(color='yellow', width=2), name="Apex Base"))
                fig.add_trace(go.Scatter(x=df_viz["dt"], y=df_viz["Apex_Upper"], line=dict(color='rgba(0,255,0,0.5)', width=1, dash='dot'), name="Upper"))
                fig.add_trace(go.Scatter(x=df_viz["dt"], y=df_viz["Apex_Lower"], line=dict(color='rgba(255,0,0,0.5)', width=1, dash='dot'), name="Lower"))
                
                fig.update_layout(template="plotly_dark", height=500, title=f"{sig_row['symbol']} God Mode Reconstruction")
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
