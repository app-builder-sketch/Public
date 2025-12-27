"""
Signals-MOBILE 
Version 18.2: Enhanced Logic + AI Analysis Integration + HTML Fix
"""

import time
import math
import sqlite3
import random
import json
import re
from typing import Dict, Optional, List
from contextlib import contextmanager

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime, timezone

# =============================================================================
# PAGE CONFIG (Mobile Friendly)
# =============================================================================
st.set_page_config(
    page_title="TITAN-SIGNALS",
    layout="wide",  # Wide layout allows full width usage on mobile
    page_icon="📱",
    initial_sidebar_state="collapsed"  # Collapsed by default for mobile screen space
)

# =============================================================================
# CUSTOM CSS (MOBILE OPTIMIZED + AI STYLES)
# =============================================================================
st.markdown("""
<style>
    .main { background-color: #0b0c10; }

    /* Mobile-First Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(31, 40, 51, 0.9);
        border: 1px solid #45a29e;
        padding: 15px; /* Larger padding for touch */
        border-radius: 12px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    /* Larger Text for Mobile Readability */
    div[data-testid="metric-container"] label {
        font-size: 14px !important;
        color: #c5c6c7 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 24px !important;
        color: #66fcf1 !important;
    }

    h1, h2, h3 {
        font-family: 'Roboto Mono', monospace;
        color: #c5c6c7;
        word-wrap: break-word; /* Prevent overflow */
    }

    /* Touch-Friendly Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        border: 1px solid #45a29e;
        color: #66fcf1;
        font-weight: bold;
        height: 3em; /* Taller buttons for thumbs */
        font-size: 16px !important;
        border-radius: 8px;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .stButton > button:hover {
        background: #45a29e;
        color: #0b0c10;
    }

    /* Report Card Styling for Mobile */
    .report-card {
        background-color: #1f2833;
        border-left: 5px solid #45a29e;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .report-header {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;
        border-bottom: 1px solid #45a29e;
        padding-bottom: 5px;
    }
    .report-item {
        margin-bottom: 8px;
        font-size: 14px;
        color: #c5c6c7;
    }
    .highlight { color: #66fcf1; font-weight: bold; }

    /* NEW: AI Analysis Card Style */
    .ai-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #66fcf1;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 0 15px rgba(102, 252, 241, 0.1);
    }
    .ai-title {
        color: #66fcf1;
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .ai-text {
        color: #e0e0e0;
        font-size: 13px;
        line-height: 1.5;
        font-family: 'Roboto Mono', monospace;
    }
    .ai-tag {
        display: inline-block;
        background: rgba(102, 252, 241, 0.15);
        color: #66fcf1;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11px;
        margin-right: 5px;
    }

</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
BINANCE_API_BASE = "https://api.binance.us/api/v3"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# A strong "popular" list to speed selection (still uses original manual Asset input too)
POPULAR_BASES = [
    "BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "LINK", "AVAX", "DOT",
    "MATIC", "LTC", "BCH", "ATOM", "XLM", "ETC", "AAVE", "UNI", "SHIB", "TRX",
    "FIL", "NEAR", "ICP", "ARB", "OP", "SUI", "APT", "INJ", "TIA", "RNDR"
]

# =============================================================================
# TICKER UNIVERSE (AUTO-LOAD FROM BINANCE US)
# =============================================================================
@st.cache_data(ttl=3600)
def get_binanceus_usdt_bases() -> List[str]:
    """
    Pull all Binance US symbols and return unique base assets for USDT pairs.
    This gives you 'lots more tickers' without hardcoding, and stays current.
    """
    try:
        r = requests.get(f"{BINANCE_API_BASE}/exchangeInfo", headers=HEADERS, timeout=6)
        if r.status_code != 200:
            return []
        js = r.json()
        bases = set()
        for s in js.get("symbols", []):
            # Keep spot-trading symbols only (Binance US)
            if s.get("status") != "TRADING":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
            base = s.get("baseAsset")
            if base:
                bases.add(base.upper())
        return sorted(bases)
    except Exception:
        return []

# =============================================================================
# LIVE TICKER WIDGET (Expanded)
# =============================================================================
def render_ticker_tape(selected_symbol: str):
    base = selected_symbol.replace("USDT", "")
    tape_bases = []
    # Start with original three (no omission)
    tape_bases.extend(["BTC", "ETH", "SOL"])
    # Add a larger popular set
    tape_bases.extend([
        "XRP", "BNB", "ADA", "DOGE", "LINK", "AVAX", "DOT", "MATIC", "LTC", "BCH",
        "ATOM", "XLM", "ETC", "AAVE", "UNI", "SHIB", "TRX", "FIL", "NEAR", "ICP"
    ])
    # Add the currently selected base
    if base and base not in tape_bases:
        tape_bases.insert(0, base)

    # De-dupe while preserving order
    seen = set()
    tape_bases = [x for x in tape_bases if not (x in seen or seen.add(x))]

    symbols_json = json.dumps(
        [{"proName": f"BINANCE:{b}USDT", "title": b} for b in tape_bases],
        separators=(",", ":")
    )

    components.html(
        f"""
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
          {{
            "symbols": {symbols_json},
            "showSymbolLogo": true,
            "colorTheme": "dark",
            "isTransparent": true,
            "displayMode": "adaptive",
            "locale": "en"
          }}
          </script>
        </div>
        """,
        height=50
    )

# HEADER with JS Clock (Stacked for Mobile)
st.title("💠 TITAN-SIGNALS")
st.caption("v18.2 | AI ENHANCED ENGINE")

# Mobile Clock
components.html(
    """
    <div id="live_clock"></div>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@700&display=swap');
        body { margin: 0; background-color: transparent; text-align: center; }
        #live_clock {
            font-family: 'Roboto Mono', monospace;
            font-size: 20px;
            color: #39ff14;
            text-shadow: 0 0 10px rgba(57, 255, 20, 0.8);
            font-weight: 800;
            padding: 5px;
        }
    </style>
    <script>
    function updateTime() {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-GB', { timeZone: 'UTC' });
        document.getElementById('live_clock').innerHTML = 'UTC: ' + timeString;
    }
    setInterval(updateTime, 1000);
    updateTime();
    </script>
    """,
    height=40
)

# =============================================================================
# SIDEBAR (Settings)
# =============================================================================
with st.sidebar:
    st.header("⚙️ CONTROL")
    if st.button("🔄 REFRESH", use_container_width=True):
        st.rerun()

    st.subheader("📡 FEED")

    # ---- Load ticker universe (Binance US) ----
    bases_all = get_binanceus_usdt_bases()

    if "symbol_input" not in st.session_state:
        st.session_state.symbol_input = "BTC"

    # Quick select controls
    with st.expander("🧬 Ticker Universe (Quick Select)", expanded=True):
        if bases_all:
            list_mode = st.selectbox(
                "List",
                ["Popular", "All Binance US (USDT)"],
                index=0
            )

            if list_mode == "Popular":
                options = [b for b in POPULAR_BASES if b in bases_all] or POPULAR_BASES
            else:
                options = POPULAR_BASES + [b for b in bases_all if b not in POPULAR_BASES]

            quick_base = st.selectbox("Quick Ticker", options, index=(options.index("BTC") if "BTC" in options else 0))
            q1, q2 = st.columns([1, 1])
            with q1:
                if st.button("Use Quick Ticker", use_container_width=True):
                    st.session_state.symbol_input = quick_base
            with q2:
                st.caption(f"{len(bases_all)} tickers loaded")
        else:
            st.warning("Ticker universe unavailable. Manual input still works.")

    # ---- ORIGINAL manual input kept (no omission) ----
    symbol_input = st.text_input("Asset", value=st.session_state.symbol_input)
    st.session_state.symbol_input = symbol_input
    symbol = symbol_input.strip().upper().replace("/", "").replace("-", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    c1, c2 = st.columns(2)
    with c1:
        timeframe = st.selectbox("TF", ["15m", "1h", "4h", "1d"], index=1)
    with c2:
        limit = st.slider("Depth", 100, 500, 200, 50)

    st.markdown("---")
    st.subheader("🧠 LOGIC")
    amplitude = st.number_input("Amp", 2, 200, 10)
    channel_dev = st.number_input("Dev", 0.5, 10.0, 3.0, 0.1)
    hma_len = st.number_input("HMA", 2, 400, 50)
    gann_len = st.number_input("Gann", 1, 50, 3)

    with st.expander("🎯 Targets"):
        tp1_r = st.number_input("TP1 (R)", value=1.5)
        tp2_r = st.number_input("TP2 (R)", value=3.0)
        tp3_r = st.number_input("TP3 (R)", value=5.0)

    st.markdown("---")
    st.subheader("🤖 NOTIFICATIONS")
    tg_token = st.text_input("Bot Token", value=st.secrets.get("TELEGRAM_TOKEN", ""), type="password")
    tg_chat = st.text_input("Chat ID", value=st.secrets.get("TELEGRAM_CHAT_ID", ""))

render_ticker_tape(symbol)

# =============================================================================
# LOGIC ENGINES (ENHANCED)
# =============================================================================
def calculate_hma(series, length):
    half_len = int(length / 2)
    sqrt_len = int(math.sqrt(length))
    wma_f = series.rolling(length).mean()
    wma_h = series.rolling(half_len).mean()
    diff = 2 * wma_h - wma_f
    return diff.rolling(sqrt_len).mean()

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = calculate_ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_stoch_rsi(df, period=14, smooth_k=3, smooth_d=3):
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1).mean()
    avg_loss = loss.ewm(com=period-1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    stoch_k = stoch_rsi.rolling(smooth_k).mean() * 100
    stoch_d = stoch_k.rolling(smooth_d).mean()
    return stoch_k, stoch_d

def calculate_fibonacci(df, lookback=50):
    recent = df.iloc[-lookback:]
    h, l = recent['high'].max(), recent['low'].min()
    d = h - l
    fibs = {
        'fib_382': h - (d * 0.382),
        'fib_500': h - (d * 0.500),
        'fib_618': h - (d * 0.618),
        'high': h, 'low': l
    }
    return fibs

def calculate_fear_greed_index(df):
    try:
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        vol_score = 50 - ((df['log_ret'].rolling(30).std().iloc[-1] - df['log_ret'].rolling(90).std().iloc[-1]) / df['log_ret'].rolling(90).std().iloc[-1]) * 100
        vol_score = max(0, min(100, vol_score))
        rsi = df['rsi'].iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        dist = (df['close'].iloc[-1] - sma_50) / sma_50
        trend_score = 50 + (dist * 1000)
        fg = (vol_score * 0.3) + (rsi * 0.4) + (max(0, min(100, trend_score)) * 0.3)
        return int(fg)
    except:
        return 50

def run_backtest(df, tp1_r):
    trades = []
    signals = df[(df['buy']) | (df['sell'])]
    for idx, row in signals.iterrows():
        future = df.loc[idx+1: idx+20]
        if future.empty:
            continue
        entry = row['close']; stop = row['entry_stop']; tp1 = row['tp1']; is_long = row['is_bull']
        outcome = "PENDING"; pnl = 0
        if is_long:
            if future['high'].max() >= tp1:
                outcome = "WIN"; pnl = abs(entry - stop) * tp1_r
            elif future['low'].min() <= stop:
                outcome = "LOSS"; pnl = -abs(entry - stop)
        else:
            if future['low'].min() <= tp1:
                outcome = "WIN"; pnl = abs(entry - stop) * tp1_r
            elif future['high'].max() >= stop:
                outcome = "LOSS"; pnl = -abs(entry - stop)
        if outcome != "PENDING":
            trades.append({'outcome': outcome, 'pnl': pnl})

    if not trades:
        return 0, 0, 0
    df_res = pd.DataFrame(trades)
    total = len(df_res)
    win_rate = (len(df_res[df_res['outcome'] == 'WIN']) / total) * 100
    net_r = (len(df_res[df_res['outcome'] == 'WIN']) * tp1_r) - len(df_res[df_res['outcome'] == 'LOSS'])
    return total, win_rate, net_r

# --- AI ANALYSIS GENERATOR ---
def generate_ai_narrative(df, last, fibs, fg_index, symbol, tf):
    # Logic to create a text summary
    price = last['close']
    rsi = last['rsi']
    macd_val = last['macd']
    macd_sig = last['macd_signal']
    stoch_k = last['stoch_k']
    stoch_d = last['stoch_d']
    
    trend_state = "NEUTRAL"
    if price > last['ema_50'] > last['ema_200']:
        trend_state = "STRONG BULLISH"
    elif price < last['ema_50'] < last['ema_200']:
        trend_state = "STRONG BEARISH"
    
    mom_state = "MOMENTUM"
    if macd_val > macd_sig:
        mom_state = "BULLISH MOMENTUM"
    else:
        mom_state = "BEARISH MOMENTUM"

    vol_state = "NORMAL"
    if last['rvol'] > 2.0:
        vol_state = "HIGH VOLATILITY BREAKOUT"
    elif last['in_squeeze']:
        vol_state = "COMPRESSION (SQUEEZE)"

    # Construct Narrative
    analysis = (
        f"<span class='ai-tag'>AI-SCAN</span> "
        f"Analyzing {symbol} ({tf}).<br><br>"
        f"<b>Trend Structure:</b> {trend_state}. Price action is respecting EMA layers.<br>"
        f"<b>Momentum:</b> {mom_state} (MACD: {macd_val:.4f}).<br>"
        f"<b>Oscillators:</b> Stoch RSI ({stoch_k:.1f}/{stoch_d:.1f}) "
    )
    
    if stoch_k > 80:
        analysis += "indicates Overbought zone. "
    elif stoch_k < 20:
        analysis += "indicates Oversold zone. "
    else:
        analysis += "is neutral. "

    analysis += (
        f"<br><b>Volume & Volatility:</b> {vol_state} with RVOL at {last['rvol']:.2f}.<br>"
        f"<b>Market Sentiment:</b> Fear/Greed Index at {fg_index}/100."
    )
    
    return analysis

# --- MOBILE OPTIMIZED REPORT GENERATOR (FIXED & ENHANCED) ---
def generate_mobile_report(row, symbol, tf, fibs, fg_index, smart_stop, ai_html):
    is_bull = row['is_bull']
    direction = "LONG 🐂" if is_bull else "SHORT 🐻"

    titan_sig = 1 if row['is_bull'] else -1
    apex_sig = row['apex_trend']
    gann_sig = row['gann_trend']

    score_val = 0
    if titan_sig == apex_sig: score_val += 1
    if titan_sig == gann_sig: score_val += 1

    confidence = "LOW"
    if score_val == 2: confidence = "MAX 🔥"
    elif score_val == 1: confidence = "HIGH"

    vol_desc = "Normal"
    if row['rvol'] > 2.0: vol_desc = "IGNITION 🚀"

    squeeze_txt = "⚠️ SQUEEZE ACTIVE" if row['in_squeeze'] else "⚪ NO SQUEEZE"

    # HTML Card Construction
    report_html = f"""
    <div class="report-card">
        <div class="report-header">💠 SIGNAL: {direction}</div>
        <div class="report-item">Confidence: <span class="highlight">{confidence}</span></div>
        <div class="report-item">Sentiment: <span class="highlight">{fg_index}/100</span></div>
        <div class="report-item">Squeeze: <span class="highlight">{squeeze_txt}</span></div>
    </div>

    <div class="report-card">
        <div class="report-header">🌊 FLOW & VOL</div>
        <div class="report-item">RVOL: <span class="highlight">{row['rvol']:.2f} ({vol_desc})</span></div>
        <div class="report-item">Money Flow: <span class="highlight">{row['money_flow']:.2f}</span></div>
        <div class="report-item">VWAP Relation: <span class="highlight">{'Above' if row['close'] > row['vwap'] else 'Below'}</span></div>
    </div>

    <div class="report-card">
        <div class="report-header">🎯 EXECUTION PLAN</div>
        <div class="report-item">Entry: <span class="highlight">{row['close']:.4f}</span></div>
        <div class="report-item">🛑 SMART STOP: <span class="highlight">{smart_stop:.4f}</span></div>
        <div class="report-item">1️⃣ TP1 (1.5R): <span class="highlight">{row['tp1']:.4f}</span></div>
        <div class="report-item">2️⃣ TP2 (3.0R): <span class="highlight">{row['tp2']:.4f}</span></div>
        <div class="report-item">3️⃣ TP3 (5.0R): <span class="highlight">{row['tp3']:.4f}</span></div>
    </div>
    
    <!-- AI ANALYSIS CARD INTEGRATION -->
    <div class="ai-card">
        <div class="ai-title">🤖 AI TECHNICAL ANALYSIS</div>
        <div class="ai-text">{ai_html}</div>
    </div>
    """
    return report_html

def send_telegram_msg(token, chat, msg):
    if not token or not chat:
        return False
    try:
        # Strip HTML tags for plain text Telegram message to fix "text like this" errors
        clean_msg = re.sub('<[^<]+?>', '', msg) 
        clean_msg = clean_msg.replace('&nbsp;', ' ')
        clean_msg = html.unescape(clean_msg)
        
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": clean_msg, "parse_mode": "Markdown"},
            timeout=5
        )
        return r.status_code == 200
    except:
        return False

import html

@st.cache_data(ttl=5)
def get_klines(symbol_bin, interval, limit):
    try:
        r = requests.get(
            f"{BINANCE_API_BASE}/klines",
            params={"symbol": symbol_bin, "interval": interval, "limit": limit},
            headers=HEADERS,
            timeout=4
        )
        if r.status_code == 200:
            df = pd.DataFrame(r.json(), columns=['t','o','h','l','c','v','T','q','n','V','Q','B'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
            return df[['timestamp','open','high','low','close','volume']]
    except:
        pass
    return pd.DataFrame()

def run_engines(df, amp, dev, hma_l, tp1, tp2, tp3, mf_l, vol_l, gann_l):
    if df.empty:
        return df
    df = df.copy().reset_index(drop=True)

    # --- BASE INDICATORS (Original) ---
    df['tr'] = np.maximum(
        df['high']-df['low'],
        np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1)))
    )
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    df['hma'] = calculate_hma(df['close'], hma_l)

    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vol_tp'] = df['tp'] * df['volume']
    df['vwap'] = df['vol_tp'].cumsum() / df['volume'].cumsum()

    bb_basis = df['close'].rolling(20).mean()
    bb_dev = df['close'].rolling(20).std() * 2.0
    kc_basis = df['close'].rolling(20).mean()
    kc_dev = df['atr'] * 1.5
    df['in_squeeze'] = ((bb_basis - bb_dev) > (kc_basis - kc_dev)) & ((bb_basis + bb_dev) < (kc_basis + kc_dev))

    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain/loss)))
    df['rvol'] = df['volume'] / df['volume'].rolling(vol_l).mean()

    rsi_source = df['rsi'] - 50
    vol_sma = df['volume'].rolling(mf_l).mean()
    df['money_flow'] = (rsi_source * (df['volume'] / vol_sma)).ewm(span=3).mean()

    pc = df['close'].diff()
    ds_pc = pc.ewm(span=25).mean().ewm(span=13).mean()
    ds_abs_pc = abs(pc).ewm(span=25).mean().ewm(span=13).mean()
    df['hyper_wave'] = (100 * (ds_pc / ds_abs_pc)) / 2

    # --- NEW INDICATORS (AI & Trend Confirmation) ---
    df['ema_50'] = calculate_ema(df['close'], 50)
    df['ema_200'] = calculate_ema(df['close'], 200)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    df['stoch_k'], df['stoch_d'] = calculate_stoch_rsi(df)

    # --- TREND LOGIC ---
    df['ll'] = df['low'].rolling(amp).min()
    df['hh'] = df['high'].rolling(amp).max()
    trend = np.zeros(len(df))
    stop = np.full(len(df), np.nan)
    curr_t = 0
    curr_s = np.nan
    for i in range(amp, len(df)):
        c = df.at[i,'close']
        d = df.at[i,'atr']*dev
        if curr_t == 0:
            s = df.at[i,'ll'] + d
            curr_s = max(curr_s, s) if not np.isnan(curr_s) else s
            if c < curr_s:
                curr_t = 1
                curr_s = df.at[i,'hh'] - d
        else:
            s = df.at[i,'hh'] - d
            curr_s = min(curr_s, s) if not np.isnan(curr_s) else s
            if c > curr_s:
                curr_t = 0
                curr_s = df.at[i,'ll'] + d
        trend[i] = curr_t
        stop[i] = curr_s

    df['is_bull'] = trend == 0
    df['entry_stop'] = stop

    # Signals (Enhanced with Stochastic filter)
    stoch_bull = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'] < 80)
    stoch_bear = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'] > 20)
    
    cond_buy = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False)) & (df['rvol']>1.0) & (stoch_bull)
    cond_sell = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True)) & (df['rvol']>1.0) & (stoch_bear)
    
    df['buy'] = cond_buy
    df['sell'] = cond_sell

    # Targets
    df['sig_id'] = (df['buy']|df['sell']).cumsum()
    df['entry'] = df.groupby('sig_id')['close'].ffill()
    df['stop_val'] = df.groupby('sig_id')['entry_stop'].ffill()
    risk = abs(df['entry'] - df['stop_val'])
    df['tp1'] = np.where(df['is_bull'], df['entry']+(risk*tp1), df['entry']-(risk*tp1))
    df['tp2'] = np.where(df['is_bull'], df['entry']+(risk*tp2), df['entry']-(risk*tp2))
    df['tp3'] = np.where(df['is_bull'], df['entry']+(risk*tp3), df['entry']-(risk*tp3))

    # Apex & Gann
    apex_base = calculate_hma(df['close'], 55)
    apex_atr = df['atr'] * 1.5
    df['apex_upper'] = apex_base + apex_atr
    df['apex_lower'] = apex_base - apex_atr
    apex_t = np.zeros(len(df))
    for i in range(1, len(df)):
        if df.at[i, 'close'] > df.at[i, 'apex_upper']:
            apex_t[i] = 1
        elif df.at[i, 'close'] < df.at[i, 'apex_lower']:
            apex_t[i] = -1
        else:
            apex_t[i] = apex_t[i-1]
    df['apex_trend'] = apex_t

    sma_h = df['high'].rolling(gann_l).mean()
    sma_l = df['low'].rolling(gann_l).mean()
    g_trend = np.full(len(df), np.nan)
    g_act = np.full(len(df), np.nan)
    curr_g_t = 1
    curr_g_a = sma_l.iloc[gann_l] if len(sma_l) > gann_l else np.nan
    for i in range(gann_l, len(df)):
        c = df.at[i,'close']
        h_ma = sma_h.iloc[i]
        l_ma = sma_l.iloc[i]
        prev_a = g_act[i-1] if (i>0 and not np.isnan(g_act[i-1])) else curr_g_a
        if curr_g_t == 1:
            if c < prev_a:
                curr_g_t = -1
                curr_g_a = h_ma
            else:
                curr_g_a = l_ma
        else:
            if c > prev_a:
                curr_g_t = 1
                curr_g_a = l_ma
            else:
                curr_g_a = h_ma
        g_trend[i] = curr_g_t
        g_act[i] = curr_g_a
    df['gann_trend'] = g_trend
    df['gann_act'] = g_act

    return df

# =============================================================================
# APP MAIN
# =============================================================================
df = get_klines(symbol, timeframe, limit)

if not df.empty:
    df = df.dropna(subset=['close'])
    df = run_engines(df, int(amplitude), channel_dev, int(hma_len), tp1_r, tp2_r, tp3_r, 14, 20, int(gann_len))

    last = df.iloc[-1]
    fibs = calculate_fibonacci(df)
    fg_index = calculate_fear_greed_index(df)

    if last['is_bull']:
        smart_stop = min(last['entry_stop'], fibs['fib_618'] * 0.9995)
    else:
        smart_stop = max(last['entry_stop'], fibs['fib_618'] * 1.0005)
        
    # --- Generate AI Narrative ---
    ai_narrative = generate_ai_narrative(df, last, fibs, fg_index, symbol, timeframe)

    # ----------------------------------------------------
    # MOBILE METRICS (2x2 Grid)
    # ----------------------------------------------------
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        tv_symbol = f"BINANCE:{symbol}"
        components.html(f"""
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js" async>
          {{ "symbol": "{tv_symbol}", "width": "100%", "colorTheme": "dark", "isTransparent": true, "locale": "en" }}
          </script>
        </div>
        """, height=120)
    with c_m2:
        st.metric("TREND", "BULL 🐂" if last['gann_trend'] == 1 else "BEAR 🐻")

    c_m3, c_m4 = st.columns(2)
    with c_m3:
        st.metric("STOP", f"{smart_stop:.2f}")
    with c_m4:
        st.metric("TP3", f"{last['tp3']:.2f}")

    # ----------------------------------------------------
    # REPORT & ACTIONS (Stacked with AI)
    # ----------------------------------------------------
    report_html = generate_mobile_report(last, symbol, timeframe, fibs, fg_index, smart_stop, ai_narrative)

    # Display the HTML Report Card directly
    st.markdown(report_html, unsafe_allow_html=True)

    st.markdown("### ⚡ ACTION")

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("🔥 ALERT TG", use_container_width=True):
            msg = f"TITAN SIGNAL: {symbol} | {'LONG' if last['is_bull'] else 'SHORT'} | EP: {last['close']}"
            if send_telegram_msg(tg_token, tg_chat, msg):
                st.success("SENT")
            else:
                st.error("FAIL")

    with b_col2:
        if st.button("📝 REPORT TG", use_container_width=True):
            # Improved HTML stripping
            txt_rep = re.sub(r'<[^>]+>', '', report_html)
            if send_telegram_msg(tg_token, tg_chat, f"REPORT: {symbol}\n{txt_rep}"):
                st.success("SENT")
            else:
                st.error("FAIL")

    # Backtest Mini-Stat
    b_total, b_win, b_net = run_backtest(df, tp1_r)
    st.caption(f"📊 Live Stats: {b_win:.1f}% Win Rate | {b_net:.1f}R Net ({b_total} Trades)")

    # ----------------------------------------------------
    # MAIN CHART
    # ----------------------------------------------------
    fig = go.Figure()
    fig.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', name='HMA', line=dict(color='#66fcf1', width=1)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vwap'], mode='lines', name='VWAP', line=dict(color='#9933ff', width=2)))

    buys = df[df['buy']]
    sells = df[df['sell']]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low'], mode='markers',
                                 marker=dict(symbol='triangle-up', size=10, color='#00ff00'), name='BUY'))
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high'], mode='markers',
                                 marker=dict(symbol='triangle-down', size=10, color='#ff0000'), name='SELL'))

    fig.update_layout(height=400, template='plotly_dark', margin=dict(l=0, r=0, t=20, b=20),
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # INDICATORS (Tabs)
    # ----------------------------------------------------
    t1, t2, t3, t4 = st.tabs(["📊 GANN", "🌊 FLOW", "🧠 SENT", "🤖 AI DEEP"])

    with t1:
        f1 = go.Figure()
        f1.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])
        df_g = df.dropna(subset=['gann_act'])
        f1.add_trace(go.Scatter(
            x=df_g['timestamp'],
            y=df_g['gann_act'],
            mode='markers',
            marker=dict(color=np.where(df_g['gann_trend'] == 1, '#00ff00', '#ff0000'), size=3)
        ))
        f1.update_layout(height=300, template='plotly_dark', margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(f1, use_container_width=True)

    with t2:
        f2 = go.Figure()
        cols = ['#00e676' if x > 0 else '#ff1744' for x in df['money_flow']]
        f2.add_trace(go.Bar(x=df['timestamp'], y=df['money_flow'], marker_color=cols))
        f2.update_layout(height=300, template='plotly_dark', margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(f2, use_container_width=True)

    with t3:
        f3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fg_index,
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [0, 25], 'color': '#ff1744'},
                    {'range': [75, 100], 'color': '#00b0ff'}
                ]
            }
        ))
        f3.update_layout(height=250, template='plotly_dark', margin=dict(l=20, r=20, t=30, b=0))
        st.plotly_chart(f3, use_container_width=True)

    with t4:
        # Detailed AI View
        st.markdown(f"<div class='ai-card'><div class='ai-title'>🤖 DIVE DEEP</div><div class='ai-text'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("RSI", f"{last['rsi']:.2f}")
        c2.metric("MACD Hist", f"{last['macd_hist']:.4f}")
        
        st.markdown(f"<b>EMA Trend Alignment:</b> Price { '>' if last['close'] > last['ema_50'] else '<'} EMA50 | EMA50 { '>' if last['ema_50'] > last['ema_200'] else '<'} EMA200", unsafe_allow_html=True)
        st.markdown(f"<b>Stochastic RSI:</b> K={last['stoch_k']:.2f}, D={last['stoch_d']:.2f}", unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

else:
    st.error("No data returned. Check ticker, timeframe, or Binance US availability.")
