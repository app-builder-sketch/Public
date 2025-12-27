"""
Signals-MOBILE 
Version 18.2: AI-Powered Analysis + Enhanced Signal Validation
"""

import time
import math
import sqlite3
import random
import json
from typing import Dict, Optional, List, Tuple
from contextlib import contextmanager

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components
from datetime import datetime, timezone

# =============================================================================
# PAGE CONFIG (Mobile Friendly)
# =============================================================================
st.set_page_config(
    page_title="TITAN-SIGNALS",
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM CSS (MOBILE OPTIMIZED)
# =============================================================================
st.markdown("""
<style>
    .main { background-color: #0b0c10; }

    /* Mobile-First Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(31, 40, 51, 0.9);
        border: 1px solid #45a29e;
        padding: 15px;
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
        word-wrap: break-word;
    }

    /* Touch-Friendly Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        border: 1px solid #45a29e;
        color: #66fcf1;
        font-weight: bold;
        height: 3em;
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
    
    /* AI Analysis Cards */
    .ai-card {
        background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%);
        border: 1px solid #45a29e;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(69, 162, 158, 0.2);
    }
    .ai-score {
        font-size: 32px;
        font-weight: bold;
        color: #66fcf1;
        text-align: center;
        padding: 10px;
    }
    .ai-recommendation {
        background: rgba(69, 162, 158, 0.1);
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
        border-left: 4px solid #45a29e;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
BINANCE_API_BASE = "https://api.binance.us/api/v3"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

POPULAR_BASES = [
    "BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "LINK", "AVAX", "DOT",
    "MATIC", "LTC", "BCH", "ATOM", "XLM", "ETC", "AAVE", "UNI", "SHIB", "TRX",
    "FIL", "NEAR", "ICP", "ARB", "OP", "SUI", "APT", "INJ", "TIA", "RNDR"
]

# AI Asset Knowledge Base
ASSET_PROFILES = {
    "BTC": {"type": "Macro Asset", "vol_regime": "Low", "session": "Global", "correlation": "Risk-On"},
    "ETH": {"type": "Smart Contract Leader", "vol_regime": "Medium", "session": "US/EU", "correlation": "BTC Beta"},
    "SOL": {"type": "High-Performance Chain", "vol_regime": "High", "session": "US", "correlation": "ETH Beta"},
    "XRP": {"type": "Cross-Border Payments", "vol_regime": "Medium", "session": "EU/ASIA", "correlation": "Uncorrelated"}
}

# =============================================================================
# AI ANALYSIS ENGINE
# =============================================================================
def analyze_asset_and_timeframe(symbol: str, timeframe: str, df: pd.DataFrame) -> Dict:
    """
    AI-driven analysis of asset characteristics and timeframe suitability
    """
    base = symbol.replace("USDT", "")
    profile = ASSET_PROFILES.get(base, {
        "type": "Altcoin", 
        "vol_regime": "High", 
        "session": "US", 
        "correlation": "High Beta"
    })
    
    # Timeframe Suitability Score
    tf_scores = {
        "15m": {"score": 70, "note": "Scalping & Intraday"},
        "1h": {"score": 85, "note": "Day Trading & Swing Entry"},
        "4h": {"score": 90, "note": "Swing Trading"},
        "1d": {"score": 80, "note": "Position Trading"}
    }
    
    tf_data = tf_scores.get(timeframe, {"score": 50, "note": "Uncommon TF"})
    
    # Calculate current market regime
    if not df.empty:
        recent_vol = df['close'].pct_change().rolling(20).std().iloc[-1] * 100
        avg_vol = df['close'].pct_change().rolling(60).std().iloc[-1] * 100
        
        if recent_vol > avg_vol * 1.5:
            regime = "High Volatility"
            regime_color = "#ff1744"
        elif recent_vol < avg_vol * 0.7:
            regime = "Low Volatility"
            regime_color = "#00e676"
        else:
            regime = "Normal Volatility"
            regime_color = "#ffd740"
    else:
        regime = "Unknown"
        regime_color = "#9e9e9e"
    
    # Generate recommendation
    if tf_data["score"] >= 85:
        rec = "OPTIMAL TIMEFRAME"
        rec_color = "#00e676"
    elif tf_data["score"] >= 70:
        rec = "SUITABLE"
        rec_color = "#ffd740"
    else:
        rec = "SUBOPTIMAL - Consider 1h or 4h"
        rec_color = "#ff1744"
    
    return {
        "asset_profile": profile,
        "timeframe_score": tf_data["score"],
        "timeframe_note": tf_data["note"],
        "market_regime": regime,
        "regime_color": regime_color,
        "recommendation": rec,
        "rec_color": rec_color,
        "current_vol": f"{recent_vol:.2f}%" if not df.empty else "N/A"
    }

# =============================================================================
# TICKER UNIVERSE (AUTO-LOAD FROM BINANCE US)
# =============================================================================
@st.cache_data(ttl=3600)
def get_binanceus_usdt_bases() -> List[str]:
    """
    Pull all Binance US symbols and return unique base assets for USDT pairs.
    """
    try:
        r = requests.get(f"{BINANCE_API_BASE}/exchangeInfo", headers=HEADERS, timeout=6)
        if r.status_code != 200:
            return []
        js = r.json()
        bases = set()
        for s in js.get("symbols", []):
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
# LIVE TICKER WIDGET (Fixed - Removed trailing spaces)
# =============================================================================
def render_ticker_tape(selected_symbol: str):
    # Ensure selected_symbol is in proName format
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

    # FIX: Properly escape JSON for JavaScript
    symbols_json = json.dumps(
        [{"proName": f"BINANCE:{b}USDT", "title": b} for b in tape_bases],
        separators=(",", ":")
    )

    # FIX: Removed trailing space in URL and fixed JSON injection
    components.html(
        f"""
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
          {symbols_json}
          </script>
        </div>
        """,
        height=50
    )

# HEADER with JS Clock (Stacked for Mobile)
st.title("💠 TITAN-SIGNALS")
st.caption("v18.2 | AI TRADING ENGINE | Enhanced Signal Validation")

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

    # Load ticker universe
    bases_all = get_binanceus_usdt_bases()

    # Session state for original manual asset input
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

    # Original manual input kept
    symbol_input = st.text_input("Asset", value=st.session_state.symbol_input)
    st.session_state.symbol_input = symbol_input
    symbol = symbol_input.strip().upper().replace("/", "").replace("-", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    # Use cols for compact settings
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

# Render expanded ticker tape AFTER we know the chosen symbol
render_ticker_tape(symbol)

# =============================================================================
# LOGIC ENGINES
# =============================================================================
def calculate_hma(series, length):
    half_len = int(length / 2)
    sqrt_len = int(math.sqrt(length))
    wma_f = series.rolling(length).mean()
    wma_h = series.rolling(half_len).mean()
    diff = 2 * wma_h - wma_f
    return diff.rolling(sqrt_len).mean()

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
        return 0, 0, 0, pd.DataFrame()
    df_res = pd.DataFrame(trades)
    total = len(df_res)
    win_rate = (len(df_res[df_res['outcome'] == 'WIN']) / total) * 100
    net_r = (len(df_res[df_res['outcome'] == 'WIN']) * tp1_r) - len(df_res[df_res['outcome'] == 'LOSS'])
    avg_trade = df_res['pnl'].mean()
    return total, win_rate, net_r, df_res

# --- MOBILE OPTIMIZED REPORT GENERATOR ---
def generate_mobile_report(row, symbol, tf, fibs, fg_index, smart_stop, ai_analysis: Dict):
    is_bull = row['is_bull']
    direction = "LONG 🐂" if is_bull else "SHORT 🐻"

    # Enhanced Logic Score (3-Layer Confirmation)
    titan_sig = 1 if row['is_bull'] else -1
    apex_sig = row['apex_trend']
    gann_sig = row['gann_trend']
    momentum_sig = 1 if row['money_flow'] > 0 else -1
    volume_sig = 1 if row['rvol'] > 1.5 else 0

    score_val = 0
    if titan_sig == apex_sig: score_val += 1
    if titan_sig == gann_sig: score_val += 1
    if titan_sig == momentum_sig: score_val += 1
    if volume_sig == 1: score_val += 1

    confidence = "LOW ⚠️"
    if score_val >= 3: confidence = "MAX 🔥🔥🔥"
    elif score_val >= 2: confidence = "HIGH 🔥"
    elif score_val >= 1: confidence = "MEDIUM ⚡"

    vol_desc = "Normal"
    if row['rvol'] > 2.0: vol_desc = "IGNITION 🚀🚀🚀"
    elif row['rvol'] > 1.5: vol_desc = "Above Avg 🚀"

    squeeze_txt = "⚠️ SQUEEZE ACTIVE" if row['in_squeeze'] else "✅ NO SQUEEZE"

    # HTML Card Construction
    report_html = f"""
    <div class="report-card">
        <div class="report-header">💠 SIGNAL: {direction}</div>
        <div class="report-item">Confidence: <span class="highlight">{confidence}</span></div>
        <div class="report-item">Layers: <span class="highlight">{score_val}/4 Confirmed</span></div>
        <div class="report-item">Squeeze: <span class="highlight">{squeeze_txt}</span></div>
    </div>

    <div class="report-card">
        <div class="report-header">🌊 FLOW & VOL</div>
        <div class="report-item">RVOL: <span class="highlight">{row['rvol']:.2f} ({vol_desc})</span></div>
        <div class="report-item">Money Flow: <span class="highlight">{row['money_flow']:.2f}</span></div>
        <div class="report-item">VWAP: <span class="highlight">{'Above' if row['close'] > row['vwap'] else 'Below'}</span></div>
    </div>

    <div class="report-card">
        <div class="report-header">🎯 EXECUTION PLAN</div>
        <div class="report-item">Entry: <span class="highlight">{row['close']:.4f}</span></div>
        <div class="report-item">🛑 SMART STOP: <span class="highlight">{smart_stop:.4f}</span></div>
        <div class="report-item">1️⃣ TP1 ({tp1_r}R): <span class="highlight">{row['tp1']:.4f}</span></div>
        <div class="report-item">2️⃣ TP2 ({tp2_r}R): <span class="highlight">{row['tp2']:.4f}</span></div>
        <div class="report-item">3️⃣ TP3 ({tp3_r}R): <span class="highlight">{row['tp3']:.4f}</span></div>
    </div>
    """
    return report_html

def send_telegram_msg(token, chat, msg):
    if not token or not chat:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"},
            timeout=5
        )
        return r.status_code == 200
    except:
        return False

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

    # Core Indicators (Preserved)
    df['tr'] = np.maximum(
        df['high']-df['low'],
        np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1)))
    )
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    df['hma'] = calculate_hma(df['close'], hma_l)

    # VWAP
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vol_tp'] = df['tp'] * df['volume']
    df['vwap'] = df['vol_tp'].cumsum() / df['volume'].cumsum()

    # Squeeze (TMM)
    bb_basis = df['close'].rolling(20).mean()
    bb_dev = df['close'].rolling(20).std() * 2.0
    kc_basis = df['close'].rolling(20).mean()
    kc_dev = df['atr'] * 1.5
    df['in_squeeze'] = ((bb_basis - bb_dev) > (kc_basis - kc_dev)) & ((bb_basis + bb_dev) < (kc_basis + kc_dev))

    # Momentum
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain/loss)))
    df['rvol'] = df['volume'] / df['volume'].rolling(vol_l).mean()

    # Money Flow (Enhanced)
    rsi_source = df['rsi'] - 50
    vol_sma = df['volume'].rolling(mf_l).mean()
    df['money_flow'] = (rsi_source * (df['volume'] / vol_sma)).ewm(span=3).mean()

    # Hyper Wave (Preserved)
    pc = df['close'].diff()
    ds_pc = pc.ewm(span=25).mean().ewm(span=13).mean()
    ds_abs_pc = abs(pc).ewm(span=25).mean().ewm(span=13).mean()
    df['hyper_wave'] = (100 * (ds_pc / ds_abs_pc)) / 2

    # Titan Trend (Preserved)
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

    # Enhanced Signals with Volume Filter
    cond_buy = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False)) & (df['rvol']>1.2) & (df['money_flow']>0)
    cond_sell = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True)) & (df['rvol']>1.2) & (df['money_flow']<0)
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

    # Apex & Gann (Preserved)
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

    # AI Analysis
    ai_analysis = analyze_asset_and_timeframe(symbol, timeframe, df)
    
    # ----------------------------------------------------
    # AI ANALYSIS CARD (New)
    # ----------------------------------------------------
    st.markdown("### 🤖 AI MARKET ANALYSIS")
    ai_col1, ai_col2 = st.columns([1, 2])
    with ai_col1:
        st.markdown(f"""
        <div class="ai-card">
            <div style="text-align:center;">
                <div style="font-size:14px; color:#c5c6c7;">Suitability Score</div>
                <div class="ai-score">{ai_analysis['timeframe_score']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with ai_col2:
        st.markdown(f"""
        <div class="ai-card">
            <div class="report-item"><strong>Asset Type:</strong> {ai_analysis['asset_profile']['type']}</div>
            <div class="report-item"><strong>Vol Regime:</strong> <span style="color:{ai_analysis['regime_color']};">{ai_analysis['market_regime']}</span></div>
            <div class="report-item"><strong>Current Vol:</strong> {ai_analysis['current_vol']}</div>
            <div class="report-item"><strong>Best Session:</strong> {ai_analysis['asset_profile']['session']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="ai-card ai-recommendation" style="border-left-color: {ai_analysis['rec_color']};">
        <strong>Recommendation:</strong> <span style="color:{ai_analysis['rec_color']};">{ai_analysis['recommendation']}</span><br>
        <small>{ai_analysis['timeframe_note']}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # MOBILE METRICS (2x2 Grid)
    # ----------------------------------------------------
    # Row 1: Price Widget + Trend
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        # FIX: Removed trailing space in URL
        tv_symbol = f"BINANCE:{symbol}"
        components.html(f"""
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js" async>
          {json.dumps({"symbol": tv_symbol, "width": "100%", "colorTheme": "dark", "isTransparent": true, "locale": "en"})}
          </script>
        </div>
        """, height=120)
    with c_m2:
        st.metric("TREND", "BULL 🐂" if last['gann_trend'] == 1 else "BEAR 🐻")

    # Row 2: Stops & Targets
    c_m3, c_m4 = st.columns(2)
    with c_m3:
        st.metric("STOP", f"{smart_stop:.2f}")
    with c_m4:
        st.metric("TP3", f"{last['tp3']:.2f}")

    # ----------------------------------------------------
    # REPORT & ACTIONS (Stacked for Mobile)
    # ----------------------------------------------------
    report_html = generate_mobile_report(last, symbol, timeframe, fibs, fg_index, smart_stop, ai_analysis)

    # Display the HTML Report Card directly
    st.markdown(report_html, unsafe_allow_html=True)

    # Action Buttons
    st.markdown("### ⚡ ACTION")
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("🔥 ALERT TG", use_container_width=True):
            msg = f"TITAN SIGNAL: {symbol} | {'LONG' if last['is_bull'] else 'SHORT'} | EP: {last['close']:.4f} | Score: {ai_analysis['timeframe_score']}/100"
            if send_telegram_msg(tg_token, tg_chat, msg):
                st.success("SENT")
            else:
                st.error("FAIL")

    with b_col2:
        if st.button("📝 REPORT TG", use_container_width=True):
            # Create text version of report
            txt_rep = f"""
SIGNAL: {symbol} {'LONG' if last['is_bull'] else 'SHORT'}
Confidence: {('HIGH' if last['rvol'] > 1.5 else 'MEDIUM')}
Entry: {last['close']:.4f}
Stop: {smart_stop:.4f}
TP1: {last['tp1']:.4f}
Timeframe Score: {ai_analysis['timeframe_score']}/100
Market Regime: {ai_analysis['market_regime']}
            """
            if send_telegram_msg(tg_token, tg_chat, f"REPORT: {symbol}\n{txt_rep}"):
                st.success("SENT")
            else:
                st.error("FAIL")

    # Backtest Mini-Stat
    b_total, b_win, b_net, b_df = run_backtest(df, tp1_r)
    if b_total > 0:
        st.caption(f"📊 Live Stats: {b_win:.1f}% Win Rate | {b_net:.1f}R Net ({b_total} Trades) | Avg: {b_df['pnl'].mean():.2f}R")
    else:
        st.caption("📊 No completed trades in lookback period")

    # ----------------------------------------------------
    # MAIN CHART (Enhanced with AI levels)
    # ----------------------------------------------------
    fig = go.Figure()
    fig.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', name='HMA', line=dict(color='#66fcf1', width=1)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vwap'], mode='lines', name='VWAP', line=dict(color='#9933ff', width=2)))
    
    # Add Fibonacci levels
    fig.add_hline(y=fibs['fib_618'], line_dash="dash", line_color="#ffd740", annotation_text="FIB 0.618")
    fig.add_hline(y=fibs['fib_500'], line_dash="dash", line_color="#ff9800", annotation_text="FIB 0.5")
    fig.add_hline(y=fibs['fib_382'], line_dash="dash", line_color="#ff5722", annotation_text="FIB 0.382")

    buys = df[df['buy']]
    sells = df[df['sell']]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low'], mode='markers',
                                 marker=dict(symbol='triangle-up', size=12, color='#00ff00', line=dict(width=2, color='white')),
                                 name='BUY'))
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high'], mode='markers',
                                 marker=dict(symbol='triangle-down', size=12, color='#ff0000', line=dict(width=2, color='white')),
                                 name='SELL'))

    fig.update_layout(height=400, template='plotly_dark', margin=dict(l=0, r=0, t=20, b=20),
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # INDICATORS (Tabs)
    # ----------------------------------------------------
    t1, t2, t3 = st.tabs(["📊 GANN", "🌊 FLOW", "🧠 SENT"])
    
    with t1:
        f1 = go.Figure()
        f1.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])
        df_g = df.dropna(subset=['gann_act'])
        f1.add_trace(go.Scatter(
            x=df_g['timestamp'],
            y=df_g['gann_act'],
            mode='markers',
            marker=dict(color=np.where(df_g['gann_trend'] == 1, '#00ff00', '#ff0000'), size=4)
        ))
        f1.update_layout(height=300, template='plotly_dark', margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(f1, use_container_width=True)

    with t2:
        f2 = go.Figure()
        colors = ['#00e676' if x > 0 else '#ff1744' for x in df['money_flow']]
        f2.add_trace(go.Bar(x=df['timestamp'], y=df['money_flow'], marker_color=colors, name='Money Flow'))
        # Add RVOL overlay
        f2.add_trace(go.Scatter(x=df['timestamp'], y=df['rvol'], mode='lines', name='RVOL', 
                                line=dict(color='#ffd740', width=1), yaxis='y2'))
        f2.update_layout(height=300, template='plotly_dark', margin=dict(l=0, r=0, t=0, b=0),
                        yaxis=dict(title='Money Flow'),
                        yaxis2=dict(title='RVOL', overlaying='y', side='right'))
        st.plotly_chart(f2, use_container_width=True)

    with t3:
        f3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fg_index,
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [0, 25], 'color': '#ff1744'},
                    {'range': [25, 50], 'color': '#ff9800'},
                    {'range': [50, 75], 'color': '#ffd740'},
                    {'range': [75, 100], 'color': '#00e676'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        f3.update_layout(height=250, template='plotly_dark', margin=dict(l=20, r=20, t=30, b=0))
        st.plotly_chart(f3, use_container_width=True)
        
        # AI Sentiment Breakdown
        st.markdown(f"""
        <div class="ai-card">
            <div class="report-item"><strong>Extreme Fear:</strong> 0-25 (Oversold)</div>
            <div class="report-item"><strong>Fear:</strong> 25-50 (Caution)</div>
            <div class="report-item"><strong>Greed:</strong> 50-75 (Momentum)</div>
            <div class="report-item"><strong>Extreme Greed:</strong> 75-100 (Overbought)</div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # SIGNAL VALIDATION SUMMARY (New)
    # ----------------------------------------------------
    st.markdown("### ✅ SIGNAL VALIDATION CHECKLIST")
    validation_items = {
        "Trend Confirmation": titan_sig == apex_sig,
        "Volume Surge": row['rvol'] > 1.5,
        "Momentum Align": titan_sig == momentum_sig,
        "No Squeeze": not row['in_squeeze'],
        "Timeframe Suitability": ai_analysis['timeframe_score'] >= 70
    }
    
    for item, passed in validation_items.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        color = "#00e676" if passed else "#ff1744"
        st.markdown(f'<div class="report-item" style="color:{color};">{item}: <strong>{status}</strong></div>', unsafe_allow_html=True)
    
    # Overall Signal Grade
    pass_count = sum(validation_items.values())
    if pass_count >= 4:
        grade = "A+ (EXCELLENT)"
        grade_color = "#00e676"
    elif pass_count >= 3:
        grade = "B+ (GOOD)"
        grade_color = "#ffd740"
    elif pass_count >= 2:
        grade = "C (MEDIUM)"
        grade_color = "#ff9800"
    else:
        grade = "D (POOR)"
        grade_color = "#ff1744"
    
    st.markdown(f"""
    <div class="ai-card" style="text-align:center; border: 2px solid {grade_color};">
        <div style="font-size:24px; color:{grade_color};"><strong>SIGNAL GRADE: {grade}</strong></div>
        <div style="font-size:14px;">{pass_count}/5 Validation Checks Passed</div>
    </div>
    """, unsafe_allow_html=True)
    
else:
    st.error("No data returned. Check ticker, timeframe, or Binance US availability.")
    st.info("Tip: Use Quick Ticker selector or verify asset is listed on Binance US")
