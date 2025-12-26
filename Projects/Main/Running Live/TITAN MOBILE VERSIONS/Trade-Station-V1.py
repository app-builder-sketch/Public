"""
Trade-Station-MOBILE EDITION
Version 18.4: Professional Telegram Reporting + Integrated Apex SMC
(Upgraded & Optimized)
"""

import time
import math
import sqlite3
import random
import json
from typing import Dict, Optional, List, Tuple, Any
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
    page_title="Trade-Station-MOBILE v18.4",
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
        height: 3.5em;
        font-size: 16px !important;
        border-radius: 8px;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .stButton > button:hover {
        background: #45a29e;
        color: #0b0c10;
    }

    /* Report Card Styling */
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
    
    /* Strategy Tags */
    .strategy-tag { background-color: #45a29e; color: #000; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-right: 5px; }
    .apex-tag-bull { background-color: #00E676; color: #000; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    .apex-tag-bear { background-color: #FF1744; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold; }

</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
BINANCE_API_BASE = "https://api.binance.us/api/v3"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# =============================================================================
# TICKER UNIVERSE
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_binanceus_usdt_bases() -> List[str]:
    """Fetches available USDT pairs from Binance.US with error handling."""
    try:
        r = requests.get(f"{BINANCE_API_BASE}/exchangeInfo", headers=HEADERS, timeout=10)
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
        return sorted(list(bases))
    except Exception as e:
        # Silently fail or log in a real app; returning empty list ensures non-breaking
        return []

# =============================================================================
# LIVE TICKER WIDGET
# =============================================================================
def render_ticker_tape(selected_symbol: str):
    base = selected_symbol.replace("USDT", "")
    tape_bases = ["BTC", "ETH", "SOL"]
    tape_bases.extend([
        "XRP", "BNB", "ADA", "DOGE", "LINK", "AVAX", "DOT", "MATIC", "LTC", "BCH",
        "ATOM", "XLM", "ETC", "AAVE", "UNI", "SHIB", "TRX", "FIL", "NEAR", "ICP"
    ])
    if base and base not in tape_bases:
        tape_bases.insert(0, base)
    
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

# HEADER
st.title("💠 TITAN MOBILE v18.4")
st.caption("AI TRADING ENGINE | PRO TELEGRAM SIGNALS")

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
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ CONTROL")
    if st.button("🔄 REFRESH", use_container_width=True):
        st.rerun()

    st.subheader("📡 FEED")
    bases_all = get_binanceus_usdt_bases()
    if "symbol_input" not in st.session_state:
        st.session_state.symbol_input = "BTC"

    with st.expander("🧬 Ticker Universe (Live Menu)", expanded=True):
        if bases_all:
            # Check if current input exists in the list to prevent index errors
            current_index = 0
            if st.session_state.symbol_input in bases_all:
                current_index = bases_all.index(st.session_state.symbol_input)
            elif "BTC" in bases_all:
                current_index = bases_all.index("BTC")

            selected_base = st.selectbox(
                "Select Asset (Searchable)",
                options=bases_all,
                index=current_index,
                key="uni_select"
            )
            if selected_base != st.session_state.symbol_input:
                st.session_state.symbol_input = selected_base
            st.caption(f"⚡ {len(bases_all)} Live Pairs Loaded")
        else:
            st.warning("Ticker universe unavailable. Manual input active.")

    symbol_input = st.text_input("Active Asset", value=st.session_state.symbol_input)
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
    st.subheader("🧠 TITAN LOGIC")
    amplitude = st.number_input("Amp", 2, 200, 10)
    channel_dev = st.number_input("Dev", 0.5, 10.0, 3.0, 0.1)
    hma_len = st.number_input("Titan HMA", 2, 400, 50)
    gann_len = st.number_input("Gann", 1, 50, 3)

    st.markdown("---")
    st.subheader("🦅 APEX SMC LOGIC")
    apex_ma_type = st.selectbox("Apex MA Type", ["HMA", "EMA", "SMA", "RMA"], index=0)
    apex_len = st.number_input("Apex Trend Len", 10, 200, 55)
    apex_mult = st.number_input("Apex Cloud Mult", 0.1, 5.0, 1.5, 0.1)
    pivot_len = st.number_input("Pivot Lookback", 2, 50, 10)

    with st.expander("🎯 Targets"):
        tp1_r = st.number_input("TP1 (R)", value=1.5)
        tp2_r = st.number_input("TP2 (R)", value=3.0)
        tp3_r = st.number_input("TP3 (R)", value=5.0)

    st.markdown("---")
    st.subheader("🤖 NOTIFICATIONS")
    
    # Safe secret retrieval
    default_token = st.secrets.get("TELEGRAM_TOKEN", "") if "TELEGRAM_TOKEN" in st.secrets else ""
    default_chat = st.secrets.get("TELEGRAM_CHAT_ID", "") if "TELEGRAM_CHAT_ID" in st.secrets else ""
    
    tg_token = st.text_input("Bot Token", value=default_token, type="password")
    tg_chat = st.text_input("Chat ID", value=default_chat)

render_ticker_tape(symbol)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_ma(series, length, ma_type):
    if ma_type == "SMA":
        return series.rolling(length).mean()
    elif ma_type == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    elif ma_type == "RMA":
        return series.ewm(alpha=1/length, adjust=False).mean()
    else: # HMA
        half_len = int(length / 2)
        sqrt_len = int(math.sqrt(length))
        wma_f = series.rolling(length).mean()
        wma_h = series.rolling(half_len).mean()
        diff = 2 * wma_h - wma_f
        return diff.rolling(sqrt_len).mean()

def calculate_adx(df, length=14):
    df = df.copy()
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['pdm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['ndm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    
    # Improved efficiency
    df['tr_s'] = df['tr'].rolling(length).sum()
    df['pdm_s'] = df['pdm'].rolling(length).sum()
    df['ndm_s'] = df['ndm'].rolling(length).sum()
    
    df['pdi'] = 100 * (df['pdm_s'] / df['tr_s'])
    df['ndi'] = 100 * (df['ndm_s'] / df['tr_s'])
    # Avoid division by zero
    den = df['pdi'] + df['ndi']
    df['dx'] = np.where(den != 0, 100 * abs(df['pdi'] - df['ndi']) / den, 0)
    return df['dx'].rolling(length).mean()

def calculate_wavetrend(df, chlen=10, avg=21):
    ap = (df['high'] + df['low'] + df['close']) / 3
    esa = ap.ewm(span=chlen, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=chlen, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d)
    tci = ci.ewm(span=avg, adjust=False).mean()
    return tci

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
        df = df.copy()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        # Protect against empty slices
        if len(df) < 90: return 50
        
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

# =============================================================================
# ENGINE
# =============================================================================
@st.cache_data(show_spinner=True)
def run_engines(df, amp, dev, hma_l, tp1, tp2, tp3, mf_l, vol_l, gann_l, apex_len, apex_mult, apex_ma_type, liq_len):
    """
    Main logic engine. Cached to improve performance during UI interactions.
    """
    if df.empty:
        return df, []
    df = df.copy().reset_index(drop=True)

    # --- TITAN INDICATORS ---
    df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    df['hma'] = get_ma(df['close'], hma_l, "HMA") # Titan always HMA

    # VWAP
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vol_tp'] = df['tp'] * df['volume']
    df['vwap'] = df['vol_tp'].cumsum() / df['volume'].cumsum()

    # Squeeze
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

    # Money Flow
    rsi_source = df['rsi'] - 50
    vol_sma = df['volume'].rolling(mf_l).mean()
    df['money_flow'] = (rsi_source * (df['volume'] / vol_sma)).ewm(span=3).mean()

    # --- TITAN TREND ---
    df['ll'] = df['low'].rolling(amp).min()
    df['hh'] = df['high'].rolling(amp).max()
    trend = np.zeros(len(df))
    stop = np.full(len(df), np.nan)
    curr_t = 0
    curr_s = np.nan
    
    # Loop Optimization: Pre-fetch columns as numpy arrays for speed
    # Note: State dependence prevents full vectorization of this block
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
    
    # Titan Signals
    cond_buy = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False)) & (df['rvol']>1.0)
    cond_sell = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True)) & (df['rvol']>1.0)
    df['buy'] = cond_buy
    df['sell'] = cond_sell

    # Titan Targets
    df['sig_id'] = (df['buy']|df['sell']).cumsum()
    df['entry'] = df.groupby('sig_id')['close'].ffill()
    df['stop_val'] = df.groupby('sig_id')['entry_stop'].ffill()
    risk = abs(df['entry'] - df['stop_val'])
    df['tp1'] = np.where(df['is_bull'], df['entry']+(risk*tp1), df['entry']-(risk*tp1))
    df['tp2'] = np.where(df['is_bull'], df['entry']+(risk*tp2), df['entry']-(risk*tp2))
    df['tp3'] = np.where(df['is_bull'], df['entry']+(risk*tp3), df['entry']-(risk*tp3))

    # --- APEX SMC INTEGRATION ---
    df['apex_base'] = get_ma(df['close'], apex_len, apex_ma_type)
    df['apex_upper'] = df['apex_base'] + (df['atr'] * apex_mult)
    df['apex_lower'] = df['apex_base'] - (df['atr'] * apex_mult)
    df['apex_adx'] = calculate_adx(df)
    df['apex_tci'] = calculate_wavetrend(df)
    df['apex_vol_avg'] = df['volume'].rolling(20).mean()

    # Arrays for Loop
    apex_trend = np.zeros(len(df))
    apex_buy_sig = np.zeros(len(df))
    apex_sell_sig = np.zeros(len(df))
    apex_trail_stop = np.full(len(df), np.nan)
    bos_bull = np.zeros(len(df))
    bos_bear = np.zeros(len(df))
    
    # Zone Storage for Visuals
    visual_zones = [] # List of dicts {'type', 'x0', 'x1', 'y0', 'y1', 'color'}

    curr_apex_t = 0
    last_ph = np.nan
    last_pl = np.nan
    
    # Apex Trailing Stop Logic Var
    curr_trail = np.nan

    for i in range(max(apex_len, liq_len, 20), len(df)):
        c = df.at[i, 'close']
        upper = df.at[i, 'apex_upper']
        lower = df.at[i, 'apex_lower']
        
        # 1. Apex Trend
        if c > upper:
            curr_apex_t = 1
        elif c < lower:
            curr_apex_t = -1
        apex_trend[i] = curr_apex_t
        
        # 2. Apex Trailing Stop (ATR * 2)
        trail_atr = df.at[i, 'atr'] * 2.0
        if curr_apex_t == 1:
            val = c - trail_atr
            curr_trail = max(curr_trail, val) if not np.isnan(curr_trail) else val
            if apex_trend[i-1] == -1: curr_trail = val # Reset
        elif curr_apex_t == -1:
            val = c + trail_atr
            curr_trail = min(curr_trail, val) if not np.isnan(curr_trail) else val
            if apex_trend[i-1] == 1: curr_trail = val # Reset
        apex_trail_stop[i] = curr_trail

        # 3. Apex Signals
        vol_ok = df.at[i, 'volume'] > df.at[i, 'apex_vol_avg']
        adx_ok = df.at[i, 'apex_adx'] > 20
        if curr_apex_t == 1 and apex_trend[i-1] != 1 and vol_ok and df.at[i, 'apex_tci'] < 60 and adx_ok:
            apex_buy_sig[i] = 1
        if curr_apex_t == -1 and apex_trend[i-1] != -1 and vol_ok and df.at[i, 'apex_tci'] > -60 and adx_ok:
            apex_sell_sig[i] = 1

        # 4. Supply/Demand (Pivots)
        p_idx = i - liq_len
        is_ph = True
        for k in range(1, liq_len + 1):
            if df.at[p_idx, 'high'] <= df.at[p_idx-k, 'high'] or df.at[p_idx, 'high'] <= df.at[p_idx+k, 'high']:
                is_ph = False; break
        if is_ph:
            last_ph = df.at[p_idx, 'high']
            visual_zones.append({
                'type': 'SUPPLY', 'x0': df.at[p_idx, 'timestamp'], 'x1': df.at[i, 'timestamp'] + pd.Timedelta(minutes=timeframe_to_min(timeframe)*20),
                'y0': df.at[p_idx, 'high'], 'y1': max(df.at[p_idx, 'open'], df.at[p_idx, 'close']),
                'color': 'rgba(229, 57, 53, 0.3)' # Red
            })

        is_pl = True
        for k in range(1, liq_len + 1):
            if df.at[p_idx, 'low'] >= df.at[p_idx-k, 'low'] or df.at[p_idx, 'low'] >= df.at[p_idx+k, 'low']:
                is_pl = False; break
        if is_pl:
            last_pl = df.at[p_idx, 'low']
            visual_zones.append({
                'type': 'DEMAND', 'x0': df.at[p_idx, 'timestamp'], 'x1': df.at[i, 'timestamp'] + pd.Timedelta(minutes=timeframe_to_min(timeframe)*20),
                'y0': df.at[p_idx, 'low'], 'y1': min(df.at[p_idx, 'open'], df.at[p_idx, 'close']),
                'color': 'rgba(67, 160, 71, 0.3)' # Green
            })

        # 5. Structure Break (BOS)
        x_ph = curr_apex_t == 1 and not np.isnan(last_ph) and c > last_ph and df.at[i-1, 'close'] <= last_ph
        x_pl = curr_apex_t == -1 and not np.isnan(last_pl) and c < last_pl and df.at[i-1, 'close'] >= last_pl
        if x_ph: bos_bull[i] = 1
        if x_pl: bos_bear[i] = 1

        # 6. Order Blocks (OB)
        if x_ph:
            for k in range(1, 21):
                idx_ob = i - k
                if idx_ob < 0: break
                if df.at[idx_ob, 'close'] < df.at[idx_ob, 'open']: # Down Candle
                    visual_zones.append({
                        'type': 'OB_BULL', 'x0': df.at[idx_ob, 'timestamp'], 'x1': df.at[i, 'timestamp'] + pd.Timedelta(minutes=timeframe_to_min(timeframe)*20),
                        'y0': df.at[idx_ob, 'high'], 'y1': df.at[idx_ob, 'low'],
                        'color': 'rgba(185, 246, 202, 0.4)' # Pale Mint
                    })
                    break
        if x_pl:
            for k in range(1, 21):
                idx_ob = i - k
                if idx_ob < 0: break
                if df.at[idx_ob, 'close'] > df.at[idx_ob, 'open']: # Up Candle
                    visual_zones.append({
                        'type': 'OB_BEAR', 'x0': df.at[idx_ob, 'timestamp'], 'x1': df.at[i, 'timestamp'] + pd.Timedelta(minutes=timeframe_to_min(timeframe)*20),
                        'y0': df.at[idx_ob, 'high'], 'y1': df.at[idx_ob, 'low'],
                        'color': 'rgba(255, 205, 210, 0.4)' # Pale Rose
                    })
                    break

        # 7. FVG Detection
        if i >= 2:
            if df.at[i, 'low'] > df.at[i-2, 'high']:
                gap_size = df.at[i, 'low'] - df.at[i-2, 'high']
                if gap_size > df.at[i, 'atr'] * 0.5:
                    visual_zones.append({
                         'type': 'FVG_BULL', 'x0': df.at[i-2, 'timestamp'], 'x1': df.at[i, 'timestamp'] + pd.Timedelta(minutes=timeframe_to_min(timeframe)*10),
                         'y0': df.at[i, 'low'], 'y1': df.at[i-2, 'high'],
                         'color': 'rgba(185, 246, 202, 0.3)'
                    })
            if df.at[i, 'high'] < df.at[i-2, 'low']:
                gap_size = df.at[i-2, 'low'] - df.at[i, 'high']
                if gap_size > df.at[i, 'atr'] * 0.5:
                     visual_zones.append({
                         'type': 'FVG_BEAR', 'x0': df.at[i-2, 'timestamp'], 'x1': df.at[i, 'timestamp'] + pd.Timedelta(minutes=timeframe_to_min(timeframe)*10),
                         'y0': df.at[i-2, 'low'], 'y1': df.at[i, 'high'],
                         'color': 'rgba(255, 205, 210, 0.3)'
                    })

    df['apex_trend'] = apex_trend
    df['apex_buy'] = apex_buy_sig
    df['apex_sell'] = apex_sell_sig
    df['apex_trail'] = apex_trail_stop
    df['bos_bull'] = bos_bull
    df['bos_bear'] = bos_bear
    
    # Gann Logic (Preserved)
    sma_h = df['high'].rolling(gann_l).mean()
    sma_l = df['low'].rolling(gann_l).mean()
    g_trend = np.full(len(df), np.nan)
    curr_g_t = 1
    for i in range(gann_l, len(df)):
        c = df.at[i,'close']
        if curr_g_t == 1:
            if c < sma_l.iloc[i-1] if i>0 else 0: curr_g_t = -1
        else:
            if c > sma_h.iloc[i-1] if i>0 else 999999: curr_g_t = 1
        g_trend[i] = curr_g_t
    df['gann_trend'] = g_trend

    # Limit visual zones
    if len(visual_zones) > 20:
        visual_zones = visual_zones[-20:]

    return df, visual_zones

def timeframe_to_min(tf):
    if tf == '15m': return 15
    if tf == '1h': return 60
    if tf == '4h': return 240
    if tf == '1d': return 1440
    return 60

def run_backtest(df, tp1_r):
    trades = []
    signals = df[(df['buy']) | (df['sell'])]
    for idx, row in signals.iterrows():
        future = df.loc[idx+1: idx+20]
        if future.empty: continue
        entry = row['close']; stop = row['entry_stop']; tp1 = row['tp1']; is_long = row['is_bull']
        outcome = "PENDING"; pnl = 0
        if is_long:
            if future['high'].max() >= tp1: outcome = "WIN"; pnl = abs(entry - stop) * tp1_r
            elif future['low'].min() <= stop: outcome = "LOSS"; pnl = -abs(entry - stop)
        else:
            if future['low'].min() <= tp1: outcome = "WIN"; pnl = abs(entry - stop) * tp1_r
            elif future['high'].max() >= stop: outcome = "LOSS"; pnl = -abs(entry - stop)
        if outcome != "PENDING": trades.append({'outcome': outcome, 'pnl': pnl})
    if not trades: return 0, 0, 0
    df_res = pd.DataFrame(trades)
    return len(df_res), (len(df_res[df_res['outcome']=='WIN'])/len(df_res))*100, (len(df_res[df_res['outcome']=='WIN'])*tp1_r)-len(df_res[df_res['outcome']=='LOSS'])

# --- SPECIAL SETUPS ---
def detect_special_setups(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    setups = { "squeeze_breakout": False, "gann_reversal": False, "rvol_ignition": False, "apex_buy": False, "apex_sell": False, "bos": False }
    if prev['in_squeeze'] and not last['in_squeeze']: setups["squeeze_breakout"] = True
    if last['gann_trend'] != prev['gann_trend']: setups["gann_reversal"] = True
    if last['rvol'] > 3.0: setups["rvol_ignition"] = True
    if last['apex_buy'] == 1: setups['apex_buy'] = True
    if last['apex_sell'] == 1: setups['apex_sell'] = True
    if last['bos_bull'] == 1 or last['bos_bear'] == 1: setups['bos'] = True
    return setups

# --- REPORT ---
def generate_mobile_report(row, fg_index, smart_stop, special_setups):
    direction = "LONG 🐂" if row['is_bull'] else "SHORT 🐻"
    setup_tags = ""
    if special_setups['gann_reversal']: setup_tags += "<span class='strategy-tag'>GANN FLIP</span>"
    if special_setups['squeeze_breakout']: setup_tags += "<span class='strategy-tag'>SQZ BREAK</span>"
    if special_setups['rvol_ignition']: setup_tags += "<span class='strategy-tag'>HIGH VOL</span>"
    if special_setups['apex_buy']: setup_tags += "<span class='apex-tag-bull'>APEX BUY</span>"
    if special_setups['apex_sell']: setup_tags += "<span class='apex-tag-bear'>APEX SELL</span>"
    if special_setups['bos']: setup_tags += "<span class='strategy-tag'>BOS</span>"
    if setup_tags == "": setup_tags = "<span>Standard Trend</span>"
    
    return f"""
    <div class="report-card">
        <div class="report-header">💠 SIGNAL: {direction}</div>
        <div class="report-item">Strategy: {setup_tags}</div>
        <div class="report-item">Sentiment: <span class="highlight">{fg_index}/100</span></div>
    </div>
    <div class="report-card">
        <div class="report-header">🌊 APEX & FLOW</div>
        <div class="report-item">Apex Trend: <span class="highlight">{'BULL 🟢' if row['apex_trend']==1 else 'BEAR 🔴'}</span></div>
        <div class="report-item">RVOL: <span class="highlight">{row['rvol']:.2f}</span></div>
    </div>
    <div class="report-card">
        <div class="report-header">🎯 EXECUTION</div>
        <div class="report-item">Entry: <span class="highlight">{row['close']:.4f}</span></div>
        <div class="report-item">🛑 STOP: <span class="highlight">{smart_stop:.4f}</span></div>
        <div class="report-item">3️⃣ TP3: <span class="highlight">{row['tp3']:.4f}</span></div>
    </div>
    """

def send_telegram_msg(token, chat, msg):
    if not token or not chat: return False
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"}, timeout=5)
        return True
    except: return False

@st.cache_data(ttl=5, show_spinner=False)
def get_klines(symbol_bin, interval, limit):
    """
    Fetch klines with retry logic for robust mobile connections.
    """
    retries = 3
    for attempt in range(retries):
        try:
            r = requests.get(f"{BINANCE_API_BASE}/klines", params={"symbol": symbol_bin, "interval": interval, "limit": limit}, headers=HEADERS, timeout=4)
            if r.status_code == 200:
                df = pd.DataFrame(r.json(), columns=['t','o','h','l','c','v','T','q','n','V','Q','B'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
                return df[['timestamp','open','high','low','close','volume']]
        except requests.exceptions.RequestException:
            if attempt < retries - 1:
                time.sleep(0.5) # Short backoff
                continue
            pass
    return pd.DataFrame()

# =============================================================================
# APP EXECUTION
# =============================================================================
with st.spinner("Connecting to Binance Engine..."):
    df = get_klines(symbol, timeframe, limit)

if not df.empty:
    df = df.dropna(subset=['close'])
    # Cached Execution
    df, visual_zones = run_engines(df, int(amplitude), channel_dev, int(hma_len), tp1_r, tp2_r, tp3_r, 14, 20, int(gann_len), int(apex_len), apex_mult, apex_ma_type, int(pivot_len))

    last = df.iloc[-1]
    fibs = calculate_fibonacci(df)
    fg_index = calculate_fear_greed_index(df)
    special_setups = detect_special_setups(df)
    smart_stop = min(last['entry_stop'], fibs['fib_618'] * 0.9995) if last['is_bull'] else max(last['entry_stop'], fibs['fib_618'] * 1.0005)

    # METRICS
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        tv_symbol = f"BINANCE:{symbol}"
        components.html(f"""<div class="tradingview-widget-container"><div class="tradingview-widget-container__widget"></div><script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js" async>{{ "symbol": "{tv_symbol}", "width": "100%", "colorTheme": "dark", "isTransparent": true, "locale": "en" }}</script></div>""", height=120)
    with c_m2: st.metric("APEX TREND", "BULL 🟢" if last['apex_trend'] == 1 else "BEAR 🔴")
    c_m3, c_m4 = st.columns(2)
    with c_m3: st.metric("STOP", f"{smart_stop:.2f}")
    with c_m4: st.metric("TP3", f"{last['tp3']:.2f}")

    # REPORT
    report_html = generate_mobile_report(last, fg_index, smart_stop, special_setups)
    st.markdown(report_html, unsafe_allow_html=True)

    # ACTIONS (PROFESSIONAL TELEGRAM REPORTING)
    st.markdown("### ⚡ SIGNAL ACTIONS")
    
    # Message Builders
    def build_titan_msg():
        trend = "LONG 🐂" if last['is_bull'] else "SHORT 🐻"
        return f"""
🚀 *TITAN TRADE ALERT* 🚀
Symbol: *{symbol}* ({timeframe})
Side: *{trend}*
Entry: `{last['close']:.4f}`

🛑 Stop: `{smart_stop:.4f}`
🎯 TP1: `{last['tp1']:.4f}`
🎯 TP2: `{last['tp2']:.4f}`
🎯 TP3: `{last['tp3']:.4f}`

📊 *Analysis:*
• RVOL: `{last['rvol']:.2f}`
• Sentiment: `{fg_index}/100`
• Titan Trend: *ACTIVE*
"""

    def build_apex_msg(mode):
        icon = "🟢" if mode == "BUY" else "🔴"
        return f"""
{icon} *APEX {mode} SIGNAL* {icon}
Symbol: *{symbol}*
Price: `{last['close']:.4f}`

🌊 *Momentum Metrics:*
• Apex Trend: *{mode}*
• TCI: `{last['apex_tci']:.2f}`
• ADX: `{last['apex_adx']:.2f}`

⚡ *Action:* Check chart for execution.
"""

    if st.button("📢 STANDARD TITAN ALERT", use_container_width=True):
        if send_telegram_msg(tg_token, tg_chat, build_titan_msg()): st.success("SENT")
        else: st.error("FAIL")

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        if special_setups['apex_buy']: 
            if st.button("🟢 APEX BUY ALERT", use_container_width=True):
                send_telegram_msg(tg_token, tg_chat, build_apex_msg("BUY"))
        elif special_setups['apex_sell']: 
            if st.button("🔴 APEX SELL ALERT", use_container_width=True):
                send_telegram_msg(tg_token, tg_chat, build_apex_msg("SELL"))
    with c_s2:
        if special_setups['bos']: 
            if st.button("🏛️ BOS ALERT", use_container_width=True):
                send_telegram_msg(tg_token, tg_chat, f"🏛️ *MARKET STRUCTURE BREAK* 🏛️\nSymbol: {symbol}\nPrice: `{last['close']:.4f}`\n\nStructure has shifted. Watch for retest.")

    # Full Report Button (Text Only Version of HTML Card)
    if st.button("📝 SEND FULL REPORT", use_container_width=True):
        txt_rep = f"""
📝 *FULL ANALYSIS REPORT*
Symbol: {symbol} | TF: {timeframe}

💠 *SIGNAL:* {'LONG 🐂' if last['is_bull'] else 'SHORT 🐻'}
• Conf: {'MAX 🔥' if (last['is_bull'] and last['apex_trend']==1) else 'HIGH'}
• Sent: {fg_index}/100

🌊 *FLOW:*
• Apex: {'BULL 🟢' if last['apex_trend']==1 else 'BEAR 🔴'}
• RVOL: {last['rvol']:.2f}

🎯 *LEVELS:*
• EP: `{last['close']:.4f}`
• SL: `{smart_stop:.4f}`
• TP3: `{last['tp3']:.4f}`
"""
        if send_telegram_msg(tg_token, tg_chat, txt_rep): st.success("SENT");
        else: st.error("FAIL");

    # BACKTEST
    b_total, b_win, b_net = run_backtest(df, tp1_r)
    st.caption(f"📊 Live Stats: {b_win:.1f}% Win Rate | {b_net:.1f}R Net ({b_total} Trades)")

    # CHART (VISUAL ZONES INTEGRATED)
    fig = go.Figure()

    # 1. Apex Cloud
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['apex_upper'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['apex_lower'], fill='tonexty', 
                             fillcolor='rgba(0, 105, 92, 0.2)' if last['apex_trend'] == 1 else 'rgba(183, 28, 28, 0.2)',
                             line=dict(width=0), name='Apex Cloud', hoverinfo='skip'))

    # 2. SMC Visual Zones (Rectangles)
    for zone in visual_zones:
        fig.add_shape(type="rect",
            x0=zone['x0'], y0=zone['y0'], x1=zone['x1'], y1=zone['y1'],
            fillcolor=zone['color'], line=dict(width=0),
        )

    # 3. Candles & Lines
    fig.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', name='HMA', line=dict(color='#66fcf1', width=1)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['apex_trail'], mode='lines', name='Apex Trail', line=dict(color='#FFA726', width=1, dash='dot')))

    # 4. Signals
    apex_buys = df[df['apex_buy'] == 1]
    apex_sells = df[df['apex_sell'] == 1]
    if not apex_buys.empty:
         fig.add_trace(go.Scatter(x=apex_buys['timestamp'], y=apex_buys['low']*0.999, mode='markers', marker=dict(symbol='arrow-bar-up', size=12, color='#00E676'), name='APEX BUY'))
    if not apex_sells.empty:
         fig.add_trace(go.Scatter(x=apex_sells['timestamp'], y=apex_sells['high']*1.001, mode='markers', marker=dict(symbol='arrow-bar-down', size=12, color='#FF1744'), name='APEX SELL'))

    # Update layout for mobile touch friendly behavior
    fig.update_layout(
        height=450, 
        template='plotly_dark', 
        margin=dict(l=0, r=0, t=20, b=20), 
        xaxis_rangeslider_visible=False, 
        legend=dict(orientation="h", y=1, x=0),
        hovermode="x unified" # Mobile Optimization: Unified hover box
    )
    st.plotly_chart(fig, use_container_width=True)

    # INDICATORS TABS
    t1, t2, t3 = st.tabs(["📊 GANN", "🌊 FLOW", "🧠 SENT"])
    with t1:
        f1 = go.Figure()
        f1.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])
        f1.update_layout(height=300, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), hovermode="x unified")
        st.plotly_chart(f1, use_container_width=True)
    with t2:
        f2 = go.Figure()
        cols = ['#00e676' if x > 0 else '#ff1744' for x in df['money_flow']]
        f2.add_trace(go.Bar(x=df['timestamp'], y=df['money_flow'], marker_color=cols))
        f2.update_layout(height=300, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), hovermode="x unified")
        st.plotly_chart(f2, use_container_width=True)
    with t3:
        f3 = go.Figure(go.Indicator(mode="gauge+number", value=fg_index, gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "white"}}))
        f3.update_layout(height=250, template='plotly_dark', margin=dict(l=20,r=20,t=30,b=0))
        st.plotly_chart(f3, use_container_width=True)
else:
    st.error("No data returned. Check ticker/Binance availability.")
