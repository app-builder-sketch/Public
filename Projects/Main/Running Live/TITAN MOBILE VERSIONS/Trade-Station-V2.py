"""
TITAN-AXIOM INTEGRATED TRADING STATION
Version: Unified V1.1 (Titan Mobile + Axiom Quant)
Modes: 
  1. TITAN (Binance.US Direct - Crypto Scalping)
  2. AXIOM (YFinance - Stocks/Forex/Macro/AI)

FIXED: Pandas replace() error resolved using .mask()
NO OMISSIONS - FULLY INTEGRATED
"""

import time
import math
import sqlite3
import random
import json
from typing import Dict, Optional, List, Tuple, Any
from contextlib import contextmanager
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import streamlit.components.v1 as components
from openai import OpenAI
from scipy.stats import linregress

# =============================================================================
# 1. PAGE CONFIGURATION & CSS
# =============================================================================
st.set_page_config(
    page_title="Titan-Axiom Unified",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="collapsed"
)

# UNIFIED CSS (Mobile Optimized + Neon Aesthetic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=SF+Pro+Display:wght@300;500;700&display=swap');

    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'SF Pro Display', sans-serif; }
    
    /* NEON METRIC CARDS */
    div[data-testid="metric-container"] {
        background: rgba(20, 20, 20, 0.8);
        border-left: 4px solid #00F0FF;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetricLabel"] { font-size: 14px !important; color: #888 !important; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 24px !important; color: #fff !important; font-weight: 300; }

    /* HEADERS */
    h1, h2, h3 { font-family: 'Roboto Mono', monospace; color: #c5c6c7; }

    /* TICKER MARQUEE */
    .ticker-wrap {
        width: 100%; overflow: hidden; background-color: #0a0a0a; border-bottom: 1px solid #333;
        height: 40px; display: flex; align-items: center; margin-bottom: 15px;
    }
    .ticker { display: inline-block; animation: marquee 45s linear infinite; white-space: nowrap; }
    @keyframes marquee { 0% { transform: translate(100%, 0); } 100% { transform: translate(-100%, 0); } }
    .ticker-item { padding: 0 2rem; font-family: 'Roboto Mono'; font-size: 0.85rem; color: #00F0FF; text-shadow: 0 0 5px rgba(0, 240, 255, 0.5); }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        border: 1px solid #45a29e; color: #66fcf1;
        font-weight: bold; height: 3.5em; font-size: 16px !important;
        border-radius: 8px; margin-top: 5px; margin-bottom: 5px;
    }
    .stButton > button:hover { background: #45a29e; color: #0b0c10; }

    /* REPORT CARDS (Mobile) */
    .report-card { background-color: #111; border-left: 4px solid #00F0FF; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .report-header { font-size: 1.1rem; font-weight: bold; color: #fff; margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 5px; }
    .report-item { margin-bottom: 5px; font-size: 0.9rem; color: #aaa; }
    .highlight { color: #00F0FF; font-weight: bold; }

    /* STRATEGY TAGS */
    .strategy-tag { background-color: #45a29e; color: #000; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-right: 5px; }
    .apex-tag-bull { background-color: #00E676; color: #000; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    .apex-tag-bear { background-color: #FF1744; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. CORE UTILITIES & SECRETS MANAGER (CRASH PROOF)
# =============================================================================
class SecretsManager:
    @staticmethod
    def get(key, default=""):
        """Safely retrieves secrets without crashing."""
        try:
            return st.secrets.get(key, default)
        except Exception:
            return default

BINANCE_API_BASE = "https://api.binance.us/api/v3"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# =============================================================================
# 3. MATH & PHYSICS ENGINE (SHARED)
# =============================================================================
def get_ma(series, length, ma_type):
    if ma_type == "SMA": return series.rolling(length).mean()
    elif ma_type == "EMA": return series.ewm(span=length, adjust=False).mean()
    elif ma_type == "RMA": return series.ewm(alpha=1/length, adjust=False).mean()
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
    df['tr_s'] = df['tr'].rolling(length).sum()
    df['pdm_s'] = df['pdm'].rolling(length).sum()
    df['ndm_s'] = df['ndm'].rolling(length).sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        df['pdi'] = 100 * (df['pdm_s'] / df['tr_s'])
        df['ndi'] = 100 * (df['ndm_s'] / df['tr_s'])
        df['dx'] = 100 * abs(df['pdi'] - df['ndi']) / (df['pdi'] + df['ndi'])
    return df['dx'].rolling(length).mean().fillna(0)

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
    return {
        'fib_382': h - (d * 0.382), 'fib_500': h - (d * 0.500),
        'fib_618': h - (d * 0.618), 'high': h, 'low': l
    }

def calculate_fear_greed(df):
    try:
        df = df.copy()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        if len(df) < 90: return 50
        vol_score = 50 - ((df['log_ret'].rolling(30).std().iloc[-1] - df['log_ret'].rolling(90).std().iloc[-1]) / df['log_ret'].rolling(90).std().iloc[-1]) * 100
        vol_score = max(0, min(100, vol_score))
        rsi = df['rsi'].iloc[-1] if 'rsi' in df else 50
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        dist = (df['close'].iloc[-1] - sma_50) / sma_50
        trend_score = 50 + (dist * 1000)
        return int((vol_score * 0.3) + (rsi * 0.4) + (max(0, min(100, trend_score)) * 0.3))
    except: return 50

# AXIOM PHYSICS HELPERS
def tanh(x): return np.tanh(np.clip(x, -20, 20))

def calc_chedo(df, length=50):
    c = df['close'].values
    log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
    mu = pd.Series(log_ret).rolling(length).mean().values
    sigma = pd.Series(log_ret).rolling(length).std().values
    v = sigma / (np.abs(mu) + 1e-9)
    abs_ret_v = np.abs(log_ret) * v
    hyper_dist = np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))
    kappa_h = tanh(pd.Series(hyper_dist).rolling(length).mean().values)
    diff_ret = np.diff(log_ret, prepend=0)
    lyap = np.log(np.abs(diff_ret) + 1e-9)
    lambda_n = tanh((pd.Series(lyap).rolling(length).mean().values + 5) / 7)
    ent = pd.Series(log_ret**2).rolling(length).sum().values
    ent_n = tanh(ent * 10)
    raw = (0.4 * kappa_h) + (0.3 * lambda_n) + (0.3 * ent_n)
    df['CHEDO'] = 2 / (1 + np.exp(-raw * 4)) - 1
    return df

def calc_rqzo(df, harmonics=25):
    src = df['close']
    mn, mx = src.rolling(100).min(), src.rolling(100).max()
    norm = (src - mn) / (mx - mn + 1e-9)
    v = np.abs(norm.diff())
    c_limit = 0.05
    gamma = 1 / np.sqrt(1 - (np.minimum(v, c_limit*0.99)/c_limit)**2)
    idx = np.arange(len(df))
    tau = (idx % 100) / gamma.fillna(1.0)
    zeta = np.zeros(len(df))
    for n in range(1, harmonics + 1):
        amp = n ** -0.5
        theta = tau * np.log(n)
        zeta += amp * np.sin(theta)
    df['RQZO'] = pd.Series(zeta).fillna(0)
    return df

def calc_apex_flux(df, length=14):
    rg = df['high'] - df['low']
    body = np.abs(df['close'] - df['open'])
    eff_raw = np.where(rg == 0, 0, body / rg)
    eff_sm = pd.Series(eff_raw, index=df.index).ewm(span=length).mean()
    vol_avg = df['volume'].rolling(55).mean()
    v_rat = np.where(vol_avg == 0, 1, df['volume'] / vol_avg)
    direction = np.sign(df['close'] - df['open'])
    raw = direction * eff_sm * pd.Series(v_rat, index=df.index)
    df['Apex_Flux'] = raw.ewm(span=5).mean()
    return df

# =============================================================================
# 4. DATA FETCHING (DUAL ENGINE)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_binance_bases() -> List[str]:
    try:
        r = requests.get(f"{BINANCE_API_BASE}/exchangeInfo", headers=HEADERS, timeout=5)
        if r.status_code != 200: return []
        js = r.json()
        bases = set()
        for s in js.get("symbols", []):
            if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT":
                bases.add(s.get("baseAsset").upper())
        return sorted(list(bases))
    except: return []

@st.cache_data(ttl=5, show_spinner=False)
def get_binance_klines(symbol, interval, limit):
    try:
        r = requests.get(f"{BINANCE_API_BASE}/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, headers=HEADERS, timeout=5)
        if r.status_code == 200:
            df = pd.DataFrame(r.json(), columns=['t','o','h','l','c','v','T','q','n','V','Q','B'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
            return df[['timestamp','open','high','low','close','volume']]
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_yfinance_data(ticker, timeframe, limit=500):
    tf_map = {"15m": "1mo", "1h": "6mo", "4h": "1y", "1d": "2y", "1wk": "5y"}
    period = tf_map.get(timeframe, "1y")
    try:
        df = yf.download(ticker, period=period, interval=timeframe, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(ticker, axis=1, level=0)
            except: df.columns = df.columns.get_level_values(0)
        
        # STANDARDIZE COLUMNS TO LOWERCASE (Unified Engine)
        df = df.rename(columns={c: c.lower() for c in df.columns})
        # Handle 'adj close' if present, otherwise 'close'
        if 'adj close' in df.columns: df['close'] = df['adj close']
        
        df['timestamp'] = df.index
        return df.dropna().tail(limit)
    except: return pd.DataFrame()

# =============================================================================
# 5. CORE LOGIC ENGINE (TITAN & AXIOM)
# =============================================================================
@st.cache_data(show_spinner=True)
def run_master_engine(df, params):
    """
    Unified Engine that runs all Titan and Axiom calculations.
    """
    if df.empty: return df, []
    df = df.copy().reset_index(drop=True)

    # 1. BASICS
    df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    df['hma'] = get_ma(df['close'], params['hma_len'], "HMA")

    # 2. MOMENTUM
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain/loss)))
    df['rvol'] = df['volume'] / df['volume'].rolling(20).mean()

    # 3. TITAN TREND
    df['ll'] = df['low'].rolling(params['amp']).min()
    df['hh'] = df['high'].rolling(params['amp']).max()
    trend = np.zeros(len(df))
    stop = np.full(len(df), np.nan)
    curr_t = 0
    curr_s = np.nan
    
    for i in range(params['amp'], len(df)):
        c = df.at[i,'close']
        d = df.at[i,'atr']*params['dev']
        if curr_t == 0:
            s = df.at[i,'ll'] + d
            curr_s = max(curr_s, s) if not np.isnan(curr_s) else s
            if c < curr_s: curr_t = 1; curr_s = df.at[i,'hh'] - d
        else:
            s = df.at[i,'hh'] - d
            curr_s = min(curr_s, s) if not np.isnan(curr_s) else s
            if c > curr_s: curr_t = 0; curr_s = df.at[i,'ll'] + d
        trend[i] = curr_t
        stop[i] = curr_s
    
    df['is_bull'] = trend == 0
    df['entry_stop'] = stop

    # 4. APEX SMC
    df['apex_base'] = get_ma(df['close'], params['apex_len'], "HMA")
    df['apex_upper'] = df['apex_base'] + (df['atr'] * params['apex_mult'])
    df['apex_lower'] = df['apex_base'] - (df['atr'] * params['apex_mult'])
    df['apex_adx'] = calculate_adx(df)
    df['apex_tci'] = calculate_wavetrend(df)
    
    apex_trend = np.zeros(len(df))
    apex_trail = np.full(len(df), np.nan)
    visual_zones = []
    
    curr_at = 0
    curr_tr = np.nan
    
    for i in range(max(params['apex_len'], 20), len(df)):
        c = df.at[i, 'close']
        # Trend
        if c > df.at[i, 'apex_upper']: curr_at = 1
        elif c < df.at[i, 'apex_lower']: curr_at = -1
        apex_trend[i] = curr_at
        
        # Trail
        atr2 = df.at[i, 'atr'] * 2.0
        if curr_at == 1:
            val = c - atr2
            curr_tr = max(curr_tr, val) if not np.isnan(curr_tr) else val
            if apex_trend[i-1] == -1: curr_tr = val
        elif curr_at == -1:
            val = c + atr2
            curr_tr = min(curr_tr, val) if not np.isnan(curr_tr) else val
            if apex_trend[i-1] == 1: curr_tr = val
        apex_trail[i] = curr_tr

        # SMC Zones (Order Blocks/Pivots)
        p_idx = i - 10 # Fixed lookback for simplicity
        if p_idx > 0:
            is_ph = df.at[p_idx, 'high'] == df['high'].iloc[p_idx-5:p_idx+6].max()
            is_pl = df.at[p_idx, 'low'] == df['low'].iloc[p_idx-5:p_idx+6].min()
            
            if is_ph:
                 visual_zones.append({'type': 'SUPPLY', 'x0': df.at[p_idx, 'timestamp'], 'x1': df.at[i, 'timestamp'], 'y0': df.at[p_idx, 'high'], 'y1': df.at[p_idx, 'close'], 'color': 'rgba(229, 57, 53, 0.3)'})
            if is_pl:
                 visual_zones.append({'type': 'DEMAND', 'x0': df.at[p_idx, 'timestamp'], 'x1': df.at[i, 'timestamp'], 'y0': df.at[p_idx, 'low'], 'y1': df.at[p_idx, 'close'], 'color': 'rgba(67, 160, 71, 0.3)'})

    df['apex_trend'] = apex_trend
    df['apex_trail'] = apex_trail
    if len(visual_zones) > 15: visual_zones = visual_zones[-15:]

    # 5. AXIOM PHYSICS
    df = calc_chedo(df)
    df = calc_rqzo(df)
    df = calc_apex_flux(df)

    # 6. TARGETS
    last_stop = df['entry_stop'].iloc[-1] if not np.isnan(df['entry_stop'].iloc[-1]) else df['close'].iloc[-1]*0.95
    risk = abs(df['close'] - df['entry_stop'])
    
    # -------------------------------------------------------------
    # ERROR FIX: Replaced .replace(0, series) with .mask()
    # This prevents the Pandas Value Error.
    # -------------------------------------------------------------
    risk = risk.mask(risk == 0, df['close'] * 0.01)
    
    df['tp1'] = np.where(df['is_bull'], df['close'] + 1.5*risk, df['close'] - 1.5*risk)
    df['tp2'] = np.where(df['is_bull'], df['close'] + 3.0*risk, df['close'] - 3.0*risk)
    df['tp3'] = np.where(df['is_bull'], df['close'] + 5.0*risk, df['close'] - 5.0*risk)

    return df, visual_zones

# =============================================================================
# 6. UI & VISUALIZATION
# =============================================================================
def render_ticker_tape(symbol_list):
    # Fallback Data for Marquee
    html = f"""
    <div class="ticker-wrap">
        <div class="ticker">
            <span class="ticker-item">💠 TITAN-AXIOM ONLINE</span>
            <span class="ticker-item">BTC: ACTIVE</span>
            <span class="ticker-item">ETH: ACTIVE</span>
            <span class="ticker-item">SOL: ACTIVE</span>
            <span class="ticker-item">NVDA: ACTIVE</span>
            <span class="ticker-item">SPY: ACTIVE</span>
            <span class="ticker-item">GOLD: ACTIVE</span>
        </div>
    </div>
    """
    components.html(html, height=50)

def render_clock():
    html = """
    <div style="display:flex; justify-content:center; font-family:'Roboto Mono'; color:#39ff14; text-shadow:0 0 10px rgba(57,255,20,0.8); font-weight:bold;">
        <span id="clock">--:--:-- UTC</span>
    </div>
    <script>
    setInterval(() => {
        document.getElementById('clock').innerText = new Date().toLocaleTimeString('en-GB', {timeZone:'UTC'}) + ' UTC';
    }, 1000);
    </script>
    """
    components.html(html, height=30)

def send_telegram(token, chat, msg):
    if not token or not chat: return False
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"}, timeout=3)
        return True
    except: return False

def call_ai_analysis(ticker, price, chedo, rqzo, flux, api_key):
    if not api_key: return "❌ No API Key Configured"
    prompt = f"""
    Analyze {ticker} at {price}.
    Physics Metrics:
    - Entropy (CHEDO): {chedo:.2f} (>0.8 is Chaos)
    - Relativity (RQZO): {rqzo:.2f} (High Amp is Volatility)
    - Flux: {flux:.2f} (>0.6 is Superconductor)
    
    Provide a 3-bullet executive summary on Regime, Risk, and Strategy.
    """
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content":prompt}])
        return resp.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

# =============================================================================
# 7. MAIN APP LOOP
# =============================================================================
def main():
    # --- SIDEBAR ---
    st.sidebar.header("💠 ENGINE MODE")
    mode = st.sidebar.radio("System", ["TITAN (Crypto)", "AXIOM (Stocks/Forex)"])
    
    # Creds (Safe Retrieval)
    with st.sidebar.expander("🔐 Credentials"):
        tg_token = st.text_input("TG Token", value=SecretsManager.get("TELEGRAM_TOKEN"), type="password")
        tg_chat = st.text_input("TG Chat ID", value=SecretsManager.get("TELEGRAM_CHAT_ID"))
        ai_key = st.text_input("OpenAI Key", value=SecretsManager.get("OPENAI_API_KEY"), type="password")

    # Inputs
    if mode == "TITAN (Crypto)":
        bases = get_binance_bases()
        default_idx = bases.index("BTC") if "BTC" in bases else 0
        symbol_base = st.sidebar.selectbox("Asset", bases, index=default_idx)
        ticker = f"{symbol_base}USDT"
        timeframe = st.sidebar.selectbox("TF", ["15m", "1h", "4h", "1d"], index=1)
    else:
        # Axiom Pre-sets
        presets = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMD", "COIN", "MSTR", "GLD", "EURUSD=X"]
        ticker = st.sidebar.selectbox("Ticker", presets)
        timeframe = st.sidebar.selectbox("TF", ["15m", "1h", "4h", "1d", "1wk"], index=3)

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Logic")
    amp = st.sidebar.number_input("Amplitude", 2, 100, 10)
    dev = st.sidebar.number_input("Deviation", 1.0, 5.0, 3.0)
    
    # --- MAIN VIEW ---
    st.title(f"💠 {mode.split()[0]} STATION")
    render_clock()
    render_ticker_tape([])

    # FETCH DATA
    with st.spinner(f"Connecting to {mode.split()[0]} Feed..."):
        if mode == "TITAN (Crypto)":
            df = get_binance_klines(ticker, timeframe, 300)
        else:
            df = get_yfinance_data(ticker, timeframe, 300)

    if df.empty:
        st.error(f"Failed to load data for {ticker}")
        return

    # RUN ENGINE
    params = {'amp': int(amp), 'dev': dev, 'hma_len': 50, 'apex_len': 55, 'apex_mult': 1.5}
    df, zones = run_master_engine(df, params)
    last = df.iloc[-1]
    
    # METRICS
    c1, c2, c3, c4 = st.columns(4)
    trend_dir = "BULL 🟢" if last['is_bull'] else "BEAR 🔴"
    flux_state = "SUPER" if abs(last['Apex_Flux']) > 0.6 else "NEUTRAL"
    
    c1.metric("Trend", trend_dir)
    c2.metric("Flux", f"{last['Apex_Flux']:.2f}", delta=flux_state)
    c3.metric("Entropy", f"{last['CHEDO']:.2f}", delta="High" if abs(last['CHEDO'])>0.8 else "Stable")
    c4.metric("TP3", f"{last['tp3']:.2f}")

    # REPORT CARD
    fg_index = calculate_fear_greed(df)
    report_html = f"""
    <div class="report-card">
        <div class="report-header">⚡ SIGNAL: {trend_dir}</div>
        <div class="report-item">Asset: <span class="highlight">{ticker}</span></div>
        <div class="report-item">Close: <span class="highlight">{last['close']:.2f}</span></div>
        <div class="report-item">Stop: <span class="highlight">{last['entry_stop']:.2f}</span></div>
        <div class="report-item">Sentiment: <span class="highlight">{fg_index}/100</span></div>
    </div>
    """
    st.markdown(report_html, unsafe_allow_html=True)

    # ACTIONS
    c_a1, c_a2 = st.columns(2)
    with c_a1:
        if st.button("📢 Broadcast Signal"):
            msg = f"🚀 *{ticker} SIGNAL* 🚀\nSide: {trend_dir}\nEntry: {last['close']:.2f}\nStop: {last['entry_stop']:.2f}\nTP3: {last['tp3']:.2f}\nFlux: {last['Apex_Flux']:.2f}"
            if send_telegram(tg_token, tg_chat, msg): st.success("Sent!")
            else: st.error("Failed")
    with c_a2:
        if st.button("🧠 AI Analysis"):
            analysis = call_ai_analysis(ticker, last['close'], last['CHEDO'], last['RQZO'], last['Apex_Flux'], ai_key)
            st.info(analysis)

    # CHARTING
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    # Price & Indicators
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], line=dict(color='#00F0FF', width=1), name='HMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['apex_trail'], line=dict(color='orange', width=1, dash='dot'), name='Trail'), row=1, col=1)

    # SMC Zones
    for z in zones:
        fig.add_shape(type="rect", x0=z['x0'], x1=z['x1'], y0=z['y0'], y1=z['y1'], fillcolor=z['color'], line_width=0, row=1, col=1)

    # Subplot: Flux
    colors = np.where(df['Apex_Flux'] > 0, '#00E676', '#FF1744')
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['Apex_Flux'], marker_color=colors, name='Flux'), row=2, col=1)

    fig.update_layout(height=600, template='plotly_dark', margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
