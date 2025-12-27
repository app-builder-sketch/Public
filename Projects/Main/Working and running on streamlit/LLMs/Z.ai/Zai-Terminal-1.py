# =============================================================================
# 0. SYSTEM MANIFEST & DEVELOPMENT RULES
# =============================================================================
APP_MANIFEST = {
    "TITAN_MOBILE_ENGINE": [
        "Binance.US API Direct Connection",
        "Multi-Timeframe Scalping Logic (Amplitude/Deviation)",
        "HMA (Hull Moving Average) Trend Filtering",
        "Apex SMC (Smart Money Concepts) & Trail Stops",
        "Gann Swing Theory Implementation",
        "Flux Momentum & Volume Flow Analysis",
        "Volatility Squeeze Detection (BB/KC)",
        "Fear & Greed Sentiment Calculation",
        "Laddered Take Profit Calculation (TP1, TP2, TP3)",
        "VWAP (Volume Weighted Average Price) Relation",
        "HTML/CSS Mobile-First Signal Reporting",
        "Automated Telegram Broadcasting"
    ],
    "AXIOM_QUANT_ENGINE": [
        "YFinance Multi-Asset Data Aggregation",
        "Asset Class Filtering (Crypto, Tech, Indices, Forex)",
        "Glassmorphic World Clock (NY/LON/TOK)",
        "Physics-Based Indicators: CHEDO (Entropy)",
        "Physics-Based Indicators: RQZO (Relativity)",
        "Macro-Economic Dashboard (SPY/VIX Correlation)",
        "Fundamental Data Integration (Market Cap/PE)",
        "AI Strategy Analysis via OpenAI (GPT-4o)",
        "Monte Carlo Simulation (Probability Paths)",
        "Volume Profile & Point of Control (POC)",
        "Day-of-Week DNA Return Analysis"
    ],
    "VISUALIZATION": [
        "Plotly Interactive Candlestick Charts",
        "Dynamic Supply/Demand Zones",
        "Neon/Cyberpunk CSS Styling",
        "Ticker Tapes & Banners"
    ]
}

DEVELOPMENT_RULES = {
    "CORE_DIRECTIVE": "NO OMISSIONS PERMITTED. ALL FEATURES IN APP_MANIFEST ARE MANDATORY.",
    "RULE_1_PERSISTENCE": "When editing or refactoring, EVERY feature listed in APP_MANIFEST must be retained. Do not delete, truncate, or simplify logic.",
    "RULE_2_INTEGRITY": "Calculation engines (Titan/Axiom) must maintain 100% parity with original formulas (Physics, SMC, Gann).",
    "RULE_3_UI": "Visual elements (Clocks, Tickers, CSS) are core features and must not be removed.",
    "RULE_4_REPORTING": "Signal Reports must strictly adhere to the 'Card' based HTML layout with Confidence/VWAP data.",
    "FUTURE_DEPRECATION_ONLY": "Legacy Standalone Functions (at bottom) are the ONLY allowed omission, provided they are fully migrated to Class Methods first."
}

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
# 1. PAGE CONFIG & CSS
# =============================================================================
st.set_page_config(
    page_title="Titan-Axiom Omega",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=SF+Pro+Display:wght@300;500;700&display=swap');

    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'SF Pro Display', sans-serif; }
    
    /* TITAN STYLE ELEMENTS */
    .titan-metric { background: rgba(31, 40, 51, 0.9); border: 1px solid #45a29e; padding: 10px; border-radius: 8px; }
    
    /* AXIOM NEON METRICS */
    div[data-testid="metric-container"] {
        background: rgba(22, 27, 34, 0.9);
        border-left: 4px solid #00F0FF;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 10px;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] { font-size: 14px !important; color: #8b949e !important; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 24px !important; color: #f0f6fc !important; font-weight: 300; }

    /* UNIVERSAL HEADERS & BUTTONS */
    h1, h2, h3 { font-family: 'Roboto Mono', monospace; color: #58a6ff; }
    .stButton > button {
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        border: 1px solid #238636; color: #ffffff;
        font-weight: bold; height: 3.5em; font-size: 16px !important;
        border-radius: 6px;
    }
    .stButton > button:hover { background: #238636; color: #ffffff; }

    /* AXIOM TICKER MARQUEE */
    .ticker-wrap {
        width: 100%; overflow: hidden; background-color: #0d1117; border-bottom: 1px solid #30363d;
        height: 40px; display: flex; align-items: center; margin-bottom: 15px;
    }
    .ticker { display: inline-block; animation: marquee 45s linear infinite; white-space: nowrap; }
    @keyframes marquee { 0% { transform: translate(100%, 0); } 100% { transform: translate(-100%, 0); } }
    .ticker-item { padding: 0 2rem; font-family: 'Roboto Mono'; font-size: 0.85rem; color: #58a6ff; text-shadow: 0 0 5px rgba(88, 166, 255, 0.3); }

    /* TITAN MOBILE REPORT CARDS (MATCHING SCREENSHOT) */
    .report-card { 
        background-color: #161b22; 
        border-left: 4px solid #3fb950; /* Default Green, overridden by inline styles */
        padding: 16px; 
        border-radius: 6px; 
        margin-bottom: 16px; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.5);
        border: 1px solid #30363d;
    }
    .report-header { 
        font-size: 1.1rem; 
        font-weight: 700; 
        color: #f0f6fc; 
        margin-bottom: 12px; 
        border-bottom: 1px solid #30363d; 
        padding-bottom: 8px; 
        display: flex; 
        align-items: center; 
        gap: 8px; 
        font-family: 'Roboto Mono', monospace;
    }
    .report-item { 
        margin-bottom: 8px; 
        font-size: 0.95rem; 
        color: #8b949e; 
        display: flex; 
        align-items: center; 
        gap: 8px; 
        justify-content: flex-start;
    }
    .value-cyan { color: #38bdf8; font-weight: 600; font-family: 'Roboto Mono'; }
    .value-green { color: #3fb950; font-weight: 600; font-family: 'Roboto Mono'; }
    .value-red { color: #ff6b6b; font-weight: 600; font-family: 'Roboto Mono'; }
    .value-yellow { color: #e3b341; font-weight: 600; font-family: 'Roboto Mono'; }
    
    /* TAGS */
    .strategy-tag { background-color: #1f6feb; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-right: 5px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. SHARED UTILITIES
# =============================================================================
class SecretsManager:
    @staticmethod
    def get(key, default=""):
        try: return st.secrets.get(key, default)
        except: return default

def send_telegram(token, chat, msg):
    if not token or not chat: return False
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"}, timeout=3)
        return True
    except: return False

def timeframe_to_min(tf):
    if tf == '15m': return 15
    if tf == '1h': return 60
    if tf == '4h': return 240
    if tf == '1d': return 1440
    return 60

# =============================================================================
# 3. VISUALS ENGINE (DISTINCT MODES)
# =============================================================================
class Visuals:
    @staticmethod
    def render_titan_clock():
        """Simple Digital UTC Clock for Titan Mobile"""
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

    @staticmethod
    def render_axiom_clock():
        """Glassmorphic World Clock for Axiom Quant"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@500&display=swap');
            body { margin: 0; background: transparent; font-family: 'Roboto Mono', monospace; color: #fff; overflow: hidden; }
            .clock-container {
                display: flex; justify-content: space-between; align-items: center;
                background: rgba(0, 0, 0, 0.3); border: 1px solid #222;
                padding: 8px 15px; border-radius: 4px; backdrop-filter: blur(5px);
            }
            .clock-box { text-align: center; width: 32%; }
            .city { font-size: 0.65rem; color: #666; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 2px; }
            .time { font-size: 1.1rem; font-weight: bold; color: #e0e0e0; text-shadow: 0 0 8px rgba(255, 255, 255, 0.1); }
            .accent-ny { border-bottom: 2px solid #00F0FF; }
            .accent-lon { border-bottom: 2px solid #FF0055; }
            .accent-tok { border-bottom: 2px solid #D500F9; }
        </style>
        </head>
        <body>
            <div class="clock-container">
                <div class="clock-box accent-ny"><div class="city">NEW YORK</div><div class="time" id="ny">--:--:--</div></div>
                <div class="clock-box accent-lon"><div class="city">LONDON</div><div class="time" id="lon">--:--:--</div></div>
                <div class="clock-box accent-tok"><div class="city">TOKYO</div><div class="time" id="tok">--:--:--</div></div>
            </div>
            <script>
                function update() {
                    const now = new Date();
                    const fmt = (tz) => new Intl.DateTimeFormat('en-US', {timeZone: tz, hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false}).format(now);
                    document.getElementById('ny').innerText = fmt('America/New_York');
                    document.getElementById('lon').innerText = fmt('Europe/London');
                    document.getElementById('tok').innerText = fmt('Asia/Tokyo');
                }
                setInterval(update, 1000); update();
            </script>
        </body>
        </html>
        """
        components.html(html, height=80)

    @staticmethod
    def render_titan_tape(selected_symbol):
        base = selected_symbol.replace("USDT", "")
        tape_bases = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE"]
        if base not in tape_bases: tape_bases.insert(0, base)
        symbols_json = json.dumps([{"proName": f"BINANCE:{b}USDT", "title": b} for b in tape_bases], separators=(",", ":"))
        components.html(f"""<div class="tradingview-widget-container"><div class="tradingview-widget-container__widget"></div><script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>{{ "symbols": {symbols_json}, "showSymbolLogo": true, "colorTheme": "dark", "isTransparent": true, "displayMode": "adaptive", "locale": "en" }}</script></div>""", height=50)

    @staticmethod
    def render_axiom_banner():
        html = """
        <div class="ticker-wrap">
            <div class="ticker">
                <span class="ticker-item">💠 AXIOM QUANTITATIVE SYSTEM ONLINE</span>
                <span class="ticker-item">BTC-USD: LIVE</span>
                <span class="ticker-item">SPY: LIVE</span>
                <span class="ticker-item">VIX: LIVE</span>
                <span class="ticker-item">GOLD: LIVE</span>
                <span class="ticker-item">NVDA: LIVE</span>
                <span class="ticker-item">EUR/USD: LIVE</span>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# 4. TITAN ENGINE (BINANCE / SCALPING)
# =============================================================================
BINANCE_API_BASE = "https://api.binance.us/api/v3"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

class TitanEngine:
    @staticmethod
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

    @staticmethod
    @st.cache_data(ttl=5, show_spinner=False)
    def get_klines(symbol, interval, limit):
        try:
            r = requests.get(f"{BINANCE_API_BASE}/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, headers=HEADERS, timeout=5)
            if r.status_code == 200:
                df = pd.DataFrame(r.json(), columns=['t','o','h','l','c','v','T','q','n','V','Q','B'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
                return df[['timestamp','open','high','low','close','volume']]
        except: pass
        return pd.DataFrame()

    @staticmethod
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

    @staticmethod
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
        return df['dx'].rolling(length).mean()

    @staticmethod
    def calculate_wavetrend(df, chlen=10, avg=21):
        ap = (df['high'] + df['low'] + df['close']) / 3
        esa = ap.ewm(span=chlen, adjust=False).mean()
        d = (ap - esa).abs().ewm(span=chlen, adjust=False).mean()
        ci = (ap - esa) / (0.015 * d)
        tci = ci.ewm(span=avg, adjust=False).mean()
        return tci

    @staticmethod
    def calculate_fear_greed(df):
        try:
            df = df.copy()
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            if len(df) < 90: return 50
            vol_score = 50 - ((df['log_ret'].rolling(30).std().iloc[-1] - df['log_ret'].rolling(90).std().iloc[-1]) / df['log_ret'].rolling(90).std().iloc[-1]) * 100
            vol_score = max(0, min(100, vol_score))
            rsi = df['rsi'].iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            dist = (df['close'].iloc[-1] - sma_50) / sma_50
            trend_score = 50 + (dist * 1000)
            return int((vol_score * 0.3) + (rsi * 0.4) + (max(0, min(100, trend_score)) * 0.3))
        except: return 50

    @staticmethod
    @st.cache_data(show_spinner=True)
    def run_engine(df, amp, dev, hma_l, gann_l, apex_len, apex_mult, liq_len):
        if df.empty: return df, []
        df = df.copy().reset_index(drop=True)

        # Basics
        df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
        df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
        df['hma'] = TitanEngine.get_ma(df['close'], hma_l, "HMA")

        # Momentum
        delta = df['close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain/loss)))
        df['rvol'] = df['volume'] / df['volume'].rolling(20).mean()

        # Money Flow
        rsi_source = df['rsi'] - 50
        vol_sma = df['volume'].rolling(14).mean()
        df['money_flow'] = (rsi_source * (df['volume'] / vol_sma)).ewm(span=3).mean()
        
        # Squeeze
        bb_basis = df['close'].rolling(20).mean()
        bb_dev = df['close'].rolling(20).std() * 2.0
        kc_basis = df['close'].rolling(20).mean()
        kc_dev = df['atr'] * 1.5
        df['in_squeeze'] = ((bb_basis - bb_dev) > (kc_basis - kc_dev)) & ((bb_basis + bb_dev) < (kc_basis + kc_dev))

        # Titan Trend
        df['ll'] = df['low'].rolling(amp).min()
        df['hh'] = df['high'].rolling(amp).max()
        trend = np.zeros(len(df))
        stop = np.full(len(df), np.nan)
        curr_t = 0; curr_s = np.nan
        
        for i in range(amp, len(df)):
            c = df.at[i,'close']
            d = df.at[i,'atr']*dev
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
        
        # Apex SMC
        df['apex_base'] = TitanEngine.get_ma(df['close'], apex_len, "HMA")
        df['apex_upper'] = df['apex_base'] + (df['atr'] * apex_mult)
        df['apex_lower'] = df['apex_base'] - (df['atr'] * apex_mult)
        df['apex_adx'] = TitanEngine.calculate_adx(df)
        df['apex_tci'] = TitanEngine.calculate_wavetrend(df)
        
        apex_trend = np.zeros(len(df)); apex_trail = np.full(len(df), np.nan)
        visual_zones = []
        curr_at = 0; curr_tr = np.nan
        
        for i in range(max(apex_len, liq_len, 20), len(df)):
            c = df.at[i, 'close']
            if c > df.at[i, 'apex_upper']: curr_at = 1
            elif c < df.at[i, 'apex_lower']: curr_at = -1
            apex_trend[i] = curr_at
            
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

            # Pivots/Zones
            p_idx = i - liq_len
            is_ph = True
            for k in range(1, liq_len + 1):
                if df.at[p_idx, 'high'] <= df.at[p_idx-k, 'high'] or df.at[p_idx, 'high'] <= df.at[p_idx+k, 'high']: is_ph = False; break
            if is_ph:
                 visual_zones.append({'type': 'SUPPLY', 'x0': df.at[p_idx, 'timestamp'], 'x1': df.at[i, 'timestamp'], 'y0': df.at[p_idx, 'high'], 'y1': df.at[p_idx, 'close'], 'color': 'rgba(229, 57, 53, 0.3)'})

            is_pl = True
            for k in range(1, liq_len + 1):
                if df.at[p_idx, 'low'] >= df.at[p_idx-k, 'low'] or df.at[p_idx, 'low'] >= df.at[p_idx+k, 'low']: is_pl = False; break
            if is_pl:
                 visual_zones.append({'type': 'DEMAND', 'x0': df.at[p_idx, 'timestamp'], 'x1': df.at[i, 'timestamp'], 'y0': df.at[p_idx, 'low'], 'y1': df.at[p_idx, 'close'], 'color': 'rgba(67, 160, 71, 0.3)'})

        df['apex_trend'] = apex_trend
        df['apex_trail'] = apex_trail
        
        # --- MISSING FLUX FIX ---
        rg = df['high'] - df['low']
        body = np.abs(df['close'] - df['open'])
        eff_raw = np.where(rg == 0, 0, body / rg)
        eff_sm = pd.Series(eff_raw, index=df.index).ewm(span=14).mean()
        vol_avg = df['volume'].rolling(55).mean()
        v_rat = np.where(vol_avg == 0, 1, df['volume'] / vol_avg)
        direction = np.sign(df['close'] - df['open'])
        raw = direction * eff_sm * pd.Series(v_rat, index=df.index)
        df['Apex_Flux'] = raw.ewm(span=5).mean()
        # ------------------------

        # Gann
        sma_h = df['high'].rolling(gann_l).mean()
        sma_l = df['low'].rolling(gann_l).mean()
        g_trend = np.zeros(len(df))
        curr_g = 1
        for i in range(gann_l, len(df)):
            if curr_g == 1:
                if df.at[i,'close'] < sma_l.iloc[i-1]: curr_g = -1
            else:
                if df.at[i,'close'] > sma_h.iloc[i-1]: curr_g = 1
            g_trend[i] = curr_g
        df['gann_trend'] = g_trend

        # Targets
        risk = abs(df['close'] - df['entry_stop'])
        # --- PANDAS ERROR FIX ---
        risk = risk.mask(risk == 0, df['close'] * 0.01)
        # ------------------------
        df['tp1'] = np.where(df['is_bull'], df['close'] + 1.5*risk, df['close'] - 1.5*risk)
        df['tp2'] = np.where(df['is_bull'], df['close'] + 3.0*risk, df['close'] - 3.0*risk)
        df['tp3'] = np.where(df['is_bull'], df['close'] + 5.0*risk, df['close'] - 5.0*risk)

        # VWAP for Reporting
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['vol_tp'] = df['tp'] * df['volume']
        df['vwap'] = df['vol_tp'].cumsum() / df['volume'].cumsum()

        if len(visual_zones) > 20: visual_zones = visual_zones[-20:]
        return df, visual_zones

    @staticmethod
    def detect_special_setups(df):
        last = df.iloc[-1]; prev = df.iloc[-2]
        setups = { "squeeze_breakout": False, "gann_reversal": False, "rvol_ignition": False }
        if prev['in_squeeze'] and not last['in_squeeze']: setups["squeeze_breakout"] = True
        if last['gann_trend'] != prev['gann_trend']: setups["gann_reversal"] = True
        if last['rvol'] > 3.0: setups["rvol_ignition"] = True
        return setups

    @staticmethod
    def generate_mobile_report(row, fg_index, special_setups):
        # Direction Logic
        is_bull = row['is_bull']
        direction_text = "LONG 🐂" if is_bull else "SHORT 🐻"
        sig_color = "#38bdf8" # Cyan
        
        # Confidence Logic
        titan_s = 1 if is_bull else -1
        apex_s = row['apex_trend']
        gann_s = row['gann_trend']
        score = 0
        if titan_s == apex_s: score += 1
        if titan_s == gann_s: score += 1
        
        conf_text = "LOW"
        if score == 2: conf_text = "HIGH"
        elif score == 1: conf_text = "MEDIUM"
        
        # Squeeze Logic
        squeeze_active = row['in_squeeze']
        sqz_text = "SQUEEZE ACTIVE" if squeeze_active else "NO SQUEEZE"
        sqz_icon = "⚠️" if squeeze_active else "⚪"
        sqz_class = "value-yellow" if squeeze_active else "value-cyan"

        # Vol/Flow Logic
        rvol_val = row['rvol']
        rvol_desc = "(Normal)"
        if rvol_val > 2.0: rvol_desc = "(Ignition)"
        elif rvol_val < 0.5: rvol_desc = "(Low)"
        
        vwap_rel = "Above" if row['close'] > row['vwap'] else "Below"
        vwap_color_class = "value-cyan" # Default

        return f"""
        <div class="report-card" style="border-left: 4px solid #38bdf8;">
            <div class="report-header">
                <span>💠 SIGNAL: {direction_text}</span>
            </div>
            <div class="report-item">Confidence: <span class="value-cyan">{conf_text}</span></div>
            <div class="report-item">Sentiment: <span class="value-cyan">{fg_index}/100</span></div>
            <div class="report-item">Squeeze: {sqz_icon} <span class="{sqz_class}">{sqz_text}</span></div>
        </div>

        <div class="report-card" style="border-left: 4px solid #38bdf8;">
            <div class="report-header">
                <span>🌊 FLOW & VOL</span>
            </div>
            <div class="report-item">RVOL: <span class="value-cyan">{rvol_val:.2f} {rvol_desc}</span></div>
            <div class="report-item">Money Flow: <span class="value-cyan">{row['money_flow']:.2f}</span></div>
            <div class="report-item">VWAP Relation: <span class="{vwap_color_class}">{vwap_rel}</span></div>
        </div>
        
        <div class="report-card" style="border-left: 4px solid #38bdf8;">
            <div class="report-header">
                <span>🎯 EXECUTION PLAN</span>
            </div>
            <div class="report-item">Entry: <span class="value-cyan">{row['close']:.4f}</span></div>
            <div class="report-item">🛑 SMART STOP: <span class="value-cyan">{row['entry_stop']:.4f}</span></div>
            <div class="report-item">1️⃣ TP1 (1.5R): <span class="value-cyan">{row['tp1']:.4f}</span></div>
            <div class="report-item">2️⃣ TP2 (3.0R): <span class="value-cyan">{row['tp2']:.4f}</span></div>
            <div class="report-item">3️⃣ TP3 (5.0R): <span class="value-cyan">{row['tp3']:.4f}</span></div>
        </div>
        """

# =============================================================================
# 5. AXIOM ENGINE (STOCKS / PHYSICS)
# =============================================================================
class AxiomEngine:
    @staticmethod
    def get_asset_classes():
        return [
            "Crypto (Major)", "Crypto (Alt/Meme)", "Indices & ETFs", 
            "US Tech (Mag 7+)", "US Large Cap", "High Volatility", 
            "Commodities & Forex"
        ]

    @staticmethod
    def get_tickers_by_class(asset_class):
        universe = {
            "Crypto (Major)": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD"],
            "Crypto (Alt/Meme)": ["SHIB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FLOKI-USD", "DOGE-USD"],
            "Indices & ETFs": ["SPY", "QQQ", "IWM", "DIA", "TLT", "VXX", "UVXY", "SQQQ", "TQQQ", "SOXL"],
            "US Tech (Mag 7+)": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD"],
            "US Large Cap": ["JPM", "BAC", "WFC", "JNJ", "LLY", "PFE", "PG", "KO", "PEP", "COST", "WMT"],
            "High Volatility": ["MSTR", "COIN", "MARA", "RIOT", "CLSK", "GME", "AMC", "HOOD", "DKNG"],
            "Commodities & Forex": ["GLD", "SLV", "USO", "EURUSD=X", "GBPUSD=X", "JPY=X", "GC=F", "CL=F"]
        }
        return universe.get(asset_class, [])

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_data(ticker, timeframe, limit=500):
        tf_map = {"15m": "1mo", "1h": "6mo", "4h": "1y", "1d": "2y", "1wk": "5y"}
        period = tf_map.get(timeframe, "1y")
        try:
            df = yf.download(ticker, period=period, interval=timeframe, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs(ticker, axis=1, level=0)
                except: df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={c: c.capitalize() for c in df.columns}) # Keep Capital for Axiom
            if 'Adj close' in df.columns: df['Close'] = df['Adj close']
            return df.dropna().tail(limit)
        except: return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_fundamentals(ticker):
        if "-" in ticker or "=" in ticker: return None
        try:
            stock = yf.Ticker(ticker)
            return { "Market Cap": stock.info.get("marketCap", "N/A"), "P/E Ratio": stock.info.get("trailingPE", "N/A") }
        except: return None

    @staticmethod
    @st.cache_data(ttl=1800)
    def get_macro_data():
        assets = {"S&P 500": "SPY", "VIX": "^VIX"}
        try:
            data = yf.download(list(assets.values()), period="5d", interval="1d", progress=False)['Close']
            prices = {k: data[v].iloc[-1] for k,v in assets.items() if v in data}
            changes = {k: ((data[v].iloc[-1]-data[v].iloc[-2])/data[v].iloc[-2])*100 for k,v in assets.items() if v in data}
            return prices, changes
        except: return {}, {}

    @staticmethod
    def tanh(x): return np.tanh(np.clip(x, -20, 20))

    @staticmethod
    def calc_chedo(df, length=50):
        c = df['Close'].values
        log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
        mu = pd.Series(log_ret).rolling(length).mean().values
        sigma = pd.Series(log_ret).rolling(length).std().values
        v = sigma / (np.abs(mu) + 1e-9)
        abs_ret_v = np.abs(log_ret) * v
        hyper_dist = np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))
        kappa_h = AxiomEngine.tanh(pd.Series(hyper_dist).rolling(length).mean().values)
        diff_ret = np.diff(log_ret, prepend=0)
        lyap = np.log(np.abs(diff_ret) + 1e-9)
        lambda_n = AxiomEngine.tanh((pd.Series(lyap).rolling(length).mean().values + 5) / 7)
        ent = pd.Series(log_ret**2).rolling(length).sum().values
        ent_n = AxiomEngine.tanh(ent * 10)
        raw = (0.4 * kappa_h) + (0.3 * lambda_n) + (0.3 * ent_n)
        df['CHEDO'] = 2 / (1 + np.exp(-raw * 4)) - 1
        return df

    @staticmethod
    def calc_rqzo(df, harmonics=25):
        src = df['Close']
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

    @staticmethod
    def calc_apex_flux(df, length=14):
        rg = df['High'] - df['Low']
        body = np.abs(df['Close'] - df['Open'])
        eff_raw = np.where(rg == 0, 0, body / rg)
        eff_sm = pd.Series(eff_raw, index=df.index).ewm(span=length).mean()
        vol_avg = df['Volume'].rolling(55).mean()
        v_rat = np.where(vol_avg == 0, 1, df['Volume'] / vol_avg)
        direction = np.sign(df['Close'] - df['Open'])
        raw = direction * eff_sm * pd.Series(v_rat, index=df.index)
        df['Apex_Flux'] = raw.ewm(span=5).mean()
        df['Apex_State'] = np.select([df['Apex_Flux'] > 0.6, df['Apex_Flux'] < -0.6], ["Super Bull", "Super Bear"], default="Neutral")
        return df

    @staticmethod
    def calc_smc(df, length=55):
        def wma(s, l):
            w = np.arange(1, l+1)
            return s.rolling(l).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
        close = df['Close']
        half = int(length/2); sqrt = int(np.sqrt(length))
        wma_f = wma(close, length); wma_h = wma(close, half)
        df['HMA_Trend'] = wma(2*wma_h - wma_f, sqrt)
        df['Trend_Dir'] = np.where(close > df['HMA_Trend'], 1, -1)
        return df

    @staticmethod
    def calc_fear_greed_v4(df):
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain/loss))
        macd = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['FG_Index'] = (rsi + (macd * 10)).clip(0, 100).rolling(5).mean()
        return df

    @staticmethod
    @st.cache_data(ttl=600)
    def run_monte_carlo(df, days=30, sims=100):
        last_price = df['Close'].iloc[-1]
        returns = df['Close'].pct_change().dropna()
        mu = returns.mean(); sigma = returns.std()
        sim_rets = np.random.normal(mu, sigma, (days, sims))
        paths = np.zeros((days, sims)); paths[0] = last_price
        for t in range(1, days): paths[t] = paths[t-1] * (1 + sim_rets[t])
        return paths

    @staticmethod
    def calc_volume_profile(df, bins=50):
        price_min = df['Low'].min(); price_max = df['High'].max()
        price_bins = np.linspace(price_min, price_max, bins)
        df['Mid'] = (df['Close'] + df['Open']) / 2
        df['Bin'] = pd.cut(df['Mid'], bins=price_bins, labels=price_bins[:-1], include_lowest=True)
        vp = df.groupby('Bin', observed=False)['Volume'].sum().reset_index()
        vp['Price'] = vp['Bin'].astype(float)
        poc = vp.loc[vp['Volume'].idxmax(), 'Price']
        return vp, poc

    @staticmethod
    @st.cache_data(ttl=3600)
    def calc_day_of_week_dna(ticker):
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            df['Day'] = df.index.day_name()
            df['Ret'] = df['Close'].pct_change() * 100
            return df.groupby('Day')['Ret'].mean()
        except: return None

    @staticmethod
    def analyze_ai(ticker, price, chedo, rqzo, flux, api_key):
        if not api_key: return "❌ Missing OpenAI API Key."
        prompt = f"Analyze {ticker} at {price}. Metrics: Entropy (CHEDO) {chedo:.2f}, Relativity (RQZO) {rqzo:.2f}, Flux {flux:.2f}. >0.8 Entropy is chaos. >0.6 Flux is breakout. Brief strategy."
        try:
            client = OpenAI(api_key=api_key)
            return client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content":prompt}]).choices[0].message.content
        except Exception as e: return f"AI Error: {e}"

# =============================================================================
# 6. MAIN CONTROLLER
# =============================================================================
def main():
    # SIDEBAR CONTROLLER
    st.sidebar.header("💠 MEGA-STATION")
    mode = st.sidebar.radio("ENGINE MODE", ["TITAN MOBILE (Crypto)", "AXIOM QUANT (Stocks)"])
    
    with st.sidebar.expander("🔐 KEYS"):
        tg_token = st.text_input("Bot Token", value=SecretsManager.get("TELEGRAM_TOKEN"), type="password")
        tg_chat = st.text_input("Chat ID", value=SecretsManager.get("TELEGRAM_CHAT_ID"))
        ai_key = st.text_input("AI Key", value=SecretsManager.get("OPENAI_API_KEY"), type="password")

    # -------------------------------------------------------------------------
    # MODE 1: TITAN MOBILE (Binance)
    # -------------------------------------------------------------------------
    if mode == "TITAN MOBILE (Crypto)":
        st.sidebar.subheader("📡 BINANCE FEED")
        bases = TitanEngine.get_binance_bases()
        idx = bases.index("BTC") if "BTC" in bases else 0
        base = st.sidebar.selectbox("Asset", bases, index=idx)
        ticker = f"{base}USDT"
        
        c1, c2 = st.sidebar.columns(2)
        with c1: timeframe = st.selectbox("TF", ["15m", "1h", "4h", "1d"], index=1)
        with c2: limit = st.slider("Depth", 100, 500, 200, 50)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧠 LOGIC")
        amp = st.sidebar.number_input("Amplitude", 2, 100, 10)
        dev = st.sidebar.number_input("Deviation", 0.5, 5.0, 3.0)
        hma_len = st.sidebar.number_input("HMA Len", 10, 200, 50)
        gann_len = st.sidebar.number_input("Gann Len", 2, 50, 3)

        # MAIN UI TITAN
        st.title(f"💠 TITAN: {base}")
        Visuals.render_titan_clock()
        Visuals.render_titan_tape(ticker)
        
        with st.spinner("Connecting to Binance..."):
            df = TitanEngine.get_klines(ticker, timeframe, limit)
        
        if not df.empty:
            df, zones = TitanEngine.run_engine(df, int(amp), dev, int(hma_len), int(gann_len), 55, 1.5, 10)
            last = df.iloc[-1]
            
            # METRICS
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("TREND", "BULL 🟢" if last['is_bull'] else "BEAR 🔴")
            c2.metric("FLUX", f"{last['Apex_Flux']:.2f}")
            c3.metric("STOP", f"{last['entry_stop']:.2f}")
            c4.metric("TP3", f"{last['tp3']:.2f}")
            
            # HTML REPORT
            fg = TitanEngine.calculate_fear_greed(df)
            spec = TitanEngine.detect_special_setups(df)
            st.markdown(TitanEngine.generate_mobile_report(last, fg, spec), unsafe_allow_html=True)
            
            # TELEGRAM
            if st.button("📢 SEND SIGNAL"):
                msg = f"🚀 *TITAN SIGNAL* 🚀\nSymbol: {ticker}\nSide: {'LONG' if last['is_bull'] else 'SHORT'}\nEntry: {last['close']}\nStop: {last['entry_stop']}\nTP1: {last['tp1']}\nTP2: {last['tp2']}\nTP3: {last['tp3']}"
                if send_telegram(tg_token, tg_chat, msg): st.success("SENT")
                else: st.error("FAIL")
            
            # CHART
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], line=dict(color='#00F0FF', width=1), name='HMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['apex_trail'], line=dict(color='orange', width=1, dash='dot'), name='Trail'), row=1, col=1)
            for z in zones: fig.add_shape(type="rect", x0=z['x0'], x1=z['x1'], y0=z['y0'], y1=z['y1'], fillcolor=z['color'], line_width=0, row=1, col=1)
            colors = np.where(df['Apex_Flux'] > 0, '#00E676', '#FF1744')
            fig.add_trace(go.Bar(x=df['timestamp'], y=df['Apex_Flux'], marker_color=colors, name='Flux'), row=2, col=1)
            fig.update_layout(height=500, template='plotly_dark', margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # MODE 2: AXIOM QUANT (Stocks/YFinance)
    # -------------------------------------------------------------------------
    else:
        st.sidebar.subheader("📡 MARKET DATA")
        ac_list = AxiomEngine.get_asset_classes()
        ac = st.sidebar.selectbox("Sector", ac_list)
        ticks = AxiomEngine.get_tickers_by_class(ac)
        ticker = st.sidebar.selectbox("Ticker", ticks)
        tf = st.sidebar.selectbox("TF", ["15m", "1h", "4h", "1d", "1wk"], index=3)
        
        st.title(f"💠 AXIOM: {ticker}")
        Visuals.render_axiom_clock()
        Visuals.render_axiom_banner()
        
        with st.spinner("Crunching Physics..."):
            df = AxiomEngine.fetch_data(ticker, tf)
        
        if not df.empty:
            # RUN AXIOM LOGIC
            df = AxiomEngine.calc_chedo(df)
            df = AxiomEngine.calc_rqzo(df)
            df = AxiomEngine.calc_apex_flux(df)
            df = AxiomEngine.calc_smc(df)
            last = df.iloc[-1]
            fund = AxiomEngine.get_fundamentals(ticker)
            macro_p, macro_c = AxiomEngine.get_macro_data()
            
            # DASHBOARD METRICS
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PRICE", f"{last['Close']:.2f}")
            c2.metric("ENTROPY", f"{last['CHEDO']:.2f}", delta="Risk" if abs(last['CHEDO'])>0.8 else "Stable")
            c3.metric("FLUX", f"{last['Apex_Flux']:.2f}", delta=last['Apex_State'])
            c4.metric("TREND", "BULL" if last['Trend_Dir']==1 else "BEAR")

            # TABS
            tabs = st.tabs(["📉 TECH", "🌍 MACRO", "📅 DNA", "🧠 AI", "📊 VOL", "🔮 SIM"])
            
            with tabs[0]: # TECH CHART
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.02)
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['HMA_Trend'], line=dict(color='#fff', width=1, dash='dot'), name='HMA'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], line=dict(color='#00F0FF', width=2), fill='tozeroy', fillcolor='rgba(0,240,255,0.1)', name='Entropy'), row=2, col=1)
                colors = np.where(df['Apex_Flux'] > 0.6, '#00E676', np.where(df['Apex_Flux'] < -0.6, '#FF1744', '#2979FF'))
                fig.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], marker_color=colors, name='Flux'), row=3, col=1)
                fig.update_layout(height=700, template='plotly_dark', margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]: # MACRO
                c1, c2 = st.columns(2)
                c1.metric("S&P 500", f"${macro_p.get('S&P 500',0):.2f}", f"{macro_c.get('S&P 500',0):.2f}%")
                c2.metric("VIX", f"{macro_p.get('VIX',0):.2f}", f"{macro_c.get('VIX',0):.2f}%")
                if fund: st.write(f"**Fundamentals**: Cap {fund['Market Cap']} | PE {fund['P/E Ratio']}")

            with tabs[2]: # DNA
                dna = AxiomEngine.calc_day_of_week_dna(ticker)
                if dna is not None: st.bar_chart(dna)

            with tabs[3]: # AI
                if st.button("RUN INTELLIGENCE"):
                    res = AxiomEngine.analyze_ai(ticker, last['Close'], last['CHEDO'], last['RQZO'], last['Apex_Flux'], ai_key)
                    st.info(res)

            with tabs[4]: # VOLUME
                vp, poc = AxiomEngine.calc_volume_profile(df)
                st.bar_chart(vp.set_index('Price')['Volume'])
                st.caption(f"POC: {poc:.2f}")

            with tabs[5]: # MONTE CARLO
                mc = AxiomEngine.run_monte_carlo(df)
                st.line_chart(mc[:, :20])

if __name__ == "__main__":
    main()
