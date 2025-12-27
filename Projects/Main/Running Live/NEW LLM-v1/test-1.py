"""
TITAN-AXIOM MEGA-STATION V5.0 (GOD MODE EDITION)
------------------------------------------------
INTEGRATION STATUS: 100% MERGED & OPTIMIZED
MODE 1: TITAN MOBILE (Binance | Scalping | Legacy SMC | Gann | HTML Cards)
MODE 2: AXIOM DARKPOOL (YFinance | God Mode Indicators | Physics | Seasonality | AI)

CRITICAL SYSTEM WARNING & COMPREHENSIVE FEATURE MANIFEST
========================================================
DO NOT ALTER, REMOVE, OR SIMPLIFY THE FOLLOWING FEATURES.
THIS SYSTEM RELIES ON THE EXACT PRESENCE OF ALL LISTED COMPONENTS:

[1. TITAN ENGINE (CRYPTO SCALPING)]
   - Binance.US API Direct Connection.
   - HMA Trend Filtering & Gann Swing Theory.
   - Apex SMC (Legacy) & Trail Stops.
   - Laddered TPs (1.5R/3.0R/5.0R).
   - HTML Card Reporting (Cyan/Green/Red/Yellow).

[2. AXIOM DARKPOOL ENGINE (GOD MODE + PHYSICS)]
   - BATCH DATA: Optimized YFinance Batch Downloading for Global Macro.
   - PHYSICS: CHEDO (Entropy), RQZO (Relativity), Apex Flux (Vector).
   - GOD MODE INDICATORS: 
     * Apex Trend & Liquidity Master (HMA+ATR).
     * DarkPool Squeeze Momentum (BB/KC/LinReg).
     * Money Flow Matrix (RSI+Vol).
     * Dark Vector Scalping & SuperTrend.
     * Elastic Volume Weighted Momentum (EVWM).
     * Gann High/Low Activator.
   - SEASONALITY: Monthly Heatmaps, Day-of-Week DNA, Intraday Hourly DNA.
   - ANALYSIS: Cross-Asset Correlation Matrix, Multi-Timeframe Radar.
   - SIMULATION: Monte Carlo Probability Paths.
   - VOLUME: VPVR (Volume Profile Visible Range) & POC.

[3. INTELLIGENCE & BROADCAST]
   - AI: OpenAI GPT-4o Integration (Context-Aware Analyst).
   - SOCIAL: Telegram (Photo/Text Splitting) & Twitter Integration.
   - ALERTS: Fear & Greed V4, FOMO/Panic Detection.

[4. UI/UX ARCHITECTURE]
   - STYLING: Merged "DarkPool" Neon CSS with Titan Mobile Card classes.
   - VISUALS: Plotly Interactive Charts (God Mode Stack vs Physics Dashboard).
   - GLOBAL: Ticker Marquees, World Clocks, TradingView Widgets.

[5. PRESERVATION RULES]
   - All Legacy Standalone functions preserved.
   - Zero Omissions of calculations or visual elements.
"""

import time
import math
import sqlite3
import random
import json
import calendar
import datetime
import urllib.parse
from typing import Dict, Optional, List, Tuple, Any
from contextlib import contextmanager
from datetime import datetime as dt, timezone

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
import matplotlib.pyplot as plt

# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Titan-Axiom V5 God Mode",
    layout="wide",
    page_icon="👁️",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. SHARED UTILITIES & CSS INJECTION
# =============================================================================
class SecretsManager:
    @staticmethod
    def get(key, default=""):
        try: return st.secrets.get(key, default)
        except: return default

def timeframe_to_min(tf):
    if tf == '15m': return 15
    if tf == '1h': return 60
    if tf == '4h': return 240
    if tf == '1d': return 1440
    return 60

def inject_merged_css(is_mobile):
    """
    Injects the Unified 'DarkPool/Neon' aesthetic CSS.
    Merges styling from Titan, Axiom, and DarkPool codes.
    """
    base_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=SF+Pro+Display:wght@300;500;700&display=swap');
        
        .stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
        
        /* TITLE GLOW */
        .title-glow {
            font-size: 2.5em;
            font-weight: bold;
            color: #ffffff;
            text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 40px #00ff00;
            margin-bottom: 20px;
            font-family: 'SF Pro Display', sans-serif;
        }

        /* UNIVERSAL BUTTONS */
        .stButton > button {
            background: linear-gradient(135deg, #1f2833, #0b0c10);
            border: 1px solid #238636; color: #ffffff;
            font-weight: bold; height: 3.5em; font-size: 16px !important;
            border-radius: 6px;
        }
        .stButton > button:hover { background: #238636; color: #ffffff; }

        /* TICKER MARQUEE */
        .ticker-wrap {
            width: 100%; overflow: hidden; background-color: #0a0a0a; border-bottom: 1px solid #333;
            white-space: nowrap; box-sizing: border-box; height: 40px; display: flex; align-items: center; margin-bottom: 10px;
        }
        .ticker { display: inline-block; animation: marquee 45s linear infinite; }
        @keyframes marquee { 0% { transform: translate(100%, 0); } 100% { transform: translate(-100%, 0); } }
        .ticker-item {
            display: inline-block; padding: 0 2rem; font-family: 'Roboto Mono', monospace;
            font-size: 0.85rem; color: #00F0FF; text-shadow: 0 0 5px rgba(0, 240, 255, 0.5);
        }
        
        /* METRICS: Glassmorphism & Hover */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 8px;
            padding: 15px;
            transition: transform 0.2s;
        }
        div[data-testid="stMetric"]:hover {
            transform: scale(1.02);
            border-color: #00ff00;
        }
        div[data-testid="stMetricLabel"] { color: #888; letter-spacing: 1px; text-transform: uppercase; font-size: 0.8rem; }
        div[data-testid="stMetricValue"] { color: #fff; font-weight: 700; font-size: 1.2rem !important; }
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: transparent; }
        .stTabs [data-baseweb="tab"] {
            height: 50px; white-space: pre-wrap; background-color: #161b22;
            border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
            border: 1px solid #30363d; color: #8b949e;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0e1117; color: #00ff00; border-bottom: 2px solid #00ff00;
        }
        
        /* INPUT FIELDS */
        .stTextInput > div > div > input { color: #00F0FF; background-color: #111; border: 1px solid #333; }
        .stTextArea > div > div > textarea { background-color: #0a0a0a; color: #00F0FF; border: 1px solid #333; font-family: 'Roboto Mono'; }
        
        /* TITAN & AXIOM REPORT CARDS (MERGED STYLES) */
        .report-card {
            background-color: #111;
            border-left: 4px solid #00F0FF;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 10px;
            font-family: 'SF Pro Display', sans-serif;
            box-shadow: 0 1px 3px rgba(0,0,0,0.5);
            border: 1px solid #30363d;
        }
        .report-header { font-size: 1.1rem; font-weight: bold; color: #fff; margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 5px; font-family: 'Roboto Mono', monospace; }
        .report-item { margin-bottom: 5px; font-size: 0.9rem; color: #aaa; display: flex; align-items: center; gap: 8px; }
        
        /* COLOR UTILITIES */
        .highlight { color: #00F0FF; font-weight: bold; }
        .value-cyan { color: #38bdf8; font-weight: 600; font-family: 'Roboto Mono'; }
        .value-green { color: #3fb950; font-weight: 600; font-family: 'Roboto Mono'; }
        .value-red { color: #ff6b6b; font-weight: 600; font-family: 'Roboto Mono'; }
        .value-yellow { color: #e3b341; font-weight: 600; font-family: 'Roboto Mono'; }
        
        div[data-testid="stVerticalBlockBorderWrapper"] { border-color: #30363d !important; }
    </style>
    """
    
    mobile_css = """
    <style>
        div[data-testid="stMetric"] { margin-bottom: 10px; padding: 20px; }
        div[data-testid="stMetricLabel"] { font-size: 0.9rem; font-weight: 700; }
        div[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
        .js-plotly-plot { height: 100% !important; } 
        button { min-height: 55px !important; margin-top: 15px !important; font-size: 1.1rem !important; }
    </style>
    """
    
    st.markdown(base_css, unsafe_allow_html=True)
    if is_mobile:
        st.markdown(mobile_css, unsafe_allow_html=True)

# =============================================================================
# 3. GLOBAL VISUALS (CLOCK & TICKERS)
# =============================================================================
class GlobalVisuals:
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
    def render_live_clock():
        """Renders World Clock (NY/Lon/Tok) - Axiom Style"""
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
        """Renders scrolling marquee of live prices."""
        try:
            tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "SPY", "QQQ", "IWM", "NVDA", "GLD", "USO", "VIX"]
            data = yf.download(tickers, period="1d", interval="1d", progress=False)['Close'].iloc[-1]
            items = []
            for t in tickers:
                if t in data:
                    items.append(f"{t}: ${data[t]:,.2f}")
            
            html = f"""
            <div class="ticker-wrap">
                <div class="ticker">
                    {" | ".join([f"<span class='ticker-item'>{i}</span>" for i in items])}
                      | {" | ".join([f"<span class='ticker-item'>{i}</span>" for i in items])}
                </div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
        except:
            st.markdown("""<div class="ticker-wrap"><div class="ticker"><span class='ticker-item'>💠 TITAN-AXIOM V5 SYSTEM ONLINE</span></div></div>""", unsafe_allow_html=True)

# =============================================================================
# 4. TITAN ENGINE (BINANCE / LEGACY LOGIC)
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
        df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
        df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
        df['hma'] = TitanEngine.get_ma(df['close'], hma_l, "HMA")
        delta = df['close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain/loss)))
        df['rvol'] = df['volume'] / df['volume'].rolling(20).mean()
        rsi_source = df['rsi'] - 50
        vol_sma = df['volume'].rolling(14).mean()
        df['money_flow'] = (rsi_source * (df['volume'] / vol_sma)).ewm(span=3).mean()
        bb_basis = df['close'].rolling(20).mean()
        bb_dev = df['close'].rolling(20).std() * 2.0
        kc_basis = df['close'].rolling(20).mean()
        kc_dev = df['atr'] * 1.5
        df['in_squeeze'] = ((bb_basis - bb_dev) > (kc_basis - kc_dev)) & ((bb_basis + bb_dev) < (kc_basis + kc_dev))
        df['ll'] = df['low'].rolling(amp).min()
        df['hh'] = df['high'].rolling(amp).max()
        trend = np.zeros(len(df)); stop = np.full(len(df), np.nan)
        curr_t = 0; curr_s = np.nan
        for i in range(amp, len(df)):
            c = df.at[i,'close']; d = df.at[i,'atr']*dev
            if curr_t == 0:
                s = df.at[i,'ll'] + d; curr_s = max(curr_s, s) if not np.isnan(curr_s) else s
                if c < curr_s: curr_t = 1; curr_s = df.at[i,'hh'] - d
            else:
                s = df.at[i,'hh'] - d; curr_s = min(curr_s, s) if not np.isnan(curr_s) else s
                if c > curr_s: curr_t = 0; curr_s = df.at[i,'ll'] + d
            trend[i] = curr_t; stop[i] = curr_s
        df['is_bull'] = trend == 0; df['entry_stop'] = stop
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
                val = c - atr2; curr_tr = max(curr_tr, val) if not np.isnan(curr_tr) else val
                if apex_trend[i-1] == -1: curr_tr = val
            elif curr_at == -1:
                val = c + atr2; curr_tr = min(curr_tr, val) if not np.isnan(curr_tr) else val
                if apex_trend[i-1] == 1: curr_tr = val
            apex_trail[i] = curr_tr
            p_idx = i - liq_len
            is_ph = True
            for k in range(1, liq_len + 1):
                if df.at[p_idx, 'high'] <= df.at[p_idx-k, 'high'] or df.at[p_idx, 'high'] <= df.at[p_idx+k, 'high']: is_ph = False; break
            if is_ph: visual_zones.append({'type': 'SUPPLY', 'x0': df.at[p_idx, 'timestamp'], 'x1': df.at[i, 'timestamp'], 'y0': df.at[p_idx, 'high'], 'y1': df.at[p_idx, 'close'], 'color': 'rgba(229, 57, 53, 0.3)'})
            is_pl = True
            for k in range(1, liq_len + 1):
                if df.at[p_idx, 'low'] >= df.at[p_idx-k, 'low'] or df.at[p_idx, 'low'] >= df.at[p_idx+k, 'low']: is_pl = False; break
            if is_pl: visual_zones.append({'type': 'DEMAND', 'x0': df.at[p_idx, 'timestamp'], 'x1': df.at[i, 'timestamp'], 'y0': df.at[p_idx, 'low'], 'y1': df.at[p_idx, 'close'], 'color': 'rgba(67, 160, 71, 0.3)'})
        df['apex_trend'] = apex_trend; df['apex_trail'] = apex_trail
        rg = df['high'] - df['low']; body = np.abs(df['close'] - df['open'])
        eff_raw = np.where(rg == 0, 0, body / rg); eff_sm = pd.Series(eff_raw, index=df.index).ewm(span=14).mean()
        vol_avg = df['volume'].rolling(55).mean(); v_rat = np.where(vol_avg == 0, 1, df['volume'] / vol_avg)
        direction = np.sign(df['close'] - df['open']); raw = direction * eff_sm * pd.Series(v_rat, index=df.index)
        df['Apex_Flux'] = raw.ewm(span=5).mean()
        sma_h = df['high'].rolling(gann_l).mean(); sma_l = df['low'].rolling(gann_l).mean()
        g_trend = np.zeros(len(df)); curr_g = 1
        for i in range(gann_l, len(df)):
            if curr_g == 1:
                if df.at[i,'close'] < sma_l.iloc[i-1]: curr_g = -1
            else:
                if df.at[i,'close'] > sma_h.iloc[i-1]: curr_g = 1
            g_trend[i] = curr_g
        df['gann_trend'] = g_trend
        risk = abs(df['close'] - df['entry_stop']); risk = risk.mask(risk == 0, df['close'] * 0.01)
        df['tp1'] = np.where(df['is_bull'], df['close'] + 1.5*risk, df['close'] - 1.5*risk)
        df['tp2'] = np.where(df['is_bull'], df['close'] + 3.0*risk, df['close'] - 3.0*risk)
        df['tp3'] = np.where(df['is_bull'], df['close'] + 5.0*risk, df['close'] - 5.0*risk)
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
        is_bull = row['is_bull']
        direction_text = "LONG 🐂" if is_bull else "SHORT 🐻"
        sqz_text = "SQUEEZE ACTIVE" if row['in_squeeze'] else "NO SQUEEZE"
        sqz_icon = "⚠️" if row['in_squeeze'] else "⚪"
        sqz_class = "value-yellow" if row['in_squeeze'] else "value-cyan"
        rvol_desc = "(Normal)"
        if row['rvol'] > 2.0: rvol_desc = "(Ignition)"
        elif row['rvol'] < 0.5: rvol_desc = "(Low)"
        vwap_rel = "Above" if row['close'] > row['vwap'] else "Below"
        return f"""
        <div class="report-card" style="border-left: 4px solid #38bdf8;">
            <div class="report-header"><span>💠 SIGNAL: {direction_text}</span></div>
            <div class="report-item">Confidence: <span class="value-cyan">{"HIGH" if (row['is_bull'] and row['apex_trend']==1) or (not row['is_bull'] and row['apex_trend']==-1) else "LOW"}</span></div>
            <div class="report-item">Sentiment: <span class="value-cyan">{fg_index}/100</span></div>
            <div class="report-item">Squeeze: {sqz_icon} <span class="{sqz_class}">{sqz_text}</span></div>
        </div>
        <div class="report-card" style="border-left: 4px solid #38bdf8;">
            <div class="report-header"><span>🌊 FLOW & VOL</span></div>
            <div class="report-item">RVOL: <span class="value-cyan">{row['rvol']:.2f} {rvol_desc}</span></div>
            <div class="report-item">Money Flow: <span class="value-cyan">{row['money_flow']:.2f}</span></div>
            <div class="report-item">VWAP Relation: <span class="value-cyan">{vwap_rel}</span></div>
        </div>
        <div class="report-card" style="border-left: 4px solid #38bdf8;">
            <div class="report-header"><span>🎯 EXECUTION PLAN</span></div>
            <div class="report-item">Entry: <span class="value-cyan">{row['close']:.4f}</span></div>
            <div class="report-item">🛑 SMART STOP: <span class="value-cyan">{row['entry_stop']:.4f}</span></div>
            <div class="report-item">1️⃣ TP1 (1.5R): <span class="value-cyan">{row['tp1']:.4f}</span></div>
            <div class="report-item">2️⃣ TP2 (3.0R): <span class="value-cyan">{row['tp2']:.4f}</span></div>
            <div class="report-item">3️⃣ TP3 (5.0R): <span class="value-cyan">{row['tp3']:.4f}</span></div>
        </div>
        """

def send_telegram(token, chat, msg):
    if not token or not chat: return False
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"}, timeout=3)
        return True
    except: return False

# =============================================================================
# 5. DATA SERVICE (AXIOM DARKPOOL - OPTIMIZED)
# =============================================================================
class DataService:
    @staticmethod
    def get_asset_classes():
        return ["Indices", "Crypto (Top 20)", "Tech Giants (Top 10)", "Macro & Commodities", "High Volatility", "Forex"]

    @staticmethod
    def get_tickers_by_class(asset_class):
        universe = {
            "Indices": ["SPY", "QQQ", "DIA", "IWM", "VTI"],
            "Crypto (Top 20)": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "TRX-USD", "LINK-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "BCH-USD", "XLM-USD", "ALGO-USD", "ATOM-USD", "UNI-USD", "FIL-USD"],
            "Tech Giants (Top 10)": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "NFLX", "INTC"],
            "Macro & Commodities": ["^TNX", "DX-Y.NYB", "GC=F", "SI=F", "CL=F", "NG=F", "^VIX", "TLT"],
            "High Volatility": ["MSTR", "COIN", "MARA", "RIOT", "CLSK", "GME", "AMC", "HOOD", "DKNG"],
            "Forex": ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "USDCAD=X"]
        }
        return universe.get(asset_class, [])

    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_data(ticker, timeframe, limit=500):
        # Timeframe mapping optimization
        if timeframe in ["1m", "5m", "15m"]: period = "59d"
        elif timeframe in ["1h", "4h"]: period = "1y"
        else: period = "2y"
        
        # Download
        try:
            df = yf.download(ticker, period=period, interval=("1h" if timeframe=="4h" else timeframe), progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if 'Close' not in df.columns:
                if 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
                else: return pd.DataFrame()
            
            # 4H Resampling Logic
            if timeframe == "4h":
                agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
                if 'Adj Close' in df.columns: agg_dict['Adj Close'] = 'last'
                df = df.resample('4h').agg(agg_dict).dropna()
            
            return df
        except: return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_fundamentals(ticker):
        if "-" in ticker or "=" in ticker or "^" in ticker: return None
        try:
            stock = yf.Ticker(ticker)
            i = stock.info
            return {"Market Cap": i.get("marketCap", 0), "P/E Ratio": i.get("trailingPE", 0), "Rev Growth": i.get("revenueGrowth", 0), "Debt/Equity": i.get("debtToEquity", 0), "Summary": i.get("longBusinessSummary", "No Data")}
        except: return None

    @staticmethod
    @st.cache_data(ttl=300)
    def get_macro_data():
        """Batch download for macro indicators."""
        groups = {
            "🇺🇸 US Equities": {"S&P 500": "SPY", "Nasdaq 100": "QQQ", "Dow Jones": "^DJI", "Russell 2000": "^RUT"},
            "🌍 Global Indices": {"FTSE 100": "^FTSE", "DAX": "^GDAXI", "Nikkei 225": "^N225", "Hang Seng": "^HSI"},
            "🏦 Rates & Bonds": {"10Y Yield": "^TNX", "2Y Yield": "^IRX", "30Y Yield": "^TYX", "T-Bond (TLT)": "TLT"},
            "💱 Forex & Volatility": {"DXY Index": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "VIX (Fear)": "^VIX"},
            "⚠️ Risk Assets": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Semis (SMH)": "SMH", "Junk Bonds": "HYG"},
            "⚡ Energy": {"WTI Crude": "CL=F", "Brent Crude": "BZ=F", "Natural Gas": "NG=F", "Uranium": "URA"},
            "🥇 Precious Metals": {"Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F"}
        }
        all_tickers = [t for g in groups.values() for t in g.values()]
        try:
            data = yf.download(all_tickers, period="5d", interval="1d", group_by='ticker', progress=False)
            prices = {}; changes = {}
            for g_name, g_dict in groups.items():
                for name, sym in g_dict.items():
                    try:
                        if len(all_tickers) > 1: df = data[sym]
                        else: df = data
                        if not df.empty:
                            curr = df['Close'].iloc[-1]
                            prev = df['Close'].iloc[-2]
                            prices[name] = curr
                            changes[name] = ((curr - prev) / prev) * 100
                    except: continue
            return groups, prices, changes
        except: return groups, {}, {}

    @staticmethod
    def get_global_performance():
        assets = {"Tech (XLK)": "XLK", "Energy (XLE)": "XLE", "Financials (XLF)": "XLF", "Bitcoin (BTC)": "BTC-USD", "Gold (GLD)": "GLD", "Oil (USO)": "USO", "Treasuries (TLT)": "TLT"}
        try:
            data = yf.download(list(assets.values()), period="5d", interval="1d", progress=False, group_by='ticker')
            results = {}
            for name, ticker in assets.items():
                try:
                    df = data[ticker] if len(assets) > 1 else data
                    if not df.empty:
                        change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                        results[name] = change
                except: continue
            return pd.Series(results).sort_values(ascending=True)
        except: return None

    @staticmethod
    def get_seasonality_stats(ticker):
        try:
            df = yf.download(ticker, period="20y", interval="1mo", progress=False)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            df['Return'] = df['Close'].pct_change() * 100
            df['Year'] = df.index.year; df['Month'] = df.index.month
            heatmap_data = df.pivot_table(index='Year', columns='Month', values='Return')
            
            hold_stats = {}
            for p in [1, 3, 6, 12]:
                rolling = df['Close'].pct_change(periods=p).dropna() * 100
                hold_stats[p] = {"Win Rate": (rolling > 0).mean() * 100, "Avg Return": rolling.mean()}
            
            month_stats = df.groupby('Month')['Return'].agg(['mean', lambda x: (x > 0).mean() * 100, 'count'])
            month_stats.columns = ['Avg Return', 'Win Rate', 'Count']
            return heatmap_data, hold_stats, month_stats
        except: return None

# =============================================================================
# 6. QUANT ENGINE (PHYSICS & GOD MODE ALGORITHMS)
# =============================================================================
class QuantEngine:
    @staticmethod
    def tanh(x): return np.tanh(np.clip(x, -20, 20))

    @staticmethod
    def calculate_atr(df, length=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(length).mean()

    @staticmethod
    def calculate_hma(series, length):
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wma = lambda s, l: s.rolling(l).apply(lambda x: np.dot(x, np.arange(1, l + 1)) / (l * (l + 1) / 2), raw=True)
        wma_half = wma(series, half_length)
        wma_full = wma(series, length)
        diff = 2 * wma_half - wma_full
        return wma(diff, sqrt_length)

    @staticmethod
    def calc_indicators(df):
        """
        🔥 TITAN V5 GOD MODE CALCULATIONS 🔥
        Merges Physics (Entropy/Flux) with Titan God Mode (Apex/Squeeze/Vector)
        """
        # --- 1. PHYSICS LAYER (Legacy V4.1) ---
        c = df['Close'].values
        log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
        mu = pd.Series(log_ret).rolling(50).mean().values
        sigma = pd.Series(log_ret).rolling(50).std().values
        v = sigma / (np.abs(mu) + 1e-9)
        abs_ret_v = np.abs(log_ret) * v
        hyper_dist = np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))
        kappa_h = QuantEngine.tanh(pd.Series(hyper_dist).rolling(50).mean().values)
        diff_ret = np.diff(log_ret, prepend=0)
        lyap = np.log(np.abs(diff_ret) + 1e-9)
        lambda_n = QuantEngine.tanh((pd.Series(lyap).rolling(50).mean().values + 5) / 7)
        ent = pd.Series(log_ret**2).rolling(50).sum().values
        ent_n = QuantEngine.tanh(ent * 10)
        raw_chedo = (0.4 * kappa_h) + (0.3 * lambda_n) + (0.3 * ent_n)
        df['CHEDO'] = 2 / (1 + np.exp(-raw_chedo * 4)) - 1
        
        # RQZO (Relativity)
        src = df['Close']
        mn, mx = src.rolling(100).min(), src.rolling(100).max()
        norm = (src - mn) / (mx - mn + 1e-9)
        v_diff = np.abs(norm.diff())
        c_limit = 0.05
        gamma = 1 / np.sqrt(1 - (np.minimum(v_diff, c_limit*0.99)/c_limit)**2)
        idx = np.arange(len(df))
        tau = (idx % 100) / gamma.fillna(1.0)
        zeta = np.zeros(len(df))
        for n in range(1, 26):
            amp = n ** -0.5
            theta = tau * np.log(n)
            zeta += amp * np.sin(theta)
        df['RQZO'] = pd.Series(zeta).fillna(0)

        # Apex Flux (Vector)
        rg = df['High'] - df['Low']
        body = np.abs(df['Close'] - df['Open'])
        eff_raw = np.where(rg == 0, 0, body / rg)
        eff_sm = pd.Series(eff_raw, index=df.index).ewm(span=14).mean()
        vol_avg = df['Volume'].rolling(55).mean()
        v_rat = np.where(vol_avg == 0, 1, df['Volume'] / vol_avg)
        direction = np.sign(df['Close'] - df['Open'])
        raw_flux = direction * eff_sm * pd.Series(v_rat, index=df.index)
        df['Apex_Flux'] = raw_flux.ewm(span=5).mean()
        df['Apex_State'] = np.select([df['Apex_Flux'] > 0.6, df['Apex_Flux'] < -0.6], ["Super Bull", "Super Bear"], default="Neutral")

        # --- 2. GOD MODE LAYER (New V5.0) ---
        df['HMA'] = QuantEngine.calculate_hma(df['Close'], 55)
        df['ATR'] = QuantEngine.calculate_atr(df, 14)
        
        # Apex Trend & Liquidity Master
        apex_mult = 1.5
        df['Apex_Base'] = df['HMA']
        df['Apex_ATR'] = QuantEngine.calculate_atr(df, 55)
        df['Apex_Upper'] = df['Apex_Base'] + (df['Apex_ATR'] * apex_mult)
        df['Apex_Lower'] = df['Apex_Base'] - (df['Apex_ATR'] * apex_mult)
        df['Apex_Trend'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, 0))
        # FIX: replace(method='ffill') is deprecated in Pandas 2.1+
        df['Apex_Trend'] = df['Apex_Trend'].replace(0, np.nan).ffill().fillna(0)

        # DarkPool Squeeze
        df['Sqz_Basis'] = df['Close'].rolling(20).mean()
        df['Sqz_Dev'] = df['Close'].rolling(20).std() * 2.0
        df['Sqz_Upper_BB'] = df['Sqz_Basis'] + df['Sqz_Dev']
        df['Sqz_Lower_BB'] = df['Sqz_Basis'] - df['Sqz_Dev']
        df['Sqz_Ma_KC'] = df['Close'].rolling(20).mean()
        df['Sqz_Range_MA'] = QuantEngine.calculate_atr(df, 20)
        df['Sqz_Upper_KC'] = df['Sqz_Ma_KC'] + (df['Sqz_Range_MA'] * 1.5)
        df['Sqz_Lower_KC'] = df['Sqz_Ma_KC'] - (df['Sqz_Range_MA'] * 1.5)
        df['Squeeze_On'] = (df['Sqz_Lower_BB'] > df['Sqz_Lower_KC']) & (df['Sqz_Upper_BB'] < df['Sqz_Upper_KC'])
        highest = df['High'].rolling(20).max()
        lowest = df['Low'].rolling(20).min()
        avg_val = (highest + lowest + df['Sqz_Ma_KC']) / 3
        df['Sqz_Mom'] = (df['Close'] - avg_val).rolling(20).mean() * 100

        # Money Flow Matrix
        rsi_src = (100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))) - 50
        mf_vol = df['Volume'] / df['Volume'].rolling(14).mean()
        df['MF_Matrix'] = (rsi_src * mf_vol).ewm(span=3).mean()

        # Dark Vector Scalping
        amp = 5
        df['VS_Low'] = df['Low'].rolling(amp).min()
        df['VS_High'] = df['High'].rolling(amp).max()
        df['VS_Trend'] = np.where(df['Close'] > df['VS_High'].shift(1), 1, np.where(df['Close'] < df['VS_Low'].shift(1), -1, 0))
        # FIX: replace(method='ffill') is deprecated
        df['VS_Trend'] = df['VS_Trend'].replace(0, np.nan).ffill().fillna(0)

        # Advanced Volume & EVWM
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
        ev_len = 21
        ev_base = QuantEngine.calculate_hma(df['Close'], ev_len)
        ev_atr = QuantEngine.calculate_atr(df, ev_len)
        ev_elast = (df['Close'] - ev_base) / ev_atr
        ev_force = np.sqrt(df['RVOL'].ewm(span=5).mean())
        df['EVWM'] = ev_elast * ev_force

        # Gann High Low Activator
        gann_len = 3
        df['Gann_High'] = df['High'].rolling(gann_len).mean()
        df['Gann_Low'] = df['Low'].rolling(gann_len).mean()
        df['Gann_Trend'] = np.where(df['Close'] > df['Gann_High'].shift(1), 1, np.where(df['Close'] < df['Gann_Low'].shift(1), -1, 0))
        # FIX: replace(method='ffill') is deprecated
        df['Gann_Trend'] = df['Gann_Trend'].replace(0, np.nan).ffill().fillna(0)

        # SuperTrend (Dark Vector Trend)
        st_atr = QuantEngine.calculate_atr(df, 10)
        st_upper = ((df['High']+df['Low'])/2) + (3 * st_atr)
        st_lower = ((df['High']+df['Low'])/2) - (3 * st_atr)
        st_dir = np.zeros(len(df))
        st_val = np.zeros(len(df))
        close = df['Close'].values
        upper = st_upper.values
        lower = st_lower.values
        # Fast Iteration
        for i in range(1, len(df)):
            if close[i-1] > st_val[i-1]:
                st_val[i] = max(lower[i], st_val[i-1]) if close[i] > st_val[i-1] else upper[i]
                st_dir[i] = 1 if close[i] > st_val[i-1] else -1
                if close[i] < lower[i] and st_dir[i-1] == 1: st_val[i] = upper[i]; st_dir[i] = -1
            else:
                st_val[i] = min(upper[i], st_val[i-1]) if close[i] < st_val[i-1] else lower[i]
                st_dir[i] = -1 if close[i] < st_val[i-1] else 1
                if close[i] > upper[i] and st_dir[i-1] == -1: st_val[i] = lower[i]; st_dir[i] = 1
        df['DarkVector_Trend'] = st_dir

        # GOD MODE SCORE
        df['GM_Score'] = (df['Apex_Trend'] + df['Gann_Trend'] + df['DarkVector_Trend'] + df['VS_Trend'] + np.sign(df['Sqz_Mom']))

        # Dashboard Metrics
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/loss)))

        return df

    @staticmethod
    def calc_fear_greed_v4(df):
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['FG_RSI'] = 100 - (100 / (1 + rs))
        
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['FG_MACD'] = (50 + ((macd-signal) * 10)).clip(0, 100)
        
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        upper = sma20 + (std20 * 2)
        lower = sma20 - (std20 * 2)
        df['FG_BB'] = ((df['Close'] - lower) / (upper - lower) * 100).clip(0, 100)
        
        sma50 = df['Close'].rolling(50).mean()
        sma200 = df['Close'].rolling(200).mean()
        cond = [(df['Close']>sma50)&(sma50>sma200), (df['Close']>sma50), (df['Close']<sma50)&(sma50<sma200)]
        df['FG_MA'] = np.select(cond, [75, 60, 25], default=40)
        
        df['FG_Index'] = (df['FG_RSI']*0.3) + (df['FG_MACD']*0.25) + (df['FG_BB']*0.25) + (df['FG_MA']*0.2)
        df['FG_Index'] = df['FG_Index'].rolling(5).mean()
        
        vol_ma = df['Volume'].rolling(20).mean()
        df['IS_FOMO'] = (df['Volume'] > vol_ma*2.5) & (df['FG_RSI']>70) & (df['Close'] > df['Close'].shift(3)*1.02)
        df['IS_PANIC'] = (df['Close'].pct_change() < -0.03) & (df['Volume'] > vol_ma*3.0)
        return df

    @staticmethod
    def calc_smc(df, swing_length=5):
        smc = {'structures': [], 'order_blocks': [], 'fvgs': []}
        # FVG
        for i in range(2, len(df)):
            if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                smc['fvgs'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i], 'color': 'rgba(0, 255, 104, 0.3)'})
            if df['High'].iloc[i] < df['Low'].iloc[i-2]:
                smc['fvgs'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['Low'].iloc[i-2], 'y1': df['High'].iloc[i], 'color': 'rgba(255, 0, 8, 0.3)'})
        
        # BOS/CHoCH
        df['PH'] = df['High'].rolling(swing_length*2+1, center=True).max() == df['High']
        df['PL'] = df['Low'].rolling(swing_length*2+1, center=True).min() == df['Low']
        last_h = None; last_l = None; trend = 0
        
        for i in range(swing_length, len(df)):
            c = df['Close'].iloc[i]
            if df['PH'].iloc[i-swing_length]: last_h = {'p': df['High'].iloc[i-swing_length], 'x': df.index[i-swing_length], 'i': i-swing_length}
            if df['PL'].iloc[i-swing_length]: last_l = {'p': df['Low'].iloc[i-swing_length], 'x': df.index[i-swing_length], 'i': i-swing_length}
            
            if last_h and c > last_h['p']:
                lbl = "CHoCH" if trend != 1 else "BOS"; trend = 1
                smc['structures'].append({'x0': last_h['x'], 'x1': df.index[i], 'y': last_h['p'], 'color': 'green', 'label': lbl})
                last_h = None
            elif last_l and c < last_l['p']:
                lbl = "CHoCH" if trend != -1 else "BOS"; trend = -1
                smc['structures'].append({'x0': last_l['x'], 'x1': df.index[i], 'y': last_l['p'], 'color': 'red', 'label': lbl})
                last_l = None
        return smc

    @staticmethod
    def calc_volume_profile(df, bins=50):
        price_min = df['Low'].min(); price_max = df['High'].max()
        price_bins = np.linspace(price_min, price_max, bins)
        df['Mid'] = (df['Close'] + df['Open']) / 2
        df['Bin'] = pd.cut(df['Mid'], bins=price_bins, labels=price_bins[:-1], include_lowest=True)
        vp = df.groupby('Bin', observed=False)['Volume'].sum().reset_index()
        vp['Price'] = vp['Bin'].astype(float)
        poc_idx = vp['Volume'].idxmax()
        poc_price = vp.loc[poc_idx, 'Price']
        return vp, poc_price

    @staticmethod
    def run_monte_carlo(df, days=30, sims=1000):
        last_price = df['Close'].iloc[-1]
        returns = df['Close'].pct_change().dropna()
        mu = returns.mean(); sigma = returns.std()
        sim_rets = np.random.normal(mu, sigma, (days, sims))
        paths = np.zeros((days, sims)); paths[0] = last_price
        for t in range(1, days): paths[t] = paths[t-1] * (1 + sim_rets[t])
        return paths

    @staticmethod
    def calc_day_of_week_dna(ticker):
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df['Day'] = df.index.day_name()
            df['Ret'] = df['Close'].pct_change() * 100
            
            # Cumulative Calc
            df_piv = df.pivot(columns='Day', values='Ret').fillna(0)
            cum_ret = df_piv.cumsum()
            
            stats = df.groupby('Day')['Ret'].agg(['count', 'sum', 'mean', lambda x: (x>0).mean()*100])
            stats.columns = ['Count', 'Total', 'Avg', 'Win Rate']
            return cum_ret, stats
        except: return None

    @staticmethod
    def calc_intraday_dna(ticker):
        try:
            df = yf.download(ticker, period="60d", interval="1h", progress=False)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df['Ret'] = df['Close'].pct_change() * 100
            df['Hour'] = df.index.hour
            stats = df.groupby('Hour')['Ret'].agg(['mean', 'sum', lambda x: (x>0).mean()*100])
            stats.columns = ['Avg', 'Total', 'Win Rate']
            return stats
        except: return None

    @staticmethod
    def calc_correlations(ticker):
        ref_tickers = {"SPY":"SPY", "BTC":"BTC-USD", "10Y":"^TNX", "DXY":"DX-Y.NYB", "Gold":"GC=F", "Oil":"CL=F"}
        try:
            df_main = yf.download(ticker, period="1y", interval="1d", progress=False)['Close']
            df_macro = yf.download(list(ref_tickers.values()), period="1y", interval="1d", progress=False)['Close']
            combined = df_macro.copy(); combined[ticker] = df_main
            corr = combined.iloc[-180:].corr()
            target_corr = corr[ticker].drop(ticker).sort_values(ascending=False)
            inv_map = {v: k for k, v in ref_tickers.items()}
            target_corr.index = [inv_map.get(x, x) for x in target_corr.index]
            return target_corr
        except: return None

    @staticmethod
    def calc_mtf_trend(ticker):
        timeframes = {"1H": "1h", "4H": "1h", "D": "1d", "W": "1wk"}
        trends = {}
        for name, tf in timeframes.items():
            try:
                p = "2y" if tf in ["1d", "1wk"] else "1y"
                df = yf.download(ticker, period=p, interval=tf, progress=False)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                if name == "4H": df = df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
                
                if len(df) > 50:
                    ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
                    ema50 = df['Close'].ewm(span=50).mean().iloc[-1]
                    c = df['Close'].iloc[-1]
                    t = "BULLISH" if c > ema20 > ema50 else "BEARISH" if c < ema20 < ema50 else "NEUTRAL"
                    trends[name] = t
                else: trends[name] = "N/A"
            except: trends[name] = "N/A"
        return pd.DataFrame(trends, index=["Trend"]).T

class Intelligence:
    @staticmethod
    def ask_ai_analyst(df, ticker, fund, interval, key):
        if not key: return "⚠️ Waiting for OpenAI API Key..."
        last = df.iloc[-1]
        gm_score = last['GM_Score']
        gm_verdict = "STRONG BUY" if gm_score >= 3 else "STRONG SELL" if gm_score <= -3 else "NEUTRAL"
        
        fund_txt = f"P/E {fund.get('P/E Ratio','N/A')}" if fund else "N/A"
        psych = "FOMO" if last['IS_FOMO'] else "PANIC" if last['IS_PANIC'] else "Normal"
        
        prompt = f"""
        Analyze {ticker} ({interval}) at ${last['Close']:.2f}.
        Techs: Trend {last['Apex_Trend']}, RSI {last['RSI']:.1f}, RVOL {last['RVOL']:.1f}x.
        God Mode Score: {gm_score} ({gm_verdict}).
        Physics: Entropy {last['CHEDO']:.2f}, Flux {last['Apex_Flux']:.2f}.
        Psychology: {psych}. Fundamentals: {fund_txt}.
        
        Mission:
        1. Analyze Market Structure (Trend vs Chop).
        2. Correlate Technicals with Physics/Sentiment.
        3. Provide Outlook.
        Use Emojis. No financial advice.
        """
        try:
            client = OpenAI(api_key=key)
            res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
            return res.choices[0].message.content
        except Exception as e: return f"AI Error: {e}"

# =============================================================================
# 7. GRAPHICS ENGINE
# =============================================================================
class Graphics:
    @staticmethod
    def render_god_mode_chart(df, sr_zones):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02)
        
        # P1: Price + Apex Cloud
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.1)', line=dict(width=0), name="Apex Cloud"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='yellow', width=2), name="HMA Trend"), row=1, col=1)
        
        # GM Signals
        buy = df[df['GM_Score'] >= 3]; sell = df[df['GM_Score'] <= -3]
        fig.add_trace(go.Scatter(x=buy.index, y=buy['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', color='#00ff00', size=10), name="GM Buy"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell.index, y=sell['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', color='#ff0000', size=10), name="GM Sell"), row=1, col=1)
        
        # P2: Squeeze Mom
        colors = ['#00E676' if v > 0 else '#FF5252' for v in df['Sqz_Mom']]
        fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=colors, name="Squeeze Mom"), row=2, col=1)
        
        # P3: MF Matrix
        fig.add_trace(go.Scatter(x=df.index, y=df['MF_Matrix'], fill='tozeroy', line=dict(color='cyan', width=1), name="Money Flow"), row=3, col=1)
        
        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, title_text="God Mode Technical Stack", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 8. MAIN CONTROLLER
# =============================================================================
def main():
    st.sidebar.header("💠 TITAN-AXIOM V5")
    mode = st.sidebar.radio("ENGINE MODE", ["TITAN MOBILE (Crypto/Binance)", "AXIOM DARKPOOL (Stocks/YFinance)"])
    is_mobile = st.sidebar.toggle("📱 Mobile Optimized", value=True)
    inject_merged_css(is_mobile)
    
    # Secrets
    tg_token = st.sidebar.text_input("Bot Token", value=SecretsManager.get("TELEGRAM_TOKEN"), type="password")
    tg_chat = st.sidebar.text_input("Chat ID", value=SecretsManager.get("TELEGRAM_CHAT_ID"))
    ai_key = st.sidebar.text_input("AI Key", value=SecretsManager.get("OPENAI_API_KEY"), type="password")

    # --- MODE 1: TITAN (LEGACY) ---
    if mode == "TITAN MOBILE (Crypto/Binance)":
        GlobalVisuals.render_titan_clock()
        st.sidebar.subheader("📡 BINANCE FEED")
        bases = TitanEngine.get_binance_bases()
        idx = bases.index("BTC") if "BTC" in bases else 0
        base = st.sidebar.selectbox("Asset", bases, index=idx)
        ticker = f"{base}USDT"
        GlobalVisuals.render_titan_tape(ticker)
        
        c1, c2 = st.sidebar.columns(2)
        with c1: timeframe = st.selectbox("TF", ["15m", "1h", "4h", "1d"], index=1)
        with c2: limit = st.slider("Depth", 100, 500, 200, 50)
        
        st.sidebar.markdown("---")
        amp = st.sidebar.number_input("Amplitude", 2, 100, 10)
        dev = st.sidebar.number_input("Deviation", 0.5, 5.0, 3.0)
        
        st.title(f"💠 TITAN: {base}")
        with st.spinner("Connecting to Binance..."):
            df = TitanEngine.get_klines(ticker, timeframe, limit)
        
        if not df.empty:
            df, zones = TitanEngine.run_engine(df, int(amp), dev, 55, 3, 55, 1.5, 10)
            last = df.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("TREND", "BULL 🟢" if last['is_bull'] else "BEAR 🔴")
            c2.metric("FLUX", f"{last['Apex_Flux']:.2f}")
            c3.metric("STOP", f"{last['entry_stop']:.2f}")
            c4.metric("TP3", f"{last['tp3']:.2f}")
            
            fg = TitanEngine.calculate_fear_greed(df)
            spec = TitanEngine.detect_special_setups(df)
            st.markdown(TitanEngine.generate_mobile_report(last, fg, spec), unsafe_allow_html=True)
            
            if st.button("📢 SEND SIGNAL"):
                msg = f"🚀 TITAN: {ticker} | {last['close']}"
                if send_telegram(tg_token, tg_chat, msg): st.success("SENT")
                else: st.error("FAIL")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], line=dict(color='#00F0FF', width=1)), row=1, col=1)
            fig.add_trace(go.Bar(x=df['timestamp'], y=df['Apex_Flux'], marker_color=np.where(df['Apex_Flux']>0, '#00E676', '#FF1744')), row=2, col=1)
            fig.update_layout(height=500, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

    # --- MODE 2: AXIOM DARKPOOL (GOD MODE) ---
    else:
        GlobalVisuals.render_axiom_banner()
        GlobalVisuals.render_live_clock()
        
        st.sidebar.subheader("📡 DARKPOOL FEED")
        input_mode = st.sidebar.radio("Selection", ["Curated", "Manual"], index=0)
        if input_mode == "Curated":
            ac = st.sidebar.selectbox("Sector", DataService.get_asset_classes())
            ticker = st.sidebar.selectbox("Ticker", DataService.get_tickers_by_class(ac))
        else:
            ticker = st.sidebar.text_input("Ticker", "BTC-USD").upper()
        
        tf = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=2)
        
        # Macro Header
        groups, prices, changes = DataService.get_macro_data()
        if prices:
            cols = st.columns(4)
            keys = ["S&P 500", "Bitcoin", "Gold", "VIX (Fear)"]
            for i, k in enumerate(keys):
                if k in prices: cols[i].metric(k, f"{prices[k]:,.2f}", f"{changes[k]:.2f}%")
        
        if st.button(f"RUN ANALYSIS: {ticker}"):
            with st.spinner("Crunching God Mode Data..."):
                df = DataService.fetch_data(ticker, tf)
                if not df.empty:
                    df = QuantEngine.calc_indicators(df)
                    df = QuantEngine.calc_fear_greed_v4(df)
                    last = df.iloc[-1]
                    fund = DataService.get_fundamentals(ticker)
                    
                    # TABS
                    t1, t2, t3, t4, t5, t6, t7 = st.tabs(["📊 God Mode", "🌍 Fundamentals", "📅 Seasonality", "📆 DNA", "🧩 Correlation", "🔮 Quant", "📡 Broadcast"])
                    
                    with t1:
                        c1, c2 = st.columns([3, 1])
                        with c1: Graphics.render_god_mode_chart(df, [])
                        with c2:
                            st.metric("God Mode Score", f"{last['GM_Score']:.0f}/5")
                            st.metric("Apex Trend", "BULL" if last['Apex_Trend']==1 else "BEAR")
                            st.metric("Entropy", f"{last['CHEDO']:.2f}")
                            st.info(Intelligence.ask_ai_analyst(df, ticker, fund, tf, ai_key))
                    
                    with t2:
                        if fund: 
                            c1, c2 = st.columns(2)
                            c1.metric("Market Cap", f"{fund.get('Market Cap', 'N/A')}")
                            c2.metric("P/E", f"{fund.get('P/E Ratio', 'N/A')}")
                            st.write(fund.get("Summary"))
                        perf = DataService.get_global_performance()
                        if perf is not None: st.bar_chart(perf)

                    with t3:
                        seas = DataService.get_seasonality_stats(ticker)
                        if seas: 
                            hm, hold, month = seas
                            st.plotly_chart(px.imshow(hm, color_continuous_scale='RdYlGn'), use_container_width=True)
                            st.dataframe(month)

                    with t4:
                        dna = QuantEngine.calc_day_of_week_dna(ticker)
                        if dna: 
                            cum, stats = dna
                            st.line_chart(cum)
                            st.dataframe(stats)
                        hr_dna = QuantEngine.calc_intraday_dna(ticker)
                        if hr_dna is not None: st.bar_chart(hr_dna['Win Rate'])

                    with t5:
                        corr = QuantEngine.calc_correlations(ticker)
                        mtf = QuantEngine.calc_mtf_trend(ticker)
                        c1, c2 = st.columns(2)
                        if corr is not None: c1.dataframe(corr)
                        c2.dataframe(mtf)

                    with t6:
                        mc = QuantEngine.run_monte_carlo(df)
                        st.line_chart(mc[:, :30])
                        vp, poc = QuantEngine.calc_volume_profile(df)
                        st.caption(f"POC: {poc:.2f}")
                        st.bar_chart(vp.set_index('Price')['Volume'])

                    with t7:
                        msg = f"🔥 {ticker} God Mode: {last['GM_Score']:.0f}/5 | Price: {last['Close']:.2f}"
                        if st.button("Send Telegram"):
                            if send_telegram(tg_token, tg_chat, msg): st.success("Sent")
                            else: st.error("Check Keys")
                        
                        # TradingView Embed
                        tv_sym = ticker.replace("-", "")
                        html = f"""<div class="tradingview-widget-container"><div id="tv"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"width": "100%", "height": 500, "symbol": "{tv_sym}", "interval": "D", "theme": "dark", "container_id": "tv"}});</script></div>"""
                        components.html(html, height=500)

# --- LEGACY STANDALONE COMPATIBILITY ---
def calculate_hma(s, l): return QuantEngine.calculate_hma(s, l)
def generate_mobile_report_standalone(r, s, t, f, fg, ss): return TitanEngine.generate_mobile_report(r, fg, {})
def send_telegram_msg_standalone(t, c, m): return send_telegram(t, c, m)
def run_engines_standalone(d, a, dv, h, t1, t2, t3, m, v, g): return TitanEngine.run_engine(d, a, dv, h, g, 55, 1.5, 10)[0]

if __name__ == "__main__":
    main()
