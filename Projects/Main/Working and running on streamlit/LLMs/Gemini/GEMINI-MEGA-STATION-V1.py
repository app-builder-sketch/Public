# ==================================================================================================
# CONSTRAINTS WARNING (NON-NEGOTIABLE) — MUST REMAIN AT TOP OF FILE IN EVERY EDIT
# --------------------------------------------------------------------------------------------------
# 1) NO OMISSIONS. NO ASSUMPTIONS. BASE PRESERVED.
#    - Start from the latest COMPLETE code provided by the user.
#    - Keep it 100% intact: no deletions, no omissions, no placeholders (“...”), no partial snippets.
#
# 2) FULL SCRIPT OUTPUT — ALWAYS
#    - Any change requires outputting the ENTIRE updated script(s), not fragments or diffs.
#
# 3) CONTINUITY + CONFLICTS
#    - Never remove features unless explicitly instructed.
#    - If a new request conflicts with existing behavior: implement behind a toggle OR preserve both,
#      and document conflicts explicitly.
#
# 4) SECRETS + SECURITY
#    - Load secrets from st.secrets first, env fallback: OPENAI_API_KEY, GEMINI_API_KEY,
#      TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.
#    - Never print or log secrets.
#
# 5) ALWAYS SUGGEST IMPROVEMENTS
#    - End every response with “Next Upgrade Options” unless truly finished.
# ==================================================================================================

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
import datetime
import requests
import urllib.parse
from scipy.stats import linregress
import sqlite3
import json
import io
import time
import threading
import websocket
import math
from typing import Dict, Optional, List, Tuple, Any
import streamlit.components.v1 as components

# ==========================================
# 1. PAGE CONFIGURATION & DATABASE INIT
# ==========================================
st.set_page_config(layout="wide", page_title="Omni-Sentient Titan Terminal v5.1", page_icon="💠")

def init_db():
    """Initializes the SQLite database for signals and watchlists."""
    conn = sqlite3.connect('titan_vault.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS signals 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp DATETIME, 
                  symbol TEXT, 
                  interval TEXT, 
                  score REAL, 
                  price REAL, 
                  message TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist 
                 (symbol TEXT PRIMARY KEY)''')
    conn.commit()
    conn.close()

init_db()

# --- PERSISTENT STATE ---
if 'whale_data' not in st.session_state:
    st.session_state.whale_data = []
if 'last_ai_summary' not in st.session_state:
    st.session_state.last_ai_summary = "Awaiting institutional analysis..."

# --- CUSTOM CSS (HYBRID STYLE) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=SF+Pro+Display:wght@300;500;700&display=swap');

    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'SF Pro Display', sans-serif; }
    
    /* TITAN STYLE ELEMENTS */
    .titan-metric { background: rgba(31, 40, 51, 0.9); border: 1px solid #45a29e; padding: 10px; border-radius: 8px; }
    .title-glow { font-size: 3em; font-weight: bold; color: #ffffff; text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00; margin-bottom: 20px; font-family: 'Roboto Mono'; }
    
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

    /* AXIOM TICKER MARQUEE */
    .ticker-wrap {
        width: 100%; overflow: hidden; background-color: #0d1117; border-bottom: 1px solid #30363d;
        height: 40px; display: flex; align-items: center; margin-bottom: 15px;
    }
    .ticker { display: inline-block; animation: marquee 45s linear infinite; white-space: nowrap; }
    @keyframes marquee { 0% { transform: translate(100%, 0); } 100% { transform: translate(-100%, 0); } }
    .ticker-item { padding: 0 2rem; font-family: 'Roboto Mono'; font-size: 0.85rem; color: #58a6ff; }

    /* REPORT CARDS */
    .report-card { 
        background-color: #161b22; 
        border-left: 4px solid #3fb950; 
        padding: 16px; 
        border-radius: 6px; 
        margin-bottom: 16px; 
        border: 1px solid #30363d;
    }
    .report-header { 
        font-size: 1.1rem; font-weight: 700; color: #f0f6fc; margin-bottom: 12px; 
        border-bottom: 1px solid #30363d; padding-bottom: 8px; font-family: 'Roboto Mono';
    }
    .report-item { margin-bottom: 8px; font-size: 0.95rem; color: #8b949e; display: flex; gap: 8px; }
    .value-cyan { color: #38bdf8; font-weight: 600; font-family: 'Roboto Mono'; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SHARED UTILITIES & SECRETS
# ==========================================
class SecretsManager:
    @staticmethod
    def get(key, default=""):
        try:
            if key in st.secrets: return st.secrets[key]
            return default
        except: return default

# API Key Management
if 'api_key' not in st.session_state: st.session_state.api_key = None
st.session_state.api_key = SecretsManager.get("OPENAI_API_KEY", st.session_state.api_key)

# ==========================================
# 3. WHALE TRACKER ENGINE (WEBSOCKET)
# ==========================================
class WhaleTracker:
    def __init__(self, symbol):
        self.symbol = symbol.lower().replace("-usd", "usdt")
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@aggTrade"
        self.ws = None
        self.thread = None

    def on_message(self, ws, message):
        data = json.loads(message)
        qty = float(data['q'])
        price = float(data['p'])
        usd_val = qty * price
        if usd_val >= 100000: 
            side = "BUY" if not data['m'] else "SELL"
            whale_event = {
                "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                "Price": f"{price:.2f}",
                "Value": f"${usd_val:,.0f}",
                "Side": side,
                "RawVal": usd_val
            }
            st.session_state.whale_data.append(whale_event)
            if len(st.session_state.whale_data) > 30: st.session_state.whale_data.pop(0)

    def start(self):
        # NOTE: websocket.WebSocketApp comes from the websocket-client library
        self.ws = websocket.WebSocketApp(self.ws_url, on_message=self.on_message)
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()

# ==========================================
# 4. TITAN ENGINE (BINANCE / SCALPING / SMC)
# ==========================================
BINANCE_API_BASE = "https://api.binance.us/api/v3"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

class TitanEngine:
    @staticmethod
    def get_ma(series, length, ma_type="HMA"):
        if ma_type == "SMA": return series.rolling(length).mean()
        elif ma_type == "EMA": return series.ewm(span=length, adjust=False).mean()
        else: # HMA
            half_len = int(length / 2)
            sqrt_len = int(math.sqrt(length))
            wma_f = series.rolling(length).mean()
            wma_h = series.rolling(half_len).mean()
            diff = 2 * wma_h - wma_f
            return diff.rolling(sqrt_len).mean()

    @staticmethod
    def calc_volumetric_delta(df):
        range_total = (df['high'] - df['low']).replace(0, 0.0001)
        buy_vol = ((df['close'] - df['low']) / range_total) * df['volume']
        df['Net_Delta'] = buy_vol - (df['volume'] - buy_vol)
        df['Volume_Bias_Pct'] = (df['Net_Delta'] / df['volume']) * 100
        return df

    @staticmethod
    @st.cache_data(ttl=5)
    def get_klines(symbol, interval, limit=200):
        try:
            r = requests.get(f"{BINANCE_API_BASE}/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, headers=HEADERS, timeout=5)
            if r.status_code == 200:
                df = pd.DataFrame(r.json(), columns=['t','o','h','l','c','v','T','q','n','V','Q','B'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
                return df[['timestamp','open','high','low','close','volume']]
        except: return pd.DataFrame()

    @staticmethod
    def run_engine(df, amp=10, dev=3.0, hma_l=50, gann_l=3):
        if df.empty: return df, []
        df = df.copy().reset_index(drop=True)

        # Base Indicators
        df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
        df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
        df['hma'] = TitanEngine.get_ma(df['close'], hma_l, "HMA")
        
        # Volumetrics
        df = TitanEngine.calc_volumetric_delta(df)
        
        # Momentum & Flow
        delta = df['close'].diff()
        gain, loss = delta.clip(lower=0).ewm(alpha=1/14).mean(), -delta.clip(upper=0).ewm(alpha=1/14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain/loss)))
        df['rvol'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Squeeze
        bb_basis = df['close'].rolling(20).mean()
        bb_dev = df['close'].rolling(20).std() * 2.0
        kc_dev = df['atr'] * 1.5
        df['in_squeeze'] = ((bb_basis - bb_dev) > (bb_basis - kc_dev)) & ((bb_basis + bb_dev) < (bb_basis + kc_dev))

        # Titan Trend logic
        df['ll'], df['hh'] = df['low'].rolling(amp).min(), df['high'].rolling(amp).max()
        trend, stop = np.zeros(len(df)), np.full(len(df), np.nan)
        curr_t, curr_s = 0, np.nan
        for i in range(amp, len(df)):
            c, d = df.at[i,'close'], df.at[i,'atr']*dev
            if curr_t == 0:
                s = df.at[i,'ll'] + d
                curr_s = max(curr_s, s) if not np.isnan(curr_s) else s
                if c < curr_s: curr_t = 1; curr_s = df.at[i,'hh'] - d
            else:
                s = df.at[i,'hh'] - d
                curr_s = min(curr_s, s) if not np.isnan(curr_s) else s
                if c > curr_s: curr_t = 0; curr_s = df.at[i,'ll'] + d
            trend[i], stop[i] = curr_t, curr_s
        df['is_bull'], df['entry_stop'] = trend == 0, stop

        # Gann & Flux
        rg, body = df['high'] - df['low'], np.abs(df['close'] - df['open'])
        eff = np.where(rg == 0, 0, body / rg)
        df['Apex_Flux'] = (pd.Series(eff).ewm(span=14).mean() * np.sign(df['close']-df['open'])).ewm(span=5).mean()
        
        # GM Score
        df['GM_Score'] = np.where(df['is_bull'], 1, -1) + np.where(df['rsi']>50,1,-1) + np.sign(df['Apex_Flux'])
        
        # SMC Zones
        zones = []
        for i in range(len(df)-10, len(df)):
             if df.at[i, 'high'] == df['high'].rolling(10, center=True).max().iloc[i]:
                 zones.append({'x0': df.at[i-5,'timestamp'], 'x1': df.at[i,'timestamp'], 'y0': df.at[i,'high'], 'y1': df.at[i,'close'], 'color': 'rgba(255,0,0,0.2)'})
        
        # Risk Levels
        risk = abs(df['close'] - df['entry_stop']).mask(df['close'] == df['entry_stop'], df['close']*0.01)
        df['tp1'] = np.where(df['is_bull'], df['close'] + 1.5*risk, df['close'] - 1.5*risk)
        df['tp2'] = np.where(df['is_bull'], df['close'] + 3.0*risk, df['close'] - 3.0*risk)
        df['tp3'] = np.where(df['is_bull'], df['close'] + 5.0*risk, df['close'] - 5.0*risk)

        return df, zones

# ==========================================
# 5. AXIOM ENGINE (YFINANCE / PHYSICS)
# ==========================================
class AxiomEngine:
    @staticmethod
    def fetch_data(ticker, timeframe, limit=500):
        tf_map = {"15m": "1mo", "1h": "6mo", "4h": "1y", "1d": "2y", "1wk": "5y"}
        try:
            df = yf.download(ticker, period=tf_map.get(timeframe, "1y"), interval=timeframe, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={c: c.capitalize() for c in df.columns})
            return df.dropna().tail(limit)
        except: return pd.DataFrame()

    @staticmethod
    def calc_chedo(df, length=50):
        c = df['Close'].values
        log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
        mu = pd.Series(log_ret).rolling(length).mean().values
        sigma = pd.Series(log_ret).rolling(length).std().values
        v = sigma / (np.abs(mu) + 1e-9)
        abs_ret_v = np.abs(log_ret) * v
        kappa_h = np.tanh(pd.Series(np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))).rolling(length).mean().values)
        df['CHEDO'] = kappa_h
        return df

    @staticmethod
    def calc_rqzo(df, harmonics=25):
        src = df['Close']
        mn, mx = src.rolling(100).min(), src.rolling(100).max()
        norm = (src - mn) / (mx - mn + 1e-9)
        v = np.abs(norm.diff())
        gamma = 1 / np.sqrt(1 - (np.minimum(v, 0.049)/0.05)**2)
        zeta = np.zeros(len(df))
        tau = (np.arange(len(df)) % 100) / gamma.fillna(1.0)
        for n in range(1, harmonics + 1):
            zeta += (n**-0.5) * np.sin(tau * np.log(n))
        df['RQZO'] = pd.Series(zeta).fillna(0)
        return df

# ==========================================
# 6. VISUALS & REPORTING
# ==========================================
class Visuals:
    @staticmethod
    def render_axiom_clock():
        html = """
        <div style="display:flex; justify-content:space-around; font-family:'Roboto Mono'; color:#00F0FF; padding:10px; background:rgba(0,0,0,0.2); border-radius:5px;">
            <div>NY: <span id="ny">--:--</span></div>
            <div>LON: <span id="lon">--:--</span></div>
            <div>TOK: <span id="tok">--:--</span></div>
        </div>
        <script>
            setInterval(() => {
                const fmt = (tz) => new Date().toLocaleTimeString('en-GB', {timeZone:tz, hour:'2-digit', minute:'2-digit'});
                document.getElementById('ny').innerText = fmt('America/New_York');
                document.getElementById('lon').innerText = fmt('Europe/London');
                document.getElementById('tok').innerText = fmt('Asia/Tokyo');
            }, 1000);
        </script>
        """
        components.html(html, height=60)

    @staticmethod
    def generate_titan_html_report(row, ticker, whale_sent, vol_conf):
        return f"""
        <div class="report-card" style="border-left: 4px solid #00ff00;">
            <div class="report-header">🏛️ TITAN SIGNAL: {ticker}</div>
            <div class="report-item">Bias: <span class="value-cyan">{'LONG 🐂' if row['is_bull'] else 'SHORT 🐻'}</span></div>
            <div class="report-item">Score: <span class="value-cyan">{row['GM_Score']:.0f} / 5</span></div>
            <div class="report-item">Whale Sentiment: <span class="value-cyan">{whale_sent}</span></div>
            <div class="report-item">Confirmation: <span class="value-cyan">{vol_conf}</span></div>
        </div>
        <div class="report-card" style="border-left: 4px solid #38bdf8;">
            <div class="report-header">🎯 EXECUTION PLAN</div>
            <div class="report-item">Entry: <span class="value-cyan">${row['close']:.4f}</span></div>
            <div class="report-item">Stop: <span class="value-cyan">${row['entry_stop']:.4f}</span></div>
            <div class="report-item">TP1: <span class="value-cyan">${row['tp1']:.4f}</span></div>
            <div class="report-item">TP2: <span class="value-cyan">${row['tp2']:.4f}</span></div>
        </div>
        """

# ==========================================
# 7. MAIN CONTROLLER
# ==========================================
def main():
    st.sidebar.title("💠 OMNI-SENTIENT")
    mode = st.sidebar.radio("MODE", ["TITAN (Binance/Crypto)", "AXIOM (Quant/Macro)"])
    
    # Global Secrets
    tg_token = SecretsManager.get("TELEGRAM_TOKEN")
    tg_chat = SecretsManager.get("TELEGRAM_CHAT_ID")

    if mode == "TITAN (Binance/Crypto)":
        ticker_base = st.sidebar.selectbox("Asset", ["BTC", "ETH", "SOL", "BNB", "XRP"])
        ticker = f"{ticker_base}USDT"
        tf = st.sidebar.selectbox("TF", ["15m", "1h", "4h", "1d"], index=1)
        
        # Start Whale WebSocket
        if 'current_tracker' not in st.session_state or st.session_state.current_tracker.symbol != ticker.lower().replace("-usd", "usdt"):
            st.session_state.whale_data = []
            st.session_state.current_tracker = WhaleTracker(ticker)
            st.session_state.current_tracker.start()

        st.markdown(f'<div class="title-glow">TITAN: {ticker_base}</div>', unsafe_allow_html=True)
        Visuals.render_axiom_clock()
        
        if st.button("RUN TITAN ANALYSIS"):
            df = TitanEngine.get_klines(ticker, tf)
            if not df.empty:
                df, zones = TitanEngine.run_engine(df)
                last = df.iloc[-1]
                
                # Whale Stats
                w_buy = sum(1 for d in st.session_state.whale_data if d['Side'] == 'BUY')
                w_sell = sum(1 for d in st.session_state.whale_data if d['Side'] == 'SELL')
                w_sent = "ACCUMULATION" if w_buy > w_sell else "DISTRIBUTION"
                vol_conf = "CONFIRMED" if (last['is_bull'] and last['Net_Delta'] > 0) else "DIVERGENT"

                # Layout
                c1, c2 = st.columns([0.7, 0.3])
                with c1:
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.03)
                    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
                    fig.add_trace(go.Bar(x=df['timestamp'], y=df['Net_Delta'], marker_color='cyan', name="Net Delta"), row=2, col=1)
                    fig.add_trace(go.Bar(x=df['timestamp'], y=df['Apex_Flux'], name="Flux Momentum"), row=3, col=1)
                    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with c2:
                    st.markdown(Visuals.generate_titan_html_report(last, ticker, w_sent, vol_conf), unsafe_allow_html=True)
                    st.metric("Whale Count", f"{len(st.session_state.whale_data)}")
                    
                    if st.button("📢 BROADCAST SIGNAL"):
                        msg = f"🏛️ TITAN SIGNAL: {ticker}\nBias: {'LONG' if last['is_bull'] else 'SHORT'}\nScore: {last['GM_Score']}\nWhale: {w_sent}\nEntry: {last['close']}"
                        requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={"chat_id":tg_chat, "text":msg})
                        st.success("Broadcasted")

    else:
        ticker = st.sidebar.text_input("Ticker (YFinance)", value="NVDA")
        tf = st.sidebar.selectbox("TF", ["1h", "4h", "1d", "1wk"], index=2)
        
        st.markdown(f'<div class="title-glow">AXIOM: {ticker}</div>', unsafe_allow_html=True)
        Visuals.render_axiom_clock()

        df = AxiomEngine.fetch_data(ticker, tf)
        if not df.empty:
            df = AxiomEngine.calc_chedo(df)
            df = AxiomEngine.calc_rqzo(df)
            last = df.iloc[-1]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Entropy (CHEDO)", f"{last['CHEDO']:.2f}", delta="Chaos" if abs(last['CHEDO'])>0.8 else "Stable")
            c2.metric("Relativity (RQZO)", f"{last['RQZO']:.2f}")
            c3.metric("Price", f"{last['Close']:.2f}")

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
            fig.update_layout(height=600, template="plotly_dark", title="Axiom Quant Visualization")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
