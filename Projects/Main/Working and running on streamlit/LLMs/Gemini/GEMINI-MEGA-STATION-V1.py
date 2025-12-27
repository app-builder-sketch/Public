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
# 1. DATABASE & PERSISTENCE LAYER
# ==========================================
def init_db():
    conn = sqlite3.connect('titan_vault.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS signals 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp DATETIME, symbol TEXT, interval TEXT, 
                  score REAL, price REAL, message TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist 
                 (symbol TEXT PRIMARY KEY)''')
    conn.commit()
    conn.close()

init_db()

# --- SYSTEM STATE ---
if 'whale_data' not in st.session_state:
    st.session_state.whale_data = []
if 'last_ai_summary' not in st.session_state:
    st.session_state.last_ai_summary = "Awaiting institutional analysis..."

# ==========================================
# 2. GLOBAL UI & CUSTOM CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Omni-Sentient Titan Axiom v5.5", page_icon="💠")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=SF+Pro+Display:wght@300;500;700&display=swap');
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'SF Pro Display', sans-serif; }
    .title-glow { font-size: 3em; font-weight: bold; color: #ffffff; text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00; margin-bottom: 20px; font-family: 'Roboto Mono'; }
    div[data-testid="metric-container"] {
        background: rgba(22, 27, 34, 0.9); border-left: 4px solid #00F0FF;
        padding: 15px; border-radius: 6px; backdrop-filter: blur(5px);
    }
    .ticker-wrap { width: 100%; overflow: hidden; background-color: #0d1117; border-bottom: 1px solid #30363d; height: 40px; display: flex; align-items: center; }
    .ticker { display: inline-block; animation: marquee 45s linear infinite; white-space: nowrap; }
    @keyframes marquee { 0% { transform: translate(100%, 0); } 100% { transform: translate(-100%, 0); } }
    .ticker-item { padding: 0 2rem; font-family: 'Roboto Mono'; font-size: 0.85rem; color: #58a6ff; }
    .report-card { background-color: #161b22; border-left: 4px solid #3fb950; padding: 16px; border-radius: 6px; margin-bottom: 16px; border: 1px solid #30363d; }
    .report-header { font-size: 1.1rem; font-weight: 700; color: #f0f6fc; margin-bottom: 12px; border-bottom: 1px solid #30363d; padding-bottom: 8px; font-family: 'Roboto Mono'; }
    .report-item { margin-bottom: 8px; font-size: 0.95rem; color: #8b949e; display: flex; gap: 8px; }
    .value-cyan { color: #38bdf8; font-weight: 600; font-family: 'Roboto Mono'; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. CORE ENGINES (MATH & WHALES)
# ==========================================

class WhaleTracker:
    def __init__(self, symbol):
        self.symbol = symbol.lower().replace("-usd", "usdt")
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@aggTrade"
        self.ws = None
        self.thread = None

    def on_message(self, ws, message):
        data = json.loads(message)
        qty, price = float(data['q']), float(data['p'])
        usd_val = qty * price
        if usd_val >= 100000:
            side = "BUY" if not data['m'] else "SELL"
            st.session_state.whale_data.append({
                "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                "Price": f"{price:.2f}", "Value": f"${usd_val:,.0f}",
                "Side": side, "RawVal": usd_val
            })
            if len(st.session_state.whale_data) > 30: st.session_state.whale_data.pop(0)

    def start(self):
        self.ws = websocket.WebSocketApp(self.ws_url, on_message=self.on_message)
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()

class QuantMath:
    @staticmethod
    def get_ma(series, length, ma_type="HMA"):
        if ma_type == "SMA": return series.rolling(length).mean()
        if ma_type == "EMA": return series.ewm(span=length, adjust=False).mean()
        half, sqrt = int(length/2), int(math.sqrt(length))
        wma_f = series.rolling(length).mean()
        wma_h = series.rolling(half).mean()
        return (2 * wma_h - wma_f).rolling(sqrt).mean()

    @staticmethod
    def calc_volumetric_delta(df):
        rg = (df['high'] - df['low']).replace(0, 0.0001)
        buy_vol = ((df['close'] - df['low']) / rg) * df['volume']
        df['Net_Delta'] = buy_vol - (df['volume'] - buy_vol)
        df['Volume_Bias_Pct'] = (df['Net_Delta'] / df['volume']) * 100
        return df

    @staticmethod
    def calc_physics(df):
        c = df['Close'].values if 'Close' in df.columns else df['close'].values
        # CHEDO (Entropy)
        log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
        mu = pd.Series(log_ret).rolling(50).mean()
        sigma = pd.Series(log_ret).rolling(50).std()
        abs_ret_v = np.abs(log_ret) * (sigma / (np.abs(mu) + 1e-9))
        df['CHEDO'] = np.tanh(pd.Series(np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))).rolling(50).mean())
        # RQZO (Relativity)
        mn, mx = pd.Series(c).rolling(100).min(), pd.Series(c).rolling(100).max()
        norm = (c - mn) / (mx - mn + 1e-9)
        gamma = 1 / np.sqrt(1 - (np.minimum(np.abs(norm.diff()), 0.049)/0.05)**2)
        zeta = np.zeros(len(df))
        tau = (np.arange(len(df)) % 100) / gamma.fillna(1.0)
        for n in range(1, 25): zeta += (n**-0.5) * np.sin(tau * np.log(n))
        df['RQZO'] = pd.Series(zeta).fillna(0)
        return df

# ==========================================
# 4. TITAN MODE (CRYPTO/BINANCE)
# ==========================================
class TitanEngine:
    @staticmethod
    @st.cache_data(ttl=5)
    def fetch_klines(symbol, interval, limit=200):
        try:
            url = f"https://api.binance.us/api/v3/klines"
            r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit})
            df = pd.DataFrame(r.json(), columns=['t','o','h','l','c','v','T','q','n','V','Q','B'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
            return df
        except: return pd.DataFrame()

    @staticmethod
    def run_titan_logic(df, amp=10, dev=3.0):
        df = df.copy()
        df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
        df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
        df['hma'] = QuantMath.get_ma(df['close'], 50)
        df = QuantMath.calc_volumetric_delta(df)
        
        # Titan Trend & Gann logic
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
        
        # Apex Flux
        body = np.abs(df['close'] - df['open'])
        eff = np.where((df['high']-df['low']) == 0, 0, body / (df['high']-df['low']))
        df['Apex_Flux'] = (pd.Series(eff).ewm(span=14).mean() * np.sign(df['close']-df['open'])).ewm(span=5).mean()
        
        risk = abs(df['close'] - df['entry_stop']).replace(0, df['close']*0.01)
        df['tp1'] = np.where(df['is_bull'], df['close'] + 1.5*risk, df['close'] - 1.5*risk)
        df['tp2'] = np.where(df['is_bull'], df['close'] + 3.0*risk, df['close'] - 3.0*risk)
        df['GM_Score'] = np.where(df['is_bull'], 1, -1) + np.sign(df['Apex_Flux'])
        return df

# ==========================================
# 5. AXIOM MODE (STOCKS/QUANT)
# ==========================================
class AxiomEngine:
    @staticmethod
    def fetch_axiom_data(ticker, tf):
        tf_map = {"15m": "1mo", "1h": "6mo", "4h": "1y", "1d": "2y"}
        df = yf.download(ticker, period=tf_map.get(tf, "1y"), interval=tf, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df.dropna()

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_dna(ticker):
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        df['Day'] = df.index.day_name()
        df['Ret'] = df['Close'].pct_change() * 100
        return df.groupby('Day')['Ret'].mean()

# ==========================================
# 6. VISUALS & CLOCKS
# ==========================================
class Visuals:
    @staticmethod
    def render_clocks():
        components.html("""
        <div style="display:flex; justify-content:space-around; font-family:'Roboto Mono'; color:#00F0FF; background:rgba(0,0,0,0.2); padding:10px; border-radius:5px;">
            <div>NYC: <span id="ny"></span></div><div>LON: <span id="lon"></span></div><div>TOK: <span id="tok"></span></div>
        </div>
        <script>
            setInterval(() => {
                const fmt = (tz) => new Date().toLocaleTimeString('en-GB', {timeZone:tz, hour:'2-digit', minute:'2-digit'});
                document.getElementById('ny').innerText = fmt('America/New_York');
                document.getElementById('lon').innerText = fmt('Europe/London');
                document.getElementById('tok').innerText = fmt('Asia/Tokyo');
            }, 1000);
        </script>
        """, height=50)

    @staticmethod
    def render_marquee():
        st.markdown("""
        <div class="ticker-wrap"><div class="ticker">
            <span class="ticker-item">💠 SYSTEM ONLINE</span><span class="ticker-item">TITAN ENGINE ACTIVE</span>
            <span class="ticker-item">AXIOM PHYSICS READY</span><span class="ticker-item">WHALE TRACKER LIVE</span>
        </div></div>
        """, unsafe_allow_html=True)

# ==========================================
# 7. MAIN APPLICATION CONTROLLER
# ==========================================
def main():
    st.sidebar.title("💠 OMNI-STATION")
    mode = st.sidebar.radio("MODE", ["TITAN (Crypto/Whales)", "AXIOM (Quant/Stocks)"])
    
    # Secrets Fallback
    tg_token = st.secrets.get("TELEGRAM_TOKEN", "")
    tg_chat = st.secrets.get("TELEGRAM_CHAT_ID", "")
    ai_key = st.secrets.get("OPENAI_API_KEY", "")

    Visuals.render_marquee()
    Visuals.render_clocks()

    if mode == "TITAN (Crypto/Whales)":
        ticker_base = st.sidebar.selectbox("Asset", ["BTC", "ETH", "SOL", "BNB", "XRP"])
        ticker = f"{ticker_base}USDT"
        interval = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=1)
        
        # WebSocket Threading
        if 'tracker' not in st.session_state or st.session_state.tracker.symbol != ticker.lower().replace("-usd","usdt"):
            st.session_state.whale_data = []
            st.session_state.tracker = WhaleTracker(ticker)
            st.session_state.tracker.start()

        if st.sidebar.button("RUN TITAN ANALYSIS"):
            df = TitanEngine.fetch_klines(ticker, interval)
            if not df.empty:
                df = TitanEngine.run_titan_logic(df)
                last = df.iloc[-1]
                
                c1, c2 = st.columns([0.7, 0.3])
                with c1:
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03)
                    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
                    fig.add_trace(go.Bar(x=df['timestamp'], y=df['Net_Delta'], marker_color='cyan', name="Delta"), row=2, col=1)
                    fig.add_trace(go.Bar(x=df['timestamp'], y=df['Apex_Flux'], name="Flux"), row=3, col=1)
                    fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with c2:
                    st.markdown(f"""
                    <div class="report-card">
                        <div class="report-header">🏛️ TITAN SIGNAL: {ticker}</div>
                        <div class="report-item">Regime: <span class="value-cyan">{'BULLISH 🐂' if last['is_bull'] else 'BEARISH 🐻'}</span></div>
                        <div class="report-item">Score: <span class="value-cyan">{last['GM_Score']:.0f}/5</span></div>
                        <div class="report-item">Entry: <span class="value-cyan">${last['close']:.4f}</span></div>
                        <div class="report-item">Stop: <span class="value-cyan">${last['entry_stop']:.4f}</span></div>
                        <div class="report-item">TP1: <span class="value-cyan">${last['tp1']:.4f}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.subheader("🐳 Whale Tape")
                    st.dataframe(pd.DataFrame(st.session_state.whale_data).sort_index(ascending=False), use_container_width=True)

    else: # AXIOM MODE
        ticker = st.sidebar.text_input("Ticker (YFinance)", "NVDA")
        tf = st.sidebar.selectbox("Interval", ["1h", "4h", "1d"], index=2)
        
        if st.sidebar.button("RUN AXIOM QUANT"):
            df = AxiomEngine.fetch_axiom_data(ticker, tf)
            if not df.empty:
                df = QuantMath.calc_physics(df)
                last = df.iloc[-1]
                
                tabs = st.tabs(["📉 Chart", "🧪 Physics", "📅 DNA", "🔮 Monte Carlo", "📊 Volume"])
                with tabs[0]:
                    st.plotly_chart(px.line(df, y='Close', template="plotly_dark"), use_container_width=True)
                with tabs[1]:
                    st.metric("Entropy (CHEDO)", f"{last['CHEDO']:.2f}", delta="Chaos" if abs(last['CHEDO'])>0.8 else "Stable")
                    st.metric("Relativity (RQZO)", f"{last['RQZO']:.2f}")
                with tabs[2]:
                    st.bar_chart(AxiomEngine.get_dna(ticker))
                with tabs[3]:
                    rets = df['Close'].pct_change().dropna()
                    paths = np.zeros((30, 20)); paths[0] = last['Close']
                    for t in range(1, 30): paths[t] = paths[t-1] * (1 + np.random.normal(rets.mean(), rets.std(), 20))
                    st.line_chart(paths)
                with tabs[4]:
                    price_bins = np.linspace(df['Low'].min(), df['High'].max(), 50)
                    vp = df.groupby(pd.cut(df['Close'], bins=price_bins), observed=False)['Volume'].sum()
                    st.bar_chart(vp)

if __name__ == "__main__":
    main()
