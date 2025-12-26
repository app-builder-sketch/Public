
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import math
import json
import urllib.parse
from scipy.signal import argrelextrema
from scipy.stats import linregress
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# ==========================================
# 0. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TITAN OMNI-DASHBOARD",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LIBRARY CHECKS ---
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import yfinance as yf
except ImportError:
    st.error("CRITICAL: `yfinance` library missing. Install via: pip install yfinance")
    st.stop()

# ==========================================
# 1. UI ENGINE: TITAN AESTHETIC (CSS & WIDGETS)
# ==========================================
def inject_titan_css():
    st.markdown("""
    <style>
        /* FONTS & BACKGROUND */
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&family=Roboto+Mono:wght@400;700&display=swap');
        .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Rajdhani', sans-serif; }

        /* NEON HEADERS */
        .titan-header {
            font-size: 3.5rem; font-weight: 700;
            background: -webkit-linear-gradient(45deg, #00E5FF, #2979FF);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px rgba(0, 229, 255, 0.4);
            letter-spacing: 2px; margin-bottom: 5px;
        }
        .titan-subheader {
            font-family: 'Roboto Mono', monospace; font-size: 0.9rem; color: #8b949e;
            letter-spacing: 1px; border-bottom: 1px solid #333; padding-bottom: 15px; margin-bottom: 20px;
        }

        /* METRIC CARDS */
        div[data-testid="metric-container"] {
            background: linear-gradient(145deg, #111, #161b22);
            border: 1px solid #333; border-left: 4px solid #00E5FF;
            padding: 15px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            transition: all 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-5px); box-shadow: 0 0 20px rgba(0, 229, 255, 0.2); border-color: #00E5FF;
        }
        div[data-testid="stMetricValue"] {
            font-family: 'Roboto Mono', monospace; color: #fff !important; font-size: 1.8rem !important; font-weight: 700;
            text-shadow: 0 0 10px rgba(255,255,255,0.3);
        }
        div[data-testid="stMetricLabel"] { color: #00E5FF !important; font-size: 0.8rem !important; letter-spacing: 1px; }

        /* TABS */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
        .stTabs [data-baseweb="tab"] {
            background-color: #0d1117; border: 1px solid #30363d; color: #8b949e; border-radius: 4px;
            font-family: 'Rajdhani', sans-serif; font-weight: 600; font-size: 1.1rem;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #00E5FF, #2979FF); color: #000 !important; font-weight: 700; border: none;
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
        }

        /* BUTTONS */
        .stButton > button {
            background: linear-gradient(135deg, #1f2833, #0b0c10); border: 1px solid #00E5FF; color: #00E5FF;
            font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 1.2rem; letter-spacing: 1px;
            text-transform: uppercase; border-radius: 6px; transition: all 0.3s ease; width: 100%;
        }
        .stButton > button:hover { background: #00E5FF; color: #000; box-shadow: 0 0 20px #00E5FF; }

        /* MOBILE REPORT CARD */
        .report-card {
            background: rgba(22, 27, 34, 0.95); border: 1px solid #30363d; border-radius: 12px; padding: 20px;
            margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .card-title { font-size: 1.2rem; color: #fff; font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }
        .card-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-family: 'Roboto Mono'; font-size: 0.9rem; }
        .bull { color: #00E676; font-weight: bold; text-shadow: 0 0 10px rgba(0, 230, 118, 0.3); }
        .bear { color: #FF1744; font-weight: bold; text-shadow: 0 0 10px rgba(255, 23, 68, 0.3); }
    </style>
    """, unsafe_allow_html=True)

def render_ticker_tape(symbol):
    """TradingView Ticker Tape Widget"""
    t_sym = f"BINANCE:{symbol}USDT" if "BTC" in symbol or "ETH" in symbol else symbol
    html = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
      {{
      "symbols": [ {{ "proName": "{t_sym}", "title": "{symbol}" }}, {{ "proName": "FOREXCOM:SPXUSD", "title": "S&P 500" }}, {{ "proName": "BITSTAMP:BTCUSD", "title": "Bitcoin" }} ],
      "showSymbolLogo": true, "colorTheme": "dark", "isTransparent": true, "displayMode": "adaptive", "locale": "en"
      }}
      </script>
    </div>
    """
    components.html(html, height=50)

def render_mobile_card(last):
    """Titan Mobile Summary Card"""
    score = last['GM_Score']
    sentiment = "BULLISH" if score > 0 else "BEARISH" if score < 0 else "NEUTRAL"
    s_class = "bull" if score > 0 else "bear" if score < 0 else ""
    
    html = f"""
    <div class="report-card">
        <div class="card-title">💠 FIELD REPORT</div>
        <div class="card-row"><span>God Mode Score:</span><span class="{s_class}">{score:.0f}/4</span></div>
        <div class="card-row"><span>Trend (MCM):</span><span class="{'bull' if last['MCM_Trend']==1 else 'bear'}">{'BULL' if last['MCM_Trend']==1 else 'BEAR'}</span></div>
        <div class="card-row"><span>Vector Gate:</span><span class="{'bear' if last['Vector_Locked'] else 'bull'}">{'LOCKED' if last['Vector_Locked'] else 'OPEN'}</span></div>
        <div class="card-row"><span>Entropy:</span><span>{last['CHEDO']:.2f}</span></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ==========================================
# 2. QUANTUM MATH KERNEL (PHYSICS + PINE LOGIC)
# ==========================================
class QuantumCore:
    """The Brain: Physics, Pine Translations, and Statistical Models."""
    
    # --- HELPER FUNCTIONS ---
    @staticmethod
    def tanh_clamp(x): return (np.exp(2.0 * np.clip(x, -20.0, 20.0)) - 1.0) / (np.exp(2.0 * np.clip(x, -20.0, 20.0)) + 1.0)

    @staticmethod
    def wma(s, l): return s.rolling(l).apply(lambda x: np.dot(x, np.arange(1, l + 1)) / np.arange(1, l + 1).sum(), raw=True)

    @staticmethod
    def hma(s, l):
        hl = int(l / 2); sl = int(np.sqrt(l))
        return QuantumCore.wma(2 * QuantumCore.wma(s, hl) - QuantumCore.wma(s, l), sl)

    @staticmethod
    def zlema(s, l):
        lag = int((l - 1) / 2)
        return (s + (s - s.shift(lag))).ewm(span=l, adjust=False).mean()

    @staticmethod
    def atr(df, l=14):
        hl = df['High'] - df['Low']; hc = (df['High'] - df['Close'].shift()).abs(); lc = (df['Low'] - df['Close'].shift()).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(l).mean()

    # --- 1. PHYSICS ENGINE (CHEDO + OMEGA) ---
    @staticmethod
    def calc_physics(df):
        # CHEDO
        l = 50; ret = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        mu = ret.rolling(l).mean(); sigma = ret.rolling(l).std()
        v = sigma / (np.abs(mu) + 1e-9)
        kappa = QuantumCore.tanh_clamp(np.log(np.abs(ret)*v + np.sqrt((np.abs(ret)*v)**2 + 1)).rolling(l).mean())
        
        def shannon(x):
            c, _ = np.histogram(x, bins=10); p = c[c > 0] / c.sum()
            return (np.log(10) + (p * np.log(p)).sum()) / np.log(10)
        
        ent = ret.rolling(l).apply(shannon, raw=True)
        df['CHEDO'] = (2 / (1 + np.exp(-(0.4 * kappa + 0.6 * ent) * 4)) - 1).rolling(3).mean()

        # OMEGA (Fluid Dynamics)
        atr = QuantumCore.atr(df, 14); rho = df['Volume']; u = df['Close'].diff().abs()
        mu_price = df['Close'].rolling(252).std()
        Re = (rho * u * atr) / (mu_price + 1e-9)
        df['Reynolds'] = Re
        
        # Quantum Tunneling
        k_q = np.sqrt(np.maximum(0, mu_price - (df['Close'] - df['Close'].shift(252)).abs())) / atr
        df['Omega_Mag'] = (1 / np.cosh(k_q)**2) * QuantumCore.tanh_clamp(Re / (Re.rolling(252).mean() + 1e-9))
        return df

    # --- 2. PINE SCRIPT TRANSLATIONS ---
    @staticmethod
    def calc_pine_algos(df):
        # A. F&G v4 (Composite)
        delta = df['Close'].diff()
        rsi = 100 - (100 / (1 + (delta.where(delta>0,0).ewm(alpha=1/14).mean() / -delta.where(delta<0,0).ewm(alpha=1/14).mean())))
        
        ema12 = df['Close'].ewm(span=12).mean(); ema26 = df['Close'].ewm(span=26).mean()
        macd = ema12 - ema26; sig = macd.ewm(span=9).mean(); hist = macd - sig
        macd_score = (50 + (hist / (hist.rolling(100).std()+1e-9) * 16.6)).clip(0, 100)
        
        sma20 = df['Close'].rolling(20).mean(); std20 = df['Close'].rolling(20).std()
        bb_score = ((df['Close'] - (sma20 - 2*std20)) / (4*std20 + 1e-9) * 100).clip(0, 100)
        
        zs = QuantumCore.zlema(df['Close'], 50); zl = QuantumCore.zlema(df['Close'], 200)
        trend_score = np.where((df['Close']>zs)&(zs>zl), 75, np.where(df['Close']>zs, 60, np.where((df['Close']<zs)&(zs<zl), 25, 40)))
        
        df['FG_Index'] = (rsi*0.3 + macd_score*0.25 + bb_score*0.25 + trend_score*0.2).rolling(3).mean()

        # B. Gann HiLo
        high_ma = df['High'].rolling(3).mean(); low_ma = df['Low'].rolling(3).mean()
        close = df['Close'].values; h_ma = high_ma.values; l_ma = low_ma.values
        act = np.zeros(len(df)); trend = np.zeros(len(df)); act[0] = l_ma[0]; trend[0] = 1
        
        for i in range(1, len(df)):
            if trend[i-1] == 1:
                if close[i] < act[i-1]: trend[i] = -1; act[i] = h_ma[i]
                else: trend[i] = 1; act[i] = l_ma[i]
            else:
                if close[i] > act[i-1]: trend[i] = 1; act[i] = l_ma[i]
                else: trend[i] = -1; act[i] = h_ma[i]
        df['Gann_Activator'] = act; df['Gann_Trend'] = trend

        # C. Dark Vector (SuperTrend + Chop)
        atr10 = QuantumCore.atr(df, 10); hl2 = (df['High']+df['Low'])/2
        up = hl2 + 4*atr10; dn = hl2 - 4*atr10
        st = np.zeros(len(df)); st_dir = np.zeros(len(df)); st[0] = dn[0]; st_dir[0] = 1
        u_val = up.values; d_val = dn.values
        
        for i in range(1, len(df)):
            if close[i-1] > st[i-1]:
                st[i] = max(d_val[i], st[i-1]) if close[i] > st[i-1] else u_val[i]
                st_dir[i] = 1 if close[i] > st[i-1] else -1
                if close[i] < d_val[i] and st_dir[i-1] == 1: st[i] = u_val[i]; st_dir[i] = -1
            else:
                st[i] = min(u_val[i], st[i-1]) if close[i] < st[i-1] else d_val[i]
                st_dir[i] = -1 if close[i] < st[i-1] else 1
                if close[i] > u_val[i] and st_dir[i-1] == -1: st[i] = d_val[i]; st_dir[i] = 1
        
        df['Vector_Trend'] = st_dir
        tr1 = QuantumCore.atr(df, 1).rolling(14).sum()
        rng = (df['High'].rolling(14).max() - df['Low'].rolling(14).min())
        df['Chop_Index'] = 100 * np.log10(tr1 / (rng + 1e-9)) / np.log10(14)
        df['Vector_Locked'] = df['Chop_Index'] > 60

        # D. Market Cycle Master (MCM)
        hma55 = QuantumCore.hma(df['Close'], 55); atr55 = QuantumCore.atr(df, 55)
        df['MCM_Upper'] = hma55 + atr55*1.5; df['MCM_Lower'] = hma55 - atr55*1.5
        df['MCM_Trend'] = np.select([(df['Close']>df['MCM_Upper']), (df['Close']<df['MCM_Lower'])], [1, -1], default=0)
        df['MCM_Trend'] = df['MCM_Trend'].replace(0, method='ffill')
        df['MCM_Stop'] = np.where(df['MCM_Trend']==1, df['MCM_Lower'], df['MCM_Upper'])

        return df

    @staticmethod
    def calc_god_mode(df):
        # Squeeze
        bb_u = df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std()
        bb_l = df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std()
        kc_u = df['Close'].rolling(20).mean() + 1.5*QuantumCore.atr(df, 20)
        kc_l = df['Close'].rolling(20).mean() - 1.5*QuantumCore.atr(df, 20)
        df['Squeeze_On'] = (bb_l > kc_l) & (bb_u < kc_u)
        
        # Momentum
        x = np.arange(20)
        df['Sqz_Mom'] = df['Close'].rolling(20).apply(lambda y: linregress(x, y)[0], raw=True) * 100
        
        # Confluence Score
        df['GM_Score'] = np.where(df['MCM_Trend']==1, 1, -1) + np.where(df['Gann_Trend']==1, 1, -1) + np.where(df['Vector_Trend']==1, 1, -1) + np.sign(df['Sqz_Mom'])
        return df

# ==========================================
# 3. DATA & INTELLIGENCE LAYERS
# ==========================================
class DataEngine:
    @staticmethod
    def fetch_data(ticker, interval):
        p_map = {"15m": ("60d", "15m"), "1h": ("730d", "1h"), "4h": ("730d", "1h"), "1d": ("5y", "1d"), "1wk": ("5y", "1wk")}
        p, i = p_map.get(interval, ("1y", "1d"))
        try:
            df = yf.download(ticker, period=p, interval=i, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if df.empty: return None
            if interval == "4h": df = df.resample("4h").agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            return df
        except: return None
        
    @staticmethod
    def get_macro():
        tickers = {"SPX": "SPY", "NDX": "QQQ", "BTC": "BTC-USD", "DXY": "DX-Y.NYB", "VIX": "^VIX"}
        try:
            d = yf.download(list(tickers.values()), period="5d", interval="1d", group_by='ticker', progress=False)
            res = {}
            for k, v in tickers.items():
                df = d[v]; last = df['Close'].iloc[-1]; prev = df['Close'].iloc[-2]
                res[k] = (last, (last-prev)/prev*100)
            return res
        except: return {}

    @staticmethod
    def get_seasonality(ticker):
        try:
            df = yf.download(ticker, period="10y", interval="1mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df['R'] = df['Close'].pct_change()*100; df['M'] = df.index.month; df['Y'] = df.index.year
            return df.pivot_table(index='Y', columns='M', values='R')
        except: return None

class Intelligence:
    @staticmethod
    def load_secrets():
        k = {"gem": "", "oai": "", "tg_t": "", "tg_c": ""}
        s = {"gem": False, "oai": False, "tg": False}
        try:
            k["gem"] = st.secrets.get("GEMINI_API_KEY", "")
            k["oai"] = st.secrets.get("OPENAI_API_KEY", "")
            k["tg_t"] = st.secrets.get("TELEGRAM_TOKEN", "")
            k["tg_c"] = st.secrets.get("TELEGRAM_CHAT_ID", "")
        except: pass
        if k["gem"]: s["gem"] = True
        if k["oai"]: s["oai"] = True
        if k["tg_t"] and k["tg_c"]: s["tg"] = True
        return k, s

    @staticmethod
    def broadcast(msg, t, c):
        if not t or not c: return False, "No Credentials"
        try:
            for chunk in [msg[i:i+2000] for i in range(0, len(msg), 2000)]:
                requests.post(f"https://api.telegram.org/bot{t}/sendMessage", json={"chat_id": c, "text": chunk, "parse_mode": "Markdown"})
            return True, "Sent"
        except Exception as e: return False, str(e)

# ==========================================
# 4. CHART RENDERING
# ==========================================
def render_omni_chart(df, ticker):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.15, 0.2], vertical_spacing=0.03)
    
    # 1. Price + Clouds + Gann
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MCM_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MCM_Lower'], fill='tonexty', fillcolor='rgba(0, 229, 255, 0.1)', line=dict(width=0), name="Titan Cloud"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Gann_Activator'], line=dict(color='#FFD700', dash='dot'), name="Gann"), row=1, col=1)
    
    # 2. Squeeze
    colors = ['#00E676' if v >= 0 else '#FF1744' for v in df['Sqz_Mom']]
    fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=colors, name="Mom"), row=2, col=1)
    
    # 3. Omega
    fig.add_trace(go.Scatter(x=df.index, y=df['Omega_Mag'], fill='tozeroy', line=dict(color='#2979FF'), name="Omega"), row=3, col=1)
    
    # 4. Entropy
    fig.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], line=dict(color='#EA80FC'), name="Entropy"), row=4, col=1)
    
    fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False, paper_bgcolor="#050505", plot_bgcolor="#050505")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    inject_titan_css()
    keys, status = Intelligence.load_secrets()
    
    with st.sidebar:
        st.markdown("## ⚙️ TITAN CORE")
        ticker = st.text_input("SYMBOL", "BTC-USD").upper()
        timeframe = st.selectbox("TIMEFRAME", ["15m", "1h", "4h", "1d", "1wk"], index=2)
        
        st.markdown("---")
        with st.expander("💰 RISK & CAPITAL", expanded=True):
            acc_size = st.number_input("Account ($)", value=10000)
            risk_pct = st.slider("Risk %", 0.1, 5.0, 1.0)
            
        with st.expander("🔐 CREDENTIALS"):
            keys["gem"] = st.text_input("Gemini", value=keys["gem"], type="password")
            keys["oai"] = st.text_input("OpenAI", value=keys["oai"], type="password")
            keys["tg_t"] = st.text_input("TG Token", value=keys["tg_t"], type="password")
            keys["tg_c"] = st.text_input("TG Chat", value=keys["tg_c"])
            if status["gem"]: st.caption("✅ Gemini Loaded")
            if status["oai"]: st.caption("✅ OpenAI Loaded")
            
        if st.button("🚀 INITIATE SYSTEM"): st.session_state['run'] = True

    st.markdown('<div class="titan-header">TITAN OMNI</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="titan-subheader">INSTITUTIONAL DASHBOARD // TARGET: {ticker}</div>', unsafe_allow_html=True)
    
    render_ticker_tape(ticker)
    
    if st.session_state.get('run', False):
        with st.spinner("QUANTUM CORES COMPUTING..."):
            df = DataEngine.fetch_data(ticker, timeframe)
            if df is not None:
                # PIPELINE
                df = QuantumCore.calc_physics(df)
                df = QuantumCore.calc_pine_algos(df)
                df = QuantumCore.calc_god_mode(df)
                last = df.iloc[-1]
                
                # METRICS
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("PRICE", f"{last['Close']:.2f}", f"{df['Close'].pct_change().iloc[-1]*100:.2f}%")
                
                sc = last['GM_Score']
                c2.metric("GOD MODE", f"{sc:.0f}/4", "STRONG" if abs(sc)>2 else "WEAK")
                c3.metric("SENTIMENT", f"{last['FG_Index']:.0f}", "GREED" if last['FG_Index']>60 else "FEAR")
                c4.metric("VECTOR", "LOCKED" if last['Vector_Locked'] else "OPEN", "CHOP" if last['Vector_Locked'] else "TREND")
                c5.metric("ENTROPY", f"{last['CHEDO']:.2f}", "CHAOS" if last['CHEDO']>0.5 else "ORDER")
                
                # TABS
                t_main, t_risk, t_macro, t_ai = st.tabs(["📈 OMNI CHART", "⚖️ RISK & SMC", "🌍 MACRO LAB", "🧠 TITAN AI"])
                
                with t_main:
                    render_omni_chart(df, ticker)
                    st.divider()
                    render_mobile_card(last)
                    
                with t_risk:
                    st.markdown("### ⚖️ INSTITUTIONAL RISK CALCULATOR")
                    dist = abs(last['Close'] - last['MCM_Stop']) / last['Close']
                    size = (acc_size * (risk_pct/100)) / (last['Close'] * dist) if dist > 0 else 0
                    
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("STOP LOSS", f"${last['MCM_Stop']:.2f}")
                    r2.metric("RISK ($)", f"${acc_size*(risk_pct/100):.2f}")
                    r3.metric("POS SIZE", f"{size:.4f} Units")
                    r4.metric("POS VALUE", f"${size*last['Close']:.2f}")
                    
                    st.markdown("### 🏦 SMART MONEY CONCEPTS")
                    st.info("SMC Structures (BOS/CHoCH) are visualized via the Swing Pivots in the main chart.")
                    
                with t_macro:
                    st.markdown("### 🌍 GLOBAL MACRO PULSE")
                    md = DataEngine.get_macro()
                    mc = st.columns(len(md))
                    for i, (k, v) in enumerate(md.items()): mc[i].metric(k, f"{v[0]:.2f}", f"{v[1]:.2f}%")
                    
                    st.divider()
                    st.markdown("### 📅 SEASONALITY HEATMAP")
                    hm = DataEngine.get_seasonality(ticker)
                    if hm is not None:
                        fig_hm = px.imshow(hm, color_continuous_scale='RdYlGn', text_auto='.1f')
                        fig_hm.update_layout(template="plotly_dark", height=500, paper_bgcolor="#050505")
                        st.plotly_chart(fig_hm, use_container_width=True)

                with t_ai:
                    c_gen, c_cast = st.columns(2)
                    with c_gen:
                        st.markdown("### 🧠 NEURAL ANALYSIS")
                        if st.button("GENERATE REPORT"):
                            prompt = f"Analyze {ticker}. Price {last['Close']}. GM Score {sc}. Entropy {last['CHEDO']}. Trending or Chopping?"
                            res = "Checking keys..."
                            if keys["gem"] and genai:
                                genai.configure(api_key=keys["gem"])
                                res = genai.GenerativeModel('gemini-pro').generate_content(prompt).text
                            elif keys["oai"] and OpenAI:
                                res = OpenAI(api_key=keys["oai"]).chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}]).choices[0].message.content
                            else: res = "⚠️ NO API KEYS FOUND."
                            st.success("ANALYSIS COMPLETE")
                            st.markdown(res)
                            
                    with c_cast:
                        st.markdown("### 📡 TELEGRAM UPLINK")
                        msg = st.text_area("PAYLOAD", f"🔥 {ticker} SIGNAL\nScore: {sc}\nPrice: {last['Close']}")
                        if st.button("BROADCAST"):
                            s, r = Intelligence.broadcast(msg, keys["tg_t"], keys["tg_c"])
                            if s: st.success(r)
                            else: st.error(r)
            else:
                st.error("DATA FEED FAILED.")

if __name__ == "__main__":
    main()
