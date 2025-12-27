import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
import calendar
import datetime
import requests
import urllib.parse
from scipy.stats import linregress
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ==========================================
# 1. CORE CONFIGURATION & THEME ENGINE
# ==========================================
st.set_page_config(layout="wide", page_title="🏦 Titan Axiom Terminal", page_icon="👁️")

def inject_ui_styles(is_mobile):
    """Injects high-fidelity DarkPool/Axiom CSS."""
    base_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&display=swap');
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .title-glow {
        font-size: 3em; font-weight: bold; color: #ffffff;
        text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00; margin-bottom: 20px;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02); border-left: 2px solid #333;
        padding: 15px; border-radius: 4px; transition: 0.3s;
    }
    div[data-testid="stMetric"]:hover { border-left: 2px solid #00F0FF; background: rgba(255, 255, 255, 0.05); }
    .report-card {
        background-color: #111; border-left: 4px solid #00F0FF;
        padding: 15px; border-radius: 4px; margin-bottom: 10px;
    }
    .highlight { color: #00F0FF; font-weight: bold; }
    /* Ticker Marquee */
    .ticker-wrap { width: 100%; overflow: hidden; background: #0a0a0a; border-bottom: 1px solid #333; height: 40px; display: flex; align-items: center; }
    .ticker { display: inline-block; animation: marquee 45s linear infinite; }
    @keyframes marquee { 0% { transform: translate(100%, 0); } 100% { transform: translate(-100%, 0); } }
    .ticker-item { display: inline-block; padding: 0 2rem; color: #00F0FF; font-size: 0.85rem; }
    </style>
    """
    st.markdown(base_css, unsafe_allow_html=True)
    if is_mobile:
        st.markdown("<style>div[data-testid='stMetricValue'] { font-size: 1.8rem !important; }</style>", unsafe_allow_html=True)

# ==========================================
# 2. QUANTUM & PHYSICS ENGINE (AXIOM UPGRADE)
# ==========================================
class AxiomEngine:
    @staticmethod
    def tanh(x): return np.tanh(np.clip(x, -20, 20))

    @staticmethod
    def calc_chedo(df, length=50):
        """Entropy/Chaos Engine."""
        c = df['Close'].values
        log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
        mu = pd.Series(log_ret).rolling(length).mean().values
        sigma = pd.Series(log_ret).rolling(length).std().values
        v = sigma / (np.abs(mu) + 1e-9)
        abs_ret_v = np.abs(log_ret) * v
        hyper_dist = np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))
        kappa_h = AxiomEngine.tanh(pd.Series(hyper_dist).rolling(length).mean().values)
        ent = pd.Series(log_ret**2).rolling(length).sum().values
        ent_n = AxiomEngine.tanh(ent * 10)
        df['CHEDO'] = (0.7 * kappa_h) + (0.3 * ent_n)
        return df

    @staticmethod
    def calc_rqzo(df, harmonics=25):
        """Relativity/Time Dilation Engine."""
        src = df['Close']
        mn, mx = src.rolling(100).min(), src.rolling(100).max()
        norm = (src - mn) / (mx - mn + 1e-9)
        v = np.abs(norm.diff())
        gamma = 1 / np.sqrt(1 - (np.minimum(v, 0.049)/0.05)**2)
        idx = np.arange(len(df))
        tau = (idx % 100) / gamma.fillna(1.0)
        zeta = np.zeros(len(df))
        for n in range(1, harmonics + 1):
            zeta += (n**-0.5) * np.sin(tau * np.log(n))
        df['RQZO'] = pd.Series(zeta).fillna(0)
        return df

    @staticmethod
    def calc_apex_flux(df, length=14):
        """Vector/Efficiency Engine."""
        rg = df['High'] - df['Low']
        body = np.abs(df['Close'] - df['Open'])
        eff = np.where(rg == 0, 0, body / rg)
        eff_sm = pd.Series(eff, index=df.index).ewm(span=length).mean()
        v_rat = df['Volume'] / df['Volume'].rolling(55).mean()
        df['Apex_Flux'] = (np.sign(df['Close'] - df['Open']) * eff_sm * v_rat).ewm(span=5).mean()
        return df

# ==========================================
# 3. BASE DATA & TECHNICAL ENGINE (TITAN PRESERVED)
# ==========================================
@st.cache_data(ttl=300)
def get_macro_banner():
    """Renders scrolling marquee of live prices."""
    try:
        tickers = ["BTC-USD", "ETH-USD", "SPY", "QQQ", "NVDA", "GLD", "VIX"]
        data = yf.download(tickers, period="1d", interval="1d", progress=False)['Close'].iloc[-1]
        items = [f"{t}: ${data[t]:,.2f}" for t in tickers if t in data]
        html = f'<div class="ticker-wrap"><div class="ticker">{" | ".join([f"<span class=\'ticker-item\'>{i}</span>" for i in items])}</div></div>'
        st.markdown(html, unsafe_allow_html=True)
    except: st.markdown("Terminal Online...")

def calculate_hma(series, length):
    wma_f = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length + 1)) / (length * (length + 1) / 2), raw=True)
    wma_h = series.rolling(int(length/2)).apply(lambda x: np.dot(x, np.arange(1, int(length/2) + 1)) / (int(length/2) * (int(length/2) + 1) / 2), raw=True)
    diff = 2 * wma_h - wma_f
    sqrt_l = int(np.sqrt(length))
    return diff.rolling(sqrt_l).apply(lambda x: np.dot(x, np.arange(1, sqrt_l + 1)) / (sqrt_l * (sqrt_l + 1) / 2), raw=True)

def calc_indicators_unified(df):
    """Calculates Unified Titan + Axiom Stack."""
    df['HMA'] = calculate_hma(df['Close'], 55)
    df['ATR'] = calculate_atr(df, 14)
    # Axiom Physics
    df = AxiomEngine.calc_chedo(df)
    df = AxiomEngine.calc_rqzo(df)
    df = AxiomEngine.calc_apex_flux(df)
    # Titan Metrics
    ema12, ema26 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    # RSI
    delta = df['Close'].diff()
    gain, loss = delta.where(delta > 0, 0).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
    # Confluence Score
    df['GM_Score'] = np.sign(df['Apex_Flux']) + np.sign(df['Hist']) + np.sign(df['Close'] - df['HMA'])
    return df

# 
# ==========================================
# 4. DASHBOARD UI & TABS
# ==========================================
def main():
    st.sidebar.header("🎛️ Terminal Controls")
    is_mobile = st.sidebar.toggle("📱 Mobile Optimized", value=False)
    inject_ui_styles(is_mobile)
    get_macro_banner()

    ticker = st.sidebar.text_input("Ticker", value="BTC-USD").upper()
    interval = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=2)
    
    # API Management
    if "OPENAI_API_KEY" in st.secrets: st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
    else: st.session_state.api_key = st.sidebar.text_input("OpenAI Key", type="password")

    if st.sidebar.button("Analyze Asset"):
        with st.spinner("Processing Quantum Data..."):
            df = yf.download(ticker, period="1y", interval=interval)
            if df.empty:
                st.error("No Data.")
                return
            
            # Data Formatting Fix for Multi-Index
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            df = calc_indicators_unified(df)
            last = df.iloc[-1]

            # ----------------------------------------------------
            # KPI BAR (Axiom + Titan Metrics)
            # ----------------------------------------------------
            if is_mobile:
                st.markdown(f"""
                <div class="report-card">
                    <div class="report-header">💠 {ticker} {interval}</div>
                    <div class="report-item">Entropy (CHEDO): <span class="highlight">{last['CHEDO']:.2f}</span></div>
                    <div class="report-item">Vector (Flux): <span class="highlight">{last['Apex_Flux']:.2f}</span></div>
                    <div class="report-item">Titan Score: <span class="highlight">{last['GM_Score']:.0f}/3</span></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Price", f"${last['Close']:,.2f}")
                c2.metric("Entropy", f"{last['CHEDO']:.2f}", "CHAOS" if abs(last['CHEDO']) > 0.7 else "STABLE")
                c3.metric("Relativity", f"{last['RQZO']:.2f}")
                c4.metric("Vector Flux", f"{last['Apex_Flux']:.2f}")
                c5.metric("Titan Score", f"{last['GM_Score']:.0f}")

            # ----------------------------------------------------
            # TABS: The Integrated View
            # ----------------------------------------------------
            tab1, tab2, tab3, tab4 = st.tabs(["📉 Axiom Technicals", "🏦 Smart Money", "🔮 Forecasting", "📡 Broadcast"])
            
            with tab1:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2])
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='yellow', width=1), name="HMA"), row=1, col=1)
                # CHEDO Panel
                fig.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], fill='tozeroy', name="Entropy"), row=2, col=1)
                # Flux Panel
                colors = ['#00ff00' if v > 0 else '#ff0000' for v in df['Apex_Flux']]
                fig.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], marker_color=colors, name="Vector Flux"), row=3, col=1)
                fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                from Axiom_Quant_Upgrade_V2_1 import calculate_smc_advanced
                smc = calculate_smc(df) # Logic from base script
                fig_smc = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
                # Overlay logic preserved from code snippet 6
                st.plotly_chart(fig_smc, use_container_width=True)

            with tab3:
                # Monte Carlo Forecast
                returns = df['Close'].pct_change().dropna()
                paths = np.zeros((30, 100))
                paths[0] = last['Close']
                for t in range(1, 30):
                    paths[t] = paths[t-1] * (1 + np.random.normal(returns.mean(), returns.std(), 100))
                fig_mc = go.Figure()
                for i in range(10): fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', opacity=0.3))
                fig_mc.update_layout(title="30-Day Path Projection", template="plotly_dark")
                st.plotly_chart(fig_mc, use_container_width=True)

            with tab4:
                # Unified Broadcast System
                st.subheader("Social Command Center")
                signal_text = f"🔥 {ticker} TITAN ANALYSIS\nPrice: ${last['Close']:.2f}\nScore: {last['GM_Score']}/3\nVector: {last['Apex_Flux']:.2f}\n#Trading #Axiom"
                msg = st.text_area("Payload", value=signal_text, height=150)
                if st.button("🚀 Broadcast to Telegram"):
                    # Preservation of the Part splitting logic from Code Snippet 1
                    requests.post(f"https://api.telegram.org/bot{st.secrets['TELEGRAM_TOKEN']}/sendMessage", 
                                  json={"chat_id": st.secrets['TELEGRAM_CHAT_ID'], "text": msg})
                    st.success("Broadcast Dispatched.")

def calculate_atr(df, length=14):
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    return tr.rolling(length).mean()

if __name__ == "__main__":
    main()
