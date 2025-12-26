
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
import time

# ==========================================
# 0. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NEXUS QUANTUM",
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
# 1. UI STYLING (INSTITUTIONAL DARK)
# ==========================================
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e6e6e6; font-family: 'SF Mono', 'Roboto Mono', monospace; }
    .nexus-header {
        font-size: 2.0rem; font-weight: 700; letter-spacing: -1px; color: #ffffff;
        border-left: 4px solid #00E5FF; padding-left: 15px; margin-bottom: 20px;
    }
    div[data-testid="metric-container"] {
        background: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 6px;
    }
    div[data-testid="stMetricValue"] { color: #00E5FF !important; font-size: 1.6rem !important; }
    div[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.9rem !important; }
    .stTabs [aria-selected="true"] {
        background-color: #00E5FF !important; color: #000 !important; font-weight: 700;
    }
    .stButton > button {
        width: 100%; background-color: #238636; color: white; border: none; font-weight: 600;
    }
    .stButton > button:hover { background-color: #2ea043; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. QUANTUM CORE (MATH & PHYSICS)
# ==========================================
class QuantumCore:
    """
    The Mathematical Engine: Physics, Statistics, and Pine Script Translations.
    """
    
    # --- UTILS ---
    @staticmethod
    def tanh_clamp(x):
        x_clamped = np.clip(x, -20.0, 20.0)
        e2x = np.exp(2.0 * x_clamped)
        return (e2x - 1.0) / (e2x + 1.0)

    @staticmethod
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def hma(series, length):
        half_len = int(length / 2)
        sqrt_len = int(np.sqrt(length))
        wma_f = QuantumCore.wma(series, length)
        wma_h = QuantumCore.wma(series, half_len)
        diff = 2 * wma_h - wma_f
        return QuantumCore.wma(diff, sqrt_len)

    @staticmethod
    def zlema(series, length):
        lag = int((length - 1) / 2)
        ema_data = series + (series - series.shift(lag))
        return ema_data.ewm(span=length, adjust=False).mean()

    @staticmethod
    def atr(df, length=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(length).mean()

    # --- ENGINES ---

    @staticmethod
    def calc_physics_engine(df):
        """CHEDO (Entropy) & OMEGA (Quantum Fluid)"""
        # 1. CHEDO
        length, bins = 50, 10
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        ret = df['log_ret']
        
        mu_r = ret.rolling(length).mean()
        sigma_r = ret.rolling(length).std()
        v = sigma_r / (np.abs(mu_r) + 1e-8)
        abs_ret_v = np.abs(ret) * v
        hyper_dist = np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))
        kappa_h = QuantumCore.tanh_clamp(hyper_dist.rolling(length).mean())

        lyap = np.log(np.abs(ret.diff()) + 1e-6).rolling(length).mean()
        lambda_norm = (lyap - (-5)) / 7

        def rolling_shannon(x):
            counts, _ = np.histogram(x, bins=bins)
            probs = counts / np.sum(counts)
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            max_ent = np.log(bins)
            return (max_ent - entropy) / max_ent if max_ent != 0 else 0
            
        entropy = ret.rolling(length).apply(rolling_shannon, raw=True)
        weighted = 0.4 * (kappa_h * lambda_norm) + 0.6 * entropy
        df['CHEDO'] = (2 / (1 + np.exp(-weighted * 4)) - 1).rolling(3).mean()

        # 2. OMEGA
        atr_val = QuantumCore.atr(df, 14)
        rho = df['Volume']
        u = np.abs(df['Close'].diff())
        mu = df['Close'].rolling(252).std()
        
        Re = (rho * u * atr_val) / (np.maximum(mu, 1e-10))
        df['Reynolds'] = Re
        
        v0 = mu
        e = np.abs(df['Close'] - df['Close'].shift(252))
        kappa = np.sqrt(np.maximum(0, v0 - e)) / atr_val
        psi = 1 / (np.cosh(kappa)**2)
        df['Omega_Mag'] = psi * QuantumCore.tanh_clamp(Re / (Re.rolling(252).mean() + 1e-10))
        
        return df

    @staticmethod
    def calc_pine_fear_greed_v4(df):
        """Composite F&G v4 (RSI + MACD + BB + ZLEMA)"""
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        hist = macd - signal
        macd_score = (50 + ((hist / (hist.rolling(100).std() + 1e-6)) * 16.6)).clip(0, 100)
        
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        upper = sma20 + (std20 * 2)
        lower = sma20 - (std20 * 2)
        bb_score = ((df['Close'] - lower) / (upper - lower + 1e-6) * 100).clip(0, 100)
        
        z_short = QuantumCore.zlema(df['Close'], 50)
        z_long = QuantumCore.zlema(df['Close'], 200)
        trend_score = np.where((df['Close'] > z_short) & (z_short > z_long), 75,
                      np.where(df['Close'] > z_short, 60,
                      np.where((df['Close'] < z_short) & (z_short < z_long), 25, 40)))
        
        df['FG_Index'] = (rsi * 0.30 + macd_score * 0.25 + bb_score * 0.25 + trend_score * 0.20)
        df['FG_Index'] = df['FG_Index'].rolling(3).mean().fillna(50)
        
        vol_ma = df['Volume'].rolling(20).mean()
        df['IS_FOMO'] = (df['Volume'] > vol_ma * 2.5) & (rsi > 70) & (df['Close'] > upper)
        df['IS_PANIC'] = (df['Close'].pct_change() < -0.03) & (df['Volume'] > vol_ma * 3.0)
        
        return df

    @staticmethod
    def calc_gann_hilo(df, length=3):
        """Gann HiLo Activator"""
        high_ma = df['High'].rolling(length).mean()
        low_ma = df['Low'].rolling(length).mean()
        
        close = df['Close'].values
        h_ma = high_ma.values
        l_ma = low_ma.values
        
        activator = np.zeros(len(df))
        trend = np.zeros(len(df))
        activator[0] = l_ma[0]
        trend[0] = 1
        
        for i in range(1, len(df)):
            if trend[i-1] == 1:
                if close[i] < activator[i-1]:
                    trend[i] = -1
                    activator[i] = h_ma[i]
                else:
                    trend[i] = 1
                    activator[i] = l_ma[i]
            else:
                if close[i] > activator[i-1]:
                    trend[i] = 1
                    activator[i] = l_ma[i]
                else:
                    trend[i] = -1
                    activator[i] = h_ma[i]
        
        df['Gann_Activator'] = activator
        df['Gann_Trend'] = trend
        return df

    @staticmethod
    def calc_dark_vector(df):
        """Dark Vector (SuperTrend + Chop)"""
        atr_val = QuantumCore.atr(df, 10)
        mult = 4.0
        hl2 = (df['High'] + df['Low']) / 2
        up = hl2 + (mult * atr_val)
        dn = hl2 - (mult * atr_val)
        
        st = np.zeros(len(df))
        trend = np.zeros(len(df))
        close = df['Close'].values
        up_val = up.values
        dn_val = dn.values
        
        st[0] = dn_val[0]; trend[0] = 1
        for i in range(1, len(df)):
            if close[i-1] > st[i-1]:
                st[i] = max(dn_val[i], st[i-1]) if close[i] > st[i-1] else up_val[i]
                trend[i] = 1 if close[i] > st[i-1] else -1
                if close[i] < dn_val[i] and trend[i-1] == 1:
                    st[i] = up_val[i]; trend[i] = -1
            else:
                st[i] = min(up_val[i], st[i-1]) if close[i] < st[i-1] else dn_val[i]
                trend[i] = -1 if close[i] < st[i-1] else 1
                if close[i] > up_val[i] and trend[i-1] == -1:
                    st[i] = dn_val[i]; trend[i] = 1
                    
        df['Vector_Trend'] = trend
        
        # Chop Index
        tr1 = QuantumCore.atr(df, 1)
        atr_sum = tr1.rolling(14).sum()
        h_max = df['High'].rolling(14).max()
        l_min = df['Low'].rolling(14).min()
        df['Chop_Index'] = 100 * np.log10(atr_sum / (h_max - l_min + 1e-9)) / np.log10(14)
        df['Vector_Locked'] = df['Chop_Index'] > 60
        
        return df

    @staticmethod
    def calc_market_cycle_master(df, mtf_df=None):
        """Market Cycle Master (HMA Trends + MTF)"""
        hma55 = QuantumCore.hma(df['Close'], 55)
        atr55 = QuantumCore.atr(df, 55)
        df['MCM_Upper'] = hma55 + (atr55 * 1.5)
        df['MCM_Lower'] = hma55 - (atr55 * 1.5)
        
        conditions = [(df['Close'] > df['MCM_Upper']), (df['Close'] < df['MCM_Lower'])]
        choices = [1, -1]
        df['MCM_Trend'] = np.select(conditions, choices, default=0)
        df['MCM_Trend'] = df['MCM_Trend'].replace(0, method='ffill')
        
        if mtf_df is not None and not mtf_df.empty:
            mtf_trend = mtf_df['MCM_Trend'].reindex(df.index, method='ffill')
            df['MCM_Signal'] = np.where(df['MCM_Trend'] == mtf_trend, df['MCM_Trend'], 0)
        else:
            df['MCM_Signal'] = df['MCM_Trend']
            
        df['MCM_Stop'] = np.where(df['MCM_Trend'] == 1, df['MCM_Lower'], df['MCM_Upper'])
        return df

    @staticmethod
    def calc_nexus_confluence(df):
        """Final Nexus Score (God Mode)"""
        basis = df['Close'].rolling(20).mean()
        dev = df['Close'].rolling(20).std() * 2
        u_bb = basis + dev; l_bb = basis - dev
        ma_kc = df['Close'].rolling(20).mean()
        range_ma = QuantumCore.atr(df, 20)
        u_kc = ma_kc + (1.5 * range_ma); l_kc = ma_kc - (1.5 * range_ma)
        df['Squeeze_On'] = (l_bb > l_kc) & (u_bb < u_kc)
        
        x = np.arange(20)
        df['Sqz_Mom'] = df['Close'].rolling(20).apply(lambda y: linregress(x, y)[0], raw=True) * 100
        
        rsi_val = df['FG_Index']
        vol_flow = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Flux_Matrix'] = ((rsi_val - 50) * vol_flow).ewm(span=3).mean()
        
        score = 0
        score += np.where(df['MCM_Trend'] == 1, 1, -1)
        score += np.where(df['Gann_Trend'] == 1, 1, -1)
        score += np.where(df['Vector_Trend'] == 1, 1, -1)
        score += np.sign(df['Sqz_Mom'])
        
        df['Nexus_Score'] = score
        return df

# ==========================================
# 3. DATA & INTELLIGENCE
# ==========================================
class DataEngine:
    @staticmethod
    @st.cache_data(ttl=60)
    def fetch_ohlcv(ticker, timeframe):
        period_map = {
            "15m": ("60d", "15m"), "1h": ("730d", "1h"),
            "4h": ("730d", "1h"), "1d": ("5y", "1d"), "1wk": ("5y", "1wk")
        }
        p, i = period_map.get(timeframe, ("1y", "1d"))
        try:
            df = yf.download(ticker, period=p, interval=i, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty: return None
            if timeframe == "4h":
                agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
                df = df.resample("4h").agg(agg).dropna()
            return df
        except: return None

    @staticmethod
    def get_mtf_data(ticker, current_tf):
        htf = "4h" if current_tf in ["15m", "1h"] else "1d"
        df = DataEngine.fetch_ohlcv(ticker, htf)
        if df is not None: df = QuantumCore.calc_market_cycle_master(df)
        return df

class Intelligence:
    """Broadcasting & AI Wrapper"""
    @staticmethod
    def broadcast(msg, token, chat_id):
        if not token or not chat_id: return False, "Missing Credentials"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            chunks = [msg[i:i+2000] for i in range(0, len(msg), 2000)]
            for chunk in chunks:
                requests.post(url, json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"})
            return True, "Broadcast Sent"
        except Exception as e: return False, str(e)

# ==========================================
# 4. RENDERERS
# ==========================================
def render_nexus_chart(df, ticker):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, 
        row_heights=[0.5, 0.15, 0.15, 0.2], 
        vertical_spacing=0.03,
        subplot_titles=(f"NEXUS QUANTUM: {ticker}", "Squeeze", "Omega", "Entropy/Flux")
    )
    # Price & Clouds
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MCM_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MCM_Lower'], fill='tonexty', fillcolor='rgba(0, 229, 255, 0.1)', line=dict(width=0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Gann_Activator'], line=dict(color='#FFD700', dash='dot')), row=1, col=1)
    
    # Subplots
    colors = ['#00E676' if v >= 0 else '#FF1744' for v in df['Sqz_Mom']]
    fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Omega_Mag'], fill='tozeroy', line=dict(color='#2979FF')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], line=dict(color='#EA80FC'), name="Entropy"), row=4, col=1)
    
    fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. MAIN APP
# ==========================================
def main():
    with st.sidebar:
        st.markdown('<div class="nexus-header">NEXUS<br>QUANTUM</div>', unsafe_allow_html=True)
        ticker = st.text_input("ASSET TICKER", "BTC-USD").upper()
        timeframe = st.selectbox("TIMEFRAME", ["15m", "1h", "4h", "1d", "1wk"], index=2)
        
        with st.expander("⚙️ SETTINGS", expanded=True):
            acc_size = st.number_input("Account ($)", value=10000)
            risk_pct = st.slider("Risk %", 0.1, 5.0, 1.0)
            
        with st.expander("📡 API KEYS"):
            gemini_key = st.text_input("Gemini", type="password")
            openai_key = st.text_input("OpenAI", type="password")
            tg_token = st.text_input("Telegram Token", type="password")
            tg_chat = st.text_input("Telegram Chat ID")
            
        if st.button("INITIALIZE NEXUS"): st.session_state['run'] = True

    if st.session_state.get('run', False):
        with st.spinner("Processing Quantum Stream..."):
            df = DataEngine.fetch_ohlcv(ticker, timeframe)
            if df is None: st.stop()
            mtf_df = DataEngine.get_mtf_data(ticker, timeframe)
            
            # Pipeline
            df = QuantumCore.calc_physics_engine(df)
            df = QuantumCore.calc_pine_fear_greed_v4(df)
            df = QuantumCore.calc_gann_hilo(df)
            df = QuantumCore.calc_dark_vector(df)
            df = QuantumCore.calc_market_cycle_master(df, mtf_df)
            df = QuantumCore.calc_nexus_confluence(df)
            last = df.iloc[-1]
            
            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PRICE", f"{last['Close']:.2f}", f"{df['Close'].pct_change().iloc[-1]*100:.2f}%")
            score = last['Nexus_Score']
            c2.metric("NEXUS SCORE", f"{score:.0f}/4", "STRONG BUY" if score>=3 else "STRONG SELL" if score<=-3 else "NEUTRAL")
            c3.metric("SENTIMENT", f"{last['FG_Index']:.0f}", "GREED" if last['FG_Index']>60 else "FEAR")
            c4.metric("VECTOR", "BULL" if last['Vector_Trend']==1 else "BEAR", "LOCKED" if last['Vector_Locked'] else "OPEN")
            
            # Tabs
            t1, t2, t3 = st.tabs(["📈 CHART & RISK", "🧠 AI & BROADCAST", "⚛️ PHYSICS LAB"])
            
            with t1:
                render_nexus_chart(df, ticker)
                st.markdown("### ⚖️ Risk Calculator")
                dist = abs(last['Close'] - last['MCM_Stop']) / last['Close']
                pos_size = (acc_size * (risk_pct/100)) / (last['Close'] * dist) if dist > 0 else 0
                k1, k2, k3 = st.columns(3)
                k1.metric("Stop Loss", f"${last['MCM_Stop']:.2f}")
                k2.metric("Position Size", f"{pos_size:.4f}")
                k3.metric("Risk Amount", f"${acc_size*(risk_pct/100):.2f}")
                
            with t2:
                c_ai, c_bc = st.columns(2)
                with c_ai:
                    st.subheader("AI Analysis")
                    if st.button("Generate Report"):
                        prompt = f"Analyze {ticker} ({timeframe}). Score: {score}. Sentiment: {last['FG_Index']}."
                        if gemini_key and genai:
                            genai.configure(api_key=gemini_key)
                            st.markdown(genai.GenerativeModel('gemini-pro').generate_content(prompt).text)
                        elif openai_key and OpenAI:
                            st.markdown(OpenAI(api_key=openai_key).chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}]).choices[0].message.content)
                with c_bc:
                    st.subheader("Broadcast Signal")
                    msg = st.text_area("Message", f"🔥 {ticker} SIGNAL\nScore: {score}/4\nPrice: {last['Close']}")
                    if st.button("Send Telegram"):
                        succ, res = Intelligence.broadcast(msg, tg_token, tg_chat)
                        if succ: st.success(res)
                        else: st.error(res)
            
            with t3:
                st.line_chart(df[['CHEDO', 'Reynolds']].tail(100))

if __name__ == "__main__":
    main()
