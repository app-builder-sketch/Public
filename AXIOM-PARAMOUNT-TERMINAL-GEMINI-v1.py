import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime
import requests
import json

# ==========================================
# 1. CORE CONFIGURATION & SECRETS (Protocol #6)
# ==========================================
st.set_page_config(layout="wide", page_title="AXIOM PARAMOUNT TERMINAL", page_icon="💠")

# Auto-load secrets from Streamlit
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    TG_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    TG_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
except Exception:
    st.error("STRICT STOP: Missing API credentials in st.secrets.")
    st.stop()

# ==========================================
# 2. UI/UX ENGINE (DarkPool Aesthetic)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .ticker-wrap { width: 100%; overflow: hidden; background: #0a0a0a; border-bottom: 1px solid #00F0FF; height: 40px; display: flex; align-items: center; }
    .ticker { display: inline-block; animation: marquee 30s linear infinite; white-space: nowrap; color: #00F0FF; }
    @keyframes marquee { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }
    div[data-testid="stMetric"] { background: rgba(0, 240, 255, 0.02); border-left: 2px solid #00F0FF; padding: 15px; }
</style>
""", unsafe_allow_html=True)

def render_marquee():
    st.markdown('<div class="ticker-wrap"><div class="ticker">💠 AXIOM QUANT ONLINE | SYSTEM STATUS: OPTIMAL | PARAMOUNT BROADCAST ACTIVE | QUANTUM ENGINES ENGAGED 💠</div></div>', unsafe_allow_html=True)

# ==========================================
# 3. AUTHORIZED QUANT ENGINE (Protocol #1, #2, #5)
# ==========================================
class AuthorizedQuantEngine:
    """Vectorized implementation of indicators explicitly provided in Knowledge."""
    
    @staticmethod
    def calculate_hma(series, length):
        def wma(s, l):
            weights = np.arange(1, l + 1)
            return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        
        half_len = int(length / 2)
        sqrt_len = int(np.sqrt(length))
        wma_half = wma(series, half_len)
        wma_full = wma(series, length)
        diff = 2 * wma_half - wma_full
        return wma(diff, sqrt_len)

    @staticmethod
    def calc_chedo_entropy(df, length=50):
        """Axiom Explicit: Entropy calculation."""
        c = df['Close'].values
        log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
        mu = pd.Series(log_ret).rolling(length).mean().values
        sigma = pd.Series(log_ret).rolling(length).std().values
        v = sigma / (np.abs(mu) + 1e-9)
        abs_ret_v = np.abs(log_ret) * v
        hyper_dist = np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))
        raw = np.tanh(pd.Series(hyper_dist).rolling(length).mean().values)
        return 2 / (1 + np.exp(-raw * 4)) - 1

    @staticmethod
    def calc_rqzo_relativity(df, harmonics=25):
        """Axiom Explicit: Relativistic Time-Dilation."""
        src = df['Close']
        mn, mx = src.rolling(100).min(), src.rolling(100).max()
        norm = (src - mn) / (mx - mn + 1e-9)
        v = np.abs(norm.diff())
        gamma = 1 / np.sqrt(1 - (np.clip(v, 0, 0.049)/0.05)**2)
        idx = np.arange(len(df))
        tau = (idx % 100) / gamma.fillna(1.0)
        zeta = np.zeros(len(df))
        for n in range(1, harmonics + 1):
            zeta += (n**-0.5) * np.sin(tau * np.log(n))
        return pd.Series(zeta).fillna(0)

    @staticmethod
    def calc_apex_flux(df, length=14):
        """Vector Explicit: Candle Efficiency & Volume Flux."""
        rg = df['High'] - df['Low']
        body = np.abs(df['Close'] - df['Open'])
        eff = (body / rg.replace(0, 1)).ewm(span=length).mean()
        v_rat = df['Volume'] / df['Volume'].rolling(55).mean()
        direction = np.sign(df['Close'] - df['Open'])
        return (direction * eff * v_rat).ewm(span=5).mean()

# ==========================================
# 4. PARAMOUNT BROADCAST ENGINE (Protocol #4)
# ==========================================
class TelegramParamountEngine:
    @staticmethod
    def broadcast(report_type, ticker, latest_data, ai_analysis):
        """Routes 3 distinct institutional report types."""
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        headers = {
            "SIGNAL": "🚨 *INSTANT TRADE SIGNAL*",
            "RISK": "⚠️ *QUANT RISK ASSESSMENT*",
            "SUMMARY": "📊 *MARKET STRATEGY BRIEF*"
        }
        
        emoji = "🟢" if latest_data['Trend'] == 1 else "🔴"
        
        msg = f"{headers[report_type]}\n"
        msg += f"Asset: `{ticker}` | {ts}\n"
        msg += f"Price: `${latest_data['Close']:,.2f}`\n\n"
        
        if report_type == "SIGNAL":
            msg += f"Wave: {emoji} {'BULLISH' if emoji=='🟢' else 'BEARISH'}\n"
            msg += f"Flux: `{latest_data['Flux']:.4f}`\n"
        elif report_type == "RISK":
            msg += f"Entropy (CHEDO): `{latest_data['CHEDO']:.4f}`\n"
            msg += f"Relativity (RQZO): `{latest_data['RQZO']:.4f}`\n"
        
        msg += f"\n*AI SYNTHESIS:*\n{ai_analysis}"
        
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "Markdown"})

# ==========================================
# 5. AI ANALYST SYNTHESIS
# ==========================================
def get_ai_synthesis(ticker, data, report_type):
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompts = {
        "SIGNAL": "Analyze technical efficiency and trend direction.",
        "RISK": "Analyze mathematical entropy and market chaos levels.",
        "SUMMARY": "Synthesize a broad market outlook based on physics indicators."
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are a Quant Architect. {prompts[report_type]}"},
                {"role": "user", "content": f"Asset: {ticker}, Data: {data}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Synthesis Error: {e}"

# ==========================================
# 6. MAIN APPLICATION EXECUTION
# ==========================================
render_marquee()
st.title("💠 Axiom Paramount Terminal")

# Sidebar
asset = st.sidebar.text_input("Ticker Symbol", value="BTC-USD").upper()
tf = st.sidebar.selectbox("Interval", ["1h", "4h", "1d"], index=1)
report_type = st.sidebar.radio("Broadcast Report Type", ["SIGNAL", "RISK", "SUMMARY"])

if st.button("RUN ENGINE & BROADCAST"):
    with st.spinner(f"Computing Logic for {asset}..."):
        # Data Ingestion
        df = yf.download(asset, period="1y", interval=tf)
        if df.empty:
            st.error("Data failed to load.")
            st.stop()
            
        # Calculation
        eng = AuthorizedQuantEngine()
        df['HMA'] = eng.calculate_hma(df['Close'], 55)
        df['CHEDO'] = eng.calc_chedo_entropy(df)
        df['RQZO'] = eng.calc_rqzo_relativity(df)
        df['Flux'] = eng.calc_apex_flux(df)
        df['Trend'] = np.where(df['Close'] > df['HMA'], 1, -1)
        
        latest = df.iloc[-1].to_dict()
        latest['Close'] = float(df['Close'].iloc[-1])
        
        # AI Synthesis
        ai_out = get_ai_synthesis(asset, latest, report_type)
        
        # Broadcast
        TelegramParamountEngine.broadcast(report_type, asset, latest, ai_out)
        
        # UI Display
        col1, col2, col3 = st.columns(3)
        col1.metric("Entropy (CHEDO)", f"{latest['CHEDO']:.4f}")
        col2.metric("Relativity (RQZO)", f"{latest['RQZO']:.4f}")
        col3.metric("Flux Vector", f"{latest['Flux']:.4f}")
        
        # Plots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='yellow', width=1), name="HMA Trend"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], fill='tozeroy', name="Entropy"), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Flux'], name="Vector Flux"), row=3, col=1)
        
        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🤖 Paramount AI Analysis")
        st.info(ai_out)
        st.success(f"Paramount {report_type} Broadcast Dispatched.")

# TradingView Widget Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Live View")
tv_code = f"""<div id="tv-widget"></div><script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"width": "100%", "height": 300, "symbol": "{asset}", "interval": "D", "theme": "dark"}});</script>"""
st.sidebar.components.v1.html(tv_code, height=320)
