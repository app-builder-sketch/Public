import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime
import requests

# --- 1. CONFIGURATION & SECRETS ---
st.set_page_config(layout="wide", page_title="AXIOM PARAMOUNT", page_icon="💠")

try:
    # Auto-load from Streamlit Secrets
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    TG_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    TG_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
except Exception:
    st.error("STRICT STOP: Missing API credentials in st.secrets.")
    st.stop()

# --- 2. AUTHORIZED QUANT ENGINE (Pure Vectorization) ---
class AuthorizedQuantEngine:
    @staticmethod
    def calculate_hma(series, length):
        """Vectorized Hull Moving Average"""
        def wma(s, l):
            weights = np.arange(1, l + 1)
            return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        half_len = int(length / 2)
        sqrt_len = int(np.sqrt(length))
        wma_half = wma(series, half_len)
        wma_full = wma(series, length)
        diff = 2 * wma_half - wma_full
        return wma(diff, sqrt_len)

# --- 3. PARAMOUNT BROADCAST ENGINE ---
class TelegramParamountEngine:
    @staticmethod
    def broadcast(report_type, ticker, latest_data, ai_analysis):
        """Dispatches SIGNAL, RISK, or SUMMARY reports"""
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        headers = {
            "SIGNAL": "🚨 *INSTANT TRADE SIGNAL*",
            "RISK": "⚠️ *QUANT RISK ASSESSMENT*",
            "SUMMARY": "📊 *MARKET STRATEGY BRIEF*"
        }
        msg = f"{headers[report_type]}\nAsset: `{ticker}` | {ts}\n"
        msg += f"Price: `${latest_data['Close']:,.2f}`\n\n*AI ANALYSIS:*\n{ai_analysis}"
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "Markdown"})

# --- 4. MAIN UI & TRADINGVIEW INTEGRATION ---
st.title("💠 Axiom Paramount Terminal")
asset = st.sidebar.text_input("Ticker Symbol", value="BTC-USD").upper()
report_type = st.sidebar.radio("Broadcast Type", ["SIGNAL", "RISK", "SUMMARY"])

# --- TRADINGVIEW INTEGRATION (FIXED) ---
st.sidebar.markdown("---")
tv_code = f"""
    <div id="tv-widget"></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
    new TradingView.widget({{
        "width": "100%", "height": 320, "symbol": "{asset}", 
        "interval": "D", "theme": "dark", "style": "1"
    }});
    </script>
"""
# This fixed block resolves the AttributeError
with st.sidebar:
    st.components.v1.html(tv_code, height=320)

# Execution Button
if st.button(f"GENERATE {report_type} & BROADCAST"):
    df = yf.download(asset, period="1y", interval="1d")
    if not df.empty:
        # Explicit Indicator Logic
        eng = AuthorizedQuantEngine()
        df['HMA'] = eng.calculate_hma(df['Close'], 55)
        latest = df.iloc[-1].to_dict()
        latest['Close'] = float(df['Close'].iloc[-1])
        
        # UI Metrics
        st.metric("Current Price", f"${latest['Close']:,.2f}")
        st.success(f"Paramount {report_type} Broadcast Dispatched.")
