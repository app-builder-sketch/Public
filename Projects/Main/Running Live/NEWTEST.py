import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import math
import json
import os
from scipy.signal import argrelextrema
from scipy.stats import linregress
from datetime import datetime
import streamlit.components.v1 as components

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="TITAN OMNI V3",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# ENVIRONMENT & SECRETS SETUP
# ===============================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # dotenv not installed, relying on st.secrets or os.environ

def load_key(key_name):
    """
    Priority:
    1. Streamlit Secrets (st.secrets)
    2. Environment Variables (os.environ / .env)
    3. Empty String (User must input manually)
    """
    # Check Streamlit Secrets
    if key_name in st.secrets:
        return st.secrets[key_name]
    
    # Check OS Environment
    return os.getenv(key_name, "")

# ===============================
# SESSION STATE
# ===============================
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
if "ai_report" not in st.session_state:
    st.session_state.ai_report = ""

# ===============================
# OPTIONAL LIBS
# ===============================
try:
    import yfinance as yf
except:
    st.error("Install yfinance: pip install yfinance")
    st.stop()

try:
    import google.generativeai as genai
except:
    genai = None

try:
    from openai import OpenAI
except:
    OpenAI = None

# ===============================
# TELEGRAM ENGINE (FULL)
# ===============================
class TelegramEngine:
    API_URL = "https://api.telegram.org/bot{}/sendMessage"

    @staticmethod
    def escape_md(text: str) -> str:
        # Escapes special characters for Telegram MarkdownV2
        escape_chars = r"_*[]()~`>#+-=|{}.!"
        return "".join("\\" + c if c in escape_chars else c for c in text)

    @staticmethod
    def send(token: str, chat_id: str, message: str):
        if not token or not chat_id:
            return False, "Missing Telegram credentials"

        payload = {
            "chat_id": chat_id,
            "text": TelegramEngine.escape_md(message),
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True
        }

        try:
            r = requests.post(
                TelegramEngine.API_URL.format(token),
                json=payload,
                timeout=10
            )
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}: {r.text}"

            res = r.json()
            if not res.get("ok"):
                return False, f"Telegram API Error: {res}"

            return True, "Sent"
        except Exception as e:
            return False, str(e)

# ===============================
# DATA ENGINE
# ===============================
class DataEngine:
    @staticmethod
    @st.cache_data(ttl=60)
    def fetch(ticker, timeframe):
        tf_map = {
            "15m": ("60d", "15m"),
            "1h": ("730d", "1h"),
            "4h": ("730d", "1h"),
            "1d": ("5y", "1d"),
            "1wk": ("10y", "1wk")
        }
        period, interval = tf_map.get(timeframe, ("1y", "1d"))
        
        # Download data
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty:
            return None

        # FIX: Handle MultiIndex columns (common in new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.droplevel(1)
            except:
                pass

        # Handle 4h resampling from 1h data
        if timeframe == "4h":
            df = df.resample("4h").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }).dropna()
            
        return df

# ===============================
# CORE STRATEGY (MINIMAL SAFE)
# ===============================
class QuantumCore:
    @staticmethod
    def atr(df, n=14):
        tr = pd.concat([
            df["High"] - df["Low"],
            abs(df["High"] - df["Close"].shift()),
            abs(df["Low"] - df["Close"].shift())
        ], axis=1).max(axis=1)
        return tr.rolling(n).mean()

    @staticmethod
    def pipeline(df):
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["EMA200"] = df["Close"].ewm(span=200).mean()
        df["ATR"] = QuantumCore.atr(df)

        df["Trend"] = np.where(df["EMA50"] > df["EMA200"], 1, -1)
        # Calculate trailing stop logic
        df["Stop"] = df["Close"] - df["ATR"] * df["Trend"]
        df["GM_Score"] = df["Trend"]
        return df.dropna()

# ===============================
# INTELLIGENCE
# ===============================
class Intelligence:
    @staticmethod
    def signal_message(ticker, tf, last):
        price = last["Close"]
        stop = last["Stop"]
        direction = "🐂 LONG" if last["GM_Score"] > 0 else "🐻 SHORT"
        risk = abs(price - stop)

        return f"""
SIGNAL: {direction}
Asset: {ticker}
Timeframe: {tf}

Entry: {price:.2f}
Stop: {stop:.2f}
TP1: {price + risk*1.5:.2f}
TP2: {price + risk*3:.2f}

TITAN OMNI V3
"""

    @staticmethod
    def outlook_message(ticker, tf, last):
        state = "BULLISH" if last["GM_Score"] > 0 else "BEARISH"
        return f"""
TITAN MARKET OUTLOOK
{ticker} ({tf})

Bias: {state}
Price: {last['Close']:.2f}
Stop: {last['Stop']:.2f}
"""

# ===============================
# UI
# ===============================
def main():
    st.title("💠 TITAN OMNI V3")

    with st.sidebar:
        ticker = st.text_input("Ticker", "BTC-USD")
        timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d", "1wk"])

        st.markdown("### 🔐 API KEYS")
        # Auto-load keys from environment/secrets, allow override
        gem_key = st.text_input("Gemini", value=load_key("GEMINI_API_KEY"), type="password")
        oai_key = st.text_input("OpenAI", value=load_key("OPENAI_API_KEY"), type="password")

        st.markdown("### 📡 Telegram")
        # Auto-load keys from environment/secrets, allow override
        tg_token = st.text_input("Bot Token", value=load_key("TELEGRAM_TOKEN"), type="password")
        tg_chat = st.text_input("Chat ID", value=load_key("TELEGRAM_CHAT_ID"))

        if st.button("🧪 Test Telegram"):
            ok, msg = TelegramEngine.send(
                tg_token, tg_chat, "✅ TITAN OMNI Telegram connected"
            )
            if ok:
                st.success(msg)
            else:
                st.error(msg)

        if st.button("🚀 Run Analysis"):
            st.session_state.run_analysis = True

    if not st.session_state.run_analysis:
        st.info("Awaiting execution")
        return

    df = DataEngine.fetch(ticker, timeframe)
    if df is None:
        st.error("No data found or connection failed.")
        return

    df = QuantumCore.pipeline(df)
    last = df.iloc[-1]

    st.metric("Price", f"{last['Close']:.2f}")
    st.metric("GM Score", last["GM_Score"])

    # Charting
    fig = go.Figure(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["Stop"], name="Stop", line=dict(color='orange')))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 📡 Broadcast")

    sig_msg = Intelligence.signal_message(ticker, timeframe, last)
    out_msg = Intelligence.outlook_message(ticker, timeframe, last)

    st.text_area("Signal Preview", sig_msg, height=200)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("SEND SIGNAL"):
            ok, msg = TelegramEngine.send(tg_token, tg_chat, sig_msg)
            if ok:
                st.success("Signal sent")
            else:
                st.error(msg)
    
    with col2:
        if st.button("SEND OUTLOOK"):
            ok, msg = TelegramEngine.send(tg_token, tg_chat, out_msg)
            if ok:
                st.success("Outlook sent")
            else:
                st.error(msg)

if __name__ == "__main__":
    main()
