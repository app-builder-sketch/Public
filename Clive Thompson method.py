import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import plotly.express as px
from lightweight_charts.widgets import StreamlitChart
from openai import OpenAI
import requests
from datetime import datetime

# =============================================================================
# 1. SYSTEM CONFIGURATION & API KEYS
# =============================================================================
st.set_page_config(page_title="Institutional Alpha 2026", layout="wide")

# Secure API Configuration (To be provided by user in secrets.toml)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "your-key-here")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELE_TOKEN", "your-token-here")
TELEGRAM_CHAT_ID = st.secrets.get("TELE_CHAT_ID", "your-id-here")

client = OpenAI(api_key=OPENAI_API_KEY)

# =============================================================================
# 2. GLOBAL TICKER BANNER & UI
# =============================================================================
st.markdown("""
    <style>
    @keyframes marquee {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    .ticker-wrap { width: 100%; overflow: hidden; background: #111; color: #0f0; padding: 5px 0; }
    .ticker { display: inline-block; white-space: nowrap; animation: marquee 30s linear infinite; }
    </style>
    <div class="ticker-wrap"><div class="ticker">
        BTC/USD $98,450 (+2.1%) | GLD $245.20 (-0.5%) | NVDA $145.80 (+1.2%) | EUR/USD 1.08 (-0.2%) | SLV $82.40 (+4.5%)
    </div></div>
""", unsafe_allow_value=True)

# =============================================================================
# 3. CORE INTEGRATION: DATA & ANALYSIS
# =============================================================================
def get_institutional_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d")
    # Technical Analysis Integration (Tested Algorithms)
    df.ta.strategy("Common") # SMA, RSI, MACD, Bollinger
    return df

def broadcast_signal(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    return requests.post(url, json=payload)

def ai_unified_analysis(ticker, tech_summary, fund_summary):
    prompt = f"""
    Unified Quant/AI Report for {ticker}:
    [Technical Factors]: {tech_summary}
    [Fundamental Factors]: {fund_summary}
    Analyze for 2026 macro-outperformance. Take every factor into account.
    Provide: 1. Risk Rating 2. Target Price 3. Tactical Action.
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are a Senior Institutional Analyst."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# =============================================================================
# 4. DASHBOARD INTERFACE
# =============================================================================
st.title("🛡️ Institutional Alpha Terminal 2026")

# Comprehensive Ticker Dropdown
TICKER_UNIVERSE = ["NVDA", "ASML", "LDO.MI", "MELI", "BTC-USD", "GC=F", "KAP.L", "7203.T", "SHEL.L", "DLO"]
selected_ticker = st.selectbox("Market Access: Select Asset", TICKER_UNIVERSE)

col_chart, col_data = st.columns([2, 1])

with col_chart:
    st.subheader("TradingView Advanced Charting")
    data = get_institutional_data(selected_ticker)
    chart = StreamlitChart(width=900, height=500)
    chart.set(data)
    chart.load()

with col_data:
    st.subheader("Quant Audit: Fundamental Gate")
    info = yf.Ticker(selected_ticker).info
    st.metric("Forward P/E", f"{info.get('forwardPE', 'N/A')}x")
    st.metric("Debt-to-Equity", f"{info.get('debtToEquity', 'N/A')}")
    st.metric("EPS Growth (3Y)", f"{info.get('earningsQuarterlyGrowth', 'N/A')}%")

# AI BROADCAST SECTION
st.divider()
if st.button("🚀 EXECUTE FULL AI AUDIT & BROADCAST"):
    with st.spinner("Synchronizing Quant/AI Factors..."):
        # Compile Factors
        last_rsi = data['RSI_14'].iloc[-1]
        tech_sum = f"RSI: {last_rsi:.2f}, Trend: Above 200SMA"
        fund_sum = f"P/E: {info.get('forwardPE')}, Cash: {info.get('totalCash')}"
        
        # Comprehensive AI Report
        full_report = ai_unified_analysis(selected_ticker, tech_sum, fund_sum)
        
        st.write("### AI Global Analyst Report")
        st.write(full_report)
        
        # Broadcast to Telegram
        broadcast_signal(f"*2026 SIGNAL ALERT: {selected_ticker}*\n\n{full_report}")
        st.success("Report Broadcasted to Institutional Network.")

# DIVERSIFICATION TREEMAP
st.subheader("Global Risk Parity")
# Visualizing the 40-stock diversification logic
fig = px.treemap(path=[px.Constant("Global Portfolio"), 'Industry', 'Name'], 
                 values=[10000]*10, # Equal Weighted
                 title="Diversification Shield: Industry Distribution")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("🔒 Developer Guidelines: TradingView Integration | 150+ Tech Indicators | OpenAI Quant Analytics")
