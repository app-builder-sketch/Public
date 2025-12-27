# ==================================================================================================
# TITAN TERMINAL: ABSOLUTE EDITION (WITH HEDGE MODE & EXTENDED AI)
# --------------------------------------------------------------------------------------------------
# NO OMISSIONS. NO ASSUMPTIONS. FULL SCRIPT.
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
import sqlite3
from scipy.stats import linregress

# ==========================================
# 1. CORE CONFIG & DATABASE
# ==========================================
st.set_page_config(layout="wide", page_title="🏦 Titan Absolute", page_icon="👁️")

def init_db():
    conn = sqlite3.connect('titan_vault.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS signals 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp DATETIME, symbol TEXT, interval TEXT, 
                  score REAL, price REAL, message TEXT, logic_blob TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .report-card {
        background-color: #111; border-left: 4px solid #00F0FF;
        padding: 20px; border-radius: 4px; margin-bottom: 15px;
    }
    .highlight { color: #00F0FF; font-weight: bold; }
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.02); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. THE PHYSICS & HEDGE ENGINES
# ==========================================
def calc_physics_metrics(df):
    """Integrates Axiom Quant Physics into the Dataframe."""
    c = df['Close'].values
    log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
    # CHEDO (Entropy)
    v = pd.Series(log_ret).rolling(50).std() / (np.abs(pd.Series(log_ret).rolling(50).mean()) + 1e-9)
    df['CHEDO'] = np.tanh(pd.Series(np.log(np.abs(log_ret) * v + 1e-9)).rolling(50).mean())
    # RQZO (Relativity/Velocity)
    mn, mx = df['Close'].rolling(100).min(), df['Close'].rolling(100).max()
    norm = (df['Close'] - mn) / (mx - mn + 1e-9)
    df['RQZO'] = norm.diff().rolling(25).mean() * 10
    # Apex Flux (Vector Efficiency)
    body = np.abs(df['Close'] - df['Open'])
    range_ = df['High'] - df['Low']
    eff = np.where(range_ == 0, 0, body / range_)
    df['Apex_Flux'] = pd.Series(eff).ewm(span=14).mean() * np.sign(df['Close'] - df['Open'])
    return df

def get_hedge_correlations(target_ticker):
    """Constructs the Correlation Matrix for Hedge Mode."""
    macro_assets = {
        "S&P 500": "SPY", "Nasdaq": "QQQ", "TLT Bonds": "TLT", 
        "Gold": "GLD", "Dollar (DXY)": "DX-Y.NYB", "VIX": "^VIX", "Bitcoin": "BTC-USD"
    }
    tickers = list(macro_assets.values())
    if target_ticker not in tickers: tickers.append(target_ticker)
    
    data = yf.download(tickers, period="1y", interval="1d", progress=False)['Close']
    corr = data.corr()[target_ticker].sort_values()
    return corr

# ==========================================
# 3. THE AI ANALYST: ABSOLUTE EDITION
# ==========================================
def run_absolute_ai(df, ticker, interval):
    """AI Analyst with Physics and Hedge Insights."""
    if not st.session_state.get('api_key'): return "⚠️ OpenAI Key Required."
    
    last = df.iloc[-1]
    physics_state = "Superconductor" if abs(last['Apex_Flux']) > 0.6 else "Friction-Heavy"
    entropy_risk = "CRITICAL CHAOS" if abs(last['CHEDO']) > 0.8 else "Stable"
    
    prompt = f"""
    Act as a Quantum Trading Lead. Analyze {ticker} ({interval}).
    Price: {last['Close']:.2f}. Titan Score: {last.get('GM_Score', 0)}.
    
    QUANTUM DATA:
    - Entropy (CHEDO): {last['CHEDO']:.2f} ({entropy_risk})
    - Velocity (RQZO): {last['RQZO']:.2f}
    - Flux Vector: {last['Apex_Flux']:.2f} ({physics_state})
    
    MISSION:
    1. Define the current market regime (Energy vs Chaos).
    2. Explain the risk of holding this asset without a hedge.
    3. Use technical terminology (Order Blocks, Flux, Entropy) but remain engaging with Emojis.
    """
    
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
        return res.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

# ==========================================
# 4. MAIN APP INTERFACE
# ==========================================
st.sidebar.title("💠 Titan Absolute")
ticker = st.sidebar.text_input("Asset Ticker", value="BTC-USD").upper()
interval = st.sidebar.selectbox("Interval", ["1h", "4h", "1d"], index=1)

if st.sidebar.button("⚡ EXECUTE SYSTEM"):
    df = yf.download(ticker, period="60d", interval=interval)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    if not df.empty:
        df = calc_physics_metrics(df)
        last = df.iloc[-1]
        
        # --- PHYSICS DASHBOARD ---
        m1, m2, m3 = st.columns(3)
        m1.metric("CHEDO (ENTROPY)", f"{last['CHEDO']:.3f}")
        m2.metric("RQZO (RELATIVITY)", f"{last['RQZO']:.3f}")
        m3.metric("APEX FLUX", f"{last['Apex_Flux']:.3f}")

        tab1, tab2, tab3 = st.tabs(["📊 Physics Terminal", "🛡️ Hedge Mode", "🤖 AI Report"])

        with tab1:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], name="Entropy", line=dict(color="#00F0FF")), row=2, col=1)
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("🛡️ Hedge Mode: Correlation Matrix")
            corr_data = get_hedge_correlations(ticker)
            fig_corr = px.bar(x=corr_data.values, y=corr_data.index, orientation='h', 
                               color=corr_data.values, color_continuous_scale='RdBu',
                               title="Asset Correlation to Target")
            st.plotly_chart(fig_corr, use_container_width=True)
            
            negative_corrs = corr_data[corr_data < 0].index.tolist()
            if negative_corrs:
                st.info(f"💡 **Delta-Neutral Strategy:** Consider hedging {ticker} with a long position in **{negative_corrs[0]}**.")
            else:
                st.warning("⚠️ High market correlation detected. No natural hedge currently exists in the standard basket.")

        with tab3:
            st.subheader("🤖 Quantum AI Briefing")
            report = run_absolute_ai(df, ticker, interval)
            st.markdown(f'<div class="report-card">{report}</div>', unsafe_allow_html=True)
