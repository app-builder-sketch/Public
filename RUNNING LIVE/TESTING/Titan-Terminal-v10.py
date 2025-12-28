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
import time
import math
import io
import xlsxwriter
import streamlit.components.v1 as components
import asyncio
import logging

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM UI
# ==========================================
st.set_page_config(layout="wide", page_title="🏦 Titan Terminal v10", page_icon="👁️")

# --- CUSTOM CSS FOR "DARKPOOL" & "MOBILE" AESTHETIC ---
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #e0e0e0;
    font-family: 'Roboto Mono', monospace;
}
.title-glow {
    font-size: 3em;
    font-weight: bold;
    color: #ffffff;
    text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 40px #00ff00;
    margin-bottom: 20px;
}
/* METRICS: Glassmorphism */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    padding: 10px;
    border-radius: 8px;
    transition: transform 0.2s;
}
div[data-testid="stMetric"]:hover {
    transform: scale(1.02);
    border-color: #00ff00;
}
div[data-testid="stMetricValue"] {
    font-size: 1.2rem !important;
    font-weight: 700;
}
/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #161b22;
    border-radius: 4px 4px 0px 0px;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
    border: 1px solid #30363d;
    color: #8b949e;
}
.stTabs [aria-selected="true"] {
    background-color: #0e1117;
    color: #00ff00;
    border-bottom: 2px solid #00ff00;
}
/* KIMI MOBILE REPORT CARDS */
.report-card {
    background-color: #161b22;
    border-left: 4px solid #00ff00;
    padding: 15px;
    border-radius: 4px;
    margin-bottom: 10px;
    font-family: 'Roboto Mono', monospace;
}
.report-header { font-size: 1.1rem; font-weight: bold; color: #fff; margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 5px; }
.report-item { margin-bottom: 5px; font-size: 0.9rem; color: #aaa; }
.highlight { color: #00ff00; font-weight: bold; }
/* AI CARD */
.ai-card {
    background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%);
    border: 1px solid #45a29e;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="title-glow">👁️ Titan Terminal v10</div>', unsafe_allow_html=True)
st.markdown("##### *Master Analysis Engine | Axiom Physics | Enterprise Broadcast*")
st.markdown("---")

# --- API Key Management ---
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if "OPENAI_API_KEY" in st.secrets:
    st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
else:
    if not st.session_state.api_key:
        st.session_state.api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key here to unlock the AI Analyst features."
        )

# ==========================================
# 2. DATABASE LAYER (From Kimis-Signals)
# ==========================================
class SignalDatabase:
    """SQLite backend for signal persistence."""
    def __init__(self, db_path: str = "titan_signals.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timestamp TEXT,
                    direction TEXT,
                    entry_price REAL,
                    stop_price REAL,
                    tp1 REAL,
                    tp2 REAL,
                    tp3 REAL,
                    confidence_score INTEGER,
                    outcome TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def save_signal(self, symbol: str, signal_data: dict, outcome: str = "PENDING"):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO signals (symbol, timestamp, direction, entry_price, stop_price, 
                                   tp1, tp2, tp3, confidence_score, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                datetime.datetime.now().isoformat(),
                signal_data.get('direction', 'NEUTRAL'),
                signal_data.get('entry', 0),
                signal_data.get('stop', 0),
                signal_data.get('tp1', 0),
                signal_data.get('tp2', 0),
                signal_data.get('tp3', 0),
                signal_data.get('conf', 0),
                outcome
            ))

# Initialize DB
db = SignalDatabase()

# ==========================================
# 3. DATA ENGINE & UTILS
# ==========================================
@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    if "-" in ticker or "=" in ticker or "^" in ticker:
        return None
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info:
            return None
        return {
            "Market Cap": info.get("marketCap", 0),
            "P/E Ratio": info.get("trailingPE", 0),
            "Rev Growth": info.get("revenueGrowth", 0),
            "Debt/Equity": info.get("debtToEquity", 0),
            "Summary": info.get("longBusinessSummary", "No Data Available"),
            "Sector": info.get("sector", "Unknown"),
            "Industry": info.get("industry", "Unknown"),
            "ROE": info.get("returnOnEquity", 0)
        }
    except:
        return None

def safe_download(ticker, period, interval):
    try:
        dl_interval = "1h" if interval == "4h" else interval
        df = yf.download(ticker, period=period, interval=dl_interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            else:
                return None
        return df
    except:
        return None

@st.cache_data(ttl=300)
def get_macro_data():
    groups = {
        "🇺🇸 US Equities": {"S&P 500": "SPY", "Nasdaq 100": "QQQ", "Dow Jones": "^DJI"},
        "⚠️ Risk Assets": {"Bitcoin": "BTC-USD", "Semis (SMH)": "SMH", "Junk Bonds": "HYG"},
        "🏦 Rates & Vol": {"10Y Yield": "^TNX", "VIX": "^VIX", "DXY": "DX-Y.NYB"},
        "🥇 Metals/Energy": {"Gold": "GC=F", "Oil": "CL=F", "Silver": "SI=F"}
    }
    all_tickers = [sym for g in groups.values() for sym in g.values()]
    try:
        data = yf.download(all_tickers, period="5d", interval="1d", group_by='ticker', progress=False)
        prices, changes = {}, {}
        for sym in all_tickers:
            try:
                df = data[sym] if len(all_tickers) > 1 else data
                if len(df) >= 2:
                    col = 'Close' if 'Close' in df.columns else 'Adj Close'
                    curr, prev = df[col].iloc[-1], df[col].iloc[-2]
                    prices[sym] = curr
                    changes[sym] = ((curr - prev) / prev) * 100
            except: continue
        return groups, prices, changes
    except: return groups, {}, {}

# ==========================================
# 4. MATH & INDICATOR LIBRARY (Merged Titan + Axiom + Kimis)
# ==========================================
def calculate_wma(series, length):
    w = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)

def calculate_hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = calculate_wma(series, half_length)
    wma_full = calculate_wma(series, length)
    return calculate_wma(2 * wma_half - wma_full, sqrt_length)

def calculate_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def calculate_rsi(series, length=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def calculate_supertrend(df, period=10, multiplier=3.0):
    atr = calculate_atr(df, period)
    hl2 = (df['High'] + df['Low']) / 2
    upper = hl2 + (multiplier * atr)
    lower = hl2 - (multiplier * atr)
    close = df['Close'].values
    st = np.zeros(len(df)); trend = np.zeros(len(df))
    st[0] = lower.iloc[0]; trend[0] = 1
    
    for i in range(1, len(df)):
        if close[i-1] > st[i-1]:
            st[i] = max(lower.iloc[i], st[i-1]) if close[i] > st[i-1] else upper.iloc[i]
            trend[i] = 1 if close[i] > st[i-1] else -1
            if close[i] < lower.iloc[i] and trend[i-1] == 1:
                st[i] = upper.iloc[i]; trend[i] = -1
        else:
            st[i] = min(upper.iloc[i], st[i-1]) if close[i] < st[i-1] else lower.iloc[i]
            trend[i] = -1 if close[i] < st[i-1] else 1
            if close[i] > upper.iloc[i] and trend[i-1] == -1:
                st[i] = lower.iloc[i]; trend[i] = 1
    return pd.Series(st, index=df.index), pd.Series(trend, index=df.index)

# --- AXIOM PHYSICS ENGINES ---
def tanh(x): return np.tanh(np.clip(x, -20, 20))

def calc_chedo(df, length=50):
    c = df['Close'].values
    log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
    mu = pd.Series(log_ret).rolling(length).mean().values
    sigma = pd.Series(log_ret).rolling(length).std().values
    v = sigma / (np.abs(mu) + 1e-9)
    abs_ret_v = np.abs(log_ret) * v
    hyper_dist = np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))
    kappa_h = tanh(pd.Series(hyper_dist).rolling(length).mean().values)
    diff_ret = np.diff(log_ret, prepend=0)
    lyap = np.log(np.abs(diff_ret) + 1e-9)
    lambda_n = tanh((pd.Series(lyap).rolling(length).mean().values + 5) / 7)
    ent = pd.Series(log_ret**2).rolling(length).sum().values
    ent_n = tanh(ent * 10)
    raw = (0.4 * kappa_h) + (0.3 * lambda_n) + (0.3 * ent_n)
    df['CHEDO'] = 2 / (1 + np.exp(-raw * 4)) - 1
    return df

def calc_rqzo(df, harmonics=25):
    src = df['Close']
    mn, mx = src.rolling(100).min(), src.rolling(100).max()
    norm = (src - mn) / (mx - mn + 1e-9)
    v = np.abs(norm.diff())
    c_limit = 0.05
    gamma = 1 / np.sqrt(1 - (np.minimum(v, c_limit*0.99)/c_limit)**2)
    idx = np.arange(len(df))
    tau = (idx % 100) / gamma.fillna(1.0)
    zeta = np.zeros(len(df))
    for n in range(1, harmonics + 1):
        amp = n ** -0.5
        theta = tau * np.log(n)
        zeta += amp * np.sin(theta)
    df['RQZO'] = pd.Series(zeta).fillna(0)
    return df

def calc_apex_flux(df, length=14):
    rg = df['High'] - df['Low']
    body = np.abs(df['Close'] - df['Open'])
    eff_raw = np.where(rg == 0, 0, body / rg)
    eff_series = pd.Series(eff_raw, index=df.index) 
    eff_sm = eff_series.ewm(span=length).mean()
    vol_avg = df['Volume'].rolling(55).mean()
    v_rat_raw = np.where(vol_avg == 0, 1, df['Volume'] / vol_avg)
    v_rat_series = pd.Series(v_rat_raw, index=df.index)
    direction = np.sign(df['Close'] - df['Open'])
    raw = direction * eff_sm * v_rat_series
    df['Apex_Flux'] = raw.ewm(span=5).mean()
    return df

# --- INDICATOR AGGREGATOR ---
def calc_indicators(df, apex_mult=1.5):
    df = df.copy()
    
    # 1. Base
    df['HMA'] = calculate_hma(df['Close'], 55)
    df['ATR'] = calculate_atr(df, 14)
    df['Pivot_Resist'] = df['High'].rolling(20).max()
    df['Pivot_Support'] = df['Low'].rolling(20).min()
    
    # 2. Apex Trend (Master)
    base = calculate_hma(df['Close'], 55)
    atr_band = df['ATR'] * apex_mult
    df['Apex_Upper'] = base + atr_band
    df['Apex_Lower'] = base - atr_band
    trend = np.zeros(len(df)); trend[0] = 1
    for i in range(1, len(df)):
        c = df['Close'].iloc[i]
        if c > df['Apex_Upper'].iloc[i]: trend[i] = 1
        elif c < df['Apex_Lower'].iloc[i]: trend[i] = -1
        else: trend[i] = trend[i-1]
    df['Apex_Trend'] = trend
    
    # 3. Squeeze
    df['Sqz_Mom'] = (df['Close'] - df['Close'].rolling(20).mean()).rolling(20).mean() * 100
    
    # 4. Money Flow
    df['RSI'] = calculate_rsi(df['Close'])
    df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['MF_Matrix'] = ((df['RSI']-50)*2 * df['RVOL']).ewm(span=3).mean()
    
    # 5. Gann Activator
    gann_len = 3
    df['Gann_High'] = df['High'].rolling(gann_len).mean()
    df['Gann_Low'] = df['Low'].rolling(gann_len).mean()
    df['Gann_Trend'] = np.where(df['Close'] > df['Gann_High'].shift(1), 1, np.where(df['Close'] < df['Gann_Low'].shift(1), -1, 0))
    df['Gann_Trend'] = pd.Series(df['Gann_Trend']).replace(0, method='ffill').values

    # 6. Vector / SuperTrend
    st_val, st_dir = calculate_supertrend(df, 10, 4.0)
    df['DarkVector_Trend'] = st_dir
    
    # 7. Axiom Physics
    df = calc_chedo(df)
    df = calc_rqzo(df)
    df = calc_apex_flux(df)
    
    # 8. Composite Scores
    df['GM_Score'] = df['Apex_Trend'] + df['Gann_Trend'] + df['DarkVector_Trend'] + np.sign(df['Sqz_Mom'])
    
    # 9. Fear & Greed v4
    df['FG_Index'] = (df['RSI'] + (df['Close'].diff().rolling(5).mean() * 10)).clip(0, 100).rolling(5).mean()
    
    return df

# ==========================================
# 5. SMC & STRUCTURE
# ==========================================
def calculate_smc(df, swing=5):
    smc = {'structures': [], 'order_blocks': [], 'fvgs': []}
    
    # FVGs
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            smc['fvgs'].append({'x0':df.index[i-2], 'x1':df.index[i], 'y0':df['High'].iloc[i-2], 'y1':df['Low'].iloc[i], 'color':'rgba(0,255,104,0.3)'})
        if df['High'].iloc[i] < df['Low'].iloc[i-2]:
            smc['fvgs'].append({'x0':df.index[i-2], 'x1':df.index[i], 'y0':df['Low'].iloc[i-2], 'y1':df['High'].iloc[i], 'color':'rgba(255,0,8,0.3)'})
            
    # Structure
    df['PH'] = df['High'].rolling(swing*2+1, center=True).max() == df['High']
    df['PL'] = df['Low'].rolling(swing*2+1, center=True).min() == df['Low']
    last_h, last_l = None, None
    trend = 0
    
    for i in range(swing, len(df)):
        idx = df.index[i]; c = df['Close'].iloc[i]
        if df['PH'].iloc[i-swing]: last_h = {'p':df['High'].iloc[i-swing], 'x':df.index[i-swing]}
        if df['PL'].iloc[i-swing]: last_l = {'p':df['Low'].iloc[i-swing], 'x':df.index[i-swing]}
        
        if last_h and c > last_h['p']:
            lbl = "BOS" if trend == 1 else "CHoCH"
            trend = 1
            smc['structures'].append({'x0':last_h['x'], 'x1':idx, 'y':last_h['p'], 'color':'green', 'label':lbl})
            last_h = None
        if last_l and c < last_l['p']:
            lbl = "BOS" if trend == -1 else "CHoCH"
            trend = -1
            smc['structures'].append({'x0':last_l['x'], 'x1':idx, 'y':last_l['p'], 'color':'red', 'label':lbl})
            last_l = None
            
    return smc

# ==========================================
# 6. REPORTING & BROADCAST (Merged Kimis + Axiom + Titan)
# ==========================================
def generate_mobile_card(last, ticker, smart_stop, tp1, tp2, tp3, fg_index):
    is_bull = last['Apex_Trend'] == 1
    direction = "LONG 🐂" if is_bull else "SHORT 🐻"
    return f"""
    <div class="report-card">
        <div class="report-header">💠 SIGNAL: {direction}</div>
        <div class="report-item">Asset: <span class="highlight">{ticker}</span></div>
        <div class="report-item">Confidence: <span class="highlight">{"HIGH" if abs(last['GM_Score'])>=3 else "MED"}</span></div>
        <div class="report-item">Sentiment: <span class="highlight">{fg_index:.0f}/100</span></div>
    </div>
    <div class="report-card">
        <div class="report-header">🎯 EXECUTION</div>
        <div class="report-item">Entry: <span class="highlight">{last['Close']:.2f}</span></div>
        <div class="report-item">🛑 STOP: <span class="highlight">{smart_stop:.2f}</span></div>
        <div class="report-item">1️⃣ TP1: <span class="highlight">{tp1:.2f}</span></div>
        <div class="report-item">2️⃣ TP2: <span class="highlight">{tp2:.2f}</span></div>
        <div class="report-item">3️⃣ TP3: <span class="highlight">{tp3:.2f}</span></div>
    </div>
    """

def generate_axiom_report(last, ticker):
    trend_emoji = "🐂 BULLISH" if last['Apex_Trend'] == 1 else "🐻 BEARISH"
    flux_emoji = "🟢 Superconductor" if abs(last['Apex_Flux']) > 0.6 else "⚪ Neutral"
    entropy = "⚠️ CHAOS" if abs(last['CHEDO']) > 0.8 else "✅ STABLE"
    return f"""🚨 *AXIOM DETAILED REPORT* 🚨
💎 Asset: *{ticker}*
💰 Price: *${last['Close']:,.2f}*

🌊 *MARKET STRUCTURE*
• Trend: {trend_emoji}
• Baseline (HMA): ${last['HMA']:,.2f}

⚛️ *QUANTUM PHYSICS*
• Flux: {flux_emoji} ({last['Apex_Flux']:.2f})
• Entropy: {entropy} ({last['CHEDO']:.2f})
• Relativity: {last['RQZO']:.2f}

🛡️ *OUTLOOK*
{"High-Energy Event Likely" if abs(last['RQZO']) > 1 else "Standard Volatility"}
"""

def send_telegram(token, chat_id, text, excel_buffer=None, filename=None):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        # Simple splitting to avoid 4096 limit
        if len(text) > 4000:
            for i in range(0, len(text), 4000):
                requests.post(url, json={"chat_id": chat_id, "text": text[i:i+4000], "parse_mode": "Markdown"})
        else:
            requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
        
        # Send Document if provided
        if excel_buffer:
            url_doc = f"https://api.telegram.org/bot{token}/sendDocument"
            files = {'document': (filename, excel_buffer, 'application/vnd.ms-excel')}
            requests.post(url_doc, data={"chat_id": chat_id}, files=files)
            
        return True
    except: return False

# ==========================================
# 7. SCREENER ENGINE (From KIMI-Stocks)
# ==========================================
@st.cache_data(ttl=3600)
def run_screener(tickers):
    results = []
    for t in tickers:
        try:
            df = yf.download(t, period="3mo", interval="1d", progress=False)
            if df.empty or len(df) < 50: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            last = df.iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            rsi = calculate_rsi(df['Close']).iloc[-1]
            
            # Simple Criteria
            score = 0
            if last['Close'] > ma50: score += 1
            if 30 < rsi < 70: score += 1
            
            results.append({
                "Ticker": t,
                "Price": last['Close'],
                "RSI": rsi,
                "Above MA50": last['Close'] > ma50,
                "Score": score
            })
        except: continue
    if not results: return pd.DataFrame()
    return pd.DataFrame(results).sort_values("Score", ascending=False)

# ==========================================
# 8. AI ANALYST (Merged Personas)
# ==========================================
def ask_ai(df, ticker, persona, api_key):
    if not api_key: return "⚠️ Missing API Key"
    
    last = df.iloc[-1]
    stats = f"Price: {last['Close']:.2f} | Trend: {last['Apex_Trend']} | Flux: {last['Apex_Flux']:.2f} | GM Score: {last['GM_Score']}"
    
    if persona == "Axiom Physicist":
        sys_prompt = "You are Axiom, a quantitative physicist. Analyze using First Principles, Entropy, and Relativity."
    elif persona == "Junior Mining Expert":
        sys_prompt = "You are a specialized Mining Analyst. Focus on jurisdiction, drill results, and commodity cycles."
    else: # Titan Standard
        sys_prompt = "You are a Senior Market Analyst. Provide a concise technical breakdown and outlook."
        
    try:
        client = OpenAI(api_key=api_key)
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"Analyze {ticker}. Data: {stats}"}]
        )
        return res.choices[0].message.content
    except Exception as e: return f"Error: {e}"

def run_monte_carlo(df, days=30, simulations=1000):
    last_price = df['Close'].iloc[-1]
    returns = df['Close'].pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    sim_rets = np.random.normal(mu, sigma, (days, simulations))
    paths = np.zeros((days, simulations)); paths[0] = last_price
    for t in range(1, days): paths[t] = paths[t-1] * (1 + sim_rets[t])
    return paths

# ==========================================
# 9. MAIN DASHBOARD LAYOUT
# ==========================================
# Sidebar Config
st.sidebar.header("🎛️ Terminal Controls")

# API Keys
with st.sidebar.expander("🔐 Credentials", expanded=False):
    tg_token = st.text_input("TG Token", value=st.session_state.get('tg_token', ''), type="password")
    tg_chat = st.text_input("TG Chat ID", value=st.session_state.get('tg_chat', ''))
    
# Mode Selection
app_mode = st.sidebar.radio("Mode", ["Single Ticker", "Market Screener"])

# Ticker Selector
if app_mode == "Single Ticker":
    assets = ["BTC-USD", "ETH-USD", "SOL-USD", "SPY", "QQQ", "NVDA", "TSLA", "AAPL", "GC=F", "CL=F"]
    ticker = st.sidebar.selectbox("Ticker", assets)
    custom = st.sidebar.text_input("Or Type Ticker", "")
    if custom: ticker = custom.upper()
    interval = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=3)
    
    # Apex Config
    with st.sidebar.expander("🌊 Indicator Config"):
        apex_mult = st.number_input("Apex Mult", 1.0, 3.0, 1.5)

else:
    st.sidebar.info("Screener Mode Active")

st.sidebar.markdown("---")
balance = st.sidebar.number_input("Capital", value=10000)
risk = st.sidebar.slider("Risk %", 0.5, 5.0, 1.0)

# --- MACRO HEADER ---
m_gr, m_pr, m_ch = get_macro_data()
c1, c2, c3, c4 = st.columns(4)
c1.metric("S&P 500", f"{m_pr.get('SPY',0):.2f}", f"{m_ch.get('SPY',0):.2f}%")
c2.metric("Bitcoin", f"{m_pr.get('BTC-USD',0):.2f}", f"{m_ch.get('BTC-USD',0):.2f}%")
c3.metric("Gold", f"{m_pr.get('GC=F',0):.2f}", f"{m_ch.get('GC=F',0):.2f}%")
c4.metric("VIX", f"{m_pr.get('^VIX',0):.2f}", f"{m_ch.get('^VIX',0):.2f}%")
st.markdown("---")

# --- MAIN LOGIC ---
if app_mode == "Single Ticker":
    if st.button("Analyze"):
        with st.spinner("Crunching Physics & Technicals..."):
            df = safe_download(ticker, "1y", interval)
            if df is not None:
                # Run All Engines
                df = calc_indicators(df, apex_mult)
                fund = get_fundamentals(ticker)
                
                # Signal Logic
                last = df.iloc[-1]
                atr = last['ATR']
                stop = last['Close'] - (atr*2) if last['Apex_Trend']==1 else last['Close'] + (atr*2)
                risk_amt = abs(last['Close'] - stop)
                tp1 = last['Close'] + risk_amt if last['Apex_Trend']==1 else last['Close'] - risk_amt
                
                # Save to DB
                sig_data = {
                    'direction': 'LONG' if last['Apex_Trend']==1 else 'SHORT',
                    'entry': last['Close'], 'stop': stop,
                    'tp1': tp1, 'tp2': tp1 + risk_amt, 'tp3': tp1 + (risk_amt*2),
                    'conf': int(last['GM_Score'])
                }
                db.save_signal(ticker, sig_data)
                
                # TABS
                t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
                    "📊 God Mode", "⚛️ Axiom Physics", "🌍 Fund/Macro", 
                    "🧠 AI Analyst", "🔮 Quant", "🏦 SMC", "🔍 Volume", "📡 Broadcast"
                ])
                
                with t1: # God Mode
                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.15, 0.2])
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0,255,0,0.1)', line=dict(width=0), name='Apex Cloud'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='yellow'), name='HMA'), row=1, col=1)
                    
                    # Signals
                    buys = df[df['GM_Score']>=3]; sells = df[df['GM_Score']<=-3]
                    fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=10), name='Buy'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell'), row=1, col=1)
                    
                    # Subplots
                    fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], name='Squeeze'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['MF_Matrix'], fill='tozeroy', name='Money Flow'), row=3, col=1)
                    fig.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], name='Flux'), row=4, col=1)
                    
                    fig.update_layout(height=800, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                    
                with t2: # Axiom Physics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Entropy (CHEDO)", f"{last['CHEDO']:.2f}", delta="Chaos" if abs(last['CHEDO'])>0.8 else "Stable")
                    c2.metric("Relativity (RQZO)", f"{last['RQZO']:.2f}")
                    c3.metric("Flux Vector", f"{last['Apex_Flux']:.2f}")
                    
                    fig_p = make_subplots(rows=3, cols=1, shared_xaxes=True)
                    fig_p.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], name='Entropy', line=dict(color='cyan')), row=1, col=1)
                    fig_p.add_trace(go.Scatter(x=df.index, y=df['RQZO'], name='Relativity', line=dict(color='magenta')), row=2, col=1)
                    fig_p.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], name='Flux'), row=3, col=1)
                    fig_p.update_layout(height=600, template="plotly_dark")
                    st.plotly_chart(fig_p, use_container_width=True)
                    
                with t3: # Fund/Macro
                    if fund: st.json(fund)
                    else: st.warning("No Fundamentals Found")
                    
                with t4: # AI
                    persona = st.selectbox("Analyst Persona", ["Titan Standard", "Axiom Physicist", "Junior Mining Expert"])
                    if st.button("Run AI Analysis"):
                        rpt = ask_ai(df, ticker, persona, st.session_state.api_key)
                        st.markdown(rpt)
                        
                with t5: # Quant
                    paths = run_monte_carlo(df)
                    st.line_chart(paths[:, :20])
                    
                with t6: # SMC
                    smc = calculate_smc(df)
                    fig_s = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
                    for s in smc['structures']:
                        fig_s.add_shape(type="line", x0=s['x0'], y0=s['y'], x1=s['x1'], y1=s['y'], line=dict(color=s['color'], dash="dot"))
                    for f in smc['fvgs']:
                        fig_s.add_shape(type="rect", x0=f['x0'], x1=f['x1'], y0=f['y0'], y1=f['y1'], fillcolor=f['color'], line_width=0)
                    fig_s.update_layout(height=600, template="plotly_dark")
                    st.plotly_chart(fig_s, use_container_width=True)
                    
                with t8: # Broadcast
                    st.subheader("📡 Broadcast Center")
                    
                    # Mobile Card Preview
                    html_card = generate_mobile_card(last, ticker, stop, tp1, tp2, tp3, last['FG_Index'])
                    st.markdown(html_card, unsafe_allow_html=True)
                    
                    # Report Text
                    rpt_text = generate_axiom_report(last, ticker)
                    st.text_area("Report Preview", rpt_text, height=200)
                    
                    if st.button("Send to Telegram 🚀"):
                        if tg_token and tg_chat:
                            # Send Text
                            ok1 = send_telegram(tg_token, tg_chat, rpt_text)
                            # Send HTML Card as simplified text for mobile
                            card_text = f"MOBILE CARD\n{ticker} {sig_data['direction']}\nEntry: {sig_data['entry']:.2f}\nStop: {sig_data['stop']:.2f}\nTP3: {sig_data['tp3']:.2f}"
                            ok2 = send_telegram(tg_token, tg_chat, card_text)
                            if ok1 and ok2: st.success("Broadcast Sent!")
                            else: st.error("Failed.")
                        else: st.error("Missing Credentials")

                    # Signal Validation Checklist (Visual)
                    st.markdown("---")
                    st.subheader("✅ Signal Validation (Kimis Engine)")
                    
                    titan_sig = 1 if last['Apex_Trend'] == 1 else -1
                    apex_sig = last['Apex_Trend']
                    gann_sig = last['Gann_Trend']
                    
                    val_items = {
                        "Trend Confirmation": titan_sig == apex_sig,
                        "Volume Surge": last['RVOL'] > 1.2,
                        "Momentum Align": last['Sqz_Mom'] > 0 if titan_sig==1 else last['Sqz_Mom'] < 0,
                        "No Squeeze": last['Sqz_Mom'] != 0, 
                        "Sentiment Align": last['FG_Index'] > 50 if titan_sig==1 else last['FG_Index'] < 50
                    }
                    
                    for item, passed in val_items.items():
                        status = "✅ PASS" if passed else "❌ FAIL"
                        color = "#00e676" if passed else "#ff1744"
                        st.markdown(f'<div class="report-item" style="color:{color};">{item}: <strong>{status}</strong></div>', unsafe_allow_html=True)

            else: st.error("Data Download Failed")

elif app_mode == "Market Screener":
    st.subheader("🔍 Market Screener (Kimis Engine)")
    univ = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
    custom_univ = st.text_area("Custom Tickers (comma sep)", "")
    if custom_univ: univ = [x.strip() for x in custom_univ.split(",")]
    
    if st.button("Run Screen"):
        with st.spinner("Scanning..."):
            res_df = run_screener(univ)
            if not res_df.empty:
                st.dataframe(res_df)
                
                # Excel Export
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    res_df.to_excel(writer, index=False)
                
                st.download_button("Download Excel", buffer, "screener_results.xlsx")
                
                if st.button("Broadcast Top Pick to TG"):
                    if tg_token and tg_chat:
                        top = res_df.iloc[0]
                        msg = f"🔍 TOP PICK: {top['Ticker']} | Score: {top['Score']}"
                        send_telegram(tg_token, tg_chat, msg)
            else:
                st.warning("No matches found or data error.")
