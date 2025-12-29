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
import os

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM UI
# ==========================================
st.set_page_config(layout="wide", page_title="🏦 Titan Terminal v11", page_icon="👁️")

# --- CUSTOM CSS FOR DARKPOOL & GRADE CARDS ---
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
/* METRICS */
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
/* MOBILE CARDS & GRADING */
.report-card {
    background-color: #161b22;
    border-left: 4px solid #00ff00;
    padding: 15px;
    border-radius: 4px;
    margin-bottom: 10px;
    font-family: 'Roboto Mono', monospace;
}
.grade-card {
    background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%);
    border: 2px solid #00ff00;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 0 15px rgba(0, 255, 0, 0.2);
}
.grade-score { font-size: 3em; font-weight: bold; color: #00ff00; margin: 0; }
.grade-label { font-size: 1.2em; color: #fff; margin-bottom: 10px; }
.report-header { font-size: 1.1rem; font-weight: bold; color: #fff; margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 5px; }
.report-item { margin-bottom: 5px; font-size: 0.9rem; color: #aaa; }
.highlight { color: #00ff00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="title-glow">👁️ Titan Terminal v11</div>', unsafe_allow_html=True)
st.markdown("##### *Master Analysis Engine | Axiom Physics | Signal Grading | Linked Broadcast*")
st.markdown("---")

# ==========================================
# 2. CREDENTIALS MANAGEMENT
# ==========================================
def load_secret(key):
    try:
        if key in st.secrets: return st.secrets[key]
    except FileNotFoundError: pass
    if key in os.environ: return os.environ[key]
    return None

if 'api_key' not in st.session_state: st.session_state.api_key = load_secret("OPENAI_API_KEY")
if 'tg_token' not in st.session_state: st.session_state.tg_token = load_secret("TELEGRAM_TOKEN")
if 'tg_chat' not in st.session_state: st.session_state.tg_chat = load_secret("TELEGRAM_CHAT_ID")

# ==========================================
# 3. DATABASE LAYER
# ==========================================
class SignalDatabase:
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
                    grade TEXT,
                    outcome TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def save_signal(self, symbol: str, signal_data: dict, outcome: str = "PENDING"):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO signals (symbol, timestamp, direction, entry_price, stop_price, 
                                   tp1, tp2, tp3, confidence_score, grade, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                signal_data.get('grade', 'N/A'),
                outcome
            ))

db = SignalDatabase()

# ==========================================
# 4. DATA ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    if any(x in ticker for x in ["-", "=", "^"]): return None
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info: return None
        return {
            "Market Cap": info.get("marketCap", 0),
            "P/E Ratio": info.get("trailingPE", 0),
            "Rev Growth": info.get("revenueGrowth", 0),
            "Debt/Equity": info.get("debtToEquity", 0),
            "Summary": info.get("longBusinessSummary", "No Data Available"),
            "Sector": info.get("sector", "Unknown")
        }
    except: return None

def safe_download(ticker, period, interval):
    try:
        dl_interval = "1h" if interval == "4h" else interval
        df = yf.download(ticker, period=period, interval=dl_interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty: return None
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
            else: return None
        return df
    except: return None

@st.cache_data(ttl=300)
def get_macro_data():
    groups = {
        "🇺🇸 Indices": {"S&P 500": "SPY", "Nasdaq": "QQQ"},
        "⚠️ Risk": {"Bitcoin": "BTC-USD", "VIX": "^VIX"},
        "🥇 Commodities": {"Gold": "GC=F", "Oil": "CL=F"}
    }
    all_tickers = [sym for g in groups.values() for sym in g.values()]
    try:
        data = yf.download(all_tickers, period="5d", interval="1d", group_by='ticker', progress=False)
        prices, changes = {}, {}
        for sym in all_tickers:
            try:
                df = data[sym] if len(all_tickers) > 1 else data
                if len(df) >= 2:
                    curr = df['Close'].iloc[-1]
                    prev = df['Close'].iloc[-2]
                    prices[sym] = curr
                    changes[sym] = ((curr - prev) / prev) * 100
            except: continue
        return groups, prices, changes
    except: return groups, {}, {}

# ==========================================
# 5. MATH & PHYSICS ENGINE
# ==========================================
def calculate_wma(series, length):
    w = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)

def calculate_hma(series, length):
    half = int(length / 2)
    sqrt = int(np.sqrt(length))
    wma_half = calculate_wma(series, half)
    wma_full = calculate_wma(series, length)
    return calculate_wma(2 * wma_half - wma_full, sqrt)

def calculate_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def calculate_rsi(series, length=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/length, adjust=False).mean()
    loss = -delta.where(delta < 0, 0.0).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
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
    lambda_n = tanh((pd.Series(np.log(np.abs(np.diff(log_ret, prepend=0)) + 1e-9)).rolling(length).mean().values + 5) / 7)
    ent_n = tanh(pd.Series(log_ret**2).rolling(length).sum().values * 10)
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
    eff_sm = pd.Series(eff_raw, index=df.index).ewm(span=length).mean()
    vol_avg = df['Volume'].rolling(55).mean()
    v_rat = np.where(vol_avg == 0, 1, df['Volume'] / vol_avg)
    raw = np.sign(df['Close'] - df['Open']) * eff_sm * v_rat
    df['Apex_Flux'] = raw.ewm(span=5).mean()
    return df

def run_monte_carlo(df, days=30, simulations=200):
    last_price = df['Close'].iloc[-1]
    returns = df['Close'].pct_change().dropna()
    sim_rets = np.random.normal(returns.mean(), returns.std(), (days, simulations))
    paths = np.zeros((days, simulations)); paths[0] = last_price
    for t in range(1, days): paths[t] = paths[t-1] * (1 + sim_rets[t])
    return paths

# ==========================================
# 6. SIGNAL GRADING ENGINE (KIMIS RESTORED)
# ==========================================
def calculate_grade(row):
    """
    Restores the specific Kimis Signals grading logic.
    Inputs: Apex Trend, Gann, Momentum (Squeeze), Volume (RVOL), Sentiment.
    """
    titan_sig = 1 if row['Apex_Trend'] == 1 else -1
    
    # 1. Validation Checks
    checks = {
        "Trend": titan_sig == row['Gann_Trend'],
        "Volume": row['RVOL'] > 1.2,
        "Momentum": (row['Sqz_Mom'] > 0 if titan_sig==1 else row['Sqz_Mom'] < 0),
        "Flux": (row['Apex_Flux'] > 0 if titan_sig==1 else row['Apex_Flux'] < 0),
        "No Squeeze": row['Sqz_Mom'] != 0
    }
    
    score = sum(checks.values())
    total = len(checks)
    
    if score == 5: grade = "A+ (PERFECT)"
    elif score == 4: grade = "A (EXCELLENT)"
    elif score == 3: grade = "B (GOOD)"
    elif score == 2: grade = "C (WEAK)"
    else: grade = "D (AVOID)"
    
    return grade, score, checks

def calc_indicators(df, apex_mult=1.5):
    df = df.copy()
    df['HMA'] = calculate_hma(df['Close'], 55)
    df['ATR'] = calculate_atr(df, 14)
    
    # Apex Trend
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
    
    # Squeeze
    df['Sqz_Mom'] = (df['Close'] - df['Close'].rolling(20).mean()).rolling(20).mean() * 100
    
    # Volume & Money Flow
    df['RSI'] = calculate_rsi(df['Close'])
    df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['MF_Matrix'] = ((df['RSI']-50)*2 * df['RVOL']).ewm(span=3).mean()
    
    # Gann
    gann_len = 3
    df['Gann_High'] = df['High'].rolling(gann_len).mean()
    df['Gann_Low'] = df['Low'].rolling(gann_len).mean()
    df['Gann_Trend'] = np.where(df['Close'] > df['Gann_High'].shift(1), 1, np.where(df['Close'] < df['Gann_Low'].shift(1), -1, 0))
    df['Gann_Trend'] = pd.Series(df['Gann_Trend']).replace(0, method='ffill').values

    # Vector
    st_val, st_dir = calculate_supertrend(df, 10, 4.0)
    df['DarkVector_Trend'] = st_dir
    
    # Physics
    df = calc_chedo(df)
    df = calc_rqzo(df)
    df = calc_apex_flux(df)
    
    df['GM_Score'] = df['Apex_Trend'] + df['Gann_Trend'] + df['DarkVector_Trend'] + np.sign(df['Sqz_Mom'])
    df['FG_Index'] = (df['RSI'] + (df['Close'].diff().rolling(5).mean() * 10)).clip(0, 100).rolling(5).mean()
    
    return df

# ==========================================
# 7. BROADCASTING SUITE (LINKED)
# ==========================================
def generate_html_card(row, ticker, stop, tp1, tp2, tp3, grade):
    direction = "LONG 🐂" if row['Apex_Trend'] == 1 else "SHORT 🐻"
    color = "#00e676" if "A" in grade or "B" in grade else "#ff1744"
    return f"""
    <div class="grade-card" style="border-color: {color};">
        <div class="grade-label">SIGNAL GRADE</div>
        <div class="grade-score" style="color: {color};">{grade.split()[0]}</div>
        <div>{grade.split()[1] if len(grade.split())>1 else ""}</div>
    </div>
    <div class="report-card">
        <div class="report-header">💠 {direction} | {ticker}</div>
        <div class="report-item">Entry: <span class="highlight">{row['Close']:.4f}</span></div>
        <div class="report-item">🛑 STOP: <span class="highlight">{stop:.4f}</span></div>
        <div class="report-item">🎯 TP3: <span class="highlight">{tp3:.4f}</span></div>
        <div class="report-item">Flux: {row['Apex_Flux']:.2f} | RVOL: {row['RVOL']:.1f}</div>
    </div>
    """

def generate_quick_signal_text(row, ticker, stop, tp3, grade):
    return f"""⚡ *TITAN FLASH: {ticker}*
{("🐂 LONG" if row['Apex_Trend']==1 else "🐻 SHORT")}
Grade: {grade}
Entry: {row['Close']:.4f}
Stop: {stop:.4f}
Target: {tp3:.4f}
RVOL: {row['RVOL']:.1f}x"""

def send_linked_broadcast(token, chat, row, ticker, stop, tp3, grade, ai_analysis, full_report):
    """
    Sends a chain of messages:
    1. Quick Flash (Instant notification)
    2. AI Deep Dive (Context)
    3. Full Physics Report (Detail)
    """
    if not token or not chat: return False
    
    base_url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    # 1. Quick Flash
    quick_text = generate_quick_signal_text(row, ticker, stop, tp3, grade)
    requests.post(base_url, json={"chat_id": chat, "text": quick_text, "parse_mode": "Markdown"})
    
    # 2. AI Context
    if ai_analysis:
        ai_text = f"🧠 *AI ANALYST INSIGHT*\n{ai_analysis}"
        requests.post(base_url, json={"chat_id": chat, "text": ai_text, "parse_mode": "Markdown"})
        
    # 3. Full Report
    if full_report:
        # Split logic for long reports
        if len(full_report) > 4000:
            for i in range(0, len(full_report), 4000):
                requests.post(base_url, json={"chat_id": chat, "text": full_report[i:i+4000], "parse_mode": "Markdown"})
        else:
            requests.post(base_url, json={"chat_id": chat, "text": full_report, "parse_mode": "Markdown"})
            
    return True

# ==========================================
# 8. MAIN DASHBOARD LAYOUT
# ==========================================
# Sidebar
st.sidebar.header("🎛️ Terminal Controls")
with st.sidebar.expander("🔐 Keys", expanded=False):
    tg_token_input = st.text_input("TG Token", value=st.session_state.get('tg_token', '') or "", type="password")
    tg_chat_input = st.text_input("TG Chat ID", value=st.session_state.get('tg_chat', '') or "")
    if tg_token_input: st.session_state.tg_token = tg_token_input
    if tg_chat_input: st.session_state.tg_chat = tg_chat_input
    ai_key_input = st.text_input("OpenAI Key", value=st.session_state.get('api_key', '') or "", type="password")
    if ai_key_input: st.session_state.api_key = ai_key_input

app_mode = st.sidebar.radio("Mode", ["Terminal (Single)", "Scanner (Multi)"])

if app_mode == "Terminal (Single)":
    assets = ["BTC-USD", "ETH-USD", "SOL-USD", "SPY", "QQQ", "NVDA", "TSLA", "GC=F", "CL=F"]
    ticker = st.sidebar.selectbox("Ticker", assets)
    custom = st.sidebar.text_input("Or Type Ticker", "")
    if custom: ticker = custom.upper()
    interval = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=3)
    apex_mult = st.sidebar.number_input("Apex Mult", 1.0, 3.0, 1.5)

    if st.button("Analyze"):
        with st.spinner(f"Running Titan v11 Engine on {ticker}..."):
            df = safe_download(ticker, "1y", interval)
            if df is not None:
                df = calc_indicators(df, apex_mult)
                last = df.iloc[-1]
                
                # Logic
                atr = last['ATR']
                is_bull = last['Apex_Trend'] == 1
                stop = last['Close'] - (atr*2) if is_bull else last['Close'] + (atr*2)
                risk = abs(last['Close'] - stop)
                tp1, tp2, tp3 = (last['Close'] + risk*x if is_bull else last['Close'] - risk*x for x in [1, 2, 3])
                
                # Grading
                grade, score, checks = calculate_grade(last)
                
                # Save DB
                sig_data = {
                    'direction': 'LONG' if is_bull else 'SHORT',
                    'entry': last['Close'], 'stop': stop,
                    'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
                    'conf': int(last['GM_Score']), 'grade': grade
                }
                db.save_signal(ticker, sig_data)
                
                # TABS
                t1, t2, t3, t4, t5 = st.tabs(["📊 God Mode", "⚛️ Physics", "🧠 AI Analyst", "🔮 Quant", "📡 Broadcast Suite"])
                
                with t1: # Technicals
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2])
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0,255,0,0.1)', line=dict(width=0), name='Apex Cloud'), row=1, col=1)
                    fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], name='Squeeze'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['MF_Matrix'], fill='tozeroy', name='Money Flow'), row=3, col=1)
                    fig.update_layout(height=700, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                    
                with t2: # Physics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Entropy (CHEDO)", f"{last['CHEDO']:.2f}", delta="Chaos" if abs(last['CHEDO'])>0.8 else "Stable")
                    c2.metric("Relativity (RQZO)", f"{last['RQZO']:.2f}")
                    c3.metric("Flux Vector", f"{last['Apex_Flux']:.2f}")
                    fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True)
                    fig_p.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], name='Entropy', line=dict(color='cyan')), row=1, col=1)
                    fig_p.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], name='Flux'), row=2, col=1)
                    fig_p.update_layout(height=500, template="plotly_dark")
                    st.plotly_chart(fig_p, use_container_width=True)
                    
                with t3: # AI
                    persona = st.selectbox("Persona", ["Titan Standard", "Axiom Physicist"])
                    if st.button("Generate AI Insight"):
                        analysis = ask_ai(df, ticker, persona, st.session_state.api_key)
                        st.session_state['last_ai_analysis'] = analysis # Store for broadcast
                        st.markdown(analysis)
                    elif 'last_ai_analysis' in st.session_state:
                        st.markdown(st.session_state['last_ai_analysis'])
                        
                with t4: # Quant
                    paths = run_monte_carlo(df)
                    st.line_chart(paths[:, :20])
                    
                with t5: # Broadcast Suite (The 100% Fix)
                    st.subheader("📡 Unified Broadcast Command")
                    
                    # 1. Visual Card
                    html_card = generate_html_card(last, ticker, stop, tp1, tp2, tp3, grade)
                    st.markdown(html_card, unsafe_allow_html=True)
                    
                    # 2. Text Reports
                    c_b1, c_b2 = st.columns(2)
                    with c_b1:
                        st.markdown("#### ✅ Grading Checklist")
                        for k, v in checks.items():
                            st.caption(f"{k}: {'✅' if v else '❌'}")
                    
                    # 3. Unified Button
                    if st.button("🚀 TRANSMIT LINKED BROADCAST"):
                        if st.session_state.tg_token and st.session_state.tg_chat:
                            # Gather Data
                            ai_txt = st.session_state.get('last_ai_analysis', "No AI analysis generated.")
                            full_rpt = f"""🚨 *DETAILED TECHNICAL REPORT*
Asset: {ticker} | Price: {last['Close']:.2f}
Trend: {'Bull' if is_bull else 'Bear'} | Flux: {last['Apex_Flux']:.2f}
Entropy: {last['CHEDO']:.2f} | Squeeze: {last['Sqz_Mom']:.1f}
CHECKLIST: {score}/5 Checks Passed ({grade})
"""
                            # Send Chain
                            sent = send_linked_broadcast(
                                st.session_state.tg_token, 
                                st.session_state.tg_chat, 
                                last, ticker, stop, tp3, grade, 
                                ai_txt, full_rpt
                            )
                            if sent: st.success("✅ Full Linked Broadcast Sent!")
                            else: st.error("Failed to send.")
                        else: st.error("Missing Credentials")

            else: st.error("Download Failed.")

elif app_mode == "Scanner (Multi)":
    st.subheader("🔍 Market Screener")
    univ = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "AAPL", "MSFT", "TSLA", "NVDA", "AMD", "COIN"]
    custom = st.text_area("Custom Tickers", "")
    if custom: univ = [x.strip() for x in custom.split(",")]
    
    if st.button("Scan Market"):
        results = []
        progress = st.progress(0)
        for i, t in enumerate(univ):
            df = safe_download(t, "3mo", "1d")
            if df is not None:
                df = calc_indicators(df)
                last = df.iloc[-1]
                grade, score, _ = calculate_grade(last)
                results.append({
                    "Ticker": t, "Price": last['Close'], 
                    "Trend": "Bull" if last['Apex_Trend']==1 else "Bear",
                    "Grade": grade, "Score": score, "RSI": last['RSI']
                })
            time.sleep(0.1) # Rate limit protection
            progress.progress((i+1)/len(univ))
            
        if results:
            res_df = pd.DataFrame(results).sort_values("Score", ascending=False)
            st.dataframe(res_df)
            
            # Export
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
                
            st.download_button("Download Excel", buffer, "scan_results.xlsx")
