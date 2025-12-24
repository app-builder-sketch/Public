"""
TITAN x APEX: ARCHITECT EDITION (v20.0)
Unified Quantum Execution Model | DPC Mobile Architecture
"""
import time
import math
import requests
import io
from typing import Dict, Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import yfinance as yf
import openai
import xlsxwriter
from datetime import datetime, timezone

# =============================================================================
# 1. SYSTEM CONFIGURATION & DPC ARCHITECTURE
# =============================================================================
st.set_page_config(
    page_title="TITAN x APEX ARCHITECT",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="collapsed"
)

# DPC CSS Architecture: Dark Mode, Neon Glows, Mobile Optimization
st.markdown("""
<style>
    /* CORE THEME */
    .stApp { background-color: #0e1117; }
    
    /* TYPOGRAPHY */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p { 
        font-family: 'Roboto Mono', monospace !important; 
        color: #e0e0e0;
    }

    /* METRIC CONTAINERS (Cards) */
    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #58a6ff;
    }
    div[data-testid="metric-container"] label {
        font-size: 12px !important; 
        color: #8b949e !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 22px !important;
        color: #58a6ff !important;
        text-shadow: 0 0 10px rgba(88, 166, 255, 0.3);
    }

    /* BUTTONS */
    .stButton > button {
        background: #238636;
        color: #ffffff;
        border: none;
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
        border-radius: 6px;
        height: 3.5em; /* Touch friendly */
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #2ea043;
        box-shadow: 0 0 15px rgba(46, 160, 67, 0.6);
    }
    
    /* REPORT CARD STYLING */
    .report-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .report-card.bull { border-left-color: #3fb950; }
    .report-card.bear { border-left-color: #f85149; }
    
    .report-header {
        font-size: 16px;
        font-weight: 700;
        color: #e0e0e0;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #30363d;
        padding-bottom: 5px;
    }
    .report-item {
        font-size: 13px;
        color: #8b949e;
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
    }
    .value-glow { color: #e0e0e0; font-weight: 600; }
    .value-bull { color: #3fb950; font-weight: 700; }
    .value-bear { color: #f85149; font-weight: 700; }
    .value-warn { color: #d29922; font-weight: 700; }
    
    /* EXPANDER STYLING */
    .streamlit-expanderHeader {
        background-color: #161b22;
        color: #e0e0e0;
        border: 1px solid #30363d;
        border-radius: 4px;
    }
    
    /* TAB STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: #161b22;
        border-radius: 4px;
        color: #8b949e;
        font-size: 12px;
        flex: 1; /* Equal width */
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #238636;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. DATA LAYER (Hybrid: Binance + YFinance)
# =============================================================================
BINANCE_API_BASE = "https://api.binance.us/api/v3"
HEADERS = { "User-Agent": "TitanApex/20.0", "Accept": "application/json" }

# --- Titan Data Fetcher (Intraday/Fast) ---
@st.cache_data(ttl=15, show_spinner=False)
def get_klines_binance(symbol_bin: str, interval: str, limit: int) -> pd.DataFrame:
    try:
        params = {"symbol": symbol_bin, "interval": interval, "limit": limit}
        r = requests.get(f"{BINANCE_API_BASE}/klines", params=params, headers=HEADERS, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if not data: return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=['t','o','h','l','c','v','T','q','n','V','Q','B'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            cols = ['open','high','low','close','volume']
            df[cols] = df[['o','h','l','c','v']].astype(float)
            return df[['timestamp'] + cols]
    except Exception as e:
        st.error(f"Binance API Error: {str(e)}")
    return pd.DataFrame()

# --- Apex Data Fetcher (Scanner/Historical) ---
@st.cache_data(ttl=3600)
def get_financials_yf(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "ticker": ticker,
            "name": info.get('shortName', ticker),
            "sector": "Crypto",
            "market_cap": info.get('marketCap', 0)
        }
    except: return None

def get_history_yf(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        # 60 days required for HMA 55 + lags
        return stock.history(period="3mo", interval="1d") 
    except: return None

def send_telegram_msg(token, chat, msg):
    if not token or not chat: return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat, "text": msg, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=5)
        return r.status_code == 200
    except: return False

def send_telegram_doc(token, chat_id, text, file_buf, fname):
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat_id, "text": text})
        file_buf.seek(0)
        requests.post(f"https://api.telegram.org/bot{token}/sendDocument",
                      data={"chat_id": chat_id},
                      files={"document": (fname, file_buf, "application/vnd.ms-excel")})
        return True
    except: return False

# =============================================================================
# 3. TITAN ENGINE (Vectorized Intraday Logic)
# =============================================================================
class TitanEngine:
    @staticmethod
    def calculate_hma(series, length):
        half_len = int(length / 2)
        sqrt_len = int(math.sqrt(length))
        wma_f = series.rolling(length).mean()
        wma_h = series.rolling(half_len).mean()
        diff = 2 * wma_h - wma_f
        return diff.rolling(sqrt_len).mean()

    @staticmethod
    def calculate_fibonacci(df, lookback=50):
        recent = df.iloc[-lookback:]
        h, l = recent['high'].max(), recent['low'].min()
        d = h - l
        return {
            'fib_382': h - (d * 0.382), 'fib_500': h - (d * 0.500),
            'fib_618': h - (d * 0.618), 'high': h, 'low': l
        }

    @staticmethod
    def calculate_fear_greed_index(df):
        try:
            log_ret = np.log(df['close'] / df['close'].shift(1))
            vol_roll = log_ret.rolling(30).std(); vol_base = log_ret.rolling(90).std()
            if len(vol_roll) < 90: return 50
            vol_score = max(0, min(100, 50 - ((vol_roll.iloc[-1] - vol_base.iloc[-1]) / vol_base.iloc[-1]) * 100))
            
            rsi = df['rsi'].iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            trend_score = 50 + ((df['close'].iloc[-1] - sma_50) / sma_50 * 1000)
            return int((vol_score * 0.3) + (rsi * 0.4) + (max(0, min(100, trend_score)) * 0.3))
        except: return 50

    @staticmethod
    def run_engine(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        if df.empty: return df
        df = df.copy().reset_index(drop=True)
        amp, dev, hma_l = params['amp'], params['dev'], params['hma_len']
        tp1, tp2, tp3 = params['tp1'], params['tp2'], params['tp3']
        gann_l, vol_l, mf_l = params['gann_len'], 20, 14

        # Indicators
        prev_close = df['close'].shift(1)
        tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - prev_close), abs(df['low'] - prev_close)))
        df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
        df['hma'] = TitanEngine.calculate_hma(df['close'], hma_l)
        
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
        
        bb_basis = df['close'].rolling(20).mean(); bb_dev = df['close'].rolling(20).std() * 2.0
        kc_dev = df['atr'] * 1.5
        df['in_squeeze'] = ((bb_basis - bb_dev) > (bb_basis - kc_dev)) & ((bb_basis + bb_dev) < (bb_basis + kc_dev))

        delta = df['close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14).mean(); loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain/loss)))
        df['rvol'] = df['volume'] / df['volume'].rolling(vol_l).mean()
        df['money_flow'] = ((df['rsi'] - 50) * (df['volume'] / df['volume'].rolling(mf_l).mean())).ewm(span=3).mean()

        # Vectorized Titan Trend
        n = len(df)
        close_arr = df['close'].values; atr_arr = df['atr'].values
        ll_arr = df['low'].rolling(amp).min().values; hh_arr = df['high'].rolling(amp).max().values
        trend = np.zeros(n, dtype=int); stop = np.full(n, np.nan)
        curr_t = 0; curr_s = np.nan
        
        for i in range(amp, n):
            c = close_arr[i]; d = atr_arr[i] * dev
            if curr_t == 0:
                s = ll_arr[i] + d
                curr_s = max(curr_s, s) if not np.isnan(curr_s) else s
                if c < curr_s: curr_t = 1; curr_s = hh_arr[i] - d
            else:
                s = hh_arr[i] - d
                curr_s = min(curr_s, s) if not np.isnan(curr_s) else s
                if c > curr_s: curr_t = 0; curr_s = ll_arr[i] + d
            trend[i] = curr_t; stop[i] = curr_s

        df['is_bull'] = trend == 0
        df['entry_stop'] = stop
        
        # Signals & Targets
        df['buy'] = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False)) & (df['rvol'] > 1.0)
        df['sell'] = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True)) & (df['rvol'] > 1.0)
        
        df['entry'] = df['close'].where(df['buy'] | df['sell']).ffill()
        df['stop_val'] = df['entry_stop'].where(df['buy'] | df['sell']).ffill()
        risk = abs(df['entry'] - df['stop_val'])
        bull_mask = df['is_bull']
        df['tp1'] = np.where(bull_mask, df['entry'] + (risk * tp1), df['entry'] - (risk * tp1))
        df['tp2'] = np.where(bull_mask, df['entry'] + (risk * tp2), df['entry'] - (risk * tp2))
        df['tp3'] = np.where(bull_mask, df['entry'] + (risk * tp3), df['entry'] - (risk * tp3))

        # Apex Logic (Titan Version)
        apex_base = TitanEngine.calculate_hma(df['close'], 55); apex_atr = df['atr'] * 1.5
        cond_up = df['close'] > (apex_base + apex_atr); cond_down = df['close'] < (apex_base - apex_atr)
        df['apex_trend'] = pd.Series(np.select([cond_up, cond_down], [1, -1], default=0)).replace(0, np.nan).ffill().fillna(0).values

        # Gann Logic
        sma_h = df['high'].rolling(gann_l).mean().values; sma_l = df['low'].rolling(gann_l).mean().values
        g_trend = np.zeros(n, dtype=int); g_act = np.full(n, np.nan)
        curr_g_t = 1; curr_g_a = sma_l[gann_l] if n > gann_l else np.nan
        
        for i in range(gann_l, n):
            c = close_arr[i]
            if curr_g_t == 1:
                if c < curr_g_a: curr_g_t = -1; curr_g_a = sma_h[i]
                else: curr_g_a = sma_l[i]
            else:
                if c > curr_g_a: curr_g_t = 1; curr_g_a = sma_l[i]
                else: curr_g_a = sma_h[i]
            g_trend[i] = curr_g_t; g_act[i] = curr_g_a
        df['gann_trend'] = g_trend; df['gann_act'] = g_act

        return df

    @staticmethod
    def run_backtest(df, tp1_r):
        if df.empty: return 0, 0, 0
        signals = df[df['buy'] | df['sell']]
        if signals.empty: return 0, 0, 0
        
        trades = []
        for idx, row in signals.iterrows():
            if idx + 20 >= len(df): continue
            future = df.iloc[idx+1 : idx+21]
            outcome = "PENDING"; pnl = 0
            
            if row['is_bull']:
                if future['high'].max() >= row['tp1']: outcome = "WIN"; pnl = abs(row['close'] - row['entry_stop']) * tp1_r
                elif future['low'].min() <= row['entry_stop']: outcome = "LOSS"; pnl = -abs(row['close'] - row['entry_stop'])
            else:
                if future['low'].min() <= row['tp1']: outcome = "WIN"; pnl = abs(row['close'] - row['entry_stop']) * tp1_r
                elif future['high'].max() >= row['entry_stop']: outcome = "LOSS"; pnl = -abs(row['close'] - row['entry_stop'])
            
            if outcome != "PENDING": trades.append({'outcome': outcome, 'pnl': pnl})
            
        if not trades: return 0, 0, 0
        df_res = pd.DataFrame(trades)
        wins = len(df_res[df_res['outcome']=='WIN'])
        return len(df_res), (wins / len(df_res)) * 100, (wins * tp1_r) - len(df_res[df_res['outcome']=='LOSS'])

# =============================================================================
# 4. APEX ENGINE (SMC + Scanner Logic)
# =============================================================================
class ApexEngine:
    @staticmethod
    def calculate_atr(df, length=14):
        h_l = df['High'] - df['Low']
        h_c = (df['High'] - df['Close'].shift()).abs()
        l_c = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
        return tr.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def calculate_adx(df, length=14):
        up = df['High'].diff(); down = -df['Low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = ApexEngine.calculate_atr(df, length)
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/length, adjust=False).mean() / tr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/length, adjust=False).mean() / tr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        return dx.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def calculate_wavetrend(df):
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=10, adjust=False).mean()
        d = (ap - esa).abs().ewm(span=10, adjust=False).mean().replace(0, 0.0001)
        ci = (ap - esa) / (0.015 * d)
        return ci.ewm(span=21, adjust=False).mean()

    @staticmethod
    def detect_smc(df):
        lookback = 10
        recent_high = df['High'].shift(1).rolling(20).max()
        bos_bull = (df['Close'] > recent_high) & (df['Close'].shift(1) <= recent_high.shift(1))
        fvg_bull = (df['Low'] > df['High'].shift(2))
        fvg_size = (df['Low'] - df['High'].shift(2))
        return bos_bull, fvg_bull, fvg_size

    @staticmethod
    def run_full_analysis(df):
        if len(df) < 60: return None
        # Rename cols for YF compatibility
        df = df.copy() # YF uses Capitalized Cols
        
        # Apex Trend (HMA 55)
        def wma(s, l): return s.rolling(l).apply(lambda x: np.dot(x, np.arange(1, l+1))/np.arange(1, l+1).sum(), raw=True)
        def hma(s, l):
             w_h = wma(s, int(l/2)); w_f = wma(s, l)
             return wma(2*w_h - w_f, int(np.sqrt(l)))
             
        baseline = hma(df['Close'], 55)
        atr = ApexEngine.calculate_atr(df, 55)
        upper = baseline + (atr * 1.5); lower = baseline - (atr * 1.5)
        
        df['Apex_Trend'] = np.select([df['Close'] > upper, df['Close'] < lower], [1, -1], default=0)
        
        # Signals
        df['ADX'] = ApexEngine.calculate_adx(df)
        df['WaveTrend'] = ApexEngine.calculate_wavetrend(df)
        
        buy_signal = ((df['Apex_Trend'] == 1) & (df['WaveTrend'] < 60) & (df['WaveTrend'] > df['WaveTrend'].shift(1)) & (df['ADX'] > 20))
        bos_bull, fvg_bull, fvg_size = ApexEngine.detect_smc(df)
        
        last = df.iloc[-1]
        return {
            "Price": last['Close'],
            "Trend": "Bullish 🟢" if last['Apex_Trend'] == 1 else "Bearish 🔴",
            "WaveTrend": last['WaveTrend'], "ADX": last['ADX'],
            "Apex_Buy_Signal": buy_signal.tail(3).any(),
            "BOS_Alert": bos_bull.tail(3).any(),
            "FVG_Detected": fvg_bull.iloc[-1], "FVG_Size": fvg_size.iloc[-1] if fvg_bull.iloc[-1] else 0
        }

def analyze_smc_with_ai(row, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        prompt = f"""
        Act as a Smart Money Concepts (SMC) Crypto Trader. Analyze setup:
        ASSET: {row['ticker']} (${row['Price']:.4f})
        TREND: {row['Trend']}
        SIGNALS: Apex Buy: {row['Apex_Buy_Signal']}, BOS: {row['BOS_Alert']}, FVG: {row['FVG_Detected']}
        MOMENTUM: WaveTrend: {row['WaveTrend']:.1f}, ADX: {row['ADX']:.1f}
        
        Provide: 1. VERDICT (Strong Long/Scalp/Wait/Short). 2. STRUCTURE (BOS/FVG Context). 3. EXECUTION Plan.
        Keep it under 100 words.
        """
        response = client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

# =============================================================================
# 5. UI COMPONENTS
# =============================================================================
def render_titan_report(row, fibs, fg, smart_stop):
    is_bull = row['is_bull']
    trend_color = "bull" if is_bull else "bear"
    direction = "LONG 🐂" if is_bull else "SHORT 🐻"
    
    score = (1 if is_bull else -1) + row['apex_trend'] + row['gann_trend'] # Rough alignment
    conf = "MAX 🔥" if abs(score) >= 2 else ("HIGH" if abs(score) == 1 else "LOW")
    conf_c = "value-bull" if is_bull else "value-bear"

    return f"""
    <div class="report-card {trend_color}">
        <div class="report-header"><span>💠 SIGNAL</span><span class="{conf_c}">{direction}</span></div>
        <div class="report-item">Confidence <span class="{conf_c}">{conf}</span></div>
        <div class="report-item">Sentiment <span class="value-glow">{fg}/100</span></div>
        <div class="report-item">Squeeze <span class="{'value-warn' if row['in_squeeze'] else 'value-glow'}">{'⚠️ ACTIVE' if row['in_squeeze'] else '⚪ NONE'}</span></div>
    </div>
    <div class="report-card">
        <div class="report-header">🌊 FLOW METRICS</div>
        <div class="report-item">RVOL <span class="value-glow">{row['rvol']:.2f} ({'🚀' if row['rvol'] > 2 else 'Norm'})</span></div>
        <div class="report-item">Money Flow <span class="{'value-bull' if row['money_flow']>0 else 'value-bear'}">{row['money_flow']:.2f}</span></div>
        <div class="report-item">VWAP Bias <span class="value-glow">{'Above' if row['close'] > row['vwap'] else 'Below'}</span></div>
    </div>
    <div class="report-card">
        <div class="report-header">🎯 TACTICAL PLAN</div>
        <div class="report-item">ENTRY <span class="value-glow">{row['close']:.4f}</span></div>
        <div class="report-item">STOP (Smart) <span class="value-warn">{smart_stop:.4f}</span></div>
        <div class="report-item">TP1 (1.5R) <span class="value-bull">{row['tp1']:.4f}</span></div>
        <div class="report-item">TP2 (3.0R) <span class="value-bull">{row['tp2']:.4f}</span></div>
        <div class="report-item">TP3 (5.0R) <span class="value-bull">{row['tp3']:.4f}</span></div>
    </div>
    """

def render_chart(df):
    fig = go.Figure()
    fig.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                        increasing_line_color='#3fb950', decreasing_line_color='#f85149', name='Price')
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', name='HMA', line=dict(color='#58a6ff', width=1.5)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vwap'], mode='lines', name='VWAP', line=dict(color='#a371f7', width=1.5, dash='dot')))
    
    buys = df[df['buy']]; sells = df[df['sell']]
    if not buys.empty: fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low']*0.999, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#3fb950'), name='BUY'))
    if not sells.empty: fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high']*1.001, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#f85149'), name='SELL'))
    
    fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), template='plotly_dark', xaxis_visible=False, yaxis=dict(side='right', gridcolor='#30363d'), showlegend=False)
    return fig

# =============================================================================
# 6. MAIN APP LOGIC
# =============================================================================
def main():
    # --- Sidebar Config ---
    with st.sidebar:
        st.header("💠 MODE SELECTOR")
        app_mode = st.radio("System Mode", ["TITAN TERMINAL", "APEX SCANNER"])
        st.divider()
        
        # Shared Config
        st.subheader("🤖 INTEGRATIONS")
        if "OPENAI_API_KEY" in st.secrets: api_key = st.secrets["OPENAI_API_KEY"]
        else: api_key = st.text_input("OpenAI Key", type="password")
        
        tg_token = st.text_input("TG Token", value=st.secrets.get("TELEGRAM_TOKEN", ""), type="password")
        tg_chat = st.text_input("TG Chat ID", value=st.secrets.get("TELEGRAM_CHAT_ID", ""))

        if app_mode == "TITAN TERMINAL":
            st.divider()
            st.subheader("📡 LIVE FEED")
            symbol_in = st.text_input("Asset", value="BTC")
            symbol = f"{symbol_in.upper()}USDT" if not symbol_in.upper().endswith("USDT") else symbol_in.upper()
            c1, c2 = st.columns(2)
            with c1: tf = st.selectbox("TF", ["15m", "1h", "4h", "1d"], index=1)
            with c2: limit = st.slider("Depth", 100, 500, 200)
            
            st.subheader("🧠 ENGINE PARAMS")
            titan_params = {
                "amp": st.number_input("Amp", 2, 200, 10),
                "dev": st.number_input("Dev", 0.5, 10.0, 3.0, 0.1),
                "hma_len": st.number_input("HMA", 2, 400, 50),
                "gann_len": st.number_input("Gann", 1, 50, 3),
                "tp1": st.number_input("TP1 (R)", 1.5), "tp2": 3.0, "tp3": 5.0
            }

        else: # Apex Scanner Mode
            st.divider()
            st.info("Universe: Major Crypto Pairs (25+ Assets)")

    # --- Header ---
    st.title(f"💠 {app_mode}")
    components.html("""<div id="clk" style="font-family:'Roboto Mono';color:#3fb950;text-align:center;font-weight:bold;text-shadow:0 0 10px rgba(63,185,80,0.4);">Loading Clock...</div><script>setInterval(()=>{document.getElementById('clk').innerText='UTC: '+new Date().toLocaleTimeString('en-GB',{timeZone:'UTC'})},1000);</script>""", height=30)

    # ==================================================
    # MODE 1: TITAN TERMINAL (Single Asset)
    # ==================================================
    if app_mode == "TITAN TERMINAL":
        df = get_klines_binance(symbol, tf, limit)
        
        if not df.empty:
            df = TitanEngine.run_engine(df, titan_params)
            last = df.iloc[-1]
            fibs = TitanEngine.calculate_fibonacci(df)
            fg = TitanEngine.calculate_fear_greed_index(df)
            smart_stop = min(last['entry_stop'], fibs['fib_618']*0.9995) if last['is_bull'] else max(last['entry_stop'], fibs['fib_618']*1.0005)

            # Top Metrics
            c_head1, c_head2 = st.columns([2, 1])
            with c_head1:
                 components.html(f"""<div class="tradingview-widget-container"><div class="tradingview-widget-container__widget"></div><script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js" async>{{ "symbol": "BINANCE:{symbol}", "width": "100%", "colorTheme": "dark", "isTransparent": true, "locale": "en" }}</script></div>""", height=120)
            with c_head2:
                st.metric("TREND", "BULL" if last['gann_trend']==1 else "BEAR", delta=f"{last['close']:.2f}", delta_color="off")
                st.caption(f"TF: {tf} | HMA: {titan_params['hma_len']}")

            # Main Chart
            st.plotly_chart(render_chart(df), use_container_width=True, config={'displayModeBar': False})
            
            # Report & Actions
            report_html = render_titan_report(last, fibs, fg, smart_stop)
            st.markdown(report_html, unsafe_allow_html=True)
            
            b1, b2 = st.columns(2)
            with b1:
                if st.button("🔥 ALERT TG", use_container_width=True):
                    msg = f"<b>TITAN SIGNAL: {symbol}</b>\nTrend: {'LONG' if last['is_bull'] else 'SHORT'}\nPrice: {last['close']}"
                    if send_telegram_msg(tg_token, tg_chat, msg): st.toast("Sent!", icon="✅")
                    else: st.toast("Failed", icon="❌")
            with b2:
                if st.button("📄 FULL REPORT", use_container_width=True):
                    clean_rep = report_html.replace("<div", "\n<div").replace("</span>", "")
                    if send_telegram_msg(tg_token, tg_chat, f"REPORT: {symbol}\n{clean_rep}"): st.toast("Sent!", icon="✅")
            
            # Secondary Visuals
            t1, t2 = st.tabs(["📊 GANN", "🌊 FLOW"])
            with t1:
                f_gann = go.Figure()
                f_gann.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])
                df_g = df.dropna(subset=['gann_act'])
                f_gann.add_trace(go.Scatter(x=df_g['timestamp'], y=df_g['gann_act'], mode='markers', marker=dict(color=np.where(df_g['gann_trend']==1, '#3fb950', '#f85149'), size=4)))
                f_gann.update_layout(height=250, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False)
                st.plotly_chart(f_gann, use_container_width=True)
            with t2:
                f_flow = go.Figure()
                f_flow.add_trace(go.Bar(x=df['timestamp'], y=df['money_flow'], marker_color=['#3fb950' if x > 0 else '#f85149' for x in df['money_flow']]))
                f_flow.update_layout(height=250, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False)
                st.plotly_chart(f_flow, use_container_width=True)
            
            # Backtest
            total, win_rate, net_r = TitanEngine.run_backtest(df, titan_params['tp1'])
            st.caption(f"📊 Session Stats: {win_rate:.1f}% Win Rate | {net_r:.1f}R Net ({total} Trades)")

    # ==================================================
    # MODE 2: APEX SCANNER (Multi Asset)
    # ==================================================
    elif app_mode == "APEX SCANNER":
        if "apex_results" not in st.session_state: st.session_state.apex_results = None
        
        CRYPTO_UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "MATIC-USD", "DOT-USD", "NEAR-USD", "ATOM-USD", "ARB-USD", "OP-USD", "SUI-USD", "APT-USD", "UNI-USD", "AAVE-USD", "MKR-USD", "INJ-USD", "RUNE-USD", "DOGE-USD", "SHIB-USD", "PEPE-USD", "WIF-USD", "FET-USD"]

        if st.button("🏛️ EXECUTE SCANNER", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            results = []
            
            for i, ticker in enumerate(CRYPTO_UNIVERSE):
                status.text(f"Scanning: {ticker}...")
                progress.progress((i+1)/len(CRYPTO_UNIVERSE))
                
                data = get_financials_yf(ticker)
                if not data: continue
                
                hist = get_history_yf(ticker)
                if hist is None or len(hist) < 60: continue
                
                apex = ApexEngine.run_full_analysis(hist)
                if not apex: continue
                
                # Scoring
                score = 0; tags = []
                if apex['Trend'] == "Bullish 🟢": score += 1
                if apex['Apex_Buy_Signal']: score += 3; tags.append("BUY SIGNAL")
                if apex['BOS_Alert']: score += 2; tags.append("BOS")
                if apex['FVG_Detected']: score += 1; tags.append("FVG")
                
                if score >= 1:
                    row = data.copy(); row.update(apex)
                    row['Score'] = score; row['Tags'] = ", ".join(tags)
                    results.append(row)
            
            progress.empty(); status.empty()
            if results:
                df_res = pd.DataFrame(results).sort_values(by='Score', ascending=False).reset_index(drop=True)
                
                # Run AI Analysis on Top 5
                if api_key:
                    st.toast("Running AI Analysis on Top 5...")
                    for idx in df_res.head(5).index:
                        df_res.loc[idx, 'AI_Analysis'] = analyze_smc_with_ai(df_res.loc[idx], api_key)
                
                st.session_state.apex_results = df_res
            else:
                st.warning("No Setups Found.")

        # Render Results
        if st.session_state.apex_results is not None:
            df = st.session_state.apex_results
            
            st.markdown("### 🏛️ MARKET MATRIX")
            for i, row in df.iterrows():
                with st.expander(f"{row['ticker']} | {row['Tags']} (Score: {row['Score']})"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Price", f"${row['Price']:.4f}")
                    c2.metric("Trend", row['Trend'])
                    c3.metric("WaveTrend", f"{row['WaveTrend']:.1f}")
                    c4.metric("FVG", "YES" if row['FVG_Detected'] else "NO")
                    
                    if 'AI_Analysis' in row and pd.notna(row['AI_Analysis']):
                        st.info(f"🤖 **AI SCALPER:** {row['AI_Analysis']}")

            # Export Tools
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name="Apex SMC")
            buf.seek(0)
            
            c_ex1, c_ex2 = st.columns(2)
            with c_ex1:
                st.download_button("📥 DOWNLOAD XLS", buf, "Apex_SMC_Scan.xlsx", "application/vnd.ms-excel", use_container_width=True)
            with c_ex2:
                if st.button("📡 BROADCAST TOP PICK", use_container_width=True):
                    top = df.iloc[0]
                    msg = f"🏛️ **APEX TOP PICK**\n\nAsset: {top['ticker']}\nTrend: {top['Trend']}\nTags: {top['Tags']}"
                    if send_telegram_doc(tg_token, tg_chat, msg, buf, "Apex_Scan.xlsx"): st.success("Broadcast Sent!")
                    else: st.error("Broadcast Failed")

if __name__ == "__main__":
    main()
