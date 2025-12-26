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
import streamlit.components.v1 as components

# ==========================================
# 0. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TITAN OMNI V2",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Persistent State Initialization
if 'run_analysis' not in st.session_state: st.session_state['run_analysis'] = False
if 'ai_report' not in st.session_state: st.session_state['ai_report'] = ""

# --- LIBRARY CHECKS ---
try:
    import google.generativeai as genai
except ImportError: genai = None
try:
    from openai import OpenAI
except ImportError: OpenAI = None
try:
    import yfinance as yf
except ImportError:
    st.error("CRITICAL: `yfinance` missing. Install via: pip install yfinance")
    st.stop()

# ==========================================
# 1. UI ENGINE & CSS
# ==========================================
def inject_titan_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;800&family=Roboto+Mono:wght@400;700&display=swap');
        
        /* THEME */
        .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Rajdhani', sans-serif; }
        
        /* NEON HEADER */
        .titan-title {
            font-size: 3.2rem; font-weight: 800; letter-spacing: 2px;
            background: linear-gradient(90deg, #00E5FF, #2979FF);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(0, 229, 255, 0.4); margin: 0;
        }
        
        /* CLOCK CONTAINER */
        .clock-container {
            display: flex; justify-content: space-around; background: #111; 
            border: 1px solid #333; border-radius: 8px; padding: 10px; margin-bottom: 20px;
            font-family: 'Roboto Mono', monospace; color: #00E5FF;
        }
        .clock-item { text-align: center; }
        .clock-time { font-size: 1.2rem; font-weight: bold; color: #fff; }
        .clock-label { font-size: 0.8rem; color: #888; }

        /* METRICS */
        div[data-testid="metric-container"] {
            background: #0f0f0f; border: 1px solid #333; border-left: 3px solid #00E5FF;
            border-radius: 6px; padding: 10px; transition: 0.3s;
        }
        div[data-testid="metric-container"]:hover {
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.15); border-color: #00E5FF;
        }
        div[data-testid="stMetricValue"] { color: #fff !important; font-family: 'Roboto Mono'; font-weight: 700; }
        div[data-testid="stMetricLabel"] { color: #00E5FF !important; }

        /* BUTTONS */
        .stButton > button {
            background: linear-gradient(135deg, #1a1a1a, #000); border: 1px solid #00E5FF; color: #00E5FF;
            font-family: 'Rajdhani'; font-weight: 800; font-size: 1.1rem; width: 100%; transition: 0.3s;
        }
        .stButton > button:hover { background: #00E5FF; color: #000; box-shadow: 0 0 20px #00E5FF; }
        
        /* TABS */
        .stTabs [aria-selected="true"] { background: #00E5FF; color: #000 !important; font-weight: bold; }
        
        /* MOBILE CARD */
        .report-card {
            background: #111; border: 1px solid #333; border-radius: 8px; padding: 15px; margin-bottom: 15px;
        }
        .card-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-family: 'Roboto Mono'; font-size: 0.9rem; border-bottom: 1px solid #222; padding-bottom: 4px; }
        .bull { color: #00E676; font-weight: bold; }
        .bear { color: #FF1744; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- LIVE WORLD CLOCK WIDGET ---
def render_live_clock():
    html = """
    <div style="display: flex; justify-content: space-around; background: #0a0a0a; border: 1px solid #333; padding: 10px; border-radius: 8px; font-family: 'Roboto Mono', monospace;">
        <div style="text-align: center;"><div id="utc-time" style="font-size: 1.5em; color: #fff; font-weight: bold;"></div><div style="color: #00E5FF; font-size: 0.8em;">UTC (ZULU)</div></div>
        <div style="text-align: center;"><div id="ny-time" style="font-size: 1.5em; color: #fff; font-weight: bold;"></div><div style="color: #888; font-size: 0.8em;">NEW YORK</div></div>
        <div style="text-align: center;"><div id="lon-time" style="font-size: 1.5em; color: #fff; font-weight: bold;"></div><div style="color: #888; font-size: 0.8em;">LONDON</div></div>
        <div style="text-align: center;"><div id="tok-time" style="font-size: 1.5em; color: #fff; font-weight: bold;"></div><div style="color: #888; font-size: 0.8em;">TOKYO</div></div>
    </div>
    <script>
    function updateClock() {
        const now = new Date();
        const opts = {hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false};
        document.getElementById('utc-time').innerText = now.toLocaleTimeString('en-US', {...opts, timeZone: 'UTC'});
        document.getElementById('ny-time').innerText = now.toLocaleTimeString('en-US', {...opts, timeZone: 'America/New_York'});
        document.getElementById('lon-time').innerText = now.toLocaleTimeString('en-US', {...opts, timeZone: 'Europe/London'});
        document.getElementById('tok-time').innerText = now.toLocaleTimeString('en-US', {...opts, timeZone: 'Asia/Tokyo'});
    }
    setInterval(updateClock, 1000); updateClock();
    </script>
    """
    components.html(html, height=85)

# --- TRADINGVIEW WIDGETS ---
def render_tv_widgets(symbol):
    # Map Symbol to TV Format
    if "BTC" in symbol: tv_sym = "BINANCE:BTCUSD"
    elif "ETH" in symbol: tv_sym = "BINANCE:ETHUSD"
    elif "SPY" in symbol: tv_sym = "AMEX:SPY"
    elif "EURUSD" in symbol: tv_sym = "FX:EURUSD"
    else: tv_sym = symbol 

    # Ticker Tape
    tape = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
      {{ "symbols": [ {{ "proName": "{tv_sym}", "title": "{symbol}" }}, {{ "proName": "FOREXCOM:SPXUSD", "title": "S&P 500" }}, {{ "proName": "FOREXCOM:NSXUSD", "title": "Nasdaq 100" }}, {{ "proName": "FX_IDC:EURUSD", "title": "EUR/USD" }} ], "showSymbolLogo": true, "colorTheme": "dark", "isTransparent": true, "displayMode": "adaptive", "locale": "en" }}
      </script>
    </div>
    """
    components.html(tape, height=50)

def render_mobile_card(last):
    sc = last['GM_Score']
    s_cls = "bull" if sc > 0 else "bear" if sc < 0 else ""
    html = f"""
    <div class="report-card">
        <h3 style="margin-top:0; color:#00E5FF">💠 FIELD REPORT</h3>
        <div class="card-row"><span>GOD MODE SCORE</span><span class="{s_cls}">{sc:.0f}/4</span></div>
        <div class="card-row"><span>VECTOR GATE</span><span class="{'bear' if last['Vector_Locked'] else 'bull'}">{'LOCKED' if last['Vector_Locked'] else 'OPEN'}</span></div>
        <div class="card-row"><span>APEX FLUX</span><span>{last['Apex_Flux']:.2f}</span></div>
        <div class="card-row"><span>TRAILING STOP</span><span>{last['Dark_Vector_Stop']:.2f}</span></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ==========================================
# 2. QUANTUM MATH KERNEL (FULL LOGIC)
# ==========================================
class QuantumCore:
    """
    Integrates Dark Vector (Trend + Chop) and Apex Vector (Flux + Efficiency)
    """
    
    @staticmethod
    def tanh_clamp(x): 
        return (np.exp(2.0 * np.clip(x, -20.0, 20.0)) - 1.0) / (np.exp(2.0 * np.clip(x, -20.0, 20.0)) + 1.0)
    
    @staticmethod
    def wma(s, l): 
        w = np.arange(1, l + 1)
        return s.rolling(l).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

    @staticmethod
    def hma(s, l):
        return QuantumCore.wma(2 * QuantumCore.wma(s, int(l/2)) - QuantumCore.wma(s, l), int(np.sqrt(l)))

    @staticmethod
    def zlema(s, l):
        return (s + (s - s.shift(int((l-1)/2)))).ewm(span=l, adjust=False).mean()

    @staticmethod
    def atr(df, l=14):
        h, l_low, c_prev = df['High'], df['Low'], df['Close'].shift()
        return pd.concat([h-l_low, (h-c_prev).abs(), (l_low-c_prev).abs()], axis=1).max(axis=1).rolling(l).mean()

    @staticmethod
    def calc_pipeline(df):
        # 1. PHYSICS (CHEDO Entropy + Reynolds)
        ret = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        v = ret.rolling(50).std() / (np.abs(ret.rolling(50).mean()) + 1e-9)
        kappa = QuantumCore.tanh_clamp(np.log(np.abs(ret)*v + np.sqrt((np.abs(ret)*v)**2 + 1)).rolling(50).mean())
        
        def shannon(x): 
            c, _ = np.histogram(x, bins=10)
            p = c[c>0]/c.sum()
            return -np.sum(p*np.log(p))
        
        ent = ret.rolling(50).apply(shannon, raw=True)
        df['CHEDO'] = (2/(1+np.exp(-(0.4*kappa + 0.6*ent)*4)) - 1).rolling(3).mean()

        atr14 = QuantumCore.atr(df, 14)
        Re = (df['Volume'] * df['Close'].diff().abs() * atr14) / (df['Close'].rolling(252).std() + 1e-9)
        df['Reynolds'] = Re
        k_q = np.sqrt(np.maximum(0, df['Close'].rolling(252).std() - (df['Close']-df['Close'].shift(252)).abs()))/atr14
        df['Omega_Mag'] = (1/np.cosh(k_q)**2) * QuantumCore.tanh_clamp(Re/(Re.rolling(252).mean()+1e-9))

        # =========================================================
        # 2. DARK VECTOR [SuperTrend Logic + Noise Gate]
        # =========================================================
        # SuperTrend Architecture (Trend Factor 4.0, ATR 10)
        hl2 = (df['High'] + df['Low']) / 2
        matr = 4.0 * QuantumCore.atr(df, 10)
        
        up = hl2 + matr
        dn = hl2 - matr
        
        st = np.zeros(len(df)) # SuperTrend Line (Stop Loss)
        d = np.zeros(len(df))  # Direction (1=Bull, -1=Bear)
        
        # Initialize
        st[0] = dn[0]
        d[0] = 1
        
        vals = df['Close'].values
        u_v = up.values
        d_v = dn.values
        
        for i in range(1, len(df)):
            if vals[i-1] > st[i-1]:
                st[i] = max(d_v[i], st[i-1]) if vals[i] > st[i-1] else u_v[i]
                d[i] = 1 if vals[i] > st[i-1] else -1
                if vals[i] < d_v[i] and d[i-1] == 1: 
                    st[i] = u_v[i]
                    d[i] = -1
            else:
                st[i] = min(u_v[i], st[i-1]) if vals[i] < st[i-1] else d_v[i]
                d[i] = -1 if vals[i] < st[i-1] else 1
                if vals[i] > u_v[i] and d[i-1] == -1: 
                    st[i] = d_v[i]
                    d[i] = 1
                    
        df['Vector_Trend'] = d
        df['Dark_Vector_Stop'] = st # EXPLICIT TRAILING STOP COLUMN

        # Choppiness Index (Noise Gate)
        # Log10(Sum(ATR, 14) / (MaxHigh - MinLow)) / Log10(14) * 100
        atr1 = QuantumCore.atr(df, 1)
        sum_atr = atr1.rolling(14).sum()
        range_max_min = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
        
        ci_num = np.log10(sum_atr / (range_max_min + 1e-9))
        ci_denom = np.log10(14)
        df['Chop_Index'] = 100 * ci_num / ci_denom
        df['Vector_Locked'] = df['Chop_Index'] > 60 # Threshold 60

        # =========================================================
        # 3. APEX VECTOR [Flux + Efficiency]
        # =========================================================
        # Efficiency: Body / Range
        range_abs = df['High'] - df['Low']
        body_abs = (df['Close'] - df['Open']).abs()
        
        # FIX: Ensure result uses the Index
        raw_eff = np.where(range_abs == 0, 0.0, body_abs / range_abs)
        efficiency = pd.Series(raw_eff, index=df.index).ewm(span=14, adjust=False).mean() # Force Index
        
        # Volume Flux: Volume / SMA(Vol, 55)
        vol_avg = df['Volume'].rolling(55).mean()
        
        # FIX: Ensure result uses the Index
        vol_fact_vals = np.where(vol_avg == 0, 1.0, df['Volume'] / vol_avg)
        vol_fact = pd.Series(vol_fact_vals, index=df.index)
        
        # Vector Calculation
        direction_sign = np.sign(df['Close'] - df['Open'])
        
        # NOTE: All operands now share the same index, preventing ValueError
        vector_raw = direction_sign * efficiency * vol_fact
        
        # Flux (Smoothing EMA 5)
        df['Apex_Flux'] = pd.Series(vector_raw).ewm(span=5, adjust=False).mean()
        
        # Superconductor Logic
        # > 0.6 = Super Bull, < -0.6 = Super Bear
        # Abs < 0.3 = Resistive
        df['Apex_State'] = np.where(df['Apex_Flux'] > 0.6, "SUPER_BULL",
                           np.where(df['Apex_Flux'] < -0.6, "SUPER_BEAR",
                           np.where(df['Apex_Flux'].abs() < 0.3, "RESISTIVE", "HEAT")))

        # =========================================================
        # 4. APEX DIVERGENCE ENGINE
        # =========================================================
        # We need pivots on Flux to compare with Price
        lookback = 5
        # Find local max/min indices using rolling window
        flux_series = df['Apex_Flux']
        price_high = df['High']
        price_low = df['Low']
        
        # Simple pivot detection (equivalent to ta.pivothigh/low)
        # We mark 1 where a pivot occurs
        df['Piv_H_Flux'] = (flux_series == flux_series.rolling(window=lookback*2+1, center=True).max()).astype(int)
        df['Piv_L_Flux'] = (flux_series == flux_series.rolling(window=lookback*2+1, center=True).min()).astype(int)
        
        # Divergence Logic (Simplified for Python vectorization)
        # In a real streaming app, we'd iterate, but here we can check recent pivots
        # For visualization, we will just pass the flux series for plotting
        
        # =========================================================
        # 5. REMAINING INDICATORS (GANN, MCM, F&G)
        # =========================================================
        # MCM Cloud
        hma55 = QuantumCore.hma(df['Close'], 55)
        atr55 = QuantumCore.atr(df, 55)
        df['MCM_Upper'] = hma55 + atr55*1.5
        df['MCM_Lower'] = hma55 - atr55*1.5
        df['MCM_Trend'] = np.where(df['Close'] > df['MCM_Upper'], 1, np.where(df['Close'] < df['MCM_Lower'], -1, 0))
        df['MCM_Trend'] = df['MCM_Trend'].replace(to_replace=0, method='ffill')
        
        # Gann
        h_ma = df['High'].rolling(3).mean()
        l_ma = df['Low'].rolling(3).mean()
        act, gt = np.zeros(len(df)), np.zeros(len(df))
        act[0] = l_ma[0]; gt[0] = 1
        hm_v, lm_v = h_ma.values, l_ma.values
        for i in range(1, len(df)):
            if gt[i-1] == 1:
                if vals[i] < act[i-1]: gt[i], act[i] = -1, hm_v[i]
                else: gt[i], act[i] = 1, lm_v[i]
            else:
                if vals[i] > act[i-1]: gt[i], act[i] = 1, lm_v[i]
                else: gt[i], act[i] = -1, hm_v[i]
        df['Gann_Activator'], df['Gann_Trend'] = act, gt
        
        # F&G v4
        delta = df['Close'].diff()
        rsi = 100 - (100/(1+(delta.where(delta>0,0).ewm(alpha=1/14).mean()/(-delta.where(delta<0,0).ewm(alpha=1/14).mean()))))
        zs, zl = QuantumCore.zlema(df['Close'],50), QuantumCore.zlema(df['Close'],200)
        ts = np.where((df['Close']>zs)&(zs>zl), 75, np.where(df['Close']>zs, 60, np.where((df['Close']<zs)&(zs<zl), 25, 40)))
        df['FG_Index'] = (rsi*0.3 + ts*0.2 + 50*0.5).rolling(3).mean()
        
        # SMC & Squeeze
        df['Pivot_H'] = df['High'][(df['High'].shift(1)<df['High']) & (df['High'].shift(-1)<df['High'])]
        df['Pivot_L'] = df['Low'][(df['Low'].shift(1)>df['Low']) & (df['Low'].shift(-1)>df['Low'])]
        df['Sqz_Mom'] = df['Close'].rolling(20).apply(lambda y: linregress(np.arange(20), y)[0], raw=True) * 100

        # God Mode Score
        df['GM_Score'] = np.where(df['MCM_Trend']==1, 1, -1) + np.where(df['Gann_Trend']==1, 1, -1) + np.where(df['Vector_Trend']==1, 1, -1) + np.sign(df['Sqz_Mom'])
        
        return df.dropna()

    @staticmethod
    def run_monte_carlo(df):
        ret = df['Close'].pct_change().dropna()
        m, s = ret.mean(), ret.std()
        sim_r = np.random.normal(m, s, (30, 50))
        paths = np.zeros((30, 50))
        paths[0] = df['Close'].iloc[-1]
        for t in range(1, 30): 
            paths[t] = paths[t-1] * (1 + sim_r[t])
        return paths

    @staticmethod
    def calc_vp(df, bins=70):
        price_range = df['High'].max() - df['Low'].min()
        if price_range == 0: return None
        
        hist, bin_edges = np.histogram(df['Close'], bins=bins, weights=df['Volume'])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        max_idx = np.argmax(hist)
        poc_vol = hist[max_idx]
        poc_price = bin_centers[max_idx]
        
        total_vol = np.sum(hist)
        target_vol = total_vol * 0.70
        current_vol = poc_vol
        l_idx, r_idx = max_idx, max_idx
        
        while current_vol < target_vol:
            l_vol = hist[l_idx-1] if l_idx > 0 else 0
            r_vol = hist[r_idx+1] if r_idx < len(hist)-1 else 0
            if l_vol == 0 and r_vol == 0: break
            if l_vol > r_vol:
                current_vol += l_vol; l_idx -= 1
            else:
                current_vol += r_vol; r_idx += 1
                
        return {
            "hist": hist,
            "bins": bin_centers,
            "poc": poc_price,
            "vah": bin_edges[r_idx+1],
            "val": bin_edges[l_idx],
            "va_indices": list(range(l_idx, r_idx+1))
        }

# ==========================================
# 3. DATA ENGINE & CREDENTIALS
# ==========================================
def load_secrets():
    k = {"gem": "", "oai": "", "tg_t": "", "tg_c": ""}
    s = {"gem": False, "oai": False, "tg": False}
    try:
        if hasattr(st, 'secrets'):
            k["gem"] = st.secrets.get("GEMINI_API_KEY", "")
            k["oai"] = st.secrets.get("OPENAI_API_KEY", "")
            k["tg_t"] = st.secrets.get("TELEGRAM_TOKEN", "")
            k["tg_c"] = st.secrets.get("TELEGRAM_CHAT_ID", "")
    except FileNotFoundError: pass 
    except Exception: pass
    
    if k["gem"]: s["gem"] = True
    if k["oai"]: s["oai"] = True
    if k["tg_t"] and k["tg_c"]: s["tg"] = True
    return k, s

class DataEngine:
    @staticmethod
    @st.cache_data(ttl=60)
    def fetch(ticker, timeframe):
        p_map = {
            "15m": ("60d", "15m"), 
            "1h": ("730d", "1h"), 
            "4h": ("730d", "1h"), 
            "1d": ("5y", "1d"), 
            "1wk": ("5y", "1wk")
        }
        p, i = p_map.get(timeframe, ("1y", "1d"))
        
        try:
            df = yf.download(ticker, period=p, interval=i, progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                try: df.columns = df.columns.droplevel(1) 
                except: df.columns = df.columns.get_level_values(0)
            
            if df.empty:
                st.error(f"⚠️ YFinance returned no data for {ticker}.")
                return None
            
            if timeframe == "4h":
                df = df.resample("4h").agg({
                    'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'
                }).dropna()
            
            return df
            
        except Exception as e:
            st.error(f"⚠️ Data Fetch Error: {str(e)}")
            return None

    @staticmethod
    def get_macro_correlation(ticker, period="90d"):
        basket = {
            "Ticker": ticker, "SPX": "SPY", "NDX": "QQQ", 
            "BTC": "BTC-USD", "DXY": "DX-Y.NYB", "GOLD": "GC=F", "VIX": "^VIX"
        }
        try:
            df = yf.download(list(basket.values()), period=period, interval="1d", progress=False)['Close']
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(0)
            rev_map = {v: k for k, v in basket.items()}
            df = df.rename(columns=rev_map)
            df = df.ffill().bfill()
            return df
        except: return None

    @staticmethod
    def get_seasonality(ticker):
        try:
            df = yf.download(ticker, period="10y", interval="1mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            df['R'] = df['Close'].pct_change()*100; df['M'] = df.index.month; df['Y'] = df.index.year
            return df.pivot_table(index='Y', columns='M', values='R')
        except: return None

# ==========================================
# 4. CHART RENDERING
# ==========================================
def render_charts(df, ticker, show_opt, vp_levels=None):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.15, 0.2], vertical_spacing=0.03, subplot_titles=("PRICE & DARK VECTOR STOP", "APEX FLUX VECTOR", "OMEGA", "ENTROPY"))
    
    # 1. Price
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    
    # Dark Vector Trailing Stop (New)
    fig.add_trace(go.Scatter(x=df.index, y=df['Dark_Vector_Stop'], mode='lines', line=dict(color='white', width=1, dash='dot'), name="DV Stop"), row=1, col=1)

    # Cloud Overlays
    if show_opt['clouds']:
        fig.add_trace(go.Scatter(x=df.index, y=df['MCM_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MCM_Lower'], fill='tonexty', fillcolor='rgba(0, 229, 255, 0.1)', line=dict(width=0), name="Cloud"), row=1, col=1)
    
    if show_opt['gann']:
        fig.add_trace(go.Scatter(x=df.index, y=df['Gann_Activator'], line=dict(color='#FFD700', dash='dot'), name="Gann"), row=1, col=1)
    
    if show_opt['smc']:
        fig.add_trace(go.Scatter(x=df.index, y=df['Pivot_H'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=8), name="Swing High"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Pivot_L'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=8), name="Swing Low"), row=1, col=1)

    if vp_levels:
        fig.add_hline(y=vp_levels['poc'], line_dash="solid", line_color="#FF6D00", annotation_text="POC", row=1, col=1)
        fig.add_hline(y=vp_levels['vah'], line_dash="dot", line_color="#00E5FF", annotation_text="VAH", row=1, col=1)
        fig.add_hline(y=vp_levels['val'], line_dash="dot", line_color="#00E5FF", annotation_text="VAL", row=1, col=1)

    # Signal Arrows
    gm_flip = df['GM_Score'].diff()
    buy_sigs = df[ (df['GM_Score'] > 0) & (df['GM_Score'].shift(1) <= 0) ]
    sell_sigs = df[ (df['GM_Score'] < 0) & (df['GM_Score'].shift(1) >= 0) ]
    
    if not buy_sigs.empty:
        fig.add_trace(go.Scatter(x=buy_sigs.index, y=buy_sigs['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', color='#00E676', size=12), name="BUY SIG"), row=1, col=1)
    if not sell_sigs.empty:
        fig.add_trace(go.Scatter(x=sell_sigs.index, y=sell_sigs['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', color='#FF1744', size=12), name="SELL SIG"), row=1, col=1)

    # 2. Apex Flux Vector (New)
    flux_cols = ['#00E676' if v > 0.6 else '#FF1744' if v < -0.6 else '#546E7A' for v in df['Apex_Flux']]
    fig.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], marker_color=flux_cols, name="Flux"), row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_hline(y=-0.6, line_dash="dot", line_color="red", row=2, col=1)

    # 3. Omega & Entropy
    fig.add_trace(go.Scatter(x=df.index, y=df['Omega_Mag'], fill='tozeroy', line=dict(color='#2979FF'), name="Omega"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], line=dict(color='#EA80FC'), name="Entropy"), row=4, col=1)
    
    fig.update_layout(height=1000, template="plotly_dark", xaxis_rangeslider_visible=False, paper_bgcolor="#050505", plot_bgcolor="#050505", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

def render_mc(paths):
    median_path = np.median(paths, axis=1)
    p95 = np.percentile(paths, 95, axis=1)
    p5 = np.percentile(paths, 5, axis=1)
    start_p = paths[0,0]
    end_mean = np.mean(paths[-1, :])
    exp_ret = ((end_mean - start_p) / start_p) * 100
    
    fig = go.Figure()
    for i in range(min(50, paths.shape[1])): 
        fig.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(color='rgba(41, 121, 255, 0.05)', width=1), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(y=p95, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(y=p5, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 229, 255, 0.1)', name='90% Conf. Interval', hoverinfo='skip'))
    fig.add_trace(go.Scatter(y=p95, mode='lines', line=dict(color='#00E676', width=1, dash='dot'), name=f'P95 (Bull): {p95[-1]:.2f}'))
    fig.add_trace(go.Scatter(y=p5, mode='lines', line=dict(color='#FF1744', width=1, dash='dot'), name=f'P5 (Bear): {p5[-1]:.2f}'))
    fig.add_trace(go.Scatter(y=median_path, mode='lines', line=dict(color='#FFFFFF', width=2), name=f'Median: {median_path[-1]:.2f}'))
    
    fig.update_layout(
        title=dict(text=f"MONTE CARLO (30D) | Exp. Return: {exp_ret:.2f}%", font=dict(family="Rajdhani", size=20, color="#00E5FF")),
        template="plotly_dark", height=450, paper_bgcolor="#050505", plot_bgcolor="#050505",
        margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def render_vp(vp_data):
    if vp_data is None: return
    colors = []
    for i in range(len(vp_data['hist'])):
        if vp_data['bins'][i] == vp_data['poc']: colors.append('#FF6D00')
        elif i in vp_data['va_indices']: colors.append('rgba(0, 229, 255, 0.6)')
        else: colors.append('rgba(255, 255, 255, 0.1)')
        
    fig = go.Figure(go.Bar(
        x=vp_data['hist'], y=vp_data['bins'], orientation='h', marker_color=colors, name='Volume'
    ))
    fig.add_hrect(y0=vp_data['val'], y1=vp_data['vah'], fillcolor="rgba(0, 229, 255, 0.05)", line_width=0)
    fig.add_hline(y=vp_data['poc'], line_dash="solid", line_color="#FF6D00", annotation_text="POC", annotation_position="top right")
    fig.add_hline(y=vp_data['vah'], line_dash="dot", line_color="#00E5FF", annotation_text="VAH", annotation_position="bottom right")
    fig.add_hline(y=vp_data['val'], line_dash="dot", line_color="#00E5FF", annotation_text="VAL", annotation_position="top right")

    fig.update_layout(
        title=dict(text="VOLUME PROFILE (TPO)", font=dict(family="Rajdhani", size=20, color="#00E5FF")),
        template="plotly_dark", height=500, paper_bgcolor="#050505", plot_bgcolor="#050505",
        margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=True, gridcolor='#222')
    )
    st.plotly_chart(fig, use_container_width=True)

def render_physics_dashboard(df):
    fig = make_subplots(
        rows=2, cols=2, 
        specs=[[{"colspan": 2}, None], [{"type": "xy"}, {"type": "scatter"}]], 
        row_heights=[0.5, 0.5], 
        subplot_titles=("REYNOLDS OSCILLATOR (TURBULENCE)", "ENTROPY REGIME", "PHASE SPACE ATTRACTOR")
    )
    re_mean = df['Reynolds'].mean()
    fig.add_trace(go.Scatter(x=df.index, y=df['Reynolds'], mode='lines', name='Reynolds', line=dict(color='#2979FF', width=1.5), fill='tozeroy', fillcolor='rgba(41,121,255,0.1)'), row=1, col=1)
    fig.add_hline(y=re_mean, line_dash="dot", line_color="white", row=1, col=1)
    
    c_map = ['#00E5FF' if v < 0.5 else '#EA80FC' for v in df['CHEDO']]
    fig.add_trace(go.Bar(x=df.index, y=df['CHEDO'], marker_color=c_map, name='Entropy Level'), row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#FF1744", annotation_text="CHAOS THRESHOLD", row=2, col=1)

    vol = df['Close'].pct_change().rolling(20).std()
    fig.add_trace(go.Scatter(
        x=df['CHEDO'].tail(200), y=vol.tail(200), mode='markers', 
        marker=dict(size=8, color=df.index[-200:].astype('int64'), colorscale='Viridis', showscale=False),
        name='Phase Space'
    ), row=2, col=2)
    fig.update_xaxes(title_text="Entropy (CHEDO)", row=2, col=2)
    fig.update_yaxes(title_text="Volatility", row=2, col=2)
    fig.update_layout(height=700, template="plotly_dark", paper_bgcolor="#050505", plot_bgcolor="#050505", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

def render_macro_dashboard(ticker):
    df_macro = DataEngine.get_macro_correlation(ticker)
    if df_macro is None:
        st.error("Macro Data Unavailable")
        return
    corr = df_macro.corr()
    df_norm = df_macro / df_macro.iloc[0] * 100
    last_vix = df_macro['VIX'].iloc[-1]
    regime = "RISK OFF (FEAR)" if last_vix > 20 else "RISK ON (GREED)"
    regime_col = "#FF1744" if last_vix > 20 else "#00E676"

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"### 🌍 GLOBAL REGIME")
        st.markdown(f"<div style='background:{regime_col}; padding:15px; border-radius:10px; text-align:center; font-weight:bold; color:black;'>{regime}<br><span style='font-size:0.8em'>VIX: {last_vix:.2f}</span></div>", unsafe_allow_html=True)
        st.markdown("### 🔗 CORRELATION MATRIX")
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", template="plotly_dark", aspect="auto")
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
    with c2:
        st.markdown(f"### 🏎️ RELATIVE PERFORMANCE (90D)")
        fig_perf = go.Figure()
        colors = {"Ticker": "#00E5FF", "SPX": "#FFD700", "BTC": "#FF9100", "DXY": "#9E9E9E", "VIX": "#FF1744"}
        for col in df_norm.columns:
            if col == "GOLD": continue
            c = colors.get(col, "#FFFFFF")
            w = 3 if col == "Ticker" else 1
            fig_perf.add_trace(go.Scatter(x=df_norm.index, y=df_norm[col], mode='lines', name=col, line=dict(color=c, width=w)))
        fig_perf.update_layout(template="plotly_dark", height=500, yaxis_title="% Return", paper_bgcolor="#050505", plot_bgcolor="#050505")
        st.plotly_chart(fig_perf, use_container_width=True)
    st.markdown("### 📅 HISTORICAL SEASONALITY")
    hm = DataEngine.get_seasonality(ticker)
    if hm is not None:
        fig_hm = px.imshow(hm, color_continuous_scale='RdYlGn', text_auto='.1f', template="plotly_dark", aspect="auto")
        st.plotly_chart(fig_hm, use_container_width=True)

# ==========================================
# 5. INTELLIGENCE & BROADCASTING
# ==========================================
class Intelligence:
    @staticmethod
    def generate_strategy_prompt(ticker, timeframe, last, sc, vp_levels, reynolds):
        vp_txt = f"VAH: {vp_levels['vah']:.2f}, VAL: {vp_levels['val']:.2f}, POC: {vp_levels['poc']:.2f}" if vp_levels else "N/A"
        turb = "HIGH" if abs(reynolds) > 2 else "LOW"
        
        return f"""
        ACT AS A SENIOR QUANTITATIVE HEDGE FUND MANAGER.
        ANALYZE: {ticker} ({timeframe})
        PRICE: {last['Close']:.2f}

        SYSTEM METRICS:
        - God Mode Score: {sc}/4 ({'BULL' if sc>0 else 'BEAR'})
        - Entropy: {last['CHEDO']:.2f} (High>0.5 = Chaos)
        - Trend (MCM): {'BULL' if last['MCM_Trend']==1 else 'BEAR'}
        - Apex State: {last['Apex_State']} (Flux: {last['Apex_Flux']:.2f})
        - Trailing Stop (Dark Vector): {last['Dark_Vector_Stop']:.2f}
        - Volatility Gate: {'LOCKED' if last['Vector_Locked'] else 'OPEN'}

        QUANT LEVELS (VOLUME PROFILE):
        {vp_txt}

        MISSION:
        Provide a concise, military-grade strategic assessment.
        1. REGIME IDENTIFICATION: (Trending, Mean Reverting, or Chaos?)
        2. KEY LEVELS: Use the VP levels (VAH/VAL) to define buy/sell zones.
        3. EXECUTION PLAN: Define clear Entry, Stop, and Target.
        4. MACRO CHECK: Briefly mention if this setup aligns with a Risk-On or Risk-Off environment.
        """

    @staticmethod
    def construct_telegram_msg(template, ticker, timeframe, last, sc):
        base = f"🔥 *TITAN SIGNAL: {ticker}*\n"
        # ENSURE STOP (DV) IS IN ALL TEMPLATES AS REQUESTED
        if template == "Scalp":
            return base + f"⏱️ TF: {timeframe}\n💰 Price: {last['Close']:.2f}\n🚀 Momentum: {last['Sqz_Mom']:.1f}\n🛑 Stop (DV): {last['Dark_Vector_Stop']:.2f}\n🌊 Flux: {last['Apex_Flux']:.2f}"
        elif template == "Swing":
            return base + f"🌊 Trend: {'BULL' if last['MCM_Trend']==1 else 'BEAR'}\n🎯 Score: {sc}/4\n🛡️ Stop (DV): {last['Dark_Vector_Stop']:.2f}\n🔮 Entropy: {last['CHEDO']:.2f}\n⚡ Flux: {last['Apex_Flux']:.2f}"
        elif template == "Executive":
            return base + f"📊 *EXECUTIVE BRIEF*\nPrice: {last['Close']:.2f}\nScore: {sc}\nApex: {last['Apex_State']}\nVol Lock: {last['Vector_Locked']}\nStop Ref: {last['Dark_Vector_Stop']:.2f}"
        else: # Standard
            return base + f"Price: {last['Close']:.2f}\nScore: {sc}\nStop (DV): {last['Dark_Vector_Stop']:.2f}\nFlux: {last['Apex_Flux']:.2f}"

    @staticmethod
    def construct_outlook_msg(ticker, timeframe, last, sc):
        apex_state = last['Apex_State']
        flux = last['Apex_Flux']
        dv_stop = last['Dark_Vector_Stop']
        
        icon = "🟢" if sc > 0 else "🔴"
        if last['Vector_Locked']: icon = "🔒"
        
        return f"""🌍 *TITAN MARKET OUTLOOK*
{icon} *{ticker}* ({timeframe})

*SYSTEM STATE:*
• God Mode: {sc}/4
• Apex Vector: {apex_state} ({flux:.2f})
• Volatility: {'LOCKED (Chop)' if last['Vector_Locked'] else 'OPEN (Trend)'}

*KEY LEVELS:*
• Price: {last['Close']:.2f}
• Trail Stop: {dv_stop:.2f}
• Sentiment: {last['FG_Index']:.0f}/100

_Generated by TITAN OMNI V2_"""

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main():
    inject_titan_css()
    keys, status = load_secrets()
    
    with st.sidebar:
        st.markdown("## 💠 TITAN CONTROL")
        cat = st.selectbox("ASSET CLASS", ["Crypto", "Indices", "Forex", "Custom"])
        if cat == "Crypto": ticker = st.selectbox("TICKER", ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"])
        elif cat == "Indices": ticker = st.selectbox("TICKER", ["SPY", "QQQ", "IWM", "DIA"])
        elif cat == "Forex": ticker = st.selectbox("TICKER", ["EURUSD=X", "GBPUSD=X", "USDJPY=X"])
        else: ticker = st.text_input("CUSTOM TICKER", "BTC-USD")
        
        timeframe = st.selectbox("TIMEFRAME", ["15m", "1h", "4h", "1d", "1wk"], index=2)
        
        with st.expander("🛠️ STRATEGY & RISK", expanded=False):
            st_mult = st.slider("Vector Sens.", 1.0, 6.0, 4.0)
            gann_len = st.slider("Gann Lookback", 2, 10, 3)
            acc_size = st.number_input("Capital ($)", value=10000)
            risk_pct = st.number_input("Risk %", value=1.0)
            
        with st.expander("👁️ OVERLAYS", expanded=False):
            show_clouds = st.checkbox("MCM Clouds", True)
            show_gann = st.checkbox("Gann Line", True)
            show_smc = st.checkbox("SMC Pivots", True)
            
        with st.expander("🔐 API KEYS"):
            keys["gem"] = st.text_input("Gemini", value=keys["gem"], type="password")
            keys["oai"] = st.text_input("OpenAI", value=keys["oai"], type="password")
            keys["tg_t"] = st.text_input("TG Token", value=keys["tg_t"], type="password")
            keys["tg_c"] = st.text_input("TG Chat", value=keys["tg_c"])
            if status["gem"]: st.caption("✅ Auto-Loaded")
            
        if st.button("🚀 INITIATE SYSTEM"): st.session_state['run_analysis'] = True

    render_tv_widgets(ticker)
    st.markdown('<h1 class="titan-title">TITAN NEXUS</h1>', unsafe_allow_html=True)
    render_live_clock()
    
    if st.session_state['run_analysis']:
        with st.spinner("QUANTUM CORES ENGAGED..."):
            df = DataEngine.fetch(ticker, timeframe)
            if df is not None:
                df = QuantumCore.calc_pipeline(df)
                last = df.iloc[-1]
                vp_data = QuantumCore.calc_vp(df)
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("PRICE", f"{last['Close']:.2f}", f"{df['Close'].pct_change().iloc[-1]*100:.2f}%")
                sc = last['GM_Score']
                c2.metric("GOD MODE", f"{sc:.0f}/4", "STRONG" if abs(sc)>2 else "WEAK")
                c3.metric("SENTIMENT", "GREED" if last['FG_Index']>60 else "FEAR", f"{last['FG_Index']:.0f}")
                c4.metric("VECTOR", "LOCKED" if last['Vector_Locked'] else "OPEN", "CHOP" if last['Vector_Locked'] else "TREND")
                c5.metric("APEX FLUX", f"{last['Apex_Flux']:.2f}", last['Apex_State'])
                
                t1, t2, t3, t4, t5 = st.tabs(["📈 CHART", "⚛️ PHYSICS", "🎲 QUANT", "🌍 MACRO", "🧠 AI & BROADCAST"])
                
                with t1:
                    render_charts(df, ticker, {"clouds":show_clouds, "gann":show_gann, "smc":show_smc}, vp_levels=vp_data)
                    render_mobile_card(last)
                with t2:
                    render_physics_dashboard(df)
                with t3:
                    c1, c2 = st.columns(2)
                    with c1: render_mc(QuantumCore.run_monte_carlo(df))
                    with c2: render_vp(vp_data)
                    st.markdown("### ⚖️ RISK CALCULATOR")
                    dist = abs(last['Close'] - last['Dark_Vector_Stop']) / last['Close'] # UPDATED TO DARK VECTOR STOP
                    sz = (acc_size * (risk_pct/100)) / (last['Close'] * dist) if dist > 0 else 0
                    r1, r2, r3 = st.columns(3)
                    r1.metric("STOP LOSS (DV)", f"${last['Dark_Vector_Stop']:.2f}")
                    r2.metric("POS SIZE", f"{sz:.4f}")
                    r3.metric("RISK VALUE", f"${acc_size*(risk_pct/100):.2f}")
                with t4:
                    render_macro_dashboard(ticker)
                with t5:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### 🧠 STRATEGIC INTEL")
                        if st.button("GENERATE NEURAL REPORT"):
                            p = Intelligence.generate_strategy_prompt(ticker, timeframe, last, sc, vp_data, last['Reynolds'])
                            r = "NO KEYS"
                            
                            gem_key = keys["gem"].strip()
                            if gem_key and genai:
                                try:
                                    genai.configure(api_key=gem_key)
                                    models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro', 'gemini-pro']
                                    for m_name in models:
                                        try:
                                            model = genai.GenerativeModel(m_name)
                                            r = model.generate_content(p).text
                                            if r: break
                                        except Exception as e: continue
                                except Exception as e: r = f"Gemini Error: {str(e)}"
                            
                            oai_key = keys["oai"].strip()
                            if (r == "NO KEYS" or "Error" in r) and oai_key and OpenAI:
                                try:
                                    client = OpenAI(api_key=oai_key)
                                    response = client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":p}])
                                    r = response.choices[0].message.content
                                except Exception as e: r = f"OpenAI Error: {str(e)}"
                            
                            if r == "NO KEYS": r = "⚠️ AI Service Unavailable. Please check API Keys in sidebar."
                            st.session_state['ai_report'] = r
                        
                        if st.session_state['ai_report']:
                            st.markdown(st.session_state['ai_report'])
                            
                    with c2:
                        st.markdown("### 📡 BROADCAST CENTER")
                        tmpl = st.selectbox("SIGNAL TEMPLATE", ["Standard", "Scalp", "Swing", "Executive", "AI Neural Report"])
                        
                        if tmpl == "AI Neural Report":
                             default_msg = st.session_state.get('ai_report', "⚠️ No Intelligence Report Generated. Please run the AI module first.")
                        else:
                             default_msg = Intelligence.construct_telegram_msg(tmpl, ticker, timeframe, last, sc)
                        
                        msg = st.text_area("PAYLOAD PREVIEW", default_msg, height=300)
                        
                        if st.button("SEND TELEGRAM"):
                            if keys["tg_t"] and keys["tg_c"]:
                                try:
                                    requests.post(f"https://api.telegram.org/bot{keys['tg_t']}/sendMessage", json={"chat_id": keys['tg_c'], "text": msg, "parse_mode": "Markdown"})
                                    st.success("SIGNAL TRANSMITTED 🚀")
                                except Exception as e: st.error(f"Transmission Failed: {str(e)}")
                            else: st.error("NO TELEGRAM KEYS")
                        
                        # NEW BUTTON: MARKET OUTLOOK SNAPSHOT
                        st.markdown("---")
                        if st.button("📨 SEND MARKET OUTLOOK (SNAPSHOT)"):
                            outlook_msg = Intelligence.construct_outlook_msg(ticker, timeframe, last, sc)
                            if keys["tg_t"] and keys["tg_c"]:
                                try:
                                    requests.post(f"https://api.telegram.org/bot{keys['tg_t']}/sendMessage", json={"chat_id": keys['tg_c'], "text": outlook_msg, "parse_mode": "Markdown"})
                                    st.success("OUTLOOK SENT 🌍")
                                except Exception as e: st.error(f"Transmission Failed: {str(e)}")
                            else: st.error("NO TELEGRAM KEYS")

            else: st.error("DATA ERROR")

if __name__ == "__main__":
    main()
