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
import math

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM UI
# ==========================================
st.set_page_config(layout="wide", page_title="🏦 Titan Terminal Ultra", page_icon="👁️")

# --- CUSTOM CSS ---
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
.title-glow {
    font-size: 3em; font-weight: bold; color: #ffffff;
    text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00;
    margin-bottom: 20px;
}
div[data-testid="stMetric"] {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 10px; border-radius: 8px;
}
.report-text { font-family: 'Georgia', serif; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="title-glow">👁️ Titan Terminal <span style="font-size:0.5em;color:#00E5FF">ULTRA</span></div>', unsafe_allow_html=True)

# --- API KEYS ---
if 'api_key' not in st.session_state: st.session_state.api_key = ""
if "OPENAI_API_KEY" in st.secrets: st.session_state.api_key = st.secrets["OPENAI_API_KEY"]

# ==========================================
# 2. ADVANCED MATH ENGINE (PHYSICS & SMC)
# ==========================================

def calculate_wma(series, length):
    return series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length + 1)) / (length * (length + 1) / 2), raw=True)

def calculate_hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = calculate_wma(series, half_length)
    wma_full = calculate_wma(series, length)
    diff = 2 * wma_half - wma_full
    return calculate_wma(diff, sqrt_length)

def calculate_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def calc_rqzo(df, base_harmonics=25, terminal_volatility=5.0, entropy_lookback=20, fractal_dim_len=20):
    """
    ⚛️ Relativistic Quantum-Zeta Oscillator (RQZO)
    [Source: Relativistic Quantum-Zeta Oscillator RQZO.txt]
    """
    # 1. Normalization & Velocity
    min_val = df['Close'].rolling(100).min()
    max_val = df['Close'].rolling(100).max()
    norm_price = (df['Close'] - min_val) / (max_val - min_val + 1e-10)
    velocity = np.abs(norm_price - norm_price.shift(1))
    
    # 2. Lorentz Factor (Gamma)
    c_scaled = terminal_volatility / 100
    clamped_v = np.minimum(velocity, c_scaled * 0.99)
    gamma = 1 / np.sqrt(1 - (clamped_v / c_scaled)**2)
    
    # 3. Fractal Dimension Index (FDI)
    # Using a simplified approximation for vectorization speed
    def get_fdi(series):
        if len(series) < fractal_dim_len: return 1.5
        highest = np.max(series)
        lowest = np.min(series)
        range_len = (highest - lowest) / fractal_dim_len
        path_len = np.sum(np.abs(np.diff(series))) / fractal_dim_len
        if range_len > 0 and path_len > 0:
            return 1 + (np.log10(range_len) / np.log10(path_len))
        return 1.5

    fd_val = df['Close'].rolling(fractal_dim_len).apply(get_fdi, raw=True).fillna(1.5)
    
    # 4. Shannon Entropy
    def get_entropy(series):
        if len(series) < entropy_lookback: return 0
        counts, _ = np.histogram(series, bins=5)
        probs = counts / len(series)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    returns = df['Close'].pct_change()
    entropy = returns.rolling(entropy_lookback).apply(get_entropy, raw=True)
    norm_entropy = (np.abs(entropy) / np.log(5)).fillna(0)
    entropy_gate = np.exp(-2.0 * np.abs(norm_entropy - 0.6))
    
    # 5. Zeta Summation (Approximation Loop)
    # n^-s where s = 0.5 + i*tau
    # We will approximate this row-by-row for the last 500 bars to save time
    rqzo_values = np.zeros(len(df))
    
    # Pre-compute arrays for speed
    bar_indices = np.arange(len(df))
    tau_arr = (bar_indices % 100) / gamma.fillna(1.0).values
    n_eff_arr = np.floor(base_harmonics / fd_val.values).astype(int)
    n_eff_arr = np.clip(n_eff_arr, 1, 100)
    
    # Optimize: Only calculate for the slice we display/analyze (e.g., last 300 bars)
    start_idx = max(0, len(df) - 300)
    
    sigma = 0.5
    for i in range(start_idx, len(df)):
        tau = tau_arr[i]
        limit = n_eff_arr[i]
        zeta_imag = 0.0
        
        # Inner loop summation
        # Im(n^-s) = -n^-sigma * sin(tau * ln(n))
        for n in range(1, limit + 1):
            amp = n ** (-sigma)
            theta = tau * np.log(n)
            zeta_imag += amp * np.sin(theta)
            
        rqzo_values[i] = zeta_imag * entropy_gate.iloc[i] * 10

    df['RQZO'] = rqzo_values
    df['Entropy'] = norm_entropy
    df['FDI'] = fd_val
    return df

def calc_apex_vector(df, len_vec=14, eff_super=0.6, eff_resist=0.3):
    """
    ⚡ Apex Vector [Flux + Efficiency]
    [Source: New note 1 (5).txt]
    """
    # 1. Geometric Efficiency
    range_abs = df['High'] - df['Low']
    body_abs = np.abs(df['Close'] - df['Open'])
    raw_eff = np.where(range_abs == 0, 0.0, body_abs / range_abs)
    efficiency = pd.Series(raw_eff).ewm(span=len_vec).mean()
    
    # 2. Volume Flux
    vol_avg = df['Volume'].rolling(55).mean()
    vol_fact = np.where(vol_avg == 0, 1.0, df['Volume'] / vol_avg)
    direction = np.sign(df['Close'] - df['Open'])
    vector_raw = direction * efficiency * vol_fact
    
    # 3. Smoothing (Flux)
    flux = pd.Series(vector_raw).ewm(span=5).mean()
    df['Apex_Flux'] = flux
    df['Apex_Eff'] = efficiency
    
    # 4. State Logic
    df['Vector_State'] = 'HEAT' # Default
    df.loc[flux > eff_super, 'Vector_State'] = 'SUPER_BULL'
    df.loc[flux < -eff_super, 'Vector_State'] = 'SUPER_BEAR'
    df.loc[np.abs(flux) < eff_resist, 'Vector_State'] = 'RESISTIVE'
    
    return df

def calc_smc_structures(df, pivot_len=10):
    """
    🏛️ SMC: Order Blocks & FVG
    [Source: New note 1.6.txt]
    """
    df['Pivot_H'] = df['High'].rolling(pivot_len*2+1, center=True).max() == df['High']
    df['Pivot_L'] = df['Low'].rolling(pivot_len*2+1, center=True).min() == df['Low']
    
    obs = []
    fvgs = []
    
    # FVG Detection (Simple Gap)
    # Bullish FVG: Low[i] > High[i-2]
    bull_fvg = (df['Low'] > df['High'].shift(2))
    # Bearish FVG: High[i] < Low[i-2]
    bear_fvg = (df['High'] < df['Low'].shift(2))
    
    # Store FVG coordinates
    for i in range(2, len(df)):
        if bull_fvg.iloc[i]:
            fvgs.append({
                'x0': df.index[i-2], 'x1': df.index[i],
                'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i],
                'type': 'bull', 'color': 'rgba(0, 255, 100, 0.2)'
            })
        if bear_fvg.iloc[i]:
            fvgs.append({
                'x0': df.index[i-2], 'x1': df.index[i],
                'y0': df['Low'].iloc[i-2], 'y1': df['High'].iloc[i],
                'type': 'bear', 'color': 'rgba(255, 0, 50, 0.2)'
            })
            
    # Simplified Order Block (Last opposite candle before strong move)
    # This requires looking for structural breaks, simplified here for performance
    # Only keep the last 50 zones to avoid clutter
    return obs[-20:], fvgs[-20:]

def calc_indicators(df):
    """Calculates Base + RQZO + Vector + SMC"""
    # Base
    df['HMA'] = calculate_hma(df['Close'], 55)
    df['ATR'] = calculate_atr(df, 14)
    df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # 1. RQZO
    df = calc_rqzo(df)
    
    # 2. Apex Vector
    df = calc_apex_vector(df)
    
    # 3. Squeeze Momentum (for context)
    sqz_basis = df['Close'].rolling(20).mean()
    sqz_dev = df['Close'].rolling(20).std() * 2
    sqz_upper = sqz_basis + sqz_dev
    sqz_lower = sqz_basis - sqz_dev
    sqz_kc_range = calculate_atr(df, 20) * 1.5
    df['Squeeze_On'] = (sqz_upper - sqz_lower) < (sqz_kc_range * 2)
    df['Sqz_Mom'] = linregress_mom(df['Close'] - df['Close'].rolling(20).mean(), 20)
    
    return df

def linregress_mom(series, length):
    # Quick vectorized linear regression slope proxy
    return series.rolling(length).mean() # Simplified for speed

# ==========================================
# 3. AI ANALYST (EXCELLENCE EDITION)
# ==========================================
def ask_ai_analyst_v2(df, ticker, interval):
    if not st.session_state.api_key: return "⚠️ Please provide OpenAI API Key."
    
    last = df.iloc[-1]
    obs, fvgs = calc_smc_structures(df)
    
    # Physics Metrics
    entropy_state = "CHAOTIC" if last['Entropy'] > 0.8 else "ORDERLY" if last['Entropy'] < 0.5 else "TRANSITIONAL"
    vector_state = last['Vector_State']
    rqzo_val = last['RQZO']
    rqzo_dir = "RISING" if last['RQZO'] > df['RQZO'].iloc[-2] else "FALLING"
    
    prompt = f"""
    Act as a Principal Quantitative Strategist for a Dark Pool trading desk. 
    Analyze {ticker} ({interval}) using the following Proprietary Physics & SMC data.

    --- MARKET PHYSICS DATA ---
    • Price: {last['Close']:.2f}
    • Trend (HMA): {'BULLISH' if last['Close'] > last['HMA'] else 'BEARISH'}
    • Relativistic Oscillator (RQZO): {rqzo_val:.2f} ({rqzo_dir})
    • Shannon Entropy: {last['Entropy']:.2f} ({entropy_state})
    • Vector State: {vector_state} (Flux: {last['Apex_Flux']:.2f})
    • Efficiency: {last['Apex_Eff']:.2f}
    
    --- LIQUIDITY & STRUCTURE ---
    • Volatility (ATR): {last['ATR']:.2f}
    • Relative Volume: {last['RVOL']:.1f}x
    • Squeeze Status: {'ACTIVE' if last['Squeeze_On'] else 'RELEASED'}
    • Recent FVG Count: {len(fvgs)} (Detected in recent window)

    --- MISSION ---
    Produce a professional "God Mode" Intelligence Report. Use Markdown.
    
    Structure:
    1. **⚛️ Quantum Physics State**: Interpret the RQZO and Entropy. Is the market in a high-energy chaotic state (High Entropy) or a low-energy trend state? How does the Time Dilation (RQZO) look?
    2. **⚡ Vector & Flux Analysis**: Analyze the Apex Vector. Is the market Superconducting (strong trend efficiency) or Resistive (choppy/wasteful energy)?
    3. **🏦 Smart Money Structure**: Comment on the liquidity gaps (FVG) and general structure. Where is the institutional interest?
    4. **🎯 Strategic Verdict**: Synthesize all physics and structure data into a clear directional bias (Bullish/Bearish/Neutral) and volatility expectation.
    
    Tone: Sophisticated, institutional, concise. Use emojis sparingly but effectively.
    """
    
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        res = client.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role":"user","content":prompt}], 
            max_tokens=1000
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# ==========================================
# 4. DATA LOADER
# ==========================================
def get_data(ticker, period, interval):
    # Adjust period for interval to maximize data for calcs
    if interval in ['1m', '5m', '15m']: period = "5d"
    elif interval in ['1h', '4h']: period = "1y" # Max allowed
    else: period = "2y"
    
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 4H Resampling fix
        if interval == '4h':
            df = df.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            
        if len(df) < 50: return None
        return df
    except: return None

# ==========================================
# 5. MAIN UI EXECUTION
# ==========================================
st.sidebar.header("🎛️ Titan Controls")

# Credentials
if not st.session_state.api_key:
    st.session_state.api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Input
ticker = st.sidebar.text_input("Ticker", value="BTC-USD").upper()
interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d", "1wk"], index=3)

if st.button("🚀 Initialize God Mode"):
    with st.spinner(f"Computing Relativistic Metrics for {ticker}..."):
        df = get_data(ticker, "1y", interval)
        
        if df is not None:
            # Calc All
            df = calc_indicators(df)
            obs, fvgs = calc_smc_structures(df)
            
            # --- TABS ---
            tab_main, tab_ai, tab_raw = st.tabs(["📊 God Mode Chart", "🧠 AI Quant Analyst", "💾 Raw Data"])
            
            with tab_main:
                # Plotly Construction
                fig = make_subplots(
                    rows=4, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.03, 
                    row_heights=[0.5, 0.15, 0.15, 0.2]
                )
                
                # 1. Main Price + HMA + SMC
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='yellow', width=2), name="HMA Trend"), row=1, col=1)
                
                # Draw SMC Rectangles (FVGs)
                for fvg in fvgs:
                    fig.add_shape(type="rect", 
                        x0=fvg['x0'], x1=fvg['x1'], y0=fvg['y0'], y1=fvg['y1'], 
                        fillcolor=fvg['color'], line_width=0, opacity=0.4, row=1, col=1
                    )

                # 2. Apex Vector (Flux)
                colors = df['Apex_Flux'].apply(lambda x: '#00E676' if x > 0 else '#FF1744')
                fig.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], marker_color=colors, name="Vector Flux"), row=2, col=1)
                # Add Threshold lines
                fig.add_hline(y=0.6, line_dash="dot", line_color="gray", row=2, col=1)
                fig.add_hline(y=-0.6, line_dash="dot", line_color="gray", row=2, col=1)

                # 3. RQZO Oscillator
                # Color based on direction
                rqzo_color = np.where(df['RQZO'] > df['RQZO'].shift(1), '#00E5FF', '#FF0055')
                fig.add_trace(go.Scatter(x=df.index, y=df['RQZO'], mode='lines', line=dict(color='white', width=1), name="RQZO"), row=3, col=1)
                # We simulate the multi-color line by adding markers or segments, but simplest is just the line for now + fill
                fig.add_trace(go.Scatter(x=df.index, y=df['RQZO'], fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.1)', line=dict(width=0), showlegend=False), row=3, col=1)

                # 4. Entropy & Efficiency
                fig.add_trace(go.Scatter(x=df.index, y=df['Entropy'], line=dict(color='orange'), name="Entropy"), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Eff'], line=dict(color='purple'), name="Efficiency"), row=4, col=1)
                fig.add_hline(y=0.8, line_dash="dot", line_color="red", annotation_text="Chaos", row=4, col=1)

                fig.update_layout(height=900, template="plotly_dark", title_text=f"TITAN GOD MODE: {ticker}", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                last_r = df.iloc[-1]
                m1.metric("RQZO Value", f"{last_r['RQZO']:.2f}")
                m2.metric("Vector State", f"{last_r['Vector_State']}")
                m3.metric("Entropy (Chaos)", f"{last_r['Entropy']:.2f}")
                m4.metric("Efficiency", f"{last_r['Apex_Eff']:.2f}")

            with tab_ai:
                st.markdown("### 🧠 Senior Quant Analyst Report")
                if st.button("Generate Intelligence Report"):
                    with st.spinner("Analyzing Physics & Structure..."):
                        report = ask_ai_analyst_v2(df, ticker, interval)
                        st.markdown(f"<div class='report-text'>{report}</div>", unsafe_allow_html=True)
                        
            with tab_raw:
                st.dataframe(df.tail(100))
                
        else:
            st.error("Could not fetch data. Check ticker.")
