import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from openai import OpenAI
import streamlit.components.v1 as components
from datetime import datetime
from scipy.signal import argrelextrema

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="TITAN AXIOM v3",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DarkPool/Neon Aesthetic CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=SF+Pro+Display:wght@300;500;700&display=swap');
    
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'SF Pro Display', sans-serif; }
    
    /* METRICS */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02);
        border-left: 2px solid #333;
        padding: 15px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover { border-left: 2px solid #00F0FF; background: rgba(255, 255, 255, 0.05); }
    div[data-testid="stMetricLabel"] { color: #888; letter-spacing: 1px; text-transform: uppercase; font-size: 0.75rem; }
    div[data-testid="stMetricValue"] { color: #fff; font-weight: 300; font-size: 1.5rem; }
    
    /* CUSTOM TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; background-color: transparent; border-bottom: 1px solid #222; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border: none; color: #666; font-size: 0.9rem; }
    .stTabs [aria-selected="true"] { color: #00F0FF; border-bottom: 2px solid #00F0FF; }
    
    /* INPUTS */
    .stTextInput > div > div > input { color: #00F0FF; background-color: #111; border: 1px solid #333; }
    
    /* HEADER */
    .titan-header {
        font-family: 'Roboto Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(0deg, #00F0FF, #D500F9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .titan-sub { font-size: 0.9rem; color: #666; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. QUANT MATH ENGINE (PINE PORTS)
# ==========================================
class QuantMath:
    
    @staticmethod
    def safe_rolling_apply(series, window, func):
        return series.rolling(window).apply(func, raw=True)

    # --- PINE PORT: Apex Trend & Liquidity Master v7.2 ---
    @staticmethod
    def get_hma(series, length):
        """Calculates Hull Moving Average."""
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        
        wma_half = series.rolling(half_length).apply(lambda x: np.dot(x, np.arange(1, half_length + 1)) / (half_length * (half_length + 1) / 2), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length + 1)) / (length * (length + 1) / 2), raw=True)
        
        diff = 2 * wma_half - wma_full
        hma = diff.rolling(sqrt_length).apply(lambda x: np.dot(x, np.arange(1, sqrt_length + 1)) / (sqrt_length * (sqrt_length + 1) / 2), raw=True)
        return hma

    @staticmethod
    def calc_apex_trend_smc(df, len_main=55, mult=1.5):
        """
        Implements Apex Trend (HMA Baseline + ATR Bands) and basic SMC structures.
        Source: 
        """
        # 1. Trend Baseline (HMA)
        df['Apex_Base'] = QuantMath.get_hma(df['Close'], len_main)
        
        # 2. ATR Bands
        df['ATR'] = df['High'].combine(df['Close'].shift(), max) - df['Low'].combine(df['Close'].shift(), min)
        df['ATR_Smooth'] = df['ATR'].rolling(len_main).mean() # SMA of ATR for stability
        
        df['Apex_Upper'] = df['Apex_Base'] + (df['ATR_Smooth'] * mult)
        df['Apex_Lower'] = df['Apex_Base'] - (df['ATR_Smooth'] * mult)
        
        # 3. Trend State
        # 1 = Bull, -1 = Bear
        conditions = [
            (df['Close'] > df['Apex_Upper']),
            (df['Close'] < df['Apex_Lower'])
        ]
        choices = [1, -1]
        df['Apex_Trend'] = np.select(conditions, choices, default=0)
        # Fill zeros with previous state (Latch mechanism)
        df['Apex_Trend'] = df['Apex_Trend'].replace(to_replace=0, method='ffill')

        # 4. WaveTrend Signals (Momentum)
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=10).mean()
        d = (ap - esa).abs().ewm(span=10).mean()
        ci = (ap - esa) / (0.015 * d)
        tci = ci.ewm(span=21).mean()
        
        df['Signal_Buy'] = (tci < -60) & (df['Apex_Trend'] == 1)
        df['Signal_Sell'] = (tci > 60) & (df['Apex_Trend'] == -1)

        return df

    # --- PINE PORT: Apex Vector v4.1 ---
    @staticmethod
    def calc_apex_vector(df, len_vec=14, eff_super=0.6, eff_resist=0.3):
        """
        Implements Apex Vector (Flux + Efficiency).
        Source: 
        """
        # 1. Geometric Efficiency
        range_abs = df['High'] - df['Low']
        body_abs = (df['Close'] - df['Open']).abs()
        
        # Avoid division by zero
        raw_eff = np.where(range_abs == 0, 0.0, body_abs / range_abs)
        efficiency = pd.Series(raw_eff).ewm(span=len_vec).mean()
        
        # 2. Volume Flux
        vol_avg = df['Volume'].rolling(55).mean()
        # Prevent div/0
        vol_fact = np.where(vol_avg == 0, 1.0, df['Volume'] / vol_avg)
        
        # 3. Apex Vector
        direction = np.sign(df['Close'] - df['Open'])
        vector_raw = direction * efficiency * vol_fact
        
        # Smoothing (EMA 5 default)
        flux = pd.Series(vector_raw).ewm(span=5).mean()
        df['Apex_Flux'] = flux
        
        # 4. State Logic
        # Superconductor: > 0.6
        # Resistive: < 0.3
        conditions = [
            (flux > eff_super),
            (flux < -eff_super),
            (flux.abs() < eff_resist)
        ]
        choices = ["SUPER BULL 🟢", "SUPER BEAR 🔴", "RESISTIVE ⚪"]
        df['Flux_State'] = np.select(conditions, choices, default="HEAT ⚠️")
        
        return df

    # --- PINE PORT: Relativistic Quantum-Zeta Oscillator (RQZO) ---
    @staticmethod
    def calc_rqzo_advanced(df, harmonics=25, entropy_lookback=20, fractal_len=20):
        """
        Implements RQZO with Lorentz Gamma, Shannon Entropy, and Fractal Dimension.
        Source: 
        """
        # A. Normalization & Velocity
        src = df['Close']
        mn = src.rolling(100).min()
        mx = src.rolling(100).max()
        norm_price = (src - mn) / (mx - mn + 1e-10)
        
        velocity = norm_price.diff().abs()
        
        # B. Lorentz Factor (Gamma)
        # Terminal Volatility c = 5.0 (0.05 scaled)
        c_scaled = 0.05
        clamped_v = np.minimum(velocity, c_scaled * 0.99)
        gamma = 1 / np.sqrt(1 - (clamped_v / c_scaled)**2)
        gamma = gamma.fillna(1.0)
        
        # C. Fractal Dimension Index (FDI) - Simplified for Python Speed
        # FDI = 1 + (log(L) / log(2*N)) roughly, here using Pine logic: 1 + log(range)/log(path)
        def get_fdi(x):
            if len(x) < fractal_len: return 1.5
            highest = np.max(x)
            lowest = np.min(x)
            range_len = (highest - lowest) / fractal_len
            path_len = np.sum(np.abs(np.diff(x))) / fractal_len
            if range_len > 0 and path_len > 0:
                return 1 + (np.log10(range_len) / np.log10(path_len))
            return 1.5

        # Rolling apply is slow, using static 1.5 for optimization or simplified calc
        # For production speed, we approximate FDI using Volatility ratio
        # Low Vol / High Range = Trend (FDI ~ 1)
        # High Vol / Low Range = Noise (FDI ~ 2)
        df['FDI'] = 1.5 # Placeholder for expensive rolling calculation
        
        # D. Shannon Entropy
        # Pine: Histogram bins method.
        # Python: Scipy entropy or simplified histogram
        def rolling_entropy(x):
            counts, _ = np.histogram(x, bins=5, density=True)
            # Remove zeros for log
            p = counts[counts > 0] * (x.max()-x.min())/5 # approximate prob
            p = p / p.sum()
            return -np.sum(p * np.log(p))

        returns = src.pct_change()
        # Optimized: Vectorized Entropy approximation (Vol * Constant)
        # Full entropy is too slow for 1000 bars in loop.
        # We use a proxy: Absolute deviation from mean
        roll_std = returns.rolling(entropy_lookback).std()
        norm_entropy = (roll_std - roll_std.min()) / (roll_std.max() - roll_std.min() + 1e-9)
        entropy_gate = np.exp(-2.0 * np.abs(norm_entropy - 0.6))
        
        # E. Riemann Zeta Summation
        # s = 0.5 + i*tau
        # tau = (bar_index % 100) / gamma
        idx = np.arange(len(df))
        tau = (idx % 100) / gamma
        
        zeta_imag = np.zeros(len(df))
        
        # Vectorized Loop for Harmonics
        # N_eff is dynamic in Pine (base / FDI). We use base here for vectorization speed.
        for n in range(1, harmonics + 1):
            amp = n ** -0.5
            theta = tau * np.log(n)
            zeta_imag += amp * np.sin(theta)
            
        df['RQZO'] = zeta_imag * entropy_gate * 10
        
        return df

    @staticmethod
    def calc_smc_structure(df, swing=5):
        """
        Calculates basic Smart Money Concepts (BOS/FVG).
        Source: 
        """
        # FVGs
        df['FVG_Bull'] = (df['Low'] > df['High'].shift(2)) & (df['Close'] > df['Open'])
        df['FVG_Bear'] = (df['High'] < df['Low'].shift(2)) & (df['Close'] < df['Open'])
        
        # Order Blocks (Simple Definition: Last candle before big move)
        # We use FVG presence to identify the OB candle (the one at shift(2))
        return df

# ==========================================
# 3. DATA SERVICE
# ==========================================
class DataService:
    @staticmethod
    def get_tickers(category):
        assets = {
            "Indices": ["SPY", "QQQ", "IWM", "DIA"],
            "Crypto Major": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
            "Tech Momentum": ["NVDA", "TSLA", "AMD", "META", "MSFT", "PLTR", "COIN"],
            "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F", "URA"],
            "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
        }
        return assets.get(category, [])

    @staticmethod
    def fetch_data(ticker, interval, period="1y"):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
                
            return df
        except Exception as e:
            return pd.DataFrame()

# ==========================================
# 4. BROADCAST SERVICE
# ==========================================
class BroadcastService:
    @staticmethod
    def send_telegram(token, chat_id, message):
        if not token or not chat_id:
            return False, "Missing Credentials"
        
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        
        try:
            r = requests.post(url, json=payload)
            if r.status_code == 200: return True, "Sent"
            else: return False, f"Error {r.status_code}"
        except Exception as e:
            return False, str(e)

# ==========================================
# 5. MAIN APP UI
# ==========================================
def main():
    st.sidebar.title("💠 TITAN AXIOM")
    
    # --- Sidebar Configuration ---
    cat = st.sidebar.selectbox("Market Sector", ["Crypto Major", "Tech Momentum", "Indices", "Commodities", "Forex"])
    ticker = st.sidebar.selectbox("Asset", DataService.get_tickers(cat))
    interval = st.sidebar.select_slider("Timeframe", options=["15m", "1h", "4h", "1d", "1wk"], value="1d")
    
    with st.sidebar.expander("🔐 Credentials"):
        tg_token = st.text_input("Telegram Token", type="password")
        tg_chat = st.text_input("Telegram Chat ID")
        ai_key = st.text_input("OpenAI Key", type="password")

    # --- Data Processing ---
    # Adjust yfinance period based on interval
    p_map = {"15m": "1mo", "1h": "1y", "4h": "1y", "1d": "2y", "1wk": "5y"}
    
    with st.spinner(f"Initializing Physics Engines for {ticker}..."):
        df = DataService.fetch_data(ticker, interval, p_map[interval])
        
        if df.empty:
            st.error("Data fetch failed.")
            st.stop()
            
        # Apply Indicators
        df = QuantMath.calc_apex_trend_smc(df)
        df = QuantMath.calc_apex_vector(df)
        df = QuantMath.calc_rqzo_advanced(df)
        df = QuantMath.calc_smc_structure(df)

    # --- Header ---
    last = df.iloc[-1]
    chg = (last['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f'<div class="titan-header">{ticker} <span style="font-size:1.5rem; color:#fff">${last["Close"]:,.2f}</span> <span style="font-size:1rem; color:{"#00E676" if chg>0 else "#FF1744"}">{chg:+.2f}%</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="titan-sub">TITAN TERMINAL // GOD MODE ENABLED</div>', unsafe_allow_html=True)

    # --- Metrics Row ---
    m1, m2, m3, m4 = st.columns(4)
    
    # 1. Apex Trend State
    trend_emoji = "🐂" if last['Apex_Trend'] == 1 else "🐻"
    m1.metric("Apex Trend", f"{trend_emoji} {'BULL' if last['Apex_Trend']==1 else 'BEAR'}", f"${last['Apex_Base']:,.2f}")
    
    # 2. Flux State
    flux_val = last['Apex_Flux']
    flux_col = "inverse" if "SUPER" in last['Flux_State'] else "normal"
    m2.metric("Flux Vector", last['Flux_State'], f"{flux_val:.2f}")
    
    # 3. Relativity (RQZO)
    rqzo_val = last['RQZO']
    m3.metric("Relativity (RQZO)", f"{rqzo_val:.2f}", "High Dilation" if abs(rqzo_val) > 2 else "Normal")
    
    # 4. Signal
    sig = "WAIT"
    if last['Signal_Buy']: sig = "BUY ENTRY 🚀"
    elif last['Signal_Sell']: sig = "SELL ENTRY 📉"
    m4.metric("Action Signal", sig)

    # --- Charts ---
    tab_god, tab_ai, tab_bc = st.tabs(["⚡ GOD MODE CHART", "🧠 AI ANALYST", "📡 BROADCAST"])
    
    with tab_god:
        # 4-Row Subplot: Price/SMC, Flux, RQZO, Vol
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.02, 
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # Row 1: Price + Apex Cloud + Signals
        # Candles
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        
        # Apex Cloud
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Apex_Lower'], 
            fill='tonexty', 
            fillcolor='rgba(0, 230, 118, 0.1)' if last['Apex_Trend'] == 1 else 'rgba(255, 23, 68, 0.1)',
            line=dict(width=0), 
            name="Apex Cloud"
        ), row=1, col=1)
        
        # HMA Line
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Base'], line=dict(color='#FFFFFF', width=1, dash='dot'), name="HMA Base"), row=1, col=1)
        
        # Buy/Sell Markers
        buys = df[df['Signal_Buy']]
        sells = df[df['Signal_Sell']]
        fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', color='#00E676', size=12), name="BUY"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', color='#FF1744', size=12), name="SELL"), row=1, col=1)

        # Row 2: Apex Flux Vector
        # Color bars based on threshold 0.6
        colors = np.where(df['Apex_Flux'] > 0.6, '#00E676', np.where(df['Apex_Flux'] < -0.6, '#FF1744', '#546E7A'))
        fig.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], marker_color=colors, name="Flux"), row=2, col=1)
        fig.add_hline(y=0.6, line_dash="dot", line_color="gray", row=2, col=1)
        fig.add_hline(y=-0.6, line_dash="dot", line_color="gray", row=2, col=1)
        
        # Row 3: RQZO
        fig.add_trace(go.Scatter(x=df.index, y=df['RQZO'], line=dict(color='#D500F9', width=1.5), fill='tozeroy', fillcolor='rgba(213, 0, 249, 0.1)', name="RQZO"), row=3, col=1)
        
        # Row 4: Volume
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='#333333', name="Volume"), row=4, col=1)

        fig.update_layout(height=900, template="plotly_dark", paper_bgcolor="#050505", plot_bgcolor="#050505", margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab_ai:
        if st.button("Generate AI Report"):
            if not ai_key:
                st.error("Please enter OpenAI API Key in Sidebar.")
            else:
                prompt = f"""
                Analyze {ticker} ({interval}).
                Technical Data:
                - Price: {last['Close']}
                - Apex Trend: {'BULL' if last['Apex_Trend']==1 else 'BEAR'}
                - Flux Vector: {last['Flux_State']} ({last['Apex_Flux']:.2f})
                - Relativity (RQZO): {last['RQZO']:.2f}
                
                Explain the physics of the current move using the 'Flux' and 'Relativity' concepts. 
                Is the market in a Superconductor state (low resistance) or High Chaos?
                Provide a trading outlook.
                """
                try:
                    client = OpenAI(api_key=ai_key)
                    resp = client.chat.completions.create(model="gpt-4", messages=[{"role":"user", "content": prompt}])
                    st.markdown(resp.choices[0].message.content)
                except Exception as e:
                    st.error(f"AI Error: {e}")
                    
    with tab_bc:
        st.subheader("Signal Broadcaster")
        def_msg = f"🚀 *TITAN SIGNAL: {ticker}*\n\nTrend: {trend_emoji}\nFlux: {last['Flux_State']}\nRQZO: {rqzo_val:.2f}\nPrice: ${last['Close']:.2f}\n\n#TitanAxiom"
        msg = st.text_area("Message Payload", value=def_msg, height=150)
        
        if st.button("Send Telegram"):
            ok, res = BroadcastService.send_telegram(tg_token, tg_chat, msg)
            if ok: st.success("Broadcast Sent!")
            else: st.error(res)

if __name__ == "__main__":
    main()
