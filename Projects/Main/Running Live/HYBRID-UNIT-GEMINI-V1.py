
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import math
import json
import urllib.parse
from scipy.signal import argrelextrema
from datetime import datetime
import streamlit.components.v1 as components  # FIXED: Moved to global scope to prevent NameError

# Gemini AI Import
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# OpenAI Import (Preserved)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import yfinance as yf
except ImportError:
    st.error("yfinance is missing. Please install: pip install yfinance")
    yf = None

# ==========================================
# 1. PAGE CONFIGURATION & CSS (TITAN AESTHETIC)
# ==========================================
st.set_page_config(
    page_title="TITAN AXIOM GEMINI",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed for mobile focus
)

# Inject DarkPool/Titan CSS
st.markdown("""
<style>
    /* Main Theme */
    .stApp { background-color: #0b0c10; color: #c5c6c7; font-family: 'Roboto Mono', monospace; }
    
    /* Metrics & Cards */
    div[data-testid="metric-container"] {
        background: rgba(31, 40, 51, 0.9);
        border: 1px solid #45a29e;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
        border-color: #66fcf1;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px !important;
        color: #66fcf1 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #c5c6c7 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        border: 1px solid #45a29e;
        color: #66fcf1;
        font-weight: bold;
        height: 3em;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background: #45a29e;
        color: #0b0c10;
        box-shadow: 0 0 10px #45a29e;
    }

    /* Mobile Report Card Styling */
    .report-card {
        background-color: #1f2833;
        border-left: 5px solid #45a29e;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .report-header {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;
        border-bottom: 1px solid #45a29e;
        padding-bottom: 5px;
    }
    .report-item {
        margin-bottom: 8px;
        font-size: 14px;
        color: #c5c6c7;
    }
    .highlight { color: #66fcf1; font-weight: bold; }
    .bearish { color: #ff1744; font-weight: bold; }
    .bullish { color: #00e676; font-weight: bold; }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 4px 4px 0px 0px;
        border: 1px solid #30363d;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0b0c10;
        color: #66fcf1;
        border-bottom: 2px solid #66fcf1;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MATH KERNEL (QUANT CORE) - FULL PINE TRANSLATION
# ==========================================
class QuantCore:
    """
    Exact translation of provided Pine Scripts:
    1. CHEDO v2
    2. Omega (Riemann-Quantum-Lyapunov)
    3. Apex Vector (Flux)
    4. SMC (Smart Money Concepts)
    5. Apex Trend Master
    """

    @staticmethod
    def tanh_clamp(x):
        """Custom hyperbolic tangent with clamping from CHEDO v2"""
        x_clamped = np.clip(x, -20.0, 20.0)
        e2x = np.exp(2.0 * x_clamped)
        return (e2x - 1.0) / (e2x + 1.0)

    @staticmethod
    def calc_hma(series, length):
        """Hull Moving Average"""
        if len(series) < length: return series
        half_len = int(length / 2)
        sqrt_len = int(np.sqrt(length))
        
        def wma(s, l):
            weights = np.arange(1, l + 1)
            return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

        wma_f = wma(series, length)
        wma_h = wma(series, half_len)
        diff = 2 * wma_h - wma_f
        return wma(diff, sqrt_len)

    @staticmethod
    def calc_atr(df, length=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(length).mean()

    @staticmethod
    def calc_chedo(df, length=50, smooth_len=3, bins=10):
        """Chaotic Hyperbolic Entropy Divergence Oscillator (CHEDO) v2"""
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        ret = df['log_ret']

        # 1. Hyperbolic Geodesic Curvature Proxy
        mu_r = ret.rolling(length).mean()
        sigma_r = ret.rolling(length).std()
        v = sigma_r / (np.abs(mu_r) + 1e-8)
        abs_ret_v = np.abs(ret) * v
        hyper_dist = np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))
        kappa_h = QuantCore.tanh_clamp(hyper_dist.rolling(length).mean())

        # 2. Local Lyapunov Exponent
        diff_ret = ret.diff()
        lyap_term = np.log(np.abs(diff_ret) + 1e-6)
        lam = lyap_term.rolling(length).mean()
        lambda_norm = (lam - (-5)) / (2 - (-5))

        # 3. KS-like Regime Shift Proxy
        short_vol = ret.rolling(int(length/2)).std()
        long_vol = ret.rolling(length).std()
        vol_ratio = short_vol / (long_vol + 1e-8)
        
        short_skew = ((ret - mu_r)**3).rolling(int(length/2)).mean() / (short_vol**3 + 1e-8)
        long_skew = ((ret - mu_r)**3).rolling(length).mean() / (long_vol**3 + 1e-8)
        skew_diff = np.abs(short_skew - long_skew)
        d_ks_proxy = np.minimum(vol_ratio + skew_diff, 2.0)
        d_ks_norm = (d_ks_proxy - 1.0) * -1 

        # 4. Shannon Entropy
        def rolling_shannon(x):
            counts, _ = np.histogram(x, bins=bins)
            probs = counts / np.sum(counts)
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            max_ent = np.log(bins)
            return (max_ent - entropy) / max_ent if max_ent != 0 else 0

        entropy_exhaust = ret.rolling(length).apply(rolling_shannon, raw=True)

        # 5. Edge Preserving Denoise
        alternating = ret - ret.shift(2)
        f_den = np.abs(alternating.rolling(length).mean()) / (np.abs(ret).rolling(length).mean() + 1e-8)

        # Fusion
        w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25
        weighted = w1 * (kappa_h * lambda_norm) + w2 * d_ks_norm + w3 * entropy_exhaust + w4 * f_den
        
        # Bound and Smooth
        raw_chedo = 2 / (1 + np.exp(-weighted * 4)) - 1
        df['CHEDO'] = raw_chedo.rolling(smooth_len).mean()
        return df

    @staticmethod
    def calc_omega(df, length=252, chaos_len=55):
        """Ω – Riemann–Quantum–Lyapunov–Reynolds Fusion Oscillator"""
        atr = QuantCore.calc_atr(df, 14)
        raw_change = df['Close'].diff()
        
        # 1. Reynolds Number
        rho = df['Volume']
        u = np.abs(raw_change)
        L = atr
        mu = df['Close'].rolling(length).std()
        Re = (rho * u * L) / (np.maximum(mu, 1e-10))
        Re_Star = Re.rolling(length).mean()
        
        # 2. Quantum Tunnel Probability
        v0 = mu 
        e = np.abs(df['Close'] - df['Close'].shift(length))
        kappa = np.sqrt(np.maximum(0, v0 - e)) / atr
        psi = 1 / (np.cosh(kappa)**2)
        
        # 3. Lyapunov Exponent
        df_diff = df['Close'].diff()
        lyap = np.log(np.abs(df_diff).rolling(chaos_len).mean() + 1e-10)
        
        # 4. Entropy Proxy
        ent_proxy = (df['Close'].rolling(length).std() / df['Close'].rolling(length).mean())
        ent_norm = np.clip(ent_proxy * 10, 0, 1)
        
        # 5. Complex Omega
        mod = psi * QuantCore.tanh_clamp(Re / (Re_Star + 1e-10)) * np.maximum(0, 1 - ent_norm)
        omega_im = mod * np.sin(np.pi * lyap)
        omega_mag = mod
        
        df['Omega_Im'] = omega_im
        df['Omega_Mag'] = omega_mag
        df['Reynolds'] = Re
        return df

    @staticmethod
    def calc_apex_vector(df, len_vec=14, vol_norm=55, sm_len=5):
        """Apex Vector [Flux + Efficiency] v4.1"""
        # 1. Efficiency
        range_abs = df['High'] - df['Low']
        body_abs = np.abs(df['Close'] - df['Open'])
        raw_eff = np.where(range_abs == 0, 0.0, body_abs / range_abs)
        efficiency = pd.Series(raw_eff).ewm(span=len_vec).mean()
        
        # 2. Volume Flux
        vol_avg = df['Volume'].rolling(vol_norm).mean()
        vol_fact = np.where(vol_avg == 0, 1.0, df['Volume'] / vol_avg)
        
        # 3. Apex Vector
        direction = np.sign(df['Close'] - df['Open'])
        vector_raw = direction * efficiency * vol_fact
        
        # 4. Smoothing
        df['Apex_Flux'] = vector_raw.ewm(span=sm_len).mean()
        
        # State Logic
        eff_super = 0.60
        df['AV_Super_Bull'] = df['Apex_Flux'] > eff_super
        df['AV_Super_Bear'] = df['Apex_Flux'] < -eff_super
        
        return df

    @staticmethod
    def calc_trend_master(df, len_main=55, mult=1.5):
        """Apex Trend & Liquidity Master v7.2"""
        df['TM_HMA'] = QuantCore.calc_hma(df['Close'], len_main)
        df['TM_ATR'] = QuantCore.calc_atr(df, len_main)
        df['TM_Upper'] = df['TM_HMA'] + (df['TM_ATR'] * mult)
        df['TM_Lower'] = df['TM_HMA'] - (df['TM_ATR'] * mult)
        
        conditions = [(df['Close'] > df['TM_Upper']), (df['Close'] < df['TM_Lower'])]
        choices = [1, -1]
        df['TM_Trend'] = np.select(conditions, choices, default=0)
        df['TM_Trend'] = df['TM_Trend'].replace(0, method='ffill')
        
        df['TM_Signal_Buy'] = (df['TM_Trend'] == 1) & (df['TM_Trend'].shift(1) != 1)
        df['TM_Signal_Sell'] = (df['TM_Trend'] == -1) & (df['TM_Trend'].shift(1) != -1)
        
        return df

    @staticmethod
    def run_full_pipeline(df):
        if df is None or df.empty: return None
        df = QuantCore.calc_hma(df['Close'], 55).to_frame(name='HMA_Check').join(df)
        df = QuantCore.calc_chedo(df)
        df = QuantCore.calc_omega(df)
        df = QuantCore.calc_apex_vector(df)
        df = QuantCore.calc_trend_master(df)
        
        # Extra: Fear & Greed Proxy
        df['FG_Index'] = 50 + (df['Apex_Flux']*25) + (df['CHEDO']*25)
        df['FG_Index'] = df['FG_Index'].clip(0, 100)
        return df

# ==========================================
# 3. DATA & UTILS
# ==========================================
class DataManager:
    @staticmethod
    def fetch_data(ticker, timeframe, limit=500):
        tf_map = {
            "15m": ("60d", "15m"),
            "1h": ("730d", "1h"),
            "4h": ("730d", "1h"),
            "1d": ("5y", "1d"),
            "1wk": ("10y", "1wk")
        }
        period, interval = tf_map.get(timeframe, ("1y", "1d"))
        if ticker == "BTC": ticker = "BTC-USD"
        
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty: return None
            
            if timeframe == "4h":
                df = df.resample("4h").agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }).dropna()
            return df
        except Exception as e:
            return None

    @staticmethod
    def backtest_live(df, tp_r=1.5):
        trades = []
        signals = df[df['TM_Signal_Buy'] | df['TM_Signal_Sell']]
        for idx, row in signals.iterrows():
            entry = row['Close']
            is_long = row['TM_Signal_Buy']
            atr = row['TM_ATR']
            stop = entry - (atr * 2) if is_long else entry + (atr * 2)
            tp = entry + (atr * 2 * tp_r) if is_long else entry - (atr * 2 * tp_r)
            
            future = df.loc[idx:].iloc[1:21]
            if future.empty: continue
            
            outcome = "PENDING"
            if is_long:
                if future['High'].max() >= tp: outcome = "WIN"
                elif future['Low'].min() <= stop: outcome = "LOSS"
            else:
                if future['Low'].min() <= tp: outcome = "WIN"
                elif future['High'].max() >= stop: outcome = "LOSS"
            if outcome != "PENDING": trades.append(outcome)
                
        win_rate = (trades.count("WIN") / len(trades) * 100) if trades else 0
        return len(trades), win_rate

# ==========================================
# 4. INTELLIGENCE (Broadcast + Gemini/OpenAI)
# ==========================================
class Intelligence:
    @staticmethod
    def broadcast_telegram(message, token, chat_id):
        if not token or not chat_id: return False, "No Credentials"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        max_len = 3000
        try:
            if len(message) <= max_len:
                requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
            else:
                chunks = [message[i:i+max_len] for i in range(0, len(message), max_len)]
                for i, chunk in enumerate(chunks):
                    header = f"(Part {i+1}/{len(chunks)}) "
                    requests.post(url, json={"chat_id": chat_id, "text": header + chunk, "parse_mode": "Markdown"})
            return True, "Broadcast Sent"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def generate_signal_report(df, ticker):
        last = df.iloc[-1]
        return f"""
        🚨 TITAN INTELLIGENCE REPORT 🚨
        Asset: {ticker} | Price: {last['Close']:.2f}
        
        [PHYSICS ENGINE]
        ⚛️ Omega State: {last['Omega_Mag']:.3f}
        🌪️ Reynolds: {last['Reynolds']:.1f}
        💠 Apex Flux: {last['Apex_Flux']:.2f} ({'SuperCond' if last['AV_Super_Bull'] else 'Resistive'})
        
        [STRUCTURE]
        🏛️ Trend Master: {'BULL 🐂' if last['TM_Trend']==1 else 'BEAR 🐻'}
        🌊 CHEDO Entropy: {last['CHEDO']:.2f}
        """

    @staticmethod
    def ask_ai(gemini_key, openai_key, context):
        sys_prompt = "You are the Titan Quant AI. Analyze the technical data provided. Be concise, institutional, and use emojis."
        full_prompt = f"{sys_prompt}\n\nContext: {context}"

        # 1. Try Gemini First (Default)
        if gemini_key and genai:
            try:
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(full_prompt)
                return f"✨ GEMINI: {response.text}"
            except Exception as e:
                return f"Gemini Error: {e}"
        
        # 2. Fallback to OpenAI
        elif openai_key and OpenAI:
            try:
                client = OpenAI(api_key=openai_key)
                res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role":"user", "content": full_prompt}]
                )
                return f"🤖 GPT-4: {res.choices[0].message.content}"
            except Exception as e:
                return f"OpenAI Error: {e}"
        
        else:
            return "⚠️ No Valid AI API Key Found (Gemini or OpenAI)."

# ==========================================
# 5. UI COMPONENTS
# ==========================================
def render_ticker_tape(symbol):
    """TradingView Tape Integration"""
    sym = f"BINANCE:{symbol}USDT" if "BTC" in symbol or "ETH" in symbol else symbol
    components.html(f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
      {{
      "symbols": [{{"proName": "{sym}", "title": "{symbol}"}}],
      "showSymbolLogo": true, "colorTheme": "dark", "isTransparent": true, "displayMode": "adaptive", "locale": "en"
      }}
      </script>
    </div>
    """, height=50)

def render_mobile_report_card(df, ticker):
    row = df.iloc[-1]
    is_bull = row['TM_Trend'] == 1
    direction = "LONG 🐂" if is_bull else "SHORT 🐻"
    confidence = "HIGH" if abs(row['Apex_Flux']) > 0.6 else "LOW"
    
    html = f"""
    <div class="report-card">
        <div class="report-header">💠 SIGNAL: {direction}</div>
        <div class="report-item">Confidence: <span class="highlight">{confidence}</span></div>
        <div class="report-item">Sentiment (F&G): <span class="highlight">{row['FG_Index']:.0f}/100</span></div>
        <div class="report-item">Flux Vector: <span class="highlight">{row['Apex_Flux']:.2f}</span></div>
    </div>
    <div class="report-card">
        <div class="report-header">🌊 PHYSICS ENGINE</div>
        <div class="report-item">Omega (Quant): <span class="highlight">{row['Omega_Mag']:.3f}</span></div>
        <div class="report-item">Reynolds (Turb): <span class="highlight">{row['Reynolds']:.1f}</span></div>
        <div class="report-item">Entropy: <span class="highlight">{row['CHEDO']:.2f}</span></div>
    </div>
    <div class="report-card">
        <div class="report-header">🎯 EXECUTION</div>
        <div class="report-item">Price: <span class="highlight">{row['Close']:.4f}</span></div>
        <div class="report-item">Stop Ref: <span class="bearish">{row['TM_Lower']:.4f}</span></div>
        <div class="report-item">TP Target: <span class="bullish">{(row['Close'] + (row['TM_ATR']*3)):.4f}</span></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_god_mode_chart(df, ticker):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.15, 0.2], vertical_spacing=0.02)
    
    # 1. Price & Cloud
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['TM_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.1)', line=dict(width=0), name="Cloud"), row=1, col=1)
    
    # 2. Apex Flux
    colors = ['#00E676' if v > 0.6 else '#FF1744' if v < -0.6 else '#546E7A' for v in df['Apex_Flux']]
    fig.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], marker_color=colors, name="Flux"), row=2, col=1)
    
    # 3. Omega (Im)
    fig.add_trace(go.Scatter(x=df.index, y=df['Omega_Im'], line=dict(color='orange'), name="Omega Im"), row=3, col=1)
    
    # 4. CHEDO
    fig.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], line=dict(color='#2962FF'), name="CHEDO"), row=4, col=1)
    fig.add_hline(y=0.7, line_dash="dot", row=4, col=1)
    fig.add_hline(y=-0.7, line_dash="dot", row=4, col=1)
    
    fig.update_layout(height=900, template="plotly_dark", title=f"GOD MODE: {ticker}", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. MAIN APPLICATION
# ==========================================
def main():
    # --- Sidebar ---
    st.sidebar.markdown("## ⚙️ TITAN CONTROL")
    
    # Ticker Input
    ticker_input = st.sidebar.text_input("Ticker", value="BTC-USD")
    ticker = ticker_input.upper()
    
    # Controls
    timeframe = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d", "1wk"], index=3)
    
    # DEFAULT MOBILE MODE SET TO TRUE
    is_mobile = st.sidebar.checkbox("📱 Mobile Mode", value=True)
    
    # Secrets Loading (Gemini Default + OpenAI Backup)
    gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    openai_key = st.secrets.get("OPENAI_API_KEY", "")
    
    # Allow manual override
    if not gemini_key:
        gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    if not openai_key:
        openai_key = st.sidebar.text_input("OpenAI API Key (Backup)", type="password")
        
    tg_token = st.sidebar.text_input("Telegram Token", type="password", value=st.secrets.get("TELEGRAM_TOKEN", ""))
    tg_chat = st.sidebar.text_input("Telegram Chat ID", value=st.secrets.get("TELEGRAM_CHAT_ID", ""))

    if st.sidebar.button("🔄 REFRESH SYSTEM"):
        st.rerun()

    # --- Header ---
    render_ticker_tape(ticker)
    
    # --- Data Engine ---
    with st.spinner("Initializing Quantum Engines..."):
        df = DataManager.fetch_data(ticker, timeframe)
        df = QuantCore.run_full_pipeline(df)
        
    if df is not None:
        # --- Layout Switching ---
        if is_mobile:
            render_mobile_report_card(df, ticker)
            st.markdown("### 📉 Price Action")
            fig = go.Figure(go.Candlestick(x=df.index, open=df['Open'], close=df['Close'], high=df['High'], low=df['Low']))
            fig.update_layout(height=350, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            render_god_mode_chart(df, ticker)
            trades, wr = DataManager.backtest_live(df)
            c1, c2, c3 = st.columns(3)
            c1.metric("Live Win Rate", f"{wr:.1f}%")
            c2.metric("Signals", trades)
            c3.metric("Entropy", f"{df.iloc[-1]['CHEDO']:.2f}")

        # --- AI Analyst ---
        st.divider()
        st.subheader("🧠 Titan AI Agent")
        
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Titan Quantum Link Established."}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Ask Titan about the market structure..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            # Context
            report = Intelligence.generate_signal_report(df, ticker)
            response = Intelligence.ask_ai(gemini_key, openai_key, f"Report: {report}\nUser: {prompt}")
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

        # --- Broadcast Center ---
        with st.expander("📡 COMMAND CENTER"):
            msg_edit = st.text_area("Broadcast Payload", value=f"SIGNAL: {ticker} | Price: {df.iloc[-1]['Close']}")
            if st.button("🚀 EXECUTE BROADCAST"):
                succ, res = Intelligence.broadcast_telegram(msg_edit, tg_token, tg_chat)
                if succ: st.success(res)
                else: st.error(res)

    else:
        st.error("No Data Available. Check Ticker or API Connection.")

if __name__ == "__main__":
    main()
