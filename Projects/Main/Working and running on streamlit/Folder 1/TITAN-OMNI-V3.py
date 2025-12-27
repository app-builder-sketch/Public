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
import urllib.parse
from scipy.signal import argrelextrema
from scipy.stats import linregress
from datetime import datetime
import streamlit.components.v1 as components

# ===============================
# 1. PAGE CONFIG & DARKPOOL CSS
# ===============================
st.set_page_config(
    page_title="TITAN OMNI V3 [GOD MODE]",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="auto" # Default for mobile optimization
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
.title-glow {
    font-size: 3em; font-weight: bold; color: #ffffff;
    text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 40px #00ff00;
    margin-bottom: 20px;
}
div[data-testid="stMetric"] {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 10px; border-radius: 8px; transition: transform 0.2s;
}
div[data-testid="stMetric"]:hover { transform: scale(1.02); border-color: #00ff00; }
div[data-testid="stMetricValue"] { font-size: 1.2rem !important; font-weight: 700; }
.stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: transparent; }
.stTabs [data-baseweb="tab"] {
    height: 50px; white-space: pre-wrap; background-color: #161b22;
    border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
    border: 1px solid #30363d; color: #8b949e;
}
.stTabs [aria-selected="true"] { background-color: #0e1117; color: #00ff00; border-bottom: 2px solid #00ff00; }
</style>
""", unsafe_allow_html=True)

# ===============================
# 2. ENVIRONMENT & SECRETS
# ===============================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def load_key(key_name):
    if key_name in st.secrets: return st.secrets[key_name]
    return os.getenv(key_name, "")

# ===============================
# 3. SESSION STATE
# ===============================
if "run_analysis" not in st.session_state: st.session_state.run_analysis = False
if "ai_report" not in st.session_state: st.session_state.ai_report = ""

# ===============================
# 4. LIBRARY CHECKS
# ===============================
try:
    import yfinance as yf
except ImportError:
    st.error("CRITICAL: Install yfinance -> pip install yfinance")
    st.stop()

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ===============================
# 5. TELEGRAM ENGINE
# ===============================
class TelegramEngine:
    API_URL = "https://api.telegram.org/bot{}/sendMessage"
    PHOTO_URL = "https://api.telegram.org/bot{}/sendPhoto"

    @staticmethod
    def send(token: str, chat_id: str, message: str, photo_data=None):
        if not token or not chat_id: return False, "Missing Credentials"
        try:
            if photo_data:
                files = {'photo': photo_data}
                data_photo = {'chat_id': chat_id, 'caption': "🔥 TITAN GOD MODE CHART", 'parse_mode': 'Markdown'}
                requests.post(TelegramEngine.PHOTO_URL.format(token), data=data_photo, files=files)
            
            # Chunking for long messages
            max_len = 3000
            for i in range(0, len(message), max_len):
                payload = {"chat_id": chat_id, "text": message[i:i+max_len], "disable_web_page_preview": True}
                requests.post(TelegramEngine.API_URL.format(token), json=payload, timeout=10)
            return True, "Sent"
        except Exception as e:
            return False, str(e)

# ===============================
# 6. DATA ENGINE
# ===============================
class DataEngine:
    @staticmethod
    def fetch(ticker, timeframe):
        tf_map = {
            "15m": ("60d", "15m"), "1h": ("730d", "1h"), "4h": ("730d", "1h"),
            "1d": ("5y", "1d"), "1wk": ("10y", "1wk")
        }
        period, interval = tf_map.get(timeframe, ("2y", "1d"))
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        
        # Proper 4H Resampling
        if timeframe == "4h":
            df = df.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()
        return df

    @staticmethod
    def get_fundamentals(ticker):
        if any(c in ticker for c in ["-", "=", "^"]): return None
        try:
            info = yf.Ticker(ticker).info
            return {
                "Market Cap": info.get("marketCap", 0), "P/E": info.get("trailingPE", 0),
                "Growth": info.get("revenueGrowth", 0), "Debt/Eq": info.get("debtToEquity", 0),
                "Summary": info.get("longBusinessSummary", "N/A")
            }
        except: return None

    @staticmethod
    def get_global_performance():
        assets = {"Tech": "XLK", "Energy": "XLE", "Financials": "XLF", "Crypto": "BTC-USD", "Gold": "GLD"}
        try:
            data = yf.download(list(assets.values()), period="5d", progress=False)['Close']
            res = ((data.iloc[-1] - data.iloc[-2]) / data.iloc[-2]) * 100
            res.index = [k for k,v in assets.items() if v in res.index]
            return res.sort_values()
        except: return None

# ===============================
# 7. QUANTUM CORE (ADVANCED MATH)
# ===============================
class QuantumCore:
    # --- HELPER FUNCTIONS ---
    @staticmethod
    def calculate_wma(series, length):
        return series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length + 1)) / (length * (length + 1) / 2), raw=True)

    @staticmethod
    def calculate_hma(series, length):
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wma_half = QuantumCore.calculate_wma(series, half_length)
        wma_full = QuantumCore.calculate_wma(series, length)
        return QuantumCore.calculate_wma(2 * wma_half - wma_full, sqrt_length)

    @staticmethod
    def calculate_atr(df, length=14):
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        return tr.rolling(length).mean()

    @staticmethod
    def calculate_supertrend(df, period=10, multiplier=3):
        atr = QuantumCore.calculate_atr(df, period)
        hl2 = (df['High'] + df['Low']) / 2
        final_upper = hl2 + (multiplier * atr)
        final_lower = hl2 - (multiplier * atr)
        
        close = df['Close'].values
        upper = final_upper.values
        lower = final_lower.values
        st_val = np.zeros(len(df)); trend = np.zeros(len(df))
        
        st_val[0] = lower[0]; trend[0] = 1
        
        for i in range(1, len(df)):
            if trend[i-1] == 1:
                st_val[i] = max(lower[i], st_val[i-1]) if close[i] > st_val[i-1] else upper[i]
                trend[i] = 1 if close[i] > st_val[i-1] else -1
                if trend[i] == -1: st_val[i] = upper[i]
            else:
                st_val[i] = min(upper[i], st_val[i-1]) if close[i] < st_val[i-1] else lower[i]
                trend[i] = -1 if close[i] < st_val[i-1] else 1
                if trend[i] == 1: st_val[i] = lower[i]
        
        return pd.Series(trend, index=df.index)

    # --- MAIN PIPELINE (GOD MODE LOGIC) ---
    @staticmethod
    def pipeline(df):
        # 1. Base Calcs
        df['HMA'] = QuantumCore.calculate_hma(df['Close'], 55)
        df['ATR'] = QuantumCore.calculate_atr(df, 14)
        
        # 2. Apex Trend (HMA + ATR Bands)
        apex_mult = 1.5
        df['Apex_ATR'] = QuantumCore.calculate_atr(df, 55)
        df['Apex_Upper'] = df['HMA'] + (df['Apex_ATR'] * apex_mult)
        df['Apex_Lower'] = df['HMA'] - (df['Apex_ATR'] * apex_mult)
        df['Apex_Trend'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, 0))
        df['Apex_Trend'] = df['Apex_Trend'].replace(to_replace=0, method='ffill')

        # 3. DarkPool Squeeze
        df['Sqz_Basis'] = df['Close'].rolling(20).mean()
        df['Sqz_Dev'] = df['Close'].rolling(20).std() * 2.0
        df['Sqz_Upper_BB'] = df['Sqz_Basis'] + df['Sqz_Dev']
        df['Sqz_Lower_BB'] = df['Sqz_Basis'] - df['Sqz_Dev']
        df['Sqz_Range_MA'] = QuantumCore.calculate_atr(df, 20)
        df['Sqz_Upper_KC'] = df['Sqz_Basis'] + (df['Sqz_Range_MA'] * 1.5)
        df['Sqz_Lower_KC'] = df['Sqz_Basis'] - (df['Sqz_Range_MA'] * 1.5)
        df['Squeeze_On'] = (df['Sqz_Lower_BB'] > df['Sqz_Lower_KC']) & (df['Sqz_Upper_BB'] < df['Sqz_Upper_KC'])
        
        # Momentum (LinReg Proxy)
        avg_val = (df['High'].rolling(20).max() + df['Low'].rolling(20).min() + df['Sqz_Basis']) / 3
        df['Sqz_Mom'] = (df['Close'] - avg_val).rolling(20).mean() * 100

        # 4. Money Flow Matrix
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        mf_vol = df['Volume'] / df['Volume'].rolling(14).mean()
        df['MF_Matrix'] = ((df['RSI'] - 50) * mf_vol).ewm(span=3).mean()

        # 5. Dark Vector (SuperTrend)
        df['DarkVector_Trend'] = QuantumCore.calculate_supertrend(df, 10, 4.0)

        # 6. Gann Trend
        gann_len = 3
        df['Gann_High'] = df['High'].rolling(gann_len).mean()
        df['Gann_Low'] = df['Low'].rolling(gann_len).mean()
        df['Gann_Trend'] = np.where(df['Close'] > df['Gann_High'].shift(1), 1, np.where(df['Close'] < df['Gann_Low'].shift(1), -1, 0))
        df['Gann_Trend'] = df['Gann_Trend'].replace(to_replace=0, method='ffill')

        # 7. EVWM & RVOL
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
        ev_elast = (df['Close'] - QuantumCore.calculate_hma(df['Close'], 21)) / QuantumCore.calculate_atr(df, 21)
        ev_force = np.sqrt(df['RVOL'].ewm(span=5).mean())
        df['EVWM'] = ev_elast * ev_force

        # 8. GOD MODE SCORE (Aggregated)
        df['GM_Score'] = (
            df['Apex_Trend'] + 
            df['Gann_Trend'] + 
            df['DarkVector_Trend'] + 
            np.sign(df['Sqz_Mom'])
        )

        return df.dropna()

    @staticmethod
    def calc_fear_greed_v4(df):
        # 1. RSI
        delta = df['Close'].diff()
        rs = (delta.where(delta > 0, 0).ewm(alpha=1/14).mean()) / (-delta.where(delta < 0, 0).ewm(alpha=1/14).mean())
        fg_rsi = 100 - (100 / (1 + rs))
        
        # 2. MACD
        macd = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        fg_macd = (50 + ( (macd - macd.ewm(span=9).mean()) * 10)).clip(0, 100)
        
        # 3. BB
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        fg_bb = ((df['Close'] - (sma20 - std20*2)) / (std20*4) * 100).clip(0, 100)
        
        # Composite
        df['FG_Index'] = (fg_rsi * 0.3) + (fg_macd * 0.25) + (fg_bb * 0.25) + 20 # Simple base padding
        df['FG_Index'] = df['FG_Index'].rolling(5).mean()
        
        return df

    @staticmethod
    def get_sr_channels(df, pivot=10, loopback=290):
        if len(df) < loopback: loopback = len(df)
        window = df.iloc[-loopback:].copy()
        
        window['Is_Pivot_H'] = window['High'] == window['High'].rolling(pivot*2+1, center=True).max()
        window['Is_Pivot_L'] = window['Low'] == window['Low'].rolling(pivot*2+1, center=True).min()
        
        pivots = window[window['Is_Pivot_H']]['High'].tolist() + window[window['Is_Pivot_L']]['Low'].tolist()
        pivots.sort()
        
        if not pivots: return []
        
        max_width = (window['High'].max() - window['Low'].min()) * 0.05
        zones = []
        
        for i in range(len(pivots)):
            seed = pivots[i]
            c_min, c_max = seed, seed
            count = 1
            for j in range(i+1, len(pivots)):
                if pivots[j] - seed <= max_width:
                    c_max = pivots[j]; count += 1
                else: break
            
            touches = ((window['High'] >= c_min) & (window['Low'] <= c_max)).sum()
            score = (count * 20) + touches
            zones.append({'min': c_min, 'max': c_max, 'score': score})
            
        zones.sort(key=lambda x: x['score'], reverse=True)
        final = []
        for z in zones:
            if not any((z['min'] < ex['max'] and z['max'] > ex['min']) for ex in final):
                final.append(z)
                if len(final) >= 6: break
        return final

    @staticmethod
    def calculate_smc(df):
        fvgs = []
        for i in range(2, len(df)):
            if df['Low'].iloc[i] > df['High'].iloc[i-2]: 
                fvgs.append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i], 'color': 'rgba(0, 255, 104, 0.3)'})
            if df['High'].iloc[i] < df['Low'].iloc[i-2]: 
                fvgs.append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['Low'].iloc[i-2], 'y1': df['High'].iloc[i], 'color': 'rgba(255, 0, 8, 0.3)'})
        return {'fvgs': fvgs}

    @staticmethod
    def calc_volume_profile(df, bins=50):
        mid = (df['Close'] + df['Open']) / 2
        price_bins = np.linspace(df['Low'].min(), df['High'].max(), bins)
        vp = np.zeros(bins)
        bin_idx = np.digitize(mid, price_bins)
        for i in range(len(mid)):
            if 0 <= bin_idx[i]-1 < bins: vp[bin_idx[i]-1] += df['Volume'].iloc[i]
        poc_price = price_bins[np.argmax(vp)]
        return price_bins, vp, poc_price

    @staticmethod
    def run_monte_carlo(df, days=30, sims=1000):
        returns = df['Close'].pct_change().dropna()
        paths = np.zeros((days, sims)); paths[0] = df['Close'].iloc[-1]
        for t in range(1, days):
            paths[t] = paths[t-1] * (1 + np.random.normal(returns.mean(), returns.std(), sims))
        return paths

    @staticmethod
    def calc_correlations(ticker):
        others = {"SPY": "SPY", "BTC": "BTC-USD", "10Y": "^TNX", "DXY": "DX-Y.NYB", "Gold": "GC=F"}
        try:
            df = yf.download([ticker] + list(others.values()), period="180d", progress=False)['Close']
            return df.corr()[ticker].drop(ticker).sort_values(ascending=False)
        except: return None

    @staticmethod
    def get_seasonality(ticker):
        try:
            df = yf.download(ticker, period="20y", interval="1mo", progress=False)
            df['Ret'] = df['Close'].pct_change() * 100
            df['Month'] = df.index.month
            return df.groupby('Month')['Ret'].agg(['mean', lambda x: (x>0).mean()*100]).rename(columns={'<lambda_0>': 'Win Rate'})
        except: return None

    @staticmethod
    def calc_day_dna(ticker):
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            df['Ret'] = df['Close'].pct_change() * 100
            df['Day'] = df.index.day_name()
            return df.groupby('Day')['Ret'].agg(['mean', lambda x: (x>0).mean()*100])
        except: return None

# ===============================
# 8. AI ENGINE
# ===============================
class AIEngine:
    @staticmethod
    def generate_report(df, ticker, keys, timeframe):
        last = df.iloc[-1]
        gm_verdict = "STRONG BUY" if last['GM_Score'] >= 3 else "STRONG SELL" if last['GM_Score'] <= -3 else "NEUTRAL"
        
        prompt = f"""
        ACT AS A SENIOR DARKPOOL TRADER. Analyze {ticker} ({timeframe}).
        
        --- DATA FEED ---
        Price: {last['Close']:.2f}
        Titan Score: {last['GM_Score']:.1f}/5 ({gm_verdict})
        Apex Trend: {'BULL' if last['Apex_Trend']==1 else 'BEAR'}
        Dark Vector: {'BULL' if last['DarkVector_Trend']==1 else 'BEAR'}
        Squeeze: {'FIRING' if last['Squeeze_On'] else 'OFF'}
        RVOL: {last['RVOL']:.1f}x | Money Flow: {last['MF_Matrix']:.2f}
        Fear/Greed: {last['FG_Index']:.1f}
        
        --- MISSION ---
        1. Structure: Trend vs Chop?
        2. Flow: Institutional footprint?
        3. Outlook: 30-word verdict.
        Use emojis.
        """
        
        if keys.get("gemini") and genai:
            try:
                genai.configure(api_key=keys["gemini"])
                return f"💎 GEMINI:\n{genai.GenerativeModel('gemini-pro').generate_content(prompt).text}"
            except Exception as e: return f"Gemini Error: {e}"

        if keys.get("openai") and OpenAI:
            try:
                client = OpenAI(api_key=keys["openai"])
                return f"🤖 OPENAI:\n{client.chat.completions.create(model='gpt-4', messages=[{'role':'user','content':prompt}]).choices[0].message.content}"
            except Exception as e: return f"OpenAI Error: {e}"
        
        return "⚠️ AI Unavailable"

# ===============================
# 9. UI & MAIN LOGIC
# ===============================
def main():
    st.markdown('<div class="title-glow">👁️ TITAN OMNI V3 [GOD MODE]</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🎛️ Command Center")
        input_mode = st.radio("Mode", ["Curated", "Manual"], index=1)
        if input_mode == "Curated":
            ticker = st.selectbox("Ticker", ["BTC-USD", "ETH-USD", "SPY", "QQQ", "NVDA", "TSLA"])
        else:
            ticker = st.text_input("Ticker", "BTC-USD").upper()
            
        timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d", "1wk"], index=3)
        
        st.markdown("### 🔐 Keys")
        gem_key = st.text_input("Gemini", value=load_key("GEMINI_API_KEY"), type="password")
        oai_key = st.text_input("OpenAI", value=load_key("OPENAI_API_KEY"), type="password")
        tg_token = st.text_input("TG Token", value=load_key("TELEGRAM_TOKEN"), type="password")
        tg_chat = st.text_input("TG Chat ID", value=load_key("TELEGRAM_CHAT_ID"))

        if st.button("🚀 INITIATE SCAN"):
            st.session_state.run_analysis = True
            st.session_state.ai_report = ""

    if not st.session_state.run_analysis:
        st.info("System Ready. Awaiting Target...")
        return

    # --- EXECUTION ---
    df = DataEngine.fetch(ticker, timeframe)
    if df is None: st.error("Target Acquisition Failed."); return
    
    # Pipeline
    df = QuantumCore.pipeline(df)
    df = QuantumCore.calc_fear_greed_v4(df)
    last = df.iloc[-1]
    sr_levels = QuantumCore.get_sr_channels(df)
    
    # --- METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{last['Close']:.2f}")
    c2.metric("Apex Trend", "BULL" if last['Apex_Trend']==1 else "BEAR", delta_color="normal" if last['Apex_Trend']==1 else "inverse")
    c3.metric("Titan Score", f"{last['GM_Score']:.0f}/5")
    c4.metric("Fear/Greed", f"{last['FG_Index']:.0f}")

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 God Mode Chart", "🤖 AI Intel", "🌍 Macro/Fund", 
        "🔮 Quant/SMC", "📅 Seasonality", "🧬 DNA", "📡 Broadcast"
    ])

    with tab1:
        # COMPLEX CHART
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02)
        
        # Price & Clouds
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.1)', line=dict(width=0), name="Apex Cloud"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='yellow', width=2), name="HMA"), row=1, col=1)
        
        # SR Channels
        for z in sr_levels:
            fig.add_shape(type="rect", x0=df.index[0], x1=df.index[-1], y0=z['min'], y1=z['max'], fillcolor="rgba(255,255,255,0.1)", line_width=0, row=1, col=1)

        # Squeeze
        colors = ['#00E676' if v > 0 else '#FF5252' for v in df['Sqz_Mom']]
        fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=colors, name="Squeeze"), row=2, col=1)
        
        # Money Flow
        fig.add_trace(go.Scatter(x=df.index, y=df['MF_Matrix'], fill='tozeroy', line=dict(color='cyan', width=1), name="Money Flow"), row=3, col=1)
        
        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if st.button("✨ Decrypt Market Data"):
            with st.spinner("Analyzing..."):
                keys = {"gemini": gem_key, "openai": oai_key}
                st.session_state.ai_report = AIEngine.generate_report(df, ticker, keys, timeframe)
        if st.session_state.ai_report: st.markdown(st.session_state.ai_report)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Fundamentals")
            f = DataEngine.get_fundamentals(ticker)
            if f: st.json(f)
        with c2:
            st.subheader("Global Heatmap")
            g = DataEngine.get_global_performance()
            if g is not None: st.bar_chart(g)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🔮 Monte Carlo")
            mc = QuantumCore.run_monte_carlo(df)
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(y=np.mean(mc, axis=1), mode='lines', line=dict(color='orange')))
            for i in range(20): fig_mc.add_trace(go.Scatter(y=mc[:,i], mode='lines', line=dict(color='rgba(255,255,255,0.05)'), showlegend=False))
            fig_mc.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_mc, use_container_width=True)
        with c2:
            st.subheader("📊 Volume Profile")
            p, v, poc = QuantumCore.calc_volume_profile(df)
            fig_vp = go.Figure(go.Bar(x=v, y=p, orientation='h', marker_color='rgba(0,200,255,0.3)'))
            fig_vp.add_hline(y=poc, line_color="yellow")
            fig_vp.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_vp, use_container_width=True)
            
        st.subheader("🏦 SMC Fair Value Gaps")
        smc = QuantumCore.calculate_smc(df)
        st.write(f"Detected {len(smc['fvgs'])} FVGs.")

    with tab5:
        st.subheader("📅 Monthly Seasonality")
        s = QuantumCore.get_seasonality(ticker)
        if s is not None: st.dataframe(s.style.background_gradient(subset=['Win Rate'], cmap="RdYlGn"), use_container_width=True)

    with tab6:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Day of Week DNA")
            d = QuantumCore.calc_day_dna(ticker)
            if d is not None: st.dataframe(d.style.background_gradient(subset=['<lambda_0>'], cmap="RdYlGn"))
        with c2:
            st.subheader("🔗 Correlations")
            corr = QuantumCore.calc_correlations(ticker)
            if corr is not None: st.bar_chart(corr)

    with tab7:
        st.subheader("📡 Social Command")
        tv_ticker = ticker.replace("-", "")
        components.html(f"""<div class="tradingview-widget-container"><div id="tradingview_widget"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{"width": "100%", "height": 400, "symbol": "{tv_ticker}", "interval": "D", "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "en", "toolbar_bg": "#f1f3f6", "enable_publishing": false, "allow_symbol_change": true, "container_id": "tradingview_widget"}});</script></div>""", height=400)
        
        msg = st.text_area("Preview", f"🔥 {ticker} GOD MODE\nScore: {last['GM_Score']:.0f}/5\nPrice: {last['Close']:.2f}")
        up = st.file_uploader("Chart")
        if st.button("Send"):
            ok, res = TelegramEngine.send(tg_token, tg_chat, msg, up.getvalue() if up else None)
            if ok: st.success("Sent!")
            else: st.error(res)

if __name__ == "__main__":
    main()
