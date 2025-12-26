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
    initial_sidebar_state="auto"  # CHANGED: Set to 'auto' for mobile-friendly default behavior
)

# --- CUSTOM CSS FOR "DARKPOOL" AESTHETIC ---
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
div[data-testid="stMetric"] {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
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
</style>
""", unsafe_allow_html=True)

# ===============================
# 2. ENVIRONMENT & SECRETS SETUP
# ===============================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def load_key(key_name):
    """Prioritizes Streamlit Secrets > Env Vars > Empty"""
    if key_name in st.secrets:
        return st.secrets[key_name]
    return os.getenv(key_name, "")

# ===============================
# 3. SESSION STATE
# ===============================
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
if "ai_report" not in st.session_state:
    st.session_state.ai_report = ""

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
# 5. TELEGRAM ENGINE (UPGRADED)
# ===============================
class TelegramEngine:
    API_URL = "https://api.telegram.org/bot{}/sendMessage"
    PHOTO_URL = "https://api.telegram.org/bot{}/sendPhoto"

    @staticmethod
    def escape_md(text: str) -> str:
        # Escapes special characters for Telegram MarkdownV2
        escape_chars = r"_*[]()~`>#+-=|{}.!"
        return "".join("\\" + c if c in escape_chars else c for c in text)

    @staticmethod
    def send(token: str, chat_id: str, message: str, photo_data=None):
        if not token or not chat_id:
            return False, "Missing Credentials"
        
        try:
            # 1. Send Photo if provided
            if photo_data:
                files = {'photo': photo_data}
                data_photo = {'chat_id': chat_id, 'caption': "🔥 TITAN OMNI CHART", 'parse_mode': 'Markdown'}
                requests.post(TelegramEngine.PHOTO_URL.format(token), data=data_photo, files=files)

            # 2. Send Text (Chunked for safety)
            # Remove custom markup for raw sending or use MarkdownV2 carefully. 
            # For simplicity in God Mode, we send raw text to avoid formatting errors with complex reports.
            max_length = 3000
            for i in range(0, len(message), max_length):
                chunk = message[i:i+max_length]
                payload = {
                    "chat_id": chat_id,
                    "text": chunk, # Sending plain text to avoid Markdown parsing errors with AI symbols
                    "disable_web_page_preview": True
                }
                r = requests.post(TelegramEngine.API_URL.format(token), json=payload, timeout=10)
            
            return (True, "Sent")
        except Exception as e:
            return False, str(e)

# ===============================
# 6. DATA ENGINE (FULL SUITE)
# ===============================
class DataEngine:
    @staticmethod
    def fetch(ticker, timeframe):
        tf_map = {
            "15m": ("60d", "15m"), "1h": ("730d", "1h"),
            "4h": ("730d", "1h"), "1d": ("5y", "1d"), "1wk": ("10y", "1wk")
        }
        period, interval = tf_map.get(timeframe, ("2y", "1d"))
        
        # Robust Download
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty: return None

        # Clean MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Resample for 4h
        if timeframe == "4h":
            df = df.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min", 
                "Close": "last", "Volume": "sum"
            }).dropna()
            
        return df

    @staticmethod
    def get_fundamentals(ticker):
        if "-" in ticker or "=" in ticker: return None
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "Market Cap": info.get("marketCap", 0),
                "P/E": info.get("trailingPE", 0),
                "Growth": info.get("revenueGrowth", 0),
                "Debt/Eq": info.get("debtToEquity", 0),
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
# 7. QUANTUM CORE (GOD MODE MATH)
# ===============================
class QuantumCore:
    # --- UTILS ---
    @staticmethod
    def calculate_wma(series, length):
        return series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length + 1)) / (length * (length + 1) / 2), raw=True)

    @staticmethod
    def calculate_hma(series, length):
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wma_half = QuantumCore.calculate_wma(series, half_length)
        wma_full = QuantumCore.calculate_wma(series, length)
        diff = 2 * wma_half - wma_full
        return QuantumCore.calculate_wma(diff, sqrt_length)

    @staticmethod
    def calculate_atr(df, length=14):
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        return tr.rolling(length).mean()

    # --- INDICATOR PIPELINE ---
    @staticmethod
    def pipeline(df):
        # 1. Base Indicators
        df['ATR'] = QuantumCore.calculate_atr(df, 14)
        df['HMA'] = QuantumCore.calculate_hma(df['Close'], 55)
        df['EMA20'] = df['Close'].ewm(span=20).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean()

        # 2. Apex Trend (HMA + ATR)
        apex_mult = 1.5
        df['Apex_ATR'] = QuantumCore.calculate_atr(df, 55)
        df['Apex_Upper'] = df['HMA'] + (df['Apex_ATR'] * apex_mult)
        df['Apex_Lower'] = df['HMA'] - (df['Apex_ATR'] * apex_mult)
        df['Apex_Trend'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, 0))
        df['Apex_Trend'] = df['Apex_Trend'].replace(to_replace=0, method='ffill')

        # 3. Squeeze Momentum
        df['Sqz_Basis'] = df['Close'].rolling(20).mean()
        df['Sqz_Dev'] = df['Close'].rolling(20).std() * 2.0
        df['Sqz_Upper_BB'] = df['Sqz_Basis'] + df['Sqz_Dev']
        df['Sqz_Lower_BB'] = df['Sqz_Basis'] - df['Sqz_Dev']
        df['Sqz_Range_MA'] = QuantumCore.calculate_atr(df, 20)
        df['Sqz_Upper_KC'] = df['Sqz_Basis'] + (df['Sqz_Range_MA'] * 1.5)
        df['Sqz_Lower_KC'] = df['Sqz_Basis'] - (df['Sqz_Range_MA'] * 1.5)
        df['Squeeze_On'] = (df['Sqz_Lower_BB'] > df['Sqz_Lower_KC']) & (df['Sqz_Upper_BB'] < df['Sqz_Upper_KC'])
        
        # Momentum (LinReg Proxy)
        highest = df['High'].rolling(20).max()
        lowest = df['Low'].rolling(20).min()
        avg_val = (highest + lowest + df['Sqz_Basis']) / 3
        df['Sqz_Mom'] = (df['Close'] - avg_val).rolling(20).mean() * 100

        # 4. Money Flow Matrix
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        mf_vol = df['Volume'] / df['Volume'].rolling(14).mean()
        rsi_centered = df['RSI'] - 50
        df['MF_Matrix'] = (rsi_centered * mf_vol).ewm(span=3).mean()

        # 5. Volume & EVWM
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
        ev_elast = (df['Close'] - QuantumCore.calculate_hma(df['Close'], 21)) / QuantumCore.calculate_atr(df, 21)
        ev_force = np.sqrt(df['RVOL'].ewm(span=5).mean())
        df['EVWM'] = ev_elast * ev_force

        # 6. Composite God Mode Score
        df['Trend_Dir'] = np.where(df['EMA50'] > df['EMA200'], 1, -1)
        df['GM_Score'] = (
            df['Apex_Trend'] + 
            np.sign(df['Sqz_Mom']) + 
            df['Trend_Dir'] + 
            np.where(df['RSI'] > 50, 0.5, -0.5)
        )

        # 7. Stop Loss Logic
        df['Stop'] = df['Close'] - (df['ATR'] * 2 * df['Apex_Trend'])

        # 8. Fear & Greed Components
        df['FG_Index'] = (df['RSI'] + (df['Sqz_Mom']*5 + 50))/2 # Simplified Proxy
        df['FG_Index'] = df['FG_Index'].clip(0, 100)

        return df.dropna()

    @staticmethod
    def identify_sr_levels(df):
        # Pivot High/Low S&R
        highs = df['High'].values
        lows = df['Low'].values
        max_idx = argrelextrema(highs, np.greater, order=5)[0]
        min_idx = argrelextrema(lows, np.less, order=5)[0]
        
        levels = []
        for i in max_idx[-5:]: levels.append({'price': highs[i], 'type': 'Resistance', 'date': df.index[i]})
        for i in min_idx[-5:]: levels.append({'price': lows[i], 'type': 'Support', 'date': df.index[i]})
        return levels

    @staticmethod
    def calculate_smc(df, swing_length=5):
        # Basic SMC Structure (BOS/CHoCH/FVG)
        structures = []
        fvgs = []
        
        # FVG
        for i in range(2, len(df)):
            if df['Low'].iloc[i] > df['High'].iloc[i-2]: # Bullish FVG
                fvgs.append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i], 'color': 'rgba(0, 255, 0, 0.3)'})
            if df['High'].iloc[i] < df['Low'].iloc[i-2]: # Bearish FVG
                fvgs.append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['Low'].iloc[i-2], 'y1': df['High'].iloc[i], 'color': 'rgba(255, 0, 0, 0.3)'})
        
        return {'fvgs': fvgs}

    @staticmethod
    def run_monte_carlo(df, days=30, sims=1000):
        last_price = df['Close'].iloc[-1]
        returns = df['Close'].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        
        paths = np.zeros((days, sims))
        paths[0] = last_price
        
        for t in range(1, days):
            shock = np.random.normal(mu, sigma, sims)
            paths[t] = paths[t-1] * (1 + shock)
        return paths

    @staticmethod
    def calc_volume_profile(df, bins=50):
        price_min, price_max = df['Low'].min(), df['High'].max()
        price_bins = np.linspace(price_min, price_max, bins)
        mid = (df['Close'] + df['Open']) / 2
        bin_idx = np.digitize(mid, price_bins)
        
        vp = np.zeros(bins)
        for i in range(len(mid)):
            if 0 <= bin_idx[i]-1 < bins:
                vp[bin_idx[i]-1] += df['Volume'].iloc[i]
        
        poc_idx = np.argmax(vp)
        poc_price = price_bins[poc_idx]
        return price_bins, vp, poc_price

    @staticmethod
    def get_seasonality(ticker):
        try:
            df = yf.download(ticker, period="20y", interval="1mo", progress=False)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            
            df['Ret'] = df['Close'].pct_change() * 100
            df['Month'] = df.index.month
            stats = df.groupby('Month')['Ret'].agg(['mean', lambda x: (x>0).mean()*100])
            stats.columns = ['Avg Return', 'Win Rate']
            return stats
        except: return None

# ===============================
# 8. AI ENGINE
# ===============================
class AIEngine:
    @staticmethod
    def generate_report(df, ticker, keys, timeframe):
        last = df.iloc[-1]
        gm_verdict = "STRONG BUY" if last['GM_Score'] >= 2.5 else "STRONG SELL" if last['GM_Score'] <= -2.5 else "NEUTRAL"
        
        prompt = f"""
        ACT AS A SENIOR DARKPOOL TRADER. Analyze {ticker} ({timeframe}).
        
        --- TECHNICAL STACK ---
        Price: {last['Close']:.2f}
        Titan Score: {last['GM_Score']:.1f}/4 ({gm_verdict})
        Apex Trend: {'BULL' if last['Apex_Trend']==1 else 'BEAR'}
        RSI: {last['RSI']:.1f}
        Squeeze: {'FIRING' if last['Squeeze_On'] else 'OFF'}
        Money Flow: {last['MF_Matrix']:.2f}
        RVOL: {last['RVOL']:.1f}x
        
        --- MISSION ---
        1. Market Structure: Is this a trend or chop?
        2. Institutional Flow: What does Money Flow & Volume suggest?
        3. Action Plan: 30-word outlook.
        
        Use emojis (🚀, 📉, ⚠️) and keep it professional but punchy.
        """
        
        report = "AI Analysis Unavailable (Check Keys)"

        if keys.get("gemini") and genai:
            try:
                genai.configure(api_key=keys["gemini"])
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                return f"💎 GEMINI INTEL:\n{response.text}"
            except Exception as e: report = f"Gemini Error: {e}"

        if keys.get("openai") and OpenAI:
            try:
                client = OpenAI(api_key=keys["openai"])
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                return f"🤖 OPENAI INTEL:\n{response.choices[0].message.content}"
            except Exception as e: report = f"OpenAI Error: {e}"
        
        return report

# ===============================
# 9. UI & MAIN LOGIC
# ===============================
def main():
    st.markdown('<div class="title-glow">👁️ TITAN OMNI V3 [GOD MODE]</div>', unsafe_allow_html=True)
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🎛️ Command Center")
        
        # Mode Selection
        input_mode = st.radio("Input Mode:", ["Curated Lists", "Manual Search"], index=1)
        if input_mode == "Curated Lists":
            assets = {"Indices": ["SPY", "QQQ"], "Crypto": ["BTC-USD", "ETH-USD"], "Tech": ["NVDA", "TSLA"]}
            cat = st.selectbox("Category", list(assets.keys()))
            ticker = st.selectbox("Ticker", assets[cat])
        else:
            ticker = st.text_input("Ticker", "BTC-USD").upper()
            
        timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d", "1wk"], index=3)
        
        st.markdown("### 🔐 API Keys")
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
    if df is None:
        st.error(f"Failed to acquire target: {ticker}")
        return

    # Process Data
    df = QuantumCore.pipeline(df)
    last = df.iloc[-1]
    sr_levels = QuantumCore.identify_sr_levels(df)
    funds = DataEngine.get_fundamentals(ticker)
    
    # --- HEADER METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{last['Close']:.2f}")
    c2.metric("Apex Trend", "BULL" if last['Apex_Trend']==1 else "BEAR", delta_color="normal" if last['Apex_Trend']==1 else "inverse")
    c3.metric("Titan Score", f"{last['GM_Score']:.1f}")
    c4.metric("RVOL", f"{last['RVOL']:.1f}x")

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 God Mode Chart", "🤖 AI Analyst", "🌍 Sector/Fund", 
        "🔮 Quant/SMC", "📅 Seasonality", "📡 Broadcast"
    ])

    with tab1:
        # COMPLEX CHARTING
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02)
        
        # 1. Price & Clouds
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.1)', line=dict(width=0), name="Apex Cloud"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='yellow', width=2), name="HMA Trend"), row=1, col=1)
        
        # S/R Lines
        for lvl in sr_levels:
            c = "green" if lvl['type']=="Support" else "red"
            fig.add_shape(type="line", x0=lvl['date'], x1=df.index[-1], y0=lvl['price'], y1=lvl['price'], line=dict(color=c, dash="dot"), row=1, col=1)

        # 2. Squeeze
        colors = ['#00E676' if v > 0 else '#FF5252' for v in df['Sqz_Mom']]
        fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=colors, name="Squeeze"), row=2, col=1)

        # 3. Money Flow
        fig.add_trace(go.Scatter(x=df.index, y=df['MF_Matrix'], fill='tozeroy', line=dict(color='cyan', width=1), name="Money Flow"), row=3, col=1)

        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if st.button("✨ Generate Intel Report"):
            with st.spinner("Decrypting Market Data..."):
                keys = {"gemini": gem_key, "openai": oai_key}
                st.session_state.ai_report = AIEngine.generate_report(df, ticker, keys, timeframe)
        
        if st.session_state.ai_report:
            st.markdown(st.session_state.ai_report)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Fundamentals")
            if funds: st.json(funds)
            else: st.warning("No fundamental data.")
        with c2:
            st.subheader("Global Heatmap")
            perf = DataEngine.get_global_performance()
            if perf is not None:
                st.bar_chart(perf)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🔮 Monte Carlo (30 Day)")
            mc = QuantumCore.run_monte_carlo(df)
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(y=np.mean(mc, axis=1), mode='lines', name='Mean Path', line=dict(color='orange')))
            for i in range(20): fig_mc.add_trace(go.Scatter(y=mc[:,i], mode='lines', line=dict(color='rgba(255,255,255,0.05)'), showlegend=False))
            fig_mc.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_mc, use_container_width=True)
            
        with c2:
            st.subheader("📊 Volume Profile")
            bins, vp, poc = QuantumCore.calc_volume_profile(df)
            fig_vp = go.Figure(go.Bar(x=vp, y=bins, orientation='h', marker_color='rgba(0,200,255,0.3)'))
            fig_vp.add_hline(y=poc, line_color="yellow", annotation_text="POC")
            fig_vp.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_vp, use_container_width=True)
            
        st.subheader("🏦 SMC Structures (FVG)")
        smc = QuantumCore.calculate_smc(df)
        st.write(f"Detected {len(smc['fvgs'])} Fair Value Gaps in visible range.")

    with tab5:
        st.subheader("📅 Monthly Seasonality")
        seas = QuantumCore.get_seasonality(ticker)
        if seas is not None:
            st.dataframe(seas.style.background_gradient(subset=['Win Rate'], cmap="RdYlGn"), use_container_width=True)
        else:
            st.info("Insufficient history for seasonality.")

    with tab6:
        st.subheader("📡 Social Command")
        
        # TradingView Widget
        tv_ticker = ticker.replace("-", "")
        components.html(f"""
        <div class="tradingview-widget-container">
          <div id="tradingview_widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
          "width": "100%", "height": 400, "symbol": "{tv_ticker}", "interval": "D",
          "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "en",
          "toolbar_bg": "#f1f3f6", "enable_publishing": false, "allow_symbol_change": true,
          "container_id": "tradingview_widget"
          }}
          );
          </script>
        </div>
        """, height=400)
        
        # Broadcast Form
        st.markdown("---")
        signal_txt = f"""
🔥 TITAN SIGNAL: {ticker} ({timeframe})
Price: {last['Close']:.2f}
Score: {last['GM_Score']:.1f}
Apex: {'🐂 BULL' if last['Apex_Trend']==1 else '🐻 BEAR'}
Squeeze: {'✅ ON' if last['Squeeze_On'] else '❌ OFF'}
RVOL: {last['RVOL']:.1f}x

#TitanOmni #Trading
"""
        msg_preview = st.text_area("Message Preview", signal_txt, height=150)
        uploaded_img = st.file_uploader("Attach Chart Screenshot", type=['png', 'jpg'])
        
        if st.button("📤 SEND TO TELEGRAM"):
            img_bytes = uploaded_img.getvalue() if uploaded_img else None
            ok, res = TelegramEngine.send(tg_token, tg_chat, msg_preview, img_bytes)
            if ok: st.success("Broadcast Sent!")
            else: st.error(res)

if __name__ == "__main__":
    main()
