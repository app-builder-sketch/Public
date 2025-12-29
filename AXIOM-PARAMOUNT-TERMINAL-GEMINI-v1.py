import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime
import requests
from scipy.stats import linregress

# ==========================================
# 1. SYSTEM CONFIGURATION & SECRETS
# ==========================================
st.set_page_config(layout="wide", page_title="AXIOM PARAMOUNT BROADCASTER", page_icon="📡")

# Load Secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    TG_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    TG_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
except Exception:
    st.error("STRICT STOP: Missing API credentials (OPENAI_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID) in secrets.")
    st.stop()

# ==========================================
# 2. DATA INGESTION ENGINE (Robust Multi-Index Fix)
# ==========================================
@st.cache_data(ttl=300)
def get_clean_data(ticker, period, interval):
    """Downloads and flattens data to prevent MultiIndex errors."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        # CRITICAL FIX: Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_macro_data():
    """Fetches global macro assets for ratio analysis."""
    tickers = {
        "SPY": "SPY", "TLT": "TLT", "GLD": "GC=F", 
        "BTC": "BTC-USD", "ETH": "ETH-USD", "VIX": "^VIX", "DXY": "DX-Y.NYB"
    }
    try:
        data = yf.download(list(tickers.values()), period="5d", interval="1d", progress=False)['Close']
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return pd.DataFrame()

# ==========================================
# 3. UNIVERSAL LOGIC ENGINE (ALL INDICATORS)
# ==========================================
class UniversalParamountEngine:
    
    @staticmethod
    def calculate_all(df):
        """Executes the complete Axiom + Titan indicator suite. NO OMISSIONS."""
        
        # --- A. CORE MATH HELPERS ---
        def wma(s, l):
            weights = np.arange(1, l + 1)
            return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        
        def hma(s, l):
            half = int(l/2); sqrt = int(np.sqrt(l))
            return wma(2 * wma(s, half) - wma(s, l), sqrt)

        def atr(df, l):
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return tr.rolling(l).mean()

        # --- B. TREND ARCHITECTURE (Apex Trend + HMA) ---
        df['HMA'] = hma(df['Close'], 55)
        df['ATR'] = atr(df, 14)
        
        # Apex Trend Bands
        apex_mult = 1.5
        df['Apex_Upper'] = df['HMA'] + (atr(df, 55) * apex_mult)
        df['Apex_Lower'] = df['HMA'] - (atr(df, 55) * apex_mult)
        df['Apex_Trend'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, 0))

        # --- C. DARKPOOL SQUEEZE MOMENTUM ---
        basis = df['Close'].rolling(20).mean()
        dev = df['Close'].rolling(20).std() * 2.0
        upper_bb, lower_bb = basis + dev, basis - dev
        kc_range = atr(df, 20)
        upper_kc, lower_kc = basis + (kc_range * 1.5), basis - (kc_range * 1.5)
        
        # Squeeze On/Off
        df['Squeeze_On'] = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        # Linear Regression Momentum Proxy
        df['Sqz_Mom'] = (df['Close'] - ((df['High'].rolling(20).max() + df['Low'].rolling(20).min() + basis)/3)).rolling(20).mean()

        # --- D. MONEY FLOW MATRIX ---
        diff = df['Close'].diff()
        gain = diff.clip(lower=0).rolling(14).mean()
        loss = diff.clip(upper=0).abs().rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        rsi_norm = (rsi - 50)
        mf_vol = df['Volume'] / df['Volume'].rolling(14).mean()
        df['MF_Matrix'] = (rsi_norm * mf_vol).ewm(span=3).mean()

        # --- E. GANN HIGH/LOW ACTIVATOR ---
        df['Gann_High'] = df['High'].rolling(3).mean()
        df['Gann_Low'] = df['Low'].rolling(3).mean()
        df['Gann_Trend'] = np.where(df['Close'] > df['Gann_High'].shift(1), 1, np.where(df['Close'] < df['Gann_Low'].shift(1), -1, 0))

        # --- F. QUANTUM PHYSICS (Axiom) ---
        # 1. CHEDO (Entropy)
        log_ret = np.diff(np.log(df['Close'].values), prepend=np.log(df['Close'].iloc[0]))
        mu = pd.Series(log_ret).rolling(50).mean(); sigma = pd.Series(log_ret).rolling(50).std()
        v = sigma / (np.abs(mu) + 1e-9)
        hyper_dist = np.log(np.abs(log_ret) * v + np.sqrt((np.abs(log_ret) * v)**2 + 1))
        df['CHEDO'] = np.tanh(pd.Series(hyper_dist).rolling(50).mean().values * 4)

        # 2. RQZO (Relativity)
        mn, mx = df['Close'].rolling(100).min(), df['Close'].rolling(100).max()
        norm = (df['Close'] - mn) / (mx - mn + 1e-9)
        gamma = 1 / np.sqrt(1 - (np.clip(np.abs(norm.diff()), 0, 0.049)/0.05)**2)
        tau = (np.arange(len(df)) % 100) / gamma.fillna(1.0)
        zeta = np.zeros(len(df))
        for n in range(1, 26): zeta += (n**-0.5) * np.sin(tau * np.log(n))
        df['RQZO'] = pd.Series(zeta).fillna(0)

        # 3. Apex Flux (Efficiency)
        rg = df['High'] - df['Low']; body = np.abs(df['Close'] - df['Open'])
        eff = (body / rg.replace(0, 1)).ewm(span=14).mean()
        v_rat = df['Volume'] / df['Volume'].rolling(55).mean()
        df['Flux'] = (np.sign(df['Close'] - df['Open']) * eff * v_rat).ewm(span=5).mean()

        # --- G. GOD MODE CONFLUENCE SCORE ---
        df['Titan_Score'] = (
            np.sign(df['Apex_Trend']) + 
            np.sign(df['Gann_Trend']) + 
            np.sign(df['Flux']) + 
            np.sign(df['Sqz_Mom'])
        )

        return df

    @staticmethod
    def calculate_volume_profile(df):
        """VPVR Calculation with Altair Fix."""
        price_bins = np.linspace(df['Low'].min(), df['High'].max(), 50)
        df['Bin'] = pd.cut(df['Close'], bins=price_bins, include_lowest=True)
        vp = df.groupby('Bin', observed=False)['Volume'].sum().reset_index()
        poc_idx = vp['Volume'].idxmax()
        poc = vp.loc[poc_idx, 'Bin'].mid
        # FIX: Convert Interval to Float
        vp['Price_Level'] = vp['Bin'].apply(lambda x: x.mid)
        return vp, poc

    @staticmethod
    def run_monte_carlo(df, days=30, sims=200):
        """Stochastic Forecasting."""
        returns = df['Close'].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        sim_rets = np.random.normal(mu, sigma, (days, sims))
        paths = np.zeros((days, sims))
        paths[0] = df['Close'].iloc[-1]
        for t in range(1, days):
            paths[t] = paths[t-1] * (1 + sim_rets[t])
        return paths

# ==========================================
# 4. ADVANCED BROADCAST ENGINE (UPGRADED)
# ==========================================
class AdvancedDispatcher:
    
    @staticmethod
    def _send(payload):
        """Handles chunking and transmission."""
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        max_len = 4000 # Safety margin below 4096
        
        try:
            if len(payload) <= max_len:
                requests.post(url, json={"chat_id": TG_CHAT_ID, "text": payload, "parse_mode": "Markdown"})
            else:
                # Chunking Logic
                parts = [payload[i:i+max_len] for i in range(0, len(payload), max_len)]
                for i, part in enumerate(parts):
                    header = f"*(Part {i+1}/{len(parts)})*\n" if i > 0 else ""
                    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": header + part, "parse_mode": "Markdown"})
            return True
        except Exception as e:
            st.error(f"Transmission Error: {e}")
            return False

    @staticmethod
    def dispatch_flash_signal(ticker, latest, score):
        """Option 1: Ultra-fast execution signal."""
        emoji = "🟢 BUY" if score > 0 else "🔴 SELL"
        sqz = "💥 BREAKOUT" if not latest['Squeeze_On'] else "💤 SQUEEZING"
        
        msg = f"⚡ *FLASH SIGNAL: {ticker}*\n"
        msg += f"Action: *{emoji}*\n"
        msg += f"Price: `${latest['Close']:,.2f}`\n"
        msg += f"Score: `{score:.0f}/4` | State: `{sqz}`\n"
        msg += f"Time: `{datetime.now().strftime('%H:%M:%S UTC')}`"
        
        return AdvancedDispatcher._send(msg)

    @staticmethod
    def dispatch_institutional_memo(ticker, latest, macro_data):
        """Option 2: Data-dense professional report."""
        btc_spx = 0.0
        spy_tlt = 0.0
        if not macro_data.empty and 'BTC-USD' in macro_data.columns and 'SPY' in macro_data.columns:
            btc_spx = macro_data['BTC-USD'].iloc[-1] / macro_data['SPY'].iloc[-1]
        if not macro_data.empty and 'SPY' in macro_data.columns and 'TLT' in macro_data.columns:
            spy_tlt = macro_data['SPY'].iloc[-1] / macro_data['TLT'].iloc[-1]
            
        msg = f"📑 *INSTITUTIONAL MEMO: {ticker}*\n"
        msg += f"───────────────────────\n"
        msg += f"**TECHNICALS**\n"
        msg += f"• Price:   `${latest['Close']:,.2f}`\n"
        msg += f"• HMA(55): `${latest['HMA']:,.2f}`\n"
        msg += f"• Flux:    `{latest['Flux']:.3f}` (Vec)\n"
        msg += f"• CHEDO:   `{latest['CHEDO']:.3f}` (Ent)\n\n"
        msg += f"**MACRO RISK**\n"
        msg += f"• Risk-On (SPY/TLT): `{spy_tlt:.2f}`\n"
        msg += f"• Beta (BTC/SPX):    `{btc_spx:.4f}`\n"
        msg += f"───────────────────────\n"
        msg += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
        
        return AdvancedDispatcher._send(msg)

    @staticmethod
    def dispatch_quantum_deep_dive(ticker, latest, ai_text):
        """Option 3: Full AI Synthesis + Physics."""
        msg = f"🧠 *QUANTUM DEEP DIVE: {ticker}*\n\n"
        msg += f"*PHYSICS ENGINE*\n"
        msg += f"The asset is currently in a {'Superconductor' if abs(latest['Flux']) > 0.5 else 'Resistive'} state.\n"
        msg += f"Entropy (CHEDO) is at {latest['CHEDO']:.2f}, suggesting {'Chaos' if abs(latest['CHEDO']) > 0.7 else 'Order'}.\n\n"
        msg += f"*AI SYNTHESIS*\n"
        msg += ai_text
        msg += f"\n\n_End of Transmission_"
        
        return AdvancedDispatcher._send(msg)

# ==========================================
# 5. UI & EXECUTION
# ==========================================
st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono'; }
    div[data-testid="stMetric"] { background: rgba(0, 240, 255, 0.05); border-left: 4px solid #00F0FF; padding: 10px; }
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #333; }
    .stTabs [aria-selected="true"] { color: #00F0FF; border-bottom-color: #00F0FF; }
    button[kind="secondary"] { border: 1px solid #333; color: #888; }
</style>
""", unsafe_allow_html=True)

st.title("💠 Axiom Paramount Broadcaster")

# Sidebar Controls
st.sidebar.header("📡 Transmission Control")
asset = st.sidebar.text_input("Asset Ticker", "BTC-USD").upper()
interval = st.sidebar.selectbox("Interval", ["1h", "4h", "1d"], index=2)

# Broadcast Type Selector
st.sidebar.subheader("Broadcast Protocol")
broadcast_type = st.sidebar.radio(
    "Select Transmission Mode:",
    ("⚡ Flash Signal", "📑 Institutional Memo", "🧠 Quantum Deep Dive")
)

# TradingView Widget
with st.sidebar:
    st.markdown("---")
    tv_code = f"""<script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"width": "100%","height": 300,"symbol": "{asset}","interval": "D","theme": "dark"}});</script>"""
    st.components.v1.html(tv_code, height=300)

# Main Execution Block
if st.button(f"EXECUTE: {broadcast_type.upper()}"):
    with st.spinner(f"Running Universal Engine for {asset}..."):
        # 1. Data Ingestion
        df = get_clean_data(asset, "1y", interval)
        macro = get_macro_data()
        
        if not df.empty and len(df) > 50:
            # 2. Universal Logic
            df = UniversalParamountEngine.calculate_all(df)
            latest = df.iloc[-1].to_dict()
            latest['Close'] = float(df['Close'].iloc[-1])
            
            # 3. AI Generation (Only needed for Deep Dive or specific summaries)
            ai_res = ""
            if broadcast_type == "🧠 Quantum Deep Dive":
                client = OpenAI(api_key=OPENAI_API_KEY)
                ai_res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": "You are a Quant Architect. Provide a deep technical breakdown."},
                              {"role": "user", "content": f"Data: {latest}"}]
                ).choices[0].message.content
            
            # 4. Visualization (Always show dashboard locally)
            t1, t2, t3, t4 = st.tabs(["📊 God Mode", "⚛️ Physics", "🔮 Forecasts", "🌍 Macro"])
            
            with t1:
                # GOD MODE CHART
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.15, 0.15, 0.2])
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='#00F0FF', width=1), name="HMA"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(color='rgba(0,255,0,0.3)'), name="Apex Up"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', line=dict(color='rgba(255,0,0,0.3)'), name="Apex Down"), row=1, col=1)
                
                cols = ['#00E676' if v > 0 else '#FF1744' for v in df['Sqz_Mom']]
                fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=cols, name="Squeeze"), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MF_Matrix'], fill='tozeroy', line=dict(color='yellow'), name="Money Flow"), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Flux'], line=dict(color='#D500F9'), name="Flux"), row=4, col=1)
                fig.update_layout(height=1000, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with t2:
                # PHYSICS CHART
                fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True)
                fig2.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], fill='tozeroy', name="CHEDO Entropy"), row=1, col=1)
                fig2.add_trace(go.Scatter(x=df.index, y=df['RQZO'], name="RQZO Relativity"), row=2, col=1)
                fig2.update_layout(height=600, template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)
                
            with t3:
                # FORECASTS
                c1, c2 = st.columns(2)
                with c1:
                    vp, poc = UniversalParamountEngine.calculate_volume_profile(df)
                    st.markdown(f"**POC**: ${poc:.2f}")
                    st.bar_chart(vp.set_index('Price_Level')['Volume'])
                with c2:
                    mc = UniversalParamountEngine.run_monte_carlo(df)
                    st.line_chart(pd.DataFrame(mc[:, :50]))

            with t4:
                # MACRO
                if not macro.empty:
                    chart_data = pd.DataFrame()
                    if 'BTC-USD' in macro.columns and 'SPY' in macro.columns:
                        chart_data["BTC/SPX"] = macro['BTC-USD'] / macro['SPY']
                    if 'SPY' in macro.columns and 'TLT' in macro.columns:
                        chart_data["SPY/TLT"] = macro['SPY'] / macro['TLT']
                    st.line_chart(chart_data)

            # 5. Routing Logic (Advanced Dispatcher)
            success = False
            if broadcast_type == "⚡ Flash Signal":
                success = AdvancedDispatcher.dispatch_flash_signal(asset, latest, latest['Titan_Score'])
            elif broadcast_type == "📑 Institutional Memo":
                success = AdvancedDispatcher.dispatch_institutional_memo(asset, latest, macro)
            elif broadcast_type == "🧠 Quantum Deep Dive":
                success = AdvancedDispatcher.dispatch_quantum_deep_dive(asset, latest, ai_res)

            if success:
                st.success(f"Transmission Successful: {broadcast_type}")
            
            if ai_res:
                st.markdown("### 🧬 AI Synthesis Preview")
                st.write(ai_res)
                
        else:
            st.error("Insufficient Data for Analysis.")
