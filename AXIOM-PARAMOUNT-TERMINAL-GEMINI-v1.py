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
st.set_page_config(layout="wide", page_title="AXIOM PARAMOUNT ULTIMATE", page_icon="💠")

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
        
        # CRITICAL FIX: Flatten MultiIndex for Macro Data too
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
        """
        Executes the complete Axiom + Titan indicator suite.
        NO OMISSIONS.
        """
        # --- A. CORE MATH HELPERS ---
        def wma(s, l):
            weights = np.arange(1, l + 1)
            return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        
        def hma(s, l):
            half = int(l/2); sqrt = int(np.sqrt(l))
            return wma(2 * wma(s, half) - wma(s, l), sqrt)

        def atr(df, l):
            tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
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
        
        # Squeeze On/Off (Fix applied: indexes are now flat)
        df['Squeeze_On'] = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        
        # Linear Regression Momentum Proxy
        df['Sqz_Mom'] = (df['Close'] - ((df['High'].rolling(20).max() + df['Low'].rolling(20).min() + basis)/3)).rolling(20).mean()

        # --- D. MONEY FLOW MATRIX ---
        # Normalized RSI * Volume Flow
        rsi = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))
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
        # Summation of Apex, Gann, Flux, Squeeze Direction
        df['Titan_Score'] = (
            np.sign(df['Apex_Trend']) + 
            np.sign(df['Gann_Trend']) + 
            np.sign(df['Flux']) + 
            np.sign(df['Sqz_Mom'])
        )

        return df

    @staticmethod
    def calculate_volume_profile(df):
        """VPVR Calculation."""
        price_bins = np.linspace(df['Low'].min(), df['High'].max(), 50)
        df['Bin'] = pd.cut(df['Close'], bins=price_bins, include_lowest=True)
        vp = df.groupby('Bin', observed=False)['Volume'].sum().reset_index()
        poc = vp.loc[vp['Volume'].idxmax()]['Bin'].mid
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
# 4. INTELLIGENCE & BROADCAST ENGINE
# ==========================================
class ParamountDispatcher:
    @staticmethod
    def broadcast(report_type, ticker, latest, ai_text, macro_data):
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        # Macro Ratios
        btc_spx = 0.0
        spy_tlt = 0.0
        
        # Safe extraction of macro ratios
        if not macro_data.empty:
            try:
                if 'BTC-USD' in macro_data.columns and 'SPY' in macro_data.columns:
                    btc_spx = macro_data['BTC-USD'].iloc[-1] / macro_data['SPY'].iloc[-1]
                if 'SPY' in macro_data.columns and 'TLT' in macro_data.columns:
                    spy_tlt = macro_data['SPY'].iloc[-1] / macro_data['TLT'].iloc[-1]
            except:
                pass
        
        emoji = "🟢" if latest['Titan_Score'] > 0 else "🔴"
        
        payload = f"💠 *AXIOM PARAMOUNT*: {report_type}\n"
        payload += f"Asset: `{ticker}` | {ts}\n"
        payload += f"Price: `${latest['Close']:,.2f}`\n\n"
        
        payload += f"*GOD MODE METRICS:*\n"
        payload += f"{emoji} Titan Score: `{latest['Titan_Score']:.0f}/4`\n"
        payload += f"🌊 Apex Trend: `{'BULL' if latest['Apex_Trend']==1 else 'BEAR'}`\n"
        payload += f"⚛️ Entropy (CHEDO): `{latest['CHEDO']:.3f}`\n"
        payload += f"🚀 Flux Vector: `{latest['Flux']:.3f}`\n"
        payload += f"💥 Squeeze: `{'ACTIVE' if latest['Squeeze_On'] else 'OFF'}`\n\n"
        
        payload += f"*INSTITUTIONAL MACRO:*\n"
        payload += f"BTC/SPX Ratio: `{btc_spx:.4f}`\n"
        payload += f"Risk-On (SPY/TLT): `{spy_tlt:.2f}`\n\n"
        
        payload += f"*AI SYNTHESIS:*\n{ai_text}"

        try:
            requests.post(url, json={"chat_id": TG_CHAT_ID, "text": payload, "parse_mode": "Markdown"}, timeout=10)
            return True
        except: return False

# ==========================================
# 5. UI & EXECUTION
# ==========================================
st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono'; }
    div[data-testid="stMetric"] { background: rgba(0, 240, 255, 0.05); border-left: 4px solid #00F0FF; padding: 10px; }
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #333; }
    .stTabs [data-baseweb="tab"] { color: #888; }
    .stTabs [aria-selected="true"] { color: #00F0FF; border-bottom-color: #00F0FF; }
</style>
""", unsafe_allow_html=True)

st.title("💠 Axiom Paramount Ultimate")

# Sidebar
asset = st.sidebar.text_input("Asset Ticker", "BTC-USD").upper()
interval = st.sidebar.selectbox("Interval", ["1h", "4h", "1d"], index=2)
rep_mode = st.sidebar.radio("Broadcast Mode", ["SIGNAL", "RISK", "SUMMARY"])

# TradingView Sidebar
with st.sidebar:
    st.markdown("---")
    tv_code = f"""<script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"width": "100%","height": 300,"symbol": "{asset}","interval": "D","theme": "dark"}});</script>"""
    st.components.v1.html(tv_code, height=300)

if st.button(f"EXECUTE FULL ANALYSIS"):
    with st.spinner(f"Running Universal Engine for {asset}..."):
        # 1. Data (Using the FIXED Clean Ingestion)
        df = get_clean_data(asset, "1y", interval)
        macro = get_macro_data()
        
        if not df.empty:
            # 2. Logic
            df = UniversalParamountEngine.calculate_all(df)
            latest = df.iloc[-1].to_dict()
            latest['Close'] = float(df['Close'].iloc[-1])
            
            # 3. AI
            client = OpenAI(api_key=OPENAI_API_KEY)
            ai_res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are a Quant Architect. Synthesize these technicals into a trading plan."},
                          {"role": "user", "content": str(latest)}]
            ).choices[0].message.content
            
            # 4. Visualization (Tabular Layout)
            t1, t2, t3, t4 = st.tabs(["📊 God Mode Charts", "⚛️ Quantum Physics", "🔮 Quant Forecasts", "🌍 Macro"])
            
            with t1:
                # GOD MODE CHART
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.15, 0.15, 0.2])
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='#00F0FF', width=1), name="HMA"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(color='rgba(0,255,0,0.3)'), name="Apex Up"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', line=dict(color='rgba(255,0,0,0.3)'), name="Apex Down"), row=1, col=1)
                
                # Squeeze
                cols = ['#00E676' if v > 0 else '#FF1744' for v in df['Sqz_Mom']]
                fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=cols, name="Squeeze"), row=2, col=1)
                
                # Money Flow Matrix
                fig.add_trace(go.Scatter(x=df.index, y=df['MF_Matrix'], fill='tozeroy', line=dict(color='yellow'), name="Money Flow"), row=3, col=1)
                
                # Flux
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
                # QUANT FORECASTS
                c1, c2 = st.columns(2)
                with c1:
                    vp, poc = UniversalParamountEngine.calculate_volume_profile(df)
                    st.markdown(f"**Volume Profile POC**: ${poc:.2f}")
                    st.bar_chart(vp.set_index('Bin')['Volume'])
                with c2:
                    mc = UniversalParamountEngine.run_monte_carlo(df)
                    st.markdown("**Monte Carlo (30 Day Path)**")
                    st.line_chart(mc[:, :50]) # Show 50 paths
                    
            with t4:
                # MACRO RATIOS
                if not macro.empty:
                    # Safe plot handling
                    chart_data = pd.DataFrame()
                    if 'BTC-USD' in macro.columns and 'SPY' in macro.columns:
                        chart_data["BTC/SPX"] = macro['BTC-USD'] / macro['SPY']
                    if 'SPY' in macro.columns and 'TLT' in macro.columns:
                        chart_data["SPY/TLT (Risk-On)"] = macro['SPY'] / macro['TLT']
                    
                    if not chart_data.empty:
                        st.line_chart(chart_data)

            # 5. Broadcast
            if ParamountDispatcher.broadcast(rep_mode, asset, latest, ai_res, macro):
                st.success(f"Paramount {rep_mode} Broadcast Sent to Telegram.")
            
            st.markdown("### 🧬 Detailed AI Synthesis")
            st.write(ai_res)
