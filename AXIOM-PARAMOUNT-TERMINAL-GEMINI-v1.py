import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime
import requests
import time

# ==========================================
# 1. SYSTEM CONFIGURATION & SECRETS
# ==========================================
st.set_page_config(layout="wide", page_title="AXIOM OMNI-LINK TERMINAL", page_icon="💠")

# Load Secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    TG_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    TG_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
except Exception:
    st.error("STRICT STOP: Missing API credentials (OPENAI_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID) in secrets.")
    st.stop()

# Initialize Session State
if 'report_draft' not in st.session_state: st.session_state['report_draft'] = ""
if 'last_run_dt' not in st.session_state: st.session_state['last_run_dt'] = None
if 'selected_ticker' not in st.session_state: st.session_state['selected_ticker'] = "BTC-USD"

# ==========================================
# 2. ASSET REPOSITORY (100s of Live Tickers)
# ==========================================
class AssetRepository:
    @staticmethod
    def get_assets():
        return {
            "CRYPTO (Major)": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "LINK-USD"],
            "CRYPTO (DeFi/L2)": ["UNI-USD", "MATIC-USD", "LDO-USD", "AAVE-USD", "MKR-USD", "OP-USD", "ARB-USD", "SNX-USD", "RUNE-USD"],
            "CRYPTO (AI/Meme)": ["FET-USD", "RNDR-USD", "AGIX-USD", "SHIB-USD", "PEPE-USD", "WIF-USD", "BONK-USD"],
            "US STOCKS (Mag 7)": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "US STOCKS (Tech)": ["AMD", "INTC", "CRM", "ADBE", "NFLX", "PYPL", "SQ", "COIN", "MSTR", "PLTR", "UBER", "ABNB"],
            "US STOCKS (Finance)": ["JPM", "BAC", "GS", "MS", "V", "MA", "BLK"],
            "INDICES": ["SPY", "QQQ", "DIA", "IWM", "^VIX", "TLT"],
            "FOREX": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"],
            "COMMODITIES": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"]
        }

# ==========================================
# 3. DATA INGESTION ENGINE (Robust Fixes)
# ==========================================
@st.cache_data(ttl=300)
def get_clean_data(ticker, period, interval):
    """Downloads and flattens data to prevent MultiIndex errors."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_macro_data():
    """Fetches global macro assets for ratio analysis."""
    tickers = {"SPY": "SPY", "TLT": "TLT", "BTC": "BTC-USD", "ETH": "ETH-USD"}
    try:
        data = yf.download(list(tickers.values()), period="5d", interval="1d", progress=False)['Close']
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return pd.DataFrame()

# ==========================================
# 4. UNIVERSAL LOGIC ENGINE (33 Indicators)
# ==========================================
class UniversalParamountEngine:
    @staticmethod
    def calculate_all(df):
        # A. Core Math
        def wma(s, l):
            weights = np.arange(1, l + 1)
            return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        def hma(s, l):
            half = int(l/2); sqrt = int(np.sqrt(l))
            return wma(2 * wma(s, half) - wma(s, l), sqrt)
        def atr(df, l):
            tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
            return tr.rolling(l).mean()

        # B. Trend (Apex + HMA)
        df['HMA'] = hma(df['Close'], 55)
        df['ATR'] = atr(df, 14)
        df['Apex_Upper'] = df['HMA'] + (atr(df, 55) * 1.5)
        df['Apex_Lower'] = df['HMA'] - (atr(df, 55) * 1.5)
        df['Apex_Trend'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, 0))

        # C. Squeeze Momentum
        basis = df['Close'].rolling(20).mean()
        dev = df['Close'].rolling(20).std() * 2.0
        upper_bb, lower_bb = basis + dev, basis - dev
        kc_range = atr(df, 20)
        upper_kc, lower_kc = basis + (kc_range * 1.5), basis - (kc_range * 1.5)
        df['Squeeze_On'] = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        df['Sqz_Mom'] = (df['Close'] - ((df['High'].rolling(20).max() + df['Low'].rolling(20).min() + basis)/3)).rolling(20).mean()

        # D. Money Flow Matrix
        diff = df['Close'].diff()
        gain = diff.clip(lower=0).rolling(14).mean()
        loss = diff.clip(upper=0).abs().rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        rsi_norm = (rsi - 50)
        mf_vol = df['Volume'] / df['Volume'].rolling(14).mean()
        df['MF_Matrix'] = (rsi_norm * mf_vol).ewm(span=3).mean()

        # E. Gann
        df['Gann_High'] = df['High'].rolling(3).mean()
        df['Gann_Low'] = df['Low'].rolling(3).mean()
        df['Gann_Trend'] = np.where(df['Close'] > df['Gann_High'].shift(1), 1, np.where(df['Close'] < df['Gann_Low'].shift(1), -1, 0))

        # F. Physics (CHEDO, RQZO, Flux)
        log_ret = np.diff(np.log(df['Close'].values), prepend=np.log(df['Close'].iloc[0]))
        mu = pd.Series(log_ret).rolling(50).mean(); sigma = pd.Series(log_ret).rolling(50).std()
        v = sigma / (np.abs(mu) + 1e-9)
        hyper_dist = np.log(np.abs(log_ret) * v + np.sqrt((np.abs(log_ret) * v)**2 + 1))
        df['CHEDO'] = np.tanh(pd.Series(hyper_dist).rolling(50).mean().values * 4)

        mn, mx = df['Close'].rolling(100).min(), df['Close'].rolling(100).max()
        norm = (df['Close'] - mn) / (mx - mn + 1e-9)
        gamma = 1 / np.sqrt(1 - (np.clip(np.abs(norm.diff()), 0, 0.049)/0.05)**2)
        tau = (np.arange(len(df)) % 100) / gamma.fillna(1.0)
        zeta = np.zeros(len(df))
        for n in range(1, 26): zeta += (n**-0.5) * np.sin(tau * np.log(n))
        df['RQZO'] = pd.Series(zeta).fillna(0)

        rg = df['High'] - df['Low']; body = np.abs(df['Close'] - df['Open'])
        eff = (body / rg.replace(0, 1)).ewm(span=14).mean()
        v_rat = df['Volume'] / df['Volume'].rolling(55).mean()
        df['Flux'] = (np.sign(df['Close'] - df['Open']) * eff * v_rat).ewm(span=5).mean()

        # G. Titan Score
        df['Titan_Score'] = np.sign(df['Apex_Trend']) + np.sign(df['Gann_Trend']) + np.sign(df['Flux']) + np.sign(df['Sqz_Mom'])
        return df

    @staticmethod
    def calculate_vp_poc(df):
        price_bins = np.linspace(df['Low'].min(), df['High'].max(), 50)
        df['Bin'] = pd.cut(df['Close'], bins=price_bins, include_lowest=True)
        vp = df.groupby('Bin', observed=False)['Volume'].sum().reset_index()
        poc_idx = vp['Volume'].idxmax()
        poc = vp.loc[poc_idx, 'Bin'].mid
        vp['Price_Level'] = vp['Bin'].apply(lambda x: x.mid)
        return vp, poc

    @staticmethod
    def run_monte_carlo_stats(df, days=30, sims=100):
        returns = df['Close'].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        sim_rets = np.random.normal(mu, sigma, (days, sims))
        paths = np.zeros((days, sims))
        paths[0] = df['Close'].iloc[-1]
        for t in range(1, days):
            paths[t] = paths[t-1] * (1 + sim_rets[t])
        
        final_prices = paths[-1]
        mean_price = np.mean(final_prices)
        lower_bound = np.percentile(final_prices, 5)
        return mean_price, lower_bound

# ==========================================
# 5. BROADCAST GENERATOR & TRANSMITTER
# ==========================================
class BroadcastEngine:
    @staticmethod
    def generate_content(mode, ticker, latest, macro, poc, mc_mean, mc_low):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        # Macro Calc
        btc_spx, spy_tlt = 0.0, 0.0
        if not macro.empty:
            try:
                btc_spx = macro['BTC-USD'].iloc[-1] / macro['SPY'].iloc[-1]
                spy_tlt = macro['SPY'].iloc[-1] / macro['TLT'].iloc[-1]
            except: pass
            
        emoji = "🟢 BUY" if latest['Titan_Score'] > 0 else "🔴 SELL"
        
        if mode == "⚡ Flash Signal":
            return (
                f"⚡ *FLASH SIGNAL: {ticker}*\n"
                f"Action: *{emoji}* (Score: {latest['Titan_Score']:.0f}/4)\n"
                f"Price: `${latest['Close']:,.2f}`\n"
                f"Flux: `{latest['Flux']:.3f}` | Squeeze: `{'ON' if latest['Squeeze_On'] else 'OFF'}`\n"
                f"Time: `{ts}`"
            )
            
        elif mode == "📑 Institutional Memo":
            return (
                f"📑 *INSTITUTIONAL MEMO: {ticker}*\n"
                f"───────────────────────\n"
                f"**TECHNICAL PROFILE**\n"
                f"• Price:   `${latest['Close']:,.2f}`\n"
                f"• Trend:   `{'BULL' if latest['Apex_Trend']==1 else 'BEAR'}` (HMA: ${latest['HMA']:,.2f})\n"
                f"• Entropy: `{latest['CHEDO']:.3f}` (Chaos Metric)\n"
                f"• VP POC:  `${poc:,.2f}` (Volume Center)\n\n"
                f"**RISK METRICS**\n"
                f"• Monte Carlo (30d): Exp `${mc_mean:,.2f}` | Risk `${mc_low:,.2f}`\n"
                f"• Risk-On Ratio: `{spy_tlt:.2f}`\n"
                f"───────────────────────\n"
                f"Generated: {ts}"
            )
            
        elif mode == "🧠 Quantum Deep Dive":
            return (
                f"🧠 *QUANTUM DEEP DIVE: {ticker}*\n\n"
                f"*1. PHYSICS ENGINE*\n"
                f"Asset is in a {'Superconductor' if abs(latest['Flux']) > 0.5 else 'Resistive'} state (Flux: {latest['Flux']:.2f}). "
                f"Relativistic time-dilation (RQZO) is at {latest['RQZO']:.2f}.\n\n"
                f"*2. GOD MODE CONFLUENCE*\n"
                f"Titan Score: {latest['Titan_Score']}/4. "
                f"Components: Trend={latest['Apex_Trend']}, Gann={latest['Gann_Trend']}, Flux={latest['Flux']:.1f}.\n\n"
                f"*3. PREDICTIVE MODELS*\n"
                f"Monte Carlo simulations project a mean price of ${mc_mean:,.2f} in 30 days.\n\n"
                f"_AI Synthesis to be appended below..._"
            )
            
    @staticmethod
    def transmit(payload):
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        max_len = 3000
        try:
            if len(payload) <= max_len:
                requests.post(url, json={"chat_id": TG_CHAT_ID, "text": payload, "parse_mode": "Markdown"})
            else:
                parts = [payload[i:i+max_len] for i in range(0, len(payload), max_len)]
                for i, part in enumerate(parts):
                    header = f"*(Part {i+1}/{len(parts)})*\n" if i > 0 else ""
                    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": header + part, "parse_mode": "Markdown"})
            return True
        except Exception as e:
            return str(e)

# ==========================================
# 6. UI LAYOUT & OMNI-LINK
# ==========================================
st.sidebar.title("💠 Axiom Omni-Link")

# A. OMNI-LINK ASSET MATRIX
asset_map = AssetRepository.get_assets()
category = st.sidebar.selectbox("1. Asset Class", list(asset_map.keys()))
selected_ticker = st.sidebar.selectbox("2. Select Instrument", asset_map[category])

# Manual Override
manual_ticker = st.sidebar.text_input("OR Manual Ticker", "")
if manual_ticker:
    selected_ticker = manual_ticker.upper()

# Update Session State
st.session_state['selected_ticker'] = selected_ticker

# Timeframe
interval = st.sidebar.selectbox("3. Interval", ["1h", "4h", "1d"], index=2)

# B. DYNAMIC TRADINGVIEW INTEGRATION (Updates with Dropdown)
st.sidebar.markdown("---")
tv_code = f"""
<div class="tradingview-widget-container">
  <div id="tradingview_chart"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget(
  {{
  "width": "100%",
  "height": 350,
  "symbol": "{selected_ticker}",
  "interval": "D",
  "timezone": "Etc/UTC",
  "theme": "dark",
  "style": "1",
  "locale": "en",
  "toolbar_bg": "#f1f3f6",
  "enable_publishing": false,
  "hide_side_toolbar": false,
  "allow_symbol_change": true,
  "container_id": "tradingview_chart"
  }}
  );
  </script>
</div>
"""
with st.sidebar:
    st.components.v1.html(tv_code, height=350)

# C. MAIN BROADCAST INTERFACE
st.markdown(f"## 📡 Command Center: {selected_ticker}")

col_cmd, col_rev = st.columns([1, 2])

with col_cmd:
    st.markdown("### Generate")
    b_mode = st.radio("Signal Mode", ["⚡ Flash Signal", "📑 Institutional Memo", "🧠 Quantum Deep Dive"])
    
    if st.button("🔄 Analyze & Draft"):
        with st.spinner(f"Running 33-Point Matrix on {selected_ticker}..."):
            df = get_clean_data(selected_ticker, "1y", interval)
            macro = get_macro_data()
            
            if not df.empty and len(df) > 50:
                df = UniversalParamountEngine.calculate_all(df)
                latest = df.iloc[-1].to_dict()
                latest['Close'] = float(df['Close'].iloc[-1])
                poc_data, poc = UniversalParamountEngine.calculate_vp_poc(df)
                mc_mean, mc_low = UniversalParamountEngine.run_monte_carlo_stats(df)
                
                draft = BroadcastEngine.generate_content(b_mode, selected_ticker, latest, macro, poc, mc_mean, mc_low)
                
                if b_mode == "🧠 Quantum Deep Dive":
                    client = OpenAI(api_key=OPENAI_API_KEY)
                    ai = client.chat.completions.create(
                        model="gpt-4o", messages=[{"role": "user", "content": f"Analyze: {latest}"}]
                    ).choices[0].message.content
                    draft += f"\n\n*AI SYNTHESIS:*\n{ai}"
                
                st.session_state['report_draft'] = draft
                st.session_state['last_run_dt'] = (df, poc_data) # Store for charts
            else:
                st.error("Data Unavailable or Insufficient History")

with col_rev:
    st.markdown("### Review & Dispatch")
    final_payload = st.text_area("Message Preview", value=st.session_state['report_draft'], height=350)
    
    if st.button("🚀 TRANSMIT SIGNAL", type="primary"):
        if final_payload:
            res = BroadcastEngine.transmit(final_payload)
            if res is True: st.success("✅ Signal Dispatched!")
            else: st.error(f"❌ Error: {res}")
        else: st.warning("Draft Empty")

# D. VISUAL VERIFICATION
if st.session_state['last_run_dt'] is not None:
    df, vp_data = st.session_state['last_run_dt']
    st.markdown("---")
    st.markdown("### 📊 Visual Verification Protocols")
    
    t1, t2, t3 = st.tabs(["God Mode", "Quantum Physics", "Volume Profile"])
    
    with t1:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='#00F0FF'), name="HMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(color='rgba(0,255,0,0.3)'), name="Apex Up"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', line=dict(color='rgba(255,0,0,0.3)'), name="Apex Down"), row=1, col=1)
        cols = ['#00E676' if v > 0 else '#FF1744' for v in df['Sqz_Mom']]
        fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=cols, name="Squeeze"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Flux'], line=dict(color='#D500F9'), name="Flux"), row=3, col=1)
        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig2.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], fill='tozeroy', name="Entropy"), row=1, col=1)
        fig2.add_trace(go.Scatter(x=df.index, y=df['RQZO'], name="Relativity"), row=2, col=1)
        fig2.update_layout(height=500, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        st.bar_chart(vp_data.set_index('Price_Level')['Volume'])
