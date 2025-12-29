import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime
import requests
import urllib.parse
import time

# ==========================================
# 1. SYSTEM CONFIGURATION & SECRETS
# ==========================================
st.set_page_config(layout="wide", page_title="AXIOM ZENITH TERMINAL", page_icon="💠")

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    TG_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    TG_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
except Exception:
    st.error("STRICT STOP: Missing API credentials (OPENAI_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID) in secrets.")
    st.stop()

# Session State Init
if 'report_draft' not in st.session_state: st.session_state['report_draft'] = ""
if 'last_run_dt' not in st.session_state: st.session_state['last_run_dt'] = None
if 'selected_ticker' not in st.session_state: st.session_state['selected_ticker'] = "BTC-USD"
if 'broadcast_log' not in st.session_state: st.session_state['broadcast_log'] = []

# ==========================================
# 2. OMNI-LINK ASSET REPOSITORY
# ==========================================
class AssetRepository:
    @staticmethod
    def get_assets():
        return {
            "CRYPTO (Major)": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD"],
            "CRYPTO (DeFi/L2)": ["UNI-USD", "MATIC-USD", "LDO-USD", "AAVE-USD", "MKR-USD", "OP-USD", "ARB-USD"],
            "CRYPTO (AI/Meme)": ["FET-USD", "RNDR-USD", "AGIX-USD", "SHIB-USD", "PEPE-USD", "WIF-USD", "BONK-USD"],
            "US STOCKS (Mag 7)": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "US STOCKS (Tech)": ["AMD", "INTC", "CRM", "PLTR", "UBER", "COIN", "MSTR", "MARA", "SMCI"],
            "INDICES": ["SPY", "QQQ", "DIA", "IWM", "^VIX", "TLT"],
            "FOREX": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"],
            "COMMODITIES": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"]
        }

# ==========================================
# 3. ROBUST DATA ENGINE (MultiIndex Safe)
# ==========================================
@st.cache_data(ttl=300)
def get_clean_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_macro_data():
    tickers = {"SPY": "SPY", "TLT": "TLT", "BTC": "BTC-USD", "ETH": "ETH-USD"}
    try:
        data = yf.download(list(tickers.values()), period="5d", interval="1d", progress=False)['Close']
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except: return pd.DataFrame()

# ==========================================
# 4. UNIVERSAL LOGIC ENGINE (33 Indicators)
# ==========================================
class UniversalParamountEngine:
    @staticmethod
    def calculate_all(df):
        # A. Math Helpers
        def wma(s, l): return s.rolling(l).apply(lambda x: np.dot(x, np.arange(1, l + 1)) / np.arange(1, l + 1).sum(), raw=True)
        def hma(s, l): return wma(2 * wma(s, int(l/2)) - wma(s, l), int(np.sqrt(l)))
        def atr(df, l): 
            tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
            return tr.rolling(l).mean()

        # B. Trend & Risk
        df['HMA'] = hma(df['Close'], 55)
        df['ATR'] = atr(df, 14)
        df['Apex_Upper'] = df['HMA'] + (atr(df, 55) * 1.5)
        df['Apex_Lower'] = df['HMA'] - (atr(df, 55) * 1.5)
        df['Apex_Trend'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, 0))

        # C. Squeeze Momentum
        basis = df['Close'].rolling(20).mean()
        dev = df['Close'].rolling(20).std() * 2.0
        lower_bb = basis - dev; upper_bb = basis + dev
        kc_range = atr(df, 20)
        lower_kc = basis - (kc_range * 1.5); upper_kc = basis + (kc_range * 1.5)
        df['Squeeze_On'] = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        df['Sqz_Mom'] = (df['Close'] - ((df['High'].rolling(20).max() + df['Low'].rolling(20).min() + basis)/3)).rolling(20).mean()

        # D. Physics (Entropy & Flux)
        log_ret = np.diff(np.log(df['Close'].values), prepend=np.log(df['Close'].iloc[0]))
        mu = pd.Series(log_ret).rolling(50).mean(); sigma = pd.Series(log_ret).rolling(50).std()
        v = sigma / (np.abs(mu) + 1e-9)
        hyper_dist = np.log(np.abs(log_ret) * v + np.sqrt((np.abs(log_ret) * v)**2 + 1))
        df['CHEDO'] = np.tanh(pd.Series(hyper_dist).rolling(50).mean().values * 4)

        rg = df['High'] - df['Low']; body = np.abs(df['Close'] - df['Open'])
        eff = (body / rg.replace(0, 1)).ewm(span=14).mean()
        v_rat = df['Volume'] / df['Volume'].rolling(55).mean()
        df['Flux'] = (np.sign(df['Close'] - df['Open']) * eff * v_rat).ewm(span=5).mean()

        # E. RQZO (Relativity)
        mn, mx = df['Close'].rolling(100).min(), df['Close'].rolling(100).max()
        norm = (df['Close'] - mn) / (mx - mn + 1e-9)
        gamma = 1 / np.sqrt(1 - (np.clip(np.abs(norm.diff()), 0, 0.049)/0.05)**2)
        tau = (np.arange(len(df)) % 100) / gamma.fillna(1.0)
        zeta = np.zeros(len(df))
        for n in range(1, 26): zeta += (n**-0.5) * np.sin(tau * np.log(n))
        df['RQZO'] = pd.Series(zeta).fillna(0)

        # F. Money Flow & Gann
        diff = df['Close'].diff()
        gain = diff.clip(lower=0).rolling(14).mean()
        loss = diff.clip(upper=0).abs().rolling(14).mean()
        rsi_norm = (100 - (100 / (1 + (gain / loss)))) - 50
        mf_vol = df['Volume'] / df['Volume'].rolling(14).mean()
        df['MF_Matrix'] = (rsi_norm * mf_vol).ewm(span=3).mean()
        
        df['Gann_High'] = df['High'].rolling(3).mean()
        df['Gann_Low'] = df['Low'].rolling(3).mean()
        df['Gann_Trend'] = np.where(df['Close'] > df['Gann_High'].shift(1), 1, np.where(df['Close'] < df['Gann_Low'].shift(1), -1, 0))

        # G. Titan Score
        df['Titan_Score'] = np.sign(df['Apex_Trend']) + np.sign(df['Gann_Trend']) + np.sign(df['Flux']) + np.sign(df['Sqz_Mom'])
        return df

    @staticmethod
    def calculate_risk_levels(df):
        last = df.iloc[-1]
        atr = last['ATR']
        score = last['Titan_Score']
        direction = "LONG" if score > 0 else "SHORT" if score < 0 else "NEUTRAL"
        price = last['Close']
        
        levels = {"Action": direction, "Entry": price, "SL": 0.0, "TS": 0.0, "TP1": 0.0, "TP2": 0.0, "TP3": 0.0}
        
        if direction == "LONG":
            levels.update({
                "SL": price - (atr * 2.0),
                "TS": df['High'].rolling(20).max().iloc[-1] - (atr * 3.0),
                "TP1": price + (atr * 1.5), "TP2": price + (atr * 2.5), "TP3": price + (atr * 4.0)
            })
        elif direction == "SHORT":
            levels.update({
                "SL": price + (atr * 2.0),
                "TS": df['Low'].rolling(20).min().iloc[-1] + (atr * 3.0),
                "TP1": price - (atr * 1.5), "TP2": price - (atr * 2.5), "TP3": price - (atr * 4.0)
            })
        return levels

    @staticmethod
    def calculate_vp_poc(df):
        price_bins = np.linspace(df['Low'].min(), df['High'].max(), 50)
        df['Bin'] = pd.cut(df['Close'], bins=price_bins, include_lowest=True)
        vp = df.groupby('Bin', observed=False)['Volume'].sum().reset_index()
        poc = vp.loc[vp['Volume'].idxmax()]['Bin'].mid
        vp['Price_Level'] = vp['Bin'].apply(lambda x: x.mid)
        return vp, poc

    @staticmethod
    def run_monte_carlo(df):
        returns = df['Close'].pct_change().dropna()
        sim_rets = np.random.normal(returns.mean(), returns.std(), (30, 100))
        paths = np.zeros((30, 100))
        paths[0] = df['Close'].iloc[-1]
        for t in range(1, 30): paths[t] = paths[t-1] * (1 + sim_rets[t])
        return np.mean(paths[-1]), np.percentile(paths[-1], 5)

# ==========================================
# 5. OMNI-BROADCAST ENGINE (Safety Chunking Enabled)
# ==========================================
class BroadcastEngine:
    @staticmethod
    def generate_content(mode, ticker, latest, macro, risk, poc, mc_mean, mc_low, ai_text=""):
        ts = datetime.now().strftime("%H:%M UTC")
        emoji = "🟢 BUY" if risk['Action'] == "LONG" else ("🔴 SELL" if risk['Action'] == "SHORT" else "⚪ WAIT")
        
        # Macro Ratios
        btc_spx = 0.0
        if not macro.empty and 'BTC-USD' in macro.columns: btc_spx = macro['BTC-USD'].iloc[-1] / macro['SPY'].iloc[-1]

        # 1. FLASH SIGNAL
        if mode == "⚡ Flash Signal":
            return (
                f"⚡ *FLASH: {ticker}*\n"
                f"Bias: *{emoji}*\n"
                f"Entry: `${risk['Entry']:,.2f}`\n"
                f"Stop: `${risk['SL']:,.2f}`\n"
                f"TP1: `${risk['TP1']:,.2f}`\n"
                f"Score: `{latest['Titan_Score']:.0f}/4`\n"
                f"Flux: `{latest['Flux']:.2f}`\n"
                f"Time: `{ts}`"
            )
        
        # 2. INSTITUTIONAL MEMO
        elif mode == "📑 Institutional Memo":
            return (
                f"📑 *AXIOM MEMO: {ticker}*\n"
                f"────────────────\n"
                f"**EXECUTION ({risk['Action']})**\n"
                f"• Entry: `${risk['Entry']:,.2f}`\n"
                f"• Stop:  `${risk['SL']:,.2f}`\n"
                f"• Trail: `${risk['TS']:,.2f}`\n"
                f"• Targets: `${risk['TP1']:,.0f}` -> `${risk['TP3']:,.0f}`\n\n"
                f"**DEEP METRICS**\n"
                f"• Trend: `{'BULL' if latest['Apex_Trend']==1 else 'BEAR'}`\n"
                f"• Entropy: `{latest['CHEDO']:.2f}`\n"
                f"• POC: `${poc:,.2f}`\n"
                f"• Beta: `{btc_spx:.3f}`\n"
                f"────────────────\n"
                f"{ts}"
            )
        
        # 3. EDUCATIONAL / DEEP DIVE (Added based on Source Files)
        elif mode == "🎓 Educational Breakdwon":
            return (
                f"🎓 *EDUCATION: {ticker} Setup*\n\n"
                f"*WHY THIS SIGNAL?*\n"
                f"1. **Trend Architecture**: Price is {'above' if latest['Apex_Trend']==1 else 'below'} the HMA(55) & Apex Bands.\n"
                f"2. **Vector Physics**: Flux is {latest['Flux']:.2f}, indicating {'efficient' if abs(latest['Flux'])>0.5 else 'choppy'} price delivery.\n"
                f"3. **Entropy State**: CHEDO is {latest['CHEDO']:.2f}. High values (>0.7) usually precede a reversion.\n\n"
                f"*AI THESIS*\n{ai_text}\n\n"
                f"_This is for educational purposes only._"
            )

    @staticmethod
    def transmit_telegram(payload, image_file=None):
        """
        Includes the Safety Chunking Logic from Terminal-V1-2.py
        to prevent 4096 char limit errors.
        """
        url_msg = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        url_photo = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
        
        try:
            # Step 1: Image (Priority)
            if image_file:
                image_file.seek(0)
                files = {'photo': image_file.getvalue()}
                # Caption max length is 1024
                caption = payload[:1024]
                data = {'chat_id': TG_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
                requests.post(url_photo, files=files, data=data)
                
                # If payload > 1024, send rest as text
                if len(payload) > 1024:
                    payload = payload[1024:]
                else:
                    return True # Done

            # Step 2: Text Chunking (Terminal V1.2 Logic)
            max_length = 3000 # Safe margin below 4096
            if len(payload) <= max_length:
                requests.post(url_msg, json={"chat_id": TG_CHAT_ID, "text": payload, "parse_mode": "Markdown"})
            else:
                for i in range(0, len(payload), max_length):
                    chunk = payload[i:i+max_length]
                    header = f"*(Part {i//max_length + 1})* " if i > 0 else ""
                    requests.post(url_msg, json={"chat_id": TG_CHAT_ID, "text": header + chunk, "parse_mode": "Markdown"})
            
            return True
        except Exception as e:
            return str(e)

    @staticmethod
    def generate_x_link(payload, ticker):
        """Generates Intent Tweet with Cashtags and Hashtags"""
        clean = payload.replace("*", "").replace("`", "")
        # Smart Hashtags
        tags = f"\n\n${ticker.split('-')[0]} #Trading #Crypto #Stocks #Axiom"
        final = clean[:250] + "..." + tags # Truncate for X limit safety
        encoded = urllib.parse.quote(final)
        return f"https://twitter.com/intent/tweet?text={encoded}"

# ==========================================
# 6. UI LAYOUT & OMNI-LINK
# ==========================================
# A. SIDEBAR CONTROL
st.sidebar.title("💠 Zenith Omni-Link")
asset_map = AssetRepository.get_assets()
category = st.sidebar.selectbox("1. Asset Class", list(asset_map.keys()))
selected_ticker = st.sidebar.selectbox("2. Instrument", asset_map[category])
manual = st.sidebar.text_input("Manual Ticker", "")
if manual: selected_ticker = manual.upper()
st.session_state['selected_ticker'] = selected_ticker
interval = st.sidebar.selectbox("3. Interval", ["1h", "4h", "1d"], index=2)

# B. MAINFRAME TRADINGVIEW
st.markdown(f"## 💠 Axiom Zenith: {selected_ticker}")
tv_code = f"""
<div class="tradingview-widget-container" style="height:500px;width:100%">
  <div id="tradingview_chart" style="height:calc(100% - 32px);width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({{
  "autosize": true, "symbol": "{selected_ticker}", "interval": "D", "timezone": "Etc/UTC", "theme": "dark",
  "style": "1", "locale": "en", "toolbar_bg": "#f1f3f6", "enable_publishing": false, "hide_side_toolbar": false,
  "allow_symbol_change": true, "container_id": "tradingview_chart"
  }});
  </script>
</div>
"""
st.components.v1.html(tv_code, height=500)

# C. ANALYSIS ENGINE
if st.button("🔄 INITIATE QUANTUM ANALYSIS", type="primary"):
    with st.spinner("Calculating 33-Vector Matrix..."):
        df = get_clean_data(selected_ticker, "1y", interval)
        macro = get_macro_data()
        
        if not df.empty and len(df) > 50:
            df = UniversalParamountEngine.calculate_all(df)
            latest = df.iloc[-1].to_dict()
            latest['Close'] = float(df['Close'].iloc[-1])
            risk = UniversalParamountEngine.calculate_risk_levels(df)
            poc_data, poc = UniversalParamountEngine.calculate_vp_poc(df)
            mc_mean, mc_low = UniversalParamountEngine.run_monte_carlo(df)
            
            # AI Analyst (Context Injection)
            client = OpenAI(api_key=OPENAI_API_KEY)
            ai_text = client.chat.completions.create(
                model="gpt-4o", 
                messages=[{"role": "system", "content": "You are a Quant Architect. Explain the physics of this trade."},
                          {"role": "user", "content": f"Analyze: {latest}"}]
            ).choices[0].message.content
            
            st.session_state['analysis_data'] = {
                "latest": latest, "macro": macro, "risk": risk, 
                "poc": poc, "mc_mean": mc_mean, "mc_low": mc_low, "ai": ai_text
            }
            st.session_state['last_run_dt'] = (df, poc_data)
        else:
            st.error("Data Error: Insufficient history for calculations.")

# D. TABS: CHARTS, PHYSICS, BROADCAST
t1, t2, t3 = st.tabs(["📊 God Mode Charts", "⚛️ Quantum Physics", "📡 Omni-Broadcast Center"])

if st.session_state['last_run_dt'] is not None:
    df, vp_data = st.session_state['last_run_dt']
    data = st.session_state['analysis_data']
    
    with t1:
        # VISUAL CONFIRMATION WITH RISK LEVELS
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='#00F0FF'), name="HMA"), row=1, col=1)
        
        # Risk Lines Overlay (Visual Verification)
        r = data['risk']
        if r['Action'] != "NEUTRAL":
            fig.add_hline(y=r['SL'], line_dash="dash", line_color="red", annotation_text="STOP", row=1, col=1)
            fig.add_hline(y=r['TP1'], line_dash="dot", line_color="#00E676", annotation_text="TP1", row=1, col=1)
            fig.add_hline(y=r['TP3'], line_dash="dot", line_color="#00E676", annotation_text="TP3", row=1, col=1)

        cols = ['#00E676' if v > 0 else '#FF1744' for v in df['Sqz_Mom']]
        fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=cols, name="Squeeze"), row=2, col=1)
        fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Entropy & Relativity**")
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True)
            fig2.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], fill='tozeroy', name="CHEDO"), row=1, col=1)
            fig2.add_trace(go.Scatter(x=df.index, y=df['RQZO'], name="RQZO"), row=2, col=1)
            fig2.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            st.markdown(f"**Volume Profile (POC: ${data['poc']:,.2f})**")
            st.bar_chart(vp_data.set_index('Price_Level')['Volume'])

    with t3:
        st.markdown("### 📡 Omni-Broadcast Command")
        
        # 1. Configuration
        c_mode, c_img = st.columns([1, 1])
        with c_mode:
            b_mode = st.radio("Protocol", ["⚡ Flash Signal", "📑 Institutional Memo", "🎓 Educational Breakdwon"])
            # Auto-Regenerate Draft on Mode Change
            draft = BroadcastEngine.generate_content(
                b_mode, selected_ticker, data['latest'], data['macro'], 
                data['risk'], data['poc'], data['mc_mean'], data['mc_low'], data['ai']
            )
        with c_img:
            uploaded_file = st.file_uploader("Attach Chart Image (Optional)", type=['png', 'jpg', 'jpeg'])

        # 2. Review & Edit
        final_msg = st.text_area("Review Payload (Editable)", value=draft, height=350)
        
        # 3. Dispatch Matrix
        st.markdown("---")
        d_col1, d_col2 = st.columns(2)
        
        with d_col1:
            if st.button("🚀 TRANSMIT TO TELEGRAM", use_container_width=True):
                res = BroadcastEngine.transmit_telegram(final_msg, uploaded_file)
                if res is True: 
                    st.success("Telegram Broadcast Successful")
                    st.session_state['broadcast_log'].append(f"{datetime.now()}: TG sent for {selected_ticker}")
                else: 
                    st.error(f"Failed: {res}")
        
        with d_col2:
            x_url = BroadcastEngine.generate_x_link(final_msg, selected_ticker)
            st.link_button("🐦 LAUNCH X (TWITTER) POST", x_url, use_container_width=True)
        
        # 4. History Log
        with st.expander("Broadcast History"):
            for log in st.session_state['broadcast_log']:
                st.text(log)
