
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
from typing import Tuple, Dict, Any, Optional, List

# ==========================================
# 1. CONFIGURATION & ASSETS
# ==========================================
class AxiomConfig:
    PAGE_TITLE = "Axiom Titan v3.2 (Production)"
    PAGE_ICON = "💠"
    LAYOUT = "wide"
    
    # Asset Universe
    ASSETS = {
        "Indices & Macro": ["SPY", "QQQ", "IWM", "TLT", "VXX", "^TNX", "DX-Y.NYB", "^VIX"],
        "Crypto (Major)": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD"],
        "Crypto (Alt/Meme)": ["SHIB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "ARB-USD", "RNDR-USD", "LINK-USD"],
        "US Tech (Mag 7+)": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX"],
        "Commodities": ["GLD", "SLV", "USO", "CL=F", "GC=F", "NG=F", "HG=F"]
    }

    # Visual Palette (Pine Script Match)
    COLORS = {
        'bull_trend': '#00695C', 'bear_trend': '#B71C1C',
        'bull_sig': '#00E676', 'bear_sig': '#FF1744',
        'smc_bull': 'rgba(185, 246, 202, 0.4)', # Pale Mint
        'smc_bear': 'rgba(255, 205, 210, 0.4)',  # Pale Rose
        'bos': '#FFFFFF', 'text': '#E0E0E0'
    }

    @staticmethod
    def get_secret(key: str) -> str:
        try: return st.secrets.get(key, "")
        except: return ""

# ==========================================
# 2. UX FACTORY
# ==========================================
class UXFactory:
    @staticmethod
    def inject_css(is_mobile: bool):
        base_css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=SF+Pro+Display:wght@300;500;700&display=swap');
            .stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'SF Pro Display', monospace; }
            .title-glow { font-size: 2.2em; font-weight: bold; color: #ffffff; text-shadow: 0 0 10px #00ff00; margin-bottom: 5px; }
            div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.05); border-left: 3px solid #00ff00; padding: 10px; border-radius: 4px; }
            div[data-testid="stMetricValue"] { color: #fff; font-weight: 700; font-size: 1.3rem !important; }
            .report-card { background-color: #161b22; border-left: 4px solid #00F0FF; padding: 15px; border-radius: 6px; margin-bottom: 10px; font-family: 'SF Pro Display', sans-serif; }
            .report-header { font-size: 1.1rem; font-weight: bold; color: #fff; margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 5px; }
            .report-item { margin-bottom: 5px; font-size: 0.9rem; color: #aaa; }
            .highlight { color: #00F0FF; font-weight: bold; }
            .ticker-wrap { width: 100%; overflow: hidden; background-color: #0d1117; border-bottom: 1px solid #30363d; height: 30px; display: flex; align-items: center; }
            .ticker { display: inline-block; animation: marquee 60s linear infinite; white-space: nowrap; }
            @keyframes marquee { 0% { transform: translate(100%, 0); } 100% { transform: translate(-100%, 0); } }
        </style>
        """
        mobile_css = """<style>div[data-testid="stMetric"] { margin-bottom: 8px; padding: 15px; } .js-plotly-plot { height: 100% !important; } button { min-height: 50px !important; margin-top: 10px !important; }</style>"""
        st.markdown(base_css, unsafe_allow_html=True)
        if is_mobile: st.markdown(mobile_css, unsafe_allow_html=True)

    @staticmethod
    def render_header(asset_class: str):
        items = "BITCOIN: ACTIVE • ETH: ACTIVE • SOL: VOLATILE" if "Crypto" in asset_class else "SPY: LIQUID • VIX: WATCH • YIELDS: STABLE"
        st.markdown(f"""
        <div class="title-glow">👁️ Axiom Titan v3.2</div>
        <div class="ticker-wrap"><div class="ticker">
            <span style="padding:0 2rem; color:#00ff00;">SYSTEM: ONLINE</span>
            <span style="padding:0 2rem; color:#00ff00;">PHYSICS: CALCULATING</span>
            <span style="padding:0 2rem; color:#00F0FF;">{items}</span>
        </div></div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_world_clock():
        html = """
        <div style="display: flex; gap: 20px; font-family: 'Roboto Mono'; color: #888; font-size: 0.8rem; background: rgba(0,0,0,0.2); padding: 5px;">
            <div>NYC <span id="nyc" style="color: #fff;">--:--</span></div>
            <div>LON <span id="lon" style="color: #fff;">--:--</span></div>
            <div>TOK <span id="tok" style="color: #fff;">--:--</span></div>
        </div>
        <script>
        function u(){ const d=new Date(); const f=(t)=>d.toLocaleTimeString('en-US',{timeZone:t,hour:'2-digit',minute:'2-digit',hour12:false}); 
        document.getElementById('nyc').innerText=f('America/New_York'); document.getElementById('lon').innerText=f('Europe/London'); document.getElementById('tok').innerText=f('Asia/Tokyo'); }
        setInterval(u,1000); u();
        </script>
        """
        components.html(html, height=40)

    @staticmethod
    def render_tradingview_widget(ticker: str, interval: str):
        tv_int = "240" if interval == "4h" else ("60" if interval == "1h" else "D")
        tv_sym = ticker.replace("-", "").replace("=X", "")
        if "BTC" in tv_sym: tv_sym = "BINANCE:BTCUSDT"
        elif "SPY" in tv_sym: tv_sym = "AMEX:SPY"
        
        html = f"""
        <div class="tradingview-widget-container"><div id="tradingview_widget"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({{"width": "100%", "height": 550, "symbol": "{tv_sym}", "interval": "{tv_int}", "theme": "dark", "style": "1", "container_id": "tradingview_widget"}});
        </script></div>
        """
        components.html(html, height=550)

# ==========================================
# 3. DATA CORE
# ==========================================
class DataCore:
    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_data(ticker: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        tf_map = {"15m": "60d", "1h": "730d", "4h": "730d", "1d": "5y"}
        fallback_map = {"15m": "5d", "1h": "60d", "4h": "60d", "1d": "1y"}
        
        try:
            df = yf.download(ticker, period=tf_map.get(timeframe, "1y"), interval="1h" if timeframe=="4h" else timeframe, progress=False, threads=False)
            if df.empty:
                df = yf.download(ticker, period=fallback_map.get(timeframe, "1mo"), interval="1h" if timeframe=="4h" else timeframe, progress=False, threads=False)
            
            if isinstance(df.columns, pd.MultiIndex): df = df.xs(ticker, axis=1, level=0) if ticker in df.columns else df
            df.columns = [c.capitalize() for c in df.columns]
            if 'Adj close' in df.columns: df['Close'] = df['Adj close']
            
            if timeframe == "4h":
                df = df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            
            return df.tail(limit)
        except Exception as e:
            st.error(f"Data Error: {e}"); return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=300)
    def get_macro_data():
        assets = {"S&P 500": "SPY", "VIX": "^VIX", "DXY": "DX-Y.NYB"}
        try:
            data = yf.download(list(assets.values()), period="5d", progress=False)['Close']
            prices = {k: data[v].iloc[-1] for k,v in assets.items() if v in data}
            return prices
        except: return {}

# ==========================================
# 4. QUANT ENGINE (PHYSICS + SMC)
# ==========================================
class QuantEngine:
    @staticmethod
    def tanh(x): return np.tanh(np.clip(x, -20, 20))

    @staticmethod
    def calculate_wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def calculate_hma(series, length):
        half = int(length / 2)
        sqrt = int(np.sqrt(length))
        wma_f = QuantEngine.calculate_wma(series, length)
        wma_h = QuantEngine.calculate_wma(series, half)
        return QuantEngine.calculate_wma(2 * wma_h - wma_f, sqrt)

    @staticmethod
    def calculate_atr(df, length=14):
        h, l, c = df['High'], df['Low'], df['Close']
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(length).mean()

    # --- PHYSICS (ENTROPY & FLUX) ---
    @staticmethod
    def calc_physics(df: pd.DataFrame) -> pd.DataFrame:
        c = df['Close'].values
        log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
        ent = pd.Series(log_ret**2).rolling(50).sum().values
        ent_n = QuantEngine.tanh(ent * 10)
        df['CHEDO'] = 2 / (1 + np.exp(-ent_n * 4)) - 1
        
        rg = df['High'] - df['Low']
        body = (df['Close'] - df['Open']).abs()
        eff = np.where(rg==0, 0, body/rg)
        df['Apex_Flux'] = pd.Series(np.sign(df['Close']-df['Open']) * eff).ewm(span=5).mean()
        return df

    # --- SMC & STRATEGY ENGINE ---
    @staticmethod
    def calc_apex_smc_strategy(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Integrates Trend (HMA), Momentum (WaveTrend), and Market Structure (SMC).
        """
        # 1. Trend & Volatility
        df['HMA_Trend'] = QuantEngine.calculate_hma(df['Close'], 55)
        df['ATR'] = QuantEngine.calculate_atr(df, 55)
        mult = 1.5
        df['Trend_Upper'] = df['HMA_Trend'] + (df['ATR'] * mult)
        df['Trend_Lower'] = df['HMA_Trend'] - (df['ATR'] * mult)
        
        # Trend State
        df['Trend'] = np.where(df['Close'] > df['Trend_Upper'], 1, np.where(df['Close'] < df['Trend_Lower'], -1, 0))
        df['Trend'] = df['Trend'].replace(0, method='ffill')

        # Momentum: ADX & WaveTrend
        up = df['High'] - df['High'].shift(1)
        down = df['Low'].shift(1) - df['Low']
        pos_dm = np.where((up > down) & (up > 0), up, 0)
        neg_dm = np.where((down > up) & (down > 0), down, 0)
        tr = QuantEngine.calculate_atr(df, 14)
        di_plus = 100 * pd.Series(pos_dm).ewm(alpha=1/14).mean() / tr
        di_minus = 100 * pd.Series(neg_dm).ewm(alpha=1/14).mean() / tr
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        df['ADX'] = dx.ewm(alpha=1/14).mean()

        # WaveTrend
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=10).mean()
        d = (ap - esa).abs().ewm(span=10).mean()
        ci = (ap - esa) / (0.015 * d)
        tci = ci.ewm(span=21).mean()
        df['WaveTrend'] = tci
        
        # Signals
        df['Vol_Avg'] = df['Volume'].rolling(20).mean()
        df['Sig_Buy'] = (df['Trend'] == 1) & (df['Trend'].shift(1) != 1) & (df['Volume'] > df['Vol_Avg']) & (tci < 60) & (df['ADX'] > 20)
        df['Sig_Sell'] = (df['Trend'] == -1) & (df['Trend'].shift(1) != -1) & (df['Volume'] > df['Vol_Avg']) & (tci > -60) & (df['ADX'] > 20)

        # 2. SMC Structure (Pivots, BOS, FVG, OB)
        smc_data = {'bos': [], 'ob': [], 'fvg': []}
        
        # Pivot High/Low
        piv_len = 5
        df['Pivot_High'] = df['High'].rolling(window=piv_len*2+1, center=True).max() == df['High']
        df['Pivot_Low'] = df['Low'].rolling(window=piv_len*2+1, center=True).min() == df['Low']
        
        last_ph = None
        last_pl = None
        
        # BOS Logic
        for i in range(max(0, len(df)-200), len(df)):
            curr_close = df['Close'].iloc[i]
            curr_trend = df['Trend'].iloc[i]
            
            pivot_idx = i - piv_len
            if pivot_idx >= 0:
                if df['Pivot_High'].iloc[pivot_idx]: last_ph = df['High'].iloc[pivot_idx]
                if df['Pivot_Low'].iloc[pivot_idx]: last_pl = df['Low'].iloc[pivot_idx]

            # Break of Structure
            if curr_trend == 1 and last_ph and curr_close > last_ph:
                if not smc_data['bos'] or (i - smc_data['bos'][-1]['idx'] > 5):
                    smc_data['bos'].append({'idx': i, 'price': last_ph, 'type': 'bull', 'date': df.index[i]})
                    last_ph = None 

            if curr_trend == -1 and last_pl and curr_close < last_pl:
                if not smc_data['bos'] or (i - smc_data['bos'][-1]['idx'] > 5):
                    smc_data['bos'].append({'idx': i, 'price': last_pl, 'type': 'bear', 'date': df.index[i]})
                    last_pl = None

        # FVG & Order Blocks
        fvg_bull = (df['Low'] > df['High'].shift(2)) & (df['Close'] > df['Open'])
        fvg_bear = (df['High'] < df['Low'].shift(2)) & (df['Close'] < df['Open'])
        
        swing_lookback = 10
        for i in range(len(df)-100, len(df)):
            if i < 2: continue
            # FVG
            if fvg_bull.iloc[i]:
                smc_data['fvg'].append({'type': 'bull', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'x0': df.index[i-2], 'x1': df.index[i]})
            if fvg_bear.iloc[i]:
                smc_data['fvg'].append({'type': 'bear', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'x0': df.index[i-2], 'x1': df.index[i]})
            
            # Order Blocks (Impulse Origin)
            if df['Sig_Buy'].iloc[i]:
                subset = df.iloc[i-swing_lookback:i]
                reds = subset[subset['Close'] < subset['Open']]
                if not reds.empty:
                    ob = reds.iloc[-1]
                    smc_data['ob'].append({'type': 'bull', 'top': ob['High'], 'bottom': ob['Low'], 'x': ob.name})
            
            if df['Sig_Sell'].iloc[i]:
                subset = df.iloc[i-swing_lookback:i]
                greens = subset[subset['Close'] > subset['Open']]
                if not greens.empty:
                    ob = greens.iloc[-1]
                    smc_data['ob'].append({'type': 'bear', 'top': ob['High'], 'bottom': ob['Low'], 'x': ob.name})

        return df, smc_data

# ==========================================
# 5. INTELLIGENCE (REPORTING & BROADCAST)
# ==========================================
class Intelligence:
    @staticmethod
    def generate_quick_signal(df: pd.DataFrame, ticker: str) -> str:
        last = df.iloc[-1]
        action = "LONG 🟢" if last['Trend'] == 1 else "SHORT 🔴"
        return f"""⚡ QUICK SIGNAL: {ticker}
Action: {action}
Price: {last['Close']:.2f}
Trend (HMA): {last['HMA_Trend']:.2f}
Flux: {last['Apex_Flux']:.2f}
WaveTrend: {last['WaveTrend']:.1f}
Timestamp: {datetime.now().strftime('%H:%M UTC')}"""

    @staticmethod
    def generate_signal_report(df: pd.DataFrame, ticker: str, smc: Dict) -> str:
        last = df.iloc[-1]
        trend_str = "BULLISH" if last['Trend'] == 1 else "BEARISH"
        
        last_bos = smc['bos'][-1] if smc['bos'] else None
        bos_str = f"{last_bos['type'].upper()} at {last_bos['price']:.2f}" if last_bos else "None"

        curr_price = last['Close']
        nearest_fvg = "None"
        min_dist = float('inf')
        for f in smc['fvg'][-5:]:
            mid = (f['top'] + f['bottom']) / 2
            dist = abs(curr_price - mid)
            if dist < min_dist:
                min_dist = dist
                nearest_fvg = f"{f['type'].upper()} ({f['bottom']:.2f} - {f['top']:.2f})"

        return f"""📄 ARCHITECT STRATEGIC REPORT: {ticker}
━━━━━━━━━━━━━━━━━━━━━━
🎯 STRATEGY: APEX SMC v3.2
━━━━━━━━━━━━━━━━━━━━━━
🔹 STRUCTURE ANALYSIS
• Trend State: {trend_str}
• Last BOS: {bos_str}
• Nearest FVG: {nearest_fvg}
• Key Pivot: {last['HMA_Trend']:.2f}

🔹 PHYSICS ENGINE
• Entropy (CHEDO): {last['CHEDO']:.2f}
• Vector Flux: {last['Apex_Flux']:.2f}
• WaveTrend: {last['WaveTrend']:.1f}

🔹 EXECUTION PARAMETERS
• Current Price: {last['Close']:.2f}
• Stop Loss Ref: {(last['Close'] - 2*last['ATR']):.2f} (L) / {(last['Close'] + 2*last['ATR']):.2f} (S)
• Volatility (ATR): {last['ATR']:.2f}

━━━━━━━━━━━━━━━━━━━━━━
Generated by Axiom Titan Terminal"""

    @staticmethod
    def broadcast(message: str, token: str, chat: str):
        if not token or not chat: return False, "Missing Telegram Credentials"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            # Robust Chunking for Long Reports
            max_len = 3000
            if len(message) <= max_len:
                requests.post(url, json={"chat_id": chat, "text": message})
            else:
                for i in range(0, len(message), max_len):
                    chunk = message[i:i+max_len]
                    requests.post(url, json={"chat_id": chat, "text": f"(Part {i//max_len + 1}) {chunk}"})
            return True, "Broadcast Sent Successfully"
        except Exception as e: return False, str(e)

# ==========================================
# 6. DASHBOARD
# ==========================================
class Dashboard:
    @staticmethod
    def render_charts(df: pd.DataFrame, smc: Dict, is_mobile: bool):
        row_heights = [0.5, 0.15, 0.15, 0.2]
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=row_heights, subplot_titles=("Price & SMC Structure", "Entropy (CHEDO)", "WaveTrend Momentum", "Apex Flux"))

        # 1. Price Pane
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        
        # Cloud
        fig.add_trace(go.Scatter(x=df.index, y=df['Trend_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Trend_Lower'], fill='tonexty', fillcolor='rgba(0, 105, 92, 0.1)' if df['Trend'].iloc[-1]==1 else 'rgba(183, 28, 28, 0.1)', line=dict(width=0), name="Apex Cloud"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA_Trend'], line=dict(color='yellow', width=1), name="HMA Baseline"), row=1, col=1)

        # SMC Visuals
        for bos in smc['bos']:
            if bos['idx'] > len(df) - 150:
                color = AxiomConfig.COLORS['bull_sig'] if bos['type'] == 'bull' else AxiomConfig.COLORS['bear_sig']
                fig.add_shape(type="line", x0=bos['date'], x1=df.index[-1], y0=bos['price'], y1=bos['price'], line=dict(color=color, width=1, dash="dash"), row=1, col=1)

        for ob in smc['ob']:
            color = AxiomConfig.COLORS['smc_bull'] if ob['type'] == 'bull' else AxiomConfig.COLORS['smc_bear']
            fig.add_shape(type="rect", x0=ob['x'], x1=df.index[-1], y0=ob['bottom'], y1=ob['top'], fillcolor=color, opacity=0.4, line_width=0, row=1, col=1)
        
        for fvg in smc['fvg']:
            color = AxiomConfig.COLORS['smc_bull'] if fvg['type'] == 'bull' else AxiomConfig.COLORS['smc_bear']
            fig.add_shape(type="rect", x0=fvg['x0'], x1=df.index[-1], y0=fvg['bottom'], y1=fvg['top'], fillcolor=color, opacity=0.3, line_width=0, row=1, col=1)

        # Signals
        buys = df[df['Sig_Buy']]
        sells = df[df['Sig_Sell']]
        fig.add_trace(go.Scatter(x=buys.index, y=buys['Low'], mode='markers', marker=dict(symbol='triangle-up', size=10, color=AxiomConfig.COLORS['bull_sig']), name="BUY"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells.index, y=sells['High'], mode='markers', marker=dict(symbol='triangle-down', size=10, color=AxiomConfig.COLORS['bear_sig']), name="SELL"), row=1, col=1)

        # 2. Entropy
        fig.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], fill='tozeroy', line=dict(color='#D500F9'), name="Entropy"), row=2, col=1)

        # 3. WaveTrend
        fig.add_trace(go.Scatter(x=df.index, y=df['WaveTrend'], fill='tozeroy', line=dict(color='#00F0FF'), name="WaveTrend"), row=3, col=1)

        # 4. Flux
        colors = np.where(df['Apex_Flux']>0, '#00E676', '#FF1744')
        fig.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], marker_color=colors, name="Flux"), row=4, col=1)

        fig.update_layout(height=1000 if is_mobile else 900, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0), xaxis_rangeslider_visible=False, showlegend=False, paper_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 7. MAIN ORCHESTRATION
# ==========================================
def main():
    st.set_page_config(page_title=AxiomConfig.PAGE_TITLE, page_icon=AxiomConfig.PAGE_ICON, layout=AxiomConfig.LAYOUT)
    
    # Sidebar
    st.sidebar.markdown("## 💠 SYSTEM CONTROL")
    is_mobile = st.sidebar.toggle("📱 Mobile Mode", value=True)
    
    with st.sidebar.expander("🔐 Credentials"):
        openai_key = st.text_input("OpenAI Key", value=AxiomConfig.get_secret("OPENAI_API_KEY"), type="password")
        tg_token = st.text_input("TG Token", value=AxiomConfig.get_secret("TELEGRAM_TOKEN"), type="password")
        tg_chat = st.text_input("TG Chat ID", value=AxiomConfig.get_secret("TELEGRAM_CHAT_ID"))

    cat = st.sidebar.selectbox("Sector", list(AxiomConfig.ASSETS.keys()))
    ticker = st.sidebar.selectbox("Asset", AxiomConfig.ASSETS[cat])
    tf = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=2)

    # Init UI
    UXFactory.inject_css(is_mobile)
    UXFactory.render_header(cat)
    UXFactory.render_world_clock()

    # Process
    with st.spinner("Processing Axiom Physics & SMC Logic..."):
        df = DataCore.fetch_data(ticker, tf)
        macro_p = DataCore.get_macro_data()

        if not df.empty:
            # Run Engines
            df = QuantEngine.calc_physics(df)
            df, smc_data = QuantEngine.calc_apex_smc_strategy(df)
            last = df.iloc[-1]

            # Top Metrics
            if is_mobile:
                st.markdown(f"""
                <div class="report-card">
                    <div class="report-header">💠 {ticker}: {last['Close']:.2f}</div>
                    <div class="report-item">Trend: <span class="highlight">{'BULL' if last['Trend']==1 else 'BEAR'}</span></div>
                    <div class="report-item">Flux: <span class="highlight">{last['Apex_Flux']:.2f}</span></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price", f"{last['Close']:.2f}")
                c2.metric("Trend", "BULL" if last['Trend']==1 else "BEAR", delta=f"{last['HMA_Trend']:.2f}")
                c3.metric("Entropy", f"{last['CHEDO']:.2f}")
                c4.metric("Flux", f"{last['Apex_Flux']:.2f}")

            # Tabs
            tabs = st.tabs(["📉 Axiom Charts", "📡 Broadcast", "🧠 Strategy Guide", "📊 Volume", "🌍 Macro"])
            
            with tabs[0]:
                Dashboard.render_charts(df, smc_data, is_mobile)

            with tabs[1]:
                st.markdown("### 📡 Signal Broadcaster")
                UXFactory.render_tradingview_widget(ticker, tf)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### ⚡ Quick Signal")
                    default_quick = Intelligence.generate_quick_signal(df, ticker)
                    msg_quick = st.text_area("Payload", value=default_quick, height=150, key="quick")
                    if st.button("🚀 Send Trade Signal", use_container_width=is_mobile):
                        success, info = Intelligence.broadcast(msg_quick, tg_token, tg_chat)
                        if success: st.success(info)
                        else: st.error(info)
                
                with c2:
                    st.markdown("#### 📄 Architect Report")
                    default_report = Intelligence.generate_signal_report(df, ticker, smc_data)
                    msg_report = st.text_area("Report Payload", value=default_report, height=350, key="report")
                    if st.button("📨 Send Architect Report", use_container_width=is_mobile):
                        success, info = Intelligence.broadcast(msg_report, tg_token, tg_chat)
                        if success: st.success(info)
                        else: st.error(info)

            with tabs[2]:
                st.markdown("### 🧠 Smart Money Concepts (SMC) Guide")
                
                with st.expander("1. Market Structure (BOS & CHoCH)", expanded=True):
                    st.markdown("""
                    **Break of Structure (BOS):**
                    * **Definition:** When price breaks a significant Pivot High (in an uptrend) or Pivot Low (in a downtrend) in the direction of the trend.
                    * **Significance:** Confirms trend continuation.
                    
                    **Change of Character (CHoCH):**
                    * **Definition:** When price breaks the last significant pivot *against* the trend (e.g., breaking a Higher Low in an uptrend).
                    * **Significance:** Signals a potential trend reversal.
                    """)
                
                with st.expander("2. Order Blocks (OB)"):
                    st.markdown("""
                    **Definition:** The last candle of the opposing color before a strong impulse move (BOS).
                    * **Bullish OB:** The last SELL candle before a strong BUY move.
                    * **Bearish OB:** The last BUY candle before a strong SELL move.
                    **Usage:** These zones act as high-probability support/resistance for re-entries.
                    """)
                
                with st.expander("3. Fair Value Gaps (FVG)"):
                    st.markdown("""
                    **Definition:** A three-candle pattern where the 1st candle's high/low does not overlap with the 3rd candle's low/high, leaving a gap.
                    **Physics:** Represents an inefficiency in the market (imbalance) that price often returns to "fill".
                    """)

            with tabs[3]:
                st.bar_chart(df['Volume'].tail(100))
            
            with tabs[4]:
                st.write(macro_p)

        else:
            st.error("Data Unavailable.")

if __name__ == "__main__":
    main()
