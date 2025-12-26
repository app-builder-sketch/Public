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
from scipy.stats import linregress
from typing import Tuple, Dict, Any, Optional, List

# ==========================================
# 1. CONFIGURATION & SECRETS
# ==========================================
class AxiomConfig:
    PAGE_TITLE = "Axiom Titan Terminal"
    PAGE_ICON = "💠"
    LAYOUT = "wide"
    
    # Combined Asset Universe
    ASSETS = {
        "Indices & Macro": ["SPY", "QQQ", "IWM", "TLT", "VXX", "^TNX", "DX-Y.NYB", "^VIX"],
        "Crypto (Major)": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD"],
        "Crypto (Alt/Meme)": ["SHIB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "ARB-USD", "RNDR-USD", "LINK-USD"],
        "US Tech (Mag 7+)": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX"],
        "Commodities": ["GLD", "SLV", "USO", "CL=F", "GC=F", "NG=F", "HG=F"]
    }

    @staticmethod
    def get_secret(key: str) -> str:
        """
        Auto-loads secrets from .streamlit/secrets.toml if available.
        Returns empty string if not found, allowing manual override.
       
        """
        try:
            return st.secrets.get(key, "")
        except:
            return ""

# ==========================================
# 2. UX FACTORY (STYLING & COMPONENT)
# ==========================================
class UXFactory:
    @staticmethod
    def inject_css(is_mobile: bool):
        """Injects the "DarkPool/Neon" aesthetic CSS with Mobile responsiveness."""
        base_css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=SF+Pro+Display:wght@300;500;700&display=swap');
            .stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'SF Pro Display', monospace; }
            
            /* Title Glow */
            .title-glow {
                font-size: 2.5em; font-weight: bold; color: #ffffff;
                text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00;
                margin-bottom: 10px;
            }

            /* Metric Cards */
            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-left: 3px solid #00ff00;
                padding: 10px; border-radius: 4px;
                transition: transform 0.2s;
            }
            div[data-testid="stMetric"]:hover { transform: scale(1.02); border-color: #00ff00; }
            div[data-testid="stMetricValue"] { color: #fff; font-weight: 700; font-size: 1.4rem !important; }
            div[data-testid="stMetricLabel"] { color: #8b949e; letter-spacing: 1px; font-size: 0.8rem; }

            /* Mobile Specific */
            .report-card {
                background-color: #161b22;
                border-left: 4px solid #00F0FF;
                padding: 15px; border-radius: 6px;
                margin-bottom: 10px; font-family: 'SF Pro Display', sans-serif;
            }
            .report-header { font-size: 1.1rem; font-weight: bold; color: #fff; margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 5px; }
            .report-item { margin-bottom: 5px; font-size: 0.9rem; color: #aaa; }
            .highlight { color: #00F0FF; font-weight: bold; }
        </style>
        """
        
        mobile_css = """
        <style>
            div[data-testid="stMetric"] { margin-bottom: 8px; padding: 15px; }
            .js-plotly-plot { height: 100% !important; } 
            button { min-height: 50px !important; margin-top: 10px !important; }
        </style>
        """
        
        st.markdown(base_css, unsafe_allow_html=True)
        if is_mobile:
            st.markdown(mobile_css, unsafe_allow_html=True)

    @staticmethod
    def render_header():
        """Renders the Title and Ticker Tape."""
        st.markdown('<div class="title-glow">👁️ Axiom Titan Terminal</div>', unsafe_allow_html=True)
        
        # Ticker Tape Implementation
        st.markdown("""
        <div class="ticker-wrap" style="width: 100%; overflow: hidden; background-color: #0d1117; border-bottom: 1px solid #30363d; height: 30px; display: flex; align-items: center;">
            <div class="ticker" style="display: inline-block; animation: marquee 60s linear infinite; white-space: nowrap;">
                <span style="padding:0 2rem; color:#00ff00; font-family:'Roboto Mono';">SYSTEM: ONLINE</span>
                <span style="padding:0 2rem; color:#00ff00; font-family:'Roboto Mono';">PHYSICS ENGINE: ACTIVE</span>
                <span style="padding:0 2rem; color:#00ff00; font-family:'Roboto Mono';">GOD MODE: ENABLED</span>
                <span style="padding:0 2rem; color:#00ff00; font-family:'Roboto Mono';">AI ANALYST: STANDBY</span>
            </div>
        </div>
        <style>@keyframes marquee { 0% { transform: translate(100%, 0); } 100% { transform: translate(-100%, 0); } }</style>
        """, unsafe_allow_html=True)

# ==========================================
# 3. DATA CORE (ROBUST UPGRADE)
# ==========================================
class DataCore:
    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_data(ticker: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """
        Robust data fetching with Smart Fallback logic.
        If the requested period fails, it retries with a shorter period.
       
        """
        # Primary Map: The ideal period we WANT
        primary_tf_map = {
            "15m": "60d",  # Max 60 days for 15m
            "1h": "730d",  # Max 730 days for 1h
            "4h": "730d",  # Derived from 1h
            "1d": "5y",
            "1wk": "10y"
        }
        
        # Fallback Map: A safe period we accept if Primary fails
        fallback_tf_map = {
            "15m": "5d",
            "1h": "60d", 
            "4h": "60d", 
            "1d": "1y",
            "1wk": "2y"
        }

        period = primary_tf_map.get(timeframe, "1y")
        dl_interval = "1h" if timeframe == "4h" else timeframe
        
        try:
            # ATTEMPT 1: Primary Request
            df = yf.download(ticker, period=period, interval=dl_interval, progress=False, threads=False)
            
            # ATTEMPT 2: Fallback Request (if empty)
            if df.empty:
                safe_period = fallback_tf_map.get(timeframe, "1mo")
                st.warning(f"⚠️ Primary data stream unavailable. Retrying with backup period ({safe_period})...")
                df = yf.download(ticker, period=safe_period, interval=dl_interval, progress=False, threads=False)

            if df.empty: 
                st.error(f"❌ Failed to fetch data for {ticker} ({timeframe}). Yahoo Finance may be rate-limiting.")
                return pd.DataFrame()
            
            # MultiIndex Handling (Critical for yfinance v0.2+)
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs(ticker, axis=1, level=0)
                except: df.columns = df.columns.get_level_values(0)
            
            # Standardization
            cols = {c: c.capitalize() for c in df.columns}
            df = df.rename(columns=cols)
            if 'Adj close' in df.columns: df['Close'] = df['Adj close']
            
            # Resample for 4h (Aggregating 1h data)
            if timeframe == "4h":
                # Ensure index is Datetime
                df.index = pd.to_datetime(df.index)
                agg_dict = {
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }
                # Only aggregate columns that actually exist
                agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
                df = df.resample('4h').agg(agg_dict).dropna()

            return df.tail(limit)
        
        except Exception as e:
            st.error(f"Data Core Critical Error: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_fundamentals(ticker: str) -> Optional[Dict]:
        """Fetches fundamentals safely."""
        # Skip fundamentals for crypto/forex to prevent errors
        if any(x in ticker for x in ["-", "=", "^"]): return None 
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "Market Cap": info.get("marketCap", "N/A"),
                "P/E": info.get("trailingPE", "N/A"),
                "Growth": info.get("revenueGrowth", 0),
                "Summary": info.get("longBusinessSummary", "No Data")
            }
        except: return None

    @staticmethod
    @st.cache_data(ttl=300)
    def get_macro_data():
        """Batch fetches global macro indicators."""
        assets = {"S&P 500": "SPY", "Bitcoin": "BTC-USD", "10Y Yield": "^TNX", "VIX": "^VIX", "DXY": "DX-Y.NYB", "Gold": "GC=F"}
        try:
            data = yf.download(list(assets.values()), period="5d", interval="1d", progress=False, threads=False)['Close']
            # Fallback for single-column result if only 1 asset returns
            if isinstance(data, pd.Series): return {}, {}
            
            prices = {k: data[v].iloc[-1] for k,v in assets.items() if v in data.columns}
            changes = {k: ((data[v].iloc[-1]-data[v].iloc[-2])/data[v].iloc[-2])*100 for k,v in assets.items() if v in data.columns}
            return prices, changes
        except: return {}, {}
# ==========================================
# 4. QUANT ENGINE (FIXED)
# ==========================================
class QuantEngine:
    
    @staticmethod
    def tanh(x): 
        return np.tanh(np.clip(x, -20, 20))

    # --- HELPER FUNCTIONS ---
    @staticmethod
    def calculate_wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def calculate_hma(series, length):
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wma_half = QuantEngine.calculate_wma(series, half_length)
        wma_full = QuantEngine.calculate_wma(series, length)
        diff = 2 * wma_half - wma_full
        return QuantEngine.calculate_wma(diff, sqrt_length)

    @staticmethod
    def calculate_atr(df, length=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(length).mean()

    # --- PHYSICS INDICATORS (AXIOM) ---
    @staticmethod
    def calc_chedo(df: pd.DataFrame, length: int = 50) -> pd.DataFrame:
        """Calculates CHEDO (Chaos/Entropy Dynamics)."""
        c = df['Close'].values
        # Log Returns
        log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
        
        # Volatility & Hyperbolic Distance
        mu = pd.Series(log_ret).rolling(length).mean().values
        sigma = pd.Series(log_ret).rolling(length).std().values
        v = sigma / (np.abs(mu) + 1e-9)
        abs_ret_v = np.abs(log_ret) * v
        hyper_dist = np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))
        
        # Kappa (Curvature)
        kappa_h = QuantEngine.tanh(pd.Series(hyper_dist).rolling(length).mean().values)
        
        # Lyapunov Proxy
        diff_ret = np.diff(log_ret, prepend=0)
        lyap = np.log(np.abs(diff_ret) + 1e-9)
        lambda_n = QuantEngine.tanh((pd.Series(lyap).rolling(length).mean().values + 5) / 7)
        
        # Entropy
        ent = pd.Series(log_ret**2).rolling(length).sum().values
        ent_n = QuantEngine.tanh(ent * 10)
        
        # Synthesis
        raw = (0.4 * kappa_h) + (0.3 * lambda_n) + (0.3 * ent_n)
        
        # Assign directly as numpy array to avoid index issues
        df['CHEDO'] = 2 / (1 + np.exp(-raw * 4)) - 1
        return df

    @staticmethod
    def calc_rqzo(df: pd.DataFrame, harmonics: int = 25) -> pd.DataFrame:
        """Calculates RQZO (Relativistic Quantum Oscillator)."""
        src = df['Close']
        mn, mx = src.rolling(100).min(), src.rolling(100).max()
        # Ensure numpy division to avoid index friction
        norm = (src - mn) / (mx - mn + 1e-9)
        
        v = np.abs(norm.diff())
        c_limit = 0.05
        # FillNa(1.0) on gamma to treat missing initial data as 'normal time'
        gamma_series = 1 / np.sqrt(1 - (np.minimum(v, c_limit*0.99)/c_limit)**2)
        gamma = gamma_series.fillna(1.0).values
        
        idx = np.arange(len(df))
        tau = (idx % 100) / gamma
        zeta = np.zeros(len(df))
        
        for n in range(1, harmonics + 1):
            amp = n ** -0.5
            theta = tau * np.log(n)
            zeta += amp * np.sin(theta)
            
        # FIX: Assign numpy array directly so Pandas infers alignment by order
        df['RQZO'] = zeta
        return df

    @staticmethod
    def calc_apex_flux(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        """Calculates Apex Flux (Vector Efficiency)."""
        rg = df['High'] - df['Low']
        body = np.abs(df['Close'] - df['Open'])
        
        # Calculate raw efficiency (numpy array)
        eff_raw = np.where(rg == 0, 0, body / rg)
        
        # FIX: Create Series with df.index explicitly to allow alignment
        eff_sm = pd.Series(eff_raw, index=df.index).ewm(span=length).mean()
        
        vol_avg = df['Volume'].rolling(55).mean()
        # numpy array
        v_rat = np.where(vol_avg == 0, 1, df['Volume'] / vol_avg)
        
        # Pandas Series (DatetimeIndex)
        direction = np.sign(df['Close'] - df['Open'])
        
        # Now all components align: 
        # direction (Series w/ DateIndex) * eff_sm (Series w/ DateIndex) * v_rat (Broadcasted Array)
        raw = direction * eff_sm * v_rat
        
        df['Apex_Flux'] = pd.Series(raw).ewm(span=5).mean()
        return df

    # --- GOD MODE INDICATORS ---
    @staticmethod
    def calc_god_mode(df: pd.DataFrame) -> pd.DataFrame:
        # 1. HMA Trend
        df['HMA_55'] = QuantEngine.calculate_hma(df['Close'], 55)
        df['ATR'] = QuantEngine.calculate_atr(df, 14)
        
        # 2. Apex Trend (Cloud)
        mult = 1.5
        df['Apex_Upper'] = df['HMA_55'] + (QuantEngine.calculate_atr(df, 55) * mult)
        df['Apex_Lower'] = df['HMA_55'] - (QuantEngine.calculate_atr(df, 55) * mult)
        df['Apex_Trend'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, 0))
        df['Apex_Trend'] = df['Apex_Trend'].replace(0, method='ffill')

        # 3. Squeeze Momentum
        basis = df['Close'].rolling(20).mean()
        dev = df['Close'].rolling(20).std() * 2.0
        upper_bb = basis + dev
        lower_bb = basis - dev
        
        ma_kc = df['Close'].rolling(20).mean()
        range_ma = QuantEngine.calculate_atr(df, 20)
        upper_kc = ma_kc + (range_ma * 1.5)
        lower_kc = ma_kc - (range_ma * 1.5)
        
        df['Squeeze_On'] = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        
        # Momentum (LinReg Slope)
        def get_slope(series):
            y = series.values
            x = np.arange(len(y))
            if len(y) < 20: return 0
            slope, _, _, _, _ = linregress(x, y)
            return slope

        # Efficient Rolling Slope
        delta = df['Close'] - ((df['High'].rolling(20).max() + df['Low'].rolling(20).min())/2 + ma_kc)/2
        df['Sqz_Mom'] = delta.rolling(20).apply(get_slope, raw=False) * 100

        # 4. Money Flow Matrix
        rsi_src = (100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))) - 50
        mf_vol = df['Volume'] / df['Volume'].rolling(14).mean()
        df['MF_Matrix'] = (rsi_src * mf_vol).ewm(span=3).mean()

        # 5. God Mode Score
        df['GM_Score'] = (
            df['Apex_Trend'] + 
            np.sign(df['Sqz_Mom']) + 
            np.sign(df['Apex_Flux'])
        )
        return df

    @staticmethod
    def calc_smc(df: pd.DataFrame) -> Dict:
        """Smart Money Concepts (Structures)."""
        smc = {'structures': []}
        swing = 5
        df['PH'] = df['High'].rolling(swing*2+1, center=True).max() == df['High']
        df['PL'] = df['Low'].rolling(swing*2+1, center=True).min() == df['Low']
        
        last_h = None
        last_l = None
        for i in range(swing, len(df)):
            if df['PH'].iloc[i]:
                if last_h: smc['structures'].append({'x0':last_h[0], 'y':last_h[1], 'x1':df.index[i], 'color':'red', 'type':'RES'})
                last_h = (df.index[i], df['High'].iloc[i])
            if df['PL'].iloc[i]:
                if last_l: smc['structures'].append({'x0':last_l[0], 'y':last_l[1], 'x1':df.index[i], 'color':'green', 'type':'SUP'})
                last_l = (df.index[i], df['Low'].iloc[i])
        return smc

    @staticmethod
    def run_monte_carlo(df: pd.DataFrame, days: int = 30, sims: int = 100) -> np.ndarray:
        last = df['Close'].iloc[-1]
        rets = df['Close'].pct_change().dropna()
        if rets.empty: return np.zeros((days, sims))
        mu, sigma = rets.mean(), rets.std()
        sim_rets = np.random.normal(mu, sigma, (days, sims))
        paths = np.zeros((days, sims))
        paths[0] = last
        for t in range(1, days): paths[t] = paths[t-1] * (1 + sim_rets[t])
        return paths

    @staticmethod
    def calc_volume_profile(df: pd.DataFrame, bins: int = 50):
        price_bins = np.linspace(df['Low'].min(), df['High'].max(), bins)
        df['Bin'] = pd.cut(df['Close'], bins=price_bins, include_lowest=True)
        vp = df.groupby('Bin', observed=False)['Volume'].sum().reset_index()
        vp['Price'] = [i.mid for i in vp['Bin']]
        poc_price = vp.loc[vp['Volume'].idxmax(), 'Price']
        return vp, poc_price

# ==========================================
# 5. INTELLIGENCE (AI & TELEGRAM)
# ==========================================
class Intelligence:
    @staticmethod
    def analyze_ai(df: pd.DataFrame, ticker: str, api_key: str) -> str:
        """God Mode AI Analyst."""
        if not api_key: return "⚠️ Missing OpenAI API Key."
        
        last = df.iloc[-1]
        gm_verdict = "STRONG BUY" if last['GM_Score'] >= 2 else "STRONG SELL" if last['GM_Score'] <= -2 else "NEUTRAL"
        
        system_prompt = """
        You are 'Axiom', a Tier-1 Quantitative Analyst. Analyze using FIRST PRINCIPLES.
        
        --- INDICATOR LEGEND ---
        1. CHEDO (Entropy): >0.8 = Max Chaos (Mean Reversion Risk).
        2. RQZO (Relativity): High Amp = Time Dilation/Instability.
        3. Apex Flux (Vector): >0.6 = Superconductor (Trend Efficiency).
        4. God Mode Score: Aggregate trend signal (-3 to +3).
        
        --- MISSION ---
        Provide a concise, high-level overview.
        1. Analyze Market Structure (Trend vs Chop).
        2. Correlate Physics (Entropy) with Technicals (Squeeze).
        3. Provide Outlook (Bull/Bear/Neutral).
        
        Use emojis (🚀, 📉, ⚛️) liberally. No specific financial advice.
        """
        
        user_prompt = f"""
        Asset: {ticker} | Price: {last['Close']:.2f}
        Titan Score: {last['GM_Score']} ({gm_verdict})
        Trend: {'Bullish' if last['Apex_Trend']==1 else 'Bearish'}
        Entropy (CHEDO): {last['CHEDO']:.3f}
        Relativity (RQZO): {last['RQZO']:.3f}
        Flux: {last['Apex_Flux']:.3f}
        Squeeze: {'ON' if last['Squeeze_On'] else 'OFF'}
        """
        
        try:
            client = OpenAI(api_key=api_key)
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=400
            )
            return res.choices[0].message.content
        except Exception as e: return f"AI Error: {e}"

    @staticmethod
    def generate_mobile_card(last: pd.Series, ticker: str) -> str:
        """HTML Card for Mobile."""
        direction = "LONG 🐂" if last['Apex_Trend'] == 1 else "SHORT 🐻"
        stop = last['HMA_55']
        
        return f"""
        <div class="report-card">
            <div class="report-header">💠 AXIOM SIGNAL: {direction}</div>
            <div class="report-item">Asset: <span class="highlight">{ticker}</span></div>
            <div class="report-item">GM Score: <span class="highlight">{last['GM_Score']:.0f} / 3</span></div>
        </div>
        <div class="report-card">
            <div class="report-header">⚛️ PHYSICS DATA</div>
            <div class="report-item">Entropy: <span class="highlight">{last['CHEDO']:.2f}</span></div>
            <div class="report-item">Flux: <span class="highlight">{last['Apex_Flux']:.2f}</span></div>
            <div class="report-item">Squeeze: <span class="highlight">{'ACTIVE' if last['Squeeze_On'] else 'OFF'}</span></div>
        </div>
        <div class="report-card">
            <div class="report-header">🎯 LEVELS</div>
            <div class="report-item">Price: <span class="highlight">{last['Close']:.2f}</span></div>
            <div class="report-item">Base (Stop): <span class="highlight">{stop:.2f}</span></div>
        </div>
        """

    @staticmethod
    def broadcast(message: str, token: str, chat: str):
        """
        Telegram Broadcast with message chunking.
        Handles API limits by splitting long reports.
       
        """
        if not token or not chat: return False, "Missing Credentials"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            max_len = 4000
            if len(message) <= max_len:
                requests.post(url, json={"chat_id": chat, "text": message, "parse_mode": "Markdown"})
            else:
                for i in range(0, len(message), max_len):
                    requests.post(url, json={"chat_id": chat, "text": message[i:i+max_len], "parse_mode": "Markdown"})
            return True, "Sent"
        except Exception as e: return False, str(e)

# ==========================================
# 6. DASHBOARD (CHARTS)
# ==========================================
class Dashboard:
    @staticmethod
    def render_physics_charts(df: pd.DataFrame, smc: Dict, is_mobile: bool):
        """God Mode + Physics Subplots."""
        
        # Dynamic Row Heights for Mobile vs Desktop
        if is_mobile:
            row_heights = [0.4, 0.15, 0.15, 0.15, 0.15]
            total_h = 1000
        else:
            row_heights = [0.4, 0.15, 0.15, 0.15, 0.15]
            total_h = 900
            
        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights,
            subplot_titles=("Price & Apex Cloud", "Squeeze Momentum", "Money Flow Matrix", "CHEDO (Entropy)", "Apex Flux")
        )

        # 1. Price + Apex Cloud + SMC
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 255, 0, 0.05)', line=dict(width=0), name="Apex Cloud"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA_55'], line=dict(color='white', width=1, dash='dot'), name="HMA 55"), row=1, col=1)
        
        # SMC Lines
        for s in smc['structures']:
            fig.add_shape(type="line", x0=s['x0'], x1=s['x1'], y0=s['y'], y1=s['y'], line=dict(color=s['color'], width=1, dash="dot"), row=1, col=1)

        # 2. Squeeze Momentum
        colors = ['#00E676' if v > 0 else '#FF5252' for v in df['Sqz_Mom']]
        fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=colors, name="Mom"), row=2, col=1)

        # 3. Money Flow Matrix
        fig.add_trace(go.Scatter(x=df.index, y=df['MF_Matrix'], fill='tozeroy', line=dict(color='cyan', width=1), name="MFI"), row=3, col=1)

        # 4. CHEDO (Entropy)
        fig.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], line=dict(color='#D500F9', width=2), fill='tozeroy', fillcolor='rgba(213, 0, 249, 0.1)', name="Entropy"), row=4, col=1)
        fig.add_hline(y=0.8, line_dash="dot", line_color="red", row=4, col=1)

        # 5. Flux (Vector)
        flux_cols = np.where(df['Apex_Flux'] > 0.6, '#00E676', np.where(df['Apex_Flux'] < -0.6, '#FF1744', '#2979FF'))
        fig.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], marker_color=flux_cols, name="Flux"), row=5, col=1)

        fig.update_layout(height=total_h, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False, showlegend=False, paper_bgcolor="#0e1117", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 7. MAIN ORCHESTRATION
# ==========================================
def main():
    st.set_page_config(page_title=AxiomConfig.PAGE_TITLE, page_icon=AxiomConfig.PAGE_ICON, layout=AxiomConfig.LAYOUT)
    
    # Sidebar Controls
    st.sidebar.markdown("## 💠 SYSTEM CONTROL")
    is_mobile = st.sidebar.toggle("📱 Mobile Mode", value=True)
    
    # Credentials (Auto-Load Logic)
    with st.sidebar.expander("🔐 Credentials"):
        # Auto-fill from secrets if available
        default_ai = AxiomConfig.get_secret("OPENAI_API_KEY")
        default_token = AxiomConfig.get_secret("TELEGRAM_TOKEN")
        default_chat = AxiomConfig.get_secret("TELEGRAM_CHAT_ID")

        openai_key = st.text_input("OpenAI Key", value=default_ai, type="password")
        tg_token = st.text_input("TG Token", value=default_token, type="password")
        tg_chat = st.text_input("TG Chat ID", value=default_chat)

    # Inputs
    cat = st.sidebar.selectbox("Sector", list(AxiomConfig.ASSETS.keys()))
    ticker = st.sidebar.selectbox("Asset", AxiomConfig.ASSETS[cat])
    tf = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=2)

    # Init UI
    UXFactory.inject_css(is_mobile)
    UXFactory.render_header()
    
    # Process Data
    with st.spinner("Initializing Axiom Physics Engine..."):
        df = DataCore.fetch_data(ticker, tf)
        macro_p, macro_c = DataCore.get_macro_data()
        fund = DataCore.get_fundamentals(ticker)

        if not df.empty:
            # Run Engines
            df = QuantEngine.calc_chedo(df)
            df = QuantEngine.calc_rqzo(df)
            df = QuantEngine.calc_apex_flux(df)
            df = QuantEngine.calc_god_mode(df)
            smc = QuantEngine.calc_smc(df)
            
            last = df.iloc[-1]
            
            # --- VIEW LAYER ---
            
            # 1. Top Metrics (Desktop) or Card (Mobile)
            if is_mobile:
                st.markdown(Intelligence.generate_mobile_card(last, ticker), unsafe_allow_html=True)
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price", f"{last['Close']:.2f}")
                c2.metric("God Mode", f"{last['GM_Score']:.0f}/3", delta="Strong Buy" if last['GM_Score']>2 else "Neutral")
                c3.metric("Entropy", f"{last['CHEDO']:.2f}", delta="Critical" if abs(last['CHEDO'])>0.8 else "Stable", delta_color="inverse")
                c4.metric("Flux", f"{last['Apex_Flux']:.2f}")
            
            # 2. Tabs
            tabs = st.tabs(["📉 Charts", "🧠 Intelligence", "📊 Volume", "🔮 Simulation", "📡 Broadcast", "🌍 Macro"])
            
            with tabs[0]:
                Dashboard.render_physics_charts(df, smc, is_mobile)
            
            with tabs[1]:
                if st.button("Run AI Analysis", use_container_width=True):
                    analysis = Intelligence.analyze_ai(df, ticker, openai_key)
                    st.markdown(analysis)
            
            with tabs[2]:
                vp, poc = QuantEngine.calc_volume_profile(df)
                st.bar_chart(vp.set_index('Price')['Volume'])
                st.caption(f"Point of Control (POC): {poc:.2f}")

            with tabs[3]:
                paths = QuantEngine.run_monte_carlo(df)
                st.line_chart(paths[:, :50])
                st.caption("Monte Carlo: 30-day projected price paths based on historical volatility.")

            with tabs[4]:
                st.subheader("📡 Telegram Command")
                # Pre-filled payload for speed
                default_msg = f"💠 AXIOM SIGNAL: {ticker}\nPrice: {last['Close']:.2f}\nGM Score: {last['GM_Score']:.0f}\nFlux: {last['Apex_Flux']:.2f}"
                msg = st.text_area("Broadcast Message", value=default_msg)
                
                if st.button("🚀 Transmit Signal", use_container_width=True):
                    ok, info = Intelligence.broadcast(msg, tg_token, tg_chat)
                    if ok: st.success(info)
                    else: st.error(info)

            with tabs[5]:
                c1, c2 = st.columns(2)
                c1.metric("S&P 500", f"{macro_p.get('S&P 500', 0):.2f}", f"{macro_c.get('S&P 500', 0):.2f}%")
                c2.metric("VIX", f"{macro_p.get('VIX', 0):.2f}", f"{macro_c.get('VIX', 0):.2f}%")
                if fund:
                    st.markdown("---")
                    st.write(f"**Fundamentals**: {fund.get('Summary', 'N/A')}")
        else:
            st.error("Data Unavailable. Check Ticker/Interval.")

if __name__ == "__main__":
    main()
