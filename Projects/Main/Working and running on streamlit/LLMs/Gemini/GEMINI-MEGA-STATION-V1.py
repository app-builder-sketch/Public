"""
Titan-AXIOM Mega-Station V4.0 (God Mode)
- BEST OF EVERYTHING: Full Feature Set + Scanner + Risk Desk
- Engines: Titan (Crypto) & Axiom (Stocks)
- Indicators: CHEDO, RQZO, Apex Flux, SMC HMA, Fear/Greed
- Features: Live Scanner, Heikin-Ashi, Position Sizing, AI Agent
"""

import time
import math
import sqlite3
import random
import json
import logging
from typing import Dict, Optional, List, Tuple, Any
from contextlib import contextmanager
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

# -------------------------------------------------------------------------
# STREAMLIT CONFIG & VISUALS
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Titan-AXIOM V4.0 God Mode",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "God Mode" Aesthetics
st.markdown("""
<style>
    .report-card { background-color: #121212; padding: 15px; border-radius: 8px; border: 1px solid #333; margin-bottom: 10px; }
    .report-header { font-weight: 900; color: #fff; font-size: 1.2em; letter-spacing: 1px; margin-bottom: 8px; }
    .report-sub { color: #888; font-size: 0.8em; text-transform: uppercase; margin-bottom: 4px; }
    .metric-box { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding-bottom: 5px; margin-bottom: 5px; }
    .value-cyan { color: #00e5ff; font-weight: bold; font-family: monospace; }
    .value-gold { color: #ffd700; font-weight: bold; font-family: monospace; }
    .trend-bull { color: #00E676; font-weight: bold; }
    .trend-bear { color: #FF1744; font-weight: bold; }
    .radar-box { font-size: 0.8em; padding: 5px; background: #1a1a1a; margin: 2px 0; border-radius: 4px; display:flex; justify-content:space-between;}
    /* Clean Streamlit UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# MATH KERNEL (THE BRAIN)
# -------------------------------------------------------------------------
class TitanMath:
    """Core mathematical calculations."""
    
    @staticmethod
    def get_ma(series: pd.Series, length: int) -> pd.Series:
        return series.rolling(window=length).mean()

    @staticmethod
    def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
        """Converts standard OHLC to Heikin-Ashi for trend visualization."""
        ha = df.copy()
        ha['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        # Initialize open (this is a simplified vectorized approach)
        ha['open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
        ha['open'].iloc[0] = df['open'].iloc[0]
        ha['high'] = ha[['high', 'open', 'close']].max(axis=1)
        ha['low'] = ha[['low', 'open', 'close']].min(axis=1)
        return ha

    @staticmethod
    def calculate_chedo(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].ewm(span=14, adjust=False).mean()
        # CHEDO Logic
        log_ret = np.diff(np.log(df['close']), prepend=np.log(df['close'].iloc[0]))
        mu = pd.Series(log_ret, index=df.index).rolling(14).mean().fillna(0)
        df['dx'] = 100 * (df['atr'] * 1.5 - df['atr'] * 0.9)
        min_dx = df['dx'].rolling(55).min()
        max_dx = df['dx'].rolling(55).max()
        df['CHEDO'] = (df['dx'] - min_dx) / (max_dx - min_dx + 1e-9)
        return df

    @staticmethod
    def calculate_rqzo(df: pd.DataFrame) -> pd.Series:
        src = df['close']
        mn, mx = src.rolling(100).min(), src.rolling(100).max()
        norm = (src - mn) / (mx - mn + 1e-9)
        v = np.abs(norm.diff())
        c_limit = 0.05
        gamma_input = np.minimum(v, c_limit * 0.99) / c_limit
        gamma = 1 / np.sqrt(1 - gamma_input ** 2)
        idx = np.arange(len(df))
        tau = (idx % 100) / gamma.fillna(1.0)
        zeta = np.zeros(len(df))
        for n in range(1, 15): 
            amp = n ** -0.5
            theta = tau * np.log(n)
            zeta += amp * np.sin(theta)
        df['RQZO'] = pd.Series(zeta, index=df.index).fillna(0)
        return df['RQZO']

    @staticmethod
    def calculate_apex_flux(df: pd.DataFrame) -> pd.DataFrame:
        eff_sm = df['volume'] / df['volume'].rolling(20).mean()
        v_rat = np.where(eff_sm == 0, 1, df['volume'] / (eff_sm + 1e-9))
        direction = np.sign(df['close'] - df['open'])
        raw = direction * eff_sm * pd.Series(v_rat, index=df.index)
        df['Apex_Flux'] = raw.ewm(span=5).mean()
        return df

    @staticmethod
    def calculate_smc(df: pd.DataFrame, length: int = 55) -> pd.Series:
        def wma(s, l):
            w = np.arange(1, l + 1)
            return s.rolling(l).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
        close = df['close']
        half = int(length / 2)
        sqrt = int(np.sqrt(length))
        wma_f = wma(close, length)
        wma_h = wma(close, half)
        raw_hma = 2 * wma_h - wma_f
        raw_hma_series = pd.Series(raw_hma, index=df.index)
        df['HMA_Trend'] = wma(raw_hma_series, sqrt)
        df['Trend_Dir'] = np.where(close > df['HMA_Trend'], 1, -1)
        return df['HMA_Trend']

    @staticmethod
    def calculate_fear_greed(df: pd.DataFrame) -> pd.Series:
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        macd = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['FG_Index'] = (rsi + (macd * 10)).clip(0, 100).rolling(5).mean()
        return df['FG_Index']

# -------------------------------------------------------------------------
# DATA ENGINES (Robust & Retry-Enabled)
# -------------------------------------------------------------------------
class TitanEngine:
    
    @staticmethod
    def get_binance_bases() -> List[str]:
        # Hardcoded 'God Mode' list for reliability + Scanner targets
        return ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT"]

    @staticmethod
    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def get_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        pair = f"{symbol}USDT"
        try:
            r = requests.get(
                f"https://api.binance.us/api/v3/klines",
                params={"symbol": pair, "interval": interval, "limit": limit},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                if not isinstance(data, list): return pd.DataFrame()
                df = pd.DataFrame(data, columns=['t','o','h','l','c','v','T','q','n','V','Q','B'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
                df.set_index('timestamp', inplace=True)
                return df[['open','high','low','close','volume']]
            return pd.DataFrame()
        except:
            return pd.DataFrame()

    @staticmethod
    def run_full_analysis(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df = TitanMath.calculate_chedo(df)
        df['RQZO'] = TitanMath.calculate_rqzo(df)
        df = TitanMath.calculate_apex_flux(df)
        df['HMA'] = TitanMath.calculate_smc(df, 55)
        df['FG_Index'] = TitanMath.calculate_fear_greed(df)
        
        # Volatility Squeeze
        bb_mean = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        kc_atr = (df['high'] - df['low']).rolling(20).mean()
        
        df['upper_bb'] = bb_mean + 2 * bb_std
        df['lower_bb'] = bb_mean - 2 * bb_std
        df['upper_kc'] = bb_mean + 1.5 * kc_atr
        df['lower_kc'] = bb_mean - 1.5 * kc_atr
        
        df['in_squeeze'] = (df['lower_bb'] > df['lower_kc']) & (df['upper_bb'] < df['upper_kc'])
        df['is_bull'] = df['close'] > df['HMA']
        
        # Targets
        vol_factor = df['Apex_Flux'].abs().clip(lower=1) * 2
        df['tp1'] = np.where(df['is_bull'], df['close'] * (1 + 0.01 * vol_factor), df['close'] * (1 - 0.01 * vol_factor))
        df['tp3'] = np.where(df['is_bull'], df['close'] * (1 + 0.03 * vol_factor), df['close'] * (1 - 0.03 * vol_factor))
        return df

class AxiomEngine:
    @staticmethod
    def get_tickers_by_class(asset_class: str) -> List[str]:
        mapping = {
            "Global Macro": ["SPY", "QQQ", "GLD", "USO", "UUP", "TLT"],
            "Tech Giants": ["NVDA", "AAPL", "MSFT", "TSLA", "AMD"],
            "High Volatility": ["MSTR", "COIN", "MARA", "GME"]
        }
        return mapping.get(asset_class, [])

    @staticmethod
    def fetch_data(ticker: str, period: str = "1mo") -> pd.DataFrame:
        try:
            df = yf.download(ticker, period=period, interval="1h", progress=False)
            if df.empty: return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            if 'volume' not in df.columns and 'Volume' in df.columns:
                df.rename(columns={'Volume': 'volume'}, inplace=True)
            return df
        except:
            return pd.DataFrame()

    @staticmethod
    def analyze_ai(ticker: str, last: pd.Series, api_key: str) -> str:
        if not api_key: return "⚠️ Connect OpenAI Key for Neural Analysis."
        prompt = (
            f"Analyze {ticker} at {last['close']:.2f}. "
            f"Trend: {'BULL' if last['is_bull'] else 'BEAR'}. "
            f"Flux: {last['Apex_Flux']:.2f} (Vol Flow). "
            f"Entropy (CHEDO): {last.get('CHEDO',0):.2f}. "
            f"Strategy: Brief, aggressive actionable advice."
        )
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o", messages=[{"role":"user", "content":prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI Error: {str(e)[:40]}"

# -------------------------------------------------------------------------
# NEW MODULE: SCANNER & RISK DESK
# -------------------------------------------------------------------------
def render_scanner(assets: List[str]):
    """Sidebar Radar scanning top assets for signals."""
    st.sidebar.markdown("### 📡 Market Radar")
    if st.sidebar.button("Scan Market"):
        progress = st.sidebar.progress(0)
        for i, sym in enumerate(assets):
            df = TitanEngine.get_klines(sym, "1h", limit=100)
            if not df.empty:
                df = TitanMath.calculate_apex_flux(df)
                last_flux = df['Apex_Flux'].iloc[-1]
                trend = "🟢" if last_flux > 0 else "🔴"
                
                # Visual Intensity
                flux_str = f"{last_flux:.2f}"
                if abs(last_flux) > 0.8: flux_str += " 🔥"
                
                st.sidebar.markdown(
                    f"""<div class='radar-box'>
                        <span><b>{sym}</b></span>
                        <span>{trend} {flux_str}</span>
                    </div>""", 
                    unsafe_allow_html=True
                )
            progress.progress((i + 1) / len(assets))

def render_risk_desk(last_price: float, atr: float):
    """Calculates position size based on risk."""
    st.markdown("### 🛡️ Risk Desk")
    c1, c2, c3 = st.columns(3)
    acct = c1.number_input("Account Size ($)", value=10000)
    risk_pct = c2.number_input("Risk %", value=1.0)
    stop_atr_mult = c3.number_input("Stop Width (ATR)", value=2.0)
    
    risk_amt = acct * (risk_pct / 100)
    stop_dist = atr * stop_atr_mult
    position_size = risk_amt / stop_dist
    
    st.success(f"**Position Size:** {position_size:.4f} units")
    st.info(f"**Risk Amount:** ${risk_amt:.2f} | **Stop Loss:** ${last_price - stop_dist:.2f}")

# -------------------------------------------------------------------------
# MAIN APP EXECUTION
# -------------------------------------------------------------------------
def main():
    # 1. SIDEBAR SETUP
    with st.sidebar:
        st.title("TITAN-AXIOM V4")
        st.caption("God Mode // Production Ready")
        
        mode = st.radio("Operating Mode", ["Titan Mobile (Crypto)", "Axiom Quant (Stocks)"])
        api_key = st.text_input("OpenAI Key (Optional)", type="password")
        
        # Scan Feature (Titan Only for speed)
        if mode == "Titan Mobile (Crypto)":
            render_scanner(TitanEngine.get_binance_bases()[:5]) # Scan top 5
        
        st.markdown("---")
        use_ha = st.checkbox("Heikin-Ashi Candles", value=False)
        
    # 2. MAIN LOGIC
    if mode == "Titan Mobile (Crypto)":
        st.header("⚡ Titan Crypto Terminal")
        
        col1, col2, col3 = st.columns(3)
        symbol = col1.selectbox("Asset", TitanEngine.get_binance_bases())
        tf = col2.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
        run_btn = col3.button("INITIATE PROTOCOL", type="primary")
        
        if run_btn:
            with st.spinner(f"Intercepting {symbol} stream..."):
                df = TitanEngine.get_klines(symbol, tf)
                if df.empty:
                    st.error("Connection Failed. Retrying...")
                    time.sleep(1)
                    df = TitanEngine.get_klines(symbol, tf) # One retry
                
                if not df.empty:
                    df = TitanEngine.run_full_analysis(df)
                    
                    # Heikin Ashi Transformation if toggled
                    display_df = TitanMath.calculate_heikin_ashi(df) if use_ha else df
                    last = df.iloc[-1]
                    
                    # --- TABS FOR ORGANIZED VIEW ---
                    tab_main, tab_risk, tab_raw = st.tabs(["📊 Intelligence", "🛡️ Risk Desk", "💾 Raw Data"])
                    
                    with tab_main:
                        # Top Metrics
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Price", f"${last['close']:.2f}", f"{last.get('Apex_Flux',0):.2f} Flux")
                        m2.metric("SMC Trend", "BULL" if last['is_bull'] else "BEAR")
                        m3.metric("Entropy", f"{last.get('CHEDO',0):.2f}")
                        m4.metric("Squeeze", "YES" if last['in_squeeze'] else "NO")
                        
                        # Charts
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                        
                        # Main Candle Trace
                        fig.add_trace(go.Candlestick(
                            x=display_df.index, open=display_df['open'], high=display_df['high'], 
                            low=display_df['low'], close=display_df['close'], name='Price'
                        ), row=1, col=1)
                        
                        # SMC HMA Overlay
                        fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='#E040FB', width=2), name='SMC HMA'), row=1, col=1)
                        
                        # Targets
                        if last['is_bull']:
                            fig.add_trace(go.Scatter(x=[df.index[-1]], y=[last['tp1']], mode='markers', marker=dict(color='cyan', size=10), name='TP1'), row=1, col=1)
                        
                        # Flux Bar Chart
                        colors = ['#00E676' if v > 0 else '#FF1744' for v in df['Apex_Flux']]
                        fig.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], marker_color=colors, name='Flux'), row=2, col=1)
                        
                        fig.update_layout(height=650, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # AI Section
                        if api_key:
                            st.info(f"🤖 **Titan AI:** {AxiomEngine.analyze_ai(symbol, last, api_key)}")
                    
                    with tab_risk:
                        render_risk_desk(last['close'], last['atr'])
                        
                    with tab_raw:
                        st.dataframe(df.tail(50))
                else:
                    st.error("Data interception failed after retries.")

    else:
        # AXIOM QUANT MODE
        st.header("🌐 Axiom Quant Terminal")
        c1, c2 = st.columns(2)
        ac = c1.selectbox("Market Sector", ["Global Macro", "Tech Giants", "High Volatility"])
        ticker = c2.selectbox("Ticker", AxiomEngine.get_tickers_by_class(ac))
        
        if st.button("Access Global Liquidity"):
            with st.spinner("Fetching market data..."):
                df = AxiomEngine.fetch_data(ticker)
                if not df.empty:
                    df = TitanEngine.run_full_analysis(df) # Reuse Titan logic
                    last = df.iloc[-1]
                    
                    st.success(f"Locked on {ticker} | ${last['close']:.2f}")
                    
                    tab1, tab2 = st.tabs(["Technical View", "Risk & AI"])
                    
                    with tab1:
                        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
                        # Add Bands
                        fig.add_trace(go.Scatter(x=df.index, y=df['upper_bb'], line=dict(width=1, color='gray'), name='BB'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['lower_bb'], line=dict(width=1, color='gray'), fill='tonexty'))
                        fig.update_layout(height=500, template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with tab2:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("### Neural Analysis")
                            st.write(AxiomEngine.analyze_ai(ticker, last, api_key) if api_key else "AI Key Required")
                        with col_b:
                            render_risk_desk(last['close'], last.get('atr', last['close']*0.02))

if __name__ == "__main__":
    main()
