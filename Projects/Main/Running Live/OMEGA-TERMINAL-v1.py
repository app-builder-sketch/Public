import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
import requests
import io
import xlsxwriter
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Optional, Dict

# ==========================================
# 1. UI & SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="OMEGA TERMINAL", 
    page_icon="🧬", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- PROFESSIONAL DARK THEME (MOBILE OPTIMIZED) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;800&display=swap');
    
    .stApp { background-color: #000000; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    
    /* SCROLLING TICKER */
    .ticker-wrap {
        width: 100%; overflow: hidden; background-color: #050505; border-bottom: 1px solid #222;
        padding: 5px 0; white-space: nowrap; margin-bottom: 5px;
    }
    .ticker { display: inline-block; animation: marquee 60s linear infinite; }
    .ticker-item { display: inline-block; padding: 0 2rem; font-family: 'Rajdhani'; font-weight: bold; color: #00f260; font-size: 0.8rem; }
    @keyframes marquee { 0% { transform: translate3d(0, 0, 0); } 100% { transform: translate3d(-50%, 0, 0); } }

    /* HEADER */
    .omega-header {
        font-family: 'Rajdhani', sans-serif; font-size: 2.5rem; font-weight: 700;
        color: #fff; letter-spacing: 3px; border-bottom: 1px solid #222; padding: 15px 0;
    }
    .omega-accent { color: #00f260; text-shadow: 0 0 15px rgba(0, 242, 96, 0.4); }
    
    /* METRICS (Touch Friendly) */
    div[data-testid="stMetric"] {
        background: rgba(20, 20, 20, 0.9); border: 1px solid #333; border-left: 3px solid #00f260; border-radius: 6px; 
        padding: 10px;
    }
    div[data-testid="stMetricValue"] { font-family: 'Rajdhani'; font-size: 1.6rem !important; color: #fff; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #888; }

    /* REPORT CARD (Mobile Optimized) */
    .report-card {
        background-color: #111; border-left: 4px solid #00f260;
        padding: 15px; border-radius: 8px; margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .report-header { font-family: 'Rajdhani'; font-size: 1.3rem; color: #fff; border-bottom: 1px solid #333; margin-bottom: 10px; padding-bottom: 5px;}
    .report-item { font-size: 0.95rem; color: #888; margin-bottom: 6px; display: flex; justify-content: space-between; }
    .highlight { color: #00f260; font-weight: bold; font-family: 'Rajdhani'; }
    .highlight-bear { color: #ff003c; font-weight: bold; font-family: 'Rajdhani'; }
    .highlight-warn { color: #ffcc00; font-weight: bold; font-family: 'Rajdhani'; }

    /* BUTTONS */
    .stButton > button {
        width: 100%; border-radius: 6px; font-weight: bold; font-family: 'Rajdhani';
        background: #111; color: #00f260; border: 1px solid #00f260;
    }
    .stButton > button:hover { background: #00f260; color: #000; }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #666; font-size: 0.8rem; border: none; font-weight: 700; }
    .stTabs [aria-selected="true"] { color: #00f260; border-bottom: 2px solid #00f260; }
</style>
""", unsafe_allow_html=True)

# SECRETS
api_key = st.secrets.get("OPENAI_API_KEY")
tg_token = st.secrets.get("TELEGRAM_TOKEN")
tg_chat = st.secrets.get("TELEGRAM_CHAT_ID")

# ==========================================
# 2. MASTER DATASETS & UNIVERSE
# ==========================================
ASSET_CLASSES = {
    "CRYPTO (Major)": {
        "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana", "XRP-USD": "Ripple",
        "BNB-USD": "Binance", "DOGE-USD": "Dogecoin", "ADA-USD": "Cardano"
    },
    "INDICES": {
        "^SPX": "S&P 500", "^IXIC": "Nasdaq 100", "^DJI": "Dow Jones", "^RUT": "Russell 2000"
    },
    "FOREX": {
        "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "JPY=X": "USD/JPY"
    },
    "TECH": {
        "NVDA": "Nvidia", "TSLA": "Tesla", "AAPL": "Apple", "MSFT": "Microsoft", "AMD": "AMD"
    },
    "COMMODITIES": {
        "GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil"
    }
}

# Full Crypto Scanner Universe
SCANNER_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD", "AVAX-USD", 
    "LINK-USD", "MATIC-USD", "DOT-USD", "NEAR-USD", "ATOM-USD", "ARB-USD", 
    "OP-USD", "SUI-USD", "APT-USD", "UNI-USD", "AAVE-USD", "MKR-USD", 
    "INJ-USD", "RUNE-USD", "DOGE-USD", "SHIB-USD", "PEPE-USD", "WIF-USD", 
    "FET-USD", "RNDR-USD"
]

# ==========================================
# 3. CORE ENGINE (INTEGRATED)
# ==========================================
class CoreEngine:
    @staticmethod
    def _hma(series, length):
        """Hull Moving Average (Math)"""
        if len(series) < length: return pd.Series(0, index=series.index)
        half = int(length / 2); sqrt = int(np.sqrt(length))
        wma_half = series.rolling(half).apply(lambda x: np.dot(x, np.arange(1, half+1)) / (half*(half+1)/2), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / (length*(length+1)/2), raw=True)
        diff = 2 * wma_half - wma_full
        return diff.rolling(sqrt).apply(lambda x: np.dot(x, np.arange(1, sqrt+1)) / (sqrt*(sqrt+1)/2), raw=True)

    @staticmethod
    def calculate_adx(df, length=14):
        """ADX Calculation (Strength)"""
        up = df['High'].diff()
        down = -df['Low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        tr = (df['High'] - df['Low']).ewm(alpha=1/length, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/length, adjust=False).mean() / tr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/length, adjust=False).mean() / tr)
        
        sum_di = plus_di + minus_di
        sum_di = sum_di.replace(0, 1) 
        dx = 100 * np.abs(plus_di - minus_di) / sum_di
        return dx.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def calculate_wavetrend(df):
        """WaveTrend Calculation (Momentum)"""
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=10, adjust=False).mean()
        d = (ap - esa).abs().ewm(span=10, adjust=False).mean()
        d = d.replace(0, 0.0001)
        ci = (ap - esa) / (0.015 * d)
        tci = ci.ewm(span=21, adjust=False).mean() 
        return tci

    @staticmethod
    def process_data(df, amp=10, dev=3.0, hma_len=55):
        """
        MASTER LOGIC:
        Merges Physics (Omega), Trend (Apex), and Execution (Titan)
        """
        if len(df) < 60: return df
        df = df.copy()
        
        # 1. PHYSICS (RQZO & Entropy)
        vel = df['Close'].pct_change().abs(); c = 0.05
        gamma = 1 / np.sqrt(1 - (np.minimum(vel, c*0.99)/c)**2)
        entropy = (df['Close'].pct_change().rolling(20).std() - 0) / 0.05 
        zeta = np.sin((np.arange(len(df))%100)/gamma) * np.exp(-2*np.abs(entropy-0.6))*10
        df['RQZO'] = zeta.fillna(0); df['Entropy'] = entropy.fillna(0.5)

        # 2. VECTOR FLUX
        eff = ((df['Close']-df['Open']).abs()/(df['High']-df['Low']).replace(0,1)).ewm(span=14).mean()
        vol_ma = df['Volume'].rolling(55).mean()
        flux = (np.sign(df['Close']-df['Open']) * eff * (df['Volume']/vol_ma)).ewm(span=5).mean()
        df['Flux'] = flux
        df['State'] = np.select([flux>0.6, flux<-0.6, flux.abs()<0.3], ['Super_Bull', 'Super_Bear', 'Resistive'], 'Heat')
        
        # 3. APEX CONTEXT (HMA Cloud)
        hma = CoreEngine._hma(df['Close'], hma_len)
        atr = (df['High']-df['Low']).rolling(hma_len).mean()
        df['Cloud_Upper'] = hma + atr*1.5
        df['Cloud_Lower'] = hma - atr*1.5
        df['Apex_Trend'] = np.where(df['Close'] > df['Cloud_Upper'], 1, np.where(df['Close'] < df['Cloud_Lower'], -1, np.nan))
        df['Apex_Trend'] = df['Apex_Trend'].ffill()

        # 4. MOMENTUM SUITE (Apex Integration)
        df['ADX'] = CoreEngine.calculate_adx(df)
        df['WaveTrend'] = CoreEngine.calculate_wavetrend(df)
        
        # Apex Buy Signal
        df['Apex_Buy_Signal'] = (
            (df['Apex_Trend'] == 1) & 
            (df['WaveTrend'] < 60) & 
            (df['WaveTrend'] > df['WaveTrend'].shift(1)) &
            (df['ADX'] > 20) &
            (df['Volume'] > df['Volume'].rolling(20).mean())
        )

        # 5. EXECUTION (Titan Smart Stop)
        df['atr_fast'] = (df['High']-df['Low']).ewm(alpha=1/14).mean()
        df['ll'] = df['Low'].rolling(amp).min()
        df['hh'] = df['High'].rolling(amp).max()
        
        trend = np.zeros(len(df)); stop = np.full(len(df), np.nan)
        curr_t = 0; curr_s = np.nan
        
        for i in range(amp, len(df)):
            c_p = df['Close'].iloc[i]
            d = df['atr_fast'].iloc[i] * dev
            
            if curr_t == 0: # Bull
                s = df['ll'].iloc[i] + d
                curr_s = max(curr_s, s) if not np.isnan(curr_s) else s
                if c_p < curr_s:
                    curr_t = 1 # Switch to Bear
                    curr_s = df['hh'].iloc[i] - d
            else: # Bear
                s = df['hh'].iloc[i] - d
                curr_s = min(curr_s, s) if not np.isnan(curr_s) else s
                if c_p > curr_s:
                    curr_t = 0 # Switch to Bull
                    curr_s = df['ll'].iloc[i] + d
            
            trend[i] = curr_t
            stop[i] = curr_s
            
        df['Trend'] = trend # 0=Bull, 1=Bear
        df['Smart_Stop'] = stop
        
        # 6. TARGETS (TP1/2/3)
        df['Trend_Switch'] = df['Trend'].diff().abs()
        df['Entry_Price'] = np.where(df['Trend_Switch']==1, df['Close'], np.nan)
        df['Entry_Price'] = df['Entry_Price'].ffill()
        df['Risk'] = (df['Entry_Price'] - df['Smart_Stop']).abs()
        
        df['TP1'] = np.where(df['Trend']==0, df['Entry_Price']+(df['Risk']*1.5), df['Entry_Price']-(df['Risk']*1.5))
        df['TP2'] = np.where(df['Trend']==0, df['Entry_Price']+(df['Risk']*3.0), df['Entry_Price']-(df['Risk']*3.0))
        df['TP3'] = np.where(df['Trend']==0, df['Entry_Price']+(df['Risk']*5.0), df['Entry_Price']-(df['Risk']*5.0))

        # 7. SQUEEZE & SMC ALERTS
        bb_basis = df['Close'].rolling(20).mean()
        bb_dev = df['Close'].rolling(20).std() * 2.0
        kc_dev = df['atr_fast'] * 1.5
        df['In_Squeeze'] = ((bb_basis - bb_dev) > (bb_basis - kc_dev)) & ((bb_basis + bb_dev) < (bb_basis + kc_dev))
        
        # SMC Detection (BOS/FVG)
        pivot = 10
        df['BOS_Alert'] = (df['Close'] > df['High'].shift(1).rolling(pivot*2).max()) & (df['Close'].shift(1) <= df['High'].shift(2).rolling(pivot*2).max())
        df['FVG_Detected'] = (df['Low'] > df['High'].shift(2)) | (df['High'] < df['Low'].shift(2))
        
        # Gann Sq9 (from V18)
        sqrt_p = np.sqrt(df['Close'])
        df['Gann_Res'] = (sqrt_p + 1)**2
        df['Gann_Sup'] = (sqrt_p - 1)**2
        
        return df

    @staticmethod
    def scan_smc_zones(df):
        """Identify Order Blocks & FVG for Charting"""
        structs = {'OB': [], 'FVG': []}
        pivot = 10
        for i in range(pivot, len(df)):
            # Bullish OB
            if df['Trend'].iloc[i] == 0 and df['Close'].iloc[i] > df['High'].iloc[i-pivot]:
                ob = df.iloc[i-15:i][df.iloc[i-15:i]['Close'] < df.iloc[i-15:i]['Open']]
                if not ob.empty: 
                    structs['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'type': 'bull'})
            # Bearish OB
            if df['Trend'].iloc[i] == 1 and df['Close'].iloc[i] < df['Low'].iloc[i-pivot]:
                ob = df.iloc[i-15:i][df.iloc[i-15:i]['Close'] > df.iloc[i-15:i]['Open']]
                if not ob.empty: 
                    structs['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'type': 'bear'})
            # FVG
            if i > 2:
                # Bull FVG
                if df['Low'].iloc[i-2] > df['High'].iloc[i]:
                    structs['FVG'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i], 'y1': df['Low'].iloc[i-2], 'type': 'bull'})
                # Bear FVG
                if df['High'].iloc[i-2] < df['Low'].iloc[i]:
                    structs['FVG'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i], 'type': 'bear'})
        return structs

    @staticmethod
    def get_fundamentals(ticker):
        if any(x in ticker for x in ["-", "=", "^"]): return None
        try:
            i = yf.Ticker(ticker).info
            return {"ticker": ticker, "name": i.get('shortName', ticker), "market_cap": i.get("marketCap", 0)}
        except: return None

# ==========================================
# 4. MARKET SCANNER (APEX INTEGRATION)
# ==========================================
def run_market_scanner(universe):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(universe)
    for i, ticker in enumerate(universe):
        status_text.text(f"Scanning Omega Metric Grid: {ticker}...")
        progress_bar.progress((i+1)/total)
        
        try:
            df = yf.download(ticker, period="6mo", progress=False)
            if df.empty or len(df) < 60: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # Run Full Engine
            df = CoreEngine.process_data(df)
            last = df.iloc[-1]
            
            # Scoring Logic
            score = 0
            tags = []
            
            if last['Trend'] == 0: 
                score += 1
                tags.append("BULL")
            else:
                tags.append("BEAR")
            
            if last['State'] == 'Super_Bull':
                score += 2
                tags.append("SUPER FLUX")
            
            if last['Apex_Buy_Signal']:
                score += 3
                tags.append("APEX BUY")
            
            if last['BOS_Alert']:
                score += 2
                tags.append("BOS")
            if last['FVG_Detected']:
                score += 1
                tags.append("FVG")
                
            results.append({
                "Ticker": ticker,
                "Price": last['Close'],
                "Trend": "🟢 BULL" if last['Trend'] == 0 else "🔴 BEAR",
                "Flux": last['State'],
                "WaveTrend": round(last['WaveTrend'], 1),
                "ADX": round(last['ADX'], 1),
                "Score": score,
                "Tags": ", ".join(tags)
            })
            
        except: continue
        
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results).sort_values(by="Score", ascending=False)

# ==========================================
# 5. STRATEGY ENGINE
# ==========================================
@dataclass
class Trade:
    timestamp: datetime; symbol: str; side: str; price: float; quantity: float; pnl: Optional[float] = None

class StrategyEngine:
    def __init__(self, df, mode):
        self.df = df
        self.mode = mode
        self.trades = []
        self.equity = [10000.0]
        self.cash = 10000.0
        self.position = 0
        self.entry_px = 0.0

    def run(self):
        for i in range(50, len(self.df)):
            row = self.df.iloc[i]
            prev = self.df.iloc[i-1]
            signal = "HOLD"
            
            # 1. FLUX STRATEGY
            if self.mode == "Flux":
                if row['State'] == 'Super_Bull': signal = "BUY"
                elif row['State'] == 'Super_Bear': signal = "SELL"
            
            # 2. TREND FOLLOWING
            elif self.mode == "Trend Following":
                if row['Trend'] == 0 and prev['Trend'] == 1: signal = "BUY"
                elif row['Trend'] == 1 and prev['Trend'] == 0: signal = "SELL"

            # 3. APEX SIGNAL
            elif self.mode == "Apex Signal":
                if row['Apex_Buy_Signal']: signal = "BUY"
                elif row['Trend'] == 1: signal = "SELL"
                
            # 4. ENTROPY
            elif self.mode == "Entropy":
                if row['Entropy'] < 0.5 and row['Apex_Trend'] == 1: signal = "BUY"
                elif row['Entropy'] > 0.8: signal = "SELL"

            # EXECUTION
            if signal == "BUY" and self.position == 0:
                qty = self.cash / row['Close']
                self.position = qty
                self.entry_px = row['Close']
                self.cash = 0
                self.trades.append(Trade(row.name, "TEST", "BUY", row['Close'], qty))
            elif signal == "SELL" and self.position > 0:
                val = self.position * row['Close']
                pnl = val - (self.position * self.entry_px)
                self.cash = val
                self.position = 0
                self.equity.append(val)
                self.trades.append(Trade(row.name, "TEST", "SELL", row['Close'], 0, pnl))
        
        return self.equity, self.trades

# ==========================================
# 6. VISUALIZATION (HYBRID CHART)
# ==========================================
def render_hybrid_chart(df, structs):
    last = df.iloc[-1]
    
    # Setup Subplots: Row 1 = Main, Row 2 = Oscillator
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

    # --- MAIN CHART ---
    # 1. Context Cloud
    fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Upper'], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    c_fill = 'rgba(0, 242, 96, 0.1)' if last['Apex_Trend']==1 else 'rgba(255, 0, 60, 0.1)'
    fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Lower'], fill='tonexty', fillcolor=c_fill, line=dict(width=0), name="Context Cloud"), row=1, col=1)

    # 2. SMC Zones (Liquidity)
    # 
    for ob in structs['OB'][-8:]:
        col = 'rgba(0, 230, 118, 0.2)' if ob['type'] == 'bull' else 'rgba(255, 23, 68, 0.2)'
        fig.add_shape(type="rect", x0=ob['x0'], x1=df.index[-1], y0=ob['y0'], y1=ob['y1'], fillcolor=col, line_width=0, row=1, col=1)
    
    # 3. Candles 
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                                 increasing_line_color='#00f260', decreasing_line_color='#ff003c', name="Price"), row=1, col=1)

    # 4. Smart Stop
    bull_stop = df[df['Trend']==0]['Smart_Stop']
    bear_stop = df[df['Trend']==1]['Smart_Stop']
    fig.add_trace(go.Scatter(x=bull_stop.index, y=bull_stop, mode='markers', marker=dict(color='#00f260', size=2), name="Bull Stop"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bear_stop.index, y=bear_stop, mode='markers', marker=dict(color='#ff003c', size=2), name="Bear Stop"), row=1, col=1)

    # 5. Targets & Gann
    if not np.isnan(last['TP1']):
        col_tp = '#00f260' if last['Trend']==0 else '#ff003c'
        fig.add_hline(y=last['TP1'], line_dash="dot", line_color=col_tp, annotation_text="TP1", row=1, col=1)
        fig.add_hline(y=last['TP2'], line_dash="dot", line_color=col_tp, annotation_text="TP2", row=1, col=1)
    
    fig.add_hline(y=last['Gann_Res'], line_dash="dash", line_color="rgba(255,255,255,0.2)", row=1, col=1)
    fig.add_hline(y=last['Gann_Sup'], line_dash="dash", line_color="rgba(255,255,255,0.2)", row=1, col=1)

    # --- OSCILLATOR ---
    # RQZO
    fig.add_trace(go.Scatter(x=df.index, y=df['RQZO'], line=dict(color='#2979FF', width=1.5), name="RQZO"), row=2, col=1)
    # Entropy Overlay
    fig.add_trace(go.Scatter(x=df.index, y=df['Entropy'], line=dict(color='#aa00ff', width=1), name="Entropy"), row=2, col=1)
    fig.add_hline(y=0.8, line_color="#ff003c", line_dash="dot", row=2, col=1)
    
    # Layout 
    fig.update_layout(template="plotly_dark", paper_bgcolor="#000000", plot_bgcolor="#080808", height=600, 
                      margin=dict(l=0, r=50, t=10, b=10), hovermode="x unified", showlegend=False, 
                      xaxis=dict(showgrid=False, rangeslider=dict(visible=False)), 
                      yaxis=dict(showgrid=True, gridcolor="#222", side="right"))
    return fig

# ==========================================
# 7. MAIN APPLICATION
# ==========================================
def main():
    # --- HEADER ---
    st.markdown("""<div class="ticker-wrap"><div class="ticker"><span class="ticker-item">BTC-USD</span><span class="ticker-item">ETH-USD</span><span class="ticker-item">SOL-USD</span><span class="ticker-item">SPX</span><span class="ticker-item">NDX</span><span class="ticker-item">EURUSD</span><span class="ticker-item">NVDA</span></div></div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="omega-header">OMEGA <span class="omega-accent">TERMINAL</span></div>""", unsafe_allow_html=True)

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("SYSTEM CONTROL")
        st.info("System v4.0 | Integrated")
        
        # Asset Selection
        asset_cls = st.selectbox("Class", list(ASSET_CLASSES.keys()))
        asset_tkr = st.selectbox("Asset", list(ASSET_CLASSES[asset_cls].keys()), format_func=lambda x: f"{x} ({ASSET_CLASSES[asset_cls][x]})")
        man_tkr = st.text_input("Manual Ticker")
        
        # Settings
        tf = st.selectbox("Timeframe", ["1h", "4h", "1d", "1wk"], index=2)
        
        with st.expander("Engine Tuning"):
            amp = st.slider("Amplitude", 5, 50, 10)
            dev = st.slider("Deviation", 1.0, 5.0, 3.0)
            hma_l = st.slider("Trend Length", 20, 200, 55)

    # --- DEEP DIVE ENGINE ---
    ticker = man_tkr.upper().strip() if man_tkr else asset_tkr
    p_map = {"1h": "1y", "4h": "1y", "1d": "2y", "1wk": "5y"}
    
    try:
        with st.spinner(f"INITIALIZING LINK: {ticker}..."):
            df = yf.download(ticker, interval=tf, period=p_map[tf], progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                # RUN MASTER ENGINE
                df = CoreEngine.process_data(df, amp, dev, hma_l)
                structs = CoreEngine.scan_smc_zones(df)
                last = df.iloc[-1]

        # --- TABS ---
        t1, t2, t3, t4, t5, t6 = st.tabs(["👁️ VISION", "🔭 SCANNER", "📝 REPORT", "🧪 LAB", "🧠 AI", "📡 UPLINK"])

        if not df.empty:
            with t1: # CHART
                st.plotly_chart(render_hybrid_chart(df, structs), use_container_width=True)
                
                # HUD METRICS
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("TREND", "BULL 🐂" if last['Trend']==0 else "BEAR 🐻", f"{last['Close']:.2f}")
                m2.metric("FLUX", last['State'].replace('_', ' '), f"WT: {last['WaveTrend']:.1f}")
                m3.metric("ADX", f"{last['ADX']:.1f}", "Strength")
                m4.metric("ENTROPY", f"{last['Entropy']:.2f}", f"Squeeze: {last['In_Squeeze']}")

        with t2: # MARKET SCANNER
            st.subheader("🌐 MARKET SCANNER (OMEGA METRICS)")
            if st.button("RUN SCAN"):
                scan_df = run_market_scanner(SCANNER_UNIVERSE)
                if not scan_df.empty:
                    st.dataframe(scan_df, use_container_width=True)
                    
                    # Excel Download
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                        scan_df.to_excel(writer, index=False, sheet_name="Omega Scan")
                    st.download_button("📥 DOWNLOAD REPORT", buf.getvalue(), "omega_scan.xlsx")

        if not df.empty:
            with t3: # MOBILE REPORT
                conf_score = 0
                if last['Trend'] == (0 if last['Apex_Trend']==1 else 1): conf_score += 1
                if abs(last['Flux']) > 0.5: conf_score += 1
                confidence = "MAX 🔥" if conf_score == 2 else "HIGH" if conf_score == 1 else "LOW"
                
                sqz_txt = "⚠️ ACTIVE" if last['In_Squeeze'] else "NONE"
                
                rpt_html = f"""
                <div class="report-card">
                    <div class="report-header">💠 SIGNAL: {"BULL" if last['Trend']==0 else "BEAR"}</div>
                    <div class="report-item">Confidence: <span class="highlight">{confidence}</span></div>
                    <div class="report-item">Squeeze: <span class="{ 'highlight-bear' if last['In_Squeeze'] else 'highlight'}">{sqz_txt}</span></div>
                    <div class="report-item">WaveTrend: <span class="highlight">{last['WaveTrend']:.1f}</span></div>
                    <div class="report-item">ADX: <span class="highlight">{last['ADX']:.1f}</span></div>
                </div>
                <div class="report-card">
                    <div class="report-header">🎯 EXECUTION</div>
                    <div class="report-item">Entry: <span class="highlight">{last['Entry_Price']:.2f}</span></div>
                    <div class="report-item">🛑 STOP: <span class="highlight-bear">{last['Smart_Stop']:.2f}</span></div>
                    <div class="report-item">1️⃣ TP1 (1.5R): <span class="highlight">{last['TP1']:.2f}</span></div>
                    <div class="report-item">2️⃣ TP2 (3.0R): <span class="highlight">{last['TP2']:.2f}</span></div>
                </div>
                """
                st.markdown(rpt_html, unsafe_allow_html=True)

            with t4: # BACKTEST LAB
                st.subheader("STRATEGY SIMULATION")
                strat = st.selectbox("Algorithm", ["Trend Following", "Flux", "Apex Signal", "Entropy"])
                
                if st.button("RUN SIMULATION", use_container_width=True):
                    eng = StrategyEngine(df, strat)
                    eq, trades = eng.run()
                    
                    ret = ((eq[-1] - 10000)/10000)*100
                    wins = len([t for t in trades if t.pnl and t.pnl > 0])
                    total = len([t for t in trades if t.pnl is not None])
                    wr = (wins/total*100) if total > 0 else 0
                    
                    c_b1, c_b2, c_b3 = st.columns(3)
                    c_b1.metric("Net Return", f"{ret:.2f}%")
                    c_b2.metric("Win Rate", f"{wr:.1f}%")
                    c_b3.metric("Trades", total)
                    st.area_chart(eq, color="#00f260")
                    st.dataframe([t.__dict__ for t in trades], use_container_width=True)

            with t5: # AI ANALYST (UPDATED)
                st.subheader("TACTICAL AI")
                if not api_key: st.warning("Add OPENAI_API_KEY to secrets")
                else:
                    # Apex-style Prompt Structure
                    p_prompt = f"""
                    Act as an SMC Crypto Trader. Analyze this setup based on Omega Engine metrics.
                    
                    [ASSET] {ticker} (${last['Close']:.5f})
                    [TREND] {"BULL" if last['Trend']==0 else "BEAR"}
                    
                    [SIGNALS]
                    Apex Signal: {last['Apex_Buy_Signal']}
                    Flux State: {last['State']}
                    BOS Alert: {last['BOS_Alert']}
                    FVG Detected: {last['FVG_Detected']}
                    
                    [MOMENTUM]
                    WaveTrend: {last['WaveTrend']:.1f}
                    ADX: {last['ADX']:.1f}
                    Entropy: {last['Entropy']:.2f}
                    
                    OUTPUT REQUIREMENTS:
                    1. VERDICT: "Strong Long", "Scalp Long", "Wait", or "Short".
                    2. SMC CONTEXT: Explain the BOS/FVG vs Trend.
                    3. EXECUTION: Entry strategy.
                    """
                    if st.button("GENERATE INTEL"):
                        try:
                            client = OpenAI(api_key=api_key)
                            res = client.chat.completions.create(model="gpt-4", messages=[{"role":"user", "content": p_prompt}])
                            st.success(res.choices[0].message.content)
                        except Exception as e: st.error(str(e))
                        
            with t6: # UPLINK
                st.subheader("TELEGRAM UPLINK")
                if not tg_token: st.warning("Add TELEGRAM_TOKEN to secrets")
                else:
                    msg = f"OMEGA ALERT: {ticker} | Trend: {'BULL' if last['Trend']==0 else 'BEAR'} | Flux: {last['State']} | WT: {last['WaveTrend']:.1f}"
                    if st.button("SEND ALERT"):
                        try:
                            # Send Text
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", json={"chat_id": tg_chat, "text": msg})
                            
                            # Generate & Send Excel Report (V18 Feature)
                            buf = io.BytesIO()
                            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                                df.tail(50).to_excel(writer, sheet_name="Omega Data")
                            buf.seek(0)
                            requests.post(f"https://api.telegram.org/bot{tg_token}/sendDocument", data={"chat_id": tg_chat}, files={"document": (f"{ticker}_report.xlsx", buf, "application/vnd.ms-excel")})
                            
                            st.success("SENT")
                        except Exception as e: st.error(str(e))

    except Exception as e:
        st.error(f"SYSTEM CRITICAL: {str(e)}")

if __name__ == "__main__":
    main()
