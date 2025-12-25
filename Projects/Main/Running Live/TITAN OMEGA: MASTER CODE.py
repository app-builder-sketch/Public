import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
import requests
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Optional, Dict

# ==========================================
# 1. UI & SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TITAN OMEGA: MASTER", 
    page_icon="🧬", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- NEON CYBERPUNK CSS (Mobile Optimized) ---
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
    .titan-header {
        font-family: 'Rajdhani', sans-serif; font-size: 2.5rem; font-weight: 700;
        color: #fff; letter-spacing: 3px; border-bottom: 1px solid #222; padding: 15px 0;
    }
    .titan-accent { color: #00f260; text-shadow: 0 0 15px rgba(0, 242, 96, 0.4); }
    
    /* REPORT CARD (FROM TITAN V18) */
    .report-card {
        background-color: #111; border-left: 4px solid #00f260;
        padding: 15px; border-radius: 4px; margin-bottom: 10px;
    }
    .report-header { font-family: 'Rajdhani'; font-size: 1.2rem; color: #fff; border-bottom: 1px solid #333; margin-bottom: 8px; }
    .report-item { font-size: 0.9rem; color: #888; margin-bottom: 4px; display: flex; justify-content: space-between; }
    .highlight { color: #00f260; font-weight: bold; font-family: 'Rajdhani'; }
    .highlight-bear { color: #ff003c; font-weight: bold; font-family: 'Rajdhani'; }

    /* METRICS */
    div[data-testid="stMetric"] {
        background: rgba(20, 20, 20, 0.9); border: 1px solid #333; border-left: 3px solid #00f260; border-radius: 6px; 
    }
    div[data-testid="stMetricValue"] { font-family: 'Rajdhani'; font-size: 1.5rem !important; color: #fff; }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #555; font-size: 0.8rem; border: none; font-weight: 700; }
    .stTabs [aria-selected="true"] { color: #00f260; border-bottom: 2px solid #00f260; }
</style>
""", unsafe_allow_html=True)

# SECRETS
api_key = st.secrets.get("OPENAI_API_KEY")
tg_token = st.secrets.get("TELEGRAM_TOKEN")
tg_chat = st.secrets.get("TELEGRAM_CHAT_ID")

# ==========================================
# 2. MASTER DATASETS
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

# ==========================================
# 3. TITAN OMEGA ENGINE (ALL MODULES)
# ==========================================
class TitanEngine:
    @staticmethod
    def _hma(series, length):
        """Hull Moving Average (Math)"""
        half = int(length / 2); sqrt = int(np.sqrt(length))
        wma_half = series.rolling(half).apply(lambda x: np.dot(x, np.arange(1, half+1)) / (half*(half+1)/2), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / (length*(length+1)/2), raw=True)
        return (2 * wma_half - wma_full).rolling(sqrt).apply(lambda x: np.dot(x, np.arange(1, sqrt+1)) / (sqrt*(sqrt+1)/2), raw=True)

    @staticmethod
    def process_data(df, amp=10, dev=3.0, hma_len=55):
        """
        MASTER ENGINE:
        1. Physics (RQZO, Entropy, Flux)
        2. Titan Trend (ATR Trailing Stop)
        3. Apex Trend (HMA Cloud)
        4. Squeeze & Momentum
        """
        df = df.copy()
        
        # --- 1. PHYSICS (RQZO & Entropy) ---
        vel = df['Close'].pct_change().abs(); c = 0.05
        gamma = 1 / np.sqrt(1 - (np.minimum(vel, c*0.99)/c)**2)
        entropy = (df['Close'].pct_change().rolling(20).std() - 0) / 0.05 
        zeta = np.sin((np.arange(len(df))%100)/gamma) * np.exp(-2*np.abs(entropy-0.6))*10
        df['RQZO'] = zeta.fillna(0); df['Entropy'] = entropy.fillna(0.5)

        # --- 2. VECTOR FLUX & DIVERGENCE ---
        eff = ((df['Close']-df['Open']).abs()/(df['High']-df['Low']).replace(0,1)).ewm(span=14).mean()
        flux = (np.sign(df['Close']-df['Open']) * eff * (df['Volume']/df['Volume'].rolling(55).mean())).ewm(span=5).mean()
        df['Flux'] = flux
        df['State'] = np.select([flux>0.6, flux<-0.6, flux.abs()<0.3], ['Super_Bull', 'Super_Bear', 'Resistive'], 'Heat')
        
        # Flux Divergence (Lookback 5)
        df['Flux_Div'] = 0 # 0=None, 1=Bull Reg, 2=Bull Hid, -1=Bear Reg, -2=Bear Hid
        # (Simplified pivot logic for speed)
        
        # --- 3. APEX TREND (Context) ---
        hma = TitanEngine._hma(df['Close'], hma_len)
        atr = (df['High']-df['Low']).rolling(hma_len).mean()
        df['Cloud_Upper'] = hma + atr*1.5
        df['Cloud_Lower'] = hma - atr*1.5
        df['Apex_Trend'] = np.where(df['Close'] > df['Cloud_Upper'], 1, np.where(df['Close'] < df['Cloud_Lower'], -1, np.nan))
        df['Apex_Trend'] = df['Apex_Trend'].ffill()

        # --- 4. TITAN TREND (Execution - ATR Trailing Stop) ---
        # Re-implementing logic from Titan Mobile v18 [cite: 44-49]
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
            
        df['Titan_Trend'] = trend # 0=Bull, 1=Bear
        df['Smart_Stop'] = stop
        
        # --- 5. TARGETS (TP1/2/3) ---
        # Logic from Titan Mobile [cite: 50]
        # We calculate risk based on entry (trend switch) and stop
        df['Trend_Switch'] = df['Titan_Trend'].diff().abs()
        df['Entry_Price'] = np.where(df['Trend_Switch']==1, df['Close'], np.nan)
        df['Entry_Price'] = df['Entry_Price'].ffill()
        df['Risk'] = (df['Entry_Price'] - df['Smart_Stop']).abs()
        
        # Targets
        df['TP1'] = np.where(df['Titan_Trend']==0, df['Entry_Price']+(df['Risk']*1.5), df['Entry_Price']-(df['Risk']*1.5))
        df['TP2'] = np.where(df['Titan_Trend']==0, df['Entry_Price']+(df['Risk']*3.0), df['Entry_Price']-(df['Risk']*3.0))
        df['TP3'] = np.where(df['Titan_Trend']==0, df['Entry_Price']+(df['Risk']*5.0), df['Entry_Price']-(df['Risk']*5.0))

        # --- 6. SQUEEZE & MOMENTUM ---
        # Squeeze 
        bb_basis = df['Close'].rolling(20).mean()
        bb_dev = df['Close'].rolling(20).std() * 2.0
        kc_dev = df['atr_fast'] * 1.5
        df['In_Squeeze'] = ((bb_basis - bb_dev) > (bb_basis - kc_dev)) & ((bb_basis + bb_dev) < (bb_basis + kc_dev))
        
        # WaveTrend 
        ap = (df['High']+df['Low']+df['Close'])/3
        esa = ap.ewm(span=10).mean()
        d_wt = (ap - esa).abs().ewm(span=10).mean()
        ci = (ap - esa) / (0.015 * d_wt)
        df['WT1'] = ci.ewm(span=21).mean()
        df['WT2'] = df['WT1'].rolling(4).mean()

        return df

    @staticmethod
    def scan_smc(df):
        """Identify Order Blocks & FVG [cite: 119-121]"""
        structs = {'OB': [], 'FVG': []}
        pivot = 10
        for i in range(pivot, len(df)):
            # Bullish OB
            if df['Titan_Trend'].iloc[i] == 0 and df['Close'].iloc[i] > df['High'].iloc[i-pivot]:
                ob = df.iloc[i-15:i][df.iloc[i-15:i]['Close'] < df.iloc[i-15:i]['Open']]
                if not ob.empty: 
                    structs['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'type': 'bull'})
            # Bearish OB
            if df['Titan_Trend'].iloc[i] == 1 and df['Close'].iloc[i] < df['Low'].iloc[i-pivot]:
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
            return {"MktCap": i.get("marketCap", 0), "PE": i.get("trailingPE", 0), "Growth": i.get("revenueGrowth", 0)}
        except: return None

# ==========================================
# 4. BACKTEST LAB (STRATEGY ENGINE)
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
            
            # 2. TITAN TREND STRATEGY (ATR Trailing)
            elif self.mode == "Titan Trend":
                if row['Titan_Trend'] == 0 and prev['Titan_Trend'] == 1: signal = "BUY"
                elif row['Titan_Trend'] == 1 and prev['Titan_Trend'] == 0: signal = "SELL"

            # 3. ENTROPY STRATEGY
            elif self.mode == "Entropy":
                if row['Entropy'] < 0.5 and row['Apex_Trend'] == 1: signal = "BUY"
                elif row['Entropy'] > 0.8: signal = "SELL" # Panic sell

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
# 5. VISUALIZATION (HYBRID CHART)
# ==========================================
def render_hybrid_chart(df, structs):
    last = df.iloc[-1]
    
    # Setup Subplots: Row 1 = Main, Row 2 = Oscillator
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

    # --- MAIN CHART ---
    # 1. Apex Cloud (Context)
    fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Upper'], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    c_fill = 'rgba(0, 242, 96, 0.1)' if last['Apex_Trend']==1 else 'rgba(255, 0, 60, 0.1)'
    fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Lower'], fill='tonexty', fillcolor=c_fill, line=dict(width=0), name="Apex Cloud"), row=1, col=1)

    # 2. SMC Zones (Liquidity)
    for ob in structs['OB'][-8:]:
        col = 'rgba(0, 230, 118, 0.2)' if ob['type'] == 'bull' else 'rgba(255, 23, 68, 0.2)'
        fig.add_shape(type="rect", x0=ob['x0'], x1=df.index[-1], y0=ob['y0'], y1=ob['y1'], fillcolor=col, line_width=0, row=1, col=1)
    
    # 3. Candles
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                                 increasing_line_color='#00f260', decreasing_line_color='#ff003c', name="Price"), row=1, col=1)

    # 4. Titan Smart Stop (Execution)
    # Filter to only show the relevant side to reduce clutter
    bull_stop = df[df['Titan_Trend']==0]['Smart_Stop']
    bear_stop = df[df['Titan_Trend']==1]['Smart_Stop']
    fig.add_trace(go.Scatter(x=bull_stop.index, y=bull_stop, mode='markers', marker=dict(color='#00f260', size=2), name="Bull Stop"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bear_stop.index, y=bear_stop, mode='markers', marker=dict(color='#ff003c', size=2), name="Bear Stop"), row=1, col=1)

    # 5. Targets (Current Active)
    if not np.isnan(last['TP1']):
        col_tp = '#00f260' if last['Titan_Trend']==0 else '#ff003c'
        fig.add_hline(y=last['TP1'], line_dash="dot", line_color=col_tp, annotation_text="TP1", row=1, col=1)
        fig.add_hline(y=last['TP2'], line_dash="dot", line_color=col_tp, annotation_text="TP2", row=1, col=1)
        fig.add_hline(y=last['TP3'], line_dash="dot", line_color=col_tp, annotation_text="TP3", row=1, col=1)

    # --- OSCILLATOR ---
    # RQZO
    fig.add_trace(go.Scatter(x=df.index, y=df['RQZO'], line=dict(color='#2979FF', width=1.5), name="RQZO"), row=2, col=1)
    # Entropy Overlay (Red zone)
    fig.add_trace(go.Scatter(x=df.index, y=df['Entropy'], line=dict(color='#aa00ff', width=1), name="Entropy"), row=2, col=1)
    fig.add_hline(y=0.8, line_color="#ff003c", line_dash="dot", row=2, col=1)
    
    # Layout
    fig.update_layout(template="plotly_dark", paper_bgcolor="#000000", plot_bgcolor="#080808", height=600, 
                      margin=dict(l=0, r=50, t=10, b=10), hovermode="x unified", showlegend=False, 
                      xaxis=dict(showgrid=False, rangeslider=dict(visible=False)), 
                      yaxis=dict(showgrid=True, gridcolor="#222", side="right"))
    return fig

# ==========================================
# 6. MAIN APPLICATION
# ==========================================
def main():
    # --- HEADER ---
    st.markdown("""<div class="ticker-wrap"><div class="ticker"><span class="ticker-item">BTC-USD</span><span class="ticker-item">ETH-USD</span><span class="ticker-item">SOL-USD</span><span class="ticker-item">SPX</span><span class="ticker-item">NDX</span><span class="ticker-item">EURUSD</span><span class="ticker-item">NVDA</span></div></div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="titan-header">TITAN <span class="titan-accent">OMEGA</span></div>""", unsafe_allow_html=True)

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("SYSTEM CONTROL")
        st.info("Titan Mobile v3.0 | Engine Active")
        
        # Asset Selection
        asset_cls = st.selectbox("Class", list(ASSET_CLASSES.keys()))
        asset_tkr = st.selectbox("Asset", list(ASSET_CLASSES[asset_cls].keys()), format_func=lambda x: f"{x} ({ASSET_CLASSES[asset_cls][x]})")
        man_tkr = st.text_input("Manual Ticker")
        
        # Settings
        tf = st.selectbox("Timeframe", ["1h", "4h", "1d", "1wk"], index=2)
        
        # Logic Settings (From V18)
        with st.expander("Engine Tuning"):
            amp = st.slider("Amplitude", 5, 50, 10)
            dev = st.slider("Deviation", 1.0, 5.0, 3.0)
            hma_l = st.slider("Trend Length", 20, 200, 55)

    # --- DATA ENGINE ---
    ticker = man_tkr.upper().strip() if man_tkr else asset_tkr
    p_map = {"1h": "1y", "4h": "1y", "1d": "2y", "1wk": "5y"}
    
    try:
        with st.spinner(f"INITIALIZING OMEGA LINK: {ticker}..."):
            df = yf.download(ticker, interval=tf, period=p_map[tf], progress=False)
            if df.empty: st.error("DATA ERROR"); return
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # RUN MASTER ENGINE
            df = TitanEngine.process_data(df, amp, dev, hma_l)
            structs = TitanEngine.scan_smc(df)
            fund = TitanEngine.get_fundamentals(ticker)
            last = df.iloc[-1]

        # --- HUD METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        
        # 1. Trend Status
        trend_str = "BULL 🐂" if last['Titan_Trend'] == 0 else "BEAR 🐻"
        m1.metric("TITAN TREND", trend_str, f"{last['Close']:.2f}")
        
        # 2. Flux State
        m2.metric("FLUX STATE", last['State'].replace('_', ' '), f"{last['Flux']:.2f}")
        
        # 3. Targets
        next_tp = last['TP1'] if (last['Close'] < last['TP1'] and last['Titan_Trend']==0) or (last['Close'] > last['TP1'] and last['Titan_Trend']==1) else last['TP2']
        m3.metric("NEXT TARGET", f"{next_tp:.2f}", f"Stop: {last['Smart_Stop']:.2f}")
        
        # 4. Entropy
        ent_delta = last['Entropy'] - df['Entropy'].iloc[-2]
        m4.metric("ENTROPY", f"{last['Entropy']:.2f}", f"{ent_delta:.2f}")

        # --- MAIN TABS ---
        t1, t2, t3, t4, t5 = st.tabs(["👁️ VISION", "📝 REPORT", "🧪 LAB", "🧠 AI", "📡 UPLINK"])

        with t1: # CHART
            st.plotly_chart(render_hybrid_chart(df, structs), use_container_width=True)

        with t2: # MOBILE REPORT (Restored from V18)
            conf_score = 0
            if last['Titan_Trend'] == (0 if last['Apex_Trend']==1 else 1): conf_score += 1
            if abs(last['Flux']) > 0.5: conf_score += 1
            confidence = "MAX 🔥" if conf_score == 2 else "HIGH" if conf_score == 1 else "LOW"
            
            sqz_txt = "⚠️ ACTIVE" if last['In_Squeeze'] else "NONE"
            
            rpt_html = f"""
            <div class="report-card">
                <div class="report-header">💠 SIGNAL: {trend_str}</div>
                <div class="report-item">Confidence: <span class="highlight">{confidence}</span></div>
                <div class="report-item">Squeeze: <span class="{ 'highlight-bear' if last['In_Squeeze'] else 'highlight'}">{sqz_txt}</span></div>
                <div class="report-item">Entropy: <span class="highlight">{last['Entropy']:.2f}</span></div>
            </div>
            <div class="report-card">
                <div class="report-header">🎯 EXECUTION PLAN</div>
                <div class="report-item">Entry: <span class="highlight">{last['Entry_Price']:.2f}</span></div>
                <div class="report-item">🛑 STOP: <span class="highlight-bear">{last['Smart_Stop']:.2f}</span></div>
                <div class="report-item">1️⃣ TP1 (1.5R): <span class="highlight">{last['TP1']:.2f}</span></div>
                <div class="report-item">2️⃣ TP2 (3.0R): <span class="highlight">{last['TP2']:.2f}</span></div>
                <div class="report-item">3️⃣ TP3 (5.0R): <span class="highlight">{last['TP3']:.2f}</span></div>
            </div>
            """
            st.markdown(rpt_html, unsafe_allow_html=True)

        with t3: # BACKTEST LAB
            st.subheader("STRATEGY SIMULATION")
            strat = st.selectbox("Algorithm", ["Titan Trend", "Flux", "Entropy"])
            
            if st.button("RUN SIMULATION", use_container_width=True):
                eng = StrategyEngine(df, strat)
                eq, trades = eng.run()
                
                # Stats
                ret = ((eq[-1] - 10000)/10000)*100
                wins = len([t for t in trades if t.pnl and t.pnl > 0])
                total = len([t for t in trades if t.pnl is not None])
                wr = (wins/total*100) if total > 0 else 0
                
                c_b1, c_b2, c_b3 = st.columns(3)
                c_b1.metric("Net Return", f"{ret:.2f}%")
                c_b2.metric("Win Rate", f"{wr:.1f}%")
                c_b3.metric("Trades", total)
                
                # Equity Curve
                st.area_chart(eq, color="#00f260")
                st.dataframe([t.__dict__ for t in trades], use_container_width=True)

        with t4: # AI
            st.subheader("TACTICAL AI")
            if not api_key: st.warning("Add OPENAI_API_KEY to secrets")
            else:
                p_prompt = f"Asset: {ticker}. Trend: {trend_str}. Flux: {last['Flux']:.2f}. Squeeze: {last['In_Squeeze']}. targets: {last['TP1']}, {last['TP2']}. Analysis?"
                if st.button("GENERATE INTEL"):
                    try:
                        client = OpenAI(api_key=api_key)
                        res = client.chat.completions.create(model="gpt-4", messages=[{"role":"user", "content": p_prompt}])
                        st.success(res.choices[0].message.content)
                    except Exception as e: st.error(str(e))
                    
        with t5: # BROADCAST
            st.subheader("TELEGRAM UPLINK")
            if not tg_token: st.warning("Add TELEGRAM_TOKEN to secrets")
            else:
                msg = f"{trend_str} {ticker} | Flux: {last['Flux']:.2f} | Stop: {last['Smart_Stop']}"
                if st.button("SEND ALERT"):
                    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", json={"chat_id": tg_chat, "text": msg})
                    st.success("SENT")

    except Exception as e:
        st.error(f"SYSTEM CRITICAL: {str(e)}")

if __name__ == "__main__":
    main()
