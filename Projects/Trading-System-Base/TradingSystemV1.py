
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
import requests
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict

# ==========================================
# 1. UI CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TITAN OMEGA", 
    page_icon="🧬", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- PRO CYBERPUNK CSS ENGINE ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;800&display=swap');
    
    /* GLOBAL THEME */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    
    /* SCROLLING TICKER */
    .ticker-wrap {
        width: 100%; overflow: hidden; background: #000; border-bottom: 1px solid #222;
        padding: 4px 0; white-space: nowrap; margin-bottom: 10px;
    }
    .ticker { display: inline-block; animation: marquee 60s linear infinite; }
    .ticker-item { display: inline-block; padding: 0 2rem; font-family: 'Rajdhani'; color: #00e676; font-weight: bold; letter-spacing: 1px; }
    @keyframes marquee { 0% { transform: translate3d(0, 0, 0); } 100% { transform: translate3d(-50%, 0, 0); } }

    /* CUSTOM HUD CARDS */
    .hud-grid {
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px;
    }
    .hud-card {
        background: rgba(15, 20, 25, 0.95);
        border: 1px solid #333; border-left: 4px solid #00e676;
        border-radius: 8px; padding: 15px;
        display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .hud-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; font-family: 'Inter'; margin-bottom: 5px; }
    .hud-value { font-size: 1.6rem; color: #fff; font-weight: 700; font-family: 'Rajdhani'; line-height: 1; }
    .hud-sub { font-size: 0.8rem; color: #00e676; margin-top: 5px; font-weight: 600; }
    .hud-sub.neg { color: #ff1744; }

    /* CONTROLS */
    .stTextInput>div>div>input { background-color: #111; color: #00e676; border: 1px solid #333; font-family: 'Rajdhani'; }
    .stSelectbox>div>div>div { background-color: #111; color: #fff; border: 1px solid #333; }
    button[kind="primary"] { 
        background: linear-gradient(90deg, #00e676, #00b0ff); 
        border: none; color: #000; font-weight: 800; font-family: 'Rajdhani'; text-transform: uppercase; letter-spacing: 1px;
    }
    
    /* MOBILE CARD */
    .mobile-card {
        background: #080808; border: 1px solid #333; border-radius: 8px; padding: 15px;
        margin-top: 10px; border-left: 3px solid #e040fb; font-family: 'Inter';
    }
    .mobile-prop { font-size: 0.75rem; color: #888; display: flex; justify-content: space-between; margin-bottom: 4px; }
    .mobile-val { color: #fff; font-weight: bold; font-family: 'Rajdhani'; }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #222; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #666; font-size: 0.8rem; font-weight: 700; border: none; }
    .stTabs [aria-selected="true"] { color: #00e676; border-bottom: 2px solid #00e676; }
</style>
""", unsafe_allow_html=True)

# SECRETS AUTO-LOADER
api_key = st.secrets.get("OPENAI_API_KEY")
tg_token = st.secrets.get("TELEGRAM_TOKEN")
tg_chat = st.secrets.get("TELEGRAM_CHAT_ID")

# ==========================================
# 2. DATASETS & METADATA
# ==========================================
APP_METADATA = {
    "Name": "Terminal",
    "Version": "1.0",
    "ID": "1043709",
    "Package": "com.mycompany.terminal",
    "Key": "GyGZYiqBriidIkwvlPFWAFCsIxcygvlB",
    "Author": "My Company (UK)",
    "Keystore": "my-release-key.jks",
    "BuildDate": "2025-12-23"
}

ASSET_CLASSES = {
    "CRYPTO": {
        "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana", "XRP-USD": "Ripple",
        "BNB-USD": "Binance", "DOGE-USD": "Dogecoin", "ADA-USD": "Cardano", "AVAX-USD": "Avalanche",
        "LINK-USD": "Chainlink", "LTC-USD": "Litecoin", "DOT-USD": "Polkadot"
    },
    "INDICES": {
        "^SPX": "S&P 500", "^IXIC": "Nasdaq 100", "^DJI": "Dow Jones", "^RUT": "Russell 2000",
        "^VIX": "Volatility", "^FTSE": "FTSE 100"
    },
    "FOREX": {
        "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "JPY=X": "USD/JPY", "AUDUSD=X": "AUD/USD"
    },
    "TECH": {
        "NVDA": "Nvidia", "TSLA": "Tesla", "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon",
        "GOOGL": "Google", "META": "Meta", "AMD": "AMD", "PLTR": "Palantir"
    },
    "COMMODITIES": {
        "GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil", "NG=F": "Nat Gas"
    }
}

# ==========================================
# 3. PHYSICS & MATH ENGINE
# ==========================================
class TitanEngine:
    @staticmethod
    def _hma(series, length):
        """Hull Moving Average"""
        half = int(length / 2); sqrt = int(np.sqrt(length))
        wma_half = series.rolling(half).apply(lambda x: np.dot(x, np.arange(1, half+1)) / (half*(half+1)/2), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / (length*(length+1)/2), raw=True)
        return (2 * wma_half - wma_full).rolling(sqrt).apply(lambda x: np.dot(x, np.arange(1, sqrt+1)) / (sqrt*(sqrt+1)/2), raw=True)

    @staticmethod
    def process_data(df):
        """Core Physics Processing"""
        df = df.copy()
        
        # 1. RQZO (Relativistic Quantum-Zeta)
        vel = df['Close'].pct_change().abs(); c = 0.05
        gamma = 1 / np.sqrt(1 - (np.minimum(vel, c*0.99)/c)**2)
        entropy = (df['Close'].pct_change().rolling(20).std() - 0) / 0.05 
        zeta = np.sin((np.arange(len(df))%100)/gamma) * np.exp(-2*np.abs(entropy-0.6))*10
        df['RQZO'] = zeta.fillna(0); df['Entropy'] = entropy.fillna(0.5)

        # 2. Vector Flux
        eff = ((df['Close']-df['Open']).abs()/(df['High']-df['Low']).replace(0,1)).ewm(span=14).mean()
        flux = (np.sign(df['Close']-df['Open']) * eff * (df['Volume']/df['Volume'].rolling(55).mean())).ewm(span=5).mean()
        df['Flux'] = flux; df['State'] = np.select([flux>0.6, flux<-0.6, flux.abs()<0.3], ['Super_Bull', 'Super_Bear', 'Resistive'], 'Heat')

        # 3. Trend Cloud
        hma = TitanEngine._hma(df['Close'], 55); atr = (df['High']-df['Low']).rolling(55).mean()
        df['Cloud_Upper'] = hma + atr*1.5; df['Cloud_Lower'] = hma - atr*1.5
        df['Trend'] = np.where(df['Close'] > df['Cloud_Upper'], 1, np.where(df['Close'] < df['Cloud_Lower'], -1, 0))
        df['Trend'] = df['Trend'].replace(0, method='ffill')

        # 4. Pattern Recognition
        body = (df['Close'] - df['Open']).abs()
        wick_up = df['High'] - df[['Open', 'Close']].max(axis=1)
        wick_dn = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Pat_Bull'] = (wick_dn > body*2) & (wick_up < body) # Hammer
        df['Pat_Bear'] = (wick_up > body*2) & (wick_dn < body) # Shooting Star
        
        # 5. Gann Levels
        last_price = df['Close'].iloc[-1]; sqrt_p = np.sqrt(last_price)
        df['Gann_Res'] = (sqrt_p + 1)**2; df['Gann_Sup'] = (sqrt_p - 1)**2
        
        return df

    @staticmethod
    def get_mtf_bias(ticker):
        try:
            d = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if d.empty: return 0
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            hma = TitanEngine._hma(d['Close'], 55).iloc[-1]
            return 1 if d['Close'].iloc[-1] > hma else -1
        except: return 0

    @staticmethod
    def scan_structures(df):
        structs = {'OB': [], 'FVG': []}; pivot = 10
        for i in range(pivot, len(df)):
            # OB
            if df['Trend'].iloc[i] == 1 and df['Close'].iloc[i] > df['High'].iloc[i-pivot]:
                ob = df.iloc[i-15:i][df.iloc[i-15:i]['Close'] < df.iloc[i-15:i]['Open']]
                if not ob.empty: structs['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'col': 'rgba(0, 230, 118, 0.2)', 'bor': '#00e676'})
            if df['Trend'].iloc[i] == -1 and df['Close'].iloc[i] < df['Low'].iloc[i-pivot]:
                ob = df.iloc[i-15:i][df.iloc[i-15:i]['Close'] > df.iloc[i-15:i]['Open']]
                if not ob.empty: structs['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'col': 'rgba(255, 23, 68, 0.2)', 'bor': '#ff1744'})
            # FVG
            if i > 2:
                if df['Low'].iloc[i-2] > df['High'].iloc[i]: 
                    structs['FVG'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i], 'y1': df['Low'].iloc[i-2], 'col': 'rgba(255, 214, 0, 0.15)', 'bor': '#ffd600'})
                if df['High'].iloc[i-2] < df['Low'].iloc[i]:
                    structs['FVG'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i], 'col': 'rgba(255, 214, 0, 0.15)', 'bor': '#ffd600'})
        return structs

    @staticmethod
    def get_vpvr(df):
        try:
            bins = np.linspace(df['Low'].min(), df['High'].max(), 50)
            df['Bin'] = pd.cut(df['Close'], bins=bins, include_lowest=True)
            vp = df.groupby('Bin', observed=True)['Volume'].sum().reset_index()
            return vp, vp.loc[vp['Volume'].idxmax(), df['Bin'].apply(lambda x: x.mid).astype(float)]
        except: return pd.DataFrame(), 0.0

    @staticmethod
    def run_monte_carlo(df, days=30, sims=50):
        last = df['Close'].iloc[-1]; vol = df['Close'].pct_change().std()
        sim_df = pd.DataFrame()
        for x in range(sims): sim_df[x] = [last * (1 + np.random.normal(0, vol))**d for d in range(days)]
        return sim_df

    @staticmethod
    def get_seasonality(df):
        try:
            df = df.copy(); df['Ret'] = df['Close'].pct_change() * 100
            df['Month'] = df.index.month
            return df.groupby('Month')['Ret'].mean()
        except: return None

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_fundamentals(ticker):
        if any(x in ticker for x in ["-", "=", "^"]): return None
        try:
            i = yf.Ticker(ticker).info
            return {"MktCap": i.get("marketCap", 0), "PE": i.get("trailingPE", 0), "Growth": i.get("revenueGrowth", 0)}
        except: return None

    @staticmethod
    def calc_correlations(ticker):
        try:
            a = {"SPX": "^SPX", "GOLD": "GC=F", "BTC": "BTC-USD"}
            df = yf.download(list(a.values()) + [ticker], period="3mo", progress=False)['Close']
            return df.corr()[ticker].drop(ticker).sort_values(ascending=False)
        except: return None

# ==========================================
# 4. BACKTEST ENGINE (PHYSICS ENABLED)
# ==========================================
@dataclass
class Trade:
    timestamp: datetime
    symbol: str
    side: str
    price: float
    quantity: float
    commission: float
    pnl: Optional[float] = None

@dataclass
class Position:
    symbol: str
    quantity: float
    average_entry_price: float

class PhysicsStrategy:
    def __init__(self, mode: str):
        self.mode = mode 
        self.name = f"Titan: {mode}"

    def generate_signal(self, row: pd.Series) -> str:
        # Integrated Physics Signals
        if self.mode == "Flux":
            if row['State'] == "Super_Bull": return 'BUY'
            elif row['State'] == "Super_Bear": return 'SELL'
        elif self.mode == "Trend":
            if row['Trend'] == 1: return 'BUY'
            elif row['Trend'] == -1: return 'SELL'
        elif self.mode == "Entropy":
            if row['Entropy'] < 0.5 and row['Trend'] == 1: return 'BUY'
            elif row['Entropy'] > 0.8: return 'SELL'
        elif self.mode == "RQZO":
            if row['RQZO'] > 0 and row['Trend'] == 1: return 'BUY'
            elif row['RQZO'] < 0 and row['Trend'] == -1: return 'SELL'
        return 'HOLD'

class PortfolioManager:
    def __init__(self, initial_capital: float, transaction_cost: float = 0.001):
        self.cash = initial_capital
        self.positions: List[Position] = []
        self.trade_log: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        self.transaction_cost = transaction_cost

    def execute_trade(self, signal: str, tick: pd.Series, symbol: str):
        price = tick['Close']
        timestamp = tick.name 
        position = next((p for p in self.positions if p.symbol == symbol), None)
        
        if signal == 'BUY':
            if self.cash > 0:
                qty = (self.cash * 0.99) / price
                cost = qty * price
                comm = cost * self.transaction_cost
                if self.cash >= (cost + comm):
                    self.cash -= (cost + comm)
                    if position:
                        total_qty = position.quantity + qty
                        avg_px = ((position.quantity * position.average_entry_price) + cost) / total_qty
                        position.quantity = total_qty
                        position.average_entry_price = avg_px
                    else:
                        self.positions.append(Position(symbol, qty, price))
                    self.trade_log.append(Trade(timestamp, symbol, 'BUY', price, qty, comm))

        elif signal == 'SELL' and position:
            if position.quantity > 0:
                rev = position.quantity * price
                comm = rev * self.transaction_cost
                self.cash += (rev - comm)
                pnl = (price - position.average_entry_price) * position.quantity - comm
                self.trade_log.append(Trade(timestamp, symbol, 'SELL', price, position.quantity, comm, pnl))
                self.positions.remove(position)

    def update_equity(self, current_price: float, timestamp: datetime):
        pos_val = sum([p.quantity * current_price for p in self.positions])
        self.equity_curve.append(self.cash + pos_val)
        self.timestamps.append(timestamp)

class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(portfolio: PortfolioManager, initial_capital: float) -> Dict:
        if not portfolio.equity_curve: return {}
        equity_series = pd.Series(portfolio.equity_curve)
        returns = equity_series.pct_change().dropna()
        final_equity = portfolio.equity_curve[-1]
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
        trades = [t for t in portfolio.trade_log if t.side == 'SELL']
        if not trades: return {'total_return': total_return, 'final_equity': final_equity, 'total_trades': len(portfolio.trade_log)}
        wins = [t for t in trades if t.pnl > 0]
        win_rate = (len(wins) / len(trades)) * 100
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        max_dd = ((equity_series / equity_series.expanding().max()) - 1).min() * 100
        return {'total_return': total_return, 'final_equity': final_equity, 'total_trades': len(trades), 'win_rate': win_rate, 'sharpe': sharpe, 'max_dd': max_dd}

# ==========================================
# 5. VISUALIZATION (EXPERT GRAPHICS)
# ==========================================
def render_pro_chart(df, structs, vp, ticker):
    last = df.iloc[-1]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])

    # Volume Watermark
    vol_c = ['rgba(0, 230, 118, 0.15)' if c >= o else 'rgba(255, 23, 68, 0.15)' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_c, showlegend=False, hoverinfo='skip'), row=1, col=1)
    
    # Cloud
    c_fill = 'rgba(0, 230, 118, 0.05)' if last['Trend']==1 else 'rgba(255, 23, 68, 0.05)'
    fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Upper'], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Lower'], fill='tonexty', fillcolor=c_fill, line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)

    # Zones
    for ob in structs['OB'][-5:]: fig.add_shape(type="rect", x0=ob['x0'], x1=df.index[-1], y0=ob['y0'], y1=ob['y1'], fillcolor=ob['col'], line=dict(color=ob['bor'], width=1), row=1, col=1)
    for fvg in structs['FVG'][-5:]: fig.add_shape(type="rect", x0=fvg['x0'], x1=fvg['x1'], y0=fvg['y0'], y1=fvg['y1'], fillcolor=fvg['col'], line=dict(color=fvg['bor'], width=0, dash='dot'), row=1, col=1)

    # Gann
    fig.add_hline(y=last['Gann_Res'], line_dash="dot", line_color="rgba(255, 255, 255, 0.3)")
    fig.add_hline(y=last['Gann_Sup'], line_dash="dot", line_color="rgba(255, 255, 255, 0.3)")

    # Price
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], increasing_line_color='#00e676', decreasing_line_color='#ff1744', name="Price"), row=1, col=1)
    
    # Physics Panel
    fig.add_trace(go.Scatter(x=df.index, y=df['RQZO'], line=dict(color='#00b0ff', width=1.5), name="RQZO"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Entropy'], line=dict(color='#9c27b0', width=1), name="Entropy"), row=2, col=1)
    fig.add_hline(y=0.8, line_color="#ff1744", line_dash="dot", row=2, col=1)

    # Watermark
    fig.add_annotation(text=ticker, xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=80, color="rgba(255,255,255,0.05)"), showarrow=False, row=1, col=1)

    fig.update_layout(template="plotly_dark", paper_bgcolor="#000000", plot_bgcolor="#050505", height=600, margin=dict(l=0, r=50, t=10, b=10), hovermode="x unified", showlegend=False, xaxis=dict(showgrid=False, rangeslider=dict(visible=False)), yaxis=dict(showgrid=True, gridcolor="#222", side="right"))
    return fig

# ==========================================
# 6. MAIN APP
# ==========================================
def main():
    st.markdown("""<div class="ticker-wrap"><div class="ticker"><span class="ticker-item">BTC-USD</span><span class="ticker-item">ETH-USD</span><span class="ticker-item">SOL-USD</span><span class="ticker-item">SPX</span><span class="ticker-item">NDX</span><span class="ticker-item">EURUSD</span><span class="ticker-item">NVDA</span></div></div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="titan-header">TITAN <span class="titan-accent">OMEGA</span></div>""", unsafe_allow_html=True)

    # SIDEBAR: MANUAL & APP METADATA
    with st.sidebar:
        with st.expander("📘 OPERATOR MANUAL (READ ME)", expanded=True):
            st.markdown("""
            **1. BATTLE GRID (HUD)**
            * **STRUCTURE:** HMA Cloud Trend (Above = Bull, Below = Bear).
            * **FLUX:** Institutional Volume Flow.
            * **ENTROPY:** Chaos Meter. >0.8 = Unpredictable.
            
            **2. PHYSICS LAB (BACKTEST)**
            * Select **Titan Strategy** to test.
            * **Flux:** Buys "Super Bull".
            * **Entropy:** Buys Stability.
            * **RQZO:** Buys Zero-Cross.
            """)
        
        st.markdown("---")
        # INTEGRATING UPLOADED FILE DATA (Terminal 1.0.zip info)
        with st.expander("📱 MOBILE ECOSYSTEM", expanded=False):
            st.markdown(f"""
            <div class="mobile-card">
                <div class="mobile-prop"><span class="mobile-val">{APP_METADATA['Name']}</span> <span style="color:#00f260">● LIVE</span></div>
                <div class="mobile-prop"><span>Ver</span> <span class="mobile-val">{APP_METADATA['Version']}</span></div>
                <div class="mobile-prop"><span>ID</span> <span class="mobile-val">{APP_METADATA['ID']}</span></div>
                <div class="mobile-prop"><span>Pkg</span> <span class="mobile-val" style="font-size:0.65rem">{APP_METADATA['Package']}</span></div>
                <hr style="border-color:#333; margin:8px 0;">
                <div class="mobile-prop"><span>Auth</span> <span class="mobile-val">{APP_METADATA['Author']}</span></div>
                <div class="mobile-prop"><span>Key</span> <span class="mobile-val" style="font-size:0.6rem">...{APP_METADATA['Key'][-6:]}</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.button("📥 DOWNLOAD APK (v1.0)", use_container_width=True)

    # ASSET SELECTION
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1: cls_sel = st.selectbox("CLASS", list(ASSET_CLASSES.keys()))
    with c2: tick_sel = st.selectbox("ASSET", list(ASSET_CLASSES[cls_sel].keys()), format_func=lambda x: f"{x} ({ASSET_CLASSES[cls_sel][x]})")
    with c3: man_sel = st.text_input("SEARCH OVERRIDE")
    with c4: tf_sel = st.selectbox("TF", ["1h", "4h", "1d", "1wk"], index=2)

    ticker = man_sel.upper().strip() if man_sel else tick_sel
    per_map = {"1h": "1y", "4h": "1y", "1d": "2y", "1wk": "5y"}

    try:
        with st.spinner("SYNCING OMEGA CORE..."):
            mtf = TitanEngine.get_mtf_bias(ticker)
            bias_c = "#00e676" if mtf==1 else "#ff1744" if mtf==-1 else "#666"
            bias_t = "DAILY: BULL" if mtf==1 else "DAILY: BEAR" if mtf==-1 else "DAILY: FLAT"

            df = yf.download(ticker, interval=tf_sel, period=per_map[tf_sel], progress=False)
            if df.empty: st.error("NO DATA"); return
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            df = TitanEngine.process_physics(df)
            structs = TitanEngine.scan_structures(df)
            vp, poc = TitanEngine.get_vpvr(df)
            mc_sim = TitanEngine.run_monte_carlo(df)
            seas = TitanEngine.get_seasonality(df)
            fund = TitanEngine.get_fundamentals(ticker)
            corr = TitanEngine.calc_correlations(ticker)
            last = df.iloc[-1]

        # HUD
        st.markdown(f"""
        <div class="hud-grid">
            <div class="hud-card">
                <div class="hud-label">Structure</div>
                <div class="hud-value" style="color:{bias_c}">{'BULL' if last['Trend']==1 else 'BEAR'}</div>
                <div class="hud-sub">{bias_t}</div>
            </div>
            <div class="hud-card">
                <div class="hud-label">Flux State</div>
                <div class="hud-value">{last['State'].replace('_',' ')}</div>
                <div class="hud-sub">{last['Flux']:.2f}</div>
            </div>
            <div class="hud-card">
                <div class="hud-label">Entropy</div>
                <div class="hud-value">{last['Entropy']:.2f}</div>
                <div class="hud-sub {'neg' if last['Entropy']>0.8 else ''}">{'CHAOS' if last['Entropy']>0.8 else 'STABLE'}</div>
            </div>
            <div class="hud-card">
                <div class="hud-label">POC Level</div>
                <div class="hud-value">{poc:.2f}</div>
                <div class="hud-sub">Volume Control</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # TABS
        t1, t2, t3, t4, t5 = st.tabs(["👁️ CHART", "🔬 INTEL", "🧪 LAB", "🧠 AI", "📡 UPLINK"])

        with t1:
            st.plotly_chart(render_pro_chart(df, structs, vp, ticker), use_container_width=True, config={'displayModeBar': False})

        with t2:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("MONTE CARLO")
                fig_mc = go.Figure()
                for i in range(min(15, len(mc_sim.columns))): fig_mc.add_trace(go.Scatter(y=mc_sim[i], mode='lines', line=dict(color='rgba(255,255,255,0.05)'), showlegend=False))
                fig_mc.add_trace(go.Scatter(y=mc_sim.mean(axis=1), mode='lines', line=dict(color='#00e676', width=2), name="Mean"))
                fig_mc.update_layout(template="plotly_dark", height=200, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="#000000")
                st.plotly_chart(fig_mc, use_container_width=True)
                if seas is not None:
                    st.caption("SEASONALITY")
                    fig_s = go.Figure(go.Bar(x=seas.index, y=seas.values, marker_color=seas.values, marker_colorscale="RdBu"))
                    fig_s.update_layout(template="plotly_dark", height=150, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="#000000")
                    st.plotly_chart(fig_s, use_container_width=True)
            with c2:
                if corr is not None: st.caption("CORRELATIONS"); st.dataframe(corr.to_frame().T, use_container_width=True)
                if fund: 
                    st.caption("FUNDAMENTALS")
                    c_f1, c_f2 = st.columns(2)
                    c_f1.metric("MKT CAP", f"${fund['MktCap']/1e9:.1f}B"); c_f2.metric("P/E", f"{fund['PE']:.1f}")

        with t3: # PHYSICS BACKTEST LAB
            st.subheader("QUANTUM STRATEGY SIMULATION")
            strat_name = st.selectbox("STRATEGY", ["Flux", "Trend", "Entropy", "RQZO"])
            strategy = PhysicsStrategy(strat_name)
            
            if st.button("RUN SIMULATION", use_container_width=True):
                portfolio = PortfolioManager(10000.0)
                start_idx = 50 
                for i in range(start_idx, len(df)):
                    row = df.iloc[i]
                    sig = strategy.generate_signal(row)
                    if sig != 'HOLD':
                        portfolio.execute_trade(sig, row, ticker)
                    portfolio.update_equity(row['Close'], row.name)
                
                # Metrics Calculation
                metrics = PerformanceAnalyzer.calculate_metrics(portfolio, 10000.0)
                
                if portfolio.equity_curve:
                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.metric("TOTAL RETURN", f"{metrics.get('total_return', 0):.2f}%")
                    c_m2.metric("SHARPE RATIO", f"{metrics.get('sharpe', 0):.2f}")
                    c_m3.metric("MAX DRAWDOWN", f"{metrics.get('max_dd', 0):.2f}%")

                    eq_df = pd.DataFrame({'Date': portfolio.timestamps, 'Equity': portfolio.equity_curve})
                    fig_eq = px.line(eq_df, x='Date', y='Equity', title=f"Equity Curve: {strategy.name}")
                    fig_eq.update_layout(template="plotly_dark", paper_bgcolor="#000000", height=300)
                    
                    # Trades on Equity
                    buys = [t for t in portfolio.trade_log if t.side == 'BUY']
                    sells = [t for t in portfolio.trade_log if t.side == 'SELL']
                    if buys: fig_eq.add_trace(go.Scatter(x=[t.timestamp for t in buys], y=[portfolio.equity_curve[portfolio.timestamps.index(t.timestamp)] for t in buys], mode='markers', marker=dict(color='#00e676', size=8, symbol='triangle-up'), name="Buy"))
                    if sells: fig_eq.add_trace(go.Scatter(x=[t.timestamp for t in sells], y=[portfolio.equity_curve[portfolio.timestamps.index(t.timestamp)] for t in sells], mode='markers', marker=dict(color='#ff003c', size=8, symbol='triangle-down'), name="Sell"))
                    
                    st.plotly_chart(fig_eq, use_container_width=True)
                    
                    if portfolio.trade_log:
                        st.dataframe(pd.DataFrame([{
                            'Time': t.timestamp, 'Side': t.side, 'Price': t.price, 
                            'Qty': t.quantity, 'PnL': t.pnl
                        } for t in portfolio.trade_log]), use_container_width=True)
                else:
                    st.info("No trades triggered by Physics Engine.")

        with t4: # AI OPS
            st.subheader("TACTICAL AI")
            if not api_key: st.warning("⚠️ ADD 'OPENAI_API_KEY' TO SECRETS"); api_in = st.text_input("Manual Key", type="password")
            else: api_in = api_key
            persona = st.selectbox("MODE", ["SCALPER", "SWING", "MACRO"])
            if st.button("EXECUTE", use_container_width=True):
                if api_in:
                    p = f"Role: {persona}. Asset: {ticker}. Bias: {bias_t}. Trend: {last['Trend']}. Entropy: {last['Entropy']:.2f}. Pattern: {'Hammer' if last['Pat_Bull'] else 'Star' if last['Pat_Bear'] else 'None'}. Verdict?"
                    try:
                        client = OpenAI(api_key=api_in)
                        res = client.chat.completions.create(model="gpt-4", messages=[{"role":"user", "content":p}])
                        st.info(res.choices[0].message.content)
                    except Exception as e: st.error(str(e))

        with t5: # BROADCAST
            st.subheader("BROADCAST")
            if not tg_token: st.info("⚠️ ADD 'TELEGRAM_TOKEN' & 'TELEGRAM_CHAT_ID' TO SECRETS"); u_tok = st.text_input("Token"); u_chat = st.text_input("Chat")
            else: u_tok, u_chat = tg_token, tg_chat
            
            sig_msg = f"{'🟢' if last['Trend']==1 else '🔴'} **TITAN:** {ticker}\nBias: {bias_t}\nFlux: {last['State']}\nPrice: {last['Close']:.2f}"
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                if st.button("🚀 QUICK SIGNAL", use_container_width=True):
                    if u_tok and u_chat:
                        try: requests.post(f"https://api.telegram.org/bot{u_tok}/sendMessage", data={"chat_id": u_chat, "text": sig_msg}); st.success("SENT")
                        except: st.error("FAIL")
            with c_b2:
                if st.button("📝 FULL REPORT", use_container_width=True):
                    if u_tok and u_chat:
                        full_rep = f"{sig_msg}\n\n**METRICS:**\nEntropy: {last['Entropy']:.2f}\nPOC: {poc:.2f}\nMC Forecast: ${mc_sim.mean(axis=1).iloc[-1]:.2f}"
                        try: requests.post(f"https://api.telegram.org/bot{u_tok}/sendMessage", data={"chat_id": u_chat, "text": full_rep}); st.success("SENT")
                        except: st.error("FAIL")

    except Exception as e: st.error(f"SYSTEM HALT: {e}")

if __name__ == "__main__":
    main()
