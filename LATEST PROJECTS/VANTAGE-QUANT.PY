import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
import datetime
import time
import logging
import io
import xlsxwriter
from typing import Optional, Dict, List, Tuple

# ==========================================
# 1. SYSTEM CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    layout="wide", 
    page_title="Vantage Quantitative Terminal", 
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

# Institutional Grade CSS Styling
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp { background-color: #0e1117; color: #f0f2f6; font-family: 'Roboto', sans-serif; }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #1a1c24; 
        border: 1px solid #2d303e;
        padding: 15px; 
        border-radius: 4px; 
        border-left: 4px solid #4a4e69;
    }
    div[data-testid="stMetric"]:hover { border-left: 4px solid #00ceff; }
    
    /* Headers */
    .terminal-header {
        font-size: 2.2em; font-weight: 700; color: #ffffff;
        border-bottom: 2px solid #2d303e; padding-bottom: 10px; margin-bottom: 20px;
        letter-spacing: 1px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1c24; border-radius: 4px 4px 0 0; 
        border: 1px solid #2d303e; color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d303e; color: #ffffff; border-bottom-color: #00ceff; font-weight: 600;
    }
    
    /* Alerts */
    .success-tag { color: #00ff00; font-weight: bold; }
    .fail-tag { color: #ff0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("VantageTerm")

# ==========================================
# 2. CORE ENGINES (Logic Layer)
# ==========================================

# --- UNIVERSE CONFIGURATION ---
CRYPTO_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD", "AVAX-USD", 
    "LINK-USD", "MATIC-USD", "DOT-USD", "NEAR-USD", "ATOM-USD", "ARB-USD", 
    "OP-USD", "SUI-USD", "APT-USD", "UNI-USD", "AAVE-USD", "MKR-USD", 
    "INJ-USD", "RUNE-USD", "DOGE-USD", "SHIB-USD", "PEPE-USD", "WIF-USD"
]

class DataEngine:
    """Enterprise Data Ingestion Layer"""
    
    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_price_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
        try:
            # Buffer fetch to ensure statistical significance for indicators
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
            
            # Robust MultiIndex Handling
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Standardize
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required): return pd.DataFrame()
            
            df = df.dropna(subset=['Close'])
            df['Volume'] = df['Volume'].fillna(0)
            return df
        except Exception: return pd.DataFrame()

    @staticmethod
    def get_fundamentals(ticker: str) -> Dict:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            return {
                "MarketCap": info.get("marketCap", "N/A"),
                "Sector": info.get("sector", "Crypto"),
                "Summary": info.get("longBusinessSummary", "No data available.")[:450]
            }
        except: return {}

class IndicatorEngine:
    """Deep Dive Analytics Module"""
    
    @staticmethod
    def calc_fibonacci(df: pd.DataFrame, window=50) -> Dict[str, float]:
        recent = df.iloc[-window:]
        high, low = recent['High'].max(), recent['Low'].min()
        diff = high - low
        return {
            "0.0 (Support)": low, 
            "0.382": low + 0.382 * diff,
            "0.50 (Mid)": low + 0.5 * diff, 
            "0.618 (Gold)": low + 0.618 * diff,
            "1.0 (Resist)": high
        }

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. HMA (Hull Moving Average)
        p = 55
        wma = lambda s, l: s.rolling(l).apply(lambda x: np.dot(x, np.arange(1, l+1))/np.arange(1, l+1).sum(), raw=True)
        hm = wma(2 * wma(df['Close'], p//2) - wma(df['Close'], p), int(np.sqrt(p)))
        df['HMA'] = hm
        
        # 2. RSI
        delta = df['Close'].diff()
        rs = (delta.where(delta>0,0).rolling(14).mean()) / (-delta.where(delta<0,0).rolling(14).mean())
        df['RSI'] = 100 - (100/(1+rs))
        
        # 3. ADX & Regime
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        plus = 100 * (df['High'].diff().clip(lower=0).ewm(alpha=1/14).mean() / atr)
        minus = 100 * (-df['Low'].diff().clip(upper=0).ewm(alpha=1/14).mean() / atr)
        df['ADX'] = (abs(plus-minus)/(plus+minus)*100).rolling(14).mean()
        df['Regime'] = np.where(df['ADX']>25, "Trending", "Consolidating")
        
        # 4. Squeeze & Momentum
        basis = df['Close'].rolling(20).mean()
        dev = df['Close'].rolling(20).std()
        df['Squeeze_On'] = (basis-(2*dev) > basis-(1.5*tr.rolling(20).mean())) & \
                           (basis+(2*dev) < basis+(1.5*tr.rolling(20).mean()))
        df['Momentum'] = df['Close'] - df['Close'].rolling(20).mean()
        
        # 5. Signal Logic
        cond = [(df['Close']>df['HMA'])&(df['RSI']>50)&(df['Momentum']>0),
                (df['Close']<df['HMA'])&(df['RSI']<50)&(df['Momentum']<0)]
        df['Signal'] = np.select(cond, [1, -1], 0)
        
        return df.dropna()

class ApexScannerEngine:
    """Market-Wide SMC Scanner Logic"""
    
    @staticmethod
    def calculate_wavetrend(df):
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=10).mean()
        d = (ap - esa).abs().ewm(span=10).mean()
        ci = (ap - esa) / (0.015 * d.replace(0, 0.0001))
        return ci.ewm(span=21).mean()

    @staticmethod
    def detect_smc(df):
        # BOS: Close > recent Pivot High
        recent_high = df['High'].shift(1).rolling(20).max()
        bos_bull = (df['Close'] > recent_high) & (df['Close'].shift(1) <= recent_high.shift(1))
        
        # FVG: Low(0) > High(2)
        fvg_bull = (df['Low'] > df['High'].shift(2))
        fvg_size = (df['Low'] - df['High'].shift(2))
        return bos_bull, fvg_bull, fvg_size

    @staticmethod
    def analyze_ticker(ticker):
        # Lightweight fetch for scanning
        df = DataEngine.fetch_price_data(ticker, "6mo", "1d")
        if df.empty or len(df) < 60: return None
        
        # Reuse logic
        df = IndicatorEngine.add_indicators(df) 
        df['WaveTrend'] = ApexScannerEngine.calculate_wavetrend(df)
        bos, fvg, fvg_sz = ApexScannerEngine.detect_smc(df)
        
        df['BOS_Bull'] = bos
        df['FVG_Bull'] = fvg
        df['FVG_Size'] = fvg_sz
        
        last = df.iloc[-1]
        score = 0
        tags = []
        
        # Scoring Matrix
        if last['Signal'] == 1: score += 1
        
        # Apex Buy: Trend + WaveTrend + Vol
        vol_ma = df['Volume'].rolling(20).mean().iloc[-1]
        if (last['Signal'] == 1) and (last['WaveTrend'] < 60) and \
           (last['WaveTrend'] > df['WaveTrend'].iloc[-2]) and (last['Volume'] > vol_ma):
            score += 3
            tags.append("APEX BUY")
            
        if df['BOS_Bull'].tail(3).any():
            score += 2
            tags.append("BOS")
            
        if last['FVG_Bull']:
            score += 1
            tags.append("FVG")
            
        if score > 0:
            return {
                "ticker": ticker,
                "Price": last['Close'],
                "Trend": "Bullish 🟢" if last['Signal']==1 else "Bearish 🔴",
                "WaveTrend": last['WaveTrend'],
                "ADX": last['ADX'],
                "Score": score,
                "Tags": ", ".join(tags),
                "FVG_Size": last['FVG_Size'] if last['FVG_Bull'] else 0,
                "Regime": last['Regime']
            }
        return None

    @staticmethod
    def run_screen(universe):
        results = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        for i, ticker in enumerate(universe):
            status.caption(f"📡 Scanning {ticker} for SMC Structures...")
            res = ApexScannerEngine.analyze_ticker(ticker)
            if res: results.append(res)
            progress_bar.progress((i+1)/len(universe))
            
        progress_bar.empty()
        status.empty()
        return pd.DataFrame(results)

class PaperTradingEngine:
    """Execution Simulator"""
    @staticmethod
    def initialize():
        if 'paper_trades' not in st.session_state: st.session_state.paper_trades = []
        if 'paper_balance' not in st.session_state: st.session_state.paper_balance = 100000.0

    @staticmethod
    def execute(action, ticker, price, qty):
        cost = price * qty
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        if action == "BUY" and st.session_state.paper_balance >= cost:
            st.session_state.paper_balance -= cost
            st.session_state.paper_trades.append({"Time": ts, "Ticker": ticker, "Action": "BUY", "Price": price, "Qty": qty})
            return True
        elif action == "SELL":
            st.session_state.paper_balance += cost
            st.session_state.paper_trades.append({"Time": ts, "Ticker": ticker, "Action": "SELL", "Price": price, "Qty": qty})
            return True
        return False

# ==========================================
# 3. VISUALIZATION LAYER (UI)
# ==========================================

def render_chart(df, fib_levels, ticker):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
    
    # Row 1: Price & HMA & Fibs
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='#00ceff', width=1.5), name='HMA Trend'), row=1, col=1)
    
    for n, l in fib_levels.items():
        color = "#ffd700" if "0.618" in n else "rgba(255,255,255,0.3)"
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=l, y1=l, line=dict(color=color, width=1, dash="dot"), row=1, col=1)
        if "0.618" in n: 
            fig.add_annotation(x=df.index[-1], y=l, text=n, font=dict(color="gold", size=10), row=1, col=1)

    # Row 2: Momentum
    colors = ['#00ff00' if v>0 else '#ff0000' for v in df['Momentum']]
    fig.add_trace(go.Bar(x=df.index, y=df['Momentum'], marker_color=colors, name='Momentum'), row=2, col=1)
    
    # Row 3: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#ffd700'), name='RSI'), row=3, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=70, y1=70, line=dict(color='red', dash='dot'), row=3, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=30, y1=30, line=dict(color='green', dash='dot'), row=3, col=1)
    
    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=5,r=5,t=5,b=5))
    return fig

# ==========================================
# 4. MAIN APPLICATION
# ==========================================

def main():
    PaperTradingEngine.initialize()
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("### ⚙️ Engine Controls")
        api_key = st.text_input("OpenAI API Key", type="password")
        
        st.markdown("---")
        st.markdown("### 📝 Paper Trader")
        st.metric("Equity", f"${st.session_state.paper_balance:,.2f}")
        
        c1, c2 = st.columns(2)
        qty = st.number_input("Qty", 0.01, 100.0, 1.0)
        
        if c1.button("🟢 BUY", use_container_width=True): st.session_state.signal = "BUY"
        if c2.button("🔴 SELL", use_container_width=True): st.session_state.signal = "SELL"
        
        if st.session_state.paper_trades:
            with st.expander("Ledger"):
                st.dataframe(pd.DataFrame(st.session_state.paper_trades), hide_index=True)

    st.markdown('<div class="terminal-header">VANTAGE QUANTITATIVE TERMINAL</div>', unsafe_allow_html=True)

    # --- TOP LEVEL NAVIGATION ---
    tab_term, tab_scan = st.tabs(["🏛️ Deep Dive Terminal", "🌊 Apex Market Scanner"])

    # === TAB 1: DEEP DIVE (Single Asset) ===
    with tab_term:
        c1, c2 = st.columns([1, 4])
        with c1:
            ticker = st.text_input("Asset Ticker", value="BTC-USD").upper()
            interval = st.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=3)
        
        # Logic Flow
        p_map = {"15m": "1mo", "1h": "1y", "4h": "1y", "1d": "2y", "1wk": "5y"}
        df = DataEngine.fetch_price_data(ticker, p_map.get(interval, "1y"), interval)
        
        if not df.empty:
            df = IndicatorEngine.add_indicators(df)
            fibs = IndicatorEngine.calc_fibonacci(df)
            funds = DataEngine.get_fundamentals(ticker)
            
            # Paper Trade Execution
            if "signal" in st.session_state:
                act = st.session_state.pop("signal")
                price = df['Close'].iloc[-1]
                if PaperTradingEngine.execute(act, ticker, price, qty):
                    st.toast(f"✅ {act} {qty} {ticker} @ {price}")
                else:
                    st.toast("❌ Insufficient Funds")

            # Metrics
            last = df.iloc[-1]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Price", f"${last['Close']:,.2f}", f"{last['Close']-df['Close'].iloc[-2]:.2f}")
            m2.metric("Regime", last['Regime'], f"ADX: {last['ADX']:.1f}", delta_color="normal" if last['Regime']=="Consolidating" else "off")
            m3.metric("Signal", "BULLISH" if last['Signal']==1 else "BEARISH" if last['Signal']==-1 else "NEUTRAL")
            m4.metric("Squeeze", "🔥 FIRE" if last['Squeeze_On'] else "💤 OFF")

            # Chart & AI
            t_ch, t_ai = st.tabs(["📈 Chart", "🧠 AI Analyst"])
            with t_ch:
                st.plotly_chart(render_chart(df, fibs, ticker), use_container_width=True)
            
            with t_ai:
                if st.button("🔮 Generate Insight"):
                    if not api_key: st.warning("API Key Required")
                    else:
                        # Logic-Aware Prompt
                        prompt = f"""
                        Act as a Quant Analyst. Analyze {ticker}.
                        Technical Logic:
                        - Signal: {'Bullish' if last['Signal']==1 else 'Bearish' if last['Signal']==-1 else 'Neutral'}
                        - Market Regime: {last['Regime']} (ADX: {last['ADX']:.1f})
                        - Key Fib Level: {fibs['0.618 (Gold)']:.2f}
                        - Squeeze Status: {last['Squeeze_On']}
                        
                        Provide a trading plan emphasizing Risk/Reward. Mention specific support levels.
                        """
                        
                        client = OpenAI(api_key=api_key)
                        st.markdown("### 🧠 Neural Analysis")
                        
                        # Streaming Response
                        placeholder = st.empty()
                        full_resp = ""
                        stream = client.chat.completions.create(
                            model="gpt-4o", messages=[{"role":"user","content":prompt}], stream=True
                        )
                        for chunk in stream:
                            if chunk.choices[0].delta.content:
                                full_resp += chunk.choices[0].delta.content
                                placeholder.markdown(full_resp + "▌")
                        placeholder.markdown(full_resp)
                        st.write("---")
                        st.markdown("

[Image of support and resistance levels]
") # Contextual Diagram Trigger

    # === TAB 2: APEX SCANNER (Multi Asset) ===
    with tab_scan:
        st.markdown("### 🌊 Apex SMC Scanner")
        st.caption("Hunting for Confluence: Trend + WaveTrend + BOS + FVG across Crypto Universe.")
        
        if st.button("🚀 Run System Scan"):
            scan_df = ApexScannerEngine.run_screen(CRYPTO_UNIVERSE)
            
            if not scan_df.empty:
                scan_df = scan_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
                st.session_state.scan_results = scan_df
                
                # Excel Gen
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                    scan_df.to_excel(writer, index=False)
                st.session_state.scan_excel = buf.getvalue()
            else:
                st.warning("No high-probability setups found.")

        if "scan_results" in st.session_state:
            res_df = st.session_state.scan_results
            
            for i, row in res_df.iterrows():
                with st.expander(f"{row['ticker']} | Score: {row['Score']} | Tags: {row['Tags']}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Price", f"${row['Price']:.4f}")
                    c2.metric("WaveTrend", f"{row['WaveTrend']:.1f}")
                    c3.metric("FVG Size", f"{row['FVG_Size']:.5f}")
                    
                    if st.button(f"🧠 Analyze {row['ticker']}", key=f"ai_{i}"):
                        if not api_key: st.error("API Key Required")
                        else:
                            # Logic-Aware Prompt for Scanner
                            prompt = f"""
                            Analyze {row['ticker']} based on this Scanner Output:
                            - Trend: {row['Trend']}
                            - Detected Tags: {row['Tags']}
                            - WaveTrend: {row['WaveTrend']}
                            - FVG Size: {row['FVG_Size']}
                            
                            Is this a valid entry? Explain the Break of Structure (BOS) context.
                            """
                            client = OpenAI(api_key=api_key)
                            resp = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
                            st.info(resp.choices[0].message.content)
                            st.markdown("") # Contextual Diagram Trigger
            
            if "scan_excel" in st.session_state:
                st.download_button("📥 Download Excel Report", st.session_state.scan_excel, "apex_scan.xlsx")

if __name__ == "__main__":
    main()
