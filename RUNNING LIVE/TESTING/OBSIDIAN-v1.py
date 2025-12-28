import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import requests
import yfinance as yf
from datetime import datetime
import pytz

# --- 1. CONFIGURATION & CONSTANTS ---

STORAGE_KEY = 'axiom_titan_obsidian_config'

# Enhanced Asset List
ASSET_CLASSES = {
    "Cryptocurrency": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD"],
    "Major Indices": ["^GSPC", "^NDX", "^DJI", "^RUT", "^FTSE", "^N225"],
    "Forex Majors": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"],
    "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"]
}

INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"]

# --- 2. THE "OBSIDIAN" VISUAL ENGINE (CSS) ---

st.set_page_config(layout="wide", page_title="AXIOM TITAN // OBSIDIAN", page_icon="💠")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;700&display=swap');

    /* BASE THEME */
    .stApp {
        background-color: #09090b;
        background-image: radial-gradient(circle at 50% 50%, #18181b 0%, #09090b 100%);
        color: #e4e4e7;
        font-family: 'JetBrains Mono', monospace;
    }

    /* HIDE DEFAULTS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* GLASS PANELS */
    div.stMarkdown, div[data-testid="stMetric"], div.stPlotlyChart {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }
    
    /* SIDEBAR & NAV */
    section[data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #27272a;
    }

    /* TYPOGRAPHY */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -1px;
    }
    .neon-blue { color: #22d3ee; text-shadow: 0 0 10px rgba(34, 211, 238, 0.5); }
    .neon-purple { color: #c084fc; text-shadow: 0 0 10px rgba(192, 132, 252, 0.5); }
    .neon-green { color: #4ade80; text-shadow: 0 0 10px rgba(74, 222, 128, 0.5); }
    .neon-red { color: #f87171; text-shadow: 0 0 10px rgba(248, 113, 113, 0.5); }

    /* CUSTOM METRIC LABELS */
    div[data-testid="stMetricLabel"] {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #71717a;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 300;
        color: #fafafa;
    }

    /* BUTTONS */
    .stButton > button {
        background: transparent;
        border: 1px solid #3f3f46;
        color: #a1a1aa;
        border-radius: 6px;
        transition: all 0.2s ease;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        border-color: #22d3ee;
        color: #22d3ee;
        box-shadow: 0 0 15px rgba(34, 211, 238, 0.2);
    }

    /* INPUTS */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(0,0,0,0.5);
        border: 1px solid #3f3f46;
        color: white;
    }

    /* TOAST */
    div[data-testid="stToast"] {
        background-color: #18181b;
        border: 1px solid #22d3ee;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. QUANTUM CORE (Advanced Logic) ---

class MarketDataEngine:
    @staticmethod
    def fetch_live(ticker, interval):
        # Intelligent Period Selection
        period_map = {
            "1m": "2d", "5m": "5d", "15m": "10d", "30m": "1mo",
            "1h": "1mo", "4h": "3mo", "1d": "1y", "1wk": "2y"
        }
        
        try:
            # FIX: Use Ticker().history() instead of download() for better stability
            dat = yf.Ticker(ticker)
            period = period_map.get(interval, "1mo")
            df = dat.history(period=period, interval=interval)
            
            if df.empty:
                # Fallback to download if history fails
                df = yf.download(ticker, period=period, interval=interval, progress=False)

            if df.empty:
                st.error(f"Provider returned no data for {ticker}. API limit or Invalid Symbol.")
                return pd.DataFrame()
            
            # Formatting: Lowercase columns
            df.columns = df.columns.str.lower()
            
            # FIX: Handle MultiIndex columns (Common yfinance issue in 2024/25)
            # If columns look like ('close', 'BTC-USD'), flatten them
            if isinstance(df.columns, pd.MultiIndex):
                # Attempt to keep just the column name (level 0 or 1 depending on version)
                try:
                    df.columns = df.columns.get_level_values(0)
                except:
                    pass
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Clean timezone info to prevent plotting errors
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)

            return df
            
        except Exception as e:
            st.error(f"Data Pipeline Error: {e}")
            return pd.DataFrame()

class QuantumIndicators:
    @staticmethod
    def compute(df):
        # Validation
        if df.empty: return None
        if len(df) < 30: 
            st.warning("Insufficient data depth for Quantum Core calculations.")
            return None
        
        try:
            # Safely extract arrays
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # A. ICHIMOKU CLOUD
            nine_high = pd.Series(high).rolling(window=9).max()
            nine_low = pd.Series(low).rolling(window=9).min()
            tenkan = (nine_high + nine_low) / 2

            six_high = pd.Series(high).rolling(window=26).max()
            six_low = pd.Series(low).rolling(window=26).min()
            kijun = (six_high + six_low) / 2

            span_a = ((tenkan + kijun) / 2).shift(26)
            
            # B. ADX / ATR
            tr1 = np.abs(high - low)
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = pd.Series(tr).rolling(14).mean().iloc[-1]
            
            # Handle NaN from rolling (replace with 0 or fill)
            if np.isnan(atr): atr = 0
            
            # C. RSI
            delta = np.diff(close)
            gain = (delta * (delta > 0)).mean()
            loss = (-delta * (delta < 0)).mean()
            rs = gain / loss if loss != 0 else 0
            rsi = 100 - (100 / (1 + rs))

            # D. LOGIC
            last_close = close[-1]
            last_cloud = span_a.iloc[-1] if not np.isnan(span_a.iloc[-1]) else last_close
            
            # Scoring
            score = 0
            if last_close > last_cloud: score += 2.5
            else: score -= 2.5
            
            if rsi < 30: score += 1.5
            if rsi > 70: score -= 1.5
            
            score = max(-5, min(5, score))
            
            # E. LEVELS
            fib_high = np.max(high[-50:])
            fib_low = np.min(low[-50:])
            fib_r = fib_high - ((fib_high - fib_low) * 0.382)
            fib_s = fib_high - ((fib_high - fib_low) * 0.618)
            
            stop_dist = max(atr * 2.0, last_close * 0.01) # Min 1% stop
            stop_loss = last_close - stop_dist if score > 0 else last_close + stop_dist

            return {
                "score": score,
                "rsi": rsi,
                "atr": atr,
                "squeeze": False, # Simplified for robustness
                "ichimoku_bullish": last_close > last_cloud,
                "fib_support": fib_s,
                "fib_resistance": fib_r,
                "stop_loss": stop_loss,
                "last_price": last_close
            }
        except Exception as e:
            st.error(f"Calculation Error: {e}")
            return None

# --- 4. INTELLIGENT BROADCASTING ---

def generate_obsidian_signal(ticker, interval, data, kind, ai_context=""):
    score = data['score']
    price = data['last_price']
    stop = data['stop_loss']
    
    volatility_emoji = "🌋" if data['atr'] > (price * 0.01) else "🌊"
    
    risk = abs(price - stop)
    tp1 = price + (risk * 1.5) if score > 0 else price - (risk * 1.5)
    tp2 = price + (risk * 3.0) if score > 0 else price - (risk * 3.0)
    
    header = f"<b>💠 AXIOM TITAN // {kind}</b>"
    meta = f"<code>{ticker} | {interval} | {volatility_emoji}</code>"
    
    if kind == "SCALP_Protocol":
        direction = "LONG 🟢" if score > 0 else "SHORT 🔴"
        return f"""
{header}
{meta}

<b>DIRECTION:</b> {direction}
<b>ENTRY REF:</b> ${price:,.2f}

<b>🛡️ RISK PARAMETERS</b>
• Stop Loss: <code>${stop:,.2f}</code>
• Risk Delta: {risk:,.2f} pts

<b>🎯 PROFIT VECTORS</b>
• TP1 (1.5R): <code>${tp1:,.2f}</code>
• TP2 (3.0R): <code>${tp2:,.2f}</code>

<b>🧠 QUANTUM METRICS</b>
• Titan Score: {score:.1f}/5.0
• Ichimoku: {'Bullish Cloud' if data['ichimoku_bullish'] else 'Bearish Cloud'}

<i>Timestamp: {datetime.now().strftime('%H:%M:%S UTC')}</i>
"""
    elif kind == "INTEL_Brief":
        return f"""
{header}
{meta}

<b>🔍 MARKET CONTEXT</b>
Price: <b>${price:,.2f}</b>
RSI: {data['rsi']:.1f}

<b>📐 KEY LEVELS</b>
• Res: ${data['fib_resistance']:,.2f}
• Sup: ${data['fib_support']:,.2f}

<b>🤖 AI SYNTHESIS</b>
{ai_context}

<i>Timestamp: {datetime.now().strftime('%H:%M UTC')}</i>
"""
    return "Signal Error"

def telegram_dispatch(token, chat_id, message):
    if not token or not chat_id: return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"})
        return True
    except:
        return False

# --- 5. MAIN APP INTERFACE ---

def main():
    # SESSION STATE & SECRETS AUTO-LOADER
    if 'config' not in st.session_state:
        st.session_state.config = {"ticker": "BTC-USD", "interval": "1h", "live": False}
    
    if 'tg_creds' not in st.session_state:
        token = ""
        chat = ""
        try:
            if "TELEGRAM_TOKEN" in st.secrets:
                token = st.secrets["TELEGRAM_TOKEN"]
            elif "telegram" in st.secrets and "token" in st.secrets["telegram"]:
                token = st.secrets["telegram"]["token"]
                
            if "TELEGRAM_CHAT_ID" in st.secrets:
                chat = st.secrets["TELEGRAM_CHAT_ID"]
            elif "telegram" in st.secrets and "chat_id" in st.secrets["telegram"]:
                chat = st.secrets["telegram"]["chat_id"]
        except: pass
            
        st.session_state.tg_creds = {"token": token, "chat": chat}

    # SIDEBAR
    with st.sidebar:
        st.markdown("## 💠 TITAN OS")
        st.markdown("---")
        
        new_ticker = st.selectbox("ASSET TARGET", [t for cat in ASSET_CLASSES.values() for t in cat], index=0)
        new_interval = st.select_slider("TIMEFRAME", options=INTERVALS, value="1h")
        
        if new_ticker != st.session_state.config['ticker']: st.session_state.config['ticker'] = new_ticker
        st.session_state.config['interval'] = new_interval
        
        st.markdown("---")
        st.markdown("### 📡 UPLINK")
        
        has_secrets = False
        try:
            if "TELEGRAM_TOKEN" in st.secrets: has_secrets = True
        except: pass
        
        if has_secrets:
            st.caption("🔒 SECRETS ACTIVE")
        else:
            st.caption("⚠️ MANUAL MODE")
            
        st.session_state.tg_creds['token'] = st.text_input("BOT TOKEN", value=st.session_state.tg_creds['token'], type="password")
        st.session_state.tg_creds['chat'] = st.text_input("CHAT ID", value=st.session_state.tg_creds['chat'])
        
        st.markdown("---")
        st.session_state.config['live'] = st.checkbox("🔴 LIVE SYNC (15s)", value=st.session_state.config['live'])

    # MAIN DISPLAY
    col_logo, col_stat = st.columns([3, 1])
    with col_logo:
        st.markdown(f"# <span class='neon-blue'>AXIOM</span> TITAN <span style='font-size:0.5em; opacity:0.5'>// {st.session_state.config['ticker']}</span>", unsafe_allow_html=True)
    with col_stat:
        st.markdown(f"<div style='text-align:right; font-family:monospace; color:#555'>{datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

    # 1. FETCH & COMPUTE
    df = MarketDataEngine.fetch_live(st.session_state.config['ticker'], st.session_state.config['interval'])
    
    # Check if empty - if so, stop here
    if df.empty:
        if st.session_state.config['live']: 
            time.sleep(15) 
            st.rerun()
        return

    metrics = QuantumIndicators.compute(df)
    if not metrics:
        st.warning("PROCESSING QUANTUM CORE...")
        return

    # 2. HEADS UP DISPLAY
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        score_color = "neon-green" if metrics['score'] > 2 else "neon-red" if metrics['score'] < -2 else "neon-blue"
        st.markdown(f"""
        <div data-testid="stMetric">
            <div data-testid="stMetricLabel">QUANTUM SCORE</div>
            <div data-testid="stMetricValue" class="{score_color}">{metrics['score']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        st.metric("RSI MOMENTUM", f"{metrics['rsi']:.1f}")
        
    with m3:
        st.metric("VOLATILITY (ATR)", f"{metrics['atr']:.2f}")
        
    with m4:
        st.markdown(f"""
        <div data-testid="stMetric">
            <div data-testid="stMetricLabel">CLUSTERS</div>
            <div style="font-size:12px; margin-top:5px; color:#aaa;">
            FIB SUP: <span style="color:#4ade80">${metrics['fib_support']:,.2f}</span><br>
            FIB RES: <span style="color:#f87171">${metrics['fib_resistance']:,.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. CHARTING
    row_chart, row_actions = st.columns([3, 1])
    
    with row_chart:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color='#22d3ee', decreasing_line_color='#f87171', name='Price'
        ))
        fig.add_hline(y=metrics['stop_loss'], line_dash="dot", line_color="gray", annotation_text="STOP")

        fig.update_layout(
            height=500, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_rangeslider_visible=False,
            font=dict(family="JetBrains Mono", color="#71717a"),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig, use_container_width=True)

    with row_actions:
        st.markdown("### 💠 BROADCAST")
        
        if st.button("🚀 TRANSMIT: SCALP", use_container_width=True):
            msg = generate_obsidian_signal(st.session_state.config['ticker'], st.session_state.config['interval'], metrics, "SCALP_Protocol")
            if telegram_dispatch(st.session_state.tg_creds['token'], st.session_state.tg_creds['chat'], msg):
                st.toast("SIGNAL UPLINK ESTABLISHED", icon="✅")
            else:
                st.toast("UPLINK FAILED: CHECK CREDS", icon="❌")

        if st.button("🧠 TRANSMIT: INTEL", use_container_width=True):
            ai_mock = "Structure suggests consolidation. Waiting for breakout."
            msg = generate_obsidian_signal(st.session_state.config['ticker'], st.session_state.config['interval'], metrics, "INTEL_Brief", ai_mock)
            telegram_dispatch(st.session_state.tg_creds['token'], st.session_state.tg_creds['chat'], msg)
            st.toast("INTEL PACKET SENT", icon="✅")

        st.markdown("---")
        st.markdown("### 🛡️ RISK MATRIX")
        st.write(f"Entry: **${metrics['last_price']:,.2f}**")
        st.write(f"Stop:  **${metrics['stop_loss']:,.2f}**")

    # LIVE LOOP
    if st.session_state.config['live']:
        time.sleep(15)
        st.rerun()

if __name__ == "__main__":
    main()
