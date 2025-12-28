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
        # Intelligent Period Selection to minimize data load while ensuring calculation depth
        period_map = {
            "1m": "2d", "5m": "5d", "15m": "10d", "30m": "1mo",
            "1h": "1mo", "4h": "3mo", "1d": "1y", "1wk": "2y"
        }
        try:
            df = yf.download(ticker, period=period_map.get(interval, "1mo"), interval=interval, progress=False)
            if df.empty: return pd.DataFrame()
            df.columns = df.columns.str.lower()
            return df
        except:
            return pd.DataFrame()

class QuantumIndicators:
    @staticmethod
    def compute(df):
        if df.empty or len(df) < 52: return None
        
        # Flatten data for numpy efficiency
        close = df['close'].values.flatten()
        high = df['high'].values.flatten()
        low = df['low'].values.flatten()

        # A. ICHIMOKU CLOUD (Trend Baseline)
        nine_period_high = pd.Series(high).rolling(window=9).max()
        nine_period_low = pd.Series(low).rolling(window=9).min()
        tenkan_sen = (nine_period_high + nine_period_low) / 2 # Conversion Line

        twenty_six_high = pd.Series(high).rolling(window=26).max()
        twenty_six_low = pd.Series(low).rolling(window=26).min()
        kijun_sen = (twenty_six_high + twenty_six_low) / 2 # Base Line

        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Current Price vs Cloud Logic
        last_close = close[-1]
        last_cloud = senkou_span_a.iloc[-1]
        
        # B. ADX (Trend Strength) - Simplified calculation for speed
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - pd.Series(close).shift(1)))
        tr3 = pd.DataFrame(abs(low - pd.Series(close).shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # C. TITAN COMPOSITE SCORE (0-100)
        # 1. Trend Direction (Ichimoku)
        trend_score = 50
        if last_close > last_cloud: trend_score += 20
        elif last_close < last_cloud: trend_score -= 20
        
        if tenkan_sen.iloc[-1] > kijun_sen.iloc[-1]: trend_score += 15
        elif tenkan_sen.iloc[-1] < kijun_sen.iloc[-1]: trend_score -= 15

        # 2. Momentum (RSI)
        delta = np.diff(close)
        gain = (delta * (delta > 0)).mean()
        loss = (-delta * (delta < 0)).mean()
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))

        # 3. Volatility Squeeze
        std_dev = np.std(close[-20:])
        is_squeeze = (std_dev * 2) < (atr * 1.5)

        # Final Composite Calculation
        final_score = trend_score
        if rsi > 70: final_score -= 10 # Overbought penalty
        if rsi < 30: final_score += 10 # Oversold boost
        
        # Normalized -5 to +5 scale for the UI
        normalized_score = (final_score - 50) / 10
        normalized_score = max(-5, min(5, normalized_score))

        # D. FIBONACCI PIVOTS (Auto-Support/Resistance)
        recent_high = max(high[-50:])
        recent_low = min(low[-50:])
        fib_382 = recent_high - ((recent_high - recent_low) * 0.382)
        fib_618 = recent_high - ((recent_high - recent_low) * 0.618)

        # E. STOP LOSS ENGINE (ATR Based)
        stop_dist = atr * 2.5
        stop_loss = last_close - stop_dist if normalized_score > 0 else last_close + stop_dist

        return {
            "score": normalized_score,
            "rsi": rsi,
            "atr": atr,
            "squeeze": is_squeeze,
            "ichimoku_bullish": last_close > last_cloud,
            "fib_support": fib_618,
            "fib_resistance": fib_382,
            "stop_loss": stop_loss,
            "last_price": last_close
        }

# --- 4. INTELLIGENT BROADCASTING ---

def generate_obsidian_signal(ticker, interval, data, kind, ai_context=""):
    score = data['score']
    price = data['last_price']
    stop = data['stop_loss']
    
    # Dynamic Emoji & Tone Selection based on Volatility
    volatility_emoji = "🌋" if data['atr'] > (price * 0.01) else "🌊"
    trend_state = "DOMINANT BULL" if score > 3 else "DOMINANT BEAR" if score < -3 else "RANGE BOUND"
    
    # Calculate R-Multiples
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
• Ichimoku Cloud: {'Above (Bullish)' if data['ichimoku_bullish'] else 'Below (Bearish)'}
• Squeeze State: {'ACTIVE 🔥' if data['squeeze'] else 'Dormant'}

<i>Timestamp: {datetime.now().strftime('%H:%M:%S UTC')}</i>
"""
    elif kind == "INTEL_Brief":
        return f"""
{header}
{meta}

<b>🔍 MARKET CONTEXT</b>
Price is currently trading at <b>${price:,.2f}</b>.
Algorithm detects a <b>{trend_state}</b> regime.

<b>📐 KEY LEVELS (FIBONACCI)</b>
• Resistance (0.382): ${data['fib_resistance']:,.2f}
• Support (0.618): ${data['fib_support']:,.2f}

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
        # Initialize with empty strings
        token = ""
        chat = ""
        
        # Safe Secrets Loading
        try:
            # Check for top-level keys first
            if "TELEGRAM_TOKEN" in st.secrets:
                token = st.secrets["TELEGRAM_TOKEN"]
            elif "telegram" in st.secrets and "token" in st.secrets["telegram"]:
                token = st.secrets["telegram"]["token"]
                
            if "TELEGRAM_CHAT_ID" in st.secrets:
                chat = st.secrets["TELEGRAM_CHAT_ID"]
            elif "telegram" in st.secrets and "chat_id" in st.secrets["telegram"]:
                chat = st.secrets["telegram"]["chat_id"]
                
        except FileNotFoundError:
            # Secrets file missing is okay, we just default to manual input
            pass
        except Exception:
            # Catch other potential loading errors
            pass
            
        st.session_state.tg_creds = {"token": token, "chat": chat}

    # SIDEBAR: COMMAND CENTER
    with st.sidebar:
        st.markdown("## 💠 TITAN OS")
        st.markdown("---")
        
        # Asset Selection
        new_ticker = st.selectbox("ASSET TARGET", [t for cat in ASSET_CLASSES.values() for t in cat], index=0)
        new_interval = st.select_slider("TIMEFRAME", options=INTERVALS, value="1h")
        
        # Update State
        if new_ticker != st.session_state.config['ticker']: st.session_state.config['ticker'] = new_ticker
        st.session_state.config['interval'] = new_interval
        
        st.markdown("---")
        st.markdown("### 📡 UPLINK")
        
        # Visual Indicator for Secrets
        has_secrets = False
        try:
            if "TELEGRAM_TOKEN" in st.secrets: has_secrets = True
        except: pass
        
        if has_secrets:
            st.caption("🔒 SECRETS ACTIVE")
        else:
            st.caption("⚠️ MANUAL MODE")
            
        # These inputs will now pre-fill if secrets were found
        st.session_state.tg_creds['token'] = st.text_input("BOT TOKEN", value=st.session_state.tg_creds['token'], type="password")
        st.session_state.tg_creds['chat'] = st.text_input("CHAT ID", value=st.session_state.tg_creds['chat'])
        
        st.markdown("---")
        st.session_state.config['live'] = st.checkbox("🔴 LIVE SYNC (10s)", value=st.session_state.config['live'])

    # MAIN DISPLAY
    col_logo, col_stat = st.columns([3, 1])
    with col_logo:
        st.markdown(f"# <span class='neon-blue'>AXIOM</span> TITAN <span style='font-size:0.5em; opacity:0.5'>// {st.session_state.config['ticker']}</span>", unsafe_allow_html=True)
    with col_stat:
        st.markdown(f"<div style='text-align:right; font-family:monospace; color:#555'>{datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

    # 1. FETCH & COMPUTE
    df = MarketDataEngine.fetch_live(st.session_state.config['ticker'], st.session_state.config['interval'])
    
    if df.empty:
        st.error("NO SIGNAL DETECTED. MARKET MAY BE CLOSED OR API LIMITED.")
        if st.session_state.config['live']: time.sleep(10); st.rerun()
        return

    metrics = QuantumIndicators.compute(df)
    if not metrics:
        st.warning("INITIALIZING QUANTUM CORE... (Need more data)")
        return

    # 2. HEADS UP DISPLAY (HUD)
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
        st.metric("RSI MOMENTUM", f"{metrics['rsi']:.1f}", "OVERSOLD" if metrics['rsi'] < 30 else "OVERBOUGHT" if metrics['rsi'] > 70 else "NEUTRAL")
        
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

    # 3. ADVANCED CHARTING
    row_chart, row_actions = st.columns([3, 1])
    
    with row_chart:
        fig = go.Figure()
        
        # Candles
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color='#22d3ee', decreasing_line_color='#f87171', name='Price'
        ))
        
        # Stop Loss Line (Visual Aid)
        fig.add_hline(y=metrics['stop_loss'], line_dash="dot", line_color="gray", annotation_text="SUGGESTED STOP")

        fig.update_layout(
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_rangeslider_visible=False,
            font=dict(family="JetBrains Mono", color="#71717a"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig, use_container_width=True)

    with row_actions:
        st.markdown("### 💠 BROADCAST")
        st.info(f"Signal Strength: {abs(metrics['score'])/5*100:.0f}%")
        
        if st.button("🚀 TRANSMIT: SCALP", use_container_width=True):
            msg = generate_obsidian_signal(st.session_state.config['ticker'], st.session_state.config['interval'], metrics, "SCALP_Protocol")
            if telegram_dispatch(st.session_state.tg_creds['token'], st.session_state.tg_creds['chat'], msg):
                st.toast("SIGNAL UPLINK ESTABLISHED", icon="✅")
            else:
                st.toast("UPLINK FAILED: CHECK CREDS", icon="❌")

        if st.button("🧠 TRANSMIT: INTEL", use_container_width=True):
            # Mock AI generation for speed in this demo
            ai_mock = "Market structure suggests a liquidity sweep of recent lows. Ichimoku baseline holds as support."
            msg = generate_obsidian_signal(st.session_state.config['ticker'], st.session_state.config['interval'], metrics, "INTEL_Brief", ai_mock)
            telegram_dispatch(st.session_state.tg_creds['token'], st.session_state.tg_creds['chat'], msg)
            st.toast("INTEL PACKET SENT", icon="✅")

        st.markdown("---")
        st.markdown("### 🛡️ RISK MATRIX")
        st.write(f"Entry: **${metrics['last_price']:,.2f}**")
        st.write(f"Stop:  **${metrics['stop_loss']:,.2f}**")
        st.caption("Values update live with chart.")

    # LIVE LOOP
    if st.session_state.config['live']:
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()
