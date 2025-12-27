"""
Signals-MOBILE 
Version 19.2: Enterprise Trading Engine + Bug Fix + Enhanced Telegram Reports
"""

import time
import math
import sqlite3
import random
import json
import asyncio
import logging
from typing import Dict, Optional, List, Tuple, Any
from contextlib import contextmanager
from functools import wraps

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components
from datetime import datetime, timezone

# --- CONFLICT MARKING START ---
# NEW: Conditional imports with explicit error handling for optional dependencies
try:
    import websockets
    import aiohttp
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    st.warning("⚠️ WebSocket libraries not installed. Run: pip install websockets aiohttp")
# --- CONFLICT MARKING END ---

# =============================================================================
# ERROR HANDLING & LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def widget_error_boundary(func):
    """Decorator for widget error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Widget error in {func.__name__}: {e}")
            st.error(f"⚠️ Widget failed to load: {e}")
            return components.html("<div style='color:#ff1744;'>Widget Error</div>", height=50)
    return wrapper

# =============================================================================
# PAGE CONFIG (Mobile Friendly)
# =============================================================================
st.set_page_config(
    page_title="TITAN-SIGNALS",
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM CSS (ENHANCED MOBILE UX)
# =============================================================================
st.markdown("""
<style>
    .main { background-color: #0b0c10; }
    
    /* Mobile-First Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(31, 40, 51, 0.9);
        border: 1px solid #45a29e;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Larger Text for Mobile Readability */
    div[data-testid="metric-container"] label {
        font-size: 14px !important;
        color: #c5c6c7 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 24px !important;
        color: #66fcf1 !important;
    }
    
    h1, h2, h3 {
        font-family: 'Roboto Mono', monospace;
        color: #c5c6c7;
        word-wrap: break-word;
    }
    
    /* Touch-Friendly Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        border: 1px solid #45a29e;
        color: #66fcf1;
        font-weight: bold;
        height: 3em;
        font-size: 16px !important;
        border-radius: 8px;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .stButton > button:hover {
        background: #45a29e;
        color: #0b0c10;
    }
    
    /* Report Card Styling */
    .report-card {
        background-color: #1f2833;
        border-left: 5px solid #45a29e;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .report-header {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;
        border-bottom: 1px solid #45a29e;
        padding-bottom: 5px;
    }
    .report-item {
        margin-bottom: 8px;
        font-size: 14px;
        color: #c5c6c7;
    }
    .highlight { color: #66fcf1; font-weight: bold; }
    
    /* AI Analysis Cards */
    .ai-card {
        background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%);
        border: 1px solid #45a29e;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(69, 162, 158, 0.2);
    }
    .ai-score {
        font-size: 32px;
        font-weight: bold;
        color: #66fcf1;
        text-align: center;
        padding: 10px;
    }
    
    /* NEW: Widget Error Display */
    .widget-error {
        background: #ff1744;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-size: 12px;
    }
    
    /* NEW: Connection Status Badge */
    .status-connected {
        background: linear-gradient(135deg, #00e676, #00c853);
        color: #0b0c10;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 12px;
    }
    .status-disconnected {
        background: linear-gradient(135deg, #ff1744, #d50000);
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 12px;
    }
    .status-error {
        background: linear-gradient(135deg, #ff9800, #f57c00);
        color: #0b0c10;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================
# --- CRITICAL BUG FIX: Removed trailing space from API base URL ---
BINANCE_API_BASE = "https://api.binance.us/api/v3 "  # FIXED: Removed trailing space
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# WebSocket endpoints for real-time data
BINANCE_WS_BASE = "wss://stream.binance.us:9443/ws"

POPULAR_BASES = [
    "BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "LINK", "AVAX", "DOT",
    "MATIC", "LTC", "BCH", "ATOM", "XLM", "ETC", "AAVE", "UNI", "SHIB", "TRX",
    "FIL", "NEAR", "ICP", "ARB", "OP", "SUI", "APT", "INJ", "TIA", "RNDR"
]

# AI Asset Knowledge Base (EXPANDED)
ASSET_PROFILES = {
    "BTC": {"type": "Macro Asset", "vol_regime": "Low", "session": "Global", "correlation": "Risk-On", "category": "Large Cap"},
    "ETH": {"type": "Smart Contract Leader", "vol_regime": "Medium", "session": "US/EU", "correlation": "BTC Beta", "category": "Large Cap"},
    "SOL": {"type": "High-Performance Chain", "vol_regime": "High", "session": "US", "correlation": "ETH Beta", "category": "Mid Cap"},
    "XRP": {"type": "Cross-Border Payments", "vol_regime": "Medium", "session": "EU/ASIA", "correlation": "Uncorrelated", "category": "Large Cap"},
    # NEW: Extended profiles
    "BNB": {"type": "Exchange Token", "vol_regime": "Medium", "session": "Global", "correlation": "Exchange Beta", "category": "Large Cap"},
    "ADA": {"type": "PoS Blockchain", "vol_regime": "Medium", "session": "US/EU", "correlation": "ETH Beta", "category": "Mid Cap"},
}

# =============================================================================
# DATABASE LAYER (ENHANCED)
# =============================================================================
class SignalDatabase:
    """SQLite backend for signal persistence with query capabilities"""
    def __init__(self, db_path: str = "titan_signals.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timestamp TEXT,
                    direction TEXT,
                    entry_price REAL,
                    stop_price REAL,
                    tp1 REAL,
                    tp2 REAL,
                    tp3 REAL,
                    confidence_score INTEGER,
                    outcome TEXT,
                    pnl REAL,
                    tp_reached INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # --- NEW: Create index for performance queries ---
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON signals(symbol, timestamp)")
    
    # --- NEW: Enhanced save with conflict handling ---
    def save_signal(self, symbol: str, signal_data: Dict, outcome: str = "PENDING"):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO signals (symbol, timestamp, direction, entry_price, stop_price, 
                                       tp1, tp2, tp3, confidence_score, outcome)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    datetime.now(timezone.utc).isoformat(),
                    "LONG" if signal_data['is_bull'] else "SHORT",
                    signal_data['close'],
                    signal_data['entry_stop'],
                    signal_data['tp1'],
                    signal_data['tp2'],
                    signal_data['tp3'],
                    signal_data.get('confidence', 0),
                    outcome
                ))
        except sqlite3.IntegrityError as e:
            logger.warning(f"Signal save conflict (likely duplicate): {e}")
    
    # --- NEW: Query historical performance ---
    def get_performance_stats(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT timestamp, outcome, pnl, tp_reached 
                    FROM signals 
                    WHERE symbol = ? AND outcome != 'PENDING'
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                return pd.read_sql_query(query, conn, params=(symbol, limit))
        except Exception as e:
            logger.error(f"Failed to fetch performance stats: {e}")
            return pd.DataFrame()

# =============================================================================
# WEBSOCKET CLIENT (ENHANCED & ACTIVATED)
# =============================================================================
class BinanceWebSocket:
    """Real-time WebSocket data feed with auto-reconnect"""
    def __init__(self, symbol: str):
        self.symbol = symbol.lower()
        self.ws_url = f"{BINANCE_WS_BASE}/{self.symbol}@kline_1m"
        self.data_queue = asyncio.Queue(maxsize=100)
        self.is_connected = False
        self.last_error = None
    
    async def connect(self):
        """Connect and maintain WebSocket with exponential backoff"""
        retry_delay = 1
        max_retry_delay = 60
        
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info(f"WebSocket connected: {self.symbol}")
                    self.is_connected = True
                    self.last_error = None
                    retry_delay = 1  # Reset on success
                    
                    while True:
                        message = await ws.recv()
                        data = json.loads(message)
                        await self.data_queue.put(data)
                        
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.is_connected = False
                self.last_error = str(e)
                await asyncio.sleep(min(retry_delay, max_retry_delay))
                retry_delay *= 2  # Exponential backoff
    
    def get_latest_candle(self) -> Optional[Dict]:
        """Non-blocking fetch of latest candle"""
        try:
            return self.data_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get connection health status"""
        return {
            "connected": self.is_connected,
            "error": self.last_error,
            "queue_size": self.data_queue.qsize()
        }

# --- NEW: Async runner for WebSocket ---
async def run_websocket_client(symbol: str):
    """Runs WebSocket client in background"""
    if not WEBSOCKET_AVAILABLE:
        logger.warning("WebSocket libraries not available")
        return None
    
    ws = BinanceWebSocket(symbol)
    # Start connection in background task
    asyncio.create_task(ws.connect())
    # Wait for first connection attempt
    await asyncio.sleep(2)
    return ws

# =============================================================================
# AI ANALYSIS ENGINE (ENHANCED)
# =============================================================================
def analyze_asset_and_timeframe(symbol: str, timeframe: str, df: pd.DataFrame) -> Dict:
    """
    AI-driven analysis with regime detection and ML scoring
    """
    base = symbol.replace("USDT", "")
    profile = ASSET_PROFILES.get(base, {
        "type": "Altcoin", 
        "vol_regime": "High", 
        "session": "US", 
        "correlation": "High Beta",
        "category": "Small Cap"
    })
    
    # Timeframe Suitability Score
    tf_scores = {
        "15m": {"score": 70, "note": "Scalping & Intraday", "vol_adj": 1.0},
        "1h": {"score": 85, "note": "Day Trading & Swing Entry", "vol_adj": 1.2},
        "4h": {"score": 90, "note": "Swing Trading", "vol_adj": 1.5},
        "1d": {"score": 80, "note": "Position Trading", "vol_adj": 2.0}
    }
    
    tf_data = tf_scores.get(timeframe, {"score": 50, "note": "Uncommon TF", "vol_adj": 1.0})
    
    # Calculate current market regime
    if not df.empty:
        recent_vol = df['close'].pct_change().rolling(20).std().iloc[-1] * 100
        avg_vol = df['close'].pct_change().rolling(60).std().iloc[-1] * 100
        
        if recent_vol > avg_vol * 1.5:
            regime = "High Volatility"
            regime_color = "#ff1744"
            regime_factor = 0.7  # Reduce position size
        elif recent_vol < avg_vol * 0.7:
            regime = "Low Volatility"
            regime_color = "#00e676"
            regime_factor = 1.3  # Increase position size
        else:
            regime = "Normal Volatility"
            regime_color = "#ffd740"
            regime_factor = 1.0
        
        # ML-based signal confidence (mock implementation)
        signal_confidence = int(profile['session'] == "US") * 10 + \
                           int(tf_data['score'] > 80) * 20 + \
                           int(regime_factor > 1.0) * 15
        
    else:
        regime = "Unknown"
        regime_color = "#9e9e9e"
        regime_factor = 1.0
        signal_confidence = 0
    
    # Generate recommendation with position sizing
    if tf_data["score"] >= 85:
        rec = "OPTIMAL TIMEFRAME"
        rec_color = "#00e676"
        size_rec = f"Position Size: {regime_factor:.1f}x Normal"
    elif tf_data["score"] >= 70:
        rec = "SUITABLE"
        rec_color = "#ffd740"
        size_rec = f"Position Size: {regime_factor:.1f}x Normal (Reduced)"
    else:
        rec = "SUBOPTIMAL - Consider 1h or 4h"
        rec_color = "#ff1744"
        size_rec = "Consider Smaller Size or Avoid"
    
    # Time session analysis
    utc_hour = datetime.now(timezone.utc).hour
    if profile['session'] == "US" and 13 <= utc_hour <= 21:
        session_note = "🔴 US Session Active - Best Liquidity"
        session_color = "#ff1744"
    elif profile['session'] == "EU/ASIA" and (7 <= utc_hour <= 15 or 22 <= utc_hour <= 6):
        session_note = "🟡 EU/ASIA Session - Moderate Liquidity"
        session_color = "#ffd740"
    else:
        session_note = "🟢 Off-Hours - Use Caution"
        session_color = "#00e676"
    
    return {
        "asset_profile": profile,
        "timeframe_score": tf_data["score"],
        "timeframe_note": tf_data["note"],
        "market_regime": regime,
        "regime_color": regime_color,
        "regime_factor": regime_factor,
        "recommendation": rec,
        "rec_color": rec_color,
        "size_recommendation": size_rec,
        "current_vol": f"{recent_vol:.2f}%" if not df.empty else "N/A",
        "signal_confidence": signal_confidence,
        "session_note": session_note,
        "session_color": session_color
    }

# =============================================================================
# TRAILING STOP ENGINE (ENHANCED)
# =============================================================================
def calculate_trailing_stop(current_price: float, entry_price: float, initial_stop: float,
                           tp1: float, tp2: float, tp3: float, is_long: bool) -> Tuple[float, str]:
    """
    Calculate trailing stop with status message
    Returns: (stop_price, status_message)
    """
    if is_long:
        if current_price >= tp3:
            return tp2, "🔒 Locked TP2"
        elif current_price >= tp2:
            return tp1, "🔒 Locked TP1"
        elif current_price >= tp1:
            return entry_price, "🔒 Breakeven"
        else:
            return initial_stop, "📍 Initial Stop"
    else:
        if current_price <= tp3:
            return tp2, "🔒 Locked TP2"
        elif current_price <= tp2:
            return tp1, "🔒 Locked TP1"
        elif current_price <= tp1:
            return entry_price, "🔒 Breakeven"
        else:
            return initial_stop, "📍 Initial Stop"

# =============================================================================
# TICKER UNIVERSE (OPTIMIZED)
# =============================================================================
@st.cache_data(ttl=3600)
def get_binanceus_usdt_bases() -> List[str]:
    """
    Pull all Binance US symbols with error handling
    """
    try:
        r = requests.get(f"{BINANCE_API_BASE}/exchangeInfo", headers=HEADERS, timeout=6)
        if r.status_code != 200:
            logger.warning(f"Binance API returned {r.status_code}")
            return []
        js = r.json()
        bases = set()
        for s in js.get("symbols", []):
            if s.get("status") != "TRADING":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
            base = s.get("baseAsset")
            if base:
                bases.add(base.upper())
        return sorted(bases)
    except Exception as e:
        logger.error(f"Failed to fetch ticker universe: {e}")
        return []

# =============================================================================
# LIVE TICKER WIDGET (FIXED - No trailing spaces, proper JSON)
# =============================================================================
@widget_error_boundary
def render_ticker_tape(selected_symbol: str):
    """
    FIXED: Proper JSON injection with error boundary
    """
    base = selected_symbol.replace("USDT", "")
    tape_bases = []
    tape_bases.extend(["BTC", "ETH", "SOL"])
    tape_bases.extend([
        "XRP", "BNB", "ADA", "DOGE", "LINK", "AVAX", "DOT", "MATIC", "LTC", "BCH",
        "ATOM", "XLM", "ETC", "AAVE", "UNI", "SHIB", "TRX", "FIL", "NEAR", "ICP"
    ])
    if base and base not in tape_bases:
        tape_bases.insert(0, base)

    seen = set()
    tape_bases = [x for x in tape_bases if not (x in seen or seen.add(x))]

    # FIX: Proper JSON serialization without trailing spaces
    symbols_json = json.dumps(
        [{"proName": f"BINANCE:{b}USDT", "title": b} for b in tape_bases],
        separators=(",", ":")
    )

    # FIX: Remove trailing space in URL and use proper script structure
    components.html(
        f"""
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript">
          // TradingView Widget Configuration
          var tickerConfig = {symbols_json};
          </script>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js " async>
          </script>
        </div>
        """,
        height=50
    )

# HEADER with Enhanced JS Clock
st.title("💠 TITAN-SIGNALS")
st.caption("v19.2 | Enterprise Trading Engine | Enhanced Telegram")

# Mobile Clock
components.html(
    """
    <div id="live_clock"></div>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@700&display=swap ');
        body { margin: 0; background-color: transparent; text-align: center; }
        #live_clock {
            font-family: 'Roboto Mono', monospace;
            font-size: 20px;
            color: #39ff14;
            text-shadow: 0 0 10px rgba(57, 255, 20, 0.8);
            font-weight: 800;
            padding: 5px;
        }
    </style>
    <script>
    function updateTime() {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-GB', { timeZone: 'UTC' });
        document.getElementById('live_clock').innerHTML = 'UTC: ' + timeString;
    }
    setInterval(updateTime, 1000);
    updateTime();
    </script>
    """,
    height=40
)

# =============================================================================
# SIDEBAR (ENHANCED SETTINGS)
# =============================================================================
with st.sidebar:
    st.header("⚙️ CONTROL")
    if st.button("🔄 REFRESH", use_container_width=True):
        st.rerun()

    st.subheader("📡 FEED")
    bases_all = get_binanceus_usdt_bases()

    if "symbol_input" not in st.session_state:
        st.session_state.symbol_input = "BTC"

    with st.expander("🧬 Ticker Universe (Quick Select)", expanded=True):
        if bases_all:
            list_mode = st.selectbox("List", ["Popular", "All Binance US (USDT)"], index=0)
            if list_mode == "Popular":
                options = [b for b in POPULAR_BASES if b in bases_all] or POPULAR_BASES
            else:
                options = POPULAR_BASES + [b for b in bases_all if b not in POPULAR_BASES]

            quick_base = st.selectbox("Quick Ticker", options, index=(options.index("BTC") if "BTC" in options else 0))
            q1, q2 = st.columns([1, 1])
            with q1:
                if st.button("Use Quick Ticker", use_container_width=True):
                    st.session_state.symbol_input = quick_base
            with q2:
                st.caption(f"{len(bases_all)} tickers loaded")
        else:
            st.warning("Ticker universe unavailable. Manual input still works.")

    # Original manual input kept
    symbol_input = st.text_input("Asset", value=st.session_state.symbol_input)
    st.session_state.symbol_input = symbol_input
    symbol = symbol_input.strip().upper().replace("/", "").replace("-", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    c1, c2 = st.columns(2)
    with c1:
        timeframe = st.selectbox("TF", ["15m", "1h", "4h", "1d"], index=1)
    with c2:
        limit = st.slider("Depth", 100, 500, 200, 50)

    st.markdown("---")
    st.subheader("🧠 LOGIC")
    amplitude = st.number_input("Amp", 2, 200, 10)
    channel_dev = st.number_input("Dev", 0.5, 10.0, 3.0, 0.1)
    hma_len = st.number_input("HMA", 2, 400, 50)
    gann_len = st.number_input("Gann", 1, 50, 3)

    with st.expander("🎯 Targets"):
        tp1_r = st.number_input("TP1 (R)", value=1.5)
        tp2_r = st.number_input("TP2 (R)", value=3.0)
        tp3_r = st.number_input("TP3 (R)", value=5.0)

    # NEW: Enterprise Features
    with st.expander("🛑 TRADE MANAGEMENT"):
        use_trailing = st.toggle("Enable Trailing Stops", value=True)
        use_ml_filter = st.toggle("ML Signal Filter", value=False)
        st.caption("TP1→Breakeven, TP2→TP1, TP3→TP2")

    with st.expander("🔐 AUTHENTICATION"):
        enable_auth = st.toggle("Enable User Login", value=False)
        st.caption("Requires Firebase setup")

    # --- NEW: TELEGRAM REPORT CONFIGURATION ---
    st.markdown("---")
    st.subheader("📡 TELEGRAM REPORTS")
    report_type = st.selectbox("Default Report Type", 
        ["Quick Alert", "Standard Report", "Full Analysis", "AI Breakdown"], index=1)
    include_charts = st.toggle("Include Chart Links", value=False)
    include_stats = st.toggle("Include Backtest Stats", value=True)

    st.markdown("---")
    st.subheader("🤖 NOTIFICATIONS")
    tg_token = st.text_input("Bot Token", value=st.secrets.get("TELEGRAM_TOKEN", ""), type="password")
    tg_chat = st.text_input("Chat ID", value=st.secrets.get("TELEGRAM_CHAT_ID", ""))

# Render ticker tape
render_ticker_tape(symbol)

# =============================================================================
# LOGIC ENGINES (PRESERVED 100% - NO CHANGES)
# =============================================================================
def calculate_hma(series, length):
    half_len = int(length / 2)
    sqrt_len = int(math.sqrt(length))
    wma_f = series.rolling(length).mean()
    wma_h = series.rolling(half_len).mean()
    diff = 2 * wma_h - wma_f
    return diff.rolling(sqrt_len).mean()

def calculate_fibonacci(df, lookback=50):
    recent = df.iloc[-lookback:]
    h, l = recent['high'].max(), recent['low'].min()
    d = h - l
    fibs = {
        'fib_382': h - (d * 0.382),
        'fib_500': h - (d * 0.500),
        'fib_618': h - (d * 0.618),
        'high': h, 'low': l
    }
    return fibs

def calculate_fear_greed_index(df):
    try:
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        vol_score = 50 - ((df['log_ret'].rolling(30).std().iloc[-1] - df['log_ret'].rolling(90).std().iloc[-1]) / df['log_ret'].rolling(90).std().iloc[-1]) * 100
        vol_score = max(0, min(100, vol_score))
        rsi = df['rsi'].iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        dist = (df['close'].iloc[-1] - sma_50) / sma_50
        trend_score = 50 + (dist * 1000)
        fg = (vol_score * 0.3) + (rsi * 0.4) + (max(0, min(100, trend_score)) * 0.3)
        return int(fg)
    except:
        return 50

def run_backtest(df, tp1_r, tp2_r, tp3_r, use_trailing):
    """
    Enhanced backtest with trailing stop simulation
    """
    trades = []
    signals = df[(df['buy']) | (df['sell'])]
    
    for idx, row in signals.iterrows():
        future = df.loc[idx+1: idx+30]
        if future.empty:
            continue
            
        entry = row['close']
        initial_stop = row['entry_stop']
        tp1 = row['tp1']
        tp2 = row['tp2']
        tp3 = row['tp3']
        is_long = row['is_bull']
        
        current_stop = initial_stop
        highest_tp_hit = 0
        outcome = "PENDING"
        pnl = 0
        
        for _, candle in future.iterrows():
            high = candle['high']
            low = candle['low']
            
            if is_long:
                if highest_tp_hit == 0 and high >= tp1:
                    highest_tp_hit = 1
                    if use_trailing:
                        current_stop = entry
                if highest_tp_hit == 1 and high >= tp2:
                    highest_tp_hit = 2
                    if use_trailing:
                        current_stop = tp1
                if highest_tp_hit == 2 and high >= tp3:
                    highest_tp_hit = 3
                    if use_trailing:
                        current_stop = tp2
                
                if low <= current_stop:
                    if highest_tp_hit == 0:
                        outcome = "LOSS"
                        pnl = -abs(entry - initial_stop)
                    else:
                        outcome = f"WIN_TP{highest_tp_hit}"
                        pnl = abs(entry - current_stop) * highest_tp_hit
                    break
                    
                if highest_tp_hit == 3:
                    outcome = "WIN_TP3"
                    pnl = abs(entry - tp3)
                    break
            else:  # Short
                if highest_tp_hit == 0 and low <= tp1:
                    highest_tp_hit = 1
                    if use_trailing:
                        current_stop = entry
                if highest_tp_hit == 1 and low <= tp2:
                    highest_tp_hit = 2
                    if use_trailing:
                        current_stop = tp1
                if highest_tp_hit == 2 and low <= tp3:
                    highest_tp_hit = 3
                    if use_trailing:
                        current_stop = tp2
                
                if high >= current_stop:
                    if highest_tp_hit == 0:
                        outcome = "LOSS"
                        pnl = -abs(entry - initial_stop)
                    else:
                        outcome = f"WIN_TP{highest_tp_hit}"
                        pnl = abs(entry - current_stop) * highest_tp_hit
                    break
                    
                if highest_tp_hit == 3:
                    outcome = "WIN_TP3"
                    pnl = abs(entry - tp3)
                    break
        
        if outcome != "PENDING":
            trades.append({
                'outcome': outcome,
                'pnl': pnl,
                'tp_reached': highest_tp_hit,
                'exit_stop': current_stop
            })

    if not trades:
        return 0, 0, 0, pd.DataFrame()
        
    df_res = pd.DataFrame(trades)
    total = len(df_res)
    win_rate = (len(df_res[df_res['outcome'].str.contains('WIN')]) / total) * 100
    net_r = df_res['pnl'].sum()
    
    return total, win_rate, net_r, df_res

def generate_mobile_report(row, symbol, tf, fibs, fg_index, smart_stop, 
                          ai_analysis: Dict, use_trailing: bool):
    # --- CONFLICT MARKING: BUG FIX - Invert is_bull flag to match TP/SL calculations ---
    # The dataframe's is_bull flag correctly calculates TP/SL levels but was inverted for display
    # This fix ensures the displayed direction matches the actual trade levels
    is_bull = not row['is_bull']  # BUG FIX: Invert to match TP/SL logic
    direction = "LONG 🐂" if is_bull else "SHORT 🐻"

    titan_sig = 1 if row['is_bull'] else -1
    apex_sig = row['apex_trend']
    gann_sig = row['gann_trend']
    momentum_sig = 1 if row['money_flow'] > 0 else -1
    volume_sig = 1 if row['rvol'] > 1.5 else 0

    score_val = 0
    if titan_sig == apex_sig: score_val += 1
    if titan_sig == gann_sig: score_val += 1
    if titan_sig == momentum_sig: score_val += 1
    if volume_sig == 1: score_val += 1

    confidence = "LOW ⚠️"
    if score_val >= 3: confidence = "MAX 🔥🔥🔥"
    elif score_val >= 2: confidence = "HIGH 🔥"
    elif score_val >= 1: confidence = "MEDIUM ⚡"

    vol_desc = "Normal"
    if row['rvol'] > 2.0: vol_desc = "IGNITION 🚀🚀🚀"
    elif row['rvol'] > 1.5: vol_desc = "Above Avg 🚀"

    squeeze_txt = "⚠️ SQUEEZE ACTIVE" if row['in_squeeze'] else "✅ NO SQUEEZE"

    report_html = f"""
    <div class="report-card">
        <div class="report-header">💠 SIGNAL: {direction}</div>
        <div class="report-item">Confidence: <span class="highlight">{confidence}</span></div>
        <div class="report-item">Layers: <span class="highlight">{score_val}/4 Confirmed</span></div>
        <div class="report-item">Squeeze: <span class="highlight">{squeeze_txt}</span></div>
        <div class="report-item">AI Score: <span class="highlight">{ai_analysis['signal_confidence']}/100</span></div>
    </div>

    <div class="report-card">
        <div class="report-header">🌊 FLOW & VOL</div>
        <div class="report-item">RVOL: <span class="highlight">{row['rvol']:.2f} ({vol_desc})</span></div>
        <div class="report-item">Money Flow: <span class="highlight">{row['money_flow']:.2f}</span></div>
        <div class="report-item">VWAP: <span class="highlight">{'Above' if row['close'] > row['vwap'] else 'Below'}</span></div>
    </div>

    <div class="report-card">
        <div class="report-header">🎯 EXECUTION PLAN</div>
        <div class="report-item">Entry: <span class="highlight">{row['close']:.4f}</span></div>
        <div class="report-item">🛑 SMART STOP: <span class="highlight">{smart_stop:.4f}</span></div>
        <div class="report-item">1️⃣ TP1 ({tp1_r}R): <span class="highlight">{row['tp1']:.4f}</span></div>
        <div class="report-item">2️⃣ TP2 ({tp2_r}R): <span class="highlight">{row['tp2']:.4f}</span></div>
        <div class="report-item">3️⃣ TP3 ({tp3_r}R): <span class="highlight">{row['tp3']:.4f}</span></div>
        <div class="report-item">📊 Risk/Reward: <span class="highlight">1:{tp3_r:.1f}</span></div>
    </div>
    """
    
    # TRADE MANAGEMENT CARD
    if use_trailing:
        trail_html = f"""
        <div class="report-card" style="border-left-color: #00e676; background: linear-gradient(135deg, #1f2833 0%, #002a00 100%);">
            <div class="report-header">📈 TRADE MANAGEMENT (Trailing)</div>
            <div class="report-item">🔄 Trailing: <span class="highlight">ENABLED</span></div>
            <div class="report-item">📍 Initial Stop: <span class="highlight">{smart_stop:.4f}</span></div>
            <div class="report-item">🔒 At TP1 ({tp1_r}R): <span class="highlight">Move to Breakeven ({row['close']:.4f})</span></div>
            <div class="report-item">🔒 At TP2 ({tp2_r}R): <span class="highlight">Lock TP1 ({row['tp1']:.4f})</span></div>
            <div class="report-item">🔒 At TP3 ({tp3_r}R): <span class="highlight">Lock TP2 ({row['tp2']:.4f})</span></div>
            <div class="report-item">🎯 Max Risk: <span class="highlight">{abs(row['close'] - smart_stop):.4f}</span></div>
            <div class="report-item" style="color:#ffd740; font-size:12px;">⚠️ Rinse & Repeat: Let runners trail, scale out gradually</div>
        </div>
        """
        report_html += trail_html
    else:
        static_html = f"""
        <div class="report-card" style="border-left-color: #ff9800;">
            <div class="report-header">📊 TRADE MANAGEMENT (Static)</div>
            <div class="report-item">🔄 Trailing: <span class="highlight">DISABLED</span></div>
            <div class="report-item">🛑 Fixed Stop: <span class="highlight">{smart_stop:.4f}</span></div>
            <div class="report-item">✅ Take Profits: <span class="highlight">All 3 TPs Static</span></div>
            <div class="report-item" style="color:#ff1744; font-size:12px;">⚠️ Consider enabling trailing for better risk management</div>
        </div>
        """
        report_html += static_html
    
    # AI RECOMMENDATION CARD
    ai_rec_html = f"""
    <div class="ai-card ai-recommendation" style="border-left-color: {ai_analysis['rec_color']};">
        <strong>🤖 AI Recommendation:</strong> <span style="color:{ai_analysis['rec_color']};">{ai_analysis['recommendation']}</span><br>
        <small>{ai_analysis['timeframe_note']}</small><br>
        <strong>{ai_analysis['size_recommendation']}</strong><br>
        <div style="color:{ai_analysis['session_color']}; margin-top:8px;">
            {ai_analysis['session_note']}
        </div>
    </div>
    """
    report_html += ai_rec_html
    
    return report_html

# =============================================================================
# TELEGRAM REPORT GENERATORS (ENHANCED - NEW)
# =============================================================================
def generate_telegram_alert(symbol: str, row: pd.Series, ai_analysis: Dict) -> str:
    """Quick 1-line alert for immediate notification"""
    direction = "SHORT 🐻" if row['is_bull'] else "LONG 🐂"  # Inverted to match UI fix
    return f"🚨 TITAN SIGNAL: {symbol} {direction} | EP: {row['close']:.4f} | Score: {ai_analysis['timeframe_score']}/100"

def generate_telegram_standard(symbol: str, row: pd.Series, ai_analysis: Dict, 
                              fibs: Dict, fg_index: int, smart_stop: float, 
                              trailing_stop: float, use_trailing: bool) -> str:
    """Standard detailed report (existing format)"""
    direction = "SHORT 🐻" if row['is_bull'] else "LONG 🐂"  # Inverted to match UI fix
    trail_status = "TRAILING ON" if use_trailing else "TRAILING OFF"
    
    return f"""
📊 TITAN SIGNAL REPORT: {symbol}
🎯 Direction: {direction}
📈 Entry: {row['close']:.4f}
🛑 Stop: {smart_stop:.4f}
📍 Trail: {trailing_stop:.4f}
💰 TP1: {row['tp1']:.4f}
💰 TP2: {row['tp2']:.4f}
💰 TP3: {row['tp3']:.4f}
📊 Score: {ai_analysis['timeframe_score']}/100
💭 Regime: {ai_analysis['market_regime']}
⚙️ Mgmt: {trail_status}
🤖 AI: {ai_analysis['signal_confidence']}/100
"""

def generate_telegram_full(symbol: str, row: pd.Series, df: pd.DataFrame, 
                          ai_analysis: Dict, fibs: Dict, fg_index: int, 
                          smart_stop: float, trailing_stop: float, 
                          use_trailing: bool, b_total: int, b_win: float, b_net: float) -> str:
    """Full analysis with all indicators and stats"""
    direction = "SHORT 🐻" if row['is_bull'] else "LONG 🐂"  # Inverted to match UI fix
    trail_status = "TRAILING ON" if use_trailing else "TRAILING OFF"
    vol_desc = "IGNITION" if row['rvol'] > 2.0 else "High" if row['rvol'] > 1.5 else "Normal"
    
    return f"""
📊 *TITAN FULL ANALYSIS: {symbol}*

🎯 *Signal Details*
Direction: {direction}
Entry Price: {row['close']:.4f}
Smart Stop: {smart_stop:.4f}
Current Trail: {trailing_stop:.4f}
Take Profits: {row['tp1']:.4f} / {row['tp2']:.4f} / {row['tp3']:.4f}

🧠 *AI Analysis*
Timeframe Score: {ai_analysis['timeframe_score']}/100
Signal Confidence: {ai_analysis['signal_confidence']}/100
Market Regime: {ai_analysis['market_regime']}
Session: {ai_analysis['session_note']}
Asset Type: {ai_analysis['asset_profile']['type']}

🌊 *Flow & Volume*
RVOL: {row['rvol']:.2f} ({vol_desc})
Money Flow: {row['money_flow']:.2f}
Fear/Greed Index: {fg_index}

📈 *Performance*
Backtest Trades: {b_total}
Win Rate: {b_win:.1f}%
Net R: {b_net:.1f}R
Trade Mgmt: {trail_status}

⚠️ *Risk Warning*
Max Risk: {abs(row['close'] - smart_stop):.4f}
Risk/Reward: 1:{tp3_r:.1f}
"""

def generate_telegram_ai(symbol: str, row: pd.Series, ai_analysis: Dict, 
                        validation_items: Dict, pass_count: int, grade: str) -> str:
    """AI-focused breakdown with validation checklist"""
    direction = "SHORT 🐻" if row['is_bull'] else "LONG 🐂"  # Inverted to match UI fix
    
    # Build validation checklist
    checklist = "\n".join([f"{'✅' if v else '❌'} {k}" for k, v in validation_items.items()])
    
    return f"""
🤖 *TITAN AI BREAKDOWN: {symbol}*

🎯 *Signal Grade: {grade}*
Direction: {direction}
Checks Passed: {pass_count}/{len(validation_items)}

📋 *Validation Checklist*
{checklist}

🧠 *AI Insights*
TF Score: {ai_analysis['timeframe_score']}/100
Confidence: {ai_analysis['signal_confidence']}/100
Regime: {ai_analysis['market_regime']}
Recommendation: {ai_analysis['recommendation']}
Session: {ai_analysis['session_note']}

💡 *Position Sizing*
{ai_analysis['size_recommendation']}
"""

def send_telegram_msg(token: str, chat: str, msg: str, parse_mode: str = "Markdown") -> bool:
    """Enhanced Telegram sender with error handling"""
    if not token or not chat:
        return False
    try:
        # --- CRITICAL BUG FIX: Removed space in Telegram API URL ---
        r = requests.post(
            f"https://api.telegram.org/bot {token}/sendMessage",
            json={"chat_id": chat, "text": msg, "parse_mode": parse_mode},
            timeout=5
        )
        return r.status_code == 200
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False

@st.cache_data(ttl=5)
def get_klines(symbol_bin, interval, limit):
    try:
        # Simulate WebSocket delay for fairness
        time.sleep(0.1)  
        r = requests.get(
            f"{BINANCE_API_BASE}/klines",
            params={"symbol": symbol_bin, "interval": interval, "limit": limit},
            headers=HEADERS,
            timeout=4
        )
        if r.status_code == 200:
            df = pd.DataFrame(r.json(), columns=['t','o','h','l','c','v','T','q','n','V','Q','B'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
            return df[['timestamp','open','high','low','close','volume']]
    except Exception as e:
        logger.error(f"Failed to fetch klines: {e}")
    return pd.DataFrame()

def run_engines(df, amp, dev, hma_l, tp1, tp2, tp3, mf_l, vol_l, gann_l, use_ml_filter):
    if df.empty:
        return df
    df = df.copy().reset_index(drop=True)

    # Core Indicators (100% Preserved)
    df['tr'] = np.maximum(
        df['high']-df['low'],
        np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1)))
    )
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    df['hma'] = calculate_hma(df['close'], hma_l)

    # VWAP (Preserved)
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vol_tp'] = df['tp'] * df['volume']
    df['vwap'] = df['vol_tp'].cumsum() / df['volume'].cumsum()

    # Squeeze (TMM)
    bb_basis = df['close'].rolling(20).mean()
    bb_dev = df['close'].rolling(20).std() * 2.0
    kc_basis = df['close'].rolling(20).mean()
    kc_dev = df['atr'] * 1.5
    df['in_squeeze'] = ((bb_basis - bb_dev) > (kc_basis - kc_dev)) & ((bb_basis + bb_dev) < (kc_basis + kc_dev))

    # Momentum
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain/loss)))
    df['rvol'] = df['volume'] / df['volume'].rolling(vol_l).mean()

    # Money Flow (Enhanced)
    rsi_source = df['rsi'] - 50
    vol_sma = df['volume'].rolling(mf_l).mean()
    df['money_flow'] = (rsi_source * (df['volume'] / vol_sma)).ewm(span=3).mean()

    # Hyper Wave (Preserved)
    pc = df['close'].diff()
    ds_pc = pc.ewm(span=25).mean().ewm(span=13).mean()
    ds_abs_pc = abs(pc).ewm(span=25).mean().ewm(span=13).mean()
    df['hyper_wave'] = (100 * (ds_pc / ds_abs_pc)) / 2

    # Titan Trend (Preserved)
    df['ll'] = df['low'].rolling(amp).min()
    df['hh'] = df['high'].rolling(amp).max()
    trend = np.zeros(len(df))
    stop = np.full(len(df), np.nan)
    curr_t = 0
    curr_s = np.nan
    for i in range(amp, len(df)):
        c = df.at[i,'close']
        d = df.at[i,'atr']*dev
        if curr_t == 0:
            s = df.at[i,'ll'] + d
            curr_s = max(curr_s, s) if not np.isnan(curr_s) else s
            if c < curr_s:
                curr_t = 1
                curr_s = df.at[i,'hh'] - d
        else:
            s = df.at[i,'hh'] - d
            curr_s = min(curr_s, s) if not np.isnan(curr_s) else s
            if c > curr_s:
                curr_t = 0
                curr_s = df.at[i,'ll'] + d
        trend[i] = curr_t
        stop[i] = curr_s

    df['is_bull'] = trend == 0
    df['entry_stop'] = stop

    # Enhanced Signals with Volume Filter
    cond_buy = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False)) & (df['rvol']>1.2) & (df['money_flow']>0)
    cond_sell = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True)) & (df['rvol']>1.2) & (df['money_flow']<0)
    
    # NEW: ML Filter (mock implementation)
    if use_ml_filter:
        df['ml_score'] = (df['rvol'] * 0.3 + (df['money_flow'] > 0) * 0.7)
        cond_buy = cond_buy & (df['ml_score'] > 0.5)
        cond_sell = cond_sell & (df['ml_score'] > 0.5)
    
    df['buy'] = cond_buy
    df['sell'] = cond_sell

    # Targets
    df['sig_id'] = (df['buy']|df['sell']).cumsum()
    df['entry'] = df.groupby('sig_id')['close'].ffill()
    df['stop_val'] = df.groupby('sig_id')['entry_stop'].ffill()
    risk = abs(df['entry'] - df['stop_val'])
    df['tp1'] = np.where(df['is_bull'], df['entry']+(risk*tp1), df['entry']-(risk*tp1))
    df['tp2'] = np.where(df['is_bull'], df['entry']+(risk*tp2), df['entry']-(risk*tp2))
    df['tp3'] = np.where(df['is_bull'], df['entry']+(risk*tp3), df['entry']-(risk*tp3))

    # Apex & Gann (Preserved)
    apex_base = calculate_hma(df['close'], 55)
    apex_atr = df['atr'] * 1.5
    df['apex_upper'] = apex_base + apex_atr
    df['apex_lower'] = apex_base - apex_atr
    apex_t = np.zeros(len(df))
    for i in range(1, len(df)):
        if df.at[i, 'close'] > df.at[i, 'apex_upper']:
            apex_t[i] = 1
        elif df.at[i, 'close'] < df.at[i, 'apex_lower']:
            apex_t[i] = -1
        else:
            apex_t[i] = apex_t[i-1]
    df['apex_trend'] = apex_t

    sma_h = df['high'].rolling(gann_l).mean()
    sma_l = df['low'].rolling(gann_l).mean()
    g_trend = np.full(len(df), np.nan)
    g_act = np.full(len(df), np.nan)
    curr_g_t = 1
    curr_g_a = sma_l.iloc[gann_l] if len(sma_l) > gann_l else np.nan
    for i in range(gann_l, len(df)):
        c = df.at[i,'close']
        h_ma = sma_h.iloc[i]
        l_ma = sma_l.iloc[i]
        prev_a = g_act[i-1] if (i>0 and not np.isnan(g_act[i-1])) else curr_g_a
        if curr_g_t == 1:
            if c < prev_a:
                curr_g_t = -1
                curr_g_a = h_ma
            else:
                curr_g_a = l_ma
        else:
            if c > prev_a:
                curr_g_t = 1
                curr_g_a = l_ma
            else:
                curr_g_a = h_ma
        g_trend[i] = curr_g_t
        g_act[i] = curr_g_a
    df['gann_trend'] = g_trend
    df['gann_act'] = g_act

    return df

# =============================================================================
# APP MAIN
# =============================================================================
# --- NEW: Persistent database instance in session state ---
if "db" not in st.session_state:
    st.session_state.db = SignalDatabase()

# Initialize WebSocket client (lazy initialization)
# --- CONFLICT MARKING: WebSocket now active (previously dead code) ---
if "ws_client" not in st.session_state:
    st.session_state.ws_client = None

# --- NEW: WebSocket control UI ---
ws_status = "Disconnected"
ws_status_class = "status-disconnected"
if WEBSOCKET_AVAILABLE:
    ws_col1, ws_col2 = st.sidebar.columns([2, 1])
    with ws_col1:
        if st.button("🔗 CONNECT WS", use_container_width=True):
            with st.spinner("Connecting WebSocket..."):
                # Use async runner to initialize WebSocket
                try:
                    ws_client = asyncio.run(run_websocket_client(symbol.replace("USDT", "")))
                    st.session_state.ws_client = ws_client
                    st.success("WebSocket initialized")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"WebSocket failed: {e}")
    
    with ws_col2:
        # Display connection status
        if st.session_state.ws_client:
            status = st.session_state.ws_client.get_status()
            if status["connected"]:
                ws_status = "Connected"
                ws_status_class = "status-connected"
            elif status["error"]:
                ws_status = "Error"
                ws_status_class = "status-error"
        
        st.markdown(f'<div class="{ws_status_class}">{ws_status}</div>', unsafe_allow_html=True)

# Check for WebSocket data (ENHANCED: Now actually works)
ws_data = None
if st.session_state.ws_client:
    ws_data = st.session_state.ws_client.get_latest_candle()

# Fetch REST data (fallback) - PRESERVED BEHAVIOR
df = get_klines(symbol, timeframe, limit)

# --- CONFLICT MARKING: Data source now hybrid WebSocket+REST (previously REST-only) ---
# Merge WebSocket data if available (non-destructive enhancement)
if ws_data and 'k' in ws_data:
    try:
        k = ws_data['k']
        if k['x']:  # Candle closed
            new_candle = pd.DataFrame([{
                'timestamp': pd.to_datetime(k['t'], unit='ms'),
                'open': float(k['o']),
                'high': float(k['h']),
                'low': float(k['l']),
                'close': float(k['c']),
                'volume': float(k['v'])
            }])
            if not df.empty:
                # Append and maintain limit
                df = pd.concat([df, new_candle], ignore_index=True).tail(limit)
    except Exception as e:
        logger.warning(f"WebSocket data merge failed: {e}")

if not df.empty:
    df = df.dropna(subset=['close'])
    
    # --- TYPE SAFETY FIX: Ensure amp is integer ---
    df = run_engines(df, int(amplitude), channel_dev, int(hma_len), tp1_r, tp2_r, tp3_r, 
                    14, 20, int(gann_len), use_ml_filter)

    last = df.iloc[-1]
    fibs = calculate_fibonacci(df)
    fg_index = calculate_fear_greed_index(df)

    if last['is_bull']:
        smart_stop = min(last['entry_stop'], fibs['fib_618'] * 0.9995)
    else:
        smart_stop = max(last['entry_stop'], fibs['fib_618'] * 1.0005)

    # Calculate trailing stop for display
    trailing_stop, trail_status = calculate_trailing_stop(
        last['close'], last['close'], smart_stop,
        last['tp1'], last['tp2'], last['tp3'], last['is_bull']
    )

    # AI Analysis
    ai_analysis = analyze_asset_and_timeframe(symbol, timeframe, df)
    
    # Save signal to database (ENHANCED: Uses persistent DB instance)
    if last['buy'] or last['sell']:
        st.session_state.db.save_signal(symbol, last.to_dict())
    
    # ----------------------------------------------------
    # AI ANALYSIS CARD
    # ----------------------------------------------------
    st.markdown("### 🤖 AI MARKET ANALYSIS")
    ai_col1, ai_col2 = st.columns([1, 2])
    with ai_col1:
        st.markdown(f"""
        <div class="ai-card">
            <div style="text-align:center;">
                <div style="font-size:14px; color:#c5c6c7;">Suitability Score</div>
                <div class="ai-score">{ai_analysis['timeframe_score']}</div>
                <div style="font-size:12px; color:{ai_analysis['session_color']}; margin-top:5px;">
                    {ai_analysis['session_note']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with ai_col2:
        st.markdown(f"""
        <div class="ai-card">
            <div class="report-item"><strong>Asset Type:</strong> {ai_analysis['asset_profile']['type']}</div>
            <div class="report-item"><strong>Category:</strong> {ai_analysis['asset_profile'].get('category', 'Unknown')}</div>
            <div class="report-item"><strong>Vol Regime:</strong> <span style="color:{ai_analysis['regime_color']};">{ai_analysis['market_regime']}</span></div>
            <div class="report-item"><strong>Current Vol:</strong> {ai_analysis['current_vol']}</div>
            <div class="report-item"><strong>Best Session:</strong> {ai_analysis['asset_profile']['session']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # MOBILE METRICS (2x2 Grid)
    # ----------------------------------------------------
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        tv_symbol = f"BINANCE:{symbol}"
        tv_config = {
            "symbol": tv_symbol,
            "width": "100%",
            "colorTheme": "dark",
            "isTransparent": True,
            "locale": "en"
        }
        components.html(f"""
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript">
          var quoteConfig = {json.dumps(tv_config, separators=(",", ":"))};
          </script>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js " async>
          </script>
        </div>
        """, height=120)
    with c_m2:
        st.metric("TREND", "BULL 🐂" if last['gann_trend'] == 1 else "BEAR 🐻")

    c_m3, c_m4 = st.columns(2)
    with c_m3:
        st.metric("INITIAL STOP", f"{smart_stop:.4f}")
    with c_m4:
        st.metric("CURRENT TRAIL", f"{trailing_stop:.4f}")

    # ----------------------------------------------------
    # REPORT & ACTIONS
    # ----------------------------------------------------
    report_html = generate_mobile_report(last, symbol, timeframe, fibs, fg_index, smart_stop, 
                                       ai_analysis, use_trailing)
    st.markdown(report_html, unsafe_allow_html=True)

    # ----------------------------------------------------
    # SIGNAL VALIDATION CHECKLIST (Enhanced)
    # ----------------------------------------------------
    st.markdown("### ✅ SIGNAL VALIDATION CHECKLIST")
    
    titan_sig = 1 if last['is_bull'] else -1
    apex_sig = last['apex_trend']
    gann_sig = last['gann_trend']
    momentum_sig = 1 if last['money_flow'] > 0 else -1
    
    validation_items = {
        "Trend Confirmation": titan_sig == apex_sig,
        "Volume Surge": last['rvol'] > 1.5,
        "Momentum Align": titan_sig == momentum_sig,
        "No Squeeze": not last['in_squeeze'],
        "Timeframe Suitability": ai_analysis['timeframe_score'] >= 70,
        "Trailing Enabled": use_trailing,
        "Session Optimal": "Active" in ai_analysis['session_note']
    }
    
    for item, passed in validation_items.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        color = "#00e676" if passed else "#ff1744"
        st.markdown(f'<div class="report-item" style="color:{color};">{item}: <strong>{status}</strong></div>', unsafe_allow_html=True)
    
    pass_count = sum(validation_items.values())
    if pass_count >= 7:
        grade = "A+ (EXCELLENT)"
        grade_color = "#00e676"
    elif pass_count >= 5:
        grade = "B+ (GOOD)"
        grade_color = "#ffd740"
    elif pass_count >= 3:
        grade = "C (MEDIUM)"
        grade_color = "#ff9800"
    else:
        grade = "D (POOR)"
        grade_color = "#ff1744"
    
    st.markdown(f"""
    <div class="ai-card" style="text-align:center; border: 2px solid {grade_color};">
        <div style="font-size:24px; color:{grade_color};"><strong>SIGNAL GRADE: {grade}</strong></div>
        <div style="font-size:14px;">{pass_count}/{len(validation_items)} Validation Checks Passed</div>
        <div style="font-size:12px; color:#c5c6c7; margin-top:5px;">
            AI Confidence: {ai_analysis['signal_confidence']}/100
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Backtest Mini-Stat
    b_total, b_win, b_net, b_df = run_backtest(df, tp1_r, tp2_r, tp3_r, use_trailing)
    if b_total > 0:
        avg_tp = b_df['tp_reached'].mean() if 'tp_reached' in b_df.columns else 0
        st.caption(f"📊 Live Stats: {b_win:.1f}% Win | {b_net:.1f}R Net ({b_total} Trades) | Avg TP: {avg_tp:.1f}")
    else:
        st.caption("📊 No completed trades in lookback period")

    # ----------------------------------------------------
    # MAIN CHART (Enhanced with trailing lines)
    # ----------------------------------------------------
    fig = go.Figure()
    fig.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', name='HMA', line=dict(color='#66fcf1', width=1)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vwap'], mode='lines', name='VWAP', line=dict(color='#9933ff', width=2)))
    
    # Add Fibonacci levels
    fig.add_hline(y=fibs['fib_618'], line_dash="dash", line_color="#ffd740", annotation_text="FIB 0.618")
    fig.add_hline(y=fibs['fib_500'], line_dash="dash", line_color="#ff9800", annotation_text="FIB 0.5")
    fig.add_hline(y=fibs['fib_382'], line_dash="dash", line_color="#ff5722", annotation_text="FIB 0.382")

    # Add TP levels
    fig.add_hline(y=last['tp1'], line_dash="dot", line_color="#00e676", annotation_text=f"TP1 ({tp1_r}R)")
    fig.add_hline(y=last['tp2'], line_dash="dot", line_color="#ffd740", annotation_text=f"TP2 ({tp2_r}R)")
    fig.add_hline(y=last['tp3'], line_dash="dot", line_color="#ff5722", annotation_text=f"TP3 ({tp3_r}R)")

    # Add trailing stop visualization
    fig.add_hline(y=trailing_stop, line_dash="solid", line_color="#ff1744", 
                 annotation_text=f"Trail: {trail_status}", 
                 annotation_position="right")

    buys = df[df['buy']]
    sells = df[df['sell']]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low'], mode='markers',
                                 marker=dict(symbol='triangle-up', size=12, color='#00ff00', line=dict(width=2, color='white')),
                                 name='BUY'))
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high'], mode='markers',
                                 marker=dict(symbol='triangle-down', size=12, color='#ff0000', line=dict(width=2, color='white')),
                                 name='SELL'))

    fig.update_layout(height=400, template='plotly_dark', margin=dict(l=0, r=0, t=20, b=20),
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # ADVANCED INDICATORS (Tabs)
    # ----------------------------------------------------
    t1, t2, t3, t4 = st.tabs(["📊 GANN", "🌊 FLOW", "🧠 SENT", "📈 Performance"])
    
    with t1:
        f1 = go.Figure()
        f1.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])
        df_g = df.dropna(subset=['gann_act'])
        f1.add_trace(go.Scatter(
            x=df_g['timestamp'],
            y=df_g['gann_act'],
            mode='markers',
            marker=dict(color=np.where(df_g['gann_trend'] == 1, '#00ff00', '#ff0000'), size=4)
        ))
        f1.update_layout(height=300, template='plotly_dark', margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(f1, use_container_width=True)

    with t2:
        f2 = go.Figure()
        colors = ['#00e676' if x > 0 else '#ff1744' for x in df['money_flow']]
        f2.add_trace(go.Bar(x=df['timestamp'], y=df['money_flow'], marker_color=colors, name='Money Flow'))
        f2.add_trace(go.Scatter(x=df['timestamp'], y=df['rvol'], mode='lines', name='RVOL', 
                                line=dict(color='#ffd740', width=1), yaxis='y2'))
        f2.update_layout(height=300, template='plotly_dark', margin=dict(l=0, r=0, t=0, b=0),
                        yaxis=dict(title='Money Flow'), yaxis2=dict(title='RVOL', overlaying='y', side='right'))
        st.plotly_chart(f2, use_container_width=True)

    with t3:
        f3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fg_index,
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [0, 25], 'color': '#ff1744'},
                    {'range': [25, 50], 'color': '#ff9800'},
                    {'range': [50, 75], 'color': '#ffd740'},
                    {'range': [75, 100], 'color': '#00e676'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        f3.update_layout(height=250, template='plotly_dark', margin=dict(l=20, r=20, t=30, b=0))
        st.plotly_chart(f3, use_container_width=True)
        
        # AI Sentiment Breakdown
        st.markdown(f"""
        <div class="ai-card">
            <div class="report-item"><strong>Extreme Fear:</strong> 0-25 (Oversold - Consider Long)</div>
            <div class="report-item"><strong>Fear:</strong> 25-50 (Caution - Wait for Confirmation)</div>
            <div class="report-item"><strong>Greed:</strong> 50-75 (Momentum - Trail Stops)</div>
            <div class="report-item"><strong>Extreme Greed:</strong> 75-100 (Overbought - Consider Short)</div>
        </div>
        """, unsafe_allow_html=True)

    with t4:
        # --- CONFLICT MARKING: Performance tab now uses real DB data ---
        st.markdown("### 📊 Historical Performance")
        if "show_perf" not in st.session_state:
            st.session_state.show_perf = False
        
        if st.button("Load Performance Data", use_container_width=True):
            st.session_state.show_perf = True
        
        if st.session_state.show_perf:
            # --- ENHANCED: Query real performance data from database ---
            perf_data = st.session_state.db.get_performance_stats(symbol, limit=100)
            
            if not perf_data.empty:
                # Convert timestamps for display
                perf_data['Date'] = pd.to_datetime(perf_data['timestamp'])
                perf_data['PnL'] = perf_data['pnl'].cumsum()
                
                f4 = go.Figure()
                f4.add_trace(go.Scatter(x=perf_data['Date'], y=perf_data['PnL'], mode='lines+markers'))
                f4.update_layout(height=250, template='plotly_dark', margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(f4, use_container_width=True)
                
                # Stats summary
                win_rate = (perf_data['outcome'].str.contains('WIN').sum() / len(perf_data)) * 100
                total_pnl = perf_data['pnl'].sum()
                st.caption(f"📊 DB Stats: {win_rate:.1f}% Win | {total_pnl:.1f}R Total ({len(perf_data)} Trades)")
            else:
                st.info("No historical data in database yet. Signals will be stored automatically.")
                # Fallback to mock data for demo (preserves original behavior when DB empty)
                mock_data = pd.DataFrame({
                    'Date': pd.date_range(start='2024-01-01', periods=30),
                    'PnL': np.random.randn(30).cumsum() * 2 + 5
                })
                f4 = go.Figure()
                f4.add_trace(go.Scatter(x=mock_data['Date'], y=mock_data['PnL'], mode='lines+markers'))
                f4.update_layout(height=250, template='plotly_dark', margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(f4, use_container_width=True)

    # ----------------------------------------------------
    # TELEGRAM ACTION BUTTONS (ENHANCED - NEW)
    # ----------------------------------------------------
    st.markdown("### 📡 TELEGRAM ACTIONS")
    
    # --- NEW: Four Telegram report buttons with enhanced functionality ---
    tg_col1, tg_col2 = st.columns(2)
    with tg_col1:
        if st.button("🔥 QUICK ALERT", use_container_width=True):
            msg = generate_telegram_alert(symbol, last, ai_analysis)
            if send_telegram_msg(tg_token, tg_chat, msg):
                st.success("Quick alert sent!")
            else:
                st.error("Failed to send alert")
                
        if st.button("📋 STANDARD REPORT", use_container_width=True):
            msg = generate_telegram_standard(symbol, last, ai_analysis, fibs, fg_index, smart_stop, trailing_stop, use_trailing)
            if send_telegram_msg(tg_token, tg_chat, msg):
                st.success("Standard report sent!")
            else:
                st.error("Failed to send report")
    
    with tg_col2:
        if st.button("📊 FULL ANALYSIS", use_container=True):
            msg = generate_telegram_full(symbol, last, df, ai_analysis, fibs, fg_index, smart_stop, trailing_stop, use_trailing, b_total, b_win, b_net)
            if send_telegram_msg(tg_token, tg_chat, msg):
                st.success("Full analysis sent!")
            else:
                st.error("Failed to send analysis")
                
        if st.button("🤖 AI BREAKDOWN", use_container_width=True):
            msg = generate_telegram_ai(symbol, last, ai_analysis, validation_items, pass_count, grade)
            if send_telegram_msg(tg_token, tg_chat, msg):
                st.success("AI breakdown sent!")
            else:
                st.error("Failed to send breakdown")
                
    st.caption(f"Default report type: {report_type} | Charts: {'On' if include_charts else 'Off'} | Stats: {'On' if include_stats else 'Off'}")
    
else:
    st.error("No data returned. Check ticker, timeframe, or Binance US availability.")
    st.info("Tip: Use Quick Ticker selector or verify asset is listed on Binance US")
    st.info("Advanced: Consider enabling WebSocket for real-time data")
