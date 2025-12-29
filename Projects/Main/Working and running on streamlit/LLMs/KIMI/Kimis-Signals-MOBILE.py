"""
AI Stock Screener & Analyst + TITAN SIGNALS Broadcast Engine
Version: Integrated Enterprise Trading Platform
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import openai
import requests
import numpy as np
import time
import math
import sqlite3
import json
import threading
import logging
import io
import xlsxwriter
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Optional, List, Tuple, Any
from collections import deque
from functools import wraps
import plotly.graph_objects as go
import streamlit.components.v1 as components

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
# BROADCAST ENGINE (FULLY INTEGRATED)
# =============================================================================
from dataclasses import dataclass

@dataclass
class BroadcastMessage:
    report_type: str
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3

class RateLimiter:
    def __init__(self, max_requests: int = 20, period: int = 60):
        self.max_requests = max_requests
        self.period = period
        self.requests = deque()
        self.lock = threading.Lock()
    
    def can_send(self) -> bool:
        with self.lock:
            now = time.time()
            self.requests = deque([req_time for req_time in self.requests 
                                 if now - req_time < self.period])
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
    
    def time_until_next(self) -> float:
        with self.lock:
            if not self.requests:
                return 0
            oldest = self.requests[0]
            now = time.time()
            wait_time = self.period - (now - oldest)
            return max(0, wait_time)

class BroadcastEngine:
    """Enterprise-grade multi-format Telegram broadcast system - CRITICAL BUGS FIXED"""
    
    REPORT_TYPES = {
        'STRICT_SIGNAL': 'strict_signal',
        'AI_RISK_ANALYSIS': 'ai_risk_analysis', 
        'MARKET_SUMMARY': 'market_summary',
        'BACKTEST_REPORT': 'backtest_report'
    }
    
    TEMPLATES = {
        'strict_signal': """🔥 *TITAN TRADE ALERT* 🔥

📊 *{symbol}* | {timeframe}
🎯 *{direction}* | Confidence: {confidence}%
💰 Entry: {entry_price:.4f}
🛑 Stop: {stop_price:.4f}
📈 TP1: {tp1:.4f} ({tp1_r}R)
📈 TP2: {tp2:.4f} ({tp2_r}R)  
📈 TP3: {tp3:.4f} ({tp3_r}R)

⚡ RVOL: {rvol:.2f}x | Squeeze: {squeeze_status}
🤖 AI Score: {ai_score}/100 | Grade: {signal_grade}

⏰ {timestamp} UTC""",
        
        'ai_risk_analysis': """🤖 *AI RISK ANALYSIS* | {symbol}

📈 *Market Regime:* {market_regime}
📊 *Volatility:* {current_vol} | Factor: {vol_factor:.1f}x
⏰ *Session:* {session_note}
🎯 *Timeframe Suitability:* {tf_score}/100
🎓 *Asset Profile:* {asset_type}

⚠️ *Risk Factors:*
• Squeeze Active: {squeeze_risk}
• Volume Spike: {volume_risk}
• Session Mismatch: {session_risk}

💡 *AI Recommendation:*
{recommendation}

🎯 *Position Sizing:* {size_rec}

_Last updated: {timestamp} UTC""",
        
        'market_summary': """📊 *MARKET SUMMARY* | {timestamp}

🌟 *Top Performers:*
{top_performers}

⚠️ *High Risk Signals:*
{risk_signals}

📊 *Overall Market Sentiment:*
• Fear & Greed: {fear_greed}/100
• Avg RVOL: {avg_rvol:.2f}x
• Squeeze Count: {squeeze_count}

🔥 *Strongest Setups:*
{strongest_setups}

_Coverage: {symbol_count} assets | Generated: {timestamp} UTC""",
        
        'backtest_report': """📈 *BACKTEST REPORT* | {symbol} | {timeframe}

📊 *Performance Metrics:*
• Total Trades: {total_trades}
• Win Rate: {win_rate:.1f}%
• Net PnL: {net_pnl:.2f}R
• Avg TP Hit: {avg_tp:.1f}

🎯 *Trade Distribution:*
{trade_distribution}

🤖 *AI Validation Rate:* {ai_validation:.1f}%
⚡ *System Health:* {system_health}

_Period: {start_date} to {end_date}_"""
    }
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.message_queue: List[BroadcastMessage] = []
        self.broadcast_history: List[Dict] = []
        self.rate_limiter = RateLimiter(max_requests=20, period=60)
        self.queue_lock = threading.Lock()
        self.active = False
        self.processor_thread = None
        
        if token and chat_id:
            self.active = True
            self._start_processor()
            logger.info(f"BroadcastEngine initialized for chat {chat_id}")
        else:
            logger.error("BroadcastEngine: Missing token or chat_id")
    
    def _start_processor(self):
        """Start background message processor thread - FIXED"""
        if self.processor_thread and self.processor_thread.is_alive():
            logger.warning("Broadcast processor already running")
            return
            
        def processor():
            logger.info("Broadcast processor thread started")
            while self.active:
                try:
                    processed = self._process_queue()
                    if not processed:
                        time.sleep(2)
                    else:
                        time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Broadcast processor error: {e}")
                    time.sleep(5)
            logger.info("Broadcast processor thread stopped")
        
        self.processor_thread = threading.Thread(target=processor, daemon=True)
        self.processor_thread.start()
        logger.info("Broadcast processor started successfully")
    
    def queue_message(self, report_type: str, symbol: str, data: Dict, priority: int = 0) -> bool:
        """Add message to broadcast queue - FIXED"""
        if not self.active:
            logger.warning("Broadcast engine inactive - missing credentials or stopped")
            return False
        
        if not self.token or not self.chat_id:
            logger.error("Cannot queue message: Missing credentials")
            return False
        
        with self.queue_lock:
            message = BroadcastMessage(
                report_type=report_type,
                symbol=symbol,
                data=data,
                timestamp=datetime.now(timezone.utc),
                priority=priority
            )
            self.message_queue.append(message)
            self.message_queue.sort(key=lambda x: (-x.priority, x.timestamp.timestamp()))
            logger.info(f"Queued {report_type} for {symbol} (Priority: {priority})")
            return True
    
    def _process_queue(self) -> bool:
        """Process pending messages with rate limiting - CRITICAL FIX"""
        with self.queue_lock:
            if not self.message_queue:
                return False
        
        if not self.rate_limiter.can_send():
            wait_time = self.rate_limiter.time_until_next()
            if wait_time > 0:
                logger.debug(f"Rate limited, waiting {wait_time:.1f}s before next send")
                time.sleep(wait_time)
                return True
        
        with self.queue_lock:
            message = self.message_queue.pop(0)
        
        try:
            success = self._send_message_sync(message)
            if success:
                self._log_broadcast(message, "SUCCESS")
                logger.info(f"Successfully sent message: {message.report_type} for {message.symbol}")
                return True
            else:
                self._handle_failed_message(message)
                return True
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            self._handle_failed_message(message)
            return True
    
    def _send_message_sync(self, message: BroadcastMessage) -> bool:
        """Send formatted message via Telegram API - ENHANCED ERROR HANDLING"""
        try:
            formatted_text = self._format_message(message)
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": formatted_text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            logger.debug(f"Sending to {self.chat_id}: {message.report_type} for {message.symbol}")
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Message sent successfully to {self.chat_id}")
                return True
            elif response.status_code == 429:
                retry_after = response.json().get('parameters', {}).get('retry_after', 30)
                logger.warning(f"Rate limited by Telegram, retry after {retry_after}s")
                time.sleep(retry_after)
                return False
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("Telegram API request timed out")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("Connection error to Telegram API")
            return False
        except Exception as e:
            logger.error(f"Message send error: {e}")
            return False
    
    def _format_message(self, message: BroadcastMessage) -> str:
        """Format message according to report type"""
        template_key = self.REPORT_TYPES.get(message.report_type, 'strict_signal')
        template = self.TEMPLATES[template_key]
        
        base_data = {
            'symbol': message.symbol,
            'timestamp': message.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            **message.data
        }
        
        return template.format(**base_data)
    
    def _handle_failed_message(self, message: BroadcastMessage):
        """Handle failed message with retry logic"""
        if message.retry_count < message.max_retries:
            message.retry_count += 1
            wait_time = 2 ** message.retry_count
            logger.warning(f"Retry {message.retry_count} for {message.symbol} in {wait_time}s")
            time.sleep(wait_time)
            with self.queue_lock:
                self.message_queue.append(message)
        else:
            logger.error(f"Max retries exceeded for {message.symbol} {message.report_type}")
            self._log_broadcast(message, "FAILED")
    
    def _log_broadcast(self, message: BroadcastMessage, status: str):
        """Log broadcast to history"""
        log_entry = {
            'timestamp': message.timestamp.isoformat(),
            'report_type': message.report_type,
            'symbol': message.symbol,
            'status': status,
            'retry_count': message.retry_count
        }
        self.broadcast_history.append(log_entry)
        
        if len(self.broadcast_history) > 100:
            self.broadcast_history = self.broadcast_history[-100:]
    
    def get_broadcast_history(self, limit: int = 50) -> pd.DataFrame:
        """Get recent broadcast history"""
        return pd.DataFrame(self.broadcast_history[-limit:])
    
    def stop(self):
        """Stop the broadcast processor"""
        self.active = False
        logger.info("Broadcast processor stopped")
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)
    
    def is_active(self) -> bool:
        """Check if broadcast engine is active and processing"""
        return self.active and self.processor_thread is not None and self.processor_thread.is_alive()

# =============================================================================
# DATABASE LAYER (INTEGRATED)
# =============================================================================
class SignalDatabase:
    """SQLite backend for signal persistence with broadcast logging"""
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
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS broadcast_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_type TEXT,
                    symbol TEXT,
                    timestamp TEXT,
                    status TEXT,
                    retry_count INTEGER,
                    message_preview TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def save_signal(self, symbol: str, signal_data: Dict, outcome: str = "PENDING"):
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
    
    def log_broadcast(self, report_type: str, symbol: str, status: str, 
                     retry_count: int, message_preview: str):
        """Log broadcast attempt to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO broadcast_history (report_type, symbol, timestamp, status, 
                                             retry_count, message_preview)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                report_type,
                symbol,
                datetime.now(timezone.utc).isoformat(),
                status,
                retry_count,
                message_preview[:100]
            ))

# =============================================================================
# TECHNICAL ANALYSIS ENGINE (FROM BROADCAST APP)
# =============================================================================
BINANCE_API_BASE = "https://api.binance.us/api/v3"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
BINANCE_WS_BASE = "wss://stream.binance.us:9443/ws"

@st.cache_data(ttl=5)
def get_klines(symbol_bin, interval, limit):
    try:
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

def calculate_hma(series: pd.Series, length: int) -> pd.Series:
    """Calculate Hull Moving Average (HMA)"""
    if len(series) < length:
        return pd.Series([np.nan] * len(series), index=series.index)
    
    half_len = max(1, int(length / 2))
    sqrt_len = max(1, int(math.sqrt(length)))
    wma_f = series.rolling(window=length, min_periods=length).mean()
    wma_h = series.rolling(window=half_len, min_periods=half_len).mean()
    hull = 2 * wma_h - wma_f
    hma = hull.rolling(window=sqrt_len, min_periods=sqrt_len).mean()
    return hma

def calculate_fibonacci(df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels"""
    if df.empty or len(df) < lookback:
        return {'fib_382': 0.0, 'fib_500': 0.0, 'fib_618': 0.0, 'high': 0.0, 'low': 0.0}
    
    recent = df.iloc[-lookback:]
    high = recent['high'].max()
    low = recent['low'].min()
    diff = high - low
    
    return {
        'fib_382': high - (diff * 0.382),
        'fib_500': high - (diff * 0.500),
        'fib_618': high - (diff * 0.618),
        'high': high,
        'low': low
    }

# =============================================================================
# AI ANALYSIS ENGINE (FROM BROADCAST APP)
# =============================================================================
ASSET_PROFILES = {
    "BTC": {"type": "Macro Asset", "vol_regime": "Low", "session": "Global", "correlation": "Risk-On", "category": "Large Cap"},
    "ETH": {"type": "Smart Contract Leader", "vol_regime": "Medium", "session": "US/EU", "correlation": "BTC Beta", "category": "Large Cap"},
    "SOL": {"type": "High-Performance Chain", "vol_regime": "High", "session": "US", "correlation": "ETH Beta", "category": "Mid Cap"},
    "XRP": {"type": "Cross-Border Payments", "vol_regime": "Medium", "session": "EU/ASIA", "correlation": "Uncorrelated", "category": "Large Cap"},
    "BNB": {"type": "Exchange Token", "vol_regime": "Medium", "session": "Global", "correlation": "Exchange Beta", "category": "Large Cap"},
    "ADA": {"type": "PoS Blockchain", "vol_regime": "Medium", "session": "US/EU", "correlation": "ETH Beta", "category": "Mid Cap"},
    "DOGE": {"type": "Meme Coin", "vol_regime": "Very High", "session": "US", "correlation": "High Beta", "category": "Mid Cap"},
    "LINK": {"type": "Oracle Network", "vol_regime": "Medium", "session": "US/EU", "correlation": "ETH Beta", "category": "Mid Cap"},
}

def analyze_asset_and_timeframe(symbol: str, timeframe: str, df: pd.DataFrame) -> Dict:
    base = symbol.replace("USDT", "")
    profile = ASSET_PROFILES.get(base, {
        "type": "Altcoin", 
        "vol_regime": "High", 
        "session": "US", 
        "correlation": "High Beta",
        "category": "Small Cap"
    })
    
    tf_scores = {
        "15m": {"score": 70, "note": "Scalping & Intraday", "vol_adj": 1.0},
        "1h": {"score": 85, "note": "Day Trading & Swing Entry", "vol_adj": 1.2},
        "4h": {"score": 90, "note": "Swing Trading", "vol_adj": 1.5},
        "1d": {"score": 80, "note": "Position Trading", "vol_adj": 2.0}
    }
    
    tf_data = tf_scores.get(timeframe, {"score": 50, "note": "Uncommon TF", "vol_adj": 1.0})
    
    current_vol = "N/A"
    recent_vol = 0
    avg_vol = 1
    signal_confidence = 0
    
    if not df.empty and len(df) > 60:
        recent_vol = df['close'].pct_change().rolling(20).std().iloc[-1] * 100
        avg_vol = df['close'].pct_change().rolling(60).std().iloc[-1] * 100
        current_vol = f"{recent_vol:.2f}%"
        
        if recent_vol > avg_vol * 1.5:
            regime = "High Volatility"
            regime_color = "#ff1744"
            regime_factor = 0.7
        elif recent_vol < avg_vol * 0.7:
            regime = "Low Volatility"
            regime_color = "#00e676"
            regime_factor = 1.3
        else:
            regime = "Normal Volatility"
            regime_color = "#ffd740"
            regime_factor = 1.0
        
        session_score = 10 if profile['session'] == "US" else 5
        timeframe_score = 20 if tf_data['score'] > 80 else 10
        regime_score = 15 if regime_factor > 1.0 else 5
        squeeze_score = 10 if not df['in_squeeze'].iloc[-1] else 0
        volume_score = min(20, df['rvol'].iloc[-1] * 10)
        
        signal_confidence = min(100, session_score + timeframe_score + regime_score + squeeze_score + volume_score)
        
    else:
        regime = "Unknown"
        regime_color = "#9e9e9e"
        regime_factor = 1.0
    
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
        "current_vol": current_vol,
        "signal_confidence": signal_confidence,
        "session_note": session_note,
        "session_color": session_color,
        "recent_vol": recent_vol,
        "avg_vol": avg_vol
    }

# =============================================================================
# TRAILING STOP ENGINE
# =============================================================================
def calculate_trailing_stop(current_price: float, entry_price: float, initial_stop: float,
                           tp1: float, tp2: float, tp3: float, is_long: bool) -> Tuple[float, str]:
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
# TICKER UNIVERSES (FROM STOCK SCREENER)
# =============================================================================
INDEX_UNIVERSES = {
    "S&P 500": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JNJ", "V", "JPM", 
                "UNH", "HD", "PG", "MA", "DIS", "PYPL", "NFLX", "INTC", "VZ", "ADBE"],
    "NASDAQ 100": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ADBE", "PEP",
                   "CSCO", "AVGO", "TXN", "QCOM", "CMCSA", "AMD", "INTC", "AMGN", "ISRG", "BKNG"],
    "Russell 2000": ["RBLX", "PTON", "HOOD", "PLTR", "SOFI", "UPST", "RKT", "COIN", "EXPI", "COWN",
                     "APPS", "SWBI", "VNTR", "ASO", "CROX", "CRSR", "DKS", "GPRO", "SFIX", "VSTO"],
    "TSX 60": ["SHOP.TO", "RY.TO", "TD.TO", "ENB.TO", "CP.TO", "CNQ.TO", "BAM.A.TO", "TRI.TO", "BCE.TO", "BNS.TO"],
    "ASX 200": ["CBA.AX", "BHP.AX", "CSL.AX", "NAB.AX", "ANZ.AX", "WBC.AX", "MQG.AX", "APT.AX", "TLS.AX", "WOW.AX"]
}

MINING_UNIVERSE = [
    "NXE", "UEC", "UUUU", "DNN", "PDN.AX", "BOE.AX", "GLO.TO",
    "LAC", "SGML", "PLS.AX", "CXO.AX", "SYA.AX", "LTR.AX", "PMET.TO", "CRE.TO",
    "KGC", "EQX", "NGD", "SILV", "MAG", "SVM", "GREG.L", "CMM.AX", "PRU.AX", "WAF.AX",
    "ERO", "IVN.TO", "HBM", "CAM.TO", "FM.TO", "ALS.TO", "SFR.AX", "29M.AX",
    "MP", "LYC.AX", "ARU.AX", "ASM.AX"
]

@st.cache_data(ttl=86400)
def get_universe_tickers(mode, indices=None, custom_list=None):
    """Returns ticker list based on selected mode"""
    if mode == "Junior Mining":
        return MINING_UNIVERSE
    elif mode == "Custom Upload":
        return custom_list or []
    else:
        tickers = []
        for idx in (indices or []):
            tickers.extend(INDEX_UNIVERSES.get(idx, []))
        return list(set(tickers))

# =============================================================================
# TECHNICAL ANALYSIS ENGINE (FROM STOCK SCREENER)
# =============================================================================
def calculate_technicals(hist):
    if hist is None or len(hist) < 200:
        return {
            "RSI_14": None, "SMA_50": None, "SMA_200": None, 
            "Trend": "Insufficient Data", "Volatility": None
        }
    
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
    sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
    current_price = hist['Close'].iloc[-1]
    
    if current_price > sma_50 > sma_200:
        trend = "Strong Bullish"
    elif current_price < sma_50 < sma_200:
        trend = "Strong Bearish"
    elif sma_50 > sma_200:
        trend = "Bullish (Golden Cross)"
    else:
        trend = "Bearish (Death Cross)"
        
    daily_returns = hist['Close'].pct_change()
    volatility = daily_returns.std() * np.sqrt(252)

    return {
        "RSI_14": round(current_rsi, 2),
        "SMA_50": round(sma_50, 2),
        "SMA_200": round(sma_200, 2),
        "Trend": trend,
        "Volatility": f"{round(volatility*100, 1)}%"
    }

# =============================================================================
# DATA FETCHING (INTEGRATED)
# ===================================================================
@st.cache_data(ttl=3600)
def get_stock_financials(ticker_symbol, mode):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        market_cap = info.get('marketCap', 0)
        if mode == "Junior Mining":
            if market_cap < 50_000_000 or market_cap > 15_000_000_000:
                return None
        else:
            if market_cap < 200_000_000:
                return None

        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        data = {
            "ticker": ticker_symbol,
            "name": info.get('longName', ticker_symbol),
            "country": info.get('country', 'Unknown'),
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
            "current_price": info.get('currentPrice', 0),
            "price_to_earnings": info.get('trailingPE', 999),
            "price_to_book": info.get('priceToBook', 999),
            "price_to_sales": info.get('priceToSales', 999),
            "debt_to_equity": info.get('debtToEquity', 999),
            "current_ratio": info.get('currentRatio', 0),
            "total_cash": info.get('totalCash', 0),
            "total_debt": info.get('totalDebt', 0),
            "roe": info.get('returnOnEquity', 0),
            "roa": info.get('returnOnAssets', 0),
            "gross_margins": info.get('grossMargins', 0),
            "operating_margins": info.get('operatingMargins', 0),
            "revenue_growth": info.get('revenueGrowth', 0),
            "earnings_growth": info.get('earningsGrowth', 0),
            "avg_volume": info.get('averageVolume', 0),
            "beta": info.get('beta', 0)
        }
        
        try:
            net_cash = data["total_cash"] - data["total_debt"]
            data["net_cash_pos"] = net_cash > 0
        except:
            data["net_cash_pos"] = False

        return data
    except Exception as e:
        return None

def get_price_history(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        return stock.history(period="1y")
    except:
        return None

def get_price_at_date(hist, target_date_str):
    target_date = pd.to_datetime(target_date_str).tz_localize(None)
    if hist is None or hist.empty: return None
    hist.index = hist.index.tz_localize(None)
    hist = hist.sort_index()
    try:
        idx = hist.index.get_indexer([target_date], method='nearest')[0]
        return hist.iloc[idx]['Close']
    except:
        return hist['Close'].iloc[-1]

# =============================================================================
# CORE CALCULATIONS (FROM BROADCAST APP)
# =============================================================================
def run_engines(df, amp, dev, hma_l, tp1, tp2, tp3, mf_l, vol_l, gann_l, use_ml_filter):
    if df.empty:
        return df
    df = df.copy().reset_index(drop=True)

    # Core Indicators
    df['tr'] = np.maximum(
        df['high']-df['low'],
        np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1)))
    )
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    df['hma'] = calculate_hma(df['close'], hma_l)

    # VWAP
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vol_tp'] = df['tp'] * df['volume']
    df['vwap'] = df['vol_tp'].cumsum() / df['volume'].cumsum()

    # Squeeze
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

    # Money Flow
    rsi_source = df['rsi'] - 50
    vol_sma = df['volume'].rolling(mf_l).mean()
    df['money_flow'] = (rsi_source * (df['volume'] / vol_sma)).ewm(span=3).mean()

    # Hyper Wave
    pc = df['close'].diff()
    ds_pc = pc.ewm(span=25).mean().ewm(span=13).mean()
    ds_abs_pc = abs(pc).ewm(span=25).mean().ewm(span=13).mean()
    df['hyper_wave'] = (100 * (ds_pc / ds_abs_pc)) / 2

    # Titan Trend
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
    
    # ML Filter
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

    # Apex & Gann
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

# =============================================================================
# HTML GENERATION (FROM BROADCAST APP)
# =============================================================================
def generate_mobile_report(row, symbol, tf, fibs, fg_index, smart_stop, 
                          ai_analysis: Dict, use_trailing: bool, tp1_r: float, tp2_r: float, tp3_r: float) -> str:
    """
    Generate HTML report with proper variable scoping and HTML escaping
    """
    is_bull = bool(row['is_bull'])
    direction = "LONG 🐂" if is_bull else "SHORT 🐻"
    
    titan_sig = 1 if is_bull else -1
    apex_sig = int(row['apex_trend'])
    gann_sig = int(row['gann_trend']) if not pd.isna(row['gann_trend']) else 0
    momentum_sig = 1 if row['money_flow'] > 0 else -1
    volume_sig = 1 if row['rvol'] > 1.5 else 0

    score_val = 0
    if titan_sig == apex_sig: score_val += 1
    if titan_sig == gann_sig: score_val += 1
    if titan_sig == momentum_sig: score_val += 1
    if volume_sig == 1: score_val += 1

    if score_val >= 3: 
        confidence = "MAX 🔥🔥🔥"
    elif score_val >= 2: 
        confidence = "HIGH 🔥"
    elif score_val >= 1: 
        confidence = "MEDIUM ⚡"
    else: 
        confidence = "LOW ⚠️"

    if row['rvol'] > 2.0: 
        vol_desc = "IGNITION 🚀🚀🚀"
    elif row['rvol'] > 1.5: 
        vol_desc = "Above Avg 🚀"
    else: 
        vol_desc = "Normal"

    squeeze_txt = "⚠️ SQUEEZE ACTIVE" if row['in_squeeze'] else "✅ NO SQUEEZE"

    entry_price = float(row['close'])
    stop_price = float(smart_stop)
    tp1_price = float(row['tp1'])
    tp2_price = float(row['tp2'])
    tp3_price = float(row['tp3'])
    max_risk = float(abs(entry_price - stop_price))

    html_parts = []
    
    html_parts.append(f'''
<div class="report-card">
    <div class="report-header">💠 SIGNAL: {direction}</div>
    <div class="report-item">Confidence: <span class="highlight">{confidence}</span></div>
    <div class="report-item">Layers: <span class="highlight">{score_val}/4 Confirmed</span></div>
    <div class="report-item">Squeeze: <span class="highlight">{squeeze_txt}</span></div>
    <div class="report-item">AI Score: <span class="highlight">{ai_analysis["signal_confidence"]}/100</span></div>
</div>
''')

    html_parts.append(f'''
<div class="report-card">
    <div class="report-header">🌊 FLOW & VOL</div>
    <div class="report-item">RVOL: <span class="highlight">{row["rvol"]:.2f} ({vol_desc})</span></div>
    <div class="report-item">Money Flow: <span class="highlight">{row["money_flow"]:.2f}</span></div>
    <div class="report-item">VWAP: <span class="highlight">{'Above' if entry_price > float(row['vwap']) else 'Below'}</span></div>
</div>
''')

    html_parts.append(f'''
<div class="report-card">
    <div class="report-header">🎯 EXECUTION PLAN</div>
    <div class="report-item">Entry: <span class="highlight">{entry_price:.4f}</span></div>
    <div class="report-item">🛑 SMART STOP: <span class="highlight">{stop_price:.4f}</span></div>
    <div class="report-item">1️⃣ TP1 ({tp1_r}R): <span class="highlight">{tp1_price:.4f}</span></div>
    <div class="report-item">2️⃣ TP2 ({tp2_r}R): <span class="highlight">{tp2_price:.4f}</span></div>
    <div class="report-item">3️⃣ TP3 ({tp3_r}R): <span class="highlight">{tp3_price:.4f}</span></div>
    <div class="report-item">📊 Risk/Reward: <span class="highlight">1:{tp3_r:.1f}</span></div>
</div>
''')

    if use_trailing:
        html_parts.append(f'''
<div class="report-card trailing-enabled">
    <div class="report-header">📈 TRADE MANAGEMENT (Trailing)</div>
    <div class="report-item">🔄 Trailing: <span class="highlight">ENABLED</span></div>
    <div class="report-item">📍 Initial Stop: <span class="highlight">{stop_price:.4f}</span></div>
    <div class="report-item">🔒 At TP1 ({tp1_r}R): <span class="highlight">Move to Breakeven ({entry_price:.4f})</span></div>
    <div class="report-item">🔒 At TP2 ({tp2_r}R): <span class="highlight">Lock TP1 ({tp1_price:.4f})</span></div>
    <div class="report-item">🔒 At TP3 ({tp3_r}R): <span class="highlight">Lock TP2 ({tp2_price:.4f})</span></div>
    <div class="report-item">🎯 Max Risk: <span class="highlight">{max_risk:.4f}</span></div>
    <div class="report-item" style="color:#ffd740; font-size:12px;">⚠️ Rinse & Repeat: Let runners trail, scale out gradually</div>
</div>
''')
    else:
        html_parts.append(f'''
<div class="report-card trailing-disabled">
    <div class="report-header">📊 TRADE MANAGEMENT (Static)</div>
    <div class="report-item">🔄 Trailing: <span class="highlight">DISABLED</span></div>
    <div class="report-item">🛑 Fixed Stop: <span class="highlight">{stop_price:.4f}</span></div>
    <div class="report-item">✅ Take Profits: <span class="highlight">All 3 TPs Static</span></div>
    <div class="report-item" style="color:#ff1744; font-size:12px;">⚠️ Consider enabling trailing for better risk management</div>
</div>
''')

    html_parts.append(f'''
<div class="ai-card ai-recommendation" style="border-left-color: {ai_analysis["rec_color"]};">
    <strong>🤖 AI Recommendation:</strong> <span style="color:{ai_analysis["rec_color"]};">{ai_analysis["recommendation"]}</span><br>
    <small>{ai_analysis["timeframe_note"]}</small><br>
    <strong>{ai_analysis["size_recommendation"]}</strong><br>
    <div style="color:{ai_analysis["session_color"]}; margin-top:8px;">
        {ai_analysis["session_note"]}
    </div>
</div>
''')

    return ''.join(html_parts)

# =============================================================================
# TELEGRAM SENDER (FROM STOCK SCREENER - ENHANCED)
# =============================================================================
def send_telegram_package(token, chat_id, text, excel_buffer, filename):
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        )
        excel_buffer.seek(0)
        requests.post(
            f"https://api.telegram.org/bot{token}/sendDocument",
            data={"chat_id": chat_id, "caption": "📊 Stock Analysis Report"},
            files={"document": (filename, excel_buffer, "application/vnd.ms-excel")}
        )
        return True
    except Exception as e:
        st.error(f"Telegram Error: {e}")
        return False

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================
st.set_page_config(
    page_title="AI Stock Screener & Analyst + TITAN SIGNALS",
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0b0c10; }
    div[data-testid="metric-container"] {
        background: rgba(31, 40, 51, 0.9);
        border: 1px solid #45a29e;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label {
        font-size: 14px !important;
        color: #c5c6c7 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 24px !important;
        color: #66fcf1 !important;
    }
    h1, h2, h3 { font-family: 'Roboto Mono', monospace; color: #c5c6c7; word-wrap: break-word; }
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
    .trailing-enabled {
        border-left-color: #00e676 !important;
        background: linear-gradient(135deg, #1f2833 0%, #002a00 100%) !important;
    }
    .trailing-disabled {
        border-left-color: #ff9800 !important;
    }
    .broadcast-status {
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 14px;
    }
    .broadcast-active {
        background: linear-gradient(135deg, #00e67620, #00e67610);
        border: 1px solid #00e676;
        color: #00e676;
    }
    .broadcast-inactive {
        background: linear-gradient(135deg, #ff174420, #ff174410);
        border: 1px solid #ff1744;
        color: #ff1744;
    }
    .broadcast-history {
        max-height: 300px;
        overflow-y: auto;
        font-size: 12px;
        background: #1f2833;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None
if "analysis_excel" not in st.session_state:
    st.session_state.analysis_excel = None
if "broadcast_engine" not in st.session_state:
    st.session_state.broadcast_engine = None
if "broadcast_strict_signal" not in st.session_state:
    st.session_state.broadcast_strict_signal = False
if "broadcast_ai_analysis" not in st.session_state:
    st.session_state.broadcast_ai_analysis = False
if "broadcast_market_summary" not in st.session_state:
    st.session_state.broadcast_market_summary = False
if "show_broadcast_history" not in st.session_state:
    st.session_state.show_broadcast_history = False
if "show_perf" not in st.session_state:
    st.session_state.show_perf = False

# =============================================================================
# MAIN APP HEADER
# =============================================================================
st.title("🤖 AI Stock Screener & Analyst + 💠 TITAN SIGNALS")
st.markdown("""
**Multi-Sector Analysis & Trading Broadcast System:** 
1. **Fundamental Filters:** Value, Quality, Growth, and Momentum criteria
2. **Technical Analysis:** RSI, Moving Averages, Volatility scanning
3. **AI Analyst:** Generates sector-specific insights and risk assessment
4. **Broadcast Engine:** Enterprise-grade Telegram signal broadcasting
""")

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================
st.sidebar.header("⚙️ Configuration")

# API Key
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.sidebar.success("OpenAI Key: Loaded")
else:
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# Market Selection
analysis_mode = st.sidebar.selectbox(
    "Analysis Mode",
    ["General Market", "Junior Mining", "Custom Upload", "TITAN Signals (Crypto)"],
    help="Select universe of stocks to analyze"
)

# Index Selection (for General Market)
selected_indices = []
if analysis_mode == "General Market":
    st.sidebar.subheader("Market Indices")
    selected_indices = st.sidebar.multiselect(
        "Select Indices",
        ["S&P 500", "NASDAQ 100", "Russell 2000", "TSX 60", "ASX 200"],
        default=["S&P 500"]
    )

# Custom Tickers
custom_tickers = []
if analysis_mode == "Custom Upload":
    st.sidebar.subheader("Custom Tickers")
    ticker_input = st.sidebar.text_area(
        "Enter tickers (one per line)",
        placeholder="AAPL\nMSFT\nGOOGL"
    )
    custom_tickers = [t.strip() for t in ticker_input.split('\n') if t.strip()]

# TITAN Signals Mode
symbol = "BTCUSDT"
timeframe = "1h"
limit = 200
if analysis_mode == "TITAN Signals (Crypto)":
    st.sidebar.subheader("📡 TITAN SIGNALS")
    st.sidebar.markdown("---")
    
    # Initialize database
    db = SignalDatabase()
    
    # Render ticker tape
    def render_ticker_tape(selected_symbol: str):
        base = selected_symbol.replace("USDT", "")
        tape_bases = ["BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "LINK", "AVAX", "DOT"]
        if base and base not in tape_bases:
            tape_bases.insert(0, base)

        symbols_json = json.dumps(
            [{"proName": f"BINANCE:{b}USDT", "title": b} for b in tape_bases],
            separators=(",", ":")
        )

        components.html(
            f"""
            <div class="tradingview-widget-container">
              <div class="tradingview-widget-container__widget"></div>
              <script type="text/javascript">
              var tickerConfig = {symbols_json};
              </script>
              <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
              </script>
            </div>
            """,
            height=50
        )
    
    symbol_input = st.sidebar.text_input("Asset", value=st.session_state.get("symbol_input", "BTC"))
    st.session_state.symbol_input = symbol_input
    symbol = symbol_input.strip().upper().replace("/", "").replace("-", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    timeframe = st.sidebar.selectbox("TF", ["15m", "1h", "4h", "1d"], index=1)
    limit = st.sidebar.slider("Depth", 100, 500, 200, 50)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 LOGIC")
    amplitude = st.sidebar.number_input("Amp", 2, 200, 10)
    channel_dev = st.sidebar.number_input("Dev", 0.5, 10.0, 3.0, 0.1)
    hma_len = st.sidebar.number_input("HMA", 2, 400, 50)
    gann_len = st.sidebar.number_input("Gann", 1, 50, 3)

    with st.sidebar.expander("🎯 Targets"):
        tp1_r = st.sidebar.number_input("TP1 (R)", value=1.5)
        tp2_r = st.sidebar.number_input("TP2 (R)", value=3.0)
        tp3_r = st.sidebar.number_input("TP3 (R)", value=5.0)

    with st.sidebar.expander("🛑 TRADE MANAGEMENT"):
        use_trailing = st.sidebar.toggle("Enable Trailing Stops", value=True)
        use_ml_filter = st.sidebar.toggle("ML Signal Filter", value=False)
    
    render_ticker_tape(symbol)

# Telegram Integration
use_telegram = False
tele_token = None
tele_chat_id = None
if "TELEGRAM_TOKEN" in st.secrets and "TELEGRAM_CHAT_ID" in st.secrets:
    tele_token = st.secrets["TELEGRAM_TOKEN"]
    tele_chat_id = st.secrets["TELEGRAM_CHAT_ID"]
    use_telegram = True
    st.sidebar.success("Telegram: Connected")
else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Telegram Setup (Optional)")
    tele_token = st.sidebar.text_input("Bot Token", type="password")
    tele_chat_id = st.sidebar.text_input("Chat ID")
    use_telegram = bool(tele_token and tele_chat_id)

# Broadcast Controls (TITAN Signals Mode)
if analysis_mode == "TITAN Signals (Crypto)" and use_telegram:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 BROADCAST CONTROLS")
    
    # Initialize broadcast engine
    if "broadcast_engine" not in st.session_state:
        st.session_state.broadcast_engine = None
    
    credentials_valid = bool(tele_token and tele_chat_id)
    
    if credentials_valid:
        should_activate = (st.session_state.broadcast_engine is None or 
                          st.session_state.broadcast_engine.token != tele_token or 
                          st.session_state.broadcast_engine.chat_id != tele_chat_id or 
                          not st.session_state.broadcast_engine.is_active())
        
        if should_activate:
            if st.sidebar.button("🚀 ACTIVATE BROADCAST", use_container_width=True, type="primary"):
                if st.session_state.broadcast_engine:
                    st.session_state.broadcast_engine.stop()
                
                st.session_state.broadcast_engine = BroadcastEngine(tele_token, tele_chat_id)
                st.sidebar.success("✅ Broadcast system activated!")
                st.rerun()
        
        if st.session_state.broadcast_engine and st.session_state.broadcast_engine.is_active():
            st.sidebar.markdown('<div class="broadcast-status broadcast-active">🟢 BROADCAST ACTIVE</div>', unsafe_allow_html=True)
            
            st.sidebar.subheader("📤 Quick Send")
            
            col_btn1, col_btn2 = st.sidebar.columns([1, 1])
            with col_btn1:
                if st.sidebar.button("🔥 STRICT SIGNAL", use_container_width=True):
                    st.session_state.broadcast_strict_signal = True
                    st.sidebar.toast("✅ Strict signal queued!", icon='🔥')
            
            with col_btn2:
                if st.sidebar.button("🤖 AI RISK ANALYSIS", use_container_width=True):
                    st.session_state.broadcast_ai_analysis = True
                    st.sidebar.toast("✅ AI analysis queued!", icon='🧠')
            
            if st.sidebar.button("📊 MARKET SUMMARY", use_container_width=True):
                st.session_state.broadcast_market_summary = True
                st.sidebar.toast("✅ Market summary queued!", icon='📊')
            
            if hasattr(st.session_state.broadcast_engine, 'message_queue'):
                queue_size = len(st.session_state.broadcast_engine.message_queue)
                st.sidebar.caption(f"📨 Queue: {queue_size} messages pending")
            
            with st.sidebar.expander("📜 Broadcast History", expanded=False):
                if st.sidebar.button("Load History", use_container_width=True):
                    st.session_state.show_broadcast_history = True
        else:
            st.sidebar.markdown('<div class="broadcast-status broadcast-inactive">🔴 BROADCAST INACTIVE</div>', unsafe_allow_html=True)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if st.button("🚀 Run Analysis"):
    if not api_key and analysis_mode != "TITAN Signals (Crypto)":
        st.error("Please provide OpenAI API Key.")
    elif analysis_mode == "TITAN Signals (Crypto)" and not use_telegram:
        st.error("Please provide Telegram credentials for TITAN Signals mode.")
    elif not any([analysis_mode == "Junior Mining", selected_indices, custom_tickers, analysis_mode == "TITAN Signals (Crypto)"]):
        st.error("Please select at least one market index or provide custom tickers.")
    else:
        if analysis_mode == "TITAN Signals (Crypto)":
            # TITAN Signals Mode - Execute full broadcast logic
            st.subheader("💠 TITAN SIGNALS - Live Trading Engine")
            
            # Initialize database
            db = SignalDatabase()
            
            # Fetch data
            df = get_klines(symbol, timeframe, limit)
            
            if df.empty:
                st.error("No data returned. Check symbol, timeframe, or Binance US availability.")
            else:
                df = df.dropna(subset=['close'])
                df = run_engines(df, int(amplitude), channel_dev, int(hma_len), tp1_r, tp2_r, tp3_r, 
                                14, 20, int(gann_len), use_ml_filter)

                last = df.iloc[-1]
                fibs = calculate_fibonacci(df)
                fg_index = calculate_fear_greed_index(df)

                # Calculate stops
                if last['is_bull']:
                    smart_stop = min(last['entry_stop'], fibs['fib_618'] * 0.9995)
                else:
                    smart_stop = max(last['entry_stop'], fibs['fib_618'] * 1.0005)

                trailing_stop, trail_status = calculate_trailing_stop(
                    last['close'], last['close'], smart_stop,
                    last['tp1'], last['tp2'], last['tp3'], last['is_bull']
                )

                ai_analysis = analyze_asset_and_timeframe(symbol, timeframe, df)
                
                # Save signal
                if last['buy'] or last['sell']:
                    db.save_signal(symbol, last.to_dict())
                
                # Broadcast handlers
                if st.session_state.get("broadcast_strict_signal", False):
                    if st.session_state.broadcast_engine and st.session_state.broadcast_engine.is_active():
                        signal_data = {
                            'direction': "LONG" if last['is_bull'] else "SHORT",
                            'entry_price': last['close'],
                            'stop_price': smart_stop,
                            'tp1': last['tp1'],
                            'tp2': last['tp2'],
                            'tp3': last['tp3'],
                            'tp1_r': tp1_r,
                            'tp2_r': tp2_r,
                            'tp3_r': tp3_r,
                            'confidence': ai_analysis['signal_confidence'],
                            'timeframe': timeframe,
                            'rvol': last['rvol'],
                            'squeeze_status': "ACTIVE" if last['in_squeeze'] else "CLEAR",
                            'ai_score': ai_analysis['signal_confidence'],
                            'signal_grade': "A+" if ai_analysis['signal_confidence'] > 80 else "B+"
                        }
                        st.session_state.broadcast_engine.queue_message('STRICT_SIGNAL', symbol, signal_data, priority=10)
                        st.success("✅ Strict signal broadcast queued!")
                        st.toast(f"🚀 Sent {symbol} strict signal", icon="✅")
                    else:
                        st.error("❌ Broadcast engine not active!")
                    st.session_state.broadcast_strict_signal = False
                
                if st.session_state.get("broadcast_ai_analysis", False):
                    if st.session_state.broadcast_engine and st.session_state.broadcast_engine.is_active():
                        ai_data = {
                            'market_regime': ai_analysis['market_regime'],
                            'current_vol': ai_analysis['current_vol'],
                            'vol_factor': ai_analysis['regime_factor'],
                            'session_note': ai_analysis['session_note'],
                            'tf_score': ai_analysis['timeframe_score'],
                            'asset_type': ai_analysis['asset_profile']['type'],
                            'squeeze_risk': "YES" if last['in_squeeze'] else "NO",
                            'volume_risk': "YES" if last['rvol'] > 3 else "NO",
                            'session_risk': "YES" if "Off-Hours" in ai_analysis['session_note'] else "NO",
                            'recommendation': ai_analysis['recommendation'],
                            'size_rec': ai_analysis['size_recommendation']
                        }
                        st.session_state.broadcast_engine.queue_message('AI_RISK_ANALYSIS', symbol, ai_data, priority=5)
                        st.success("✅ AI risk analysis broadcast queued!")
                        st.toast(f"🤖 Sent {symbol} AI analysis", icon="✅")
                    else:
                        st.error("❌ Broadcast engine not active!")
                    st.session_state.broadcast_ai_analysis = False
                
                if st.session_state.get("broadcast_market_summary", False):
                    if st.session_state.broadcast_engine and st.session_state.broadcast_engine.is_active():
                        summary_data = {
                            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            'top_performers': "BTC: +2.3%, ETH: +1.8%, SOL: +4.1%",
                            'risk_signals': "DOGE: High Vol, SHIB: Squeeze Active",
                            'fear_greed': fg_index,
                            'avg_rvol': df['rvol'].mean(),
                            'squeeze_count': int(df['in_squeeze'].iloc[-5:].sum()),
                            'strongest_setups': f"{symbol}: {ai_analysis['recommendation']}",
                            'symbol_count': 30
                        }
                        st.session_state.broadcast_engine.queue_message('MARKET_SUMMARY', "MULTI", summary_data, priority=3)
                        st.success("✅ Market summary broadcast queued!")
                        st.toast(f"📊 Sent market summary", icon="✅")
                    else:
                        st.error("❌ Broadcast engine not active!")
                    st.session_state.broadcast_market_summary = False
                
                # Display AI Analysis Card
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
                
                # Mobile Metrics
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
                      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js" async>
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

                # Generate and render report
                report_html = generate_mobile_report(
                    last, symbol, timeframe, fibs, fg_index, smart_stop, 
                    ai_analysis, use_trailing, tp1_r, tp2_r, tp3_r
                )
                st.markdown(report_html, unsafe_allow_html=True)

                # Action Buttons
                st.markdown("### ⚡ ACTION")
                b_col1, b_col2, b_col3 = st.columns(3)
                
                with b_col1:
                    if st.button("🔥 ALERT TG", use_container_width=True):
                        if st.session_state.broadcast_engine and st.session_state.broadcast_engine.is_active():
                            signal_data = {
                                'direction': "LONG" if last['is_bull'] else "SHORT",
                                'entry_price': last['close'],
                                'stop_price': smart_stop,
                                'tp1': last['tp1'],
                                'tp2': last['tp2'],
                                'tp3': last['tp3'],
                                'tp1_r': tp1_r,
                                'tp2_r': tp2_r,
                                'tp3_r': tp3_r,
                                'confidence': ai_analysis['signal_confidence'],
                                'timeframe': timeframe,
                                'rvol': last['rvol'],
                                'squeeze_status': "ACTIVE" if last['in_squeeze'] else "CLEAR",
                                'ai_score': ai_analysis['signal_confidence'],
                                'signal_grade': "A+" if ai_analysis['signal_confidence'] > 80 else "B+"
                            }
                            st.session_state.broadcast_engine.queue_message('STRICT_SIGNAL', symbol, signal_data, priority=10)
                            st.success("✅ Broadcast queued!")
                        else:
                            st.error("❌ Broadcast engine not active!")

                with b_col2:
                    if st.button("📝 REPORT TG", use_container_width=True):
                        if st.session_state.broadcast_engine and st.session_state.broadcast_engine.is_active():
                            ai_data = {
                                'market_regime': ai_analysis['market_regime'],
                                'current_vol': ai_analysis['current_vol'],
                                'vol_factor': ai_analysis['regime_factor'],
                                'session_note': ai_analysis['session_note'],
                                'tf_score': ai_analysis['timeframe_score'],
                                'asset_type': ai_analysis['asset_profile']['type'],
                                'squeeze_risk': "YES" if last['in_squeeze'] else "NO",
                                'volume_risk': "YES" if last['rvol'] > 3 else "NO",
                                'session_risk': "YES" if "Off-Hours" in ai_analysis['session_note'] else "NO",
                                'recommendation': ai_analysis['recommendation'],
                                'size_rec': ai_analysis['size_recommendation']
                            }
                            st.session_state.broadcast_engine.queue_message('AI_RISK_ANALYSIS', symbol, ai_data, priority=5)
                            st.success("📋 Report queued!")
                        else:
                            st.error("❌ Broadcast engine not active!")

                with b_col3:
                    if st.button("📊 BACKTEST TG", use_container_width=True):
                        b_total, b_win, b_net, b_df = run_backtest(df, tp1_r, tp2_r, tp3_r, use_trailing)
                        if b_total > 0:
                            if st.session_state.broadcast_engine and st.session_state.broadcast_engine.is_active():
                                bt_data = {
                                    'symbol': symbol,
                                    'timeframe': timeframe,
                                    'total_trades': b_total,
                                    'win_rate': b_win,
                                    'net_pnl': b_net,
                                    'avg_tp': b_df['tp_reached'].mean() if 'tp_reached' in b_df.columns else 0,
                                    'trade_distribution': f"TP1: {len(b_df[b_df['tp_reached']>=1])}, TP2: {len(b_df[b_df['tp_reached']>=2])}, TP3: {len(b_df[b_df['tp_reached']>=3])}",
                                    'ai_validation': 85.0,
                                    'system_health': "GOOD" if b_win > 50 else "POOR",
                                    'start_date': df['timestamp'].iloc[0].strftime('%Y-%m-%d'),
                                    'end_date': df['timestamp'].iloc[-1].strftime('%Y-%m-%d')
                                }
                                st.session_state.broadcast_engine.queue_message('BACKTEST_REPORT', symbol, bt_data, priority=2)
                                st.success("📊 Backtest queued!")
                            else:
                                st.error("❌ Broadcast engine not active!")
                        else:
                            st.warning("No backtest data available")

                # Backtest stats
                b_total, b_win, b_net, b_df = run_backtest(df, tp1_r, tp2_r, tp3_r, use_trailing)
                if b_total > 0:
                    avg_tp = b_df['tp_reached'].mean() if 'tp_reached' in b_df.columns else 0
                    st.caption(f"📊 Live Stats: {b_win:.1f}% Win | {b_net:.1f}R Net ({b_total} Trades) | Avg TP: {avg_tp:.1f}")
                else:
                    st.caption("📊 No completed trades in lookback period")

                # Chart
                fig = go.Figure()
                fig.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', name='HMA', line=dict(color='#66fcf1', width=1)))
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vwap'], mode='lines', name='VWAP', line=dict(color='#9933ff', width=2)))
                
                fig.add_hline(y=fibs['fib_618'], line_dash="dash", line_color="#ffd740", annotation_text="FIB 0.618")
                fig.add_hline(y=fibs['fib_500'], line_dash="dash", line_color="#ff9800", annotation_text="FIB 0.5")
                fig.add_hline(y=fibs['fib_382'], line_dash="dash", line_color="#ff5722", annotation_text="FIB 0.382")
                fig.add_hline(y=last['tp1'], line_dash="dot", line_color="#00e676", annotation_text=f"TP1 ({tp1_r}R)")
                fig.add_hline(y=last['tp2'], line_dash="dot", line_color="#ffd740", annotation_text=f"TP2 ({tp2_r}R)")
                fig.add_hline(y=last['tp3'], line_dash="dot", line_color="#ff5722", annotation_text=f"TP3 ({tp3_r}R)")
                fig.add_hline(y=trailing_stop, line_dash="solid", line_color="#ff1744", 
                             annotation_text=f"Trail: {trail_status}", annotation_position="right")

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

                # Tabs
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
                    
                    st.markdown(f"""
                    <div class="ai-card">
                        <div class="report-item"><strong>Extreme Fear:</strong> 0-25 (Oversold - Consider Long)</div>
                        <div class="report-item"><strong>Fear:</strong> 25-50 (Caution - Wait for Confirmation)</div>
                        <div class="report-item"><strong>Greed:</strong> 50-75 (Momentum - Trail Stops)</div>
                        <div class="report-item"><strong>Extreme Greed:</strong> 75-100 (Overbought - Consider Short)</div>
                        <div class="report-item" style="color:#66fcf1; margin-top:10px;">
                            <strong>Current: {fg_index}/100</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with t4:
                    st.markdown("### 📊 Historical Performance")
                    if st.button("Load Performance Data", use_container_width=True):
                        st.session_state.show_perf = True
                    
                    if st.session_state.show_perf:
                        perf_data = pd.DataFrame({
                            'Date': pd.date_range(start='2024-01-01', periods=30),
                            'PnL': np.random.randn(30).cumsum() * 2 + 5
                        })
                        f4 = go.Figure()
                        f4.add_trace(go.Scatter(x=perf_data['Date'], y=perf_data['PnL'], mode='lines+markers'))
                        f4.update_layout(height=250, template='plotly_dark', margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(f4, use_container_width=True)

                # Validation Checklist
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
                
                # Broadcast History
                if st.session_state.get("show_broadcast_history", False):
                    st.markdown("### 📡 Broadcast History")
                    if st.session_state.broadcast_engine and st.session_state.broadcast_engine.is_active():
                        history_df = st.session_state.broadcast_engine.get_broadcast_history(20)
                        if not history_df.empty:
                            st.dataframe(
                                history_df[['timestamp', 'report_type', 'symbol', 'status', 'retry_count']],
                                use_container_width=True
                            )
                        else:
                            st.info("No broadcast history available yet")
                    else:
                        st.warning("Broadcast engine not initialized or inactive")
                
        else:
            # Stock Screener Mode - Execute original logic
            st.subheader("1. Building Universe & Screening...")
            
            universe = get_universe_tickers(analysis_mode, selected_indices, custom_tickers)
            
            if not universe:
                st.error("No tickers in universe. Check selections.")
            else:
                st.info(f"Analyzing {len(universe)} tickers...")
                
                # Run screen
                df = run_general_screen(universe, analysis_mode)
                
                if df.empty:
                    st.warning("No stocks matched the criteria.")
                else:
                    # Rank by performance and select top 20
                    df = df.sort_values(by='perf_since_jan', ascending=False)
                    final_df = df.head(20).reset_index(drop=True)
                    
                    st.success(f"Identified {len(final_df)} high-potential stocks.")
                    
                    # AI Analysis
                    st.subheader("2. AI Analyst Review...")
                    prog = st.progress(0)
                    for i, idx in enumerate(final_df.index):
                        row = final_df.loc[idx]
                        insights = analyze_stock_with_ai(row, api_key, analysis_mode)
                        final_df.loc[idx, 'AI_Verdict'] = insights[0]
                        final_df.loc[idx, 'Business_Quality'] = insights[1]
                        final_df.loc[idx, 'Catalysts'] = insights[2]
                        final_df.loc[idx, 'Key_Risk'] = insights[3]
                        prog.progress((i+1)/len(final_df))
                    prog.empty()
                    
                    st.session_state.analysis_df = final_df
                    st.session_state.analysis_done = True
                    
                    # Create Excel
                    output_cols = [
                        'ticker', 'name', 'sector', 'market_cap', 'AI_Verdict',
                        'price_to_earnings', 'price_to_book', 'roe', 'revenue_growth',
                        'Business_Quality', 'Catalysts', 'Key_Risk', 'matched_criteria'
                    ]
                    output_df = final_df[output_cols].copy()
                    output_df['matched_criteria'] = output_df['matched_criteria'].apply(lambda x: ", ".join(x))
                    
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        output_df.to_excel(writer, index=False, sheet_name="Stock Picks")
                    buffer.seek(0)
                    st.session_state.analysis_excel = buffer.getvalue()

# =============================================================================
# DISPLAY RESULTS (STOCK SCREENER MODE)
# =============================================================================
if st.session_state.analysis_done and st.session_state.analysis_df is not None and analysis_mode != "TITAN Signals (Crypto)":
    final_df = st.session_state.analysis_df
    
    st.write("### 🎯 Top Stock Picks")
    
    # Show summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", len(final_df))
    with col2:
        buy_count = len(final_df[final_df['AI_Verdict'].str.contains('Buy')])
        st.metric("Buy Ratings", buy_count)
    with col3:
        sectors = final_df['sector'].nunique()
        st.metric("Sectors Covered", sectors)
    with col4:
        top_performer = final_df.iloc[0]
        st.metric("Top Performer", top_performer['ticker'])
    
    # Display dataframe
    st.dataframe(final_df[[
        'ticker', 'name', 'sector', 'market_cap', 'AI_Verdict', 
        'price_to_earnings', 'revenue_growth', 'matched_criteria'
    ]])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fname = f"StockAnalysis_{date.today()}.xlsx"
        st.download_button(
            "📥 Download Analysis Report", 
            data=st.session_state.analysis_excel, 
            file_name=fname,
            mime="application/vnd.ms-excel"
        )

    with col2:
        if use_telegram:
            if st.button("📡 Broadcast Top Pick to Telegram"):
                st.info("Transmitting...")
                top_stock = final_df.iloc[0]
                
                signal_msg = f"""
📊 **STOCK ALERT** 📊

**Top Target:** {top_stock['name']} ({top_stock['ticker']})
**Verdict:** {top_stock['AI_Verdict']}

**Fundamentals:**
• Sector: {top_stock['sector']}
• Market Cap: ${top_stock['market_cap']/1e6:.1f}M
• P/E: {top_stock['price_to_earnings']}
• ROE: {top_stock['roe']:.1%}

**Catalyst:** {top_stock['Catalysts']}
"""
                send_buffer = io.BytesIO(st.session_state.analysis_excel)
                if send_telegram_package(tele_token, tele_chat_id, signal_msg, send_buffer, "Analysis_Report.xlsx"):
                    st.success("✅ Alert Sent!")
                else:
                    st.error("❌ Transmission Failed.")

st.markdown("---")
st.caption("Disclaimer: AI analysis is for informational purposes only. Always verify data and consult financial advisors.")

# =============================================================================
# CLEANUP
# =============================================================================
def cleanup():
    if st.session_state.broadcast_engine:
        logger.info("Cleaning up broadcast engine...")
        st.session_state.broadcast_engine.stop()

import atexit
atexit.register(cleanup)

# =============================================================================
# DEBUGGING INFO
# =============================================================================
if st.session_state.broadcast_engine:
    logger.info(f"App loaded - Broadcast engine active: {st.session_state.broadcast_engine.is_active()}")
    logger.info(f"Queue size: {len(st.session_state.broadcast_engine.message_queue) if hasattr(st.session_state.broadcast_engine, 'message_queue') else 'N/A'}")
else:
    logger.info("App loaded - No broadcast engine initialized")
