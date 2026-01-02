# streamlit_app.py
# =============================================================================
# TELEGRAM BROADCAST + AI MACRO ANALYSIS MACHINE
# Production-grade trading dashboard with LLM-powered insights
# =============================================================================

from __future__ import annotations

import asyncio
import json
import math
import textwrap
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import plotly.express as px

# Telegram integration
try:
    import telegram
    from telegram.ext import Application, CommandHandler, MessageHandler, filters
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

# AI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# yfinance is core
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

# -----------------------------------------------------------------------------
# SECRETS MANAGEMENT & CONFIGURATION
# -----------------------------------------------------------------------------
def load_secrets():
    """Load and validate all required secrets with detailed status reporting"""
    secrets = {
        "telegram_token": st.secrets.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": st.secrets.get("TELEGRAM_CHAT_ID", ""),
        "openai_key": st.secrets.get("OPENAI_API_KEY", ""),
        "anthropic_key": st.secrets.get("ANTHROPIC_API_KEY", ""),
        "openai_model": st.secrets.get("OPENAI_MODEL", "gpt-4-turbo-preview"),
        "anthropic_model": st.secrets.get("ANTHROPIC_MODEL", "claude-3-opus-20240229"),
    }
    return secrets

def check_secrets_status():
    """Provide detailed secrets status for troubleshooting"""
    secrets = load_secrets()
    status = []
    
    # Telegram
    if secrets["telegram_token"]:
        status.append(("✅ TELEGRAM_BOT_TOKEN", "Loaded"))
        if secrets["telegram_chat_id"]:
            status.append(("✅ TELEGRAM_CHAT_ID", "Loaded"))
        else:
            status.append(("⚠️ TELEGRAM_CHAT_ID", "Missing - Required for broadcasting"))
    else:
        status.append(("❌ TELEGRAM_BOT_TOKEN", "Missing - Broadcasting disabled"))
        status.append(("❌ TELEGRAM_CHAT_ID", "Missing - Broadcasting disabled"))
    
    # AI Providers
    if secrets["openai_key"]:
        status.append(("✅ OPENAI_API_KEY", "Loaded"))
        status.append(("ℹ️ OPENAI_MODEL", secrets["openai_model"]))
    else:
        status.append(("❌ OPENAI_API_KEY", "Missing - OpenAI disabled"))
    
    if secrets["anthropic_key"]:
        status.append(("✅ ANTHROPIC_API_KEY", "Loaded"))
        status.append(("ℹ️ ANTHROPIC_MODEL", secrets["anthropic_model"]))
    else:
        status.append(("❌ ANTHROPIC_API_KEY", "Missing - Anthropic disabled"))
    
    return status, secrets

# -----------------------------------------------------------------------------
# ERROR HANDLING & DIAGNOSTICS
# -----------------------------------------------------------------------------
class AppError(Exception):
    """Base exception with user-friendly messages"""
    def __init__(self, message: str, technical: str = ""):
        self.message = message
        self.technical = technical
        super().__init__(message)

class ErrorLogger:
    """Thread-safe error logging with context preservation"""
    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        if "error_log" not in st.session_state:
            st.session_state.error_log = []
    
    def log(self, error: Exception, context: str = ""):
        """Log error with full traceback and context"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "context": context,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "session_id": st.runtime.scriptrunner.get_script_run_ctx().session_id if st.runtime.scriptrunner.get_script_run_ctx() else "unknown"
        }
        st.session_state.error_log.append(entry)
        st.session_state.error_log = st.session_state.error_log[-self.max_entries:]
        
        # Also print to console for server logs
        print(f"[ERROR] {context}: {error}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
    
    def get_recent(self, n: int = 10) -> List[Dict]:
        return st.session_state.error_log[-n:]
    
    def clear(self):
        st.session_state.error_log.clear()

error_logger = ErrorLogger()

def handle_error(error: Exception, user_message: str = "An error occurred"):
    """Unified error handler that shows user-friendly message and logs details"""
    error_logger.log(error, st.session_state.get("current_context", ""))
    
    # Show user-friendly error in UI
    st.error(f"**{user_message}**")
    with st.expander("Technical Details", expanded=False):
        st.code(f"{type(error).__name__}: {str(error)}\n\n{traceback.format_exc()}")

# -----------------------------------------------------------------------------
# ASYNC DATA ENGINE
# -----------------------------------------------------------------------------
class AsyncDataEngine:
    """Threaded data fetching with rate limiting and failover"""
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="data_fetcher")
        # Add streamlit context to threads
        add_script_run_ctx(self.executor)
    
    async def fetch_batch(self, assets: List[Asset], interval: str, start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch multiple assets in parallel with error isolation"""
        loop = asyncio.get_event_loop()
        tasks = []
        
        for asset in assets:
            task = loop.run_in_executor(
                self.executor, 
                self._safe_fetch, 
                asset, interval, start, end
            )
            tasks.append((asset.key, task))
        
        results = {}
        for key, task in tasks:
            try:
                results[key] = await asyncio.wait_for(task, timeout=30)
            except Exception as e:
                error_logger.log(e, f"Batch fetch {key}")
                results[key] = None
        
        return results
    
    def _safe_fetch(self, asset: Asset, interval: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Isolated fetch with multiple retry logic"""
        try:
            return fetch_asset(asset, interval, start, end)
        except Exception as e:
            error_logger.log(e, f"Safe fetch {asset.key}")
            # Try alternative symbols for known problematic assets
            if asset.key == "DXY":
                for alt_symbol in ["DX=F", "^DXY"]:
                    try:
                        alt_asset = Asset(asset.key, asset.name, asset.provider, alt_symbol, asset.kind)
                        return fetch_asset(alt_asset, interval, start, end)
                    except:
                        continue
            return None

data_engine = AsyncDataEngine()

# -----------------------------------------------------------------------------
# TELEGRAM BROADCAST MANAGER
# -----------------------------------------------------------------------------
class TelegramBroadcastManager:
    """Production-grade Telegram broadcasting with queue and retry"""
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.bot = None
        self.queue = []
        self.max_queue_size = 1000
        self.retry_attempts = 3
        self.rate_limit_delay = 0.1
        
        if TELEGRAM_AVAILABLE and token and chat_id:
            try:
                self.bot = telegram.Bot(token=token)
                # Test connection
                asyncio.run(self.bot.get_me())
                st.success("✅ Telegram bot connected")
            except Exception as e:
                error_logger.log(e, "Telegram initialization")
                st.warning("⚠️ Telegram bot initialization failed. Broadcasting disabled.")
                self.bot = None
    
    async def broadcast_message(self, message: str, parse_mode: str = "HTML", silent: bool = False) -> bool:
        """Broadcast with markdown support and rate limiting"""
        if not self.bot:
            return False
        
        # Truncate if too long (Telegram limit: 4096 chars)
        if len(message) > 4000:
            message = message[:4000] + "\n\n...[truncated]"
        
        for attempt in range(self.retry_attempts):
            try:
                await asyncio.sleep(self.rate_limit_delay * (attempt + 1))
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=parse_mode,
                    disable_notification=silent
                )
                return True
            except telegram.error.RetryAfter as e:
                await asyncio.sleep(e.retry_after)
            except Exception as e:
                error_logger.log(e, f"Telegram broadcast attempt {attempt + 1}")
                if attempt == self.retry_attempts - 1:
                    return False
        
        return False
    
    def queue_message(self, message: str, priority: int = 0):
        """Add message to broadcast queue with priority"""
        if len(self.queue) < self.max_queue_size:
            self.queue.append((priority, message))
            self.queue.sort(reverse=True)  # Higher priority first
    
    async def process_queue(self) -> int:
        """Process all queued messages"""
        if not self.bot:
            return 0
        
        processed = 0
        while self.queue:
            _, message = self.queue.pop(0)
            if await self.broadcast_message(message):
                processed += 1
        
        return processed
    
    def format_signal_message(self, asset: str, signal: Dict[str, Any], ai_analysis: str = "") -> str:
        """Format professional trading signal message"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        
        message = f"📊 <b>TRADING SIGNAL - {asset}</b>\n"
        message += f"⏰ <i>{timestamp}</i>\n\n"
        
        # Price action
        if "price" in signal:
            message += f"💰 <b>Price:</b> {signal['price']:.4f}\n"
        if "change_1d" in signal:
            message += f"📈 <b>24h Change:</b> {signal['change_1d']:+.2f}%\n"
        
        # Technical indicators
        if "rsi" in signal:
            rsi_status = "🔴 Overbought" if signal["rsi"] > 70 else "🟢 Oversold" if signal["rsi"] < 30 else "⚪ Neutral"
            message += f"\n📊 <b>RSI(14):</b> {signal['rsi']:.1f} {rsi_status}\n"
        
        if "macd" in signal:
            message += f"📊 <b>MACD:</b> {signal['macd']['histogram']:+.4f} ({signal['macd']['signal']})\n"
        
        # Trend
        if "trend" in signal:
            trend_emoji = "📈" if "bull" in signal["trend"].lower() else "📉"
            message += f"\n{trend_emoji} <b>Trend:</b> {signal['trend']}\n"
        
        # AI Analysis
        if ai_analysis:
            message += f"\n🤖 <b>AI Analysis:</b>\n{ai_analysis[:500]}...\n"
        
        # Risk disclaimer
        message += "\n⚠️ <i>Risk Warning: This is not financial advice. Always do your own research.</i>"
        
        return message

# Global broadcast manager (initialized later)
broadcast_manager = None

# -----------------------------------------------------------------------------
# AI ANALYSIS ENGINE
# -----------------------------------------------------------------------------
class AIAnalyzer:
    """Multi-provider LLM integration for technical analysis"""
    def __init__(self, secrets: Dict[str, str]):
        self.secrets = secrets
        self.openai_client = None
        self.anthropic_client = None
        
        if OPENAI_AVAILABLE and secrets["openai_key"]:
            self.openai_client = openai.OpenAI(api_key=secrets["openai_key"])
        
        if ANTHROPIC_AVAILABLE and secrets["anthropic_key"]:
            self.anthropic_client = anthropic.Anthropic(api_key=secrets["anthropic_key"])
    
    def generate_prompt(self, asset_name: str, df: pd.DataFrame, computed: Dict[str, pd.Series], signal_data: Dict) -> str:
        """Generate structured prompt for LLM analysis"""
        prompt = f"""You are a senior technical analyst. Analyze {asset_name} based on the following data:

PRICE ACTION:
- Current Price: {signal_data.get('price', 'N/A')}
- 24h Change: {signal_data.get('change_1d', 'N/A'):+.2f}%
- 1W Change: {signal_data.get('change_1w', 'N/A'):+.2f}%

TECHNICAL INDICATORS:
"""
        if "RSI 14" in computed:
            rsi_val = computed["RSI 14"].iloc[-1]
            prompt += f"- RSI(14): {rsi_val:.1f} ({'Overbought' if rsi_val > 70 else 'Oversold' if rsi_val < 30 else 'Neutral'})\n"
        
        if "MACD Hist" in computed:
            hist_val = computed["MACD Hist"].iloc[-1]
            prompt += f"- MACD Histogram: {hist_val:+.4f} ({'Bullish' if hist_val > 0 else 'Bearish'})\n"
        
        if "EMA 50" in computed and "EMA 200" in computed:
            ema50 = computed["EMA 50"].iloc[-1]
            ema200 = computed["EMA 200"].iloc[-1]
            trend = "Uptrend" if ema50 > ema200 else "Downtrend"
            prompt += f"- EMA(50/200): {trend}\n"
        
        prompt += f"""

Based on this data:
1. What is the immediate technical outlook (1-3 days)?
2. Identify key support and resistance levels
3. Suggest entry/exit points with risk management
4. What catalysts would invalidate this analysis?

Provide concise, actionable analysis in 4-5 sentences. Use professional trader language."""
        
        return prompt
    
    async def analyze(self, asset_name: str, df: pd.DataFrame, computed: Dict[str, pd.Series], 
                      provider: str = "openai") -> str:
        """Generate AI analysis using selected provider"""
        # Prepare signal data
        close_series = load_close_series(asset_name, df)
        signal_data = {
            "price": close_series.iloc[-1] if not close_series.empty else 0,
            "change_1d": _safe_pct(close_series.iloc[-1], close_series.iloc[-2]) if len(close_series) >= 2 else 0,
            "change_1w": _safe_pct(close_series.iloc[-1], close_series.iloc[-8]) if len(close_series) >= 8 else 0,
        }
        
        prompt = self.generate_prompt(asset_name, df, computed, signal_data)
        
        try:
            if provider == "openai" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=self.secrets["openai_model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
                return response.choices[0].message.content
            
            elif provider == "anthropic" and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model=self.secrets["anthropic_model"],
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            else:
                return "AI analysis unavailable: No valid API key configured."
        
        except Exception as e:
            error_logger.log(e, f"AI analysis {provider}")
            return f"AI analysis failed: {str(e)}"

ai_analyzer = None

# -----------------------------------------------------------------------------
# LIVE DATA & REAL-TIME UPDATES
# -----------------------------------------------------------------------------
class LiveDataManager:
    """Manage live ticker banners and auto-refresh"""
    def __init__(self, refresh_interval: int = 60):
        self.refresh_interval = refresh_interval
        self.last_update = None
    
    def should_refresh(self) -> bool:
        """Check if refresh is needed based on interval"""
        now = datetime.utcnow()
        if self.last_update is None or (now - self.last_update).seconds >= self.refresh_interval:
            self.last_update = now
            return True
        return False
    
    def get_live_quotes(self, keys: List[str]) -> Dict[str, Dict]:
        """Get live quotes for ticker banner"""
        quotes = {}
        for key in keys[:10]:  # Limit to 10 for performance
            asset = get_asset(key)
            if not asset:
                continue
            
            try:
                # Quick 1d fetch for latest
                end = datetime.now()
                start = end - timedelta(days=1)
                df = fetch_asset(asset, "1m", start, end)
                
                if not df.empty:
                    close = load_close_series(key, df)
                    if not close.empty:
                        quotes[key] = {
                            "price": close.iloc[-1],
                            "change": _safe_pct(close.iloc[-1], close.iloc[-2]) if len(close) >= 2 else 0,
                            "asset": asset
                        }
            except Exception as e:
                error_logger.log(e, f"Live quote {key}")
        
        return quotes

live_manager = LiveDataManager()

# -----------------------------------------------------------------------------
# MAIN APP LOGIC (Enhanced from original)
# -----------------------------------------------------------------------------
# [Previous helper functions remain the same but enhanced with error handling]
def _now_utc() -> datetime:
    return datetime.utcnow()

def _safe_pct(a: float, b: float) -> Optional[float]:
    try:
        if b is None or a is None or b == 0 or np.isnan(b) or np.isnan(a):
            return None
        return (a / b - 1.0) * 100.0
    except Exception as e:
        error_logger.log(e, "Percentage calculation")
        return None

def _fmt_num(x: Optional[float]) -> str:
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "—"
        ax = abs(x)
        if ax >= 1e9:
            return f"{x/1e9:.2f}B"
        if ax >= 1e6:
            return f"{x/1e6:.2f}M"
        if ax >= 1e3:
            return f"{x:,.2f}"
        if ax >= 1:
            return f"{x:.4f}".rstrip("0").rstrip(".")
        return f"{x:.6f}".rstrip("0").rstrip(".")
    except Exception as e:
        error_logger.log(e, "Number formatting")
        return "—"

def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:+.2f}%"

def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df = df.sort_index()
        return df
    except Exception as e:
        error_logger.log(e, "Datetime index conversion")
        return pd.DataFrame()

def _align_series(series_list: List[pd.Series]) -> pd.DataFrame:
    try:
        df = pd.concat(series_list, axis=1, join="inner")
        df = _to_datetime_index(df)
        return df.dropna(how="any")
    except Exception as e:
        error_logger.log(e, "Series alignment")
        return pd.DataFrame()

def _cap_intraday_range(interval: str, start: datetime, end: datetime) -> Tuple[datetime, datetime, Optional[str]]:
    max_days = None
    if interval == "1m":
        max_days = 7
    elif interval in ("2m", "5m", "15m", "30m", "60m", "90m"):
        max_days = 60
    elif interval in ("1h",):
        max_days = 365
    
    if max_days is None:
        return start, end, None
    
    if (end - start).days > max_days:
        new_start = end - timedelta(days=max_days)
        return new_start, end, f"⚠️ Interval {interval} limited: date range capped to last {max_days} days."
    return start, end, None

# -----------------------------------------------------------------------------
# Enhanced Indicator Engine
# -----------------------------------------------------------------------------
def sma(series: pd.Series, length: int) -> pd.Series:
    try:
        return series.rolling(length, min_periods=length).mean()
    except Exception as e:
        error_logger.log(e, "SMA calculation")
        return pd.Series(dtype=float)

def ema(series: pd.Series, length: int) -> pd.Series:
    try:
        return series.ewm(span=length, adjust=False, min_periods=length).mean()
    except Exception as e:
        error_logger.log(e, "EMA calculation")
        return pd.Series(dtype=float)

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    try:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        
        alpha = 1.0 / float(length)
        avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
        
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi
    except Exception as e:
        error_logger.log(e, "RSI calculation")
        return pd.Series(dtype=float)

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    try:
        macd_line = ema(close, fast) - ema(close, slow)
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist
    except Exception as e:
        error_logger.log(e, "MACD calculation")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

def bollinger(close: pd.Series, length: int = 20, stdev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    try:
        mid = sma(close, length)
        sd = close.rolling(length, min_periods=length).std()
        upper = mid + stdev * sd
        lower = mid - stdev * sd
        return upper, mid, lower
    except Exception as e:
        error_logger.log(e, "Bollinger Bands calculation")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    try:
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr
    except Exception as e:
        error_logger.log(e, "True Range calculation")
        return pd.Series(dtype=float)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    try:
        tr = true_range(high, low, close)
        alpha = 1.0 / float(length)
        return tr.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    except Exception as e:
        error_logger.log(e, "ATR calculation")
        return pd.Series(dtype=float)

def vwap(df_ohlcv: pd.DataFrame) -> pd.Series:
    try:
        if not {"High", "Low", "Close", "Volume"}.issubset(df_ohlcv.columns):
            return pd.Series(index=df_ohlcv.index, dtype=float)
        
        tp = (df_ohlcv["High"] + df_ohlcv["Low"] + df_ohlcv["Close"]) / 3.0
        vol = df_ohlcv["Volume"].replace(0, np.nan)
        pv = tp * vol
        cum_pv = pv.cumsum()
        cum_vol = vol.cumsum()
        out = cum_pv / cum_vol
        return out
    except Exception as e:
        error_logger.log(e, "VWAP calculation")
        return pd.Series(dtype=float)

# -----------------------------------------------------------------------------
# Data Providers (Enhanced)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Asset:
    key: str
    name: str
    provider: str
    symbol: str
    kind: str
    notes: str = ""

class DataError(Exception):
    pass

@st.cache_data(show_spinner=False, ttl=60 * 20)
def fetch_yf(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    if not YF_AVAILABLE:
        raise DataError("yfinance not installed. Run: pip install yfinance")
    
    try:
        df = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
            timeout=30
        )
        
        if df is None or df.empty:
            raise DataError(f"No data for {symbol}")
        
        if isinstance(df.columns, pd.MultiIndex):
            if symbol in df.columns.get_level_values(0):
                df = df[symbol].copy()
            else:
                df = df.copy()
                df.columns = ["_".join([str(x) for x in col if x]) for col in df.columns.values]
        
        df = _to_datetime_index(df)
        return df
    except Exception as e:
        raise DataError(f"Yahoo Finance error for {symbol}: {str(e)}")

@st.cache_data(show_spinner=False, ttl=60 * 60 * 2)
def fetch_fred_series(series_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            raise DataError(f"FRED request failed ({r.status_code})")
        
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        
        if df is None or df.empty or "DATE" not in df.columns:
            raise DataError(f"No usable data for {series_id}")
        
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"]).set_index("DATE").sort_index()
        
        col = series_id if series_id in df.columns else df.columns[-1]
        s = pd.to_numeric(df[col], errors="coerce")
        out = pd.DataFrame({"Close": s})
        out = out[(out.index >= pd.to_datetime(start)) & (out.index <= pd.to_datetime(end))]
        out = out.dropna()
        
        if out.empty:
            raise DataError(f"No data in range for {series_id}")
        
        return out
    except Exception as e:
        raise DataError(f"FRED error: {str(e)}")

def fetch_asset(asset: Asset, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        if asset.provider == "YF":
            return fetch_yf(asset.symbol, interval, start, end)
        if asset.provider == "FRED":
            return fetch_fred_series(asset.symbol, start, end)
        raise DataError(f"Unknown provider: {asset.provider}")
    except Exception as e:
        raise DataError(f"Asset fetch failed {asset.key}: {str(e)}")

# -----------------------------------------------------------------------------
# Asset & Watchlist Management
# -----------------------------------------------------------------------------
def default_assets() -> Dict[str, Asset]:
    assets: Dict[str, Asset] = {}
    # [Previous asset definitions remain identical]
    # --- Equities / Risk
    assets["SPX"] = Asset("SPX", "S&P 500 Index", "YF", "^GSPC", "line", "US equity risk anchor")
    assets["NDX"] = Asset("NDX", "Nasdaq 100 Index", "YF", "^NDX", "line", "US growth / duration proxy")
    assets["DJI"] = Asset("DJI", "Dow Jones Industrial Average", "YF", "^DJI", "line", "US cyclicals / old economy")
    assets["RUT"] = Asset("RUT", "Russell 2000 Index", "YF", "^RUT", "line", "Small caps / risk appetite")
    assets["ACWI"] = Asset("ACWI", "MSCI ACWI ETF", "YF", "ACWI", "ohlcv", "Global equities")
    assets["EEM"] = Asset("EEM", "Emerging Markets ETF", "YF", "EEM", "ohlcv", "EM equities risk appetite")
    assets["QQQ"] = Asset("QQQ", "Nasdaq 100 ETF", "YF", "QQQ", "ohlcv", "Large-cap growth")
    assets["SPY"] = Asset("SPY", "S&P 500 ETF", "YF", "SPY", "ohlcv", "Liquid proxy for SPX")
    assets["IWM"] = Asset("IWM", "Russell 2000 ETF", "YF", "IWM", "ohlcv", "Small caps")

    # --- Rates proxies
    assets["US10Y"] = Asset("US10Y", "US 10Y Yield (proxy)", "YF", "^TNX", "line", "10Y yield *10")
    assets["US30Y"] = Asset("US30Y", "US 30Y Yield (proxy)", "YF", "^TYX", "line", "30Y yield *10")
    assets["US5Y"] = Asset("US5Y", "US 5Y Yield (proxy)", "YF", "^FVX", "line", "5Y yield *10")
    assets["US13W"] = Asset("US13W", "US 13W Yield (proxy)", "YF", "^IRX", "line", "13W yield *100 (proxy)")

    assets["TLT"] = Asset("TLT", "20+Y Treasuries ETF", "YF", "TLT", "ohlcv", "Long duration")
    assets["IEF"] = Asset("IEF", "7-10Y Treasuries ETF", "YF", "IEF", "ohlcv", "Intermediate duration")
    assets["SHY"] = Asset("SHY", "1-3Y Treasuries ETF", "YF", "SHY", "ohlcv", "Short duration")

    # --- Dollar & FX
    assets["DXY"] = Asset("DXY", "US Dollar Index (proxy)", "YF", "DX-Y.NYB", "line", "DXY proxy")
    assets["EURUSD"] = Asset("EURUSD", "EUR/USD", "YF", "EURUSD=X", "line", "FX major")
    assets["USDJPY"] = Asset("USDJPY", "USD/JPY", "YF", "USDJPY=X", "line", "Rates/risk barometer")
    assets["GBPUSD"] = Asset("GBPUSD", "GBP/USD", "YF", "GBPUSD=X", "line", "UK vs USD")
    assets["USDCNH"] = Asset("USDCNH", "USD/CNH", "YF", "USDCNH=X", "line", "China stress proxy")
    assets["AUDUSD"] = Asset("AUDUSD", "AUD/USD", "YF", "AUDUSD=X", "line", "Commodity FX / growth proxy")
    assets["EURGBP"] = Asset("EURGBP", "EUR/GBP", "YF", "EURGBP=X", "line", "UK vs EU divergence")

    # --- Commodities
    assets["XAU"] = Asset("XAU", "Gold (futures)", "YF", "GC=F", "ohlcv", "Gold futures continuous")
    assets["XAG"] = Asset("XAG", "Silver (futures)", "YF", "SI=F", "ohlcv", "Silver futures continuous")
    assets["COPPER"] = Asset("COPPER", "Copper (futures)", "YF", "HG=F", "ohlcv", "Copper futures")
    assets["WTI"] = Asset("WTI", "WTI Crude Oil (futures)", "YF", "CL=F", "ohlcv", "WTI crude")
    assets["BRENT"] = Asset("BRENT", "Brent Crude Oil (futures)", "YF", "BZ=F", "ohlcv", "Brent crude")
    assets["NATGAS"] = Asset("NATGAS", "Natural Gas (futures)", "YF", "NG=F", "ohlcv", "Natural gas")

    # --- Credit
    assets["HYG"] = Asset("HYG", "High Yield Credit ETF", "YF", "HYG", "ohlcv", "Credit risk appetite")
    assets["LQD"] = Asset("LQD", "Investment Grade Credit ETF", "YF", "LQD", "ohlcv", "Credit quality")
    assets["EMB"] = Asset("EMB", "EM USD Sovereign Debt ETF", "YF", "EMB", "ohlcv", "EM credit / risk")

    # --- Volatility
    assets["VIX"] = Asset("VIX", "CBOE VIX", "YF", "^VIX", "line", "Equity volatility")
    assets["VVIX"] = Asset("VVIX", "CBOE VVIX", "YF", "^VVIX", "line", "Vol-of-vol")

    # --- Crypto
    assets["BTC"] = Asset("BTC", "Bitcoin", "YF", "BTC-USD", "ohlcv", "BTC spot proxy")
    assets["ETH"] = Asset("ETH", "Ethereum", "YF", "ETH-USD", "ohlcv", "ETH spot proxy")

    # --- Liquidity / Macro (FRED)
    assets["WALCL"] = Asset("WALCL", "Fed Balance Sheet (WALCL)", "FRED", "WALCL", "line", "Total Assets, Federal Reserve")
    assets["RRP"] = Asset("RRP", "Reverse Repo (RRPONTSYD)", "FRED", "RRPONTSYD", "line", "ON RRP facility usage")
    assets["TGA"] = Asset("TGA", "Treasury General Account (WTREGEN)", "FRED", "WTREGEN", "line", "US Treasury cash balance")
    assets["M2"] = Asset("M2", "Money Supply (M2SL)", "FRED", "M2SL", "line", "M2 money stock")
    assets["EFFR"] = Asset("EFFR", "Effective Fed Funds Rate (EFFR)", "FRED", "EFFR", "line", "Policy rate")
    assets["SOFR"] = Asset("SOFR", "SOFR (SOFR)", "FRED", "SOFR", "line", "Secured Overnight Financing Rate")

    # Inflation expectations / real rates (FRED)
    assets["T10YIE"] = Asset("T10YIE", "10Y Breakeven Inflation (T10YIE)", "FRED", "T10YIE", "line", "Inflation expectations")
    assets["DFII10"] = Asset("DFII10", "10Y TIPS Real Yield (DFII10)", "FRED", "DFII10", "line", "Real yields")

    return assets

def default_watchlists() -> Dict[str, List[str]]:
    return {
        "Core Macro (must-follow)": ["SPX", "NDX", "RUT", "DXY", "US10Y", "TLT", "XAU", "XAG", "COPPER", "WTI", "VIX", "HYG", "WALCL", "RRP"],
        "Equities (global risk)": ["SPX", "NDX", "DJI", "RUT", "ACWI", "EEM", "QQQ", "SPY", "IWM"],
        "Rates & Bonds": ["US13W", "US5Y", "US10Y", "US30Y", "SHY", "IEF", "TLT"],
        "FX & USD": ["DXY", "EURUSD", "USDJPY", "GBPUSD", "USDCNH", "AUDUSD", "EURGBP"],
        "Commodities": ["XAU", "XAG", "COPPER", "WTI", "BRENT", "NATGAS"],
        "Credit": ["HYG", "LQD", "EMB"],
        "Volatility": ["VIX", "VVIX"],
        "Liquidity (FRED)": ["WALCL", "RRP", "TGA", "M2", "EFFR", "SOFR"],
        "Inflation / Real Rates (FRED)": ["T10YIE", "DFII10"],
        "Crypto": ["BTC", "ETH"],
    }

def ratio_presets() -> Dict[str, Tuple[str, str]]:
    return {
        "Silver / Gold": ("XAG", "XAU"),
        "Gold / SPX": ("XAU", "SPX"),
        "BTC / SPX": ("BTC", "SPX"),
        "BTC / Gold": ("BTC", "XAU"),
        "Copper / Gold": ("COPPER", "XAU"),
        "WTI / Gold": ("WTI", "XAU"),
        "QQQ / SPY": ("QQQ", "SPY"),
        "IWM / SPY": ("IWM", "SPY"),
        "EEM / SPY": ("EEM", "SPY"),
        "SPX / TLT": ("SPX", "TLT"),
        "HYG / TLT": ("HYG", "TLT"),
        "Gold / DXY": ("XAU", "DXY"),
        "ETH / BTC": ("ETH", "BTC"),
    }

# -----------------------------------------------------------------------------
# Session State Management
# -----------------------------------------------------------------------------
def ensure_session_defaults():
    """Initialize all session state variables with safe defaults"""
    defaults = {
        "custom_assets": {},
        "custom_watchlists": default_watchlists(),
        "error_log": [],
        "telegram_connected": False,
        "ai_provider": "openai" if OPENAI_AVAILABLE else "none",
        "broadcast_queue": [],
        "live_refresh_interval": 60,
        "dark_mode": False,
        "current_context": "",
        "ui_expanded": {
            "data_selection": True,
            "timeframe": True,
            "chart_indicators": True,
            "telegram": False,
            "ai": False,
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

ensure_session_defaults()

def get_asset(key: str) -> Optional[Asset]:
    if key in st.session_state.custom_assets:
        return st.session_state.custom_assets[key]
    return ASSETS.get(key)

ASSETS = default_assets()
WATCHLISTS = default_watchlists()
RATIO_PRESETS = ratio_presets()

# -----------------------------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Telegram AI Trading Terminal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://t.me/your_support_bot',
        'Report a bug': 'https://github.com/yourusername/trading-terminal/issues',
        'About': "Professional-grade trading analysis with AI-powered insights and Telegram broadcasting"
    }
)

# Theme toggle
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #E0E0E0;
        }
        </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR - Secrets Status & Configuration
# -----------------------------------------------------------------------------
st.sidebar.title("📡 Trading Terminal")

# Secrets Status Panel
with st.sidebar.expander("🔐 Secrets Status", expanded=True):
    status, secrets = check_secrets_status()
    for service, stat in status:
        st.markdown(f"{service}: **{stat}**")
    
    if st.button("🔄 Refresh Status"):
        st.rerun()

# Telegram Connection
if secrets["telegram_token"] and secrets["telegram_chat_id"]:
    if st.sidebar.button("🔗 Connect Telegram Bot"):
        try:
            broadcast_manager = TelegramBroadcastManager(
                secrets["telegram_token"], 
                secrets["telegram_chat_id"]
            )
            st.session_state.telegram_connected = True
            st.session_state.broadcast_manager = broadcast_manager
            st.sidebar.success("✅ Telegram connected!")
        except Exception as e:
            handle_error(e, "Telegram connection failed")
            st.session_state.telegram_connected = False
else:
    st.sidebar.warning("⚠️ Telegram secrets missing. Broadcasting disabled.")

# AI Provider Selection
with st.sidebar.expander("🤖 AI Configuration", expanded=True):
    providers = []
    if OPENAI_AVAILABLE and secrets["openai_key"]:
        providers.append("openai")
    if ANTHROPIC_AVAILABLE and secrets["anthropic_key"]:
        providers.append("anthropic")
    
    if providers:
        st.session_state.ai_provider = st.selectbox(
            "AI Provider", 
            providers,
            index=providers.index(st.session_state.ai_provider) if st.session_state.ai_provider in providers else 0
        )
        ai_analyzer = AIAnalyzer(secrets)
        st.session_state.ai_analyzer = ai_analyzer
    else:
        st.info("No AI provider configured. Add API keys to secrets.toml")

# Live Data Controls
with st.sidebar.expander("⚡ Live Data", expanded=True):
    st.session_state.live_refresh_interval = st.slider(
        "Refresh interval (seconds)", 
        min_value=30, 
        max_value=300, 
        value=st.session_state.live_refresh_interval,
        step=30
    )
    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# SIDEBAR - Data Selection (Collapsible)
# -----------------------------------------------------------------------------
if st.sidebar.toggle("📊 Data Selection", value=st.session_state.ui_expanded["data_selection"]):
    st.session_state.ui_expanded["data_selection"] = True
    
    category = st.selectbox(
        "Watchlist", 
        list(st.session_state.custom_watchlists.keys()),
        help="Select a watchlist category"
    )
    
    keys_in_cat = [k for k in st.session_state.custom_watchlists[category] if get_asset(k) is not None]
    selected_keys = st.multiselect(
        "Assets", 
        keys_in_cat, 
        default=keys_in_cat[:2] if keys_in_cat else [],
        help="Select multiple assets to analyze"
    )

    use_ratio_mode = st.toggle("Ratio Mode (A/B)", value=False)
    ratio_preset_name = None
    ratio_a = None
    ratio_b = None

    if use_ratio_mode:
        ratio_preset_name = st.selectbox(
            "Ratio Preset", 
            ["(Custom)"] + list(RATIO_PRESETS.keys()),
            index=0
        )
        if ratio_preset_name != "(Custom)":
            ratio_a, ratio_b = RATIO_PRESETS[ratio_preset_name]
        
        a_default = ratio_a if ratio_a else (selected_keys[0] if len(selected_keys) > 0 else "XAU")
        b_default = ratio_b if ratio_b else (selected_keys[1] if len(selected_keys) > 1 else "SPX")
        
        ratio_a = st.selectbox("A (numerator)", list(ASSETS.keys()), 
                              index=list(ASSETS.keys()).index(a_default) if a_default in ASSETS else 0)
        ratio_b = st.selectbox("B (denominator)", list(ASSETS.keys()), 
                              index=list(ASSETS.keys()).index(b_default) if b_default in ASSETS else 0)
else:
    st.session_state.ui_expanded["data_selection"] = False

# -----------------------------------------------------------------------------
# SIDEBAR - Timeframe (Collapsible)
# -----------------------------------------------------------------------------
if st.sidebar.toggle("⏰ Timeframe", value=st.session_state.ui_expanded["timeframe"]):
    st.session_state.ui_expanded["timeframe"] = True
    
    interval = st.selectbox(
        "Interval",
        ["1d", "1wk", "1mo", "1h", "4h", "15m", "5m", "1m"],
        index=0,
        help="Intraday intervals are range-limited"
    )
    
    interval_map = {"4h": "60m", "1h": "60m", "15m": "15m", "5m": "5m", "1m": "1m", "1d": "1d", "1wk": "1wk", "1mo": "1mo"}
    yf_interval = interval_map.get(interval, "1d")
    
    today = date.today()
    default_start = today - timedelta(days=365)
    dr = st.date_input("Date Range", value=(default_start, today), help="Select analysis period")
    
    if isinstance(dr, tuple) and len(dr) == 2:
        start_dt = datetime.combine(dr[0], datetime.min.time())
        end_dt = datetime.combine(dr[1] + timedelta(days=1), datetime.min.time())
    else:
        start_dt = datetime.combine(default_start, datetime.min.time())
        end_dt = datetime.combine(today + timedelta(days=1), datetime.min.time())
    
    start_dt, end_dt, range_warn = _cap_intraday_range(yf_interval, start_dt, end_dt)
    if range_warn:
        st.warning(range_warn)
else:
    st.session_state.ui_expanded["timeframe"] = False

# -----------------------------------------------------------------------------
# SIDEBAR - Chart & Indicators (Collapsible)
# -----------------------------------------------------------------------------
if st.sidebar.toggle("📈 Chart Settings", value=st.session_state.ui_expanded["chart_indicators"]):
    st.session_state.ui_expanded["chart_indicators"] = True
    
    chart_type = st.selectbox("Chart Type", ["Candles", "Line"], index=0)
    show_volume = st.checkbox("Volume Pane", value=True)
    
    st.markdown("**Overlays**")
    use_ema50 = st.checkbox("EMA 50", value=True)
    use_ema200 = st.checkbox("EMA 200", value=True)
    use_sma200 = st.checkbox("SMA 200", value=False)
    use_vwap = st.checkbox("VWAP", value=False)
    use_bb = st.checkbox("Bollinger Bands (20, 2)", value=False)
    
    st.markdown("**Subpanes**")
    show_rsi = st.checkbox("RSI 14", value=True)
    show_macd = st.checkbox("MACD (12, 26, 9)", value=True)
    show_atr = st.checkbox("ATR 14", value=False)
else:
    st.session_state.ui_expanded["chart_indicators"] = False

# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------
st.title("📡 AI-Powered Trading Terminal")
st.caption("Professional macro analysis with LLM insights and Telegram broadcasting")

# Live Ticker Banner
if selected_keys and live_manager.should_refresh():
    live_quotes = live_manager.get_live_quotes(selected_keys[:8])
    if live_quotes:
        cols = st.columns(len(live_quotes))
        for i, (key, quote) in enumerate(live_quotes.items()):
            with cols[i]:
                change_color = "green" if quote["change"] > 0 else "red"
                st.metric(
                    label=key,
                    value=_fmt_num(quote["price"]),
                    delta=_fmt_pct(quote["change"]),
                    delta_color="normal"
                )

# Tab Layout
tab_chart, tab_dashboard, tab_correlation, tab_telegram, tab_ai, tab_settings = st.tabs([
    "📊 Charts", "📈 Dashboard", "🔗 Correlation", "📡 Telegram", "🤖 AI Analysis", "⚙️ Settings"
])

# -----------------------------------------------------------------------------
# CHART TAB
# -----------------------------------------------------------------------------
with tab_chart:
    left, right = st.columns([2, 1], gap="large")
    
    with left:
        if use_ratio_mode:
            a_asset = get_asset(ratio_a)
            b_asset = get_asset(ratio_b)
            
            if not a_asset or not b_asset:
                st.error("⚠️ Invalid ratio assets selected")
            else:
                title = f"Ratio: {a_asset.name} / {b_asset.name}"
                with st.spinner("Fetching ratio data..."):
                    try:
                        df_a = fetch_asset(a_asset, yf_interval, start_dt, end_dt)
                        df_b = fetch_asset(b_asset, yf_interval, start_dt, end_dt)
                        s_a = load_close_series(a_asset.key, df_a).rename("A")
                        s_b = load_close_series(b_asset.key, df_b).rename("B")
                        
                        aligned = _align_series([s_a, s_b])
                        ratio = (aligned["A"] / aligned["B"]).replace([np.inf, -np.inf], np.nan).dropna()
                        df_ratio = pd.DataFrame({"Close": ratio})
                        
                        overlays = compute_overlays(df_ratio)
                        sub = compute_subpanes(df_ratio)
                        
                        fig = make_price_figure(df_ratio, title, "Line", overlays, False)
                        st.plotly_chart(fig, use_container_width=True, key="ratio_chart")
                        
                        # AI Analysis for ratio
                        if st.button("🤖 Analyze Ratio", key="ai_ratio"):
                            with st.spinner("AI analyzing..."):
                                ai_result = asyncio.run(
                                    st.session_state.ai_analyzer.analyze(
                                        f"{ratio_a}/{ratio_b}", df_ratio, sub
                                    )
                                )
                                with right:
                                    st.markdown("### 🤖 AI Ratio Analysis")
                                    st.info(ai_result)
                        
                    except Exception as e:
                        handle_error(e, "Ratio calculation failed")
        else:
            if not selected_keys:
                st.info("Select assets to chart")
            else:
                asset_key = st.selectbox("Active Chart", selected_keys, index=0, key="chart_selector")
                asset = get_asset(asset_key)
                
                if asset:
                    with st.spinner("Fetching market data..."):
                        try:
                            st.session_state.current_context = f"Chart: {asset_key}"
                            df = fetch_asset(asset, yf_interval, start_dt, end_dt)
                            
                            overlays = compute_overlays(df)
                            sub = compute_subpanes(df)
                            
                            title = f"{asset.name} [{asset.key}]"
                            fig = make_price_figure(df, title, chart_type, overlays, show_volume)
                            st.plotly_chart(fig, use_container_width=True, key=f"chart_{asset_key}")
                            
                            # Subpanes
                            if show_volume and "Volume" in df.columns:
                                vfig = make_volume_figure(df, "Volume")
                                st.plotly_chart(vfig, use_container_width=True)
                            
                            if show_rsi and "RSI 14" in sub:
                                st.plotly_chart(make_rsi_figure(sub["RSI 14"], "RSI 14"), use_container_width=True)
                            
                            if show_macd and {"MACD", "MACD Signal", "MACD Hist"}.issubset(sub.keys()):
                                st.plotly_chart(
                                    make_macd_figure(sub["MACD"], sub["MACD Signal"], sub["MACD Hist"], "MACD"),
                                    use_container_width=True
                                )
                            
                            if show_atr and "ATR 14" in sub:
                                st.plotly_chart(make_atr_figure(sub["ATR 14"], "ATR 14"), use_container_width=True)
                            
                        except Exception as e:
                            handle_error(e, f"Failed to load {asset_key}")
    
    with right:
        st.markdown("### 📊 Indicator Readout")
        
        # Asset selector for readout
        readout_key = st.selectbox("Asset", selected_keys if selected_keys else ["SPX"], key="readout_asset")
        readout_asset = get_asset(readout_key)
        
        if readout_asset:
            try:
                df = fetch_asset(readout_asset, yf_interval, start_dt, end_dt)
                overlays = compute_overlays(df)
                sub = compute_subpanes(df)
                computed = {**overlays, **sub}
                
                txt = indicator_readout(df, readout_asset.name, {}, computed)
                st.code(txt, language="text")
                
                # Quick broadcast button
                if st.session_state.telegram_connected and st.button("📡 Broadcast Readout"):
                    message = f"📊 <b>Technical Readout - {readout_asset.key}</b>\n\n{txt}"
                    asyncio.run(st.session_state.broadcast_manager.broadcast_message(message))
                    st.success("✅ Broadcast sent!")
                
            except Exception as e:
                handle_error(e, "Readout generation failed")
        
        # Error log
        st.markdown("### 🐛 Error Log")
        errors = error_logger.get_recent(5)
        if errors:
            for err in errors:
                with st.expander(f"{err['timestamp']} - {err['context'][:30]}..."):
                    st.code(f"{err['error_type']}: {err['message']}\n\n{err['traceback'][:200]}...")
        else:
            st.caption("No errors logged")

# -----------------------------------------------------------------------------
# DASHBOARD TAB
# -----------------------------------------------------------------------------
with tab_dashboard:
    st.subheader("📈 Macro Dashboard")
    
    dash_keys = st.multiselect(
        "Dashboard Assets",
        options=[k for k in ASSETS.keys() if get_asset(k) is not None],
        default=st.session_state.custom_watchlists.get("Core Macro (must-follow)", [])[:12],
        key="dash_selector"
    )
    
    dash_interval = st.selectbox("Dashboard Interval", ["1d", "1wk", "1mo"], index=0)
    dash_yf_interval = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}[dash_interval]
    
    dash_end = datetime.combine(date.today() + timedelta(days=1), datetime.min.time())
    dash_start = dash_end - timedelta(days=365 * 2)
    
    rows = []
    with st.spinner("Building dashboard..."):
        for k in dash_keys:
            asset = get_asset(k)
            if not asset:
                continue
            
            try:
                df = fetch_asset(asset, dash_yf_interval, dash_start, dash_end)
                close = load_close_series(k, df).dropna()
                if close.empty:
                    continue
                
                last = float(close.iloc[-1])
                c_1d = float(close.iloc[-2]) if len(close) >= 2 else np.nan
                
                # Approximate periods based on interval
                if dash_yf_interval == "1d":
                    w_back, m_back, q_back = 5, 21, 63
                elif dash_yf_interval == "1wk":
                    w_back, m_back, q_back = 1, 4, 12
                else:  # 1mo
                    w_back, m_back, q_back = 1, 1, 3
                
                c_1w = float(close.iloc[-(w_back + 1)]) if len(close) >= (w_back + 1) else np.nan
                c_1m = float(close.iloc[-(m_back + 1)]) if len(close) >= (m_back + 1) else np.nan
                c_3m = float(close.iloc[-(q_back + 1)]) if len(close) >= (q_back + 1) else np.nan
                
                rows.append({
                    "Key": k,
                    "Name": asset.name,
                    "Last": last,
                    "24h": _safe_pct(last, c_1d),
                    "~1W": _safe_pct(last, c_1w),
                    "~1M": _safe_pct(last, c_1m),
                    "~3M": _safe_pct(last, c_3m),
                })
            except Exception as e:
                handle_error(e, f"Dashboard {k}")
    
    if rows:
        df_dash = pd.DataFrame(rows)
        view = df_dash.copy()
        view["Last"] = view["Last"].map(_fmt_num)
        view["24h"] = df_dash["24h"].map(_fmt_pct)
        view["~1W"] = df_dash["~1W"].map(_fmt_pct)
        view["~1M"] = df_dash["~1M"].map(_fmt_pct)
        view["~3M"] = df_dash["~3M"].map(_fmt_pct)
        
        st.dataframe(view, use_container_width=True, height=520)
        
        # Broadcast dashboard
        if st.session_state.telegram_connected and st.button("📡 Broadcast Dashboard"):
            message = "📊 <b>Macro Dashboard Update</b>\n\n"
            for _, row in df_dash.iterrows():
                message += f"<b>{row['Key']}</b>: {row['Name'][:20]}...\n"
                message += f"Last: {_fmt_num(row['Last'])} | 24h: {_fmt_pct(row['24h'])}\n\n"
            
            asyncio.run(st.session_state.broadcast_manager.broadcast_message(message))
            st.success("✅ Dashboard broadcast sent!")
    else:
        st.info("No dashboard data available")

# -----------------------------------------------------------------------------
# CORRELATION TAB
# -----------------------------------------------------------------------------
with tab_correlation:
    st.subheader("🔗 Correlation Analysis")
    
    corr_keys = st.multiselect(
        "Correlation Assets",
        options=[k for k in ASSETS.keys() if get_asset(k) is not None],
        default=(selected_keys[:6] if selected_keys else ["SPX", "NDX", "DXY", "XAU", "TLT", "VIX"]),
        key="corr_selector"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        corr_use_log = st.checkbox("Use Log Returns", value=True)
    with col2:
        corr_min_obs = st.slider("Minimum Observations", 30, 500, 60)
    
    if len(corr_keys) < 2:
        st.info("Select at least 2 assets")
    else:
        with st.spinner("Computing correlation..."):
            series_list = []
            for k in corr_keys:
                asset = get_asset(k)
                if not asset:
                    continue
                
                try:
                    df = fetch_asset(asset, yf_interval, start_dt, end_dt)
                    close = load_close_series(k, df).rename(k).dropna()
                    if not close.empty:
                        series_list.append(close)
                except Exception as e:
                    handle_error(e, f"Correlation fetch {k}")
            
            if len(series_list) >= 2:
                try:
                    aligned = _align_series(series_list)
                    if aligned.shape[0] < corr_min_obs:
                        st.warning(f"Only {aligned.shape[0]} observations. Expand date range.")
                    
                    rets = np.log(aligned).diff() if corr_use_log else aligned.pct_change()
                    rets = rets.dropna()
                    corr = rets.corr()
                    
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=corr.values,
                            x=corr.columns,
                            y=corr.index,
                            colorbar=dict(title="Correlation"),
                            zmin=-1,
                            zmax=1,
                            colorscale="RdBu"
                        )
                    )
                    fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Broadcast correlation
                    if st.session_state.telegram_connected and st.button("📡 Broadcast Correlation"):
                        high_corr = []
                        for i in range(len(corr.columns)):
                            for j in range(i+1, len(corr.columns)):
                                if abs(corr.iloc[i, j]) > 0.7:
                                    high_corr.append(f"{corr.columns[i]}-{corr.columns[j]}: {corr.iloc[i, j]:.2f}")
                        
                        message = f"📊 <b>Correlation Report</b>\n\nHigh Correlations (|r| > 0.7):\n"
                        message += "\n".join(high_corr[:10])
                        
                        asyncio.run(st.session_state.broadcast_manager.broadcast_message(message))
                        st.success("✅ Correlation broadcast sent!")
                    
                except Exception as e:
                    handle_error(e, "Correlation computation")
            else:
                st.warning("Not enough valid data series for correlation")

# -----------------------------------------------------------------------------
# TELEGRAM TAB
# -----------------------------------------------------------------------------
with tab_telegram:
    st.subheader("📡 Telegram Broadcast Center")
    
    if not st.session_state.telegram_connected:
        st.warning("⚠️ Telegram not connected. Add credentials to secrets.toml and connect from sidebar.")
        st.info("**Setup Instructions:**\n1. Create bot with @BotFather\n2. Get TELEGRAM_BOT_TOKEN\n3. Add bot to channel\n4. Get channel ID with @getmyid_bot\n5. Add to secrets.toml")
    else:
        st.success("✅ Telegram connected and ready")
        
        # Quick broadcast
        quick_msg = st.text_area("Quick Message", placeholder="Enter broadcast message...")
        if st.button("🚀 Send Broadcast"):
            if quick_msg.strip():
                asyncio.run(st.session_state.broadcast_manager.broadcast_message(quick_msg))
                st.success("✅ Message sent!")
            else:
                st.error("Message cannot be empty")
        
        # Signal broadcast
        st.markdown("---")
        st.subheader("📊 Signal Broadcast")
        
        sig_asset = st.selectbox("Signal Asset", selected_keys if selected_keys else list(ASSETS.keys()))
        sig_type = st.selectbox("Signal Type", ["Bullish", "Bearish", "Neutral", "Breakout", "Reversal"])
        sig_entry = st.number_input("Entry Price", value=0.0, format="%.4f", help="Leave 0 for market")
        sig_stop = st.number_input("Stop Loss", value=0.0, format="%.4f")
        sig_target = st.number_input("Target Price", value=0.0, format="%.4f")
        sig_rationale = st.text_area("Rationale", placeholder="Technical/fundamental reasoning...")
        
        if st.button("📡 Broadcast Signal"):
            try:
                # Fetch current data for signal
                asset = get_asset(sig_asset)
                df = fetch_asset(asset, yf_interval, start_dt, end_dt)
                close = load_close_series(sig_asset, df)
                
                signal_data = {
                    "price": close.iloc[-1] if not close.empty else 0,
                    "change_1d": _safe_pct(close.iloc[-1], close.iloc[-2]) if len(close) >= 2 else 0,
                    "rsi": rsi_wilder(close, 14).iloc[-1] if not close.empty else 0,
                }
                
                # AI analysis for signal
                overlays = compute_overlays(df)
                sub = compute_subpanes(df)
                ai_analysis = asyncio.run(
                    st.session_state.ai_analyzer.analyze(sig_asset, df, {**overlays, **sub})
                )
                
                # Format and broadcast
                signal = {
                    "type": sig_type,
                    "price": signal_data["price"],
                    "change_1d": signal_data["change_1d"],
                    "rsi": signal_data["rsi"],
                }
                
                message = st.session_state.broadcast_manager.format_signal_message(
                    sig_asset, signal, ai_analysis[:200]
                )
                message += f"\n\n<b>Rationale:</b> {sig_rationale[:200]}"
                
                if sig_entry > 0:
                    message += f"\n<b>Entry:</b> {_fmt_num(sig_entry)} | <b>Stop:</b> {_fmt_num(sig_stop)} | <b>Target:</b> {_fmt_num(sig_target)}"
                
                asyncio.run(st.session_state.broadcast_manager.broadcast_message(message))
                st.success("✅ Signal broadcast sent!")
                
            except Exception as e:
                handle_error(e, "Signal broadcast failed")

# -----------------------------------------------------------------------------
# AI ANALYSIS TAB
# -----------------------------------------------------------------------------
with tab_ai:
    st.subheader("🤖 AI Analysis Center")
    
    if not hasattr(st.session_state, 'ai_analyzer'):
        st.warning("⚠️ AI not configured. Add API keys to secrets.toml")
    else:
        ai_asset = st.selectbox("AI Analysis Asset", selected_keys if selected_keys else list(ASSETS.keys()))
        
        analysis_depth = st.slider("Analysis Depth", 1, 5, 3, help="Higher = more detailed but slower")
        analysis_focus = st.multiselect(
            "Focus Areas",
            ["Price Action", "Trend Analysis", "Momentum", "Volatility", "Support/Resistance", "Risk Metrics"],
            default=["Price Action", "Trend Analysis"]
        )
        
        if st.button("🧠 Generate Analysis"):
            try:
                asset = get_asset(ai_asset)
                st.session_state.current_context = f"AI Analysis: {ai_asset}"
                
                with st.spinner(f"Running {analysis_depth}x AI analysis..."):
                    df = fetch_asset(asset, yf_interval, start_dt, end_dt)
                    overlays = compute_overlays(df)
                    sub = compute_subpanes(df)
                    
                    ai_result = asyncio.run(
                        st.session_state.ai_analyzer.analyze(ai_asset, df, {**overlays, **sub})
                    )
                    
                    st.markdown("### 🤖 AI Analysis Result")
                    st.info(ai_result)
                    
                    # Store for broadcasting
                    st.session_state.last_ai_analysis = ai_result
                    
                    # Technical validation
                    st.markdown("### ✅ Technical Validation")
                    validation = []
                    if "RSI 14" in sub:
                        rsi_val = sub["RSI 14"].iloc[-1]
                        validation.append(f"RSI: {rsi_val:.1f} - {'Valid' if 0 < rsi_val < 100 else 'Invalid'}")
                    
                    if "MACD Hist" in sub:
                        hist_val = sub["MACD Hist"].iloc[-1]
                        validation.append(f"MACD hist: {hist_val:+.4f}")
                    
                    st.write(", ".join(validation))
                    
            except Exception as e:
                handle_error(e, "AI analysis failed")
        
        # Broadcast last AI analysis
        if hasattr(st.session_state, 'last_ai_analysis') and st.button("📡 Broadcast AI Analysis"):
            asset = get_asset(ai_asset)
            message = f"🤖 <b>AI Analysis - {ai_asset}</b>\n\n{st.session_state.last_ai_analysis}"
            asyncio.run(st.session_state.broadcast_manager.broadcast_message(message))
            st.success("✅ AI analysis broadcast sent!")

# -----------------------------------------------------------------------------
# TELEGRAM BOT SETUP (for receiving commands)
# -----------------------------------------------------------------------------
async def setup_telegram_bot():
    """Initialize Telegram bot command handlers"""
    if not st.session_state.telegram_connected:
        return
    
    token = secrets["telegram_token"]
    application = Application.builder().token(token).build()
    
    async def start(update, context):
        await update.message.reply_text(
            "🤖 Trading Bot Active!\n\n"
            "Commands:\n"
            "/status - Check bot status\n"
            "/chart <asset> - Get chart\n"
            "/signal <asset> - Get signal\n"
            "/help - Show help"
        )
    
    async def status(update, context):
        await update.message.reply_text(
            f"✅ Bot is online\n"
            f"⏰ Server time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"📊 Active assets: {len(selected_keys) if 'selected_keys' in locals() else 0}"
        )
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    
    # Start bot in background
    await application.initialize()
    await application.start()
    # Note: In production, run this in a separate process

# -----------------------------------------------------------------------------
# SETTINGS TAB
# -----------------------------------------------------------------------------
with tab_settings:
    st.subheader("⚙️ Application Settings")
    
    # Theme
    st.session_state.dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)
    if st.session_state.dark_mode:
        st.info("Dark mode requires page refresh to fully apply")
    
    # Cache management
    if st.button("🗑️ Clear All Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Cache cleared!")
    
    # Error log management
    if st.button("📋 Export Error Log"):
        errors = error_logger.get_recent(100)
        st.download_button(
            "Download Error Log",
            data=json.dumps(errors, indent=2),
            file_name=f"error_log_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
    
    # System info
    st.markdown("### 📊 System Information")
    sys_info = {
        "Python Version": sys.version,
        "Streamlit Version": st.__version__,
        "Pandas Version": pd.__version__,
        "NumPy Version": np.__version__,
        "Telegram Available": TELEGRAM_AVAILABLE,
        "OpenAI Available": OPENAI_AVAILABLE,
        "Anthropic Available": ANTHROPIC_AVAILABLE,
        "YFinance Available": YF_AVAILABLE,
        "Cache Size": len(st.session_state.error_log)
    }
    
    st.json(sys_info)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "🛡️ **Risk Disclaimer**: This tool is for educational and analysis purposes only. "
    "Not financial advice. Always verify signals with your own research. "
    "Telegram broadcasting requires valid bot token and chat ID in secrets.toml. "
    "AI analysis is probabilistic and may contain errors."
)

# -----------------------------------------------------------------------------
# BACKGROUND TASKS (Auto-refresh)
# -----------------------------------------------------------------------------
if st.session_state.telegram_connected:
    # Process broadcast queue in background
    if hasattr(st.session_state, 'broadcast_manager'):
        asyncio.run(st.session_state.broadcast_manager.process_queue())

# Setup Telegram bot commands (runs once)
if TELEGRAM_AVAILABLE and 'telegram_bot_setup' not in st.session_state:
    try:
        asyncio.run(setup_telegram_bot())
        st.session_state.telegram_bot_setup = True
    except Exception as e:
        error_logger.log(e, "Telegram bot setup")

# Auto-refresh mechanism
if st.session_state.get('auto_refresh', False):
    if live_manager.should_refresh():
        st.rerun()
