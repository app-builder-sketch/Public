# streamlit_app.py
# =============================================================================
# TELEGRAM BROADCAST + AI TRADE PLAN MACHINE (Production-grade Streamlit)
# - 500+ ticker universe w/ search, favorites, persisted watchlists
# - Robust data pipeline with transparency metrics
# - Plotly charting workbench + indicator interpretation pane
# - TradingView embed + explicit symbol mapping + overrides
# - Telegram broadcaster with retries, delivery log, multi-chat, message splitting
# - AI Trade Plan: STRICT JSON schema -> UI render + Telegram report
# =============================================================================

from __future__ import annotations

import json
import math
import os
import re
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Optional providers
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except Exception:
    ANTHROPIC_AVAILABLE = False


# =============================================================================
# Config / Secrets
# =============================================================================

def load_secrets() -> Dict[str, str]:
    return {
        "telegram_token": st.secrets.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": st.secrets.get("TELEGRAM_CHAT_ID", ""),  # can be comma-separated for multi-chat
        "openai_key": st.secrets.get("OPENAI_API_KEY", ""),
        "anthropic_key": st.secrets.get("ANTHROPIC_API_KEY", ""),
        "openai_model": st.secrets.get("OPENAI_MODEL", "gpt-4o-mini"),
        "anthropic_model": st.secrets.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
    }


def secrets_status(secrets: Dict[str, str]) -> List[Tuple[str, str]]:
    out = []
    # Telegram
    if secrets["telegram_token"]:
        out.append(("✅ TELEGRAM_BOT_TOKEN", "Loaded"))
        if secrets["telegram_chat_id"]:
            out.append(("✅ TELEGRAM_CHAT_ID", "Loaded"))
        else:
            out.append(("⚠️ TELEGRAM_CHAT_ID", "Missing (Broadcast disabled)"))
    else:
        out.append(("❌ TELEGRAM_BOT_TOKEN", "Missing (Broadcast disabled)"))
        out.append(("❌ TELEGRAM_CHAT_ID", "Missing (Broadcast disabled)"))

    # AI
    if secrets["openai_key"]:
        out.append(("✅ OPENAI_API_KEY", "Loaded"))
        out.append(("ℹ️ OPENAI_MODEL", secrets["openai_model"]))
    else:
        out.append(("❌ OPENAI_API_KEY", "Missing"))

    if secrets["anthropic_key"]:
        out.append(("✅ ANTHROPIC_API_KEY", "Loaded"))
        out.append(("ℹ️ ANTHROPIC_MODEL", secrets["anthropic_model"]))
    else:
        out.append(("❌ ANTHROPIC_API_KEY", "Missing"))

    # yfinance
    out.append(("✅ yfinance", "Available" if YF_AVAILABLE else "Missing (pip install yfinance)"))
    return out


# =============================================================================
# Error logging (never swallow)
# =============================================================================

class ErrorLogger:
    def __init__(self, max_entries: int = 200):
        self.max_entries = max_entries
        if "error_log" not in st.session_state:
            st.session_state.error_log = []

    def log(self, err: Exception, context: str = ""):
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "context": context,
            "type": type(err).__name__,
            "message": str(err),
            "traceback": traceback.format_exc(),
        }
        st.session_state.error_log.append(entry)
        st.session_state.error_log = st.session_state.error_log[-self.max_entries:]

    def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        return st.session_state.error_log[-n:]

    def clear(self):
        st.session_state.error_log = []


error_logger = ErrorLogger()


def ui_error(user_msg: str, err: Exception, context: str):
    error_logger.log(err, context)
    st.error(user_msg)
    with st.expander("Technical details", expanded=False):
        st.code(f"{type(err).__name__}: {err}\n\n{traceback.format_exc()}")


# =============================================================================
# Persistence (watchlists/favorites)
# =============================================================================

class StateStore:
    """
    Best-effort persistence:
    - tries local file write (works on many deployments; ephemeral on Streamlit Cloud)
    - falls back to session-only state
    """
    def __init__(self, path: str = "user_state.json"):
        self.path = path

    def load(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            error_logger.log(e, "StateStore.load")
        return {}

    def save(self, payload: Dict[str, Any]) -> bool:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return True
        except Exception as e:
            error_logger.log(e, "StateStore.save")
            return False


store = StateStore()


# =============================================================================
# Assets & Symbol Mapping
# =============================================================================

@dataclass(frozen=True)
class Asset:
    key: str
    name: str
    asset_class: str  # "Equity", "ETF", "FX", "Crypto", "Index", "Commodity", "Macro"
    provider: str     # "YF" or "FRED" (FRED not implemented in this upgraded file; add if needed)
    symbol: str       # provider symbol
    kind: str = "ohlcv"
    notes: str = ""


def macro_assets() -> Dict[str, Asset]:
    # Keep your macro spine as curated “known-good” tickers.
    return {
        "SPX": Asset("SPX", "S&P 500 Index", "Index", "YF", "^GSPC", "line", "US equity risk anchor"),
        "NDX": Asset("NDX", "Nasdaq 100 Index", "Index", "YF", "^NDX", "line", "US growth / duration proxy"),
        "DXY": Asset("DXY", "US Dollar Index (proxy)", "Index", "YF", "DX-Y.NYB", "line", "DXY proxy"),
        "US10Y": Asset("US10Y", "US 10Y Yield (proxy)", "Macro", "YF", "^TNX", "line", "10Y yield *10"),
        "TLT": Asset("TLT", "20+Y Treasuries ETF", "ETF", "YF", "TLT", "ohlcv", "Long duration"),
        "XAU": Asset("XAU", "Gold (futures)", "Commodity", "YF", "GC=F", "ohlcv", "Gold futures"),
        "WTI": Asset("WTI", "WTI Crude (futures)", "Commodity", "YF", "CL=F", "ohlcv", "WTI crude"),
        "VIX": Asset("VIX", "CBOE VIX", "Index", "YF", "^VIX", "line", "Equity volatility"),
        "BTC": Asset("BTC", "Bitcoin", "Crypto", "YF", "BTC-USD", "ohlcv", "BTC spot proxy"),
        "ETH": Asset("ETH", "Ethereum", "Crypto", "YF", "ETH-USD", "ohlcv", "ETH spot proxy"),
        "EURUSD": Asset("EURUSD", "EUR/USD", "FX", "YF", "EURUSD=X", "line", "FX major"),
        "USDJPY": Asset("USDJPY", "USD/JPY", "FX", "YF", "USDJPY=X", "line", "Rates/risk barometer"),
        "GBPUSD": Asset("GBPUSD", "GBP/USD", "FX", "YF", "GBPUSD=X", "line", "UK vs USD"),
    }


def default_watchlists() -> Dict[str, List[str]]:
    return {
        "Core Macro": ["SPX", "NDX", "DXY", "US10Y", "TLT", "XAU", "WTI", "VIX", "BTC", "ETH", "EURUSD"],
        "FX Majors": ["EURUSD", "USDJPY", "GBPUSD"],
        "Crypto": ["BTC", "ETH"],
    }


class SymbolMapper:
    """
    Explicit mapping rules by asset class.
    Shows mapping state: source -> yfinance symbol -> tradingview symbol.
    Allows overrides for TradingView exchange prefix (esp. equities).
    """
    DEFAULTS = {
        "FX": {"tv_exchange": "FX"},
        "Crypto": {"tv_exchange": "BINANCE", "crypto_quote": "USDT"},
        "Index": {"tv_exchange": ""},     # TradingView uses e.g. SP:SPX, TVC:DXY etc. Needs overrides often
        "Commodity": {"tv_exchange": "COMEX"},  # varies (NYMEX/CBOT/COMEX) -> needs override
        "Equity": {"tv_exchange": "NYSE"},
        "ETF": {"tv_exchange": "ARCA"},
        "Macro": {"tv_exchange": ""},
    }

    def __init__(self):
        pass

    @staticmethod
    def map_for_class(source_symbol: str, asset_class: str, overrides: Dict[str, Any]) -> Dict[str, str]:
        asset_class = asset_class or "Equity"
        cfg = dict(SymbolMapper.DEFAULTS.get(asset_class, {}))
        cfg.update(overrides or {})

        yf_symbol = source_symbol
        tv_symbol = ""

        # FX
        if asset_class == "FX":
            # source_symbol expected like EURUSD or provided already as EURUSD=X in Asset.symbol.
            pair = source_symbol.replace("=X", "")
            tv_symbol = f"{cfg.get('tv_exchange','FX')}:{pair}"
            yf_symbol = source_symbol if source_symbol.endswith("=X") else f"{pair}=X"

        # Crypto
        elif asset_class == "Crypto":
            # yfinance common: BTC-USD. TradingView common: BINANCE:BTCUSDT (quote override)
            tv_ex = cfg.get("tv_exchange", "BINANCE")
            quote = cfg.get("crypto_quote", "USDT")
            base = source_symbol.replace("-USD", "").replace("-USDT", "").replace("USD", "")
            base = base.replace("-", "")
            # best effort base extraction:
            if source_symbol.upper().startswith(("BTC", "ETH", "SOL", "XRP", "BNB")):
                base = re.split(r"[-/]", source_symbol.upper())[0]
            tv_symbol = f"{tv_ex}:{base}{quote}"
            yf_symbol = source_symbol if "-" in source_symbol else f"{base}-USD"

        # Equities / ETFs
        elif asset_class in ("Equity", "ETF"):
            tv_ex = cfg.get("tv_exchange", "NYSE")
            # TradingView needs exchange prefix; user can override
            tv_symbol = f"{tv_ex}:{source_symbol}"
            yf_symbol = source_symbol

        # Index / Commodity / Macro
        else:
            # We cannot reliably auto-map TV for these. Use override if provided.
            tv_ex = cfg.get("tv_exchange", "")
            if tv_ex:
                tv_symbol = f"{tv_ex}:{source_symbol}"
            else:
                tv_symbol = "Needs user input"
            yf_symbol = source_symbol

        return {"source": source_symbol, "yfinance": yf_symbol, "tradingview": tv_symbol}


# =============================================================================
# Ticker Universe (500+)
# =============================================================================

UNIVERSE_URLS = [
    # Large, plain-text symbol lists (GitHub raw tends to be stable)
    "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt",
]

@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_text_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Streamlit Trading Terminal)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.text

@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_equity_universe() -> pd.DataFrame:
    """
    Returns DataFrame with at least 500+ tickers when network is available.
    """
    last_err = None
    for url in UNIVERSE_URLS:
        try:
            txt = fetch_text_url(url)
            # one symbol per line
            syms = []
            for line in txt.splitlines():
                s = line.strip().upper()
                if not s or s.startswith("#"):
                    continue
                # keep common ticker chars
                if re.fullmatch(r"[A-Z0-9.\-]{1,10}", s):
                    syms.append(s)
            syms = sorted(set(syms))
            df = pd.DataFrame({"symbol": syms})
            return df
        except Exception as e:
            last_err = e
            error_logger.log(e, "load_equity_universe")
            continue
    # If we get here: network blocked or source unreachable.
    # Return empty DF; UI will show Needs user input + CSV upload option.
    if last_err:
        raise last_err
    return pd.DataFrame({"symbol": []})


# =============================================================================
# Data fetching + transparency metrics
# =============================================================================

class DataError(Exception):
    pass


def to_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    return df


def load_close_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "Close" in df.columns:
        return pd.to_numeric(df["Close"], errors="coerce")
    # fallback common variations
    for c in df.columns:
        if str(c).lower() == "close":
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(dtype=float)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV safely (only if OHLC present).
    """
    df = to_dt_index(df)
    if df.empty:
        return df
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(df.columns):
        return df
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    out = df.resample(rule).agg(agg).dropna(how="any")
    return out


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
            timeout=30,
        )
        if df is None or df.empty:
            raise DataError(f"No data for {symbol}")
        if isinstance(df.columns, pd.MultiIndex):
            # pick first level if multiindex
            if symbol in df.columns.get_level_values(0):
                df = df[symbol].copy()
            else:
                df = df.copy()
                df.columns = ["_".join([str(x) for x in col if x]) for col in df.columns.values]
        return to_dt_index(df)
    except Exception as e:
        raise DataError(f"Yahoo Finance error for {symbol}: {e}")


def fetch_with_metrics(symbol: str, interval: str, start: datetime, end: datetime) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    t0 = time.time()
    df = fetch_yf(symbol, interval, start, end)
    latency_ms = int((time.time() - t0) * 1000)

    missing = int(df.isna().sum().sum()) if not df.empty else 0
    candles = int(df.shape[0])
    metrics = {
        "symbol": symbol,
        "interval": interval,
        "candles": candles,
        "missing_values": missing,
        "latency_ms": latency_ms,
        "last_refresh_utc": datetime.utcnow().isoformat() + "Z",
        "source": "Yahoo Finance (yfinance)",
    }
    return df, metrics


def safe_pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b is None:
            return None
        if b == 0 or (isinstance(b, float) and (np.isnan(b) or np.isinf(b))):
            return None
        if isinstance(a, float) and (np.isnan(a) or np.isinf(a)):
            return None
        return (a / b - 1.0) * 100.0
    except Exception as e:
        error_logger.log(e, "safe_pct")
        return None


def fmt_num(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    ax = abs(float(x))
    if ax >= 1e9:
        return f"{x/1e9:.2f}B"
    if ax >= 1e6:
        return f"{x/1e6:.2f}M"
    if ax >= 1e3:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.6f}".rstrip("0").rstrip(".")


def fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:+.2f}%"


# =============================================================================
# Indicators
# =============================================================================

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / float(n)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    signal = line.ewm(span=sig, adjust=False, min_periods=sig).mean()
    hist = line - signal
    return line, signal, hist

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return pd.Series(dtype=float)
    tr = true_range(df["High"], df["Low"], df["Close"])
    alpha = 1.0 / float(n)
    return tr.ewm(alpha=alpha, adjust=False, min_periods=n).mean()

def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    upper = mid + k * sd
    lower = mid - k * sd
    return upper, mid, lower

def vwap(df: pd.DataFrame) -> pd.Series:
    if not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].replace(0, np.nan)
    pv = tp * vol
    return pv.cumsum() / vol.cumsum()


def compute_indicators(df: pd.DataFrame, settings: Dict[str, Any]) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    close = load_close_series(df).dropna()
    if close.empty:
        return out

    if settings.get("ema50"):
        out["EMA 50"] = ema(close, 50)
    if settings.get("ema200"):
        out["EMA 200"] = ema(close, 200)
    if settings.get("sma200"):
        out["SMA 200"] = sma(close, 200)
    if settings.get("vwap"):
        out["VWAP"] = vwap(df)

    if settings.get("bb"):
        u, m, l = bollinger(close, 20, 2.0)
        out["BB Upper"] = u
        out["BB Mid"] = m
        out["BB Lower"] = l

    if settings.get("rsi"):
        out["RSI 14"] = rsi_wilder(close, 14)

    if settings.get("macd"):
        m, s, h = macd(close, 12, 26, 9)
        out["MACD"] = m
        out["MACD Signal"] = s
        out["MACD Hist"] = h

    if settings.get("atr"):
        out["ATR 14"] = atr(df, 14)

    return out


# =============================================================================
# Plotly figures
# =============================================================================

def make_price_figure(df: pd.DataFrame, title: str, chart_type: str, ind: Dict[str, pd.Series]) -> go.Figure:
    df = to_dt_index(df)
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=title)
        return fig

    if chart_type == "Candles" and {"Open", "High", "Low", "Close"}.issubset(df.columns):
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
        ))
    else:
        close = load_close_series(df)
        fig.add_trace(go.Scatter(x=df.index, y=close, mode="lines", name="Close"))

    # overlays
    for k in ("EMA 50", "EMA 200", "SMA 200", "VWAP", "BB Upper", "BB Mid", "BB Lower"):
        if k in ind:
            fig.add_trace(go.Scatter(x=ind[k].index, y=ind[k], mode="lines", name=k))

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h"),
    )
    return fig


def make_volume_figure(df: pd.DataFrame) -> go.Figure:
    df = to_dt_index(df)
    fig = go.Figure()
    if df.empty or "Volume" not in df.columns:
        fig.update_layout(title="Volume")
        return fig
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
    fig.update_layout(title="Volume", height=220, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def make_rsi_figure(rsi: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode="lines", name="RSI 14"))
    fig.add_hline(y=70)
    fig.add_hline(y=30)
    fig.update_layout(title="RSI 14", height=240, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def make_macd_figure(macd_line: pd.Series, signal: pd.Series, hist: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=macd_line.index, y=macd_line, mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=signal.index, y=signal, mode="lines", name="Signal"))
    fig.add_trace(go.Bar(x=hist.index, y=hist, name="Hist"))
    fig.update_layout(title="MACD (12,26,9)", height=260, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def make_atr_figure(atr_s: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=atr_s.index, y=atr_s, mode="lines", name="ATR 14"))
    fig.update_layout(title="ATR 14", height=240, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# =============================================================================
# Interpretation pane + simple technical signal extraction
# =============================================================================

def infer_trend(ind: Dict[str, pd.Series]) -> str:
    if "EMA 50" in ind and "EMA 200" in ind:
        a = ind["EMA 50"].iloc[-1]
        b = ind["EMA 200"].iloc[-1]
        if pd.notna(a) and pd.notna(b):
            return "Bullish (EMA50>EMA200)" if a > b else "Bearish (EMA50<EMA200)"
    return "Neutral / Not enough data"


def interpretation_text(df: pd.DataFrame, ind: Dict[str, pd.Series], settings: Dict[str, Any]) -> str:
    close = load_close_series(df).dropna()
    if close.empty:
        return "No price data to interpret."

    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 else None
    chg = safe_pct(last, prev) if prev is not None else None

    lines = [
        f"Price: {fmt_num(last)}   (Δ: {fmt_pct(chg)})",
        f"Trend: {infer_trend(ind)}",
    ]

    if settings.get("rsi") and "RSI 14" in ind:
        r = ind["RSI 14"].iloc[-1]
        if pd.notna(r):
            state = "Overbought" if r > 70 else "Oversold" if r < 30 else "Neutral"
            lines.append(f"RSI(14): {r:.1f} -> {state}")

    if settings.get("macd") and "MACD Hist" in ind:
        h = ind["MACD Hist"].iloc[-1]
        if pd.notna(h):
            lines.append(f"MACD Hist: {h:+.4f} -> {'Bullish momentum' if h>0 else 'Bearish momentum'}")

    if settings.get("bb") and "BB Upper" in ind and "BB Lower" in ind:
        u = ind["BB Upper"].iloc[-1]
        l = ind["BB Lower"].iloc[-1]
        if pd.notna(u) and pd.notna(l):
            pos = "Upper band pressure" if last >= u else "Lower band pressure" if last <= l else "Inside bands"
            lines.append(f"Bollinger(20,2): {pos}")

    if settings.get("atr") and "ATR 14" in ind:
        a = ind["ATR 14"].iloc[-1]
        if pd.notna(a):
            lines.append(f"ATR(14): {fmt_num(float(a))} (volatility proxy)")

    lines.append("Invalidation logic: trend/level breaks should override indicator bias; use stops.")
    return "\n".join(lines)


def simple_levels(df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
    close = load_close_series(df).dropna()
    if close.empty:
        return {}
    w = close.iloc[-lookback:] if len(close) >= lookback else close
    return {
        "support": float(w.min()),
        "resistance": float(w.max()),
    }


# =============================================================================
# TradingView Embed
# =============================================================================

def tradingview_embed_html(tv_symbol: str, interval: str = "D") -> str:
    """
    Lightweight TradingView widget embed.
    Limitations: cannot fully replicate TradingView UI features in embed.
    """
    # Interval mapping for widget: "D", "60", "15", etc.
    # We'll accept "D"/"W" or minute numbers as string.
    if tv_symbol == "Needs user input" or not tv_symbol or ":" not in tv_symbol:
        return "<div style='padding:12px;border:1px solid #444;border-radius:8px;'>TradingView symbol mapping is <b>Needs user input</b>.</div>"

    return f"""
<div class="tradingview-widget-container">
  <div id="tradingview_widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget(
  {{
    "autosize": true,
    "symbol": "{tv_symbol}",
    "interval": "{interval}",
    "timezone": "Etc/UTC",
    "theme": "{'dark' if st.session_state.get('dark_mode', False) else 'light'}",
    "style": "1",
    "locale": "en",
    "enable_publishing": false,
    "allow_symbol_change": true,
    "hide_side_toolbar": true,
    "container_id": "tradingview_widget"
  }}
  );
  </script>
</div>
"""


def tradingview_deeplink(tv_symbol: str) -> str:
    if tv_symbol == "Needs user input" or not tv_symbol or ":" not in tv_symbol:
        return ""
    # Deep-link format: https://www.tradingview.com/chart/?symbol=EXCHANGE:SYMBOL
    return f"https://www.tradingview.com/chart/?symbol={tv_symbol}"


# =============================================================================
# Telegram Broadcaster (Bot API via requests)
# =============================================================================

class TelegramBroadcaster:
    def __init__(self, token: str):
        self.token = token.strip()
        self.base = f"https://api.telegram.org/bot{self.token}"
        if "delivery_log" not in st.session_state:
            st.session_state.delivery_log = []

    def _log(self, record: Dict[str, Any]):
        st.session_state.delivery_log.append(record)
        st.session_state.delivery_log = st.session_state.delivery_log[-300:]

    def split_message(self, text: str, limit: int = 3900) -> List[str]:
        # Split on paragraph boundaries if possible
        if len(text) <= limit:
            return [text]
        parts = []
        remaining = text
        while len(remaining) > limit:
            cut = remaining.rfind("\n\n", 0, limit)
            if cut < limit * 0.6:
                cut = remaining.rfind("\n", 0, limit)
            if cut < limit * 0.6:
                cut = limit
            parts.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            parts.append(remaining)
        # Add (i/n)
        total = len(parts)
        return [f"({i+1}/{total})\n{p}" for i, p in enumerate(parts)]

    def send(self, chat_id: str, text: str, parse_mode: str = "HTML", silent: bool = False,
             report_type: str = "MESSAGE", retries: int = 3) -> bool:
        if not self.token:
            raise ValueError("Telegram token missing")

        url = f"{self.base}/sendMessage"
        ok_all = True
        parts = self.split_message(text)

        for part in parts:
            success = False
            last_err = ""
            for attempt in range(retries):
                try:
                    backoff = min(2.0 ** attempt, 8.0)
                    if attempt > 0:
                        time.sleep(backoff)
                    payload = {
                        "chat_id": chat_id,
                        "text": part,
                        "parse_mode": parse_mode,
                        "disable_notification": bool(silent),
                    }
                    r = requests.post(url, data=payload, timeout=20)
                    status = r.status_code
                    snippet = r.text[:220]
                    if status == 200:
                        success = True
                        self._log({
                            "ts": datetime.utcnow().isoformat() + "Z",
                            "chat_id": chat_id,
                            "report_type": report_type,
                            "status": status,
                            "ok": True,
                            "response_snippet": snippet,
                        })
                        break
                    else:
                        last_err = snippet
                        self._log({
                            "ts": datetime.utcnow().isoformat() + "Z",
                            "chat_id": chat_id,
                            "report_type": report_type,
                            "status": status,
                            "ok": False,
                            "response_snippet": snippet,
                        })
                except Exception as e:
                    last_err = str(e)
                    self._log({
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "chat_id": chat_id,
                        "report_type": report_type,
                        "status": None,
                        "ok": False,
                        "response_snippet": last_err[:220],
                    })
            if not success:
                ok_all = False
        return ok_all


# =============================================================================
# AI Engine (Strict JSON Trade Plan)
# =============================================================================

TRADE_PLAN_SCHEMA = {
    "type": "object",
    "required": ["market", "bias", "timeframe", "plan", "risk", "disclaimer"],
    "properties": {
        "market": {"type": "object"},
        "bias": {"type": "object"},
        "timeframe": {"type": "object"},
        "plan": {"type": "object"},
        "risk": {"type": "object"},
        "disclaimer": {"type": "string"},
    }
}


def validate_trade_plan(obj: Any) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "Trade plan root must be an object"
    for k in TRADE_PLAN_SCHEMA["required"]:
        if k not in obj:
            return False, f"Missing required field: {k}"
    # Minimal deeper checks
    if not isinstance(obj.get("bias", {}), dict):
        return False, "bias must be an object"
    if "direction" not in obj["bias"] or obj["bias"]["direction"] not in ("bull", "bear", "neutral"):
        return False, "bias.direction must be bull|bear|neutral"
    if "confidence" not in obj["bias"] or not isinstance(obj["bias"]["confidence"], (int, float)):
        return False, "bias.confidence must be a number"
    if not isinstance(obj.get("plan", {}), dict):
        return False, "plan must be an object"
    if "entry" not in obj["plan"] or "invalidation" not in obj["plan"] or "tps" not in obj["plan"]:
        return False, "plan must include entry, invalidation, tps"
    if not isinstance(obj["plan"]["tps"], list) or len(obj["plan"]["tps"]) < 1:
        return False, "plan.tps must be a non-empty list"
    return True, "ok"


@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str):
    return openai.OpenAI(api_key=api_key)

@st.cache_resource(show_spinner=False)
def get_anthropic_client(api_key: str):
    return anthropic.Anthropic(api_key=api_key)


class AIEngine:
    def __init__(self, secrets: Dict[str, str]):
        self.secrets = secrets
        self.openai_client = get_openai_client(secrets["openai_key"]) if (OPENAI_AVAILABLE and secrets["openai_key"]) else None
        self.anthropic_client = get_anthropic_client(secrets["anthropic_key"]) if (ANTHROPIC_AVAILABLE and secrets["anthropic_key"]) else None

    def build_trade_plan_prompt(self, market_label: str, df: pd.DataFrame, ind: Dict[str, pd.Series],
                                levels: Dict[str, float], timeframe_label: str) -> str:
        close = load_close_series(df).dropna()
        last = float(close.iloc[-1]) if not close.empty else None

        # Extract latest indicator values, but NEVER fabricate if missing.
        def last_val(name: str) -> Optional[float]:
            s = ind.get(name)
            if s is None or len(s.dropna()) == 0:
                return None
            v = s.dropna().iloc[-1]
            return float(v) if pd.notna(v) else None

        payload = {
            "market": {
                "label": market_label,
                "last_price": last,
                "levels": levels or {},
            },
            "indicators": {
                "ema50": last_val("EMA 50"),
                "ema200": last_val("EMA 200"),
                "rsi14": last_val("RSI 14"),
                "macd_hist": last_val("MACD Hist"),
                "atr14": last_val("ATR 14"),
                "bb_upper": last_val("BB Upper"),
                "bb_lower": last_val("BB Lower"),
                "vwap": last_val("VWAP"),
            },
            "constraints": {
                "no_fabrication": True,
                "if_missing_use_null": True
            }
        }

        schema_hint = {
            "market": {"label": "string", "last_price": "number|null", "levels": {"support": "number|null", "resistance": "number|null"}},
            "bias": {"direction": "bull|bear|neutral", "confidence": "0-100", "why": ["string"]},
            "timeframe": {"label": "string", "context": "string"},
            "plan": {
                "setup_name": "string|null",
                "entry": {"trigger": "string", "entry_zone": ["number|null","number|null"], "type": "market|limit|stop"},
                "stop_loss": {"price": "number|null", "logic": "string"},
                "invalidation": {"price": "number|null", "logic": "string"},
                "trailing_stop": {"method": "ATR|Swing|MA|None", "activation": "string", "rule": "string"},
                "tps": [{"target": "number|null", "rr": "number|null", "action": "string"}],
                "when_not_to_take": ["string"],
                "vol_liq_warnings": ["string"],
                "reasons_grounded": ["string"],
            },
            "risk": {"position_sizing_note": "string", "risk_note": "string"},
            "disclaimer": "string"
        }

        return (
            "You are a senior trader. Create an AI Trade Plan using ONLY the provided JSON input values.\n"
            "RULES:\n"
            "- Output MUST be valid JSON only (no markdown, no commentary).\n"
            "- You MUST NOT invent price/indicator values. If unknown, use null.\n"
            "- Ground reasons in indicators/levels actually provided.\n"
            "- Include TP ladder with at least 3 steps when possible; if cannot compute, keep rr null.\n\n"
            f"TIMEFRAME: {timeframe_label}\n"
            "INPUT_JSON:\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            "OUTPUT_JSON_SCHEMA_EXAMPLE (types only):\n"
            f"{json.dumps(schema_hint, ensure_ascii=False)}"
        )

    def generate_trade_plan(self, provider: str, market_label: str, df: pd.DataFrame, ind: Dict[str, pd.Series],
                            levels: Dict[str, float], timeframe_label: str) -> Tuple[Optional[Dict[str, Any]], str]:
        prompt = self.build_trade_plan_prompt(market_label, df, ind, levels, timeframe_label)

        raw = ""
        try:
            if provider == "openai":
                if not self.openai_client:
                    return None, "OpenAI not configured"
                resp = self.openai_client.chat.completions.create(
                    model=self.secrets["openai_model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=900,
                )
                raw = resp.choices[0].message.content or ""
            elif provider == "anthropic":
                if not self.anthropic_client:
                    return None, "Anthropic not configured"
                resp = self.anthropic_client.messages.create(
                    model=self.secrets["anthropic_model"],
                    max_tokens=900,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text if resp.content else ""
            else:
                return None, "AI provider not selected"

            # Parse strict JSON
            try:
                obj = json.loads(raw)
            except Exception:
                # Attempt to extract JSON object if model wrapped it
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if not m:
                    return None, f"AI returned non-JSON output:\n{raw[:800]}"
                obj = json.loads(m.group(0))

            ok, msg = validate_trade_plan(obj)
            if not ok:
                return None, f"Trade plan JSON failed validation: {msg}\nRaw:\n{raw[:800]}"
            return obj, raw

        except Exception as e:
            error_logger.log(e, f"AI.generate_trade_plan({provider})")
            return None, f"AI error: {e}\nRaw:\n{raw[:800]}"


def trade_plan_to_telegram(plan: Dict[str, Any]) -> str:
    bias = plan["bias"]
    tf = plan["timeframe"]
    mk = plan["market"]
    pl = plan["plan"]

    def n(x): return "—" if x is None else fmt_num(float(x))

    tps_lines = []
    for i, tp in enumerate(pl.get("tps", []), start=1):
        tps_lines.append(f"TP{i}: {n(tp.get('target'))} | R/R: {tp.get('rr') if tp.get('rr') is not None else '—'} | {tp.get('action','')}")

    why = "\n".join([f"• {x}" for x in bias.get("why", [])][:6])
    reasons = "\n".join([f"• {x}" for x in pl.get("reasons_grounded", [])][:8])
    when_not = "\n".join([f"• {x}" for x in pl.get("when_not_to_take", [])][:6])
    warnings = "\n".join([f"• {x}" for x in pl.get("vol_liq_warnings", [])][:6])

    msg = (
        f"🤖 <b>AI TRADE PLAN</b>\n"
        f"<b>Market:</b> {mk.get('label','')}\n"
        f"<b>Timeframe:</b> {tf.get('label','')}\n"
        f"<b>Bias:</b> {bias.get('direction','').upper()} ({bias.get('confidence',0):.0f}/100)\n\n"
        f"<b>Entry:</b> {pl['entry'].get('type','')} | {pl['entry'].get('trigger','')}\n"
        f"<b>Entry Zone:</b> {n(pl['entry'].get('entry_zone',[None,None])[0])} - {n(pl['entry'].get('entry_zone',[None,None])[1])}\n"
        f"<b>Stop:</b> {n(pl.get('stop_loss',{}).get('price'))}  ({pl.get('stop_loss',{}).get('logic','')})\n"
        f"<b>Invalidation:</b> {n(pl.get('invalidation',{}).get('price'))}  ({pl.get('invalidation',{}).get('logic','')})\n\n"
        f"<b>Trailing Stop:</b> {pl.get('trailing_stop',{}).get('method','')} | {pl.get('trailing_stop',{}).get('activation','')}\n"
        f"{pl.get('trailing_stop',{}).get('rule','')}\n\n"
        f"<b>TP Ladder:</b>\n" + ("\n".join(tps_lines) if tps_lines else "—") + "\n\n"
        f"<b>Why:</b>\n{why if why else '—'}\n\n"
        f"<b>Grounded Reasons:</b>\n{reasons if reasons else '—'}\n\n"
        f"<b>When NOT to take:</b>\n{when_not if when_not else '—'}\n\n"
        f"<b>Warnings:</b>\n{warnings if warnings else '—'}\n\n"
        f"<b>Risk:</b> {plan.get('risk',{}).get('risk_note','')}\n"
        f"<i>{plan.get('disclaimer','')}</i>"
    )
    return msg[:3900 * 5]  # allow splitter to handle; protect extreme outputs


# =============================================================================
# Session defaults
# =============================================================================

def ensure_defaults():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True

    persisted = store.load()
    st.session_state.favorites = persisted.get("favorites", [])
    st.session_state.watchlists = persisted.get("watchlists", default_watchlists())
    st.session_state.tv_overrides = persisted.get("tv_overrides", {
        "Equity": {"tv_exchange": "NYSE"},
        "ETF": {"tv_exchange": "ARCA"},
        "FX": {"tv_exchange": "FX"},
        "Crypto": {"tv_exchange": "BINANCE", "crypto_quote": "USDT"},
        "Index": {"tv_exchange": ""},
        "Commodity": {"tv_exchange": ""},
        "Macro": {"tv_exchange": ""},
    })
    st.session_state.dark_mode = persisted.get("dark_mode", False)
    st.session_state.last_data_metrics = persisted.get("last_data_metrics", {})

    if "delivery_log" not in st.session_state:
        st.session_state.delivery_log = []

    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False

    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 60

ensure_defaults()


def persist_state():
    payload = {
        "favorites": st.session_state.get("favorites", []),
        "watchlists": st.session_state.get("watchlists", {}),
        "tv_overrides": st.session_state.get("tv_overrides", {}),
        "dark_mode": st.session_state.get("dark_mode", False),
        "last_data_metrics": st.session_state.get("last_data_metrics", {}),
    }
    store.save(payload)


# =============================================================================
# UI Setup
# =============================================================================

st.set_page_config(
    page_title="Telegram AI Trading Terminal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

secrets = load_secrets()
mapper = SymbolMapper()
ai_engine = AIEngine(secrets)

# Optional dark mode (simple)
if st.session_state.dark_mode:
    st.markdown(
        "<style>.stApp{background-color:#111;color:#eee}</style>",
        unsafe_allow_html=True
    )

st.sidebar.title("📡 Trading Terminal")

with st.sidebar.expander("🔐 Secrets Status", expanded=True):
    for k, v in secrets_status(secrets):
        st.markdown(f"{k}: **{v}**")
    if st.button("🔄 Refresh"):
        st.rerun()

# Telegram init
broadcaster = None
telegram_ready = bool(secrets["telegram_token"] and secrets["telegram_chat_id"])
if telegram_ready:
    broadcaster = TelegramBroadcaster(secrets["telegram_token"])

# =============================================================================
# Universe loading (500+)
# =============================================================================

ASSETS = macro_assets()

with st.sidebar.expander("🧭 Universe + Watchlists", expanded=True):
    universe_ok = False
    universe_err = ""
    equity_df = pd.DataFrame({"symbol": []})
    try:
        equity_df = load_equity_universe()
        universe_ok = (len(equity_df) >= 500)
    except Exception as e:
        universe_err = str(e)
        error_logger.log(e, "Universe load")

    st.caption(f"Equity universe loaded: **{len(equity_df)}** tickers")
    if not universe_ok:
        st.warning("Universe < 500 or unavailable. Needs user input (upload a CSV or fix network).")
        if universe_err:
            with st.expander("Universe diagnostics"):
                st.code(universe_err)

        up = st.file_uploader("Upload tickers CSV (column: symbol)", type=["csv"])
        if up is not None:
            try:
                dfu = pd.read_csv(up)
                if "symbol" not in dfu.columns:
                    st.error("CSV must include a 'symbol' column.")
                else:
                    equity_df = dfu[["symbol"]].dropna()
                    equity_df["symbol"] = equity_df["symbol"].astype(str).str.upper().str.strip()
                    equity_df = equity_df.drop_duplicates().sort_values("symbol").reset_index(drop=True)
                    st.success(f"Loaded {len(equity_df)} tickers from upload.")
                    universe_ok = (len(equity_df) >= 500)
            except Exception as e:
                ui_error("Failed to parse uploaded CSV", e, "CSV upload")

    # Watchlist manager
    watchlists = st.session_state.watchlists
    wl_names = list(watchlists.keys())
    if not wl_names:
        watchlists["Core Macro"] = default_watchlists()["Core Macro"]
        wl_names = list(watchlists.keys())

    wl = st.selectbox("Watchlist", wl_names, index=0)
    new_wl = st.text_input("Create new watchlist", value="")
    if st.button("➕ Create") and new_wl.strip():
        if new_wl.strip() not in watchlists:
            watchlists[new_wl.strip()] = []
            persist_state()
            st.rerun()

    # Search + selection
    asset_class = st.selectbox("Asset Class", ["Macro (curated)", "Equities (universe)", "FX", "Crypto", "Indices", "Commodities"])
    q = st.text_input("Search", value="", placeholder="Type to filter symbols...")

    def filter_syms(df: pd.DataFrame, query: str) -> List[str]:
        if df.empty:
            return []
        if not query.strip():
            return df["symbol"].head(800).tolist()
        qq = query.strip().upper()
        return df[df["symbol"].str.contains(qq, na=False)].head(800)["symbol"].tolist()

    if asset_class == "Macro (curated)":
        options = sorted(list(ASSETS.keys()))
    elif asset_class == "Equities (universe)":
        options = filter_syms(equity_df, q)
    elif asset_class == "FX":
        options = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "EURGBP"]
    elif asset_class == "Crypto":
        options = ["BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "AVAX", "LINK"]
    elif asset_class == "Indices":
        options = ["SPX", "NDX", "VIX", "DJI", "RUT", "DXY"]
    else:
        options = ["XAU", "WTI", "BRENT", "NATGAS", "COPPER", "XAG"]

    # Multi-select assets for workbench (favorites first)
    favs = st.session_state.favorites
    opts_sorted = sorted(options, key=lambda s: (0 if s in favs else 1, s))
    selected = st.multiselect("Selected", opts_sorted, default=opts_sorted[:2] if len(opts_sorted) >= 2 else opts_sorted)

    # Favorites toggle
    fav_pick = st.selectbox("⭐ Favorites", ["(select)"] + opts_sorted[:200])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Add Favorite") and fav_pick != "(select)":
            if fav_pick not in st.session_state.favorites:
                st.session_state.favorites.append(fav_pick)
                persist_state()
                st.rerun()
    with c2:
        if st.button("Remove Favorite") and fav_pick != "(select)":
            if fav_pick in st.session_state.favorites:
                st.session_state.favorites.remove(fav_pick)
                persist_state()
                st.rerun()

    # Add to watchlist
    if st.button("➕ Add selected to watchlist"):
        for s in selected:
            if s not in watchlists[wl]:
                watchlists[wl].append(s)
        persist_state()
        st.success("Added.")

# =============================================================================
# Timeframe + Indicator settings
# =============================================================================

with st.sidebar.expander("⏰ Timeframe", expanded=True):
    tf = st.selectbox("Interval", ["1d", "1wk", "1mo", "1h", "4h", "15m", "5m", "1m"], index=0)

    interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "4h": "60m", "1d": "1d", "1wk": "1wk", "1mo": "1mo"}
    base_interval = interval_map[tf]
    resample_rule = "4H" if tf == "4h" else None

    today = date.today()
    default_start = today - timedelta(days=365)
    dr = st.date_input("Date Range", value=(default_start, today))
    if isinstance(dr, tuple) and len(dr) == 2:
        start_dt = datetime.combine(dr[0], datetime.min.time())
        end_dt = datetime.combine(dr[1] + timedelta(days=1), datetime.min.time())
    else:
        start_dt = datetime.combine(default_start, datetime.min.time())
        end_dt = datetime.combine(today + timedelta(days=1), datetime.min.time())

    # Cap intraday (yfinance limits)
    if base_interval == "1m" and (end_dt - start_dt).days > 7:
        start_dt = end_dt - timedelta(days=7)
        st.warning("⚠️ 1m capped to last 7 days by provider limits.")
    if base_interval in ("5m", "15m", "60m") and (end_dt - start_dt).days > 60:
        start_dt = end_dt - timedelta(days=60)
        st.warning("⚠️ Intraday capped to last ~60 days by provider limits.")

with st.sidebar.expander("📈 Indicators", expanded=True):
    chart_type = st.selectbox("Chart Type", ["Candles", "Line"], index=0)
    show_volume = st.checkbox("Show Volume", value=True)

    settings = {
        "ema50": st.checkbox("EMA 50", value=True),
        "ema200": st.checkbox("EMA 200", value=True),
        "sma200": st.checkbox("SMA 200", value=False),
        "vwap": st.checkbox("VWAP", value=False),
        "bb": st.checkbox("Bollinger (20,2)", value=False),
        "rsi": st.checkbox("RSI 14", value=True),
        "macd": st.checkbox("MACD (12,26,9)", value=True),
        "atr": st.checkbox("ATR 14", value=False),
    }

with st.sidebar.expander("📺 TradingView Mapping", expanded=False):
    # Overrides by class
    tv_overrides = st.session_state.tv_overrides
    cls = st.selectbox("Override class", list(tv_overrides.keys()), index=0)
    ov = tv_overrides.get(cls, {})
    tv_ex = st.text_input("TradingView exchange prefix", value=str(ov.get("tv_exchange", "")))
    crypto_quote = st.text_input("Crypto quote (for BINANCE)", value=str(ov.get("crypto_quote", "USDT")))
    if st.button("Save overrides"):
        tv_overrides[cls] = {"tv_exchange": tv_ex.strip(), "crypto_quote": crypto_quote.strip() or "USDT"}
        persist_state()
        st.success("Saved.")

with st.sidebar.expander("⚙️ App", expanded=False):
    st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)
    st.session_state.auto_refresh = st.toggle("Auto-refresh", value=st.session_state.auto_refresh)
    st.session_state.refresh_interval = st.slider("Refresh interval (sec)", 30, 300, st.session_state.refresh_interval, step=30)
    if st.button("🗑️ Clear cached market data"):
        st.cache_data.clear()
        st.success("Cleared.")

# =============================================================================
# Main layout
# =============================================================================

st.title("📡 AI-Powered Trading Terminal")
st.caption("Charts • TradingView • Strict AI Trade Plans • Telegram Broadcasting")

tabs = st.tabs(["📊 Chart + Indicators", "📈 Dashboard", "🤖 AI Analysis", "📡 Broadcast", "🧪 Diagnostics"])

# Helper: resolve a selected symbol into a provider symbol + class
def resolve_symbol(sel: str) -> Tuple[str, str, str]:
    """
    Returns (label, asset_class, yf_symbol)
    """
    if sel in ASSETS:
        a = ASSETS[sel]
        return f"{a.name} [{a.key}]", a.asset_class, a.symbol

    # Universe symbol -> treat as Equity
    if re.fullmatch(r"[A-Z0-9.\-]{1,10}", sel or ""):
        return f"{sel}", "Equity", sel

    # FX alias like EURUSD -> yfinance EURUSD=X
    if sel in ("EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "EURGBP"):
        return f"{sel}", "FX", f"{sel}=X"

    # Crypto alias BTC -> yfinance BTC-USD
    if sel in ("BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "AVAX", "LINK"):
        return f"{sel}", "Crypto", f"{sel}-USD"

    # Commodities shortcuts
    comm_map = {"XAU": "GC=F", "XAG": "SI=F", "WTI": "CL=F", "BRENT": "BZ=F", "NATGAS": "NG=F", "COPPER": "HG=F"}
    if sel in comm_map:
        return f"{sel}", "Commodity", comm_map[sel]

    return f"{sel}", "Equity", sel


# =============================================================================
# Tab: Chart + Indicators
# =============================================================================

with tabs[0]:
    if not selected:
        st.info("Select symbols in the sidebar.")
    else:
        left, right = st.columns([2.2, 1.0], gap="large")

        with left:
            active = st.selectbox("Active symbol", selected, index=0)
            label, aclass, yf_sym = resolve_symbol(active)

            # Symbol mapping state
            overrides = st.session_state.tv_overrides.get(aclass, {})
            map_state = mapper.map_for_class(
                source_symbol=active if active not in ASSETS else ASSETS[active].symbol,
                asset_class=aclass,
                overrides=overrides
            )

            st.caption(f"Mapping: **source** `{map_state['source']}` → **yfinance** `{map_state['yfinance']}` → **TradingView** `{map_state['tradingview']}`")

            # Fetch data with metrics
            try:
                df, metrics = fetch_with_metrics(yf_sym, base_interval, start_dt, end_dt)
                if resample_rule:
                    df = resample_ohlcv(df, resample_rule)

                st.session_state.last_data_metrics[active] = metrics
                persist_state()

                ind = compute_indicators(df, settings)
                levels = simple_levels(df)

                fig = make_price_figure(df, label, chart_type, ind)
                st.plotly_chart(fig, use_container_width=True)

                if show_volume:
                    st.plotly_chart(make_volume_figure(df), use_container_width=True)

                if settings.get("rsi") and "RSI 14" in ind:
                    st.plotly_chart(make_rsi_figure(ind["RSI 14"].dropna()), use_container_width=True)

                if settings.get("macd") and {"MACD", "MACD Signal", "MACD Hist"}.issubset(ind.keys()):
                    st.plotly_chart(make_macd_figure(ind["MACD"].dropna(), ind["MACD Signal"].dropna(), ind["MACD Hist"].dropna()), use_container_width=True)

                if settings.get("atr") and "ATR 14" in ind:
                    st.plotly_chart(make_atr_figure(ind["ATR 14"].dropna()), use_container_width=True)

                # TradingView embed
                st.markdown("### 📺 TradingView")
                tv_interval = "D" if tf in ("1d", "1wk", "1mo") else ("60" if tf in ("1h","4h") else tf.replace("m",""))
                tv_html = tradingview_embed_html(map_state["tradingview"], tv_interval)
                st.components.v1.html(tv_html, height=520, scrolling=False)
                link = tradingview_deeplink(map_state["tradingview"])
                if link:
                    st.link_button("Open in TradingView", link)

            except Exception as e:
                ui_error("Failed to load chart data.", e, f"Chart fetch {active}")

        with right:
            st.markdown("### 🧠 Chart Interpretation")
            try:
                if "df" in locals() and isinstance(df, pd.DataFrame) and not df.empty:
                    txt = interpretation_text(df, ind, settings)
                    st.code(txt, language="text")

                    lv = simple_levels(df)
                    if lv:
                        st.metric("Support", fmt_num(lv["support"]))
                        st.metric("Resistance", fmt_num(lv["resistance"]))

                    st.markdown("### 📊 Data Transparency")
                    m = st.session_state.last_data_metrics.get(active, {})
                    if m:
                        st.write({
                            "source": m.get("source"),
                            "last_refresh_utc": m.get("last_refresh_utc"),
                            "candles": m.get("candles"),
                            "missing_values": m.get("missing_values"),
                            "latency_ms": m.get("latency_ms"),
                        })

                else:
                    st.info("No data loaded yet.")
            except Exception as e:
                ui_error("Interpretation pane failed.", e, "Interpretation pane")

# =============================================================================
# Tab: Dashboard (lightweight multi-symbol snapshot)
# =============================================================================

with tabs[1]:
    st.subheader("📈 Dashboard Snapshot")
    dash_syms = st.multiselect("Symbols", selected, default=selected[:6] if len(selected) >= 6 else selected)
    dash_interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    dash_base = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}[dash_interval]
    dash_end = datetime.combine(date.today() + timedelta(days=1), datetime.min.time())
    dash_start = dash_end - timedelta(days=365 * 2)

    rows = []
    if st.button("Build dashboard"):
        with st.spinner("Fetching..."):
            for s in dash_syms:
                try:
                    label, aclass, yf_sym = resolve_symbol(s)
                    df, m = fetch_with_metrics(yf_sym, dash_base, dash_start, dash_end)
                    close = load_close_series(df).dropna()
                    if close.empty:
                        continue
                    last = float(close.iloc[-1])
                    prev = float(close.iloc[-2]) if len(close) >= 2 else None
                    rows.append({
                        "Symbol": s,
                        "Last": last,
                        "Δ": safe_pct(last, prev) if prev is not None else None,
                        "Candles": int(close.shape[0]),
                        "Latency(ms)": m.get("latency_ms"),
                    })
                except Exception as e:
                    error_logger.log(e, f"Dashboard {s}")

    if rows:
        ddf = pd.DataFrame(rows)
        view = ddf.copy()
        view["Last"] = view["Last"].map(fmt_num)
        view["Δ"] = view["Δ"].map(fmt_pct)
        st.dataframe(view, use_container_width=True, height=520)
    else:
        st.info("Click 'Build dashboard' to fetch snapshot.")

# =============================================================================
# Tab: AI Analysis (Strict JSON Trade Plan)
# =============================================================================

with tabs[2]:
    st.subheader("🤖 AI Analysis (Strict JSON Trade Plan)")

    providers = []
    if OPENAI_AVAILABLE and secrets["openai_key"]:
        providers.append("openai")
    if ANTHROPIC_AVAILABLE and secrets["anthropic_key"]:
        providers.append("anthropic")

    if not providers:
        st.warning("No AI provider configured. Add OPENAI_API_KEY or ANTHROPIC_API_KEY to secrets.")
    else:
        provider = st.selectbox("AI Provider", providers, index=0)
        sym = st.selectbox("Symbol", selected, index=0)

        risk_pct = st.slider("Risk % (note only)", 0.1, 5.0, 1.0, step=0.1)
        account_size = st.number_input("Account size (optional)", min_value=0.0, value=0.0, step=100.0)
        st.caption("Risk inputs are included as notes unless you wire a full position sizing engine.")

        if st.button("Generate AI Trade Plan"):
            try:
                label, aclass, yf_sym = resolve_symbol(sym)
                df, _m = fetch_with_metrics(yf_sym, base_interval, start_dt, end_dt)
                if resample_rule:
                    df = resample_ohlcv(df, resample_rule)

                ind = compute_indicators(df, settings)
                levels = simple_levels(df)
                tf_label = f"{tf} (yf:{base_interval}{' resampled '+resample_rule if resample_rule else ''})"

                plan, raw = ai_engine.generate_trade_plan(provider, label, df, ind, levels, tf_label)
                if plan is None:
                    st.error("AI Trade Plan failed.")
                    with st.expander("Raw / diagnostics"):
                        st.code(raw)
                else:
                    # Add risk note inputs (non-destructive)
                    plan.setdefault("risk", {})
                    plan["risk"]["risk_note"] = plan["risk"].get("risk_note", "")
                    plan["risk"]["risk_note"] += f" | User risk%={risk_pct:.2f}"
                    if account_size > 0:
                        plan["risk"]["position_sizing_note"] = plan.get("risk", {}).get("position_sizing_note", "")
                        plan["risk"]["position_sizing_note"] += f" | Account={account_size:.0f}"

                    st.session_state.last_trade_plan = plan
                    st.success("Trade plan generated (validated).")

                    st.markdown("### Rendered Plan")
                    st.json(plan)

            except Exception as e:
                ui_error("AI Trade Plan generation crashed.", e, "AI tab")

# =============================================================================
# Tab: Broadcast (Templates + Preview + Multi-chat)
# =============================================================================

with tabs[3]:
    st.subheader("📡 Broadcast Center")

    if not telegram_ready:
        st.warning("Telegram not configured. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in secrets.")
    else:
        chat_ids = [c.strip() for c in secrets["telegram_chat_id"].split(",") if c.strip()]
        st.caption(f"Configured chats: {', '.join(chat_ids)}")

        report_type = st.selectbox("Report Type", ["TEST MESSAGE", "STRICT SIGNAL", "RISK", "MARKET SUMMARY", "AI TRADE PLAN"])

        sym = st.selectbox("Symbol", selected, index=0, key="broadcast_symbol")
        label, aclass, yf_sym = resolve_symbol(sym)

        preview = ""

        # Build report content
        try:
            df, _m = fetch_with_metrics(yf_sym, base_interval, start_dt, end_dt)
            if resample_rule:
                df = resample_ohlcv(df, resample_rule)
            ind = compute_indicators(df, settings)
            lv = simple_levels(df)
            close = load_close_series(df).dropna()
            last = float(close.iloc[-1]) if not close.empty else None

            if report_type == "TEST MESSAGE":
                preview = f"✅ <b>Test</b>\nServer UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}Z"

            elif report_type == "STRICT SIGNAL":
                trend = infer_trend(ind)
                rsi_v = ind.get("RSI 14", pd.Series(dtype=float)).dropna()
                rsi_last = float(rsi_v.iloc[-1]) if len(rsi_v) else None
                preview = (
                    f"📌 <b>STRICT SIGNAL</b>\n"
                    f"<b>Symbol:</b> {label}\n"
                    f"<b>Price:</b> {fmt_num(last)}\n"
                    f"<b>Trend:</b> {trend}\n"
                    f"<b>RSI:</b> {rsi_last:.1f if rsi_last is not None else '—'}\n"
                    f"<b>Support/Resistance:</b> {fmt_num(lv.get('support'))} / {fmt_num(lv.get('resistance'))}\n"
                    f"<i>Not financial advice.</i>"
                )

            elif report_type == "RISK":
                atr_v = ind.get("ATR 14", pd.Series(dtype=float)).dropna()
                atr_last = float(atr_v.iloc[-1]) if len(atr_v) else None
                preview = (
                    f"🛡️ <b>RISK CHECK</b>\n"
                    f"<b>Symbol:</b> {label}\n"
                    f"<b>Price:</b> {fmt_num(last)}\n"
                    f"<b>ATR(14):</b> {fmt_num(atr_last)}\n"
                    f"<b>Notes:</b> Higher ATR => wider stops or smaller size.\n"
                    f"<i>Risk warning: educational only.</i>"
                )

            elif report_type == "MARKET SUMMARY":
                txt = interpretation_text(df, ind, settings)
                preview = f"📰 <b>MARKET SUMMARY</b>\n<b>{label}</b>\n\n<pre>{txt}</pre>"

            elif report_type == "AI TRADE PLAN":
                plan = st.session_state.get("last_trade_plan")
                if not plan:
                    preview = "Needs user input: Generate an AI Trade Plan first in the AI tab."
                else:
                    preview = trade_plan_to_telegram(plan)

        except Exception as e:
            ui_error("Failed to build broadcast preview.", e, "Broadcast preview build")

        st.markdown("### Preview")
        st.text_area("Message preview", value=preview, height=260)

        cols = st.columns([1, 1, 2])
        with cols[0]:
            if st.button("📨 Send to all chats"):
                try:
                    if not preview.strip():
                        st.error("Empty message.")
                    else:
                        ok = True
                        for cid in chat_ids:
                            ok = broadcaster.send(cid, preview, report_type=report_type) and ok
                        st.success("Sent." if ok else "Sent with errors (check Delivery Log).")
                except Exception as e:
                    ui_error("Telegram send failed.", e, "Telegram send all")

        with cols[1]:
            one = st.selectbox("Send to one chat", ["(select)"] + chat_ids)
            if st.button("Send to one"):
                try:
                    if one == "(select)":
                        st.error("Select a chat.")
                    else:
                        ok = broadcaster.send(one, preview, report_type=report_type)
                        st.success("Sent." if ok else "Sent with errors (check Delivery Log).")
                except Exception as e:
                    ui_error("Telegram send failed.", e, "Telegram send one")

        with cols[2]:
            st.caption("Messages are auto-split if too long. Retries use bounded exponential backoff.")

# =============================================================================
# Tab: Diagnostics
# =============================================================================

with tabs[4]:
    st.subheader("🧪 System Health / Diagnostics")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Secrets")
        st.json(secrets)

        st.markdown("### Telegram Connectivity")
        if telegram_ready:
            st.success("Token + Chat IDs loaded")
        else:
            st.warning("Telegram not ready")

        st.markdown("### AI Connectivity")
        ai_ok = (OPENAI_AVAILABLE and secrets["openai_key"]) or (ANTHROPIC_AVAILABLE and secrets["anthropic_key"])
        st.write({"openai_available": OPENAI_AVAILABLE, "anthropic_available": ANTHROPIC_AVAILABLE, "ai_configured": bool(ai_ok)})

    with c2:
        st.markdown("### Latest Data Metrics")
        st.json(st.session_state.get("last_data_metrics", {}))

        st.markdown("### Delivery Log (latest 50)")
        log = st.session_state.get("delivery_log", [])[-50:]
        if log:
            st.dataframe(pd.DataFrame(log), use_container_width=True, height=260)
        else:
            st.caption("No deliveries yet.")

    st.markdown("### Error Log (latest 20)")
    errs = error_logger.recent(20)
    if errs:
        for e in reversed(errs):
            with st.expander(f"{e['ts']} | {e['context']} | {e['type']}"):
                st.code(e["message"])
                st.code(e["traceback"][:2000])
        if st.button("Clear error log"):
            error_logger.clear()
            st.rerun()
    else:
        st.caption("No errors recorded.")

# =============================================================================
# Auto-refresh
# =============================================================================

if st.session_state.auto_refresh:
    # Best-effort: rerun based on interval without background threads.
    st.caption(f"Auto-refresh enabled ({st.session_state.refresh_interval}s).")
    time.sleep(st.session_state.refresh_interval)
    st.rerun()
