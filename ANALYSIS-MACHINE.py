# streamlit_app.py
# =============================================================================
# MACRO + TECHNICAL ANALYSIS MACHINE (TradingView-style Watchlists + Ratios)
# -----------------------------------------------------------------------------
# Single-file Streamlit app. Full implementation (no partial snippets).
#
# Features:
# - Curated macro watchlists (Equities, Rates, FX, Commodities, Credit, Vol, Liquidity)
# - Unified data engine:
#     * Yahoo Finance (yfinance) for OHLCV/lines (indices via ^, FX via =X, futures via =F, ETFs)
#     * FRED (St. Louis Fed) for macro series via CSV endpoints (no API key required for many series)
# - Ratio charting (A/B), with aligned calendars and robust NaN handling
# - Plotly professional charts:
#     * Candles for OHLCV series
#     * Lines for macro series/ratios
#     * Optional overlays + subpanes (RSI, MACD, ATR, Volume)
# - Indicator “what it’s saying” panel (rule-based summary)
# - Correlation heatmap + returns table
# - Import/export watchlists (JSON), plus custom symbol mapping UI
# - Streamlit caching and transparent error reporting
#
# Run:
#   streamlit run streamlit_app.py
#
# Dependencies:
#   pip install streamlit pandas numpy plotly yfinance requests
# =============================================================================

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# yfinance is optional but expected for this app’s core experience.
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False


# -----------------------------------------------------------------------------
# App config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Macro + TA Machine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _now_utc() -> datetime:
    return datetime.utcnow()

def _safe_pct(a: float, b: float) -> Optional[float]:
    # percent change from b -> a
    if b is None or a is None:
        return None
    if b == 0 or np.isnan(b) or np.isnan(a):
        return None
    return (a / b - 1.0) * 100.0

def _fmt_num(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    # format large/small gracefully
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

def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:+.2f}%"

def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()
    return df

def _align_series(series_list: List[pd.Series]) -> pd.DataFrame:
    # inner-join aligned by datetime index
    df = pd.concat(series_list, axis=1, join="inner")
    df = _to_datetime_index(df)
    return df.dropna(how="any")

def _cap_intraday_range(interval: str, start: datetime, end: datetime) -> Tuple[datetime, datetime, Optional[str]]:
    """
    yfinance intraday limits:
    - 1m supports ~7 days max
    - 2m,5m,15m,30m,60m support ~60 days (varies)
    This function enforces conservative caps and returns an optional warning message.
    """
    max_days = None
    if interval == "1m":
        max_days = 7
    elif interval in ("2m", "5m", "15m", "30m", "60m", "90m"):
        max_days = 60
    elif interval in ("1h",):
        max_days = 365  # conservative
    else:
        max_days = None

    if max_days is None:
        return start, end, None

    if (end - start).days > max_days:
        new_start = end - timedelta(days=max_days)
        return new_start, end, f"Interval {interval} limited: date range capped to last {max_days} days for reliable intraday data."
    return start, end, None


# -----------------------------------------------------------------------------
# Indicator engine (explicit formulas)
# -----------------------------------------------------------------------------
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Wilder RSI:
    - delta = close.diff()
    - gain = max(delta, 0), loss = max(-delta, 0)
    - avg_gain = Wilder EMA (alpha=1/length)
    - avg_loss = Wilder EMA
    - RSI = 100 - (100/(1+RS))
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    alpha = 1.0 / float(length)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close: pd.Series, length: int = 20, stdev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, length)
    sd = close.rolling(length, min_periods=length).std()
    upper = mid + stdev * sd
    lower = mid - stdev * sd
    return upper, mid, lower

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    # Wilder-style smoothing
    alpha = 1.0 / float(length)
    return tr.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

def vwap(df_ohlcv: pd.DataFrame) -> pd.Series:
    """
    VWAP (session-less approximation on continuous index):
    VWAP = cumulative(sum(typical_price*volume)) / cumulative(sum(volume))
    """
    if not {"High", "Low", "Close", "Volume"}.issubset(df_ohlcv.columns):
        return pd.Series(index=df_ohlcv.index, dtype=float)
    tp = (df_ohlcv["High"] + df_ohlcv["Low"] + df_ohlcv["Close"]) / 3.0
    vol = df_ohlcv["Volume"].replace(0, np.nan)
    pv = tp * vol
    cum_pv = pv.cumsum()
    cum_vol = vol.cumsum()
    out = cum_pv / cum_vol
    return out

def returns(close: pd.Series) -> pd.Series:
    return close.pct_change()

def zscore(series: pd.Series, length: int = 252) -> pd.Series:
    mu = series.rolling(length, min_periods=length).mean()
    sd = series.rolling(length, min_periods=length).std()
    return (series - mu) / sd.replace(0.0, np.nan)


# -----------------------------------------------------------------------------
# Data providers
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Asset:
    key: str
    name: str
    provider: str  # "YF" or "FRED"
    symbol: str    # yfinance ticker or FRED series id
    kind: str      # "ohlcv" or "line"
    notes: str = ""

class DataError(Exception):
    pass

@st.cache_data(show_spinner=False, ttl=60 * 20)  # 20 minutes
def fetch_yf(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    if not YF_AVAILABLE:
        raise DataError("yfinance is not installed/available. Install it: pip install yfinance")

    # yfinance expects strings or datetime; ensure timezone-naive
    df = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if df is None or df.empty:
        raise DataError(f"No data returned from Yahoo Finance for: {symbol}")

    # Handle multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        # If user requested a single ticker, flatten
        if symbol in df.columns.get_level_values(0):
            df = df[symbol].copy()
        else:
            df = df.copy()
            df.columns = ["_".join([str(x) for x in col if x]) for col in df.columns.values]

    df = _to_datetime_index(df)

    # Standardize columns for single-asset OHLCV
    # yfinance typically: Open, High, Low, Close, Adj Close, Volume
    if "Adj Close" in df.columns and "Close" in df.columns:
        # keep both; chart uses Close by default
        pass

    return df

@st.cache_data(show_spinner=False, ttl=60 * 60 * 2)  # 2 hours
def fetch_fred_series(series_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch FRED series as CSV:
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            raise DataError(f"FRED request failed ({r.status_code}) for series {series_id}")
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
    except requests.RequestException as e:
        raise DataError(f"FRED request error for series {series_id}: {e}")

    if df is None or df.empty or "DATE" not in df.columns:
        raise DataError(f"FRED returned no usable data for series {series_id}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).set_index("DATE").sort_index()

    # FRED uses '.' for missing values
    col = series_id
    if col not in df.columns:
        # sometimes FRED column name equals series id; if not, take last column
        col = df.columns[-1]

    s = pd.to_numeric(df[col], errors="coerce")
    out = pd.DataFrame({"Close": s})
    out = out[(out.index >= pd.to_datetime(start)) & (out.index <= pd.to_datetime(end))]
    out = out.dropna()
    if out.empty:
        raise DataError(f"No data in selected date range for FRED series {series_id}")
    return out


def fetch_asset(asset: Asset, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    if asset.provider == "YF":
        return fetch_yf(asset.symbol, interval, start, end)
    if asset.provider == "FRED":
        # FRED is daily-ish; ignore intraday intervals by forcing 1d behavior
        return fetch_fred_series(asset.symbol, start, end)
    raise DataError(f"Unknown provider: {asset.provider}")


# -----------------------------------------------------------------------------
# Curated Macro Assets (keys you can chart + compute ratios on)
# -----------------------------------------------------------------------------
def default_assets() -> Dict[str, Asset]:
    """
    Comprehensive macro set: indices, FX, commodities, rates proxies, credit, vol, liquidity.
    Symbols are chosen to work broadly with Yahoo Finance (YF) and FRED (macro).
    """
    assets: Dict[str, Asset] = {}

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

    # --- Rates proxies (YF indices are scaled; still useful for trend/ratios)
    assets["US10Y"] = Asset("US10Y", "US 10Y Yield (proxy)", "YF", "^TNX", "line", "10Y yield *10")
    assets["US30Y"] = Asset("US30Y", "US 30Y Yield (proxy)", "YF", "^TYX", "line", "30Y yield *10")
    assets["US5Y"] = Asset("US5Y", "US 5Y Yield (proxy)", "YF", "^FVX", "line", "5Y yield *10")
    assets["US13W"] = Asset("US13W", "US 13W Yield (proxy)", "YF", "^IRX", "line", "13W yield *100 (proxy)")

    assets["TLT"] = Asset("TLT", "20+Y Treasuries ETF", "YF", "TLT", "ohlcv", "Long duration")
    assets["IEF"] = Asset("IEF", "7-10Y Treasuries ETF", "YF", "IEF", "ohlcv", "Intermediate duration")
    assets["SHY"] = Asset("SHY", "1-3Y Treasuries ETF", "YF", "SHY", "ohlcv", "Short duration")

    # --- Dollar & FX
    assets["DXY"] = Asset("DXY", "US Dollar Index (proxy)", "YF", "DX-Y.NYB", "line", "DXY proxy; if empty try 'DX=F'")
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
    assets["VVIX"] = Asset("VVIX", "CBOE VVIX", "YF", "^VVIX", "line", "Vol-of-vol (if available)")

    # --- Crypto (yfinance)
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
        "Rates & Bonds": ["US13W", "US5Y", "US10Y", "US30Y", "SHY", "IEF", "TLT", "MOVE_PROXY_NOTE"],
        "FX & USD": ["DXY", "EURUSD", "USDJPY", "GBPUSD", "USDCNH", "AUDUSD", "EURGBP"],
        "Commodities": ["XAU", "XAG", "COPPER", "WTI", "BRENT", "NATGAS"],
        "Credit": ["HYG", "LQD", "EMB"],
        "Volatility": ["VIX", "VVIX"],
        "Liquidity (FRED)": ["WALCL", "RRP", "TGA", "M2", "EFFR", "SOFR"],
        "Inflation / Real Rates (FRED)": ["T10YIE", "DFII10"],
        "Crypto": ["BTC", "ETH"],
    }


# -----------------------------------------------------------------------------
# Ratio presets (A/B)
# -----------------------------------------------------------------------------
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
# Charting
# -----------------------------------------------------------------------------
def make_price_figure(
    df: pd.DataFrame,
    title: str,
    chart_type: str,
    overlays: Dict[str, pd.Series],
    show_volume: bool,
) -> go.Figure:
    df = df.copy()
    df = _to_datetime_index(df)

    fig = go.Figure()

    # Decide base series
    if chart_type == "Candles" and {"Open", "High", "Low", "Close"}.issubset(df.columns):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            )
        )
    else:
        # line fallback
        close_col = "Close" if "Close" in df.columns else df.columns[0]
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[close_col],
                mode="lines",
                name=close_col,
            )
        )

    for name, series in overlays.items():
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=50, b=10),
        height=520 if not show_volume else 520,
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def make_volume_figure(df: pd.DataFrame, title: str) -> Optional[go.Figure]:
    if "Volume" not in df.columns:
        return None
    df = _to_datetime_index(df)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=35, b=10),
        height=220,
    )
    return fig

def make_rsi_figure(rsi_series: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series.values, mode="lines", name="RSI"))
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=35, b=10), height=220)
    return fig

def make_macd_figure(macd_line: pd.Series, signal_line: pd.Series, hist: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=macd_line.index, y=macd_line.values, mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line.values, mode="lines", name="Signal"))
    fig.add_trace(go.Bar(x=hist.index, y=hist.values, name="Hist"))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=35, b=10), height=240)
    return fig

def make_atr_figure(atr_series: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=atr_series.index, y=atr_series.values, mode="lines", name="ATR"))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=35, b=10), height=220)
    return fig


# -----------------------------------------------------------------------------
# Analysis text (indicator readout)
# -----------------------------------------------------------------------------
def indicator_readout(
    df: pd.DataFrame,
    asset_name: str,
    overlays_config: Dict[str, dict],
    computed: Dict[str, pd.Series],
) -> str:
    """
    Produce a concise “what the chart is saying” summary based on last values.
    """
    lines: List[str] = []
    df = _to_datetime_index(df)
    close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]

    if close.dropna().empty:
        return f"{asset_name}: No data to analyze."

    last_close = float(close.dropna().iloc[-1])
    lines.append(f"Last close: {last_close:.6g}")

    # EMA/SMA trend framing
    ema50 = computed.get("EMA 50")
    ema200 = computed.get("EMA 200")
    sma200 = computed.get("SMA 200")

    def _last(s: Optional[pd.Series]) -> Optional[float]:
        if s is None or s.dropna().empty:
            return None
        return float(s.dropna().iloc[-1])

    if ema50 is not None and ema200 is not None:
        e50 = _last(ema50)
        e200 = _last(ema200)
        if e50 is not None and e200 is not None:
            bias = "bull" if e50 > e200 else "bear"
            lines.append(f"Trend (EMA 50 vs EMA 200): {bias} ({e50:.6g} vs {e200:.6g}).")
            lines.append(f"Price vs EMA 200: {'above' if last_close > e200 else 'below'}.")

    elif sma200 is not None:
        s200 = _last(sma200)
        if s200 is not None:
            lines.append(f"Price vs SMA 200: {'above' if last_close > s200 else 'below'}.")

    # RSI
    rsi_s = computed.get("RSI 14")
    if rsi_s is not None and not rsi_s.dropna().empty:
        r = float(rsi_s.dropna().iloc[-1])
        if r >= 70:
            lines.append(f"RSI 14: {r:.1f} (overbought zone).")
        elif r <= 30:
            lines.append(f"RSI 14: {r:.1f} (oversold zone).")
        else:
            lines.append(f"RSI 14: {r:.1f} (neutral).")

    # MACD
    macd_line = computed.get("MACD")
    macd_sig = computed.get("MACD Signal")
    macd_hist = computed.get("MACD Hist")
    if macd_line is not None and macd_sig is not None and macd_hist is not None:
        if not macd_hist.dropna().empty:
            h = float(macd_hist.dropna().iloc[-1])
            state = "bullish momentum" if h > 0 else "bearish momentum"
            lines.append(f"MACD: histogram {h:.6g} ({state}).")

    # Bollinger
    bb_u = computed.get("BB Upper")
    bb_m = computed.get("BB Mid")
    bb_l = computed.get("BB Lower")
    if bb_u is not None and bb_l is not None and bb_m is not None:
        u = _last(bb_u)
        m = _last(bb_m)
        l = _last(bb_l)
        if u is not None and m is not None and l is not None:
            if last_close > u:
                lines.append("Bollinger: price above upper band (strong extension).")
            elif last_close < l:
                lines.append("Bollinger: price below lower band (strong extension).")
            else:
                # position in band
                pos = (last_close - l) / (u - l) if (u - l) != 0 else np.nan
                if not np.isnan(pos):
                    lines.append(f"Bollinger: band position {pos:.2f} (0=lower, 1=upper).")

    # ATR
    atr_s = computed.get("ATR 14")
    if atr_s is not None and not atr_s.dropna().empty:
        a = float(atr_s.dropna().iloc[-1])
        lines.append(f"ATR 14: {a:.6g} (recent volatility).")

    # VWAP
    vwap_s = computed.get("VWAP")
    if vwap_s is not None and not vwap_s.dropna().empty:
        v = float(vwap_s.dropna().iloc[-1])
        lines.append(f"VWAP: price is {'above' if last_close > v else 'below'} VWAP ({v:.6g}).")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# UI state
# -----------------------------------------------------------------------------
ASSETS = default_assets()
WATCHLISTS = default_watchlists()
RATIO_PRESETS = ratio_presets()

# Optional mapping fix: DXY sometimes fails; offer alternative
DXY_ALTS = ["DX-Y.NYB", "DX=F", "^DXY"]


def ensure_session_defaults():
    if "custom_assets" not in st.session_state:
        st.session_state.custom_assets = {}  # key -> Asset override
    if "custom_watchlists" not in st.session_state:
        st.session_state.custom_watchlists = WATCHLISTS.copy()
    if "last_errors" not in st.session_state:
        st.session_state.last_errors = []

ensure_session_defaults()


def get_asset(key: str) -> Optional[Asset]:
    if key in st.session_state.custom_assets:
        return st.session_state.custom_assets[key]
    return ASSETS.get(key)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("Macro + TA Machine")

if not YF_AVAILABLE:
    st.sidebar.error("Missing dependency: yfinance. Install with: pip install yfinance")

with st.sidebar.expander("1) Data Selection", expanded=True):
    category = st.selectbox("Watchlist category", list(st.session_state.custom_watchlists.keys()))
    keys_in_cat = st.session_state.custom_watchlists[category]
    # filter out placeholders
    keys_in_cat = [k for k in keys_in_cat if get_asset(k) is not None]
    selected_keys = st.multiselect("Assets to chart", keys_in_cat, default=keys_in_cat[:2] if keys_in_cat else [])

    use_ratio_mode = st.toggle("Ratio mode (A / B)", value=False)
    ratio_preset_name = None
    ratio_a = None
    ratio_b = None

    if use_ratio_mode:
        ratio_preset_name = st.selectbox("Ratio preset", ["(Custom)"] + list(RATIO_PRESETS.keys()), index=0)
        if ratio_preset_name != "(Custom)":
            ratio_a, ratio_b = RATIO_PRESETS[ratio_preset_name]
        # custom override
        a_default = ratio_a if ratio_a else (selected_keys[0] if len(selected_keys) > 0 else "XAU")
        b_default = ratio_b if ratio_b else (selected_keys[1] if len(selected_keys) > 1 else "SPX")
        ratio_a = st.selectbox("A (numerator)", list(ASSETS.keys()), index=list(ASSETS.keys()).index(a_default) if a_default in ASSETS else 0)
        ratio_b = st.selectbox("B (denominator)", list(ASSETS.keys()), index=list(ASSETS.keys()).index(b_default) if b_default in ASSETS else 0)

with st.sidebar.expander("2) Timeframe", expanded=True):
    interval = st.selectbox(
        "Interval",
        ["1d", "1wk", "1mo", "1h", "4h", "15m", "5m", "1m"],
        index=0,
        help="Intraday intervals may be range-limited by the data provider.",
    )
    # Normalize to yfinance interval strings
    interval_map = {"4h": "60m", "1h": "60m", "15m": "15m", "5m": "5m", "1m": "1m", "1d": "1d", "1wk": "1wk", "1mo": "1mo"}
    yf_interval = interval_map.get(interval, "1d")

    today = date.today()
    default_start = today - timedelta(days=365)
    dr = st.date_input("Date range", value=(default_start, today))
    if isinstance(dr, tuple) and len(dr) == 2:
        start_dt = datetime.combine(dr[0], datetime.min.time())
        end_dt = datetime.combine(dr[1] + timedelta(days=1), datetime.min.time())
    else:
        start_dt = datetime.combine(default_start, datetime.min.time())
        end_dt = datetime.combine(today + timedelta(days=1), datetime.min.time())

    start_dt, end_dt, range_warn = _cap_intraday_range(yf_interval, start_dt, end_dt)
    if range_warn:
        st.warning(range_warn)

with st.sidebar.expander("3) Chart & Indicators", expanded=True):
    chart_type = st.selectbox("Chart type", ["Candles", "Line"], index=0)
    show_volume = st.checkbox("Show volume pane (if available)", value=True)

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

with st.sidebar.expander("4) Tools", expanded=False):
    st.download_button(
        "Export watchlists (JSON)",
        data=json.dumps(st.session_state.custom_watchlists, indent=2),
        file_name="macro_watchlists.json",
        mime="application/json",
    )
    up = st.file_uploader("Import watchlists (JSON)", type=["json"])
    if up is not None:
        try:
            data = json.load(up)
            if isinstance(data, dict):
                st.session_state.custom_watchlists = data
                st.success("Imported watchlists.")
        except Exception as e:
            st.error(f"Import failed: {e}")

    st.markdown("---")
    st.caption("Symbol mapping editor (fix any provider ticker/series).")
    edit_key = st.selectbox("Asset to edit", sorted(list(ASSETS.keys())), index=sorted(list(ASSETS.keys())).index("DXY") if "DXY" in ASSETS else 0)
    base_asset = get_asset(edit_key)
    if base_asset:
        new_provider = st.selectbox("Provider", ["YF", "FRED"], index=0 if base_asset.provider == "YF" else 1)
        new_symbol = st.text_input("Symbol", value=base_asset.symbol)
        new_name = st.text_input("Name", value=base_asset.name)
        new_kind = st.selectbox("Kind", ["ohlcv", "line"], index=0 if base_asset.kind == "ohlcv" else 1)
        new_notes = st.text_area("Notes", value=base_asset.notes, height=80)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save asset override"):
                st.session_state.custom_assets[edit_key] = Asset(
                    key=edit_key,
                    name=new_name,
                    provider=new_provider,
                    symbol=new_symbol,
                    kind=new_kind,
                    notes=new_notes,
                )
                st.success("Saved override.")
        with c2:
            if st.button("Reset override"):
                if edit_key in st.session_state.custom_assets:
                    del st.session_state.custom_assets[edit_key]
                st.success("Reset to defaults.")


# -----------------------------------------------------------------------------
# Main Layout
# -----------------------------------------------------------------------------
st.title("Macro + Technical Analysis Machine")
st.caption("Curated macro watchlists + ratio charting + technical overlays + correlation. Designed for an all-round macro view.")

tab_chart, tab_dashboard, tab_correlation, tab_watchlists = st.tabs(["Chart", "Macro Dashboard", "Correlation", "Watchlists"])

# -----------------------------------------------------------------------------
# Data load + compute
# -----------------------------------------------------------------------------
def load_close_series(asset_key: str, df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "Close" in df.columns:
        return df["Close"].copy()
    # fallback: first numeric column
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return df[c].copy()
    return pd.Series(dtype=float)

def compute_overlays(df: pd.DataFrame) -> Dict[str, pd.Series]:
    df = _to_datetime_index(df)
    out: Dict[str, pd.Series] = {}
    close = load_close_series("x", df)

    if close.empty:
        return out

    if use_ema50:
        out["EMA 50"] = ema(close, 50)
    if use_ema200:
        out["EMA 200"] = ema(close, 200)
    if use_sma200:
        out["SMA 200"] = sma(close, 200)
    if use_vwap and {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        out["VWAP"] = vwap(df)
    if use_bb:
        upper, mid, lower = bollinger(close, 20, 2.0)
        out["BB Upper"] = upper
        out["BB Mid"] = mid
        out["BB Lower"] = lower

    return out

def compute_subpanes(df: pd.DataFrame) -> Dict[str, pd.Series]:
    df = _to_datetime_index(df)
    out: Dict[str, pd.Series] = {}
    close = load_close_series("x", df)

    if close.empty:
        return out

    if show_rsi:
        out["RSI 14"] = rsi_wilder(close, 14)
    if show_macd:
        m, s, h = macd(close, 12, 26, 9)
        out["MACD"] = m
        out["MACD Signal"] = s
        out["MACD Hist"] = h
    if show_atr and {"High", "Low"}.issubset(df.columns):
        # if no OHLC, ATR not computed
        if "Close" in df.columns:
            out["ATR 14"] = atr(df["High"], df["Low"], df["Close"], 14)
    return out


def push_error(msg: str):
    st.session_state.last_errors.append({"time": _now_utc().isoformat(timespec="seconds") + "Z", "error": msg})
    st.session_state.last_errors = st.session_state.last_errors[-50:]


# -----------------------------------------------------------------------------
# Chart tab
# -----------------------------------------------------------------------------
with tab_chart:
    left, right = st.columns([2.2, 1], gap="large")

    with left:
        if use_ratio_mode:
            a_asset = get_asset(ratio_a) if ratio_a else None
            b_asset = get_asset(ratio_b) if ratio_b else None

            if not a_asset or not b_asset:
                st.error("Ratio mode requires valid A and B assets.")
            else:
                title = f"Ratio: {a_asset.name} / {b_asset.name} ({a_asset.key}/{b_asset.key})"
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

                        fig = make_price_figure(df_ratio, title, "Line", overlays, show_volume=False)
                        st.plotly_chart(fig, use_container_width=True)

                        # Subpanes for ratio (RSI/MACD only make sense; ATR/Volume skipped)
                        if show_rsi and "RSI 14" in sub:
                            st.plotly_chart(make_rsi_figure(sub["RSI 14"], "RSI 14"), use_container_width=True)
                        if show_macd and {"MACD", "MACD Signal", "MACD Hist"}.issubset(sub.keys()):
                            st.plotly_chart(make_macd_figure(sub["MACD"], sub["MACD Signal"], sub["MACD Hist"], "MACD"), use_container_width=True)

                    except Exception as e:
                        push_error(str(e))
                        st.error(f"Ratio load failed: {e}")

        else:
            if not selected_keys:
                st.info("Select at least one asset to chart.")
            else:
                # For simplicity and clarity: render one chart at a time with selector
                asset_key = st.selectbox("Active chart", selected_keys, index=0)
                asset = get_asset(asset_key)
                if asset is None:
                    st.error("Invalid asset selection.")
                else:
                    with st.spinner("Fetching market data..."):
                        try:
                            df = fetch_asset(asset, yf_interval, start_dt, end_dt)

                            overlays = compute_overlays(df)
                            sub = compute_subpanes(df)

                            title = f"{asset.name} [{asset.key}] — {asset.provider}:{asset.symbol}"
                            fig = make_price_figure(df, title, chart_type, overlays, show_volume=show_volume)
                            st.plotly_chart(fig, use_container_width=True)

                            if show_volume:
                                vfig = make_volume_figure(df, "Volume")
                                if vfig is not None:
                                    st.plotly_chart(vfig, use_container_width=True)

                            # Subpanes
                            if show_rsi and "RSI 14" in sub:
                                st.plotly_chart(make_rsi_figure(sub["RSI 14"], "RSI 14"), use_container_width=True)
                            if show_macd and {"MACD", "MACD Signal", "MACD Hist"}.issubset(sub.keys()):
                                st.plotly_chart(make_macd_figure(sub["MACD"], sub["MACD Signal"], sub["MACD Hist"], "MACD"), use_container_width=True)
                            if show_atr and "ATR 14" in sub:
                                st.plotly_chart(make_atr_figure(sub["ATR 14"], "ATR 14"), use_container_width=True)

                        except Exception as e:
                            push_error(str(e))
                            st.error(f"Load failed: {e}")

    with right:
        st.subheader("Indicator Readout")
        st.caption("Rule-based summary of what your overlays/subpanes are currently implying.")

        if use_ratio_mode:
            a_asset = get_asset(ratio_a) if ratio_a else None
            b_asset = get_asset(ratio_b) if ratio_b else None
            if a_asset and b_asset:
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
                    computed = {**overlays, **sub}
                    txt = indicator_readout(df_ratio, f"Ratio {a_asset.key}/{b_asset.key}", {}, computed)
                    st.code(txt, language="text")
                except Exception as e:
                    st.error(f"Readout unavailable: {e}")
        else:
            if selected_keys:
                asset_key = selected_keys[0] if selected_keys else None
                asset_key = st.selectbox("Readout asset", selected_keys, index=0, key="readout_asset")
                asset = get_asset(asset_key)
                if asset:
                    try:
                        df = fetch_asset(asset, yf_interval, start_dt, end_dt)
                        overlays = compute_overlays(df)
                        sub = compute_subpanes(df)
                        computed = {**overlays, **sub}
                        txt = indicator_readout(df, asset.name, {}, computed)
                        st.code(txt, language="text")
                    except Exception as e:
                        st.error(f"Readout unavailable: {e}")
            else:
                st.info("Select assets to get a readout.")

        st.subheader("Errors & Transparency")
        if st.session_state.last_errors:
            st.dataframe(pd.DataFrame(st.session_state.last_errors), use_container_width=True, height=240)
        else:
            st.caption("No errors logged in this session.")


# -----------------------------------------------------------------------------
# Macro Dashboard tab (multi-asset summary)
# -----------------------------------------------------------------------------
with tab_dashboard:
    st.subheader("Macro Dashboard")
    st.caption("Latest value + % changes. Select a set of assets and get an at-a-glance macro read.")

    dash_keys = st.multiselect(
        "Dashboard assets",
        options=[k for k in ASSETS.keys() if get_asset(k) is not None],
        default=st.session_state.custom_watchlists.get("Core Macro (must-follow)", [])[:12],
    )

    dash_interval = st.selectbox("Dashboard interval", ["1d", "1wk", "1mo"], index=0)
    dash_yf_interval = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}[dash_interval]

    # Keep dashboard range modest and relevant
    dash_end = datetime.combine(date.today() + timedelta(days=1), datetime.min.time())
    dash_start = dash_end - timedelta(days=365 * 2)

    rows = []
    with st.spinner("Building dashboard..."):
        for k in dash_keys:
            a = get_asset(k)
            if not a:
                continue
            try:
                df = fetch_asset(a, dash_yf_interval, dash_start, dash_end)
                c = load_close_series(k, df).dropna()
                if c.empty:
                    continue

                last = float(c.iloc[-1])
                c_1d = float(c.iloc[-2]) if len(c) >= 2 else np.nan

                # Compute 1w and 1m approximations by index steps (works for daily/weekly/monthly)
                idx = c.index
                # 5 bars ~ 1w (daily), 4 bars ~ 1m (weekly), etc. It’s a rough but transparent heuristic.
                w_back = 5
                m_back = 21
                if dash_yf_interval == "1wk":
                    w_back = 1
                    m_back = 4
                if dash_yf_interval == "1mo":
                    w_back = 1
                    m_back = 1

                c_1w = float(c.iloc[-(w_back + 1)]) if len(c) >= (w_back + 1) else np.nan
                c_1m = float(c.iloc[-(m_back + 1)]) if len(c) >= (m_back + 1) else np.nan

                rows.append({
                    "Key": k,
                    "Name": a.name,
                    "Provider": a.provider,
                    "Symbol": a.symbol,
                    "Last": last,
                    "Chg (1 bar)": _safe_pct(last, c_1d),
                    "Chg (~1w)": _safe_pct(last, c_1w),
                    "Chg (~1m)": _safe_pct(last, c_1m),
                })
            except Exception as e:
                push_error(f"{k}: {e}")

    if rows:
        df_dash = pd.DataFrame(rows)
        # formatting view
        view = df_dash.copy()
        view["Last"] = view["Last"].map(_fmt_num)
        view["Chg (1 bar)"] = df_dash["Chg (1 bar)"].map(_fmt_pct)
        view["Chg (~1w)"] = df_dash["Chg (~1w)"].map(_fmt_pct)
        view["Chg (~1m)"] = df_dash["Chg (~1m)"].map(_fmt_pct)
        st.dataframe(view, use_container_width=True, height=520)
    else:
        st.info("No dashboard rows yet. Try fewer assets or a broader date range.")

    st.markdown("---")
    st.subheader("Quick Ratio Launcher")
    cols = st.columns(3)
    preset_names = list(RATIO_PRESETS.keys())
    for i, name in enumerate(preset_names):
        a, b = RATIO_PRESETS[name]
        with cols[i % 3]:
            if st.button(f"{name}  ({a}/{b})", key=f"ratio_btn_{name}"):
                st.session_state["__launch_ratio__"] = {"a": a, "b": b, "name": name}
                st.success("Go to Chart tab → enable Ratio mode and pick the preset.")


# -----------------------------------------------------------------------------
# Correlation tab
# -----------------------------------------------------------------------------
with tab_correlation:
    st.subheader("Correlation")
    st.caption("Returns correlation among selected assets over the chosen date range.")

    corr_keys = st.multiselect(
        "Assets",
        options=[k for k in ASSETS.keys() if get_asset(k) is not None],
        default=(selected_keys[:6] if selected_keys else ["SPX", "NDX", "DXY", "XAU", "TLT", "VIX"]),
    )

    corr_use_log = st.checkbox("Use log returns", value=True)
    corr_min_obs = st.slider("Minimum observations", 30, 500, 60)

    if len(corr_keys) < 2:
        st.info("Select at least two assets for correlation.")
    else:
        with st.spinner("Computing correlation..."):
            series_list = []
            labels = []
            for k in corr_keys:
                a = get_asset(k)
                if not a:
                    continue
                try:
                    df = fetch_asset(a, yf_interval, start_dt, end_dt)
                    c = load_close_series(k, df).rename(k).dropna()
                    if not c.empty:
                        series_list.append(c)
                        labels.append(k)
                except Exception as e:
                    push_error(f"Correlation {k}: {e}")

            if len(series_list) < 2:
                st.warning("Not enough data series fetched to compute correlation.")
            else:
                aligned = _align_series(series_list)
                if aligned.shape[0] < corr_min_obs:
                    st.warning(f"Only {aligned.shape[0]} aligned observations. Consider broadening your date range.")
                if corr_use_log:
                    rets = np.log(aligned).diff()
                else:
                    rets = aligned.pct_change()

                rets = rets.dropna()
                corr = rets.corr()

                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        colorbar=dict(title="Corr"),
                        zmin=-1,
                        zmax=1,
                    )
                )
                fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10), title="Returns Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Returns (latest few rows)**")
                st.dataframe(rets.tail(12), use_container_width=True)


# -----------------------------------------------------------------------------
# Watchlists tab
# -----------------------------------------------------------------------------
with tab_watchlists:
    st.subheader("Watchlists")
    st.caption("Edit watchlists and keep your macro dashboard organized.")

    wl_names = list(st.session_state.custom_watchlists.keys())
    wl = st.selectbox("Watchlist", wl_names, index=wl_names.index(category) if category in wl_names else 0)

    current = st.session_state.custom_watchlists.get(wl, [])
    # keep only valid keys
    valid_keys = [k for k in current if get_asset(k) is not None]

    st.markdown("**Current keys**")
    st.code(", ".join(valid_keys) if valid_keys else "(empty)", language="text")

    st.markdown("**Edit**")
    add_keys = st.multiselect("Add assets", options=sorted(list(ASSETS.keys())), default=[])
    remove_keys = st.multiselect("Remove assets", options=valid_keys, default=[])

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Apply changes"):
            new_list = [k for k in valid_keys if k not in remove_keys]
            for k in add_keys:
                if k not in new_list:
                    new_list.append(k)
            st.session_state.custom_watchlists[wl] = new_list
            st.success("Watchlist updated.")
    with c2:
        new_name = st.text_input("Create new watchlist", value="")
        if st.button("Create"):
            if new_name.strip():
                if new_name in st.session_state.custom_watchlists:
                    st.error("That name already exists.")
                else:
                    st.session_state.custom_watchlists[new_name.strip()] = []
                    st.success("Created.")
            else:
                st.error("Enter a watchlist name.")
    with c3:
        if st.button("Reset ALL to defaults"):
            st.session_state.custom_watchlists = WATCHLISTS.copy()
            st.session_state.custom_assets = {}
            st.success("Reset done.")


# -----------------------------------------------------------------------------
# Footer / Provider notes
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Notes: Some symbols (especially DXY) can vary by data feed. If a chart is empty, use the Asset Editor (sidebar → Tools) "
    "to swap the ticker (e.g., DXY alternatives: DX-Y.NYB, DX=F). FRED series are fetched via fredgraph CSV endpoints."
)
