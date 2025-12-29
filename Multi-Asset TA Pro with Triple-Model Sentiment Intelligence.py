# app.py
# =============================================================================
# Multi-Asset TA Pro with Triple-Model Sentiment Intelligence
# =============================================================================
# REQUIRED INSTALLS (run once):
#   pip install streamlit yfinance plotly pandas numpy transformers torch vaderSentiment textblob wordcloud matplotlib nltk requests
#
# SECRETS (recommended): create .streamlit/secrets.toml
# -----------------------------------------------------------------------------
# # Optional X (Twitter) API v2 Bearer Token (for recent search)
# X_BEARER_TOKEN = "YOUR_X_BEARER_TOKEN"
#
# # Optional Telegram broadcasting (alerts/export sharing)
# TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
# TELEGRAM_CHAT_ID   = "YOUR_TELEGRAM_CHAT_ID"
#
# Notes:
# - This app NEVER hardcodes credentials. It reads from st.secrets only.
# - If X_BEARER_TOKEN is missing, the app falls back to yfinance news headlines (optional).
# =============================================================================

import os
import re
import math
import time
import json
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import torch
from transformers import pipeline

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

# =============================================================================
# STREAMLIT CONFIG
# =============================================================================
st.set_page_config(
    page_title="Multi-Asset TA Pro with Triple-Model Sentiment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# GLOBALS / CONSTANTS
# =============================================================================
APP_TITLE = "📈 Multi-Asset TA Pro with Triple-Model Sentiment Intelligence"

INTERVALS = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m (1h)": "60m",
    "90m": "90m",
    "1d": "1d",
    "5d": "5d",
    "1wk": "1wk",
    "1mo": "1mo",
    "3mo": "3mo",
}

# Reasonable default lookbacks per interval (yfinance constraints vary)
DEFAULT_LOOKBACK_DAYS = {
    "1m": 7,
    "2m": 30,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "90m": 730,
    "1d": 3650,
    "5d": 3650,
    "1wk": 3650,
    "1mo": 3650,
    "3mo": 3650,
}

TRANSFORMER_PRIMARY = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
TRANSFORMER_FALLBACKS = [
    "ProsusAI/finbert",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
]

SENTIMENT_MODES = [
    "Hybrid (All Three)",
    "Transformer Only",
    "VADER Only",
    "TextBlob Only",
    "Custom Weights",
]

INDICATOR_OPTIONS = [
    "SMA",
    "EMA",
    "Bollinger Bands",
    "MACD",
    "RSI",
    "Stochastic",
    "Ichimoku",
    "Parabolic SAR",
    "ADX",
    "CCI",
    "ATR",
    "Supertrend",
    "VWAP",
    "Fibonacci (Auto Swing)",
    "Pivot Points (Classic)",
    "Volume Profile (Basic)",
]

STRATEGY_OPTIONS = [
    "None",
    "MA Crossover (EMA Fast/Slow)",
    "RSI Mean Reversion",
    "MACD Trend",
    "Supertrend Trend",
    "Sentiment: Extreme Sentiment Reversal",
    "Sentiment: Sentiment Momentum",
    "Sentiment: High Subjectivity + Oversold",
    "Sentiment: Consensus Signal",
]

# =============================================================================
# HELPERS: SECRETS
# =============================================================================
def get_secret(key: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(key, default)).strip()
    except Exception:
        return default


# =============================================================================
# CACHING: NLTK
# =============================================================================
@st.cache_resource(show_spinner=False)
def ensure_nltk() -> bool:
    """
    Downloads NLTK stopwords once (if missing).
    Cached as a resource to avoid repeated downloads.
    """
    try:
        nltk.data.find("corpora/stopwords")
        return True
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            return True
        except Exception:
            return False


# =============================================================================
# DATA FETCH: PRICE
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(
    ticker: str,
    start: dt.datetime,
    end: dt.datetime,
    interval: str,
) -> pd.DataFrame:
    """
    Fetch OHLCV using yfinance with caching.
    Includes robust cleanup + standardized columns.
    """
    t = ticker.strip()
    if not t:
        return pd.DataFrame()

    try:
        df = yf.download(
            tickers=t,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            prepost=False,
            actions=False,
            progress=False,
            threads=True,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        # yfinance sometimes returns multiindex cols for multiple tickers
        if isinstance(df.columns, pd.MultiIndex):
            # pick the ticker if present; else flatten
            if t in df.columns.get_level_values(0):
                df = df[t].copy()
            else:
                df.columns = ["_".join(map(str, c)).strip() for c in df.columns.values]

        df = df.rename(columns={c: c.title() for c in df.columns})
        # Ensure required columns
        for col in ["Open", "High", "Low", "Close"]:
            if col not in df.columns:
                return pd.DataFrame()

        if "Volume" not in df.columns:
            df["Volume"] = np.nan

        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Remove duplicates
        df = df[~df.index.duplicated(keep="last")]

        return df
    except Exception:
        return pd.DataFrame()


# =============================================================================
# TECHNICAL INDICATORS (PURE PANDAS / NUMPY)
# =============================================================================
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def bollinger_bands(close: pd.Series, length: int = 20, mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, length)
    std = close.rolling(length, min_periods=length).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return lower, mid, upper


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    h = m - s
    return m, s, h


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    k_pct = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    d_pct = k_pct.rolling(d, min_periods=d).mean()
    return k_pct, d_pct


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    ADX with +DI and -DI (Wilder's smoothing via EWM alpha=1/length).
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)

    atr_w = pd.Series(tr, index=high.index).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    plus_dm_w = pd.Series(plus_dm, index=high.index).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    minus_dm_w = pd.Series(minus_dm, index=high.index).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    plus_di = 100 * (plus_dm_w / atr_w.replace(0, np.nan))
    minus_di = 100 * (minus_dm_w / atr_w.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    return adx_val, plus_di, minus_di


def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(length, min_periods=length).mean()
    mad = (tp - sma_tp).abs().rolling(length, min_periods=length).mean()
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def ichimoku(high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
    """
    Classic Ichimoku:
    - Tenkan (9)
    - Kijun (26)
    - Senkou A (shift +26)
    - Senkou B (52, shift +26)
    - Chikou (close shifted -26) handled elsewhere if needed
    """
    tenkan = (high.rolling(9, min_periods=9).max() + low.rolling(9, min_periods=9).min()) / 2.0
    kijun = (high.rolling(26, min_periods=26).max() + low.rolling(26, min_periods=26).min()) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(26)
    senkou_b = ((high.rolling(52, min_periods=52).max() + low.rolling(52, min_periods=52).min()) / 2.0).shift(26)
    return {"tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a, "senkou_b": senkou_b}


def parabolic_sar(high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """
    Parabolic SAR (vectorized-ish but requires iterative state).
    Implemented with a tight loop over rows (fast enough for typical yfinance sizes).
    """
    if high.empty:
        return pd.Series(dtype=float)

    h = high.values
    l = low.values
    n = len(high)

    sar = np.full(n, np.nan, dtype=float)

    # Initialize trend by first two candles
    uptrend = True
    if n >= 2 and (h[1] + l[1]) / 2 < (h[0] + l[0]) / 2:
        uptrend = False

    af = step
    ep = h[0] if uptrend else l[0]
    sar[0] = l[0] if uptrend else h[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        sar_i = prev_sar + af * (ep - prev_sar)

        # Clamp SAR to prior extremes
        if uptrend:
            sar_i = min(sar_i, l[i - 1], l[i] if i - 2 < 0 else l[i - 2])
        else:
            sar_i = max(sar_i, h[i - 1], h[i] if i - 2 < 0 else h[i - 2])

        # Reversal check
        if uptrend:
            if l[i] < sar_i:
                uptrend = False
                sar_i = ep
                af = step
                ep = l[i]
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + step, max_step)
        else:
            if h[i] > sar_i:
                uptrend = True
                sar_i = ep
                af = step
                ep = h[i]
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + step, max_step)

        sar[i] = sar_i

    return pd.Series(sar, index=high.index, name="PSAR")


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP using typical price. Works best intraday but computed generically.
    Uses cumulative volume and typical price*volume.
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].fillna(0.0)
    pv = (tp * vol).cumsum()
    vv = vol.cumsum().replace(0, np.nan)
    return pv / vv


def supertrend(df: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Supertrend:
    - Returns: (supertrend_line, direction) where direction is 1 bull, -1 bear
    """
    atr_v = atr(df["High"], df["Low"], df["Close"], length=length)
    hl2 = (df["High"] + df["Low"]) / 2.0

    upperband = hl2 + multiplier * atr_v
    lowerband = hl2 - multiplier * atr_v

    st_line = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    # Iterative state
    for i in range(len(df)):
        if i == 0:
            st_line.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = 1
            continue

        prev_st = st_line.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]

        curr_close = df["Close"].iloc[i]
        prev_close = df["Close"].iloc[i - 1]

        # Final upper/lower bands
        curr_upper = upperband.iloc[i]
        curr_lower = lowerband.iloc[i]

        prev_upper = upperband.iloc[i - 1]
        prev_lower = lowerband.iloc[i - 1]

        final_upper = curr_upper if (curr_upper < prev_upper) or (prev_close > prev_upper) else prev_upper
        final_lower = curr_lower if (curr_lower > prev_lower) or (prev_close < prev_lower) else prev_lower

        # Direction + supertrend value
        if prev_st == prev_upper:
            if curr_close <= final_upper:
                st_line.iloc[i] = final_upper
                direction.iloc[i] = -1
            else:
                st_line.iloc[i] = final_lower
                direction.iloc[i] = 1
        else:  # prev_st == prev_lower
            if curr_close >= final_lower:
                st_line.iloc[i] = final_lower
                direction.iloc[i] = 1
            else:
                st_line.iloc[i] = final_upper
                direction.iloc[i] = -1

        # update stored bands to match st_line regime (not strictly required)
        upperband.iloc[i] = final_upper
        lowerband.iloc[i] = final_lower

    st_line.name = "Supertrend"
    direction.name = "SupertrendDir"
    return st_line, direction


def pivot_points_classic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classic daily pivot points from previous period (uses shift(1)).
    P, R1,R2,R3,S1,S2,S3
    """
    high = df["High"].shift(1)
    low = df["Low"].shift(1)
    close = df["Close"].shift(1)

    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2 * (p - low)
    s3 = low - 2 * (high - p)

    out = pd.DataFrame(
        {"P": p, "R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3},
        index=df.index,
    )
    return out


def fibonacci_auto_swing(df: pd.DataFrame, lookback: int = 200) -> Dict[str, float]:
    """
    Auto swing fib levels based on highest high / lowest low over lookback window.
    Returns dict of levels at last bar.
    """
    if df.empty:
        return {}
    d = df.tail(lookback)
    swing_high = float(d["High"].max())
    swing_low = float(d["Low"].min())
    if swing_high == swing_low:
        return {}

    # Determine direction by last close relative to midrange (simple heuristic)
    last_close = float(df["Close"].iloc[-1])
    mid = (swing_high + swing_low) / 2.0
    uptrend = last_close >= mid

    # If uptrend, fib retracements from low->high; else high->low
    if uptrend:
        a, b = swing_low, swing_high
    else:
        a, b = swing_high, swing_low

    diff = (b - a)
    levels = {
        "0.0%": b,
        "23.6%": b - 0.236 * diff,
        "38.2%": b - 0.382 * diff,
        "50.0%": b - 0.5 * diff,
        "61.8%": b - 0.618 * diff,
        "78.6%": b - 0.786 * diff,
        "100.0%": a,
        "swing_high": swing_high,
        "swing_low": swing_low,
    }
    return levels


def volume_profile_basic(df: pd.DataFrame, bins: int = 36) -> pd.DataFrame:
    """
    Basic Volume Profile:
    - Bin closes into price buckets.
    - Sum volume by bucket.
    Returns DataFrame: bucket_mid, volume
    """
    if df.empty:
        return pd.DataFrame(columns=["price", "volume"])

    closes = df["Close"].astype(float)
    vols = df["Volume"].fillna(0.0).astype(float)

    if closes.nunique() < 2:
        return pd.DataFrame(columns=["price", "volume"])

    pmin, pmax = closes.min(), closes.max()
    if pmin == pmax:
        return pd.DataFrame(columns=["price", "volume"])

    edges = np.linspace(pmin, pmax, bins + 1)
    bucket = np.digitize(closes.values, edges) - 1
    bucket = np.clip(bucket, 0, bins - 1)

    vol_by_bucket = pd.Series(vols.values).groupby(bucket).sum()
    mids = (edges[:-1] + edges[1:]) / 2.0

    vp = pd.DataFrame({"price": mids, "volume": [vol_by_bucket.get(i, 0.0) for i in range(bins)]})
    vp = vp.sort_values("price")
    return vp


# =============================================================================
# SIGNAL ENGINE (TA + SENTIMENT)
# =============================================================================
def compute_ta_bundle(df: pd.DataFrame, params: Dict) -> Dict[str, pd.Series]:
    """
    Computes selected TA indicators and returns a dict of series.
    This keeps plotting and strategies clean.
    """
    out: Dict[str, pd.Series] = {}
    if df.empty:
        return out

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Base MA params
    sma_len = int(params.get("sma_len", 20))
    ema_fast = int(params.get("ema_fast", 12))
    ema_slow = int(params.get("ema_slow", 26))

    # SMA / EMA (always computed if requested)
    out["SMA"] = sma(close, sma_len)
    out["EMA_FAST"] = ema(close, ema_fast)
    out["EMA_SLOW"] = ema(close, ema_slow)

    # Bollinger
    bb_len = int(params.get("bb_len", 20))
    bb_mult = float(params.get("bb_mult", 2.0))
    bb_l, bb_m, bb_u = bollinger_bands(close, bb_len, bb_mult)
    out["BB_L"] = bb_l
    out["BB_M"] = bb_m
    out["BB_U"] = bb_u

    # MACD
    macd_fast = int(params.get("macd_fast", 12))
    macd_slow = int(params.get("macd_slow", 26))
    macd_sig = int(params.get("macd_signal", 9))
    m, s, h = macd(close, macd_fast, macd_slow, macd_sig)
    out["MACD"] = m
    out["MACD_SIGNAL"] = s
    out["MACD_HIST"] = h

    # RSI
    rsi_len = int(params.get("rsi_len", 14))
    out["RSI"] = rsi(close, rsi_len)

    # Stochastic
    st_k = int(params.get("stoch_k", 14))
    st_d = int(params.get("stoch_d", 3))
    k_pct, d_pct = stochastic(high, low, close, st_k, st_d)
    out["STOCH_K"] = k_pct
    out["STOCH_D"] = d_pct

    # Ichimoku
    ich = ichimoku(high, low)
    out["ICH_TENKAN"] = ich["tenkan"]
    out["ICH_KIJUN"] = ich["kijun"]
    out["ICH_SENKOU_A"] = ich["senkou_a"]
    out["ICH_SENKOU_B"] = ich["senkou_b"]

    # PSAR
    psar_step = float(params.get("psar_step", 0.02))
    psar_max = float(params.get("psar_max", 0.2))
    out["PSAR"] = parabolic_sar(high, low, psar_step, psar_max)

    # ADX
    adx_len = int(params.get("adx_len", 14))
    adx_v, pdi, mdi = adx(high, low, close, adx_len)
    out["ADX"] = adx_v
    out["+DI"] = pdi
    out["-DI"] = mdi

    # CCI
    cci_len = int(params.get("cci_len", 20))
    out["CCI"] = cci(high, low, close, cci_len)

    # ATR
    atr_len = int(params.get("atr_len", 14))
    out["ATR"] = atr(high, low, close, atr_len)

    # Supertrend
    st_len = int(params.get("supertrend_len", 10))
    st_mult = float(params.get("supertrend_mult", 3.0))
    st_line, st_dir = supertrend(df, st_len, st_mult)
    out["SUPERTREND"] = st_line
    out["SUPERTREND_DIR"] = st_dir

    # VWAP
    out["VWAP"] = vwap(df)

    return out


def ta_strategy_signal(df: pd.DataFrame, ta: Dict[str, pd.Series], strategy: str, params: Dict) -> Tuple[str, str]:
    """
    Returns (signal, rationale).
    Signals: BUY / SELL / NEUTRAL
    """
    if df.empty or not ta:
        return "NEUTRAL", "No data."

    close = df["Close"]
    last = df.index[-1]

    def safe_last(s: pd.Series) -> float:
        try:
            return float(s.loc[last])
        except Exception:
            return float("nan")

    # Default
    signal = "NEUTRAL"
    why = "No strategy selected."

    if strategy == "None":
        return signal, why

    if strategy == "MA Crossover (EMA Fast/Slow)":
        fast = ta.get("EMA_FAST")
        slow = ta.get("EMA_SLOW")
        if fast is None or slow is None:
            return "NEUTRAL", "EMA series missing."

        # Cross detection
        cross_up = (fast.shift(1) <= slow.shift(1)) & (fast > slow)
        cross_dn = (fast.shift(1) >= slow.shift(1)) & (fast < slow)

        if bool(cross_up.iloc[-1]):
            return "BUY", f"EMA fast crossed above EMA slow (fast={safe_last(fast):.4f}, slow={safe_last(slow):.4f})."
        if bool(cross_dn.iloc[-1]):
            return "SELL", f"EMA fast crossed below EMA slow (fast={safe_last(fast):.4f}, slow={safe_last(slow):.4f})."
        return "NEUTRAL", f"No crossover. fast={safe_last(fast):.4f}, slow={safe_last(slow):.4f}."

    if strategy == "RSI Mean Reversion":
        r = ta.get("RSI")
        if r is None:
            return "NEUTRAL", "RSI missing."
        overbought = float(params.get("rsi_overbought", 70))
        oversold = float(params.get("rsi_oversold", 30))
        r_last = safe_last(r)
        if r_last < oversold:
            return "BUY", f"RSI oversold ({r_last:.2f} < {oversold})."
        if r_last > overbought:
            return "SELL", f"RSI overbought ({r_last:.2f} > {overbought})."
        return "NEUTRAL", f"RSI neutral ({r_last:.2f})."

    if strategy == "MACD Trend":
        m = ta.get("MACD")
        s = ta.get("MACD_SIGNAL")
        if m is None or s is None:
            return "NEUTRAL", "MACD missing."
        cross_up = (m.shift(1) <= s.shift(1)) & (m > s)
        cross_dn = (m.shift(1) >= s.shift(1)) & (m < s)
        if bool(cross_up.iloc[-1]):
            return "BUY", f"MACD crossed above signal (MACD={safe_last(m):.4f}, Sig={safe_last(s):.4f})."
        if bool(cross_dn.iloc[-1]):
            return "SELL", f"MACD crossed below signal (MACD={safe_last(m):.4f}, Sig={safe_last(s):.4f})."
        return "NEUTRAL", f"MACD no cross (MACD={safe_last(m):.4f}, Sig={safe_last(s):.4f})."

    if strategy == "Supertrend Trend":
        st_dir = ta.get("SUPERTREND_DIR")
        st_line = ta.get("SUPERTREND")
        if st_dir is None or st_line is None:
            return "NEUTRAL", "Supertrend missing."
        d_last = safe_last(st_dir)
        if d_last > 0:
            return "BUY", f"Supertrend bullish (line={safe_last(st_line):.4f})."
        if d_last < 0:
            return "SELL", f"Supertrend bearish (line={safe_last(st_line):.4f})."
        return "NEUTRAL", "Supertrend neutral."

    return "NEUTRAL", "Strategy not recognized."


# =============================================================================
# SENTIMENT: FETCH SOURCES
# =============================================================================
def minimal_clean_keep_emotion(text: str) -> str:
    """
    Minimal cleaning:
    - Remove URLs
    - Remove excessive whitespace
    IMPORTANT: Preserve emojis, capitalization, punctuation for VADER/TextBlob signals.
    """
    t = text or ""
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@st.cache_data(ttl=900, show_spinner=False)
def fetch_x_posts_v2(query: str, max_results: int) -> pd.DataFrame:
    """
    Fetch recent X (Twitter) posts via API v2 recent search.
    Requires X_BEARER_TOKEN in st.secrets.
    Returns DataFrame with columns: id, created_at, text
    """
    bearer = get_secret("X_BEARER_TOKEN", "")
    if not bearer:
        return pd.DataFrame()

    # X API limits: max_results per request is 10-100. We'll paginate up to max_results.
    max_results = int(np.clip(max_results, 10, 500))
    per_page = 100 if max_results >= 100 else max_results

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer}"}

    params = {
        "query": query,
        "max_results": per_page,
        "tweet.fields": "created_at,lang,public_metrics",
    }

    all_rows = []
    next_token = None
    remaining = max_results

    while remaining > 0:
        params["max_results"] = min(per_page, remaining)
        if next_token:
            params["next_token"] = next_token
        else:
            params.pop("next_token", None)

        try:
            r = requests.get(url, headers=headers, params=params, timeout=20)
            if r.status_code != 200:
                break
            payload = r.json()
            data = payload.get("data", [])
            for d in data:
                all_rows.append(
                    {
                        "id": d.get("id", ""),
                        "created_at": d.get("created_at", None),
                        "text": d.get("text", ""),
                    }
                )
            meta = payload.get("meta", {})
            next_token = meta.get("next_token", None)
            fetched = len(data)
            remaining -= fetched
            if fetched == 0 or not next_token:
                break
        except Exception:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["text"] = df["text"].astype(str).map(minimal_clean_keep_emotion)
    df = df.dropna(subset=["created_at"])
    df = df[df["text"].str.len() > 0]
    df = df.sort_values("created_at")
    df = df.reset_index(drop=True)
    return df


@st.cache_data(ttl=900, show_spinner=False)
def fetch_yfinance_news_headlines(ticker: str, limit: int = 50) -> pd.DataFrame:
    """
    Optional fallback: yfinance news headlines for the selected ticker.
    Returns DataFrame: created_at (best-effort), text
    """
    t = ticker.strip()
    if not t:
        return pd.DataFrame()

    try:
        tk = yf.Ticker(t)
        items = getattr(tk, "news", None)
        if not items:
            return pd.DataFrame()

        rows = []
        for it in items[: int(limit)]:
            title = str(it.get("title", "")).strip()
            if not title:
                continue
            ts = it.get("providerPublishTime", None)
            if ts is not None:
                created = pd.to_datetime(int(ts), unit="s", utc=True, errors="coerce")
            else:
                created = pd.Timestamp.utcnow().tz_localize("UTC")
            rows.append({"created_at": created, "text": minimal_clean_keep_emotion(title)})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).dropna(subset=["created_at"])
        df = df.sort_values("created_at").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


# =============================================================================
# SENTIMENT: MODELS (LOAD + RUN)
# =============================================================================
@dataclass
class TransformerLoadResult:
    ok: bool
    model_name: str
    error: str = ""


@st.cache_resource(show_spinner=False)
def load_transformer_pipeline(model_name: str):
    """
    Load a transformers pipeline (cached).
    """
    device = 0 if torch.cuda.is_available() else -1
    # Some pipelines use "text-classification" and return label/score.
    # We keep truncation to handle long social posts.
    return pipeline(
        task="text-classification",
        model=model_name,
        tokenizer=model_name,
        device=device,
        truncation=True,
        top_k=None,  # keep default output; some models return one label, some return multiple
    )


def try_load_transformer() -> Tuple[Optional[object], TransformerLoadResult]:
    """
    Attempt primary, then fallbacks.
    Returns (pipeline_or_none, load_result).
    """
    # Primary
    candidates = [TRANSFORMER_PRIMARY] + TRANSFORMER_FALLBACKS
    last_err = ""
    for name in candidates:
        try:
            pipe = load_transformer_pipeline(name)
            return pipe, TransformerLoadResult(ok=True, model_name=name, error="")
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
    return None, TransformerLoadResult(ok=False, model_name="", error=last_err)


def run_transformer_sentiment(pipe, texts: List[str]) -> pd.DataFrame:
    """
    Run transformer sentiment on texts.
    Returns DataFrame with columns:
      - transformer_label
      - transformer_conf
      - transformer_score_norm  (-1..+1)
    Normalization logic:
      - label -> sign; multiply by confidence score; neutral -> 0
    """
    if pipe is None or not texts:
        return pd.DataFrame(columns=["transformer_label", "transformer_conf", "transformer_score_norm"])

    # Batch inference
    try:
        outputs = pipe(texts, batch_size=16)
    except Exception:
        # fallback with smaller batch
        outputs = pipe(texts, batch_size=4)

    # outputs shape varies:
    # - Some models: [{'label': 'positive', 'score': 0.97}, ...]
    # - Some models: [[{'label': 'NEGATIVE', 'score': 0.9}, {'label': 'POSITIVE', 'score': 0.1}], ...]
    labels = []
    confs = []
    scores = []

    for out in outputs:
        label = "neutral"
        conf = 0.0

        if isinstance(out, list):
            # choose highest score
            best = max(out, key=lambda x: x.get("score", 0.0)) if out else {"label": "neutral", "score": 0.0}
            label = str(best.get("label", "neutral")).lower()
            conf = float(best.get("score", 0.0))
        elif isinstance(out, dict):
            label = str(out.get("label", "neutral")).lower()
            conf = float(out.get("score", 0.0))
        else:
            label = "neutral"
            conf = 0.0

        # Normalize
        # Robust label mapping
        if "pos" in label:
            s = +1.0 * conf
            label_std = "positive"
        elif "neg" in label:
            s = -1.0 * conf
            label_std = "negative"
        elif "neu" in label:
            s = 0.0
            label_std = "neutral"
        else:
            # Unknown label: treat as neutral with 0
            s = 0.0
            label_std = "neutral"

        labels.append(label_std)
        confs.append(conf)
        scores.append(float(np.clip(s, -1.0, 1.0)))

    return pd.DataFrame(
        {
            "transformer_label": labels,
            "transformer_conf": confs,
            "transformer_score_norm": scores,
        }
    )


def run_vader_sentiment(texts: List[str]) -> pd.DataFrame:
    """
    VADER sentiment:
      - compound already in [-1,1]
      - pos/neu/neg proportions
    """
    if not texts:
        return pd.DataFrame(columns=["vader_compound", "vader_pos", "vader_neu", "vader_neg"])

    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for t in texts:
        s = analyzer.polarity_scores(t)
        rows.append(
            {
                "vader_compound": float(s.get("compound", 0.0)),
                "vader_pos": float(s.get("pos", 0.0)),
                "vader_neu": float(s.get("neu", 0.0)),
                "vader_neg": float(s.get("neg", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def run_textblob_sentiment(texts: List[str]) -> pd.DataFrame:
    """
    TextBlob sentiment:
      - polarity in [-1,1]
      - subjectivity in [0,1]
    """
    if not texts:
        return pd.DataFrame(columns=["tb_polarity", "tb_subjectivity"])

    rows = []
    for t in texts:
        b = TextBlob(t)
        rows.append(
            {
                "tb_polarity": float(b.sentiment.polarity),
                "tb_subjectivity": float(b.sentiment.subjectivity),
            }
        )
    return pd.DataFrame(rows)


def compute_ensemble(
    transformer_ok: bool,
    df_scored: pd.DataFrame,
    mode: str,
    w_t: float,
    w_v: float,
    w_b: float,
) -> pd.DataFrame:
    """
    Compute final sentiment score:
      - Normalize each model to [-1,1] (transformer_score_norm, vader_compound, tb_polarity)
      - Weighted average
    Fallback logic:
      - If transformer fails: redistribute its weight proportionally to VADER/TextBlob.
    """
    out = df_scored.copy()

    # Base normalized columns expected:
    # - transformer_score_norm (may be missing)
    # - vader_compound
    # - tb_polarity
    for col in ["transformer_score_norm", "vader_compound", "tb_polarity", "tb_subjectivity"]:
        if col not in out.columns:
            out[col] = np.nan

    # Mode overrides
    if mode == "Transformer Only":
        w_t, w_v, w_b = 1.0, 0.0, 0.0
    elif mode == "VADER Only":
        w_t, w_v, w_b = 0.0, 1.0, 0.0
    elif mode == "TextBlob Only":
        w_t, w_v, w_b = 0.0, 0.0, 1.0
    elif mode == "Hybrid (All Three)":
        w_t, w_v, w_b = 0.40, 0.35, 0.25
    else:
        # Custom Weights uses provided w_t,w_v,w_b
        pass

    # Fallback: if transformer not available, reweight
    if not transformer_ok:
        # redistribute w_t to others proportional to their weights, default if both are zero
        remaining = w_v + w_b
        if remaining <= 0:
            w_v, w_b = 0.6, 0.4
            remaining = 1.0
        w_v = w_v + w_t * (w_v / remaining)
        w_b = w_b + w_t * (w_b / remaining)
        w_t = 0.0

    # Normalize to sum 1
    total = w_t + w_v + w_b
    if total <= 0:
        w_t, w_v, w_b = 0.0, 0.6, 0.4
        total = 1.0

    w_t, w_v, w_b = w_t / total, w_v / total, w_b / total

    out["w_transformer"] = w_t
    out["w_vader"] = w_v
    out["w_textblob"] = w_b

    # Weighted score
    out["hybrid_score"] = (
        out["transformer_score_norm"].fillna(0.0) * w_t
        + out["vader_compound"].fillna(0.0) * w_v
        + out["tb_polarity"].fillna(0.0) * w_b
    ).clip(-1.0, 1.0)

    # Classification thresholds
    def classify(x: float) -> str:
        if x > 0.5:
            return "Strong Positive"
        if x > 0.05:
            return "Positive"
        if x < -0.5:
            return "Strong Negative"
        if x < -0.05:
            return "Negative"
        return "Neutral"

    out["hybrid_class"] = out["hybrid_score"].map(lambda v: classify(float(v)))
    out["avg_subjectivity"] = out["tb_subjectivity"].astype(float)

    return out


@st.cache_data(ttl=900, show_spinner=False)
def analyze_sentiment_batch(
    source_df: pd.DataFrame,
    mode: str,
    w_t: float,
    w_v: float,
    w_b: float,
) -> Tuple[pd.DataFrame, Dict]:
    """
    End-to-end sentiment scoring for a batch of posts/headlines.
    Returns:
      - scored DataFrame with per-item scores
      - meta dict with model status + summary distributions
    """
    meta = {
        "transformer_ok": False,
        "transformer_model": "",
        "transformer_error": "",
        "n_items": 0,
    }

    if source_df is None or source_df.empty or "text" not in source_df.columns:
        return pd.DataFrame(), meta

    df = source_df.copy()
    df["text"] = df["text"].astype(str).map(minimal_clean_keep_emotion)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    meta["n_items"] = int(len(df))
    if len(df) == 0:
        return pd.DataFrame(), meta

    texts = df["text"].tolist()

    # Transformer load + run (best-effort)
    pipe, load_res = try_load_transformer()
    meta["transformer_ok"] = bool(load_res.ok)
    meta["transformer_model"] = load_res.model_name
    meta["transformer_error"] = load_res.error

    # Run models
    tf_df = pd.DataFrame()
    if pipe is not None:
        try:
            tf_df = run_transformer_sentiment(pipe, texts)
        except Exception:
            tf_df = pd.DataFrame(columns=["transformer_label", "transformer_conf", "transformer_score_norm"])
            meta["transformer_ok"] = False

    vader_df = run_vader_sentiment(texts)
    blob_df = run_textblob_sentiment(texts)

    scored = pd.concat([df.reset_index(drop=True), tf_df, vader_df, blob_df], axis=1)
    scored = compute_ensemble(meta["transformer_ok"], scored, mode, w_t, w_v, w_b)

    # Model agreement direction (+1 / 0 / -1)
    def dir_from_val(x: float, thr: float = 0.05) -> int:
        if x > thr:
            return 1
        if x < -thr:
            return -1
        return 0

    scored["dir_transformer"] = scored["transformer_score_norm"].fillna(0.0).map(lambda v: dir_from_val(float(v), 0.05))
    scored["dir_vader"] = scored["vader_compound"].fillna(0.0).map(lambda v: dir_from_val(float(v), 0.05))
    scored["dir_textblob"] = scored["tb_polarity"].fillna(0.0).map(lambda v: dir_from_val(float(v), 0.05))

    scored["model_agreement"] = (
        (scored["dir_transformer"] == scored["dir_vader"])
        & (scored["dir_vader"] == scored["dir_textblob"])
        & (scored["dir_transformer"] != 0)
    )

    # Summary distributions
    meta["hybrid_mean"] = float(scored["hybrid_score"].mean())
    meta["hybrid_std"] = float(scored["hybrid_score"].std(ddof=0)) if len(scored) > 1 else 0.0
    meta["subjectivity_mean"] = float(scored["tb_subjectivity"].mean())
    meta["vader_intensity_mean"] = float(scored["vader_compound"].abs().mean())

    return scored, meta


def build_sentiment_timeseries(scored: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.Series:
    """
    Convert per-post sentiment into a time series aligned to price index:
      - resample posts to the nearest price frequency using asfreq + ffill
      - default aggregation: mean per time bucket
    """
    if scored is None or scored.empty or "created_at" not in scored.columns:
        return pd.Series(index=price_index, dtype=float)

    s = scored.dropna(subset=["created_at"]).copy()
    if s.empty:
        return pd.Series(index=price_index, dtype=float)

    s["created_at"] = pd.to_datetime(s["created_at"], utc=True, errors="coerce")
    s = s.dropna(subset=["created_at"])
    if s.empty:
        return pd.Series(index=price_index, dtype=float)

    # Build raw series
    raw = pd.Series(s["hybrid_score"].astype(float).values, index=s["created_at"])
    raw = raw.sort_index()

    # Determine a reasonable resample rule from price index spacing
    if len(price_index) < 3:
        return pd.Series(index=price_index, dtype=float)

    # Estimate median spacing
    diffs = pd.Series(price_index[1:] - price_index[:-1]).dt.total_seconds()
    med = float(diffs.median()) if not diffs.empty else 86400.0

    if med <= 120:  # <=2m
        rule = "1min"
    elif med <= 600:  # <=10m
        rule = "5min"
    elif med <= 1800:  # <=30m
        rule = "15min"
    elif med <= 5400:  # <=90m
        rule = "1H"
    elif med <= 90000:  # <=~1d
        rule = "1D"
    else:
        rule = "1W"

    bucket = raw.resample(rule).mean()

    # Align to price index (convert price index to UTC for match)
    px = pd.to_datetime(price_index, utc=True)
    aligned = bucket.reindex(px, method="ffill")
    aligned.index = price_index  # restore original tz-awareness/naiveness as Streamlit displays
    return aligned


def ema_smooth_series(series: pd.Series, length: int = 6) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=max(2, length // 2)).mean()


def rolling_corr(sent: pd.Series, price: pd.Series, windows: List[int]) -> pd.DataFrame:
    """
    Rolling correlation of sentiment vs price returns.
    """
    if sent.empty or price.empty:
        return pd.DataFrame()

    df = pd.DataFrame({"sent": sent.astype(float), "px": price.astype(float)})
    df = df.dropna()
    if df.empty:
        return pd.DataFrame()

    df["ret"] = df["px"].pct_change()
    # sentiment changes can also matter; but spec asks sentiment vs price returns
    out = {}
    for w in windows:
        out[f"corr_{w}"] = df["sent"].rolling(w, min_periods=max(3, w // 3)).corr(df["ret"])
    return pd.DataFrame(out, index=df.index)


# =============================================================================
# VISUALS: PLOTLY FIGURES
# =============================================================================
def plot_price_with_indicators(
    df: pd.DataFrame,
    ta: Dict[str, pd.Series],
    indicators_selected: List[str],
    overlay_tickers_data: Dict[str, pd.DataFrame],
    show_volume: bool,
    sentiment_series: Optional[pd.Series],
    show_sentiment_overlay: bool,
    show_subjectivity_overlay: bool,
    subjectivity_series: Optional[pd.Series],
) -> go.Figure:
    """
    Primary chart: Candles + overlays + optional sentiment on secondary axis + optional volume subplot.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data to plot.")
        return fig

    rows = 2 if show_volume else 1
    specs = [[{"secondary_y": True}]] + ([ [{"secondary_y": False}] ] if show_volume else [])
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=specs,
        row_heights=[0.72, 0.28] if show_volume else [1.0],
    )

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # Overlays: other tickers (normalized)
    for tkr, odf in overlay_tickers_data.items():
        if odf is None or odf.empty:
            continue
        # align
        s = odf["Close"].reindex(df.index).ffill()
        base = float(s.dropna().iloc[0]) if not s.dropna().empty else np.nan
        if not np.isfinite(base) or base == 0:
            continue
        norm = (s / base) * float(df["Close"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=norm,
                mode="lines",
                name=f"Overlay {tkr} (norm)",
                opacity=0.7,
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    # Indicator overlays on price panel
    if "SMA" in indicators_selected and "SMA" in ta:
        fig.add_trace(go.Scatter(x=df.index, y=ta["SMA"], mode="lines", name="SMA"), row=1, col=1, secondary_y=False)

    if "EMA" in indicators_selected:
        if "EMA_FAST" in ta:
            fig.add_trace(go.Scatter(x=df.index, y=ta["EMA_FAST"], mode="lines", name="EMA Fast"), row=1, col=1, secondary_y=False)
        if "EMA_SLOW" in ta:
            fig.add_trace(go.Scatter(x=df.index, y=ta["EMA_SLOW"], mode="lines", name="EMA Slow"), row=1, col=1, secondary_y=False)

    if "Bollinger Bands" in indicators_selected and all(k in ta for k in ["BB_L", "BB_M", "BB_U"]):
        fig.add_trace(go.Scatter(x=df.index, y=ta["BB_U"], mode="lines", name="BB Upper", opacity=0.7), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=ta["BB_M"], mode="lines", name="BB Mid", opacity=0.7), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=ta["BB_L"], mode="lines", name="BB Lower", opacity=0.7), row=1, col=1, secondary_y=False)

    if "VWAP" in indicators_selected and "VWAP" in ta:
        fig.add_trace(go.Scatter(x=df.index, y=ta["VWAP"], mode="lines", name="VWAP"), row=1, col=1, secondary_y=False)

    if "Parabolic SAR" in indicators_selected and "PSAR" in ta:
        fig.add_trace(go.Scatter(x=df.index, y=ta["PSAR"], mode="markers", name="PSAR", marker=dict(size=4)), row=1, col=1, secondary_y=False)

    if "Supertrend" in indicators_selected and "SUPERTREND" in ta:
        fig.add_trace(go.Scatter(x=df.index, y=ta["SUPERTREND"], mode="lines", name="Supertrend"), row=1, col=1, secondary_y=False)

    # Ichimoku cloud (light overlay)
    if "Ichimoku" in indicators_selected and all(k in ta for k in ["ICH_TENKAN", "ICH_KIJUN", "ICH_SENKOU_A", "ICH_SENKOU_B"]):
        fig.add_trace(go.Scatter(x=df.index, y=ta["ICH_TENKAN"], mode="lines", name="Tenkan"), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=ta["ICH_KIJUN"], mode="lines", name="Kijun"), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=ta["ICH_SENKOU_A"], mode="lines", name="Senkou A", opacity=0.5), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=ta["ICH_SENKOU_B"], mode="lines", name="Senkou B", opacity=0.5), row=1, col=1, secondary_y=False)

    # Sentiment overlay (secondary y)
    if show_sentiment_overlay and sentiment_series is not None and not sentiment_series.empty:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sentiment_series.reindex(df.index).astype(float),
                mode="lines",
                name="Hybrid Sentiment (-1..+1)",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
        # lock sentiment axis range
        fig.update_yaxes(range=[-1, 1], row=1, col=1, secondary_y=True, title_text="Sentiment")

    if show_subjectivity_overlay and subjectivity_series is not None and not subjectivity_series.empty:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=subjectivity_series.reindex(df.index).astype(float),
                mode="lines",
                name="Subjectivity (0..1)",
                opacity=0.7,
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(range=[-1, 1], row=1, col=1, secondary_y=True, title_text="Sentiment/Subjectivity")

    # Volume subplot
    if show_volume and "Volume" in df.columns:
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.6),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=820 if show_volume else 700,
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        title="Price + Overlays + Sentiment Intelligence",
    )

    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)

    return fig


def gauge_figure(value: float, title: str, vmin: float, vmax: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            title={"text": title},
            gauge={"axis": {"range": [vmin, vmax]}},
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def radar_model_comparison(t_score: float, v_score: float, b_score: float) -> go.Figure:
    labels = ["Transformer", "VADER", "TextBlob"]
    values = [t_score, v_score, b_score]
    # close the loop
    labels2 = labels + [labels[0]]
    values2 = values + [values[0]]

    fig = go.Figure(
        data=go.Scatterpolar(r=values2, theta=labels2, fill="toself", name="Model Scores")
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
        showlegend=False,
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Model Comparison Radar (-1..+1)",
    )
    return fig


# =============================================================================
# WORDCLOUD
# =============================================================================
def render_wordcloud(texts: List[str]) -> Optional[plt.Figure]:
    ok = ensure_nltk()
    if not ok:
        return None

    sw = set(stopwords.words("english"))
    combined = " ".join([t for t in texts if isinstance(t, str) and t.strip()])
    if not combined.strip():
        return None

    # Preserve sentiment-bearing tokens; remove very short fragments and URLs already removed
    # WordCloud will handle tokenization; stopwords reduce noise.
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="black",
        stopwords=sw,
        collocations=False,
    ).generate(combined)

    fig = plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    return fig


# =============================================================================
# TELEGRAM BROADCAST (OPTIONAL)
# =============================================================================
def telegram_send_message(text: str) -> Tuple[bool, str]:
    token = get_secret("TELEGRAM_BOT_TOKEN", "")
    chat_id = get_secret("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False, "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in st.secrets."

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        r = requests.post(url, data=payload, timeout=15)
        if r.status_code != 200:
            return False, f"Telegram error {r.status_code}: {r.text[:200]}"
        return True, "Sent."
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# =============================================================================
# UI: SIDEBAR CONTROLS (SESSION STATE)
# =============================================================================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None
if "sentiment_cache_key" not in st.session_state:
    st.session_state.sentiment_cache_key = ""

st.title(APP_TITLE)
st.caption(
    "Triple-model sentiment intelligence (Transformer + VADER + TextBlob) with robust fallbacks, "
    "plus multi-asset technical analysis, strategies, overlays, exports, and optional broadcasting."
)

with st.sidebar:
    st.header("⚙️ Controls")

    # Asset selection
    st.subheader("Asset / Timeframe")
    ticker = st.text_input("Primary Ticker", value="BTC-USD", help="Any yfinance-supported ticker (crypto, stocks, FX, commodities, bonds, indices).")

    interval_label = st.selectbox("Interval", list(INTERVALS.keys()), index=list(INTERVALS.keys()).index("1d"))
    interval = INTERVALS[interval_label]

    # Date range
    today = dt.datetime.utcnow()
    default_days = DEFAULT_LOOKBACK_DAYS.get(interval, 365)
    start_default = today - dt.timedelta(days=int(default_days))
    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Start", value=start_default.date())
    with col_b:
        end_date = st.date_input("End", value=today.date())

    start_dt = dt.datetime.combine(start_date, dt.time.min)
    end_dt = dt.datetime.combine(end_date, dt.time.max)

    # Multi-asset overlay
    st.subheader("Multi-Asset Overlay")
    overlay_tickers = st.text_area(
        "Overlay tickers (comma-separated)",
        value="ETH-USD,SPY,GC=F",
        help="Optional comparisons. Overlays are normalized to the primary price scale.",
    )
    overlay_list = [x.strip() for x in overlay_tickers.split(",") if x.strip() and x.strip() != ticker.strip()]

    # Indicators
    st.subheader("Indicators")
    indicators_selected = st.multiselect("Select Indicators", INDICATOR_OPTIONS, default=["EMA", "Bollinger Bands", "RSI", "MACD", "VWAP", "Supertrend"])

    with st.expander("Advanced Indicator Settings", expanded=False):
        sma_len = st.slider("SMA Length", 5, 200, 20)
        ema_fast = st.slider("EMA Fast", 3, 100, 12)
        ema_slow = st.slider("EMA Slow", 10, 300, 26)
        bb_len = st.slider("BB Length", 5, 100, 20)
        bb_mult = st.slider("BB Mult", 1.0, 4.0, 2.0, 0.1)
        macd_fast = st.slider("MACD Fast", 3, 50, 12)
        macd_slow = st.slider("MACD Slow", 10, 100, 26)
        macd_signal = st.slider("MACD Signal", 3, 30, 9)
        rsi_len = st.slider("RSI Length", 5, 50, 14)
        rsi_oversold = st.slider("RSI Oversold", 5, 45, 30)
        rsi_overbought = st.slider("RSI Overbought", 55, 95, 70)
        stoch_k = st.slider("Stochastic K", 5, 50, 14)
        stoch_d = st.slider("Stochastic D", 2, 20, 3)
        psar_step = st.slider("PSAR Step", 0.01, 0.10, 0.02, 0.01)
        psar_max = st.slider("PSAR Max", 0.10, 0.50, 0.20, 0.01)
        adx_len = st.slider("ADX Length", 5, 50, 14)
        cci_len = st.slider("CCI Length", 5, 60, 20)
        atr_len = st.slider("ATR Length", 5, 60, 14)
        supertrend_len = st.slider("Supertrend Length", 5, 50, 10)
        supertrend_mult = st.slider("Supertrend Mult", 1.0, 6.0, 3.0, 0.1)
        vp_bins = st.slider("Volume Profile Bins", 12, 100, 36)
        fib_lookback = st.slider("Fib Swing Lookback Bars", 50, 600, 200)

    # Strategy
    st.subheader("Strategy Center")
    strategy = st.selectbox("Strategy", STRATEGY_OPTIONS, index=STRATEGY_OPTIONS.index("None"))
    st.caption("Signals are computed on the latest bar using selected TA + sentiment (when applicable).")

    # Sentiment controls
    st.subheader("Sentiment Intelligence")
    sentiment_mode = st.selectbox("Mode", SENTIMENT_MODES, index=SENTIMENT_MODES.index("Hybrid (All Three)"))

    with st.expander("Custom Weights (Advanced)", expanded=(sentiment_mode == "Custom Weights")):
        w_t_pct = st.slider("Transformer Weight %", 0, 100, 40)
        w_v_pct = st.slider("VADER Weight %", 0, 100, 35)
        w_b_pct = st.slider("TextBlob Weight %", 0, 100, 25)
        total_w = max(1, w_t_pct + w_v_pct + w_b_pct)
        st.write(f"Sum: **{w_t_pct + w_v_pct + w_b_pct}%** (auto-normalized internally).")

    n_posts = st.slider("Number of posts/headlines", 50, 500, 200, 10)

    # X query builder
    default_query = f'("${ticker.replace("-USD","")}" OR "{ticker}" OR "#{ticker.replace("-USD","")}") lang:en -is:retweet'
    x_query = st.text_input("X (Twitter) Query", value=default_query)

    use_x = st.toggle("Use X as primary source (requires X_BEARER_TOKEN)", value=True)
    use_news = st.toggle("Use yfinance news headlines (optional)", value=True)

    with st.expander("Overlay Toggles + Alert Thresholds", expanded=False):
        show_volume = st.toggle("Show Volume Panel", value=True)
        show_sentiment_overlay = st.toggle("Overlay sentiment on price (secondary y)", value=True)
        show_subjectivity_overlay = st.toggle("Overlay subjectivity on price (secondary y)", value=False)
        sentiment_ema_len = st.slider("Sentiment Smoothing EMA (periods)", 2, 30, 6)
        sentiment_momentum_thr = st.slider("Sentiment Momentum Threshold", 0.01, 0.50, 0.10, 0.01)
        consensus_thr = st.slider("Consensus Threshold (abs score)", 0.05, 0.80, 0.10, 0.01)

    # Refresh
    st.divider()
    refresh = st.button("🔄 Refresh Data & Sentiment", use_container_width=True)
    if refresh:
        st.session_state.last_refresh = dt.datetime.utcnow().isoformat()

    st.caption(f"Last refresh: **{st.session_state.last_refresh or '—'}**")

# Normalize weights to decimals
w_t = float(w_t_pct) / 100.0 if "w_t_pct" in locals() else 0.40
w_v = float(w_v_pct) / 100.0 if "w_v_pct" in locals() else 0.35
w_b = float(w_b_pct) / 100.0 if "w_b_pct" in locals() else 0.25

# =============================================================================
# MAIN: LOAD PRICE DATA
# =============================================================================
top_status = st.status("Loading market data...", expanded=False)
with top_status:
    st.write("Fetching OHLCV...")
    df = fetch_price_data(ticker, start_dt, end_dt, interval)
    if df.empty:
        st.error("No price data returned. Check ticker/interval/date range.")
        top_status.update(label="Failed to load market data.", state="error")
        st.stop()

    # Overlay tickers
    overlay_data: Dict[str, pd.DataFrame] = {}
    if overlay_list:
        st.write("Fetching overlay tickers...")
        for tkr in overlay_list:
            overlay_data[tkr] = fetch_price_data(tkr, start_dt, end_dt, interval)

    top_status.update(label="Market data loaded.", state="complete")

# =============================================================================
# COMPUTE TA
# =============================================================================
ta_params = {
    "sma_len": sma_len,
    "ema_fast": ema_fast,
    "ema_slow": ema_slow,
    "bb_len": bb_len,
    "bb_mult": bb_mult,
    "macd_fast": macd_fast,
    "macd_slow": macd_slow,
    "macd_signal": macd_signal,
    "rsi_len": rsi_len,
    "rsi_overbought": rsi_overbought,
    "rsi_oversold": rsi_oversold,
    "stoch_k": stoch_k,
    "stoch_d": stoch_d,
    "psar_step": psar_step,
    "psar_max": psar_max,
    "adx_len": adx_len,
    "cci_len": cci_len,
    "atr_len": atr_len,
    "supertrend_len": supertrend_len,
    "supertrend_mult": supertrend_mult,
    "vp_bins": vp_bins,
    "fib_lookback": fib_lookback,
}

ta_bundle = compute_ta_bundle(df, ta_params)

# =============================================================================
# SENTIMENT PIPELINE (FETCH + ANALYZE)
# =============================================================================
sentiment_status = st.status("Sentiment pipeline ready.", expanded=False)

source_frames = []
source_labels = []

if use_x:
    source_labels.append("X Posts")
if use_news:
    source_labels.append("yfinance News")

with sentiment_status:
    st.write("Preparing sentiment sources...")

    posts_df = pd.DataFrame()
    news_df = pd.DataFrame()

    if use_x:
        st.write("Fetching X posts (if credentials present)...")
        posts_df = fetch_x_posts_v2(x_query, n_posts)

    if use_news:
        st.write("Fetching yfinance headlines (optional)...")
        # using a smaller number for headlines; still controlled by n_posts
        news_df = fetch_yfinance_news_headlines(ticker, limit=min(200, n_posts))

    # Combine sources
    combined = []
    if not posts_df.empty:
        combined.append(posts_df[["created_at", "text"]].copy())
    if not news_df.empty:
        combined.append(news_df[["created_at", "text"]].copy())

    if combined:
        source_df = pd.concat(combined, axis=0).dropna(subset=["created_at", "text"]).sort_values("created_at").reset_index(drop=True)
    else:
        source_df = pd.DataFrame(columns=["created_at", "text"])

    if source_df.empty:
        st.warning(
            "No sentiment texts available. "
            "If you want X posts, add X_BEARER_TOKEN to st.secrets. Otherwise enable yfinance news."
        )
        sentiment_scored = pd.DataFrame()
        sentiment_meta = {"transformer_ok": False, "transformer_model": "", "transformer_error": "", "n_items": 0}
        sentiment_status.update(label="Sentiment pipeline: no sources available.", state="error")
    else:
        # Progress indicators for model analysis
        prog = st.progress(0)
        st.write("Analyzing sentiment with Transformer... VADER... TextBlob... (cached, ttl=900s)")

        prog.progress(10)
        # Run analysis (cached)
        sentiment_scored, sentiment_meta = analyze_sentiment_batch(source_df, sentiment_mode, w_t, w_v, w_b)
        prog.progress(100)
        prog.empty()

        if sentiment_scored.empty:
            sentiment_status.update(label="Sentiment pipeline failed to produce scores.", state="error")
        else:
            if sentiment_meta.get("transformer_ok", False):
                sentiment_status.update(
                    label=f"Sentiment ready (Transformer: {sentiment_meta.get('transformer_model','')})",
                    state="complete",
                )
            else:
                sentiment_status.update(
                    label="Sentiment ready (Transformer unavailable → auto-reweighted to VADER/TextBlob).",
                    state="complete",
                )

# Build time series aligned to price
sent_ts = pd.Series(index=df.index, dtype=float)
sent_ema = pd.Series(index=df.index, dtype=float)
subj_ts = pd.Series(index=df.index, dtype=float)

if sentiment_scored is not None and not sentiment_scored.empty:
    sent_ts = build_sentiment_timeseries(sentiment_scored, df.index)
    sent_ema = ema_smooth_series(sent_ts, sentiment_ema_len)

    # Subjectivity series: bucketed same way, for optional overlay
    subj_raw = pd.Series(
        sentiment_scored["tb_subjectivity"].astype(float).values,
        index=pd.to_datetime(sentiment_scored["created_at"], utc=True, errors="coerce"),
    ).dropna()
    if not subj_raw.empty:
        # resample similar to sentiment build
        subj_bucket = subj_raw.resample("1H").mean() if len(subj_raw) > 10 else subj_raw.resample("1D").mean()
        px_utc = pd.to_datetime(df.index, utc=True)
        subj_aligned = subj_bucket.reindex(px_utc, method="ffill")
        subj_aligned.index = df.index
        subj_ts = subj_aligned

# =============================================================================
# STRATEGY SIGNALS (TA + Sentiment)
# =============================================================================
ta_signal, ta_reason = ta_strategy_signal(df, ta_bundle, strategy, ta_params)

# Sentiment-driven strategy overlays
sent_signal = "NEUTRAL"
sent_reason = "Sentiment strategy not selected."
hybrid_now = float(sentiment_meta.get("hybrid_mean", 0.0)) if isinstance(sentiment_meta, dict) else 0.0
subj_now = float(sentiment_meta.get("subjectivity_mean", 0.0)) if isinstance(sentiment_meta, dict) else 0.0
vader_intensity = float(sentiment_meta.get("vader_intensity_mean", 0.0)) if isinstance(sentiment_meta, dict) else 0.0

# Most recent sentiment point (preferred over mean)
if sent_ema is not None and not sent_ema.dropna().empty:
    hybrid_now = float(sent_ema.dropna().iloc[-1])

if subj_ts is not None and not subj_ts.dropna().empty:
    subj_now = float(subj_ts.dropna().iloc[-1])

# Strategy logic
if strategy == "Sentiment: Extreme Sentiment Reversal":
    if hybrid_now < -0.5:
        sent_signal, sent_reason = "BUY", f"Strong Negative sentiment ({hybrid_now:.2f}) → contrarian buy."
    elif hybrid_now > 0.5:
        sent_signal, sent_reason = "SELL", f"Strong Positive sentiment ({hybrid_now:.2f}) → contrarian sell."
    else:
        sent_signal, sent_reason = "NEUTRAL", f"Sentiment not extreme ({hybrid_now:.2f})."

elif strategy == "Sentiment: Sentiment Momentum":
    if sent_ema is not None and len(sent_ema.dropna()) >= 8:
        recent = sent_ema.dropna().tail(8)
        slope = float(recent.iloc[-1] - recent.iloc[0])
        if slope > sentiment_momentum_thr:
            sent_signal, sent_reason = "BUY", f"Sentiment rising sharply (Δ={slope:.2f})."
        elif slope < -sentiment_momentum_thr:
            sent_signal, sent_reason = "SELL", f"Sentiment falling sharply (Δ={slope:.2f})."
        else:
            sent_signal, sent_reason = "NEUTRAL", f"Sentiment change muted (Δ={slope:.2f})."
    else:
        sent_signal, sent_reason = "NEUTRAL", "Not enough sentiment history for momentum."

elif strategy == "Sentiment: High Subjectivity + Oversold":
    r = ta_bundle.get("RSI")
    r_last = float(r.dropna().iloc[-1]) if r is not None and not r.dropna().empty else np.nan
    if np.isfinite(r_last) and subj_now > 0.7 and r_last < 30:
        sent_signal, sent_reason = "BUY", f"Subjectivity high ({subj_now:.2f}) + RSI oversold ({r_last:.2f})."
    else:
        sent_signal, sent_reason = "NEUTRAL", f"Condition not met (Subjectivity={subj_now:.2f}, RSI={r_last if np.isfinite(r_last) else np.nan})."

elif strategy == "Sentiment: Consensus Signal":
    if sentiment_scored is not None and not sentiment_scored.empty:
        last_row = sentiment_scored.iloc[-1]
        agree = bool(last_row.get("model_agreement", False))
        score = float(last_row.get("hybrid_score", 0.0))
        if agree and score > consensus_thr:
            sent_signal, sent_reason = "BUY", f"All three agree bullish, score={score:.2f}."
        elif agree and score < -consensus_thr:
            sent_signal, sent_reason = "SELL", f"All three agree bearish, score={score:.2f}."
        else:
            sent_signal, sent_reason = "NEUTRAL", f"No strong consensus (agree={agree}, score={score:.2f})."
    else:
        sent_signal, sent_reason = "NEUTRAL", "No sentiment batch available."

# Combined action (simple fusion)
combined_action = "NEUTRAL"
if ta_signal == "BUY" and sent_signal == "BUY":
    combined_action = "STRONG BUY"
elif ta_signal == "SELL" and sent_signal == "SELL":
    combined_action = "STRONG SELL"
elif ta_signal == "BUY" and sent_signal == "NEUTRAL":
    combined_action = "BUY (TA)"
elif ta_signal == "SELL" and sent_signal == "NEUTRAL":
    combined_action = "SELL (TA)"
elif ta_signal == "NEUTRAL" and sent_signal == "BUY":
    combined_action = "BUY (Sentiment)"
elif ta_signal == "NEUTRAL" and sent_signal == "SELL":
    combined_action = "SELL (Sentiment)"
else:
    combined_action = "NEUTRAL"

# =============================================================================
# LAYOUT: TABS
# =============================================================================
tab_chart, tab_ta, tab_sent, tab_strategy, tab_broadcast, tab_download = st.tabs(
    [
        "📊 Chart",
        "🧠 TA Dashboard",
        "🛰️ Sentiment Intelligence Dashboard",
        "🧩 Strategy Center",
        "📣 Broadcasting Center",
        "⬇️ Downloads",
    ]
)

# =============================================================================
# TAB: CHART
# =============================================================================
with tab_chart:
    # Primary plot (candles + overlays + sentiment)
    fig = plot_price_with_indicators(
        df=df,
        ta=ta_bundle,
        indicators_selected=indicators_selected,
        overlay_tickers_data=overlay_data,
        show_volume=show_volume,
        sentiment_series=sent_ema if show_sentiment_overlay else None,
        show_sentiment_overlay=show_sentiment_overlay,
        show_subjectivity_overlay=show_subjectivity_overlay,
        subjectivity_series=subj_ts if show_subjectivity_overlay else None,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optional Fibonacci + Pivot + Volume Profile (as separate helper panels)
    cols = st.columns(3)

    with cols[0]:
        st.subheader("Fibonacci (Auto Swing)")
        if "Fibonacci (Auto Swing)" in indicators_selected:
            fib = fibonacci_auto_swing(df, lookback=fib_lookback)
            if fib:
                fib_df = pd.DataFrame({"Level": list(fib.keys()), "Price": list(fib.values())})
                st.dataframe(fib_df, use_container_width=True, hide_index=True)
            else:
                st.info("Not enough data for fib swing.")
        else:
            st.caption("Enable in Indicators to view.")

    with cols[1]:
        st.subheader("Pivot Points (Classic)")
        if "Pivot Points (Classic)" in indicators_selected:
            piv = pivot_points_classic(df).dropna()
            if not piv.empty:
                last_p = piv.iloc[-1]
                st.write(pd.DataFrame(last_p).rename(columns={last_p.name: "Value"}))
            else:
                st.info("Not enough data for pivot points.")
        else:
            st.caption("Enable in Indicators to view.")

    with cols[2]:
        st.subheader("Volume Profile (Basic)")
        if "Volume Profile (Basic)" in indicators_selected:
            vp = volume_profile_basic(df, bins=vp_bins)
            if not vp.empty:
                vp_fig = go.Figure()
                vp_fig.add_trace(go.Bar(x=vp["volume"], y=vp["price"], orientation="h", name="Volume"))
                vp_fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Price", xaxis_title="Volume")
                st.plotly_chart(vp_fig, use_container_width=True)
                # POC
                poc_price = float(vp.loc[vp["volume"].idxmax(), "price"])
                st.caption(f"POC (approx): **{poc_price:.4f}**")
            else:
                st.info("Not enough volume profile data.")
        else:
            st.caption("Enable in Indicators to view.")

# =============================================================================
# TAB: TA DASHBOARD
# =============================================================================
with tab_ta:
    st.subheader("🧠 Technical Analysis Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    last_close = float(df["Close"].iloc[-1])
    last_ret = float(df["Close"].pct_change().iloc[-1]) if len(df) > 2 else 0.0

    c1.metric("Last Close", f"{last_close:,.4f}")
    c2.metric("Last Return", f"{last_ret*100:,.2f}%")
    if "RSI" in ta_bundle and not ta_bundle["RSI"].dropna().empty:
        c3.metric("RSI", f"{float(ta_bundle['RSI'].dropna().iloc[-1]):.2f}")
    if "ADX" in ta_bundle and not ta_bundle["ADX"].dropna().empty:
        c4.metric("ADX", f"{float(ta_bundle['ADX'].dropna().iloc[-1]):.2f}")

    st.divider()

    # Subplots: MACD / RSI / Stoch / ADX / CCI
    sub = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.22, 0.20, 0.20, 0.20, 0.18],
    )

    # MACD
    if "MACD" in ta_bundle and "MACD_SIGNAL" in ta_bundle and "MACD_HIST" in ta_bundle:
        sub.add_trace(go.Scatter(x=df.index, y=ta_bundle["MACD"], mode="lines", name="MACD"), row=1, col=1)
        sub.add_trace(go.Scatter(x=df.index, y=ta_bundle["MACD_SIGNAL"], mode="lines", name="Signal"), row=1, col=1)
        sub.add_trace(go.Bar(x=df.index, y=ta_bundle["MACD_HIST"], name="Hist", opacity=0.6), row=1, col=1)

    # RSI
    if "RSI" in ta_bundle:
        sub.add_trace(go.Scatter(x=df.index, y=ta_bundle["RSI"], mode="lines", name="RSI"), row=2, col=1)
        sub.add_hline(y=rsi_overbought, row=2, col=1)
        sub.add_hline(y=rsi_oversold, row=2, col=1)

    # Stochastic
    if "STOCH_K" in ta_bundle and "STOCH_D" in ta_bundle:
        sub.add_trace(go.Scatter(x=df.index, y=ta_bundle["STOCH_K"], mode="lines", name="%K"), row=3, col=1)
        sub.add_trace(go.Scatter(x=df.index, y=ta_bundle["STOCH_D"], mode="lines", name="%D"), row=3, col=1)
        sub.add_hline(y=80, row=3, col=1)
        sub.add_hline(y=20, row=3, col=1)

    # ADX
    if "ADX" in ta_bundle and "+DI" in ta_bundle and "-DI" in ta_bundle:
        sub.add_trace(go.Scatter(x=df.index, y=ta_bundle["ADX"], mode="lines", name="ADX"), row=4, col=1)
        sub.add_trace(go.Scatter(x=df.index, y=ta_bundle["+DI"], mode="lines", name="+DI"), row=4, col=1)
        sub.add_trace(go.Scatter(x=df.index, y=ta_bundle["-DI"], mode="lines", name="-DI"), row=4, col=1)
        sub.add_hline(y=20, row=4, col=1)

    # CCI
    if "CCI" in ta_bundle:
        sub.add_trace(go.Scatter(x=df.index, y=ta_bundle["CCI"], mode="lines", name="CCI"), row=5, col=1)
        sub.add_hline(y=100, row=5, col=1)
        sub.add_hline(y=-100, row=5, col=1)

    sub.update_layout(height=900, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10), title="TA Subpanels")
    st.plotly_chart(sub, use_container_width=True)

# =============================================================================
# TAB: SENTIMENT INTELLIGENCE DASHBOARD
# =============================================================================
with tab_sent:
    st.subheader("🛰️ Sentiment Intelligence Dashboard")

    if sentiment_scored is None or sentiment_scored.empty:
        st.info("No sentiment batch available. Add X_BEARER_TOKEN to st.secrets and/or enable yfinance news.")
    else:
        # Current readings
        current_score = float(sentiment_scored["hybrid_score"].iloc[-1])
        current_subjectivity = float(sentiment_scored["tb_subjectivity"].iloc[-1])
        current_intensity = float(abs(sentiment_scored["vader_compound"].iloc[-1]))

        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(gauge_figure(current_score, "Hybrid Sentiment (-1..+1)", -1, 1), use_container_width=True)
        with g2:
            st.plotly_chart(gauge_figure(current_subjectivity, "Subjectivity (0..1)", 0, 1), use_container_width=True)
        with g3:
            st.plotly_chart(gauge_figure(current_intensity, "Intensity (|VADER compound|)", 0, 1), use_container_width=True)

        st.divider()

        # Pos/Neu/Neg across models (distributions)
        def dist_from_scores(scores: pd.Series) -> Dict[str, float]:
            s = scores.dropna().astype(float)
            if s.empty:
                return {"Positive": 0, "Neutral": 0, "Negative": 0}
            pos = float((s > 0.05).mean())
            neg = float((s < -0.05).mean())
            neu = float(1.0 - pos - neg)
            return {"Positive": pos, "Neutral": neu, "Negative": neg}

        dist_tf = dist_from_scores(sentiment_scored["transformer_score_norm"]) if "transformer_score_norm" in sentiment_scored else {"Positive": 0, "Neutral": 0, "Negative": 0}
        dist_vd = dist_from_scores(sentiment_scored["vader_compound"])
        dist_tb = dist_from_scores(sentiment_scored["tb_polarity"])
        dist_hy = dist_from_scores(sentiment_scored["hybrid_score"])

        dist_df = pd.DataFrame(
            [
                {"Model": "Transformer", **dist_tf},
                {"Model": "VADER", **dist_vd},
                {"Model": "TextBlob", **dist_tb},
                {"Model": "Hybrid", **dist_hy},
            ]
        )
        dist_long = dist_df.melt(id_vars=["Model"], var_name="Class", value_name="Percent")
        dist_long["Percent"] = dist_long["Percent"] * 100.0

        c1, c2 = st.columns([0.55, 0.45])
        with c1:
            bar = px.bar(dist_long, x="Model", y="Percent", color="Class", barmode="group", title="% Positive / Neutral / Negative")
            bar.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(bar, use_container_width=True)

        with c2:
            # Radar comparison (use means)
            t_mean = float(sentiment_scored["transformer_score_norm"].dropna().mean()) if "transformer_score_norm" in sentiment_scored and not sentiment_scored["transformer_score_norm"].dropna().empty else 0.0
            v_mean = float(sentiment_scored["vader_compound"].dropna().mean()) if not sentiment_scored["vader_compound"].dropna().empty else 0.0
            b_mean = float(sentiment_scored["tb_polarity"].dropna().mean()) if not sentiment_scored["tb_polarity"].dropna().empty else 0.0
            st.plotly_chart(radar_model_comparison(t_mean, v_mean, b_mean), use_container_width=True)

        st.divider()

        # Histogram of individual post sentiment scores
        hist = px.histogram(
            sentiment_scored,
            x="hybrid_score",
            nbins=40,
            title="Histogram of Individual Post Hybrid Scores",
        )
        hist.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(hist, use_container_width=True)

        st.divider()

        # Word cloud
        st.subheader("☁️ Word Cloud")
        wc_fig = render_wordcloud(sentiment_scored["text"].astype(str).tolist())
        if wc_fig is None:
            st.warning("Word cloud unavailable (NLTK stopwords missing or no text).")
        else:
            st.pyplot(wc_fig, clear_figure=True)

        st.divider()

        # Top 10 positive / negative posts with per-model scores
        st.subheader("🏆 Top Posts (Most Positive / Most Negative)")
        view_cols = [
            "created_at",
            "text",
            "hybrid_score",
            "hybrid_class",
            "transformer_score_norm",
            "vader_compound",
            "tb_polarity",
            "tb_subjectivity",
            "model_agreement",
        ]
        available_cols = [c for c in view_cols if c in sentiment_scored.columns]

        top_pos = sentiment_scored.sort_values("hybrid_score", ascending=False).head(10)[available_cols]
        top_neg = sentiment_scored.sort_values("hybrid_score", ascending=True).head(10)[available_cols]

        cpos, cneg = st.columns(2)
        with cpos:
            st.markdown("**Top 10 Positive**")
            st.dataframe(top_pos, use_container_width=True, hide_index=True)
        with cneg:
            st.markdown("**Top 10 Negative**")
            st.dataframe(top_neg, use_container_width=True, hide_index=True)

        st.divider()

        # Divergence detection (simple heuristics)
        st.subheader("⚡ Divergence & Insight Detection")
        insights = []

        # Price trend vs sentiment cooling/warming
        if len(df) >= 30 and sent_ema is not None and len(sent_ema.dropna()) >= 30:
            px_slope = float(df["Close"].tail(30).iloc[-1] / df["Close"].tail(30).iloc[0] - 1.0)
            se = sent_ema.dropna().tail(30)
            se_slope = float(se.iloc[-1] - se.iloc[0])

            if px_slope > 0.03 and se_slope < -0.10:
                insights.append("Price rising **but sentiment cooling** → potential exhaustion / distribution risk.")
            if px_slope < -0.03 and se_slope > 0.10:
                insights.append("Price falling **but sentiment improving** → potential bottoming / accumulation signal.")

        # High subjectivity spike
        if subj_ts is not None and len(subj_ts.dropna()) >= 10:
            sub_recent = subj_ts.dropna().tail(10)
            sub_spike = float(sub_recent.iloc[-1] - sub_recent.mean())
            if sub_spike > 0.15 and sub_recent.iloc[-1] > 0.75:
                insights.append("Subjectivity spike detected → chatter/opinion intensity rising (risk of noise-driven moves).")

        # VADER intensity high
        if vader_intensity > 0.35:
            insights.append(f"High VADER intensity (avg |compound|={vader_intensity:.2f}) → emotionally charged discourse (volatility risk).")

        if not insights:
            st.info("No strong divergences detected under current heuristics.")
        else:
            for it in insights:
                st.warning(it)

        st.divider()

        # Correlation panel
        st.subheader("🔗 Rolling Correlation: Sentiment vs Price Returns")
        corr_df = rolling_corr(sent_ema.fillna(method="ffill"), df["Close"], windows=[7, 14, 30])
        if corr_df is None or corr_df.empty:
            st.info("Not enough data for rolling correlations.")
        else:
            corr_fig = go.Figure()
            for col in corr_df.columns:
                corr_fig.add_trace(go.Scatter(x=corr_df.index, y=corr_df[col], mode="lines", name=col))
            corr_fig.update_layout(height=380, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(corr_fig, use_container_width=True)

# =============================================================================
# TAB: STRATEGY CENTER
# =============================================================================
with tab_strategy:
    st.subheader("🧩 Strategy Center")

    c1, c2, c3 = st.columns(3)
    c1.metric("TA Signal", ta_signal)
    c2.metric("Sentiment Signal", sent_signal)
    c3.metric("Combined Action", combined_action)

    st.markdown("### Rationale")
    st.write(f"**TA:** {ta_reason}")
    st.write(f"**Sentiment:** {sent_reason}")

    # Detailed signal table (latest)
    st.divider()
    st.markdown("### Signals Table (Latest Bar)")

    latest = {
        "Ticker": ticker,
        "Time": str(df.index[-1]),
        "Last Close": float(df["Close"].iloc[-1]),
        "TA Signal": ta_signal,
        "Sentiment (Hybrid)": float(hybrid_now),
        "Subjectivity": float(subj_now),
        "Model Consensus": bool(
            sentiment_scored.iloc[-1]["model_agreement"] if sentiment_scored is not None and not sentiment_scored.empty and "model_agreement" in sentiment_scored.columns else False
        ),
        "Sentiment Trend (EMA Δ)": float(sent_ema.dropna().tail(8).iloc[-1] - sent_ema.dropna().tail(8).iloc[0]) if sent_ema is not None and len(sent_ema.dropna()) >= 8 else np.nan,
        "Combined Action": combined_action,
    }
    st.dataframe(pd.DataFrame([latest]), use_container_width=True, hide_index=True)

    # Add a quick strategy preview matrix for overlay tickers (alerts-style)
    st.divider()
    st.markdown("### Multi-Asset Alerts (Primary + Overlays)")

    tickers_for_alerts = [ticker] + overlay_list[:10]  # limit to keep UI responsive
    alerts_rows = []
    with st.spinner("Computing alerts snapshot..."):
        for tkr in tickers_for_alerts:
            dfx = df if tkr == ticker else overlay_data.get(tkr, pd.DataFrame())
            if dfx is None or dfx.empty:
                continue
            tax = compute_ta_bundle(dfx, ta_params)
            sig, why = ta_strategy_signal(dfx, tax, "MA Crossover (EMA Fast/Slow)", ta_params)

            # Use same sentiment snapshot for all (source query is typically market-wide; per-ticker custom queries are possible via sidebar)
            cons = bool(
                sentiment_scored.iloc[-1]["model_agreement"]
                if sentiment_scored is not None and not sentiment_scored.empty and "model_agreement" in sentiment_scored.columns
                else False
            )

            alerts_rows.append(
                {
                    "Ticker": tkr,
                    "TA Signal (MA Xover)": sig,
                    "Hybrid Sentiment": float(hybrid_now),
                    "Subjectivity": float(subj_now),
                    "Model Consensus": cons,
                    "Combined Action": ("BUY" if sig == "BUY" and hybrid_now > 0.05 else "SELL" if sig == "SELL" and hybrid_now < -0.05 else "WATCH"),
                }
            )

    if alerts_rows:
        st.dataframe(pd.DataFrame(alerts_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No alerts computed (missing data).")

# =============================================================================
# TAB: BROADCASTING CENTER
# =============================================================================
with tab_broadcast:
    st.subheader("📣 Broadcasting Center Enhancements")

    st.markdown(
        "This center assembles real-time alert rows with **Ticker, TA Signal, Hybrid Sentiment, Subjectivity, Model Consensus, Combined Action**.\n\n"
        "Optional: broadcast a snapshot to Telegram if TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are present in `st.secrets`."
    )

    # Use latest alerts snapshot from Strategy tab (recompute small snapshot for safety)
    tickers_for_alerts = [ticker] + overlay_list[:10]
    rows = []
    for tkr in tickers_for_alerts:
        dfx = df if tkr == ticker else overlay_data.get(tkr, pd.DataFrame())
        if dfx is None or dfx.empty:
            continue
        tax = compute_ta_bundle(dfx, ta_params)
        sig, _ = ta_strategy_signal(dfx, tax, "MA Crossover (EMA Fast/Slow)", ta_params)
        cons = bool(
            sentiment_scored.iloc[-1]["model_agreement"]
            if sentiment_scored is not None and not sentiment_scored.empty and "model_agreement" in sentiment_scored.columns
            else False
        )
        rows.append(
            {
                "Ticker": tkr,
                "TA Signal": sig,
                "Hybrid Sentiment": float(hybrid_now),
                "Subjectivity": float(subj_now),
                "Model Consensus": cons,
                "Combined Action": ("BUY" if sig == "BUY" and hybrid_now > 0.05 else "SELL" if sig == "SELL" and hybrid_now < -0.05 else "WATCH"),
            }
        )

    alerts_df = pd.DataFrame(rows)
    st.dataframe(alerts_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Telegram Broadcast (Optional)")
    msg = st.text_area(
        "Message Preview",
        value=(
            f"[{dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}] Alerts Snapshot\n"
            f"Primary: {ticker}\n"
            f"TA Signal: {ta_signal}\n"
            f"Hybrid Sentiment: {hybrid_now:.2f}\n"
            f"Subjectivity: {subj_now:.2f}\n"
            f"Combined: {combined_action}\n"
        ),
        height=160,
    )

    col1, col2 = st.columns([0.25, 0.75])
    with col1:
        send_now = st.button("📨 Send to Telegram")
    with col2:
        st.caption("Requires TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in st.secrets. No credentials are ever hardcoded.")

    if send_now:
        ok, info = telegram_send_message(msg)
        if ok:
            st.success(info)
        else:
            st.error(info)

# =============================================================================
# TAB: DOWNLOADS
# =============================================================================
with tab_download:
    st.subheader("⬇️ Downloads")

    # Build a combined export table: price + selected TA + sentiment series
    export = df.copy()
    for k, s in ta_bundle.items():
        if isinstance(s, pd.Series):
            export[k] = s

    export["Sentiment_Hybrid_TS"] = sent_ts.astype(float) if sent_ts is not None and not sent_ts.empty else np.nan
    export["Sentiment_Hybrid_EMA"] = sent_ema.astype(float) if sent_ema is not None and not sent_ema.empty else np.nan
    export["Subjectivity_TS"] = subj_ts.astype(float) if subj_ts is not None and not subj_ts.empty else np.nan

    st.markdown("### Export Market + Indicators + Sentiment (CSV)")
    csv = export.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=f"{ticker}_ta_sentiment_export.csv", mime="text/csv")

    st.divider()
    st.markdown("### Export Sentiment Items (CSV)")
    if sentiment_scored is None or sentiment_scored.empty:
        st.info("No sentiment batch to export.")
    else:
        sent_csv = sentiment_scored.to_csv(index=False).encode("utf-8")
        st.download_button("Download Sentiment CSV", data=sent_csv, file_name=f"{ticker}_sentiment_items.csv", mime="text/csv")

# =============================================================================
# FOOTER: MODEL STATUS + FALLBACK TRANSPARENCY
# =============================================================================
with st.expander("ℹ️ System Status / Model Loading / Fallbacks", expanded=False):
    st.markdown("### Transformer Model")
    if sentiment_meta.get("transformer_ok", False):
        st.success(f"Loaded: **{sentiment_meta.get('transformer_model','')}**")
    else:
        st.warning("Transformer unavailable. The app automatically reweights to VADER/TextBlob.")
        if sentiment_meta.get("transformer_error"):
            st.code(sentiment_meta.get("transformer_error"))

    st.markdown("### Data Sources")
    st.write(f"X posts fetched: **{0 if posts_df is None else len(posts_df)}**")
    st.write(f"yfinance headlines fetched: **{0 if news_df is None else len(news_df)}**")
    st.write(f"Total sentiment items analyzed: **{int(sentiment_meta.get('n_items', 0))}**")

    st.markdown("### Ensemble Weights (Effective)")
    if sentiment_scored is not None and not sentiment_scored.empty:
        w_eff = sentiment_scored[["w_transformer", "w_vader", "w_textblob"]].iloc[-1].to_dict()
        st.json({k: float(v) for k, v in w_eff.items()})
    else:
        st.write("No effective weights (no sentiment batch).")

    st.markdown("### Classification Thresholds")
    st.write("- Strong Positive: > 0.5")
    st.write("- Positive: > 0.05")
    st.write("- Neutral: -0.05 to 0.05")
    st.write("- Negative: < -0.05")
    st.write("- Strong Negative: < -0.5")
