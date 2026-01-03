# app.py
# =============================================================================
# MACRO MASTER: Expert Macro Technical + Fundamental Analyst Machine
# - Built around the user's TradingView-style macro watchlists (symbols + ratios)
# - Data source: Yahoo Finance via yfinance (with robust TV-symbol -> yfinance mapping + overrides)
# - Indicators: VWAP, MFI, Bollinger Bands, Ichimoku Cloud
# - SMC (Smart Money Concepts): pivot-based market structure + BOS/CHOCH + swing levels
# - AI Analysis: OpenAI and/or Gemini (optional, via st.secrets)
#
# IMPORTANT NOTES
# - TradingView symbols like COMEX:GC1! are mapped to yfinance tickers like GC=F.
# - Not all symbols exist on every free data source; this app is transparent about failures and lets you override mappings.
# - This is not financial advice.
# =============================================================================

from __future__ import annotations

import math
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional dependencies
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except Exception:
    OPENAI_SDK_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


# ---------------------------
# Defaults: Watchlists
# ---------------------------
DEFAULT_SYMBOLS: List[str] = [
    # Futures - Metals
    "COMEX:GC1!",
    "COMEX:SI1!",
    "COMEX:HG1!",
    "NYMEX:PL1!",
    "NYMEX:PA1!",
    # Futures - Energy
    "NYMEX:CL1!",
    "ICE:BRN1!",
    "NYMEX:NG1!",
    "NYMEX:RB1!",
    "NYMEX:HO1!",
    # Futures - Grains
    "CBOT:ZC1!",
    "CBOT:ZW1!",
    "CBOT:ZS1!",
    "CBOT:ZM1!",
    "CBOT:ZL1!",
    # Futures - Softs
    "ICEUS:KC1!",
    "ICEUS:CC1!",
    "ICEUS:SB1!",
    "ICEUS:CT1!",
    "ICEUS:OJ1!",
    # Futures - Livestock
    "CME:LE1!",
    "CME:GF1!",
    "CME:HE1!",
    # Global Indices (cash indices)
    "SP:SPX",
    "NASDAQ:NDX",
    "DJ:DJI",
    "RUSSELL:RUT",
    "TVC:VIX",
    "TVC:DXY",
    # Europe
    "TVC:DAX",
    "TVC:UKX",
    "TVC:CAC40",
    "TVC:SX5E",
    "TVC:SMI",
    # Asia
    "TVC:NI225",
    "TVC:TOPIX",
    "TVC:HSI",
    "NSE:NIFTY",
    # Major global stocks (Top 50)
    "NASDAQ:NVDA",
    "NASDAQ:AAPL",
    "NASDAQ:GOOGL",
    "NASDAQ:MSFT",
    "NASDAQ:AMZN",
    "TWSE:2330",
    "NASDAQ:AVGO",
    "NASDAQ:META",
    "TADAWUL:2222",
    "NASDAQ:TSLA",
    "NYSE:BRK.B",
    "NYSE:LLY",
    "NYSE:WMT",
    "NYSE:JPM",
    "OTC:TCEHY",
    "NYSE:V",
    "KRX:005930",
    "NYSE:ORCL",
    "NYSE:XOM",
    "NYSE:MA",
    "NYSE:JNJ",
    "EURONEXT:ASML",
    "NYSE:BAC",
    "NYSE:ABBV",
    "NASDAQ:PLTR",
    "NASDAQ:NFLX",
    "SSE:601288",
    "NASDAQ:COST",
    "EURONEXT:MC",
    "NYSE:BABA",
    "NASDAQ:AMD",
    "HKEX:1398",
    "NASDAQ:MU",
    "SSE:601939",
    "NYSE:HD",
    "NYSE:GE",
    "SWX:ROG",
    "NYSE:PG",
    "KRX:000660",
    "NYSE:CVX",
    "NYSE:WFC",
    "NYSE:UNH",
    "NASDAQ:CSCO",
    "NYSE:KO",
    "NYSE:MS",
    "NYSE:TM",
    "NASDAQ:AZN",
    "NYSE:CAT",
    "NYSE:GS",
    "XETR:SAP",
    # Crypto + crypto market series (some series may not be supported by yfinance)
    "CRYPTOCAP:TOTAL",
    "CRYPTOCAP:TOTAL2",
    "CRYPTOCAP:TOTAL3",
    "CRYPTOCAP:BTC.D",
    "CRYPTOCAP:USDT.D",
    "CRYPTOCAP:USDC.D",
    "MEXC:BTCUSDT",
    "MEXC:ETHUSDT",
    "MEXC:XRPUSDT",
    "MEXC:BNBUSDT",
    "MEXC:SOLUSDT",
    "MEXC:TRXUSDT",
    "MEXC:DOGEUSDT",
    "MEXC:ADAUSDT",
    "MEXC:BCHUSDT",
    "MEXC:LINKUSDT",
    "MEXC:LTCUSDT",
    "MEXC:XMRUSDT",
    "MEXC:XLMUSDT",
    "MEXC:SUIUSDT",
    "MEXC:AVAXUSDT",
    "MEXC:TONUSDT",
    "MEXC:HBARUSDT",
    "MEXC:DOTUSDT",
    "MEXC:NEARUSDT",
    "MEXC:ATOMUSDT",
    "MEXC:ICPUSDT",
    "MEXC:FILUSDT",
    "MEXC:ARBUSDT",
    "MEXC:OPUSDT",
    "MEXC:UNIUSDT",
    "MEXC:AAVEUSDT",
    "MEXC:RNDRUSDT",
    "MEXC:FETUSDT",
    "MEXC:SHIBUSDT",
]

DEFAULT_RATIOS: List[str] = [
    "XAUUSD/SP:SPX",
    "XAGUSD/SP:SPX",
    "COMEX:GC1!/CME_MINI:ES1!",
    "COMEX:SI1!/CME_MINI:ES1!",
    "COMEX:HG1!/COMEX:GC1!",
    "COMEX:SI1!/COMEX:GC1!",
    "XAUUSD/TVC:DXY",
    "NYMEX:CL1!/CME_MINI:ES1!",
    "ICE:BRN1!/CME_MINI:ES1!",
    "COMEX:HG1!/CME_MINI:ES1!",
    "NYMEX:CL1!/NYMEX:NG1!",
    "NYMEX:RB1!/NYMEX:CL1!",
    "NYMEX:HO1!/NYMEX:CL1!",
    "NYMEX:CL1!/COMEX:GC1!",
    "CBOT:ZC1!/CBOT:ZW1!",
    "CBOT:ZS1!/CBOT:ZC1!",
    "ICEUS:KC1!/ICEUS:SB1!",
    "ICEUS:CC1!/ICEUS:KC1!",
    "NASDAQ:NDX/SP:SPX",
    "RUSSELL:RUT/SP:SPX",
    "DJ:DJI/SP:SPX",
    "SP:SPX/TVC:VIX",
    "NASDAQ:NDX/TVC:VIX",
    "MEXC:BTCUSDT/SP:SPX",
    "MEXC:BTCUSDT/NASDAQ:NDX",
    "MEXC:ETHUSDT/SP:SPX",
    "MEXC:BTCUSDT/TVC:DXY",
    "MEXC:BTCUSDT/XAUUSD",
    "MEXC:BTCUSDT/NYMEX:CL1!",
    "MEXC:BTCUSDT/COMEX:HG1!",
    "MEXC:ETHUSDT/MEXC:BTCUSDT",
    "MEXC:SOLUSDT/MEXC:ETHUSDT",
]


# ---------------------------
# TradingView symbol -> yfinance mapping
# ---------------------------
# You can override any mapping in the UI.
TV_TO_YF: Dict[str, str] = {
    # Futures -> Yahoo Finance continuous futures symbols
    "COMEX:GC1!": "GC=F",
    "COMEX:SI1!": "SI=F",
    "COMEX:HG1!": "HG=F",
    "NYMEX:PL1!": "PL=F",
    "NYMEX:PA1!": "PA=F",
    "NYMEX:CL1!": "CL=F",
    "ICE:BRN1!": "BZ=F",
    "NYMEX:NG1!": "NG=F",
    "NYMEX:RB1!": "RB=F",
    "NYMEX:HO1!": "HO=F",
    "CBOT:ZC1!": "ZC=F",
    "CBOT:ZW1!": "ZW=F",
    "CBOT:ZS1!": "ZS=F",
    "CBOT:ZM1!": "ZM=F",
    "CBOT:ZL1!": "ZL=F",
    "ICEUS:KC1!": "KC=F",
    "ICEUS:CC1!": "CC=F",
    "ICEUS:SB1!": "SB=F",
    "ICEUS:CT1!": "CT=F",
    "ICEUS:OJ1!": "OJ=F",
    "CME:LE1!": "LE=F",
    "CME:GF1!": "GF=F",
    "CME:HE1!": "HE=F",

    # Indices
    "SP:SPX": "^GSPC",
    "NASDAQ:NDX": "^NDX",
    "DJ:DJI": "^DJI",
    "RUSSELL:RUT": "^RUT",
    "TVC:VIX": "^VIX",
    "TVC:DXY": "DX-Y.NYB",
    "TVC:DAX": "^GDAXI",
    "TVC:UKX": "^FTSE",
    "TVC:CAC40": "^FCHI",
    "TVC:SX5E": "^STOXX50E",
    "TVC:SMI": "^SSMI",
    "TVC:NI225": "^N225",
    "TVC:TOPIX": "^TOPX",
    "TVC:HSI": "^HSI",
    "NSE:NIFTY": "^NSEI",

    # Spot placeholders (we map to common FX/metal spot proxies on yfinance where possible)
    "XAUUSD": "XAUUSD=X",
    "XAGUSD": "XAGUSD=X",

    # Stocks: map exchange-qualified to plain (or specific suffix when required)
    "TWSE:2330": "2330.TW",
    "TADAWUL:2222": "2222.SR",
    "NYSE:BRK.B": "BRK-B",
    "EURONEXT:ASML": "ASML.AS",
    "EURONEXT:MC": "MC.PA",
    "SWX:ROG": "ROG.SW",
    "XETR:SAP": "SAP.DE",
    "KRX:005930": "005930.KS",
    "KRX:000660": "000660.KS",
    "HKEX:1398": "1398.HK",
    "SSE:601288": "601288.SS",
    "SSE:601939": "601939.SS",
    "OTC:TCEHY": "TCEHY",

    # Crypto exchange symbols -> yfinance crypto tickers
    "MEXC:BTCUSDT": "BTC-USD",
    "MEXC:ETHUSDT": "ETH-USD",
    "MEXC:XRPUSDT": "XRP-USD",
    "MEXC:BNBUSDT": "BNB-USD",
    "MEXC:SOLUSDT": "SOL-USD",
    "MEXC:TRXUSDT": "TRX-USD",
    "MEXC:DOGEUSDT": "DOGE-USD",
    "MEXC:ADAUSDT": "ADA-USD",
    "MEXC:BCHUSDT": "BCH-USD",
    "MEXC:LINKUSDT": "LINK-USD",
    "MEXC:LTCUSDT": "LTC-USD",
    "MEXC:XMRUSDT": "XMR-USD",
    "MEXC:XLMUSDT": "XLM-USD",
    "MEXC:SUIUSDT": "SUI-USD",
    "MEXC:AVAXUSDT": "AVAX-USD",
    "MEXC:TONUSDT": "TON-USD",
    "MEXC:HBARUSDT": "HBAR-USD",
    "MEXC:DOTUSDT": "DOT-USD",
    "MEXC:NEARUSDT": "NEAR-USD",
    "MEXC:ATOMUSDT": "ATOM-USD",
    "MEXC:ICPUSDT": "ICP-USD",
    "MEXC:FILUSDT": "FIL-USD",
    "MEXC:ARBUSDT": "ARB-USD",
    "MEXC:OPUSDT": "OP-USD",
    "MEXC:UNIUSDT": "UNI-USD",
    "MEXC:AAVEUSDT": "AAVE-USD",
    "MEXC:RNDRUSDT": "RNDR-USD",
    "MEXC:FETUSDT": "FET-USD",
    "MEXC:SHIBUSDT": "SHIB-USD",
}


# ---------------------------
# Utility: parsing + loading watchlists
# ---------------------------
def _safe_split_lines(text: str) -> List[str]:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def load_watchlist_from_upload(uploaded_file) -> List[str]:
    if uploaded_file is None:
        return []
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors="replace")
        return _safe_split_lines(content)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        return []


def is_ratio_symbol(s: str) -> bool:
    return "/" in s and not s.startswith("http")


def tv_to_yf_symbol(tv_symbol: str, overrides: Dict[str, str]) -> str:
    # If user manually overrides, respect it.
    if tv_symbol in overrides and overrides[tv_symbol].strip():
        return overrides[tv_symbol].strip()

    # Handle ratios like A/B
    if is_ratio_symbol(tv_symbol):
        return tv_symbol  # ratios handled separately

    # If it looks like "EXCHANGE:SYMBOL", try mapping with dict then fallback to SYMBOL
    if tv_symbol in TV_TO_YF:
        return TV_TO_YF[tv_symbol]
    # If it's already a yfinance-like symbol (contains = or - or ^ or . suffix), use directly
    if any(x in tv_symbol for x in ["=F", "^", "=", "-", "."]):
        return tv_symbol
    # If it's exchange-qualified but not in mapping, strip prefix
    if ":" in tv_symbol:
        return tv_symbol.split(":", 1)[1]
    return tv_symbol


# ---------------------------
# Data fetching (Yahoo Finance)
# ---------------------------
@st.cache_data(show_spinner=False, ttl=60 * 10)
def yf_download(symbol: str, interval: str, start: Optional[datetime], end: Optional[datetime]) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Download OHLCV data. Returns (df, error_message).
    df columns expected: Open, High, Low, Close, Volume
    """
    if not YF_AVAILABLE:
        return pd.DataFrame(), "Missing dependency: yfinance is not installed."

    try:
        df = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        if df is None or df.empty:
            return pd.DataFrame(), f"No data returned for {symbol} (interval={interval})."
        # Standardize
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            # Sometimes yfinance returns multiindex for multiple tickers; we only expect one
            df.columns = [c[-1] for c in df.columns]
        needed = {"Open", "High", "Low", "Close"}
        if not needed.issubset(set(df.columns)):
            return pd.DataFrame(), f"Data for {symbol} missing required OHLC columns."
        if "Volume" not in df.columns:
            df["Volume"] = np.nan
        df.index = pd.to_datetime(df.index)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"yfinance error for {symbol}: {e}"


# ---------------------------
# Indicators (pure pandas)
# ---------------------------
def add_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP computed as cumulative typical_price*volume / cumulative volume.
    Works best on intraday data. On daily data it's an anchored cumulative VWAP over the selected window.
    """
    d = df.copy()
    tp = (d["High"] + d["Low"] + d["Close"]) / 3.0
    vol = d["Volume"].fillna(0.0)
    pv = tp * vol
    cum_vol = vol.cumsum()
    cum_pv = pv.cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    return vwap.rename("VWAP")


def add_bollinger(df: pd.DataFrame, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
    close = df["Close"]
    basis = close.rolling(length).mean()
    dev = close.rolling(length).std(ddof=0)
    upper = basis + mult * dev
    lower = basis - mult * dev
    out = pd.DataFrame({"BB_MID": basis, "BB_UPPER": upper, "BB_LOWER": lower}, index=df.index)
    return out


def add_mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Money Flow Index (MFI):
    Typical Price = (H+L+C)/3
    Raw Money Flow = TP * Volume
    Positive MF if TP > TP_prev else Negative MF if TP < TP_prev
    Money Flow Ratio = sum(PosMF)/sum(NegMF)
    MFI = 100 - 100/(1+MFR)
    """
    d = df.copy()
    tp = (d["High"] + d["Low"] + d["Close"]) / 3.0
    rmf = tp * d["Volume"].fillna(0.0)
    tp_prev = tp.shift(1)

    pos = rmf.where(tp > tp_prev, 0.0)
    neg = rmf.where(tp < tp_prev, 0.0)

    pos_sum = pos.rolling(length).sum()
    neg_sum = neg.rolling(length).sum().replace(0, np.nan)

    mfr = pos_sum / neg_sum
    mfi = 100 - (100 / (1 + mfr))
    return mfi.rename("MFI")


def add_ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> pd.DataFrame:
    """
    Ichimoku components:
    Tenkan-sen = (9H + 9L)/2
    Kijun-sen  = (26H + 26L)/2
    Senkou A   = (Tenkan + Kijun)/2 shifted forward kijun periods
    Senkou B   = (52H + 52L)/2 shifted forward kijun periods
    Chikou     = Close shifted backward kijun periods
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2.0
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2.0
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun)
    senkou_b_line = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2.0).shift(kijun)
    chikou = close.shift(-kijun)

    out = pd.DataFrame(
        {
            "IC_TENKAN": tenkan_sen,
            "IC_KIJUN": kijun_sen,
            "IC_SENKOU_A": senkou_a,
            "IC_SENKOU_B": senkou_b_line,
            "IC_CHIKOU": chikou,
        },
        index=df.index,
    )
    return out


# ---------------------------
# SMC: pivot market structure + BOS/CHOCH
# ---------------------------
@dataclass
class SMCConfig:
    pivot_len: int = 3  # swing detection sensitivity
    use_close_break: bool = True  # BOS uses close break vs wick
    show_swings: bool = True
    show_bos: bool = True
    show_choch: bool = True


def detect_pivots(df: pd.DataFrame, left_right: int) -> Tuple[pd.Series, pd.Series]:
    """
    Pivot highs/lows: a pivot high is a high that is the max in a window of size 2*lr+1, centered.
    Returns (pivot_high, pivot_low) series with price at pivot or NaN.
    """
    lr = int(left_right)
    if lr < 1:
        lr = 1
    high = df["High"]
    low = df["Low"]
    win = 2 * lr + 1

    ph = high.rolling(win, center=True).max()
    pl = low.rolling(win, center=True).min()

    pivot_high = high.where(high == ph, np.nan)
    pivot_low = low.where(low == pl, np.nan)
    return pivot_high.rename("PIVOT_HIGH"), pivot_low.rename("PIVOT_LOW")


def smc_market_structure(df: pd.DataFrame, cfg: SMCConfig) -> pd.DataFrame:
    """
    Basic SMC-like structure:
    - Identify swing highs/lows via pivots
    - Track last confirmed swing high/low
    - BOS: price breaks previous swing high (bullish) or swing low (bearish)
    - CHOCH: "change of character" when breaks in opposite direction after trend flag
    """
    d = df.copy()
    ph, pl = detect_pivots(d, cfg.pivot_len)
    d["PIVOT_HIGH"] = ph
    d["PIVOT_LOW"] = pl

    # forward-fill last swing levels (confirmed pivots)
    d["LAST_SWING_HIGH"] = d["PIVOT_HIGH"].ffill()
    d["LAST_SWING_LOW"] = d["PIVOT_LOW"].ffill()

    # Break logic
    if cfg.use_close_break:
        up_break = d["Close"] > d["LAST_SWING_HIGH"].shift(1)
        dn_break = d["Close"] < d["LAST_SWING_LOW"].shift(1)
    else:
        up_break = d["High"] > d["LAST_SWING_HIGH"].shift(1)
        dn_break = d["Low"] < d["LAST_SWING_LOW"].shift(1)

    # crude trend state: +1 bullish, -1 bearish, 0 unknown
    trend = np.zeros(len(d), dtype=int)
    last = 0
    for i in range(len(d)):
        if bool(up_break.iloc[i]):
            last = 1
        elif bool(dn_break.iloc[i]):
            last = -1
        trend[i] = last
    d["SMC_TREND"] = trend

    # BOS flags (first break in that direction)
    d["BOS_BULL"] = up_break & (pd.Series(trend, index=d.index).shift(1).fillna(0).astype(int) != 1)
    d["BOS_BEAR"] = dn_break & (pd.Series(trend, index=d.index).shift(1).fillna(0).astype(int) != -1)

    # CHOCH: break opposite direction while trend is established
    prev_trend = pd.Series(trend, index=d.index).shift(1).fillna(0).astype(int)
    d["CHOCH_BULL"] = up_break & (prev_trend == -1)
    d["CHOCH_BEAR"] = dn_break & (prev_trend == 1)

    return d


# ---------------------------
# Plotting
# ---------------------------
def plot_price_with_indicators(
    df: pd.DataFrame,
    title: str,
    show_bollinger: bool,
    show_ichimoku: bool,
    show_vwap: bool,
    show_smc: bool,
    smc_cfg: SMCConfig,
    show_mfi: bool,
) -> go.Figure:
    rows = 2 if show_mfi else 1
    row_heights = [0.72, 0.28] if show_mfi else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
        specs=[[{"secondary_y": False}]] + ([[{"secondary_y": False}]] if show_mfi else []),
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
            showlegend=True,
        ),
        row=1, col=1
    )

    # Volume (as bars on price panel - optional but useful)
    if "Volume" in df.columns and df["Volume"].notna().any():
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                opacity=0.25,
                showlegend=True,
            ),
            row=1, col=1
        )

    # Bollinger
    if show_bollinger:
        bb = add_bollinger(df)
        fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_MID"], name="BB Mid", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_UPPER"], name="BB Upper", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_LOWER"], name="BB Lower", mode="lines"), row=1, col=1)

    # VWAP
    if show_vwap:
        vwap = add_vwap(df)
        fig.add_trace(go.Scatter(x=vwap.index, y=vwap, name="VWAP", mode="lines"), row=1, col=1)

    # Ichimoku
    if show_ichimoku:
        ic = add_ichimoku(df)
        fig.add_trace(go.Scatter(x=ic.index, y=ic["IC_TENKAN"], name="Ichimoku Tenkan", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ic.index, y=ic["IC_KIJUN"], name="Ichimoku Kijun", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ic.index, y=ic["IC_SENKOU_A"], name="Senkou A", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ic.index, y=ic["IC_SENKOU_B"], name="Senkou B", mode="lines"), row=1, col=1)

        # Cloud fill (A/B) — fill between traces using two scatters
        fig.add_trace(
            go.Scatter(
                x=ic.index,
                y=ic["IC_SENKOU_A"],
                name="Cloud A (fill)",
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=ic.index,
                y=ic["IC_SENKOU_B"],
                name="Cloud B (fill)",
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                opacity=0.15,
                showlegend=False,
            ),
            row=1, col=1
        )

    # SMC overlays
    if show_smc:
        smc_df = smc_market_structure(df, smc_cfg)

        if smc_cfg.show_swings:
            # Swing highs/lows as small circles (not triangles)
            ph = smc_df["PIVOT_HIGH"]
            pl = smc_df["PIVOT_LOW"]

            fig.add_trace(
                go.Scatter(
                    x=ph.index,
                    y=ph,
                    name="Swing High",
                    mode="markers",
                    marker=dict(size=7, symbol="circle"),
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=pl.index,
                    y=pl,
                    name="Swing Low",
                    mode="markers",
                    marker=dict(size=7, symbol="circle"),
                ),
                row=1, col=1
            )

        # BOS / CHOCH annotations
        def _add_event_marks(flag_col: str, label: str):
            idx = smc_df.index[smc_df[flag_col].fillna(False)]
            if len(idx) == 0:
                return
            y = smc_df.loc[idx, "Close"]
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=y,
                    name=label,
                    mode="markers+text",
                    text=[label]*len(idx),
                    textposition="top center",
                    marker=dict(size=10, symbol="circle-open"),
                ),
                row=1, col=1
            )

        if smc_cfg.show_bos:
            _add_event_marks("BOS_BULL", "BOS↑")
            _add_event_marks("BOS_BEAR", "BOS↓")
        if smc_cfg.show_choch:
            _add_event_marks("CHOCH_BULL", "CHOCH↑")
            _add_event_marks("CHOCH_BEAR", "CHOCH↓")

    # MFI pane
    if show_mfi:
        mfi = add_mfi(df)
        fig.add_trace(go.Scatter(x=mfi.index, y=mfi, name="MFI", mode="lines"), row=2, col=1)
        # Classic zones
        fig.add_hline(y=80, row=2, col=1, line_dash="dash")
        fig.add_hline(y=20, row=2, col=1, line_dash="dash")
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=850 if show_mfi else 700,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ---------------------------
# Fundamentals
# ---------------------------
def fetch_fundamentals_yf(yf_symbol: str) -> Tuple[dict, Optional[str]]:
    if not YF_AVAILABLE:
        return {}, "Missing dependency: yfinance is not installed."
    try:
        t = yf.Ticker(yf_symbol)
        info = t.info or {}
        return info, None
    except Exception as e:
        return {}, f"Fundamentals error for {yf_symbol}: {e}"


def format_fundamentals(info: dict) -> Dict[str, Optional[object]]:
    keys = [
        ("shortName", "Name"),
        ("symbol", "Symbol"),
        ("marketCap", "Market Cap"),
        ("currency", "Currency"),
        ("exchange", "Exchange"),
        ("sector", "Sector"),
        ("industry", "Industry"),
        ("trailingPE", "Trailing P/E"),
        ("forwardPE", "Forward P/E"),
        ("priceToBook", "P/B"),
        ("beta", "Beta"),
        ("dividendYield", "Dividend Yield"),
        ("payoutRatio", "Payout Ratio"),
        ("profitMargins", "Profit Margin"),
        ("grossMargins", "Gross Margin"),
        ("operatingMargins", "Operating Margin"),
        ("returnOnEquity", "ROE"),
        ("returnOnAssets", "ROA"),
        ("totalRevenue", "Total Revenue"),
        ("revenueGrowth", "Revenue Growth"),
        ("earningsGrowth", "Earnings Growth"),
        ("freeCashflow", "Free Cash Flow"),
        ("totalCash", "Total Cash"),
        ("totalDebt", "Total Debt"),
        ("currentRatio", "Current Ratio"),
    ]
    out = {}
    for k, label in keys:
        out[label] = info.get(k, None)
    return out


# ---------------------------
# AI Analysis
# ---------------------------
def build_ai_context(df: pd.DataFrame, asset_label: str, interval: str, include: Dict[str, bool], smc_cfg: SMCConfig) -> str:
    """
    Build a compact, indicator-aware context for the LLM.
    We keep it factual: key levels, latest indicator readings, basic structure notes.
    """
    last = df.iloc[-1]
    prior = df.iloc[-2] if len(df) > 1 else last

    ctx = []
    ctx.append(f"Asset: {asset_label}")
    ctx.append(f"Interval: {interval}")
    ctx.append(f"Last close: {float(last['Close']):.6g}")
    ctx.append(f"Last high/low: {float(last['High']):.6g} / {float(last['Low']):.6g}")
    if "Volume" in df.columns and pd.notna(last.get("Volume", np.nan)):
        ctx.append(f"Last volume: {float(last.get('Volume', 0.0)):.6g}")

    # Returns / momentum
    if pd.notna(prior["Close"]) and prior["Close"] != 0:
        ret = (last["Close"] / prior["Close"] - 1) * 100
        ctx.append(f"1-bar return: {ret:.2f}%")

    # Bollinger
    if include.get("bollinger", False):
        bb = add_bollinger(df)
        bb_last = bb.iloc[-1]
        ctx.append(f"Bollinger(20,2): mid={bb_last['BB_MID']:.6g}, upper={bb_last['BB_UPPER']:.6g}, lower={bb_last['BB_LOWER']:.6g}")
        if pd.notna(bb_last["BB_UPPER"]) and pd.notna(bb_last["BB_LOWER"]):
            pos = (last["Close"] - bb_last["BB_LOWER"]) / (bb_last["BB_UPPER"] - bb_last["BB_LOWER"] + 1e-12)
            ctx.append(f"Close position within bands (0=lower,1=upper): {pos:.2f}")

    # VWAP
    if include.get("vwap", False):
        vwap = add_vwap(df).iloc[-1]
        ctx.append(f"VWAP (anchored over selected window): {vwap:.6g}")
        if pd.notna(vwap):
            ctx.append(f"Close - VWAP: {(last['Close'] - vwap):.6g}")

    # MFI
    if include.get("mfi", False):
        mfi = add_mfi(df).iloc[-1]
        ctx.append(f"MFI(14): {mfi:.2f} (80 overbought / 20 oversold heuristic)")

    # Ichimoku
    if include.get("ichimoku", False):
        ic = add_ichimoku(df).iloc[-1]
        ctx.append(
            "Ichimoku: "
            f"Tenkan={ic['IC_TENKAN']:.6g}, Kijun={ic['IC_KIJUN']:.6g}, "
            f"SenkouA={ic['IC_SENKOU_A']:.6g}, SenkouB={ic['IC_SENKOU_B']:.6g}"
        )

    # SMC
    if include.get("smc", False):
        smc_df = smc_market_structure(df, smc_cfg)
        last_row = smc_df.iloc[-1]
        trend = int(last_row.get("SMC_TREND", 0))
        trend_label = "bullish" if trend == 1 else "bearish" if trend == -1 else "neutral/unknown"
        ctx.append(f"SMC structure: trend_state={trend_label}")
        # last swing levels
        lsh = last_row.get("LAST_SWING_HIGH", np.nan)
        lsl = last_row.get("LAST_SWING_LOW", np.nan)
        if pd.notna(lsh) and pd.notna(lsl):
            ctx.append(f"Last swing high/low: {float(lsh):.6g} / {float(lsl):.6g}")
        # most recent events
        recent = smc_df.tail(200)
        events = []
        for col, lab in [("BOS_BULL","BOS↑"),("BOS_BEAR","BOS↓"),("CHOCH_BULL","CHOCH↑"),("CHOCH_BEAR","CHOCH↓")]:
            hits = recent.index[recent[col].fillna(False)]
            if len(hits) > 0:
                events.append(f"{lab} last at {hits[-1].strftime('%Y-%m-%d %H:%M') if hasattr(hits[-1],'strftime') else str(hits[-1])}")
        if events:
            ctx.append("Recent structure events: " + "; ".join(events))

    return "\n".join(ctx)


def ai_analyze_openai(context: str, user_focus: str, model: str) -> Tuple[str, Optional[str]]:
    if not OPENAI_SDK_AVAILABLE:
        return "", "OpenAI SDK not installed. Install: pip install openai"
    api_key = st.secrets.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "", "OPENAI_API_KEY missing in st.secrets."
    try:
        client = OpenAI(api_key=api_key)
        system = (
            "You are an expert macro technical + fundamental analyst. "
            "You explain regimes (risk-on/off, inflation/deflation, liquidity), "
            "and you translate indicators into actionable observation. "
            "Do NOT claim certainty. Do NOT give financial advice. "
            "Provide: (1) Market regime read, (2) Technical state, (3) Key levels, "
            "(4) What would invalidate the current read, (5) 2-3 scenario paths."
        )
        prompt = f"{context}\n\nUser focus:\n{user_focus}\n"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip(), None
    except Exception as e:
        return "", f"OpenAI error: {e}"


def ai_analyze_gemini(context: str, user_focus: str, model: str) -> Tuple[str, Optional[str]]:
    if not GEMINI_AVAILABLE:
        return "", "Gemini SDK not installed. Install: pip install google-generativeai"
    api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "", "GEMINI_API_KEY missing in st.secrets."
    try:
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model)
        system = (
            "You are an expert macro technical + fundamental analyst. "
            "Explain regimes (risk-on/off, inflation/deflation, liquidity) "
            "and translate indicators into structured observations. "
            "Avoid certainty and avoid financial advice. "
            "Provide: (1) Market regime read, (2) Technical state, (3) Key levels, "
            "(4) Invalidation triggers, (5) 2-3 scenario paths."
        )
        prompt = f"{system}\n\nContext:\n{context}\n\nUser focus:\n{user_focus}\n"
        resp = m.generate_content(prompt)
        return (resp.text or "").strip(), None
    except Exception as e:
        return "", f"Gemini error: {e}"


# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Macro Master Analyst Machine", layout="wide")

st.title("📡 Macro Master — Technical + Fundamental + AI Analyst Machine")
st.caption("Macro dashboard built from your watchlists (symbols + ratios). Uses VWAP, MFI, SMC structure, Bollinger Bands, Ichimoku Cloud, and optional AI analysis. Not financial advice.")

if not YF_AVAILABLE:
    st.warning("yfinance is not installed in this environment. Install it to fetch data: `pip install yfinance`")

# Sidebar: Watchlist inputs
st.sidebar.header("📋 Watchlists")

col_a, col_b = st.sidebar.columns(2)
use_defaults = col_a.toggle("Use built-in lists", value=True)
show_raw_lists = col_b.toggle("Show lists", value=False)

uploaded_symbols = st.sidebar.file_uploader("Upload Symbols .txt (one per line)", type=["txt"], key="sym_upload")
uploaded_ratios = st.sidebar.file_uploader("Upload Ratios .txt (one per line)", type=["txt"], key="rat_upload")

symbols_list = []
ratios_list = []

if use_defaults:
    symbols_list = DEFAULT_SYMBOLS.copy()
    ratios_list = DEFAULT_RATIOS.copy()

# Merge uploads (additive)
symbols_list += load_watchlist_from_upload(uploaded_symbols)
ratios_list += load_watchlist_from_upload(uploaded_ratios)

# De-duplicate while preserving order
def dedupe(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

symbols_list = dedupe([s for s in symbols_list if not is_ratio_symbol(s)])
ratios_list = dedupe([r for r in ratios_list if is_ratio_symbol(r)])

if show_raw_lists:
    with st.sidebar.expander("Symbols list (raw)", expanded=False):
        st.write(symbols_list)
    with st.sidebar.expander("Ratios list (raw)", expanded=False):
        st.write(ratios_list)

st.sidebar.divider()

# Mode selection
mode = st.sidebar.radio("Mode", ["Single Asset", "Ratio"], index=0)

# Time controls
st.sidebar.header("⏱️ Time & Data")
interval = st.sidebar.selectbox(
    "Interval",
    options=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "1wk", "1mo"],
    index=7,
    help="Intraday intervals have limited history on yfinance.",
)

days_back_default = 365 if interval in ["1d"] else 60
days_back = st.sidebar.number_input("Lookback (days)", min_value=1, max_value=3650, value=days_back_default, step=1)
end_dt = datetime.utcnow()
start_dt = end_dt - timedelta(days=int(days_back))

st.sidebar.caption(f"Data window: {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')} (UTC)")

# Indicator toggles
st.sidebar.header("🧠 Indicators")
show_vwap = st.sidebar.toggle("VWAP", value=True)
show_mfi = st.sidebar.toggle("MFI", value=True)
show_bollinger = st.sidebar.toggle("Bollinger Bands", value=True)
show_ichimoku = st.sidebar.toggle("Ichimoku Cloud", value=True)

show_smc = st.sidebar.toggle("SMC (Market Structure)", value=True)
smc_cfg = SMCConfig(
    pivot_len=int(st.sidebar.slider("SMC pivot length", 1, 10, 3, 1, help="Higher = fewer, bigger swings")),
    use_close_break=st.sidebar.toggle("SMC breaks use Close", value=True),
    show_swings=st.sidebar.toggle("Show swing points", value=True),
    show_bos=st.sidebar.toggle("Show BOS", value=True),
    show_choch=st.sidebar.toggle("Show CHOCH", value=True),
)

st.sidebar.divider()

# Mapping overrides panel
st.sidebar.header("🧩 Symbol Mapping")
st.sidebar.caption("TradingView symbols are mapped to yfinance symbols. Override any mapping here.")

mapping_overrides: Dict[str, str] = st.session_state.get("mapping_overrides", {})

with st.sidebar.expander("Edit mapping overrides", expanded=False):
    st.write("Add an override for the currently selected TV symbol (or any symbol).")
    override_key = st.text_input("TV Symbol (exact)", value="")
    override_val = st.text_input("yfinance Symbol", value="")
    c1, c2 = st.columns(2)
    if c1.button("Add/Update override"):
        if override_key.strip() and override_val.strip():
            mapping_overrides[override_key.strip()] = override_val.strip()
            st.session_state["mapping_overrides"] = mapping_overrides
            st.success("Override saved.")
        else:
            st.warning("Provide both TV Symbol and yfinance Symbol.")
    if c2.button("Clear all overrides"):
        mapping_overrides = {}
        st.session_state["mapping_overrides"] = mapping_overrides
        st.success("Overrides cleared.")

# AI controls
st.sidebar.header("🤖 AI Analysis")
enable_ai = st.sidebar.toggle("Enable AI", value=False)
ai_provider = st.sidebar.selectbox("Provider", ["OpenAI", "Gemini"], index=0)
openai_model = st.sidebar.text_input("OpenAI model", value="gpt-4.1-mini")
gemini_model = st.sidebar.text_input("Gemini model", value="gemini-1.5-pro")

user_focus = st.sidebar.text_area(
    "AI focus (optional)",
    value="Focus on macro regime, key levels, invalidation, and what this implies for risk-on/off.",
    height=100,
)

# ---------------------------
# Main selection
# ---------------------------
left, right = st.columns([0.62, 0.38])

with left:
    if mode == "Single Asset":
        selected_tv = st.selectbox("Select symbol", symbols_list, index=0 if symbols_list else None)
        if not selected_tv:
            st.stop()
        yf_symbol = tv_to_yf_symbol(selected_tv, mapping_overrides)

        with st.spinner(f"Downloading data for {selected_tv} → {yf_symbol} ..."):
            df, err = yf_download(yf_symbol, interval=interval, start=start_dt, end=end_dt)

        if err:
            st.error(err)
            st.info("Try: (1) change interval, (2) add a mapping override, (3) switch to a supported symbol.")
            st.stop()

        # Plot
        fig = plot_price_with_indicators(
            df=df,
            title=f"{selected_tv}  (source: {yf_symbol})",
            show_bollinger=show_bollinger,
            show_ichimoku=show_ichimoku,
            show_vwap=show_vwap,
            show_smc=show_smc,
            smc_cfg=smc_cfg,
            show_mfi=show_mfi,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Quick stats
        last = df.iloc[-1]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Close", f"{last['Close']:.6g}")
        k2.metric("High", f"{last['High']:.6g}")
        k3.metric("Low", f"{last['Low']:.6g}")
        vol_val = last.get("Volume", np.nan)
        k4.metric("Volume", f"{vol_val:.6g}" if pd.notna(vol_val) else "—")

    else:
        selected_ratio = st.selectbox("Select ratio", ratios_list, index=0 if ratios_list else None)
        if not selected_ratio:
            st.stop()
        a, b = selected_ratio.split("/", 1)
        a = a.strip()
        b = b.strip()
        yf_a = tv_to_yf_symbol(a, mapping_overrides)
        yf_b = tv_to_yf_symbol(b, mapping_overrides)

        with st.spinner(f"Downloading data for ratio: {a} ({yf_a}) / {b} ({yf_b}) ..."):
            df_a, err_a = yf_download(yf_a, interval=interval, start=start_dt, end=end_dt)
            df_b, err_b = yf_download(yf_b, interval=interval, start=start_dt, end=end_dt)

        if err_a:
            st.error(f"{a} → {yf_a}: {err_a}")
        if err_b:
            st.error(f"{b} → {yf_b}: {err_b}")
        if err_a or err_b or df_a.empty or df_b.empty:
            st.info("Fix the mapping overrides or choose another ratio / interval.")
            st.stop()

        # Align on time index
        close_a = df_a["Close"].rename("A")
        close_b = df_b["Close"].rename("B")
        ratio = (close_a / close_b).dropna().rename("RATIO")

        ratio_df = pd.DataFrame({"Close": ratio})
        # Fake OHLC for ratio chart (line chart is enough)
        fig = make_subplots(rows=2 if show_mfi else 1, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.72, 0.28] if show_mfi else [1.0],
                            specs=[[{"secondary_y": False}]] + ([[{"secondary_y": False}]] if show_mfi else []))
        fig.add_trace(go.Scatter(x=ratio_df.index, y=ratio_df["Close"], name="Ratio", mode="lines"), row=1, col=1)

        # Add Bollinger/VWAP on ratio using Close only (VWAP not meaningful without volume)
        if show_bollinger:
            bb = add_bollinger(ratio_df.rename(columns={"Close": "Close"}).assign(Open=np.nan, High=np.nan, Low=np.nan, Volume=np.nan)[["Close"]].join(
                pd.DataFrame({"Open": ratio_df["Close"], "High": ratio_df["Close"], "Low": ratio_df["Close"], "Volume": 0.0}, index=ratio_df.index)
            ))
            fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_MID"], name="BB Mid", mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_UPPER"], name="BB Upper", mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_LOWER"], name="BB Lower", mode="lines"), row=1, col=1)

        # MFI requires volume; skip for ratios unless user has volume series (we don't)
        if show_mfi:
            fig.add_trace(go.Scatter(x=ratio_df.index, y=np.nan * ratio_df["Close"], name="MFI (n/a for ratio)", mode="lines"), row=2, col=1)
            fig.update_yaxes(range=[0, 100], row=2, col=1)

        fig.update_layout(
            title=f"Ratio: {selected_ratio} (source: {yf_a} / {yf_b})",
            xaxis_rangeslider_visible=False,
            height=850 if show_mfi else 650,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Note: VWAP/MFI are not meaningful on ratios unless you have volume for the ratio series. Use Bollinger + structure + regime context.")

# ---------------------------
# Right panel: Fundamentals + AI
# ---------------------------
with right:
    st.subheader("🧾 Fundamentals (where applicable)")
    st.caption("Fundamentals are best for equities. Futures/indices/crypto often have limited or no fundamentals via yfinance.")

    if mode == "Single Asset":
        tv_symbol = selected_tv
        yf_symbol = tv_to_yf_symbol(tv_symbol, mapping_overrides)

        info, ferr = fetch_fundamentals_yf(yf_symbol) if YF_AVAILABLE else ({}, "yfinance unavailable")
        if ferr:
            st.warning(ferr)
        else:
            f = format_fundamentals(info)
            st.dataframe(pd.DataFrame([f]).T.rename(columns={0: "Value"}), use_container_width=True)

    st.divider()
    st.subheader("🤖 AI Macro/TA Narrative")
    if not enable_ai:
        st.info("Enable AI in the sidebar to generate an expert regime + TA narrative.")
    else:
        if mode == "Single Asset":
            ctx = build_ai_context(
                df=df,
                asset_label=f"{selected_tv} (source: {tv_to_yf_symbol(selected_tv, mapping_overrides)})",
                interval=interval,
                include={"bollinger": show_bollinger, "vwap": show_vwap, "mfi": show_mfi, "ichimoku": show_ichimoku, "smc": show_smc},
                smc_cfg=smc_cfg,
            )
        else:
            # For ratios, build context on ratio close series only (no volume)
            ctx = f"Asset: Ratio {selected_ratio}\nInterval: {interval}\nNote: Volume-based indicators not available for ratios via this data source.\n"

        st.text_area("AI context (what the model sees)", value=ctx, height=180)

        if st.button("Generate AI Analysis", type="primary"):
            with st.spinner("Running AI analysis..."):
                if ai_provider == "OpenAI":
                    out, e = ai_analyze_openai(ctx, user_focus=user_focus, model=openai_model)
                else:
                    out, e = ai_analyze_gemini(ctx, user_focus=user_focus, model=gemini_model)

            if e:
                st.error(e)
                st.info("Check your API key in `.streamlit/secrets.toml` and ensure the required SDK is installed.")
            else:
                st.markdown(out)

    st.divider()
    st.subheader("⚠️ Transparency")
    st.write(
        "- Some TradingView symbols (especially ICE products and CRYPTOCAP series) may not be fetchable via yfinance.\n"
        "- Use the mapping override panel to map any TV symbol to a valid yfinance ticker.\n"
        "- Intraday data history is limited by Yahoo Finance."
    )
