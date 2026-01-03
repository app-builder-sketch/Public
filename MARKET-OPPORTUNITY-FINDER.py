# app.py
# =============================================================================
# MARKET OPPORTUNITY FINDER — Professional Macro + Multi-Market TA/Fundamental + AI
#
# What this app does
# - Ingests watchlists (upload .txt one symbol per line) across "all markets"
# - Fetches OHLCV via Yahoo Finance (yfinance) with transparent failures + mapping overrides
# - Runs a professional Opportunity Scanner:
#     Technical: VWAP, MFI, Bollinger Bands (squeeze/band position), Ichimoku Cloud,
#               SMC-like Market Structure (pivots, BOS/CHOCH, swing levels)
#     Fundamentals (equities only): valuation/growth/quality signals where available from yfinance
# - Ranks & filters opportunities with a composite Opportunity Score + Risk Meter (1–10)
# - Provides expert Plotly charts (candles + overlays + indicator panes)
# - Optional AI narrative analysis (OpenAI or Gemini) that explains reasoning + scenarios
#
# IMPORTANT
# - This is a research tool, not financial advice.
# - Some TradingView symbols won’t resolve via free data sources; use mapping overrides.
# - Yahoo intraday history is limited; daily/weekly is more stable.
#
# =============================================================================

from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
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

from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================================================================
# Defaults / Helpers
# =============================================================================

APP_TITLE = "Market Opportunity Finder — TA/Fundamentals + AI"
DISCLAIMER = (
    "⚠️ **Research tool only (not financial advice).** "
    "Markets can move against you. Rankings are heuristic."
)

DEFAULT_SYMBOLS: List[str] = [
    # Benchmarks / macro signals
    "^GSPC", "^NDX", "^DJI", "^VIX", "DX-Y.NYB", "GC=F", "SI=F", "HG=F", "CL=F",
    # Large tech (examples)
    "NVDA", "AMD", "AVGO", "MSFT", "AAPL", "AMZN", "GOOGL", "META",
    # Industrials / power / infra
    "VRT", "ETN", "PWR", "TT", "HUBB",
    # Miners / commodities proxies
    "FCX", "RIO", "BHP",
    # Defense
    "LMT", "NOC", "RTX",
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD",
]

DEFAULT_RATIOS: List[str] = [
    "GC=F/^GSPC",
    "SI=F/^GSPC",
    "HG=F/GC=F",
    "CL=F/^GSPC",
    "BTC-USD/^GSPC",
    "BTC-USD/GC=F",
    "^NDX/^GSPC",
    "^GSPC/^VIX",
]

# TradingView-style mapping (optional)
TV_TO_YF: Dict[str, str] = {
    "SP:SPX": "^GSPC",
    "NASDAQ:NDX": "^NDX",
    "DJ:DJI": "^DJI",
    "TVC:VIX": "^VIX",
    "TVC:DXY": "DX-Y.NYB",
    "COMEX:GC1!": "GC=F",
    "COMEX:SI1!": "SI=F",
    "COMEX:HG1!": "HG=F",
    "NYMEX:CL1!": "CL=F",
    "ICE:BRN1!": "BZ=F",
    "MEXC:BTCUSDT": "BTC-USD",
    "MEXC:ETHUSDT": "ETH-USD",
    "MEXC:SOLUSDT": "SOL-USD",
    "XAUUSD": "XAUUSD=X",
    "XAGUSD": "XAGUSD=X",
}


def safe_split_lines(text: str) -> List[str]:
    out = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def is_ratio_symbol(s: str) -> bool:
    return "/" in s and not s.startswith("http")


def tv_to_yf_symbol(symbol: str, overrides: Dict[str, str]) -> str:
    """
    Converts TradingView-ish symbols to yfinance symbols with override support.
    If it's already a yfinance-like symbol, returns as-is.
    """
    if symbol in overrides and overrides[symbol].strip():
        return overrides[symbol].strip()

    if is_ratio_symbol(symbol):
        return symbol

    if symbol in TV_TO_YF:
        return TV_TO_YF[symbol]

    # If already yfinance-like
    if any(x in symbol for x in ["=F", "^", "-USD", ".L", ".DE", ".HK", ".SS", ".SZ", ".TW", ".KS", ".SW", ".PA", "=X"]):
        return symbol

    # Strip TV exchange prefix
    if ":" in symbol:
        return symbol.split(":", 1)[1]

    return symbol


def dedupe(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# =============================================================================
# Data Fetching
# =============================================================================

@st.cache_data(show_spinner=False, ttl=60 * 10)
def yf_download(symbol: str, interval: str, start: datetime, end: datetime) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Returns OHLCV dataframe and error message if any.
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

        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[-1] for c in df.columns]

        needed = {"Open", "High", "Low", "Close"}
        if not needed.issubset(set(df.columns)):
            return pd.DataFrame(), f"Data for {symbol} missing required OHLC columns."

        if "Volume" not in df.columns:
            df["Volume"] = np.nan

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"yfinance error for {symbol}: {e}"


@st.cache_data(show_spinner=False, ttl=60 * 30)
def yf_info(symbol: str) -> Tuple[dict, Optional[str]]:
    """
    Fundamentals via yfinance .info (best-effort).
    """
    if not YF_AVAILABLE:
        return {}, "Missing dependency: yfinance is not installed."
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        return info, None
    except Exception as e:
        return {}, f"Fundamentals error for {symbol}: {e}"


# =============================================================================
# Indicators (pure pandas)
# =============================================================================

def add_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].fillna(0.0)
    pv = tp * vol
    cum_vol = vol.cumsum()
    cum_pv = pv.cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    return vwap.rename("VWAP")


def add_mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    rmf = tp * df["Volume"].fillna(0.0)
    tp_prev = tp.shift(1)

    pos = rmf.where(tp > tp_prev, 0.0)
    neg = rmf.where(tp < tp_prev, 0.0)

    pos_sum = pos.rolling(length).sum()
    neg_sum = neg.rolling(length).sum().replace(0, np.nan)

    mfr = pos_sum / neg_sum
    mfi = 100 - (100 / (1 + mfr))
    return mfi.rename("MFI")


def add_bollinger(df: pd.DataFrame, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
    close = df["Close"]
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    upper = mid + mult * sd
    lower = mid - mult * sd
    bw = (upper - lower) / mid.replace(0, np.nan)
    out = pd.DataFrame(
        {"BB_MID": mid, "BB_UPPER": upper, "BB_LOWER": lower, "BB_BW": bw},
        index=df.index,
    )
    return out


def add_ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> pd.DataFrame:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2.0
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2.0
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun)
    senkou_b_line = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2.0).shift(kijun)
    chikou = close.shift(-kijun)

    return pd.DataFrame(
        {
            "IC_TENKAN": tenkan_sen,
            "IC_KIJUN": kijun_sen,
            "IC_SENKOU_A": senkou_a,
            "IC_SENKOU_B": senkou_b_line,
            "IC_CHIKOU": chikou,
        },
        index=df.index,
    )


# =============================================================================
# SMC-lite: pivots + BOS/CHOCH
# =============================================================================

@dataclass
class SMCConfig:
    pivot_len: int = 3
    use_close_break: bool = True


def detect_pivots(df: pd.DataFrame, lr: int) -> Tuple[pd.Series, pd.Series]:
    lr = max(int(lr), 1)
    win = 2 * lr + 1

    ph = df["High"].rolling(win, center=True).max()
    pl = df["Low"].rolling(win, center=True).min()

    pivot_high = df["High"].where(df["High"] == ph, np.nan)
    pivot_low = df["Low"].where(df["Low"] == pl, np.nan)
    return pivot_high.rename("PIVOT_HIGH"), pivot_low.rename("PIVOT_LOW")


def smc_structure(df: pd.DataFrame, cfg: SMCConfig) -> pd.DataFrame:
    d = df.copy()
    ph, pl = detect_pivots(d, cfg.pivot_len)
    d["PIVOT_HIGH"] = ph
    d["PIVOT_LOW"] = pl
    d["LAST_SWING_HIGH"] = d["PIVOT_HIGH"].ffill()
    d["LAST_SWING_LOW"] = d["PIVOT_LOW"].ffill()

    if cfg.use_close_break:
        up_break = d["Close"] > d["LAST_SWING_HIGH"].shift(1)
        dn_break = d["Close"] < d["LAST_SWING_LOW"].shift(1)
    else:
        up_break = d["High"] > d["LAST_SWING_HIGH"].shift(1)
        dn_break = d["Low"] < d["LAST_SWING_LOW"].shift(1)

    trend = np.zeros(len(d), dtype=int)
    last = 0
    for i in range(len(d)):
        if bool(up_break.iloc[i]):
            last = 1
        elif bool(dn_break.iloc[i]):
            last = -1
        trend[i] = last

    d["SMC_TREND"] = trend
    prev_trend = pd.Series(trend, index=d.index).shift(1).fillna(0).astype(int)

    d["BOS_BULL"] = up_break & (prev_trend != 1)
    d["BOS_BEAR"] = dn_break & (prev_trend != -1)

    d["CHOCH_BULL"] = up_break & (prev_trend == -1)
    d["CHOCH_BEAR"] = dn_break & (prev_trend == 1)

    return d


# =============================================================================
# Opportunity Scoring
# =============================================================================

def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0
    return (s - mu) / sd


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def compute_risk_meter(df: pd.DataFrame, info: Optional[dict]) -> int:
    """
    Heuristic risk meter 1–10:
    - Higher vol => higher risk
    - Smaller market cap (if available) => higher risk
    - Higher beta (if available) => higher risk
    """
    close = df["Close"].dropna()
    if len(close) < 30:
        return 7

    rets = close.pct_change().dropna()
    vol = float(rets.std(ddof=0) * math.sqrt(252))  # annualized-ish
    vol_score = clamp((vol - 0.15) / (0.80 - 0.15), 0, 1)  # map 15%..80% to 0..1

    beta_score = 0.5
    mcap_score = 0.5

    if info:
        beta = info.get("beta", None)
        if isinstance(beta, (int, float)) and not np.isnan(beta):
            beta_score = clamp((float(beta) - 0.8) / (2.5 - 0.8), 0, 1)

        mcap = info.get("marketCap", None)
        if isinstance(mcap, (int, float)) and not np.isnan(mcap):
            # >200B => low, <2B => high
            if mcap >= 2e11:
                mcap_score = 0.0
            elif mcap <= 2e9:
                mcap_score = 1.0
            else:
                # log-scale interpolation
                mcap_score = clamp((math.log10(2e11) - math.log10(float(mcap))) / (math.log10(2e11) - math.log10(2e9)), 0, 1)

    raw = 0.55 * vol_score + 0.25 * beta_score + 0.20 * mcap_score
    risk = int(round(1 + raw * 9))
    return int(clamp(risk, 1, 10))


def fundamentals_score(info: dict) -> Tuple[float, Dict[str, Any]]:
    """
    Returns fundamentals score 0..100 and a breakdown dict.
    Best-effort: many assets will not have these fields.
    """
    if not info:
        return 50.0, {"note": "No fundamentals available."}

    pe = info.get("trailingPE", None)
    fpe = info.get("forwardPE", None)
    pb = info.get("priceToBook", None)
    rev_g = info.get("revenueGrowth", None)
    earn_g = info.get("earningsGrowth", None)
    roe = info.get("returnOnEquity", None)
    fcf = info.get("freeCashflow", None)
    margins = info.get("operatingMargins", None)
    debt = info.get("totalDebt", None)
    cash = info.get("totalCash", None)

    # Valuation subscore (lower is better; clamp to plausible ranges)
    val = 50.0
    val_parts = {}
    if isinstance(pe, (int, float)) and pe > 0 and not np.isnan(pe):
        # PE 8..35 => 100..0
        val_parts["PE"] = float(pe)
        val += clamp((35 - pe) / (35 - 8), 0, 1) * 25 - 12.5
    if isinstance(pb, (int, float)) and pb > 0 and not np.isnan(pb):
        val_parts["P/B"] = float(pb)
        val += clamp((6 - pb) / (6 - 1), 0, 1) * 15 - 7.5

    # Growth subscore
    growth = 50.0
    growth_parts = {}
    if isinstance(rev_g, (int, float)) and not np.isnan(rev_g):
        growth_parts["RevenueGrowth"] = float(rev_g)
        growth += clamp((rev_g - 0.00) / (0.30 - 0.00), 0, 1) * 25 - 12.5
    if isinstance(earn_g, (int, float)) and not np.isnan(earn_g):
        growth_parts["EarningsGrowth"] = float(earn_g)
        growth += clamp((earn_g - 0.00) / (0.40 - 0.00), 0, 1) * 25 - 12.5

    # Quality subscore
    qual = 50.0
    qual_parts = {}
    if isinstance(roe, (int, float)) and not np.isnan(roe):
        qual_parts["ROE"] = float(roe)
        qual += clamp((roe - 0.05) / (0.25 - 0.05), 0, 1) * 25 - 12.5
    if isinstance(margins, (int, float)) and not np.isnan(margins):
        qual_parts["OperatingMargins"] = float(margins)
        qual += clamp((margins - 0.05) / (0.30 - 0.05), 0, 1) * 25 - 12.5
    if isinstance(fcf, (int, float)) and not np.isnan(fcf):
        qual_parts["FreeCashFlow"] = float(fcf)
        qual += 10 if fcf > 0 else -10
    if isinstance(debt, (int, float)) and isinstance(cash, (int, float)) and not np.isnan(debt) and not np.isnan(cash):
        qual_parts["DebtMinusCash"] = float(debt - cash)
        qual += 5 if cash >= debt else -5

    # Blend
    score = (0.35 * val + 0.30 * growth + 0.35 * qual)
    score = clamp(score, 0, 100)

    breakdown = {
        "valuation": clamp(val, 0, 100),
        "growth": clamp(growth, 0, 100),
        "quality": clamp(qual, 0, 100),
        "fields": {**val_parts, **growth_parts, **qual_parts},
    }
    return float(score), breakdown


def technical_features(df: pd.DataFrame, smc_cfg: SMCConfig) -> Dict[str, Any]:
    """
    Compute technical features used in scoring and explanations.
    """
    d = df.copy()
    if len(d) < 60:
        return {"ok": False, "note": "Not enough bars for robust signals."}

    bb = add_bollinger(d)
    ic = add_ichimoku(d)
    vwap = add_vwap(d)
    mfi = add_mfi(d)

    s = smc_structure(d, smc_cfg)

    last = d.iloc[-1]
    last_bb = bb.iloc[-1]
    last_ic = ic.iloc[-1]
    last_vwap = vwap.iloc[-1]
    last_mfi = mfi.iloc[-1]
    last_s = s.iloc[-1]

    close = float(last["Close"])
    # Bollinger position + squeeze
    bb_pos = np.nan
    if pd.notna(last_bb["BB_UPPER"]) and pd.notna(last_bb["BB_LOWER"]):
        bb_pos = float((close - last_bb["BB_LOWER"]) / (last_bb["BB_UPPER"] - last_bb["BB_LOWER"] + 1e-12))

    bw = bb["BB_BW"].dropna()
    bw_pct = float((bw.rank(pct=True).iloc[-1] * 100) if len(bw) else np.nan)

    # Ichimoku trend proxy: close above cloud + tenkan>kijun
    cloud_top = np.nanmax([last_ic["IC_SENKOU_A"], last_ic["IC_SENKOU_B"]])
    cloud_bot = np.nanmin([last_ic["IC_SENKOU_A"], last_ic["IC_SENKOU_B"]])
    above_cloud = pd.notna(cloud_top) and close > cloud_top
    below_cloud = pd.notna(cloud_bot) and close < cloud_bot
    tk_bull = pd.notna(last_ic["IC_TENKAN"]) and pd.notna(last_ic["IC_KIJUN"]) and (last_ic["IC_TENKAN"] > last_ic["IC_KIJUN"])

    # VWAP distance
    vwap_dist = float((close - last_vwap) / close) if pd.notna(last_vwap) and close != 0 else np.nan

    # MFI level
    mfi_val = float(last_mfi) if pd.notna(last_mfi) else np.nan

    # SMC events in last N bars
    recent = s.tail(120)
    bos_bull_recent = bool(recent["BOS_BULL"].fillna(False).any())
    bos_bear_recent = bool(recent["BOS_BEAR"].fillna(False).any())
    choch_bull_recent = bool(recent["CHOCH_BULL"].fillna(False).any())
    choch_bear_recent = bool(recent["CHOCH_BEAR"].fillna(False).any())

    # Distance to last swing high/low
    lsh = last_s.get("LAST_SWING_HIGH", np.nan)
    lsl = last_s.get("LAST_SWING_LOW", np.nan)
    dist_to_swing_high = float((lsh - close) / close) if pd.notna(lsh) and close else np.nan
    dist_to_swing_low = float((close - lsl) / close) if pd.notna(lsl) and close else np.nan

    # Momentum returns
    close_series = d["Close"].dropna()
    r_20 = float(close / close_series.iloc[-21] - 1) if len(close_series) >= 21 else np.nan
    r_60 = float(close / close_series.iloc[-61] - 1) if len(close_series) >= 61 else np.nan

    return {
        "ok": True,
        "close": close,
        "bb_pos": bb_pos,
        "bb_bandwidth_pct": bw_pct,
        "above_cloud": bool(above_cloud),
        "below_cloud": bool(below_cloud),
        "tk_bull": bool(tk_bull),
        "vwap_dist": vwap_dist,
        "mfi": mfi_val,
        "bos_bull_recent": bos_bull_recent,
        "bos_bear_recent": bos_bear_recent,
        "choch_bull_recent": choch_bull_recent,
        "choch_bear_recent": choch_bear_recent,
        "dist_to_swing_high": dist_to_swing_high,
        "dist_to_swing_low": dist_to_swing_low,
        "r_20": r_20,
        "r_60": r_60,
    }


def opportunity_score(tech: Dict[str, Any], fund_score: Optional[float], asset_has_fundamentals: bool) -> Tuple[float, Dict[str, Any]]:
    """
    Composite score 0..100 with explanation breakdown.
    Philosophy:
    - Prefer assets that are:
        * in bullish structure / regime (Ichimoku + SMC)
        * near "value zones" (VWAP / lower BB) OR breaking out from squeeze with structure confirmation
        * improving flow (MFI not extremely overheated unless breakout)
    - If fundamentals exist, blend them in (value/growth/quality)
    """
    if not tech.get("ok", False):
        return 0.0, {"note": tech.get("note", "No signal.")}

    # Start neutral
    score = 50.0
    parts = {}

    # Regime / trend alignment
    trend_pts = 0.0
    if tech["above_cloud"] and tech["tk_bull"]:
        trend_pts += 12.0
    if tech["below_cloud"]:
        trend_pts -= 10.0
    if tech["bos_bull_recent"] or tech["choch_bull_recent"]:
        trend_pts += 8.0
    if tech["bos_bear_recent"] or tech["choch_bear_recent"]:
        trend_pts -= 8.0
    parts["trend_structure"] = trend_pts
    score += trend_pts

    # Mean-reversion-to-trend: if price is below vwap but structure not broken, good "value entry zone"
    mr_pts = 0.0
    vwap_dist = tech.get("vwap_dist", np.nan)
    if pd.notna(vwap_dist):
        if vwap_dist < -0.03:
            mr_pts += 10.0
        elif vwap_dist < -0.01:
            mr_pts += 6.0
        elif vwap_dist > 0.05:
            mr_pts -= 4.0
    parts["vwap_zone"] = mr_pts
    score += mr_pts

    # Bollinger position + squeeze logic
    bb_pts = 0.0
    bb_pos = tech.get("bb_pos", np.nan)
    bw_pct = tech.get("bb_bandwidth_pct", np.nan)

    # Squeeze: low bandwidth percentile can precede expansion; reward if structure is bullish
    if pd.notna(bw_pct) and bw_pct <= 20:
        bb_pts += 6.0
        if tech["bos_bull_recent"] or (tech["above_cloud"] and tech["tk_bull"]):
            bb_pts += 4.0  # squeeze + bullish alignment

    # Where are we inside bands? Lower band zones can be "value"
    if pd.notna(bb_pos):
        if bb_pos < 0.15:
            bb_pts += 6.0
        elif bb_pos > 0.90:
            bb_pts -= 3.0
    parts["bollinger"] = bb_pts
    score += bb_pts

    # Flow (MFI): oversold is opportunity; extremely overbought can be late
    mfi_pts = 0.0
    mfi = tech.get("mfi", np.nan)
    if pd.notna(mfi):
        if mfi < 25:
            mfi_pts += 8.0
        elif mfi < 35:
            mfi_pts += 4.0
        elif mfi > 80:
            mfi_pts -= 4.0
    parts["mfi_flow"] = mfi_pts
    score += mfi_pts

    # Momentum (prefer positive 3M, not parabolic)
    mom_pts = 0.0
    r60 = tech.get("r_60", np.nan)
    if pd.notna(r60):
        if r60 > 0.20:
            mom_pts += 6.0
        elif r60 > 0.05:
            mom_pts += 3.0
        elif r60 < -0.20:
            mom_pts -= 6.0
        elif r60 < -0.10:
            mom_pts -= 3.0
    parts["momentum_60"] = mom_pts
    score += mom_pts

    # Fundamentals blend (equities only)
    fund_pts = 0.0
    if asset_has_fundamentals and fund_score is not None:
        # Map fund_score (0..100) to -8..+8 around 50
        fund_pts = (fund_score - 50.0) / 50.0 * 8.0
    parts["fundamentals"] = fund_pts
    score += fund_pts

    score = clamp(score, 0, 100)
    parts["final_score"] = score
    return float(score), parts


# =============================================================================
# Plotly Charting
# =============================================================================

def plot_asset(df: pd.DataFrame, title: str, show_volume: bool, show_bb: bool, show_vwap: bool, show_ichimoku: bool,
               show_mfi: bool, show_smc: bool, smc_cfg: SMCConfig) -> go.Figure:
    rows = 2 if show_mfi else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.72, 0.28] if show_mfi else [1.0],
        specs=[[{"secondary_y": False}]] + ([[{"secondary_y": False}]] if show_mfi else [])
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Price"
        ),
        row=1, col=1
    )

    if show_volume and df["Volume"].notna().any():
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.25),
            row=1, col=1
        )

    if show_bb:
        bb = add_bollinger(df)
        fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_MID"], name="BB Mid", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_UPPER"], name="BB Upper", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_LOWER"], name="BB Lower", mode="lines"), row=1, col=1)

    if show_vwap:
        vwap = add_vwap(df)
        fig.add_trace(go.Scatter(x=vwap.index, y=vwap, name="VWAP", mode="lines"), row=1, col=1)

    if show_ichimoku:
        ic = add_ichimoku(df)
        fig.add_trace(go.Scatter(x=ic.index, y=ic["IC_TENKAN"], name="Tenkan", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ic.index, y=ic["IC_KIJUN"], name="Kijun", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ic.index, y=ic["IC_SENKOU_A"], name="Senkou A", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ic.index, y=ic["IC_SENKOU_B"], name="Senkou B", mode="lines"), row=1, col=1)

        # Cloud fill (A->B)
        fig.add_trace(
            go.Scatter(
                x=ic.index, y=ic["IC_SENKOU_A"],
                name="Cloud A (fill)", mode="lines",
                line=dict(width=0), showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=ic.index, y=ic["IC_SENKOU_B"],
                name="Cloud", mode="lines",
                fill="tonexty", line=dict(width=0),
                opacity=0.15, showlegend=False
            ),
            row=1, col=1
        )

    if show_smc:
        s = smc_structure(df, smc_cfg)
        # Swing points (circles, not triangles)
        ph = s["PIVOT_HIGH"]
        pl = s["PIVOT_LOW"]
        fig.add_trace(go.Scatter(x=ph.index, y=ph, name="Swing High", mode="markers",
                                 marker=dict(size=7, symbol="circle")), row=1, col=1)
        fig.add_trace(go.Scatter(x=pl.index, y=pl, name="Swing Low", mode="markers",
                                 marker=dict(size=7, symbol="circle")), row=1, col=1)

        # BOS/CHOCH marks
        def mark(flag_col: str, label: str):
            idx = s.index[s[flag_col].fillna(False)]
            if len(idx) == 0:
                return
            y = s.loc[idx, "Close"]
            fig.add_trace(
                go.Scatter(
                    x=idx, y=y, name=label,
                    mode="markers+text",
                    text=[label]*len(idx),
                    textposition="top center",
                    marker=dict(size=10, symbol="circle-open")
                ),
                row=1, col=1
            )
        mark("BOS_BULL", "BOS↑")
        mark("BOS_BEAR", "BOS↓")
        mark("CHOCH_BULL", "CHOCH↑")
        mark("CHOCH_BEAR", "CHOCH↓")

    if show_mfi:
        mfi = add_mfi(df)
        fig.add_trace(go.Scatter(x=mfi.index, y=mfi, name="MFI", mode="lines"), row=2, col=1)
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


# =============================================================================
# AI Analysis
# =============================================================================

def get_secret(name: str) -> str:
    """
    st.secrets first, then environment variables.
    """
    v = ""
    try:
        v = (st.secrets.get(name, "") or "").strip()
    except Exception:
        v = ""
    if not v:
        v = (os.getenv(name, "") or "").strip()
    return v


def build_ai_context(symbol_label: str, interval: str, tech: Dict[str, Any], fund: Optional[dict], fund_breakdown: Optional[dict],
                     score: float, risk: int) -> str:
    """
    Compact factual context for the model.
    """
    lines = []
    lines.append(f"Asset: {symbol_label}")
    lines.append(f"Interval: {interval}")
    lines.append(f"OpportunityScore(0-100): {score:.1f}")
    lines.append(f"RiskMeter(1-10): {risk}")

    if tech.get("ok"):
        lines.append("--- Technical Snapshot ---")
        lines.append(f"Close: {tech.get('close')}")
        lines.append(f"VWAP distance (close-vwap)/close: {tech.get('vwap_dist')}")
        lines.append(f"MFI(14): {tech.get('mfi')}")
        lines.append(f"Bollinger position (0=lower,1=upper): {tech.get('bb_pos')}")
        lines.append(f"BB bandwidth percentile (lower=squeeze): {tech.get('bb_bandwidth_pct')}")
        lines.append(f"Ichimoku above cloud: {tech.get('above_cloud')} | below cloud: {tech.get('below_cloud')} | tenkan>kijun: {tech.get('tk_bull')}")
        lines.append(f"SMC recent: BOS_bull={tech.get('bos_bull_recent')} BOS_bear={tech.get('bos_bear_recent')} CHOCH_bull={tech.get('choch_bull_recent')} CHOCH_bear={tech.get('choch_bear_recent')}")
        lines.append(f"Distance to swing high: {tech.get('dist_to_swing_high')} | swing low: {tech.get('dist_to_swing_low')}")
        lines.append(f"Return 20 bars: {tech.get('r_20')} | Return 60 bars: {tech.get('r_60')}")

    if fund and fund_breakdown:
        lines.append("--- Fundamentals Snapshot (best-effort) ---")
        lines.append(f"Name: {fund.get('shortName')} | Sector: {fund.get('sector')} | Industry: {fund.get('industry')}")
        lines.append(f"MarketCap: {fund.get('marketCap')} | Currency: {fund.get('currency')} | Beta: {fund.get('beta')}")
        lines.append(f"TrailingPE: {fund.get('trailingPE')} | ForwardPE: {fund.get('forwardPE')} | P/B: {fund.get('priceToBook')}")
        lines.append(f"RevenueGrowth: {fund.get('revenueGrowth')} | EarningsGrowth: {fund.get('earningsGrowth')}")
        lines.append(f"OperatingMargins: {fund.get('operatingMargins')} | ROE: {fund.get('returnOnEquity')} | FCF: {fund.get('freeCashflow')}")
        lines.append(f"FundSubscores: valuation={fund_breakdown.get('valuation')} growth={fund_breakdown.get('growth')} quality={fund_breakdown.get('quality')}")

    return "\n".join(lines)


def ai_openai(context: str, user_focus: str, model: str) -> Tuple[str, Optional[str]]:
    if not OPENAI_SDK_AVAILABLE:
        return "", "OpenAI SDK not installed. `pip install openai`"
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return "", "OPENAI_API_KEY missing in st.secrets or environment."
    try:
        client = OpenAI(api_key=api_key)
        system = (
            "You are an expert multi-asset technical + fundamental analyst and macro regime interpreter. "
            "You must be explicit about uncertainty. Do NOT give financial advice. "
            "Output must be structured:\n"
            "1) Regime/Context\n"
            "2) Technical read (VWAP, MFI, Bollinger, Ichimoku, SMC)\n"
            "3) Key levels (support/resistance, swing levels)\n"
            "4) Opportunity thesis (why this could work)\n"
            "5) Invalidation (what breaks the thesis)\n"
            "6) Scenarios (2-3 paths) + what to watch next\n"
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


def ai_gemini(context: str, user_focus: str, model: str) -> Tuple[str, Optional[str]]:
    if not GEMINI_AVAILABLE:
        return "", "Gemini SDK not installed. `pip install google-generativeai`"
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        return "", "GEMINI_API_KEY missing in st.secrets or environment."
    try:
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model)
        system = (
            "You are an expert multi-asset technical + fundamental analyst and macro regime interpreter. "
            "Be explicit about uncertainty. Do NOT give financial advice. "
            "Structured output:\n"
            "1) Regime/Context\n"
            "2) Technical read (VWAP, MFI, Bollinger, Ichimoku, SMC)\n"
            "3) Key levels\n"
            "4) Opportunity thesis\n"
            "5) Invalidation\n"
            "6) Scenarios + what to watch\n"
        )
        prompt = f"{system}\n\nContext:\n{context}\n\nUser focus:\n{user_focus}\n"
        resp = m.generate_content(prompt)
        return (resp.text or "").strip(), None
    except Exception as e:
        return "", f"Gemini error: {e}"


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("📡 Market Opportunity Finder — TA/Fundamentals + AI")
st.markdown(DISCLAIMER)

if not YF_AVAILABLE:
    st.warning("yfinance is not installed in this environment. Install it: `pip install yfinance`")

# Sidebar — Watchlists
st.sidebar.header("📋 Watchlists")

use_defaults = st.sidebar.toggle("Use built-in sample watchlist", value=True)
uploaded_symbols = st.sidebar.file_uploader("Upload Symbols .txt (one per line)", type=["txt"], key="sym_upload")
uploaded_ratios = st.sidebar.file_uploader("Upload Ratios .txt (one per line)", type=["txt"], key="rat_upload")

symbols: List[str] = []
ratios: List[str] = []
if use_defaults:
    symbols += DEFAULT_SYMBOLS
    ratios += DEFAULT_RATIOS

if uploaded_symbols is not None:
    try:
        content = uploaded_symbols.getvalue().decode("utf-8", errors="replace")
        symbols += safe_split_lines(content)
    except Exception as e:
        st.sidebar.error(f"Could not read symbols file: {e}")

if uploaded_ratios is not None:
    try:
        content = uploaded_ratios.getvalue().decode("utf-8", errors="replace")
        ratios += safe_split_lines(content)
    except Exception as e:
        st.sidebar.error(f"Could not read ratios file: {e}")

symbols = dedupe([s for s in symbols if not is_ratio_symbol(s)])
ratios = dedupe([r for r in ratios if is_ratio_symbol(r)])

# Mapping overrides
st.sidebar.divider()
st.sidebar.header("🧩 Symbol Mapping Overrides")
st.sidebar.caption("Use this when a TradingView-style symbol won't fetch via Yahoo Finance.")
mapping_overrides: Dict[str, str] = st.session_state.get("mapping_overrides", {})

with st.sidebar.expander("Edit overrides", expanded=False):
    k = st.text_input("TV Symbol (exact)", value="")
    v = st.text_input("yfinance Symbol", value="")
    c1, c2 = st.columns(2)
    if c1.button("Add/Update override"):
        if k.strip() and v.strip():
            mapping_overrides[k.strip()] = v.strip()
            st.session_state["mapping_overrides"] = mapping_overrides
            st.success("Saved.")
        else:
            st.warning("Provide both fields.")
    if c2.button("Clear overrides"):
        mapping_overrides = {}
        st.session_state["mapping_overrides"] = mapping_overrides
        st.success("Cleared.")

# Sidebar — Data settings
st.sidebar.divider()
st.sidebar.header("⏱️ Data Settings")

mode = st.sidebar.radio("Mode", ["Scanner (Rank Opportunities)", "Single Chart (Deep Dive)", "Ratio Chart"], index=0)

interval = st.sidebar.selectbox(
    "Interval",
    options=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "1wk", "1mo"],
    index=7,
    help="Intraday history is limited on Yahoo; daily/weekly works best for multi-year scanning.",
)
days_back_default = 365 if interval in ["1d"] else 60
days_back = st.sidebar.number_input("Lookback (days)", min_value=30, max_value=3650, value=days_back_default, step=10)
end_dt = datetime.utcnow()
start_dt = end_dt - timedelta(days=int(days_back))

st.sidebar.caption(f"Window (UTC): {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}")

# Sidebar — Indicator settings
st.sidebar.divider()
st.sidebar.header("🧠 Analysis Settings")

show_volume = st.sidebar.toggle("Show volume", value=True)
show_bb = st.sidebar.toggle("Bollinger Bands", value=True)
show_vwap = st.sidebar.toggle("VWAP", value=True)
show_ichimoku = st.sidebar.toggle("Ichimoku", value=True)
show_mfi = st.sidebar.toggle("MFI", value=True)
show_smc = st.sidebar.toggle("SMC Structure (BOS/CHOCH)", value=True)

smc_cfg = SMCConfig(
    pivot_len=int(st.sidebar.slider("SMC pivot length", 1, 10, 3, 1)),
    use_close_break=st.sidebar.toggle("SMC break uses Close", value=True),
)

# Sidebar — Scanner controls
st.sidebar.divider()
st.sidebar.header("🔎 Scanner Controls")

max_symbols = st.sidebar.number_input("Max symbols to scan", min_value=10, max_value=2000, value=min(250, max(10, len(symbols))), step=10)
threads = st.sidebar.slider("Parallel threads", 1, 32, 10, 1)

min_score = st.sidebar.slider("Min Opportunity Score", 0, 100, 60, 1)
max_risk = st.sidebar.slider("Max Risk Meter (1–10)", 1, 10, 10, 1)

prefer_equities = st.sidebar.toggle("Prefer equities (use fundamentals when possible)", value=True)

# Sidebar — AI controls
st.sidebar.divider()
st.sidebar.header("🤖 AI Analysis")

enable_ai = st.sidebar.toggle("Enable AI", value=False)
ai_provider = st.sidebar.selectbox("Provider", ["OpenAI", "Gemini"], index=0)
openai_model = st.sidebar.text_input("OpenAI model", value="gpt-4.1-mini")
gemini_model = st.sidebar.text_input("Gemini model", value="gemini-1.5-pro")

user_focus = st.sidebar.text_area(
    "AI focus",
    value="Explain why this is ranked as an opportunity, the regime read, key levels, invalidation, and 2-3 scenarios.",
    height=110,
)

with st.sidebar.expander("🔐 Secrets diagnostics", expanded=False):
    st.write("These are **not** printed fully; only whether they are detected.")
    o = get_secret("OPENAI_API_KEY")
    g = get_secret("GEMINI_API_KEY")
    st.write({"OPENAI_API_KEY_detected": bool(o), "GEMINI_API_KEY_detected": bool(g)})
    st.caption("If detected=False in Streamlit Cloud, ensure `.streamlit/secrets.toml` exists in the deployed app.")


# =============================================================================
# Core logic for scanner
# =============================================================================

def scan_one(tv_symbol: str) -> Dict[str, Any]:
    """
    Fetch data + (optional) fundamentals and compute:
    - tech features
    - fundamentals score (if available and equity-like)
    - opportunity score + risk meter
    """
    yf_sym = tv_to_yf_symbol(tv_symbol, mapping_overrides)
    df, err = yf_download(yf_sym, interval=interval, start=start_dt, end=end_dt)
    if err or df.empty:
        return {
            "TV_Symbol": tv_symbol,
            "YF_Symbol": yf_sym,
            "Status": "FAIL",
            "Error": err or "Empty dataframe",
        }

    # Determine if it's likely an equity for fundamentals
    # Heuristic: equities tend to have alphabetic tickers without '=' or '^' or '-USD' and not ending with '=F'
    # BUT we still just try yfinance info; if it's empty, we treat as no fundamentals.
    info = None
    fund_score = None
    fund_breakdown = None
    asset_has_fundamentals = False

    if prefer_equities:
        info, i_err = yf_info(yf_sym)
        if info and not i_err:
            # If it has a sector or marketCap, treat as equity-like
            if info.get("sector") is not None or info.get("marketCap") is not None:
                asset_has_fundamentals = True
                fund_score, fund_breakdown = fundamentals_score(info)

    tech = technical_features(df, smc_cfg=smc_cfg)
    score, score_parts = opportunity_score(tech, fund_score, asset_has_fundamentals)
    risk = compute_risk_meter(df, info if asset_has_fundamentals else None)

    # Add compact columns for scanner table
    out = {
        "TV_Symbol": tv_symbol,
        "YF_Symbol": yf_sym,
        "Status": "OK",
        "OpportunityScore": float(score),
        "RiskMeter": int(risk),
        "Close": float(df["Close"].iloc[-1]),
        "MFI": float(tech["mfi"]) if tech.get("ok") and pd.notna(tech.get("mfi")) else np.nan,
        "VWAP_Dist": float(tech["vwap_dist"]) if tech.get("ok") and pd.notna(tech.get("vwap_dist")) else np.nan,
        "BB_BW_Pctl": float(tech["bb_bandwidth_pct"]) if tech.get("ok") and pd.notna(tech.get("bb_bandwidth_pct")) else np.nan,
        "Ichimoku_AboveCloud": bool(tech.get("above_cloud", False)) if tech.get("ok") else False,
        "SMC_BOS_Bull_Recent": bool(tech.get("bos_bull_recent", False)) if tech.get("ok") else False,
        "SMC_CHOCH_Bull_Recent": bool(tech.get("choch_bull_recent", False)) if tech.get("ok") else False,
        "FundScore": float(fund_score) if asset_has_fundamentals and fund_score is not None else np.nan,
        "Name": (info.get("shortName") if isinstance(info, dict) else None) or "",
        "Sector": (info.get("sector") if isinstance(info, dict) else None) or "",
        "MarketCap": (info.get("marketCap") if isinstance(info, dict) else None) or np.nan,
        "_df": df,
        "_info": info,
        "_tech": tech,
        "_fund_breakdown": fund_breakdown,
        "_score_parts": score_parts,
    }
    return out


# =============================================================================
# Main content
# =============================================================================

if mode == "Scanner (Rank Opportunities)":
    st.subheader("🔎 Opportunity Scanner")
    st.caption("Scans your watchlist, computes TA/fundamentals features, then ranks by Opportunity Score with a Risk Meter.")

    if not symbols:
        st.warning("No symbols found. Upload a symbols .txt or enable the built-in sample list.")
        st.stop()

    scan_list = symbols[: int(max_symbols)]
    st.write(f"Symbols to scan: **{len(scan_list)}** (of {len(symbols)})")

    run = st.button("Run Scan", type="primary")
    if run:
        progress = st.progress(0)
        status = st.empty()

        results: List[Dict[str, Any]] = []
        fail_count = 0

        start_t = time.time()
        with ThreadPoolExecutor(max_workers=int(threads)) as ex:
            futures = {ex.submit(scan_one, sym): sym for sym in scan_list}
            done = 0
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                done += 1
                if res.get("Status") != "OK":
                    fail_count += 1
                progress.progress(int(done / len(scan_list) * 100))
                status.info(f"Scanned {done}/{len(scan_list)} | Failures: {fail_count}")

        elapsed = time.time() - start_t
        st.success(f"Scan complete. {len(results) - fail_count} OK, {fail_count} failed. Elapsed: {elapsed:.1f}s")

        # Build dataframe for display (strip heavy fields)
        display_rows = []
        for r in results:
            rr = {k: v for k, v in r.items() if not k.startswith("_")}
            display_rows.append(rr)
        df_res = pd.DataFrame(display_rows)

        # Filter
        ok = df_res[df_res["Status"] == "OK"].copy()
        ok = ok[(ok["OpportunityScore"] >= float(min_score)) & (ok["RiskMeter"] <= int(max_risk))].copy()
        ok = ok.sort_values(["OpportunityScore", "RiskMeter"], ascending=[False, True])

        st.markdown("### 📌 Ranked Opportunities")
        st.dataframe(
            ok,
            use_container_width=True,
            height=420
        )

        # Download
        csv_bytes = ok.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download ranked opportunities (CSV)",
            data=csv_bytes,
            file_name="ranked_opportunities.csv",
            mime="text/csv",
        )

        # Pick one to deep dive
        st.markdown("### 🧠 Deep Dive (from scan results)")
        if len(ok) == 0:
            st.info("No results matched your filters. Lower Min Score or increase Max Risk.")
            st.stop()

        pick = st.selectbox("Select an opportunity to analyze", ok["TV_Symbol"].tolist(), index=0)

        # Recover full result (with heavy fields)
        by_sym = {r["TV_Symbol"]: r for r in results}
        chosen = by_sym[pick]
        df = chosen["_df"]
        info = chosen.get("_info") if isinstance(chosen.get("_info"), dict) else None
        tech = chosen.get("_tech", {})
        fund_bd = chosen.get("_fund_breakdown", None)
        score_parts = chosen.get("_score_parts", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Opportunity Score", f"{chosen.get('OpportunityScore', np.nan):.1f}")
        c2.metric("Risk Meter", str(chosen.get("RiskMeter", "")))
        c3.metric("Close", f"{chosen.get('Close', np.nan):.6g}")
        c4.metric("MFI", f"{chosen.get('MFI', np.nan):.2f}" if pd.notna(chosen.get("MFI", np.nan)) else "—")

        fig = plot_asset(
            df=df,
            title=f"{pick}  (source: {chosen.get('YF_Symbol')})",
            show_volume=show_volume,
            show_bb=show_bb,
            show_vwap=show_vwap,
            show_ichimoku=show_ichimoku,
            show_mfi=show_mfi,
            show_smc=show_smc,
            smc_cfg=smc_cfg,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🧾 Reasoning (Score Breakdown)")
        st.json(score_parts)

        if info:
            st.markdown("### 🧾 Fundamentals (best-effort from yfinance)")
            f = {
                "Name": info.get("shortName"),
                "Symbol": info.get("symbol"),
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
                "MarketCap": info.get("marketCap"),
                "Currency": info.get("currency"),
                "Beta": info.get("beta"),
                "TrailingPE": info.get("trailingPE"),
                "ForwardPE": info.get("forwardPE"),
                "PriceToBook": info.get("priceToBook"),
                "RevenueGrowth": info.get("revenueGrowth"),
                "EarningsGrowth": info.get("earningsGrowth"),
                "OperatingMargins": info.get("operatingMargins"),
                "ROE": info.get("returnOnEquity"),
                "FreeCashflow": info.get("freeCashflow"),
                "TotalCash": info.get("totalCash"),
                "TotalDebt": info.get("totalDebt"),
            }
            st.dataframe(pd.DataFrame([f]).T.rename(columns={0: "Value"}), use_container_width=True)

            if fund_bd:
                st.markdown("#### Fundamentals subscores")
                st.json(fund_bd)

        st.markdown("### 🤖 AI Narrative (optional)")
        if not enable_ai:
            st.info("Enable AI in the sidebar to generate a structured narrative (regime, thesis, invalidation, scenarios).")
        else:
            ctx = build_ai_context(
                symbol_label=f"{pick} (source: {chosen.get('YF_Symbol')})",
                interval=interval,
                tech=tech,
                fund=info,
                fund_breakdown=fund_bd,
                score=float(chosen.get("OpportunityScore", 0.0)),
                risk=int(chosen.get("RiskMeter", 7)),
            )
            st.text_area("Context passed to AI", value=ctx, height=220)

            if st.button("Generate AI Analysis", type="primary"):
                with st.spinner("Generating AI analysis..."):
                    if ai_provider == "OpenAI":
                        out, err = ai_openai(ctx, user_focus=user_focus, model=openai_model)
                    else:
                        out, err = ai_gemini(ctx, user_focus=user_focus, model=gemini_model)
                if err:
                    st.error(err)
                else:
                    st.markdown(out)

        st.markdown("### 🧯 Data failures (transparency)")
        failed = df_res[df_res["Status"] != "OK"].copy()
        if len(failed):
            st.dataframe(failed[["TV_Symbol", "YF_Symbol", "Error"]], use_container_width=True, height=220)
        else:
            st.write("No failures in this scan.")

elif mode == "Single Chart (Deep Dive)":
    st.subheader("📈 Single Asset Deep Dive")

    if not symbols:
        st.warning("No symbols found. Upload a symbols .txt or enable the built-in sample list.")
        st.stop()

    tv_sym = st.selectbox("Select symbol", symbols, index=0)
    yf_sym = tv_to_yf_symbol(tv_sym, mapping_overrides)

    df, err = yf_download(yf_sym, interval=interval, start=start_dt, end=end_dt)
    if err:
        st.error(err)
        st.stop()

    info, i_err = yf_info(yf_sym)
    if i_err:
        info = {}

    tech = technical_features(df, smc_cfg=smc_cfg)
    f_score, f_bd = fundamentals_score(info) if info else (None, None)
    score, score_parts = opportunity_score(tech, f_score, asset_has_fundamentals=bool(info))
    risk = compute_risk_meter(df, info if info else None)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Opportunity Score", f"{score:.1f}")
    c2.metric("Risk Meter", str(risk))
    c3.metric("Close", f"{df['Close'].iloc[-1]:.6g}")
    c4.metric("MFI", f"{tech.get('mfi', np.nan):.2f}" if tech.get("ok") and pd.notna(tech.get("mfi")) else "—")

    fig = plot_asset(
        df=df,
        title=f"{tv_sym} (source: {yf_sym})",
        show_volume=show_volume,
        show_bb=show_bb,
        show_vwap=show_vwap,
        show_ichimoku=show_ichimoku,
        show_mfi=show_mfi,
        show_smc=show_smc,
        smc_cfg=smc_cfg,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧾 Score Breakdown")
    st.json(score_parts)

    if info:
        st.markdown("### 🧾 Fundamentals (best-effort)")
        st.json({
            "Name": info.get("shortName"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "MarketCap": info.get("marketCap"),
            "Beta": info.get("beta"),
            "TrailingPE": info.get("trailingPE"),
            "ForwardPE": info.get("forwardPE"),
            "PriceToBook": info.get("priceToBook"),
            "RevenueGrowth": info.get("revenueGrowth"),
            "EarningsGrowth": info.get("earningsGrowth"),
            "OperatingMargins": info.get("operatingMargins"),
            "ROE": info.get("returnOnEquity"),
            "FreeCashflow": info.get("freeCashflow"),
        })
        if f_bd:
            st.markdown("#### Fundamentals subscores")
            st.json(f_bd)

    st.markdown("### 🤖 AI Narrative (optional)")
    if enable_ai:
        ctx = build_ai_context(
            symbol_label=f"{tv_sym} (source: {yf_sym})",
            interval=interval,
            tech=tech,
            fund=info,
            fund_breakdown=f_bd,
            score=score,
            risk=risk,
        )
        st.text_area("Context passed to AI", value=ctx, height=220)

        if st.button("Generate AI Analysis", type="primary"):
            with st.spinner("Generating AI analysis..."):
                if ai_provider == "OpenAI":
                    out, e = ai_openai(ctx, user_focus=user_focus, model=openai_model)
                else:
                    out, e = ai_gemini(ctx, user_focus=user_focus, model=gemini_model)
            if e:
                st.error(e)
            else:
                st.markdown(out)
    else:
        st.info("Enable AI in the sidebar to generate a structured narrative.")

else:
    st.subheader("➗ Ratio Chart")

    if not ratios:
        st.warning("No ratios found. Upload a ratios .txt or enable the built-in sample list.")
        st.stop()

    ratio = st.selectbox("Select ratio (A/B)", ratios, index=0)
    a, b = ratio.split("/", 1)
    a = a.strip()
    b = b.strip()
    ya = tv_to_yf_symbol(a, mapping_overrides)
    yb = tv_to_yf_symbol(b, mapping_overrides)

    df_a, err_a = yf_download(ya, interval=interval, start=start_dt, end=end_dt)
    df_b, err_b = yf_download(yb, interval=interval, start=start_dt, end=end_dt)

    if err_a:
        st.error(f"{a} → {ya}: {err_a}")
    if err_b:
        st.error(f"{b} → {yb}: {err_b}")
    if err_a or err_b or df_a.empty or df_b.empty:
        st.stop()

    s = (df_a["Close"].rename("A") / df_b["Close"].rename("B")).dropna()
    ratio_df = pd.DataFrame({"Close": s})
    ratio_df["Open"] = ratio_df["Close"]
    ratio_df["High"] = ratio_df["Close"]
    ratio_df["Low"] = ratio_df["Close"]
    ratio_df["Volume"] = 0.0

    fig = make_subplots(rows=2 if show_mfi else 1, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.72, 0.28] if show_mfi else [1.0],
                        specs=[[{"secondary_y": False}]] + ([[{"secondary_y": False}]] if show_mfi else []))

    fig.add_trace(go.Scatter(x=ratio_df.index, y=ratio_df["Close"], name="Ratio", mode="lines"), row=1, col=1)

    if show_bb:
        bb = add_bollinger(ratio_df)
        fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_MID"], name="BB Mid", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_UPPER"], name="BB Upper", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_LOWER"], name="BB Lower", mode="lines"), row=1, col=1)

    if show_mfi:
        # No volume for ratios => MFI not meaningful; show placeholder
        fig.add_trace(go.Scatter(x=ratio_df.index, y=np.nan * ratio_df["Close"], name="MFI (n/a)", mode="lines"), row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    fig.update_layout(
        title=f"Ratio: {ratio} (source: {ya} / {yb})",
        xaxis_rangeslider_visible=False,
        height=850 if show_mfi else 650,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Note: Volume-based indicators (VWAP/MFI) are not meaningful on ratio series unless you have ratio-volume data.")


# Footer
st.divider()
st.caption(
    "Tip: If a symbol fails to fetch, add a mapping override (TradingView symbol → yfinance symbol). "
    "For big scans, prefer interval=1d or 1wk for stability."
)
