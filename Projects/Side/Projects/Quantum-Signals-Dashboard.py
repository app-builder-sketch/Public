# app.py
# Quantum Signals Dashboard (Streamlit Cloud)
# - Apex Vector (Flux + Efficiency) + Divergences
# - RQZO (Relativistic Quantum-Zeta Oscillator) + Chaos Zones
# - Unified Field (SMC-lite): FVG ("Wormholes"), OB ("Event Horizons"), BOS/CHoCH, HUD-style Recommendation
# - Optional Trend Engine (ATR channel + ADX + WaveTrend + Volume filter) inspired by your SMC v7.2 file
# - Pro UI/UX + Mobile mode + Plotly multi-panel chart
# - AI Brief (Gemini default, OpenAI optional) loaded from st.secrets
# - Telegram broadcasting (manual + optional auto-send on new signal)

import os
import time
import json
import math
import requests
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Optional dependencies
# ----------------------------
CCXT_OK = True
try:
    import ccxt  # type: ignore
except Exception:
    CCXT_OK = False

YF_OK = True
try:
    import yfinance as yf  # type: ignore
except Exception:
    YF_OK = False

OPENAI_OK = True
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OPENAI_OK = False

GEMINI_OK = True
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    GEMINI_OK = False

AUTOR_OK = True
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:
    AUTOR_OK = False


# ============================================================
# 0) APP CONFIG + CSS
# ============================================================
st.set_page_config(
    page_title="Quantum Signals Dashboard // Apex + RQZO",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_css():
    st.markdown(
        """
<style>
:root{
  --bg:#07090d;
  --panel:#0f131a;
  --panel2:#121826;
  --border:#1f2937;
  --text:#e5e7eb;
  --muted:#9ca3af;
  --good:#00E676;
  --bad:#FF1744;
  --warn:#FFD600;
  --cyan:#00E5FF;
  --mag:#FF0055;
  --purple:#9C27B0;
}

.stApp { background: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #05070b, #0b0f16);
  border-right: 1px solid var(--border);
}
div[data-testid="stMetric"]{
  background: rgba(15,19,26,.85);
  border: 1px solid rgba(31,41,55,.8);
  border-radius: 16px;
  padding: 10px 12px;
}
.block-container { padding-top: 1.0rem; }
.small-muted { color: var(--muted); font-size: 12px; }
.pill {
  display:inline-block; padding:6px 10px; border-radius:999px;
  border:1px solid rgba(31,41,55,.9); background: rgba(15,19,26,.8);
  font-size:12px; color: var(--muted);
}
.hr { height:1px; background: rgba(31,41,55,.9); margin: 10px 0 14px 0; }
.kbd {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  padding: 2px 6px; border-radius: 8px;
  border: 1px solid rgba(31,41,55,.9);
  background: rgba(18,24,38,.8);
  color: #c7d2fe;
  font-size: 12px;
}

/* Mobile-first helpers */
@media (max-width: 900px){
  .block-container { padding-left: 0.6rem; padding-right: 0.6rem; }
  h1 { font-size: 1.35rem !important; }
  .stPlotlyChart { margin-top: 0.25rem; }
}
</style>
""",
        unsafe_allow_html=True,
    )

inject_css()

st.title("⚛️ Quantum Signals Dashboard")
st.caption("Apex Vector + RQZO + SMC-lite structure + Pro Trend Engine, with AI + Telegram broadcasting.")


# ============================================================
# 1) SECRETS (auto-load)
# ============================================================
def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    # st.secrets first, then env
    try:
        if key in st.secrets:
            v = st.secrets[key]
            return str(v) if v is not None else default
    except Exception:
        pass
    return os.getenv(key, default)

def init_secrets():
    if st.session_state.get("_secrets_loaded"):
        return

    st.session_state.GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")
    st.session_state.GEMINI_MODEL = _get_secret("GEMINI_MODEL", "gemini-1.5-pro")
    st.session_state.OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
    st.session_state.OPENAI_MODEL = _get_secret("OPENAI_MODEL", "gpt-4.1-mini")

    st.session_state.TELEGRAM_BOT_TOKEN = _get_secret("TELEGRAM_BOT_TOKEN")
    st.session_state.TELEGRAM_CHAT_ID = _get_secret("TELEGRAM_CHAT_ID")

    st.session_state._secrets_loaded = True

init_secrets()


# ============================================================
# 2) UTILS / MATH
# ============================================================
def _retry(fn, tries=3, sleep=0.6):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(sleep * (i + 1))
    raise last

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()

def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)

def vwma(price: pd.Series, vol: pd.Series, n: int) -> pd.Series:
    pv = price * vol
    denom = vol.rolling(n).sum().replace(0, np.nan)
    return pv.rolling(n).sum() / denom

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return rma(tr, n)

def dmi_adx(df: pd.DataFrame, n: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    # Basic ADX implementation (Wilder)
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_n = rma(tr, n)

    plus_di = 100 * rma(pd.Series(plus_dm, index=df.index), n) / atr_n.replace(0, np.nan)
    minus_di = 100 * rma(pd.Series(minus_dm, index=df.index), n) / atr_n.replace(0, np.nan)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx = rma(dx, n)
    return plus_di.fillna(0.0), minus_di.fillna(0.0), adx.fillna(0.0)

def pivothigh(series: pd.Series, left: int, right: int) -> pd.Series:
    s = series.values
    out = np.full(len(s), np.nan, dtype=float)
    for i in range(left, len(s) - right):
        win = s[i - left : i + right + 1]
        if np.isfinite(s[i]) and s[i] == np.nanmax(win):
            out[i] = s[i]
    return pd.Series(out, index=series.index)

def pivotlow(series: pd.Series, left: int, right: int) -> pd.Series:
    s = series.values
    out = np.full(len(s), np.nan, dtype=float)
    for i in range(left, len(s) - right):
        win = s[i - left : i + right + 1]
        if np.isfinite(s[i]) and s[i] == np.nanmin(win):
            out[i] = s[i]
    return pd.Series(out, index=series.index)

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["open", "high", "low", "close", "volume"]
    return out.dropna()

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ============================================================
# 3) DATA LAYER (Streamlit Cloud friendly + caching)
# ============================================================
@st.cache_data(ttl=60 * 30, show_spinner=False)
def ccxt_exchange_markets(exchange_id: str) -> List[str]:
    if not CCXT_OK:
        return []
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    markets = ex.load_markets()
    return sorted(list(markets.keys()))

@st.cache_data(ttl=60 * 5, show_spinner=False)
def fetch_ohlcv_ccxt(exchange_id: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    if not CCXT_OK:
        raise RuntimeError("ccxt not installed.")
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})

    def _do():
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.drop(columns=["ts"]).set_index("time")
        return df

    return _retry(_do, tries=3)

def timeframe_map_for_yf(tf: str) -> Tuple[str, str]:
    # Stable + predictable mapping for Streamlit Cloud
    mapping = {
        "1m": ("1m", "7d"),
        "2m": ("2m", "60d"),
        "5m": ("5m", "60d"),
        "15m": ("15m", "60d"),
        "30m": ("30m", "60d"),
        "1h": ("60m", "730d"),
        "4h": ("60m", "730d"),  # resample from 1h
        "1d": ("1d", "10y"),
    }
    return mapping.get(tf, ("60m", "730d"))

@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_ohlcv_yf(ticker: str, interval: str, period: str) -> pd.DataFrame:
    if not YF_OK:
        raise RuntimeError("yfinance not installed.")

    def _do():
        df = yf.download(ticker, interval=interval, period=period, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df.index = pd.to_datetime(df.index, utc=True)
        return df[["open", "high", "low", "close", "volume"]].dropna()

    return _retry(_do, tries=3)


# ============================================================
# 4) ENGINES
# ============================================================
@dataclass
class ApexVectorParams:
    eff_super: float = 0.60
    eff_resist: float = 0.30
    vol_norm: int = 55
    len_vec: int = 14
    sm_type: str = "EMA"   # EMA/SMA/RMA/WMA/VWMA
    len_sm: int = 5
    use_vol: bool = True
    strictness: float = 1.0
    div_look: int = 5

def compute_apex_vector(df: pd.DataFrame, p: ApexVectorParams) -> Dict[str, pd.Series]:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    raw_eff = (body / rng).fillna(0.0)
    efficiency = ema(raw_eff, p.len_vec)

    vol_avg = sma(df["volume"], p.vol_norm).replace(0, np.nan)
    if p.use_vol:
        vol_fact = (df["volume"] / vol_avg).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    else:
        vol_fact = 1.0

    direction = np.sign(df["close"] - df["open"])
    vector_raw = direction * efficiency * vol_fact

    if p.sm_type == "EMA":
        flux = ema(vector_raw, p.len_sm)
    elif p.sm_type == "SMA":
        flux = sma(vector_raw, p.len_sm)
    elif p.sm_type == "RMA":
        flux = rma(vector_raw, p.len_sm)
    elif p.sm_type == "WMA":
        flux = wma(vector_raw, p.len_sm)
    elif p.sm_type == "VWMA":
        flux = vwma(vector_raw, df["volume"], p.len_sm)
    else:
        flux = ema(vector_raw, p.len_sm)

    th_super = float(p.eff_super) * float(p.strictness)
    th_resist = float(p.eff_resist) * float(p.strictness)

    is_super_bull = flux > th_super
    is_super_bear = flux < -th_super
    is_resistive = flux.abs() < th_resist
    is_heat = ~(is_super_bull | is_super_bear | is_resistive)

    # Divergence: pivot-based (REG/HID)
    div_look = int(p.div_look)
    ph = pivothigh(flux, div_look, div_look)
    pl = pivotlow(flux, div_look, div_look)

    div_bull_reg = pd.Series(False, index=df.index)
    div_bull_hid = pd.Series(False, index=df.index)
    div_bear_reg = pd.Series(False, index=df.index)
    div_bear_hid = pd.Series(False, index=df.index)

    prev_pl_flux = np.nan
    prev_pl_price = np.nan
    prev_ph_flux = np.nan
    prev_ph_price = np.nan

    lows = df["low"].values
    highs = df["high"].values

    for i in range(len(df)):
        if np.isfinite(pl.iloc[i]):
            price_at_pivot = lows[max(i - div_look, 0)]
            if np.isfinite(prev_pl_flux):
                # Regular bull: lower price, higher flux
                if price_at_pivot < prev_pl_price and pl.iloc[i] > prev_pl_flux:
                    div_bull_reg.iloc[i] = True
                # Hidden bull: higher price, lower flux
                if price_at_pivot > prev_pl_price and pl.iloc[i] < prev_pl_flux:
                    div_bull_hid.iloc[i] = True
            prev_pl_flux = pl.iloc[i]
            prev_pl_price = price_at_pivot

        if np.isfinite(ph.iloc[i]):
            price_at_pivot = highs[max(i - div_look, 0)]
            if np.isfinite(prev_ph_flux):
                # Regular bear: higher price, lower flux
                if price_at_pivot > prev_ph_price and ph.iloc[i] < prev_ph_flux:
                    div_bear_reg.iloc[i] = True
                # Hidden bear: lower price, higher flux
                if price_at_pivot < prev_ph_price and ph.iloc[i] > prev_ph_flux:
                    div_bear_hid.iloc[i] = True
            prev_ph_flux = ph.iloc[i]
            prev_ph_price = price_at_pivot

    return {
        "efficiency": efficiency,
        "flux": flux,
        "th_super": pd.Series(th_super, index=df.index),
        "th_resist": pd.Series(th_resist, index=df.index),
        "is_super_bull": is_super_bull.fillna(False),
        "is_super_bear": is_super_bear.fillna(False),
        "is_resistive": is_resistive.fillna(False),
        "is_heat": is_heat.fillna(False),
        "div_bull_reg": div_bull_reg,
        "div_bull_hid": div_bull_hid,
        "div_bear_reg": div_bear_reg,
        "div_bear_hid": div_bear_hid,
    }

@dataclass
class RQZOParams:
    base_harmonics: int = 25
    terminal_volatility: float = 5.0
    entropy_lookback: int = 20
    fractal_dim_len: int = 20
    bands_len: int = 20

def compute_rqzo(df: pd.DataFrame, p: RQZOParams) -> Dict[str, pd.Series]:
    src = df["close"].astype(float)

    # normalize to [0,1] over 100
    min_val = src.rolling(100).min()
    max_val = src.rolling(100).max()
    norm_price = (src - min_val) / ((max_val - min_val) + 1e-10)

    velocity = (norm_price - norm_price.shift(1)).abs().fillna(0.0)

    c_scaled = max(float(p.terminal_volatility) / 100.0, 1e-6)
    clamped_v = np.minimum(velocity.values, c_scaled * 0.99)
    gamma = 1.0 / np.sqrt(np.maximum(1e-12, 1.0 - np.power(clamped_v / c_scaled, 2.0)))
    gamma = pd.Series(gamma, index=df.index)

    # fractal dimension (FDI-ish) based on range_len and path_len
    L = int(p.fractal_dim_len)
    highest = src.rolling(L).max()
    lowest = src.rolling(L).min()
    range_len = (highest - lowest) / float(L)

    diffs = src.diff().abs()
    # rolling average of absolute moves as proxy to Pine looped path_len/L
    path_len = diffs.rolling(L).mean()

    fd_val = pd.Series(1.5, index=df.index)
    valid = (range_len > 0) & (path_len > 0)
    fd_val.loc[valid] = 1.0 + (np.log10(range_len[valid].values) / np.log10(path_len[valid].values))

    N_eff = np.floor(float(p.base_harmonics) / fd_val.replace(0, np.nan)).fillna(float(p.base_harmonics)).astype(int)
    N_eff = N_eff.clip(lower=1, upper=100)

    # Shannon entropy (binned returns) rolling
    ent_lb = int(p.entropy_lookback)
    returns = ((src - src.shift(1)) / src.shift(1)).replace([np.inf, -np.inf], np.nan)
    bins = 5
    ent = np.full(len(df), np.nan, dtype=float)

    ret_vals = returns.values
    for i in range(len(df)):
        if i < ent_lb:
            continue
        win = ret_vals[i - ent_lb + 1 : i + 1]
        win = win[np.isfinite(win)]
        if win.size == 0:
            continue
        mn = float(np.min(win))
        mx = float(np.max(win))
        rng = (mx - mn) + 1e-10
        # histogram
        idx = np.floor(((win - mn) / rng) * bins).astype(int)
        idx = np.clip(idx, 0, bins - 1)
        freq = np.bincount(idx, minlength=bins).astype(float)
        pvec = freq / float(ent_lb)

        e = 0.0
        for pv in pvec:
            if pv > 0:
                e += pv * math.log(pv)
        ent[i] = e

    entropy = pd.Series(ent, index=df.index)
    norm_entropy = entropy.abs() / math.log(bins)
    entropy_gate = np.exp(-2.0 * (norm_entropy - 0.6).abs())
    entropy_gate = pd.Series(entropy_gate, index=df.index)

    # Zeta imag sum
    sigma = 0.5
    bar_index = np.arange(len(df), dtype=float)
    tau = (bar_index % 100.0) / gamma.replace(0, np.nan).fillna(1.0).values

    maxN = int(np.clip(p.base_harmonics, 5, 100))
    n = np.arange(1, maxN + 1, dtype=float)
    amp = np.power(n, -sigma)
    ln_n = np.log(n)

    zeta_imag = np.zeros(len(df), dtype=float)
    N_eff_vals = N_eff.values.astype(int)

    for i in range(len(df)):
        ni = int(N_eff_vals[i])
        ni = max(1, min(ni, maxN))
        t = float(tau[i])
        # vectorized sin over n
        s = np.sin(t * ln_n[:ni])
        zeta_imag[i] = float(np.sum(amp[:ni] * s))

    rqzo = pd.Series(zeta_imag, index=df.index) * entropy_gate * 10.0

    # bands
    bands_len = int(p.bands_len)
    basis = sma(rqzo, bands_len)
    dev = rqzo.rolling(bands_len).std()
    band_width = (2.5 - fd_val) * dev
    upper = basis + band_width
    lower = basis - band_width

    chaos = (norm_entropy > 0.8).fillna(False)

    return {
        "rqzo": rqzo,
        "fd_val": fd_val,
        "norm_entropy": norm_entropy,
        "entropy_gate": entropy_gate,
        "upper_band": upper,
        "lower_band": lower,
        "chaos": chaos,
    }

@dataclass
class SMCParams:
    smc_lookback: int = 5
    show_fvg: bool = True
    show_ob: bool = True
    show_structure: bool = True

def compute_smc_lite(df: pd.DataFrame, apex_flux: pd.Series, th_resist: float, p: SMCParams) -> Dict[str, pd.Series]:
    # Wormholes (FVG): bullish if low > high[2], bearish if high < low[2]
    low = df["low"]
    high = df["high"]
    open_ = df["open"]
    close = df["close"]

    wormhole_bull = p.show_fvg & (low > high.shift(2))
    wormhole_bear = p.show_fvg & (high < low.shift(2))

    # Event Horizons (OB-ish): bullish if prev candle bearish and current close > prev high and is super bull
    # We'll keep it simple & consistent with your Unified Field logic.
    prev_bear = close.shift(1) < open_.shift(1)
    prev_bull = close.shift(1) > open_.shift(1)

    # "is super bull/bear" from apex thresholds will be computed outside; here we take apex_flux sign
    # and let calling code filter by state if desired.
    event_horizon_bull = p.show_ob & prev_bear & (close > high.shift(1))
    event_horizon_bear = p.show_ob & prev_bull & (close < low.shift(1))

    # Structure pivots
    lb = int(p.smc_lookback)
    st_ph = pivothigh(high, lb, lb)
    st_pl = pivotlow(low, lb, lb)

    last_st_high = pd.Series(np.nan, index=df.index)
    last_st_low = pd.Series(np.nan, index=df.index)

    cur_h = np.nan
    cur_l = np.nan
    for i in range(len(df)):
        if np.isfinite(st_ph.iloc[i]):
            cur_h = float(high.iloc[max(i - lb, 0)])
        if np.isfinite(st_pl.iloc[i]):
            cur_l = float(low.iloc[max(i - lb, 0)])
        last_st_high.iloc[i] = cur_h
        last_st_low.iloc[i] = cur_l

    bos_bull = p.show_structure & (last_st_high.notna()) & (close > last_st_high) & (high.shift(1) <= last_st_high)
    bos_bear = p.show_structure & (last_st_low.notna()) & (close < last_st_low) & (low.shift(1) >= last_st_low)

    choch_bull = p.show_structure & (last_st_low.notna()) & (close > last_st_low) & (apex_flux.shift(1) < -float(th_resist))
    choch_bear = p.show_structure & (last_st_high.notna()) & (close < last_st_high) & (apex_flux.shift(1) > float(th_resist))

    return {
        "wormhole_bull": wormhole_bull.fillna(False),
        "wormhole_bear": wormhole_bear.fillna(False),
        "event_horizon_bull": event_horizon_bull.fillna(False),
        "event_horizon_bear": event_horizon_bear.fillna(False),
        "last_st_high": last_st_high,
        "last_st_low": last_st_low,
        "bos_bull": bos_bull.fillna(False),
        "bos_bear": bos_bear.fillna(False),
        "choch_bull": choch_bull.fillna(False),
        "choch_bear": choch_bear.fillna(False),
    }

@dataclass
class TrendEngineParams:
    ma_type: str = "HMA"
    len_main: int = 55
    mult: float = 1.5
    adx_len: int = 14
    adx_thr: float = 20.0
    vol_len: int = 20
    wt_esa: int = 10
    wt_d: int = 10
    wt_tci: int = 21

def hma(series: pd.Series, n: int) -> pd.Series:
    # Hull MA: WMA(2*WMA(n/2)-WMA(n), sqrt(n))
    n2 = max(1, n // 2)
    sqrt_n = max(1, int(math.sqrt(n)))
    wma1 = wma(series, n2)
    wma2 = wma(series, n)
    return wma(2 * wma1 - wma2, sqrt_n)

def get_ma(ma_type: str, s: pd.Series, n: int) -> pd.Series:
    t = (ma_type or "EMA").upper()
    if t == "SMA":
        return sma(s, n)
    if t == "EMA":
        return ema(s, n)
    if t == "RMA":
        return rma(s, n)
    if t == "HMA":
        return hma(s, n)
    return ema(s, n)

def compute_trend_engine(df: pd.DataFrame, p: TrendEngineParams) -> Dict[str, pd.Series]:
    src = df["close"]
    baseline = get_ma(p.ma_type, src, p.len_main)
    a = atr(df, p.len_main)
    upper = baseline + (a * float(p.mult))
    lower = baseline - (a * float(p.mult))

    trend = pd.Series(0, index=df.index, dtype=int)
    trend[(df["close"] > upper)] = 1
    trend[(df["close"] < lower)] = -1
    # forward fill trend state so it persists until flipped
    trend = trend.replace(0, np.nan).ffill().fillna(0).astype(int)

    di_plus, di_minus, adx = dmi_adx(df, p.adx_len)
    adx_ok = adx > float(p.adx_thr)

    # WaveTrend approximation
    ap = (df["high"] + df["low"] + df["close"]) / 3.0
    esa = ema(ap, p.wt_esa)
    d = ema((ap - esa).abs(), p.wt_d)
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    tci = ema(ci.fillna(0.0), p.wt_tci)

    mom_buy = tci < 60
    mom_sell = tci > -60

    vol_avg = sma(df["volume"], p.vol_len)
    vol_ok = df["volume"] > vol_avg

    sig_buy = (trend == 1) & (trend.shift(1) != 1) & vol_ok & mom_buy & adx_ok
    sig_sell = (trend == -1) & (trend.shift(1) != -1) & vol_ok & mom_sell & adx_ok

    # Trailing stop (ATR-based)
    trail_stop = pd.Series(np.nan, index=df.index)
    trail_atr = atr(df, 14) * 2.0

    for i in range(len(df)):
        if i == 0:
            trail_stop.iloc[i] = np.nan
            continue
        if trend.iloc[i] == 1:
            prev = trail_stop.iloc[i - 1]
            base = df["close"].iloc[i] - trail_atr.iloc[i]
            if np.isfinite(prev):
                trail_stop.iloc[i] = max(prev, base)
            else:
                trail_stop.iloc[i] = base
            if trend.iloc[i - 1] == -1:
                trail_stop.iloc[i] = base
        elif trend.iloc[i] == -1:
            prev = trail_stop.iloc[i - 1]
            base = df["close"].iloc[i] + trail_atr.iloc[i]
            if np.isfinite(prev):
                trail_stop.iloc[i] = min(prev, base)
            else:
                trail_stop.iloc[i] = base
            if trend.iloc[i - 1] == 1:
                trail_stop.iloc[i] = base
        else:
            trail_stop.iloc[i] = trail_stop.iloc[i - 1]

    return {
        "baseline": baseline,
        "upper": upper,
        "lower": lower,
        "trend": trend,
        "adx": adx,
        "tci": tci,
        "trail_stop": trail_stop,
        "sig_buy": sig_buy.fillna(False),
        "sig_sell": sig_sell.fillna(False),
    }


# ============================================================
# 5) AI PROVIDERS (Gemini default, OpenAI optional)
# ============================================================
def ai_ready(provider: str) -> bool:
    if provider == "Gemini":
        return GEMINI_OK and bool(st.session_state.get("GEMINI_API_KEY"))
    if provider == "OpenAI":
        return OPENAI_OK and bool(st.session_state.get("OPENAI_API_KEY"))
    return False

def ai_generate(provider: str, prompt: str) -> str:
    if provider == "Gemini":
        if not (GEMINI_OK and st.session_state.get("GEMINI_API_KEY")):
            raise RuntimeError("Gemini not available or GEMINI_API_KEY missing.")
        genai.configure(api_key=st.session_state["GEMINI_API_KEY"])
        model_name = st.session_state.get("GEMINI_MODEL", "gemini-1.5-pro")
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "") or str(resp)

    if provider == "OpenAI":
        if not (OPENAI_OK and st.session_state.get("OPENAI_API_KEY")):
            raise RuntimeError("OpenAI not available or OPENAI_API_KEY missing.")
        client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])
        model_name = st.session_state.get("OPENAI_MODEL", "gpt-4.1-mini")
        out = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a professional trading signals analyst for pro traders. Be concise, specific, and risk-aware."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return out.choices[0].message.content

    raise RuntimeError("Unknown provider")


# ============================================================
# 6) TELEGRAM
# ============================================================
def telegram_send(text: str) -> Tuple[bool, str]:
    token = st.session_state.get("TELEGRAM_BOT_TOKEN")
    chat_id = st.session_state.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False, "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID."

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=15)
        if r.status_code == 200:
            return True, "Sent"
        return False, f"Telegram error {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, str(e)


# ============================================================
# 7) SIDEBAR (Pro but easy)
# ============================================================
with st.sidebar:
    st.markdown("### Control Panel")
    st.markdown('<div class="small-muted">Workflow: market → symbol → timeframe → run.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    mobile_mode = st.toggle("Mobile mode (stack UI)", value=False, help="Improves spacing + chart height for phones.")
    auto_refresh = st.toggle("Auto-refresh", value=False)
    if auto_refresh and AUTOR_OK:
        st_autorefresh(interval=30_000, key="autorefresh")  # 30s

    market = st.selectbox("Market", ["Crypto (CCXT)", "Stocks (yfinance)"], index=0)

    if market == "Crypto (CCXT)":
        if not CCXT_OK:
            st.error("ccxt not installed. Add it to requirements.txt.")
        exchange_id = st.selectbox("Exchange", ["binance", "bybit", "okx", "kucoin"], index=0)
        with st.spinner("Loading symbols (cached)…"):
            symbols = ccxt_exchange_markets(exchange_id) if CCXT_OK else []
        if not symbols:
            st.warning("No symbols loaded.")
        symbol = st.selectbox("Symbol (search)", symbols, index=0 if symbols else 0)
        tf = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)
        limit = st.slider("Bars", 200, 2000, 800, step=50)
    else:
        if not YF_OK:
            st.error("yfinance not installed. Add it to requirements.txt.")
        ticker = st.text_input("Ticker", value="AAPL", help="Any yfinance ticker: AAPL, SPY, BTC-USD, ^GSPC, etc.")
        tf = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
        interval, period = timeframe_map_for_yf(tf)
        st.caption(f"yfinance interval: {interval}, period: {period} (4h is resampled from 1h)")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### Apex Vector")
    colA, colB = st.columns(2)
    with colA:
        eff_super = st.slider("Super Threshold", 0.10, 1.00, 0.60, 0.05)
        eff_resist = st.slider("Resist Threshold", 0.00, 0.60, 0.30, 0.05)
        len_vec = st.slider("Vector Len", 2, 60, 14, 1)
        vol_norm = st.slider("Vol Norm", 10, 200, 55, 5)
    with colB:
        sm_type = st.selectbox("Smoothing", ["EMA", "SMA", "RMA", "WMA", "VWMA"], index=0)
        len_sm = st.slider("Smooth Len", 1, 30, 5, 1)
        strictness = st.slider("Strictness", 0.5, 2.0, 1.0, 0.1)
        div_look = st.slider("Div Lookback", 1, 12, 5, 1)

    st.markdown("### RQZO")
    colC, colD = st.columns(2)
    with colC:
        base_harm = st.slider("Harmonics (N)", 5, 100, 25, 1)
        term_vol = st.slider("Terminal Vol (c)", 0.5, 15.0, 5.0, 0.1)
    with colD:
        ent_lb = st.slider("Entropy Lookback", 5, 80, 20, 1)
        fdi_len = st.slider("Fractal Dim Len", 5, 80, 20, 1)
    bands_len = st.slider("Band Len", 10, 60, 20, 1)

    st.markdown("### Structure (SMC-lite)")
    show_fvg = st.toggle("Show FVG (wormholes)", value=True)
    show_ob = st.toggle("Show OB events (dots)", value=True)
    show_structure = st.toggle("Show BOS/CHoCH", value=True)
    smc_lb = st.slider("Pivot Lookback", 2, 20, 5, 1)

    st.markdown("### Trend Engine (Optional)")
    use_trend_engine = st.toggle("Enable Trend Engine", value=True)
    if use_trend_engine:
        ma_type = st.selectbox("Trend MA", ["HMA", "EMA", "SMA", "RMA"], index=0)
        len_main = st.slider("Trend Length", 10, 200, 55, 1)
        mult = st.slider("Vol Mult", 0.5, 5.0, 1.5, 0.1)
        adx_thr = st.slider("ADX Threshold", 5.0, 40.0, 20.0, 1.0)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### AI / Telegram")
    provider = st.selectbox("AI Provider", ["Gemini", "OpenAI", "Off"], index=0)
    send_to_telegram = st.toggle("Enable Telegram", value=False)
    auto_telegram_on_new_signal = st.toggle("Auto-send on NEW signal", value=False)

    st.markdown('<div class="small-muted">Keys auto-load from <span class="kbd">st.secrets</span>. Not displayed.</div>', unsafe_allow_html=True)


# ============================================================
# 8) LOAD DATA
# ============================================================
with st.spinner("Loading price data…"):
    if market == "Crypto (CCXT)":
        df = fetch_ohlcv_ccxt(exchange_id, symbol, tf, int(limit))
        title_symbol = f"{exchange_id.upper()} · {symbol} · {tf}"
    else:
        df = fetch_ohlcv_yf(ticker, interval, period)
        if df is not None and not df.empty and tf == "4h":
            df = _resample_ohlcv(df, "4H")
        title_symbol = f"YF · {ticker} · {tf}"

if df is None or df.empty or len(df) < 80:
    st.error("No/insufficient data returned for this selection.")
    st.stop()

df = df.copy()
for c in ["open", "high", "low", "close", "volume"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna()

# clamp for chart performance
MAX_PLOT_BARS = 900 if not mobile_mode else 450
df_plot = df.tail(MAX_PLOT_BARS).copy()


# ============================================================
# 9) COMPUTE ENGINES
# ============================================================
ap_params = ApexVectorParams(
    eff_super=float(eff_super),
    eff_resist=float(eff_resist),
    vol_norm=int(vol_norm),
    len_vec=int(len_vec),
    sm_type=str(sm_type),
    len_sm=int(len_sm),
    use_vol=True,
    strictness=float(strictness),
    div_look=int(div_look),
)
rq_params = RQZOParams(
    base_harmonics=int(base_harm),
    terminal_volatility=float(term_vol),
    entropy_lookback=int(ent_lb),
    fractal_dim_len=int(fdi_len),
    bands_len=int(bands_len),
)
smc_params = SMCParams(
    smc_lookback=int(smc_lb),
    show_fvg=bool(show_fvg),
    show_ob=bool(show_ob),
    show_structure=bool(show_structure),
)

ap = compute_apex_vector(df_plot, ap_params)
rq = compute_rqzo(df_plot, rq_params)
smc = compute_smc_lite(df_plot, ap["flux"], float(ap["th_resist"].iloc[-1]), smc_params)

trend = None
if use_trend_engine:
    te_params = TrendEngineParams(
        ma_type=str(ma_type),
        len_main=int(len_main),
        mult=float(mult),
        adx_thr=float(adx_thr),
    )
    trend = compute_trend_engine(df_plot, te_params)

# Unified recommendation (HUD-style): LONG when SuperBull + BOS+, SHORT when SuperBear + BOS-
rec = pd.Series("WAIT", index=df_plot.index)
rec[(ap["is_super_bull"]) & (smc["bos_bull"])] = "LONG"
rec[(ap["is_super_bear"]) & (smc["bos_bear"])] = "SHORT"

# Trend signals (optional)
trend_sig = pd.Series("", index=df_plot.index)
if trend is not None:
    trend_sig[trend["sig_buy"]] = "BUY"
    trend_sig[trend["sig_sell"]] = "SELL"

# "NEW signal" detection
new_long = (rec == "LONG") & (rec.shift(1) != "LONG")
new_short = (rec == "SHORT") & (rec.shift(1) != "SHORT")


# ============================================================
# 10) HUD METRICS
# ============================================================
def state_label() -> str:
    if bool(ap["is_super_bull"].iloc[-1]): return "SUPER (BULL)"
    if bool(ap["is_super_bear"].iloc[-1]): return "SUPER (BEAR)"
    if bool(ap["is_resistive"].iloc[-1]): return "RESISTIVE"
    return "HIGH HEAT"

state = state_label()
last_flux = float(ap["flux"].iloc[-1])
last_eff = float(ap["efficiency"].iloc[-1] * 100.0)
last_rqzo = float(rq["rqzo"].iloc[-1])
last_entropy = rq["norm_entropy"].iloc[-1]
last_entropy_str = f"{float(last_entropy):.3f}" if np.isfinite(last_entropy) else "NA"
last_rec = rec.iloc[-1]

bos_txt = "BOS+" if bool(smc["bos_bull"].iloc[-1]) else "BOS-" if bool(smc["bos_bear"].iloc[-1]) else "--"
choch_txt = "CH+" if bool(smc["choch_bull"].iloc[-1]) else "CH-" if bool(smc["choch_bear"].iloc[-1]) else "--"
chaos_txt = "CHAOS" if bool(rq["chaos"].iloc[-1]) else "OK"
latest_time = df_plot.index[-1].to_pydatetime().replace(tzinfo=timezone.utc).isoformat(timespec="seconds")

if mobile_mode:
    st.markdown(f"<div class='pill'>{title_symbol}</div>", unsafe_allow_html=True)
    st.metric("Recommendation", last_rec)
    c1, c2 = st.columns(2)
    c1.metric("State", state)
    c2.metric("Efficiency", f"{last_eff:.0f}%")
    c3, c4 = st.columns(2)
    c3.metric("Apex Flux", f"{last_flux:.3f}")
    c4.metric("RQZO", f"{last_rqzo:.3f}")
else:
    c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1, 1, 1, 1, 1])
    c1.markdown(f"<div class='pill'>{title_symbol}</div>", unsafe_allow_html=True)
    c2.metric("Rec", last_rec)
    c3.metric("State", state)
    c4.metric("Flux", f"{last_flux:.3f}")
    c5.metric("Efficiency", f"{last_eff:.0f}%")
    c6.metric("RQZO", f"{last_rqzo:.3f}")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# ============================================================
# 11) PLOTLY CHART (Pro multi-panel)
# Notes:
# - Avoid triangles (you said they look messy). We use circles/x/diamonds.
# ============================================================
rows = 4 if use_trend_engine else 3
row_heights = [0.52, 0.18, 0.18, 0.12] if use_trend_engine else [0.56, 0.22, 0.22]

fig = make_subplots(
    rows=rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=row_heights,
    specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]] + ([{"secondary_y": False}] if use_trend_engine else []),
)

# --- Price candles
fig.add_trace(
    go.Candlestick(
        x=df_plot.index, open=df_plot["open"], high=df_plot["high"], low=df_plot["low"], close=df_plot["close"],
        name="Price",
        increasing_line_width=1, decreasing_line_width=1,
    ),
    row=1, col=1, secondary_y=False
)

# --- Volume
fig.add_trace(
    go.Bar(x=df_plot.index, y=df_plot["volume"], name="Volume", opacity=0.30),
    row=1, col=1, secondary_y=True
)

# --- Pivot levels
fig.add_trace(go.Scatter(x=df_plot.index, y=smc["last_st_high"], name="Pivot High", mode="lines", line=dict(width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=smc["last_st_low"],  name="Pivot Low",  mode="lines", line=dict(width=1, dash="dot")), row=1, col=1)

# --- BOS/CHoCH markers (subtle)
bos_bull_idx = df_plot.index[smc["bos_bull"]]
bos_bear_idx = df_plot.index[smc["bos_bear"]]
choch_bull_idx = df_plot.index[smc["choch_bull"]]
choch_bear_idx = df_plot.index[smc["choch_bear"]]

if len(bos_bull_idx) > 0:
    fig.add_trace(go.Scatter(
        x=bos_bull_idx, y=df_plot.loc[bos_bull_idx, "close"],
        mode="markers", name="BOS+",
        marker=dict(size=7, symbol="diamond"),
        opacity=0.9
    ), row=1, col=1)

if len(bos_bear_idx) > 0:
    fig.add_trace(go.Scatter(
        x=bos_bear_idx, y=df_plot.loc[bos_bear_idx, "close"],
        mode="markers", name="BOS-",
        marker=dict(size=7, symbol="diamond-open"),
        opacity=0.9
    ), row=1, col=1)

if len(choch_bull_idx) > 0:
    fig.add_trace(go.Scatter(
        x=choch_bull_idx, y=df_plot.loc[choch_bull_idx, "close"],
        mode="markers", name="CH+",
        marker=dict(size=7, symbol="circle-open"),
        opacity=0.9
    ), row=1, col=1)

if len(choch_bear_idx) > 0:
    fig.add_trace(go.Scatter(
        x=choch_bear_idx, y=df_plot.loc[choch_bear_idx, "close"],
        mode="markers", name="CH-",
        marker=dict(size=7, symbol="x"),
        opacity=0.9
    ), row=1, col=1)

# --- Wormholes (FVG) as faint vrect shapes
# Bullish wormhole: low > high[2], draw region between high[2] and low (approx)
if show_fvg:
    wh_bull = smc["wormhole_bull"]
    wh_bear = smc["wormhole_bear"]
    for i in range(2, len(df_plot)):
        if bool(wh_bull.iloc[i]):
            x0 = df_plot.index[i - 2]
            x1 = df_plot.index[i]
            y0 = float(df_plot["high"].iloc[i - 2])
            y1 = float(df_plot["low"].iloc[i])
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, xref="x", yref="y", opacity=0.12, line_width=0, row=1, col=1)
        if bool(wh_bear.iloc[i]):
            x0 = df_plot.index[i - 2]
            x1 = df_plot.index[i]
            y0 = float(df_plot["high"].iloc[i])
            y1 = float(df_plot["low"].iloc[i - 2])
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, xref="x", yref="y", opacity=0.12, line_width=0, row=1, col=1)

# --- Event horizon dots
if show_ob:
    eh_bull = df_plot.index[smc["event_horizon_bull"]]
    eh_bear = df_plot.index[smc["event_horizon_bear"]]
    if len(eh_bull) > 0:
        fig.add_trace(go.Scatter(
            x=eh_bull, y=df_plot.loc[eh_bull, "high"],
            mode="markers", name="OB Bull",
            marker=dict(size=6, symbol="circle"),
            opacity=0.8
        ), row=1, col=1)
    if len(eh_bear) > 0:
        fig.add_trace(go.Scatter(
            x=eh_bear, y=df_plot.loc[eh_bear, "low"],
            mode="markers", name="OB Bear",
            marker=dict(size=6, symbol="x"),
            opacity=0.8
        ), row=1, col=1)

# --- Unified LONG/SHORT markers (avoid triangles)
long_idx = df_plot.index[new_long]
short_idx = df_plot.index[new_short]
if len(long_idx) > 0:
    fig.add_trace(go.Scatter(
        x=long_idx, y=df_plot.loc[long_idx, "low"],
        mode="markers", name="LONG (new)",
        marker=dict(size=10, symbol="circle"),
        opacity=0.95
    ), row=1, col=1)
if len(short_idx) > 0:
    fig.add_trace(go.Scatter(
        x=short_idx, y=df_plot.loc[short_idx, "high"],
        mode="markers", name="SHORT (new)",
        marker=dict(size=10, symbol="x"),
        opacity=0.95
    ), row=1, col=1)

# --- Apex Flux panel (histogram + thresholds + divergence markers)
fig.add_trace(go.Bar(x=df_plot.index, y=ap["flux"], name="Apex Flux", opacity=0.85), row=2, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=ap["th_super"], name="+Super Th", mode="lines", line=dict(width=1, dash="dot")), row=2, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=-ap["th_super"], name="-Super Th", mode="lines", line=dict(width=1, dash="dot")), row=2, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=ap["th_resist"], name="+Resist Th", mode="lines", line=dict(width=1, dash="dot")), row=2, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=-ap["th_resist"], name="-Resist Th", mode="lines", line=dict(width=1, dash="dot")), row=2, col=1)

div_bull = ap["div_bull_reg"] | ap["div_bull_hid"]
div_bear = ap["div_bear_reg"] | ap["div_bear_hid"]

if div_bull.any():
    fig.add_trace(go.Scatter(
        x=df_plot.index[div_bull], y=ap["flux"][div_bull],
        mode="markers", name="Div Bull",
        marker=dict(size=8, symbol="circle-open"),
        opacity=0.9
    ), row=2, col=1)
if div_bear.any():
    fig.add_trace(go.Scatter(
        x=df_plot.index[div_bear], y=ap["flux"][div_bear],
        mode="markers", name="Div Bear",
        marker=dict(size=8, symbol="x"),
        opacity=0.9
    ), row=2, col=1)

# --- RQZO panel + bands + chaos shading
fig.add_trace(go.Scatter(x=df_plot.index, y=rq["rqzo"], name="RQZO", mode="lines", line=dict(width=2)), row=3, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=rq["upper_band"], name="RQZO Upper", mode="lines", line=dict(width=1, dash="dot")), row=3, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=rq["lower_band"], name="RQZO Lower", mode="lines", line=dict(width=1, dash="dot")), row=3, col=1)
fig.add_hline(y=0, line_dash="dot", line_width=1, row=3, col=1)

# Chaos zones as vrect blocks
chaos = rq["chaos"].fillna(False)
if chaos.any():
    in_run = False
    start = None
    for t, flag in chaos.items():
        if flag and not in_run:
            in_run = True
            start = t
        if in_run and (not flag):
            end = t
            fig.add_vrect(x0=start, x1=end, opacity=0.10, line_width=0, row=3, col=1)
            in_run = False
    if in_run:
        fig.add_vrect(x0=start, x1=df_plot.index[-1], opacity=0.10, line_width=0, row=3, col=1)

# --- Optional Trend panel
if use_trend_engine and trend is not None:
    r = 4
    fig.add_trace(go.Scatter(x=df_plot.index, y=trend["baseline"], name="Trend Baseline", mode="lines", line=dict(width=1)), row=r, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=trend["upper"], name="Trend Upper", mode="lines", line=dict(width=1, dash="dot")), row=r, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=trend["lower"], name="Trend Lower", mode="lines", line=dict(width=1, dash="dot")), row=r, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=trend["trail_stop"], name="Trail Stop", mode="lines", line=dict(width=2)), row=r, col=1)

    tbuy = df_plot.index[trend["sig_buy"]]
    tsell = df_plot.index[trend["sig_sell"]]
    if len(tbuy) > 0:
        fig.add_trace(go.Scatter(
            x=tbuy, y=df_plot.loc[tbuy, "close"],
            mode="markers", name="Trend BUY",
            marker=dict(size=8, symbol="circle"),
            opacity=0.95
        ), row=r, col=1)
    if len(tsell) > 0:
        fig.add_trace(go.Scatter(
            x=tsell, y=df_plot.loc[tsell, "close"],
            mode="markers", name="Trend SELL",
            marker=dict(size=8, symbol="x"),
            opacity=0.95
        ), row=r, col=1)

chart_height = 720 if not mobile_mode else 560
fig.update_layout(
    height=chart_height,
    margin=dict(l=8, r=8, t=40, b=10),
    title=dict(text=f"{title_symbol} — Unified Signals", x=0.01),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
    xaxis_rangeslider_visible=False,
)
fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Vol", row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="Apex Flux", row=2, col=1)
fig.update_yaxes(title_text="RQZO", row=3, col=1)
if use_trend_engine:
    fig.update_yaxes(title_text="Trend", row=4, col=1)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 12) SIGNALS / EXPORT / TELEGRAM / AI
# ============================================================
left, right = st.columns([1.12, 0.88]) if not mobile_mode else (st.container(), st.container())

# ---- Signal snapshot
snap = {
    "symbol": title_symbol,
    "timestamp_utc": latest_time,
    "state": state,
    "structure_bos": bos_txt,
    "structure_choch": choch_txt,
    "recommendation": last_rec,
    "apex_flux": float(last_flux),
    "efficiency_pct": float(last_eff),
    "rqzo": float(last_rqzo),
    "norm_entropy": None if not np.isfinite(last_entropy) else float(last_entropy),
    "chaos": bool(rq["chaos"].iloc[-1]),
}
if trend is not None:
    snap.update({
        "trend_state": int(trend["trend"].iloc[-1]),
        "adx": float(trend["adx"].iloc[-1]),
        "tci": float(trend["tci"].iloc[-1]),
        "trend_sig": trend_sig.iloc[-1],
    })

signal_msg = (
    f"<b>⚛️ QUANTUM SIGNAL</b>\n"
    f"<b>Market:</b> {title_symbol}\n"
    f"<b>Time (UTC):</b> {latest_time}\n"
    f"<b>State:</b> {state}\n"
    f"<b>Struct:</b> {bos_txt} {choch_txt}\n"
    f"<b>Rec:</b> {last_rec}\n"
    f"<b>Apex Flux:</b> {last_flux:.4f}\n"
    f"<b>Efficiency:</b> {last_eff:.0f}%\n"
    f"<b>RQZO:</b> {last_rqzo:.4f}\n"
    f"<b>Entropy:</b> {last_entropy_str}\n"
    f"<b>Regime:</b> {chaos_txt}\n"
)

with left:
    st.subheader("Signals")
    # table for pro scanning
    rows_tbl = [
        ("Timestamp (UTC)", latest_time),
        ("State", state),
        ("BOS / CHoCH", f"{bos_txt} / {choch_txt}"),
        ("Recommendation", last_rec),
        ("Apex Flux", f"{last_flux:.4f}"),
        ("Efficiency", f"{last_eff:.0f}%"),
        ("RQZO", f"{last_rqzo:.4f}"),
        ("Norm Entropy", last_entropy_str),
        ("Entropy Regime", chaos_txt),
    ]
    if trend is not None:
        rows_tbl += [
            ("Trend State", str(int(trend["trend"].iloc[-1]))),
            ("ADX", f"{float(trend['adx'].iloc[-1]):.2f}"),
            ("WaveTrend (tci)", f"{float(trend['tci'].iloc[-1]):.2f}"),
            ("Trend Signal", trend_sig.iloc[-1] or "--"),
        ]

    st.dataframe(pd.DataFrame(rows_tbl, columns=["Key", "Value"]), use_container_width=True, hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    cA, cB, cC = st.columns([1, 1, 1]) if not mobile_mode else (st.columns(3))
    with cA:
        if st.button("📣 Send Telegram", use_container_width=True, disabled=not send_to_telegram):
            ok, info = telegram_send(signal_msg)
            st.success("Telegram: sent ✅" if ok else f"Telegram: failed ❌ — {info}")
    with cB:
        st.download_button(
            "⬇️ Snapshot JSON",
            data=json.dumps(snap, indent=2),
            file_name="signal_snapshot.json",
            mime="application/json",
            use_container_width=True
        )
    with cC:
        st.download_button(
            "⬇️ Data CSV (plot)",
            data=df_plot.to_csv(index=True).encode("utf-8"),
            file_name="ohlcv_plot.csv",
            mime="text/csv",
            use_container_width=True
        )

# ---- AI Brief
with right:
    st.subheader("AI Brief")
    st.caption("Default provider: Gemini (if available). Uses computed state only.")

    if provider == "Off":
        st.info("AI is off.")
    else:
        if not ai_ready(provider):
            st.warning("AI provider not ready (missing dependency or API key in secrets).")
        else:
            ai_prompt = f"""
Return:
1) Regime summary (1 line).
2) Trade plan (2-4 lines) referencing state/structure/entropy.
3) Risk checklist (3 bullets).
4) Confirmations to wait for (3 bullets).
5) One alternative scenario (1-2 lines).

DATA:
symbol={title_symbol}
timestamp_utc={latest_time}
state={state}
bos={bos_txt}
choch={choch_txt}
recommendation={last_rec}
apex_flux={last_flux}
efficiency_pct={last_eff}
rqzo={last_rqzo}
norm_entropy={last_entropy_str}
chaos={chaos_txt}
trend_engine_enabled={use_trend_engine}
trend_sig={trend_sig.iloc[-1] if trend is not None else "--"}
"""
            if st.button("🧠 Generate AI Brief", use_container_width=True):
                with st.spinner("Generating…"):
                    try:
                        text = ai_generate(provider, ai_prompt)
                        st.write(text)

                        if send_to_telegram and st.toggle("Send AI brief to Telegram now", value=False):
                            ok, info = telegram_send(f"<b>🧠 AI BRIEF</b>\n<b>{title_symbol}</b>\n\n{text}")
                            st.caption("Telegram AI brief sent ✅" if ok else f"Telegram AI brief failed ❌ — {info}")
                    except Exception as e:
                        st.error(str(e))

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# ============================================================
# 13) AUTO TELEGRAM ON NEW SIGNAL (dedupe)
# ============================================================
if "last_sent_key" not in st.session_state:
    st.session_state.last_sent_key = ""

if send_to_telegram and auto_telegram_on_new_signal:
    # Create a deterministic key to avoid repeat sends on reruns
    # Key changes only when a NEW signal appears on the latest bar.
    latest_sig = "NONE"
    if bool(new_long.iloc[-1]):
        latest_sig = "LONG"
    elif bool(new_short.iloc[-1]):
        latest_sig = "SHORT"

    send_key = f"{title_symbol}|{df_plot.index[-1]}|{latest_sig}"
    if latest_sig != "NONE" and st.session_state.last_sent_key != send_key:
        ok, info = telegram_send(signal_msg)
        if ok:
            st.session_state.last_sent_key = send_key
        else:
            st.warning(f"Auto-telegram failed: {info}")


# ============================================================
# 14) FOOTER / HEALTH
# ============================================================
deps = {
    "ccxt": CCXT_OK,
    "yfinance": YF_OK,
    "openai": OPENAI_OK,
    "gemini": GEMINI_OK,
    "autorefresh": AUTOR_OK,
}
st.markdown(
    "<div class='small-muted'>Runtime deps: "
    + " · ".join([f"{k}={'OK' if v else 'MISSING'}" for k, v in deps.items()])
    + "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='small-muted'>Secrets expected: GEMINI_API_KEY (optional), OPENAI_API_KEY (optional), TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID (optional).</div>",
    unsafe_allow_html=True,
)
