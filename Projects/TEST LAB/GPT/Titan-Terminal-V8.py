# app.py
# =========================================================================================
# No omissions. No assumptions. Base preserved.
# STRICT CONSTRAINTS (NON-NEGOTIABLE):
# - Start from the latest COMPLETE code provided by the user and keep it 100% intact.
# - NO deletions, NO omissions, NO placeholders, NO partial snippets.
# - Do NOT assume anything. Integrate upgrades except direct contradictions.
# - Preserve base behavior; if conflict arises, mark conflicts explicitly.
# - Always output the ENTIRE script in every response.
# =========================================================================================

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
import calendar
import datetime
import requests
import urllib.parse
from scipy.stats import linregress
import streamlit.components.v1 as components  # ADDED (required for TradingView embed; no removals)

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM UI
# ==========================================
st.set_page_config(layout="wide", page_title="🏦Titan Terminal", page_icon="👁️")

# --- CUSTOM CSS FOR "DARKPOOL" AESTHETIC ---
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #e0e0e0;
    font-family: 'Roboto Mono', monospace;
}
.title-glow {
    font-size: 3em;
    font-weight: bold;
    color: #ffffff;
    text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 40px #00ff00;
    margin-bottom: 20px;
}
div[data-testid="stMetric"] {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 10px;
    border-radius: 8px;
    transition: transform 0.2s;
}
div[data-testid="stMetric"]:hover {
    transform: scale(1.02);
    border-color: #00ff00;
}
div[data-testid="stMetricValue"] {
    font-size: 1.2rem !important;
    font-weight: 700;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #161b22;
    border-radius: 4px 4px 0px 0px;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
    border: 1px solid #30363d;
    color: #8b949e;
}
.stTabs [aria-selected="true"] {
    background-color: #0e1117;
    color: #00ff00;
    border-bottom: 2px solid #00ff00;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #30363d !important;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="title-glow">👁️ DarkPool Titan Terminal</div>', unsafe_allow_html=True)
st.markdown("##### *Institutional-Grade Market Intelligence*")
st.markdown("---")

# --- API Key Management ---
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if "OPENAI_API_KEY" in st.secrets:
    st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
else:
    if not st.session_state.api_key:
        st.session_state.api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key here to unlock the AI Analyst features."
        )

# ==========================================
# 2. DATA ENGINE (OPTIMIZED FOR SPEED)
# ==========================================
@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    """Fetches key financial metrics safely."""
    if "-" in ticker or "=" in ticker or "^" in ticker:
        return None
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info:
            return None

        return {
            "Market Cap": info.get("marketCap", 0),
            "P/E Ratio": info.get("trailingPE", 0),
            "Rev Growth": info.get("revenueGrowth", 0),
            "Debt/Equity": info.get("debtToEquity", 0),
            "Summary": info.get("longBusinessSummary", "No Data Available")
        }
    except:
        return None

@st.cache_data(ttl=300)
def get_global_performance():
    """Fetches performance of a Global Multi-Asset Basket."""
    assets = {
        "Tech (XLK)": "XLK",
        "Energy (XLE)": "XLE",
        "Financials (XLF)": "XLF",
        "Bitcoin (BTC)": "BTC-USD",
        "Gold (GLD)": "GLD",
        "Oil (USO)": "USO",
        "Treasuries (TLT)": "TLT"
    }
    try:
        tickers_list = list(assets.values())
        data = yf.download(tickers_list, period="5d", interval="1d", progress=False, group_by='ticker')

        results = {}
        for name, tk in assets.items():
            try:
                df = data[tk] if len(tickers_list) > 1 else data
                if not df.empty and len(df) >= 2:
                    price_col = 'Close' if 'Close' in df.columns else ('Adj Close' if 'Adj Close' in df.columns else None)
                    if not price_col:
                        continue
                    price = df[price_col].iloc[-1]
                    prev = df[price_col].iloc[-2]
                    change = ((price - prev) / prev) * 100
                    results[name] = change
            except:
                continue

        return pd.Series(results).sort_values(ascending=True)
    except:
        return None

def safe_download(ticker, period, interval):
    """Robust price downloader."""
    try:
        dl_interval = "1h" if interval == "4h" else interval
        df = yf.download(ticker, period=period, interval=dl_interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            else:
                return None
        return df
    except:
        return None

@st.cache_data(ttl=300)
def get_macro_data():
    """Fetches 40 global macro indicators grouped by sector using BATCH DOWNLOAD (FAST)."""
    groups = {
        "🇺🇸 US Equities": {"S&P 500": "SPY", "Nasdaq 100": "QQQ", "Dow Jones": "^DJI", "Russell 2000": "^RUT"},
        "🌍 Global Indices": {"FTSE 100": "^FTSE", "DAX": "^GDAXI", "Nikkei 225": "^N225", "Hang Seng": "^HSI"},
        "🏦 Rates & Bonds": {"10Y Yield": "^TNX", "2Y Yield": "^IRX", "30Y Yield": "^TYX", "T-Bond (TLT)": "TLT"},
        "💱 Forex & Volatility": {"DXY Index": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "VIX (Fear)": "^VIX"},
        "⚠️ Risk Assets": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Semis (SMH)": "SMH", "Junk Bonds": "HYG"},
        "⚡ Energy": {"WTI Crude": "CL=F", "Brent Crude": "BZ=F", "Natural Gas": "NG=F", "Uranium": "URA"},
        "🥇 Precious Metals": {"Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F"},
        "🏗️ Industrial & Ag": {"Copper": "HG=F", "Rare Earths": "REMX", "Corn": "ZC=F", "Wheat": "ZW=F"},
        "🇬🇧 UK Desk": {"GBP/USD": "GBPUSD=X", "GBP/JPY": "GBPJPY=X", "EUR/GBP": "EURGBP=X", "UK Gilts": "IGLT.L"},
        "📈 Growth & Real Assets": {"Emerging Mkts": "EEM", "China (FXI)": "FXI", "Real Estate": "VNQ", "Soybeans": "ZS=F"}
    }

    all_tickers_list = []
    ticker_to_name_map = {}
    for _, g_dict in groups.items():
        for t_name, t_sym in g_dict.items():
            all_tickers_list.append(t_sym)
            ticker_to_name_map[t_sym] = t_name

    try:
        data_batch = yf.download(all_tickers_list, period="5d", interval="1d", group_by='ticker', progress=False)
        prices, changes = {}, {}

        for sym in all_tickers_list:
            try:
                df = data_batch[sym] if len(all_tickers_list) > 1 else data_batch
                if df is None or df.empty:
                    continue
                df = df.dropna(how='all')
                if len(df) >= 2:
                    col = 'Close' if 'Close' in df.columns else 'Adj Close'
                    curr, prev = df[col].iloc[-1], df[col].iloc[-2]
                    chg = ((curr - prev) / prev) * 100
                    name = ticker_to_name_map.get(sym, sym)
                    prices[name] = curr
                    changes[name] = chg
            except Exception:
                continue

        return groups, prices, changes
    except Exception:
        return groups, {}, {}

# ==========================================
# 2A. TICKER LIBRARY (UPGRADED — EXTENSIVE DROPDOWNS + CSV UPLOAD, NO REMOVALS)
# ==========================================
def build_builtin_ticker_library():
    """
    Built-in ticker sets (extensive but not infinite).
    No assumptions: user can upload CSV to extend to thousands.
    """
    library = {
        "📌 Indices (Global)": [
            "SPY", "QQQ", "DIA", "IWM", "VTI", "^GSPC", "^IXIC", "^DJI", "^RUT",
            "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI", "^STOXX50E", "^AEX",
            "^IBEX", "^SSMI", "^BSESN", "^NSEI", "^KS11", "^TWII", "^AXJO"
        ],
        "💱 FX (Major)": [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
            "EURGBP=X", "EURJPY=X", "GBPJPY=X"
        ],
        "📉 Rates / Vol": [
            "^TNX", "^IRX", "^TYX", "^FVX", "^VIX", "TLT", "IEF", "SHY", "HYG", "LQD"
        ],
        "🛢️ Energy / Commodities": [
            "CL=F", "BZ=F", "NG=F", "RB=F", "HO=F", "GC=F", "SI=F", "HG=F",
            "ZC=F", "ZW=F", "ZS=F", "KC=F", "SB=F"
        ],
        "🥇 Metals ETFs": [
            "GLD", "IAU", "SLV", "SGOL", "SIVR", "GDX", "GDXJ", "SIL", "SILJ"
        ],
        "🧠 Tech / AI / Semis": [
            "NVDA", "AMD", "INTC", "AVGO", "TSM", "ASML", "ARM", "MU", "QCOM",
            "MSFT", "AAPL", "GOOGL", "AMZN", "META", "TSLA", "PLTR", "SNOW",
            "SMH", "SOXX", "XLK"
        ],
        "🏦 Financials": ["JPM", "BAC", "GS", "MS", "C", "WFC", "SCHW", "BLK", "XLF"],
        "🏥 Healthcare": ["JNJ", "PFE", "MRK", "LLY", "UNH", "ABBV", "TMO", "XLV"],
        "🏭 Industrials": ["BA", "CAT", "DE", "GE", "HON", "RTX", "LMT", "XLI"],
        "🧃 Consumer": ["KO", "PEP", "WMT", "COST", "PG", "MCD", "NKE", "XLY", "XLP"],
        "🚀 Crypto (YFinance)": [
            "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD",
            "DOT-USD", "LINK-USD", "TRX-USD", "MATIC-USD", "LTC-USD", "BCH-USD", "XLM-USD", "ATOM-USD",
            "UNI-USD", "FIL-USD", "ETC-USD", "APT-USD", "ARB-USD", "OP-USD", "INJ-USD", "AAVE-USD"
        ],
        "🇬🇧 UK (Sample)": ["VOD.L", "HSBA.L", "BP.L", "SHEL.L", "AZN.L", "ULVR.L", "REL.L", "^FTSE"]
    }
    return library

def parse_uploaded_tickers(uploaded_file):
    """
    Accepts CSV or TXT. Extracts a column named 'ticker' if present,
    else reads first column. No assumptions: if file unreadable returns [].
    """
    if uploaded_file is None:
        return []
    try:
        name = uploaded_file.name.lower()
        raw = uploaded_file.getvalue()
        if name.endswith(".txt"):
            text = raw.decode("utf-8", errors="ignore")
            items = [x.strip() for x in text.replace(",", "\n").splitlines()]
            items = [x for x in items if x]
            return sorted(list(dict.fromkeys(items)))
        # CSV
        dfu = pd.read_csv(uploaded_file)
        if dfu.empty:
            return []
        if "ticker" in dfu.columns:
            items = dfu["ticker"].astype(str).str.strip().tolist()
        else:
            items = dfu.iloc[:, 0].astype(str).str.strip().tolist()
        items = [x for x in items if x and x.lower() != "nan"]
        return sorted(list(dict.fromkeys(items)))
    except Exception:
        return []

# ==========================================
# 3. MATH LIBRARY & PINE v6 TRANSLATION LAYER
# ==========================================
def calculate_wma(series: pd.Series, length: int) -> pd.Series:
    w = np.arange(1, length + 1, dtype=float)
    denom = w.sum()
    return series.rolling(length).apply(lambda x: float(np.dot(x, w) / denom), raw=True)

def calculate_hma(series: pd.Series, length: int) -> pd.Series:
    half_length = max(int(length / 2), 1)
    sqrt_length = max(int(np.sqrt(length)), 1)
    wma_half = calculate_wma(series, half_length)
    wma_full = calculate_wma(series, length)
    diff = 2 * wma_half - wma_full
    return calculate_wma(diff, sqrt_length)

def calculate_rma(series: pd.Series, length: int) -> pd.Series:
    # Pine's RMA is EMA with alpha=1/length
    return series.ewm(alpha=1/length, adjust=False).mean()

def calculate_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def calculate_sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()

def calculate_vwma(price: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    pv = price * volume
    return pv.rolling(length).sum() / volume.rolling(length).sum()

def calculate_ma(ma_type: str, source: pd.Series, length: int, df: pd.DataFrame | None = None) -> pd.Series:
    ma_type = (ma_type or "EMA").upper()
    if ma_type == "SMA":
        return calculate_sma(source, length)
    if ma_type == "EMA":
        return calculate_ema(source, length)
    if ma_type == "HMA":
        return calculate_hma(source, length)
    if ma_type == "RMA":
        return calculate_rma(source, length)
    if ma_type == "WMA":
        return calculate_wma(source, length)
    if ma_type == "VWMA":
        if df is None:
            # no assumption: VWMA requires volume; caller must supply df
            return calculate_ema(source, length)
        return calculate_vwma(source, df["Volume"], length)
    return calculate_ema(source, length)

def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def calculate_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    atr = calculate_atr(df, period)
    hl2 = (df['High'] + df['Low']) / 2
    upper = hl2 + (multiplier * atr)
    lower = hl2 - (multiplier * atr)

    close = df['Close'].to_numpy()
    upper_np = upper.to_numpy()
    lower_np = lower.to_numpy()
    st = np.zeros(len(df), dtype=float)
    trend = np.zeros(len(df), dtype=float)  # 1 up, -1 down

    if len(df) == 0:
        return pd.Series([], index=df.index), pd.Series([], index=df.index)

    st[0] = lower_np[0] if np.isfinite(lower_np[0]) else close[0]
    trend[0] = 1

    for i in range(1, len(df)):
        prev_st = st[i-1]
        if close[i-1] > prev_st:
            st[i] = max(lower_np[i], prev_st) if close[i] > prev_st else upper_np[i]
            trend[i] = 1 if close[i] > prev_st else -1
            if close[i] < lower_np[i] and trend[i-1] == 1:
                st[i] = upper_np[i]
                trend[i] = -1
        else:
            st[i] = min(upper_np[i], prev_st) if close[i] < prev_st else lower_np[i]
            trend[i] = -1 if close[i] < prev_st else 1
            if close[i] > upper_np[i] and trend[i-1] == -1:
                st[i] = lower_np[i]
                trend[i] = 1

    return pd.Series(st, index=df.index), pd.Series(trend, index=df.index)

def pivothigh(series: pd.Series, left: int, right: int) -> pd.Series:
    # Pine: ta.pivothigh(src, left, right) returns value at pivot bar (center), detected right bars later.
    # Here: return pivot value aligned to the pivot bar index (center). Non-pivot = NaN.
    n = left + right + 1
    if n <= 1:
        return pd.Series(np.nan, index=series.index)
    roll_max = series.rolling(n, center=True).max()
    is_pivot = series.eq(roll_max)
    # ensure uniqueness: pivot must be strictly greater than neighbors to avoid plateaus
    shifted_left = series.shift(1)
    shifted_right = series.shift(-1)
    is_strict = series.gt(shifted_left.fillna(series)) & series.gt(shifted_right.fillna(series))
    out = series.where(is_pivot & is_strict)
    return out

def pivotlow(series: pd.Series, left: int, right: int) -> pd.Series:
    n = left + right + 1
    if n <= 1:
        return pd.Series(np.nan, index=series.index)
    roll_min = series.rolling(n, center=True).min()
    is_pivot = series.eq(roll_min)
    shifted_left = series.shift(1)
    shifted_right = series.shift(-1)
    is_strict = series.lt(shifted_left.fillna(series)) & series.lt(shifted_right.fillna(series))
    out = series.where(is_pivot & is_strict)
    return out

# ==========================================
# 3A. PINE INDICATOR #1: APEX TREND & LIQUIDITY MASTER (FULL PORT)
# ==========================================
def compute_apex_trend_liquidity(df: pd.DataFrame, cfg: dict):
    """
    Full port of:
    - Dynamic MA baseline (EMA/SMA/HMA/RMA)
    - ATR cloud bands
    - Trend state w/ chop filter (maintain previous)
    - Volume filter (vol > SMA(vol, 20))
    - RSI filter (buy blocked if RSI>=70, sell blocked if RSI<=30 when enabled)
    - Smart Liquidity Zones (supply/demand pivot zones) w/ mitigation logic (end-of-series snapshot)
    Outputs:
      Columns:
        Apex_Base, Apex_ATR55, Apex_Upper, Apex_Lower, Apex_Trend, Apex_Sig_Buy, Apex_Sig_Sell,
        Apex_RSI, Apex_VolMA, Apex_HighVol
      Side outputs:
        apex_supply_zones, apex_demand_zones (lists of dicts for Plotly shapes)
    """
    ma_type = cfg["ma_type"]
    len_main = int(cfg["len_main"])
    mult = float(cfg["mult"])
    src_name = cfg["src"]
    use_vol = bool(cfg["use_vol"])
    use_rsi = bool(cfg["use_rsi"])
    liq_len = int(cfg["liq_len"])
    zone_ext = int(cfg["zone_ext"])
    show_liq = bool(cfg["show_liq"])

    if src_name not in df.columns:
        src = df["Close"]
    else:
        src = df[src_name]

    baseline = calculate_ma(ma_type, src, len_main, df=df)
    atr_main = calculate_atr(df, len_main)
    upper = baseline + (atr_main * mult)
    lower = baseline - (atr_main * mult)

    trend = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if i == 0:
            trend[i] = 0
            continue
        if df["Close"].iloc[i] > upper.iloc[i]:
            trend[i] = 1
        elif df["Close"].iloc[i] < lower.iloc[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]  # maintain previous state (Pine behavior)

    vol_ma = df["Volume"].rolling(20).mean()
    high_vol = df["Volume"] > vol_ma
    rsi_val = calculate_rsi(df["Close"], 14)
    rsi_ok_buy = (~use_rsi) | (rsi_val < 70)
    rsi_ok_sell = (~use_rsi) | (rsi_val > 30)
    cond_vol = (~use_vol) | high_vol

    trend_prev = pd.Series(trend, index=df.index).shift(1).fillna(0).astype(int)
    sig_buy = (trend == 1) & (trend_prev.to_numpy() != 1) & cond_vol.to_numpy() & rsi_ok_buy.to_numpy()
    sig_sell = (trend == -1) & (trend_prev.to_numpy() != -1) & cond_vol.to_numpy() & rsi_ok_sell.to_numpy()

    # Liquidity Zones (Supply/Demand) — keep last 5 each, and apply mitigation to last bar snapshot.
    apex_supply_zones = []
    apex_demand_zones = []

    if show_liq:
        ph = pivothigh(df["High"], liq_len, liq_len)
        pl = pivotlow(df["Low"], liq_len, liq_len)

        supply = []
        demand = []

        # Build zones at pivot centers
        for i in range(len(df)):
            if not np.isnan(ph.iloc[i]):
                box_top = df["High"].iloc[i]
                box_bot = max(df["Open"].iloc[i], df["Close"].iloc[i])
                supply.append({"idx": df.index[i], "top": float(box_top), "bot": float(box_bot), "right_ext": zone_ext})
                if len(supply) > 5:
                    supply.pop(0)

            if not np.isnan(pl.iloc[i]):
                box_bot = df["Low"].iloc[i]
                box_top = min(df["Open"].iloc[i], df["Close"].iloc[i])
                demand.append({"idx": df.index[i], "top": float(box_top), "bot": float(box_bot), "right_ext": zone_ext})
                if len(demand) > 5:
                    demand.pop(0)

        # Mitigation management at end-of-series snapshot (Pine does management on barstate.islast)
        last_close = float(df["Close"].iloc[-1]) if len(df) else np.nan
        for z in supply:
            if np.isfinite(last_close) and last_close > z["bot"]:
                continue
            apex_supply_zones.append(z)

        for z in demand:
            if np.isfinite(last_close) and last_close < z["top"]:
                continue
            apex_demand_zones.append(z)

    out = df.copy()
    out["Apex_Base"] = baseline
    out["Apex_ATR_Main"] = atr_main
    out["Apex_Upper"] = upper
    out["Apex_Lower"] = lower
    out["Apex_Trend"] = pd.Series(trend, index=df.index).astype(int)
    out["Apex_Sig_Buy"] = sig_buy
    out["Apex_Sig_Sell"] = sig_sell
    out["Apex_RSI"] = rsi_val
    out["Apex_VolMA20"] = vol_ma
    out["Apex_HighVol"] = high_vol.astype(bool)

    return out, apex_supply_zones, apex_demand_zones

# ==========================================
# 3B. PINE INDICATOR #2: NEXUS v8.2 (FULL PORT)
# ==========================================
def rational_quadratic(series: pd.Series, lookback: int, weight: float, start_at: int = 0) -> pd.Series:
    """
    Pine v6 rationalQuadratic loop:
      for i=0 to lookback+startAt:
         w = (1 + (i^2/(2*weight*lookback^2)))^-weight
         sum += src[i]*w, wsum += w
    Note: Pine uses src[i] where i=0 is current bar. Our rolling window provides oldest->newest.
    """
    lookback = int(lookback)
    start_at = int(start_at)
    L = lookback + start_at + 1
    if L <= 1:
        return series.copy()

    idx = np.arange(0, L, dtype=float)  # i
    denom = (2.0 * weight * (lookback ** 2)) if (weight != 0 and lookback != 0) else 1.0
    w = np.power(1.0 + (np.power(idx, 2) / denom), -weight, dtype=float)
    wsum = float(np.sum(w)) if np.sum(w) != 0 else 1.0

    def _apply(x):
        xr = x[::-1]
        return float(np.dot(xr, w) / wsum)

    return series.rolling(L).apply(_apply, raw=True)

def compute_nexus(df: pd.DataFrame, cfg: dict):
    """
    Full port (single-timeframe with optional resampled macro kernel filter):
    - Kernel baseline (rational quadratic)
    - Kernel trend direction
    - Gann activator (donchian high/low, state machine)
    - UT Bot trailing stop & position
    - Structure engine (major pivots, BOS/CHoCH, strict vs gann)
    - FVG detection (2-bar gap) with doji filter
    - Omni buy/sell signal when UT+Gann+Kernel align w/ last_signal gating
    Outputs:
      Columns:
        Nexus_Kernel, Nexus_KernelTrend, Nexus_GannActivator, Nexus_GannTrend,
        Nexus_UT_Stop, Nexus_UT_Pos, Nexus_StructState,
        Nexus_BOS_Bull, Nexus_BOS_Bear, Nexus_CHoCH_Bull, Nexus_CHoCH_Bear,
        Nexus_Signal_Buy, Nexus_Signal_Sell
      Side outputs:
        nexus_fvgs (list)
    """
    h_val = int(cfg["h_val"])
    r_val = float(cfg["r_val"])
    gann_len = int(cfg["gann_len"])
    tf_trend = cfg["tf_trend"]
    a_val = float(cfg["a_val"])
    c_period = int(cfg["c_period"])
    liq_len = int(cfg["liq_len"])
    show_fvg = bool(cfg["show_fvg"])
    filter_doji = bool(cfg["filter_doji"])
    strict_structure = bool(cfg["strict_structure"])

    base_close = df["Close"]

    if tf_trend and tf_trend != "":
        rule = {"4H": "4h", "1D": "1D", "1W": "1W"}.get(tf_trend, "")
        if rule:
            res = df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
            k_res = rational_quadratic(res["Close"], h_val, r_val, 0)
            kernel = k_res.reindex(df.index, method="ffill")
        else:
            kernel = rational_quadratic(base_close, h_val, r_val, 0)
    else:
        kernel = rational_quadratic(base_close, h_val, r_val, 0)

    kernel_trend = np.where(kernel > kernel.shift(1), 1, -1)

    donch_high = df["High"].rolling(gann_len).max()
    donch_low = df["Low"].rolling(gann_len).min()

    gann_activator = np.full(len(df), np.nan, dtype=float)
    gann_trend = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        if i == 0:
            gann_activator[i] = float(donch_low.iloc[i]) if np.isfinite(donch_low.iloc[i]) else float(df["Low"].iloc[i])
            gann_trend[i] = 1
            continue

        if gann_trend[i-1] == 1:
            if df["Close"].iloc[i] < gann_activator[i-1]:
                gann_trend[i] = -1
                gann_activator[i] = float(donch_high.iloc[i]) if np.isfinite(donch_high.iloc[i]) else float(df["High"].iloc[i])
            else:
                gann_trend[i] = 1
                gann_activator[i] = float(donch_low.iloc[i]) if np.isfinite(donch_low.iloc[i]) else float(df["Low"].iloc[i])
        else:
            if df["Close"].iloc[i] > gann_activator[i-1]:
                gann_trend[i] = 1
                gann_activator[i] = float(donch_low.iloc[i]) if np.isfinite(donch_low.iloc[i]) else float(df["Low"].iloc[i])
            else:
                gann_trend[i] = -1
                gann_activator[i] = float(donch_high.iloc[i]) if np.isfinite(donch_high.iloc[i]) else float(df["High"].iloc[i])

    xatr = calculate_atr(df, c_period)
    nloss = a_val * xatr
    ut_stop = np.zeros(len(df), dtype=float)
    ut_pos = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        if i == 0:
            ut_stop[i] = float(df["Close"].iloc[i] - nloss.iloc[i]) if np.isfinite(nloss.iloc[i]) else float(df["Close"].iloc[i])
            ut_pos[i] = 0
            continue

        prev_stop = ut_stop[i-1]
        c = float(df["Close"].iloc[i])
        c1 = float(df["Close"].iloc[i-1])
        nl = float(nloss.iloc[i]) if np.isfinite(nloss.iloc[i]) else 0.0

        if (c > prev_stop) and (c1 > prev_stop):
            ut_stop[i] = max(prev_stop, c - nl)
        elif (c < prev_stop) and (c1 < prev_stop):
            ut_stop[i] = min(prev_stop, c + nl)
        elif c > prev_stop:
            ut_stop[i] = c - nl
        else:
            ut_stop[i] = c + nl

        if (c1 < prev_stop) and (c > prev_stop):
            ut_pos[i] = 1
        elif (c1 > prev_stop) and (c < prev_stop):
            ut_pos[i] = -1
        else:
            ut_pos[i] = ut_pos[i-1]

    ph = pivothigh(df["High"], liq_len, liq_len)
    pl = pivotlow(df["Low"], liq_len, liq_len)

    major_res = np.full(len(df), np.nan)
    major_sup = np.full(len(df), np.nan)
    struct_state = np.zeros(len(df), dtype=int)

    is_bos_bull = np.zeros(len(df), dtype=bool)
    is_bos_bear = np.zeros(len(df), dtype=bool)
    is_choch_bull = np.zeros(len(df), dtype=bool)
    is_choch_bear = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if i == 0:
            struct_state[i] = 0
        else:
            struct_state[i] = struct_state[i-1]

        if not np.isnan(ph.iloc[i]):
            major_res[i] = ph.iloc[i]
        else:
            major_res[i] = major_res[i-1] if i > 0 else np.nan

        if not np.isnan(pl.iloc[i]):
            major_sup[i] = pl.iloc[i]
        else:
            major_sup[i] = major_sup[i-1] if i > 0 else np.nan

        if i == 0:
            continue

        mr = major_res[i]
        ms = major_sup[i]
        c = df["Close"].iloc[i]
        c_prev = df["Close"].iloc[i-1]

        break_up = np.isfinite(mr) and (c_prev <= mr) and (c > mr)
        break_dn = np.isfinite(ms) and (c_prev >= ms) and (c < ms)

        valid_bos_bull = (not strict_structure) or (strict_structure and gann_trend[i] == 1)
        valid_bos_bear = (not strict_structure) or (strict_structure and gann_trend[i] == -1)

        if break_up:
            if struct_state[i-1] == -1:
                is_choch_bull[i] = True
                struct_state[i] = 1
            else:
                if valid_bos_bull:
                    is_bos_bull[i] = True
                    struct_state[i] = 1

        if break_dn:
            if struct_state[i-1] == 1:
                is_choch_bear[i] = True
                struct_state[i] = -1
            else:
                if valid_bos_bear:
                    is_bos_bear[i] = True
                    struct_state[i] = -1

    global_atr = calculate_atr(df, 14)
    body_size = (df["Close"] - df["Open"]).abs()
    is_significant = (~filter_doji) | (body_size > (global_atr * 0.5))

    nexus_fvgs = []
    if show_fvg and len(df) >= 3:
        for i in range(2, len(df)):
            if not bool(is_significant.iloc[i]):
                continue
            if df["Low"].iloc[i] > df["High"].iloc[i-2] and (df["Low"].iloc[i] - df["High"].iloc[i-2]) > 0:
                nexus_fvgs.append({
                    "x0": df.index[i-2], "x1": df.index[i],
                    "y0": float(df["High"].iloc[i-2]), "y1": float(df["Low"].iloc[i]),
                    "dir": "BULL"
                })
            if df["High"].iloc[i] < df["Low"].iloc[i-2] and (df["Low"].iloc[i-2] - df["High"].iloc[i]) > 0:
                nexus_fvgs.append({
                    "x0": df.index[i-2], "x1": df.index[i],
                    "y0": float(df["Low"].iloc[i-2]), "y1": float(df["High"].iloc[i]),
                    "dir": "BEAR"
                })

    is_full_bull = (ut_pos == 1) & (gann_trend == 1) & (kernel_trend == 1)
    is_full_bear = (ut_pos == -1) & (gann_trend == -1) & (kernel_trend == -1)

    last_signal = 0
    signal_buy = np.zeros(len(df), dtype=bool)
    signal_sell = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if is_full_bull[i] and last_signal != 1:
            signal_buy[i] = True
            last_signal = 1
        if is_full_bear[i] and last_signal != -1:
            signal_sell[i] = True
            last_signal = -1

    out = df.copy()
    out["Nexus_Kernel"] = kernel
    out["Nexus_KernelTrend"] = pd.Series(kernel_trend, index=df.index).astype(int)
    out["Nexus_GannActivator"] = pd.Series(gann_activator, index=df.index)
    out["Nexus_GannTrend"] = pd.Series(gann_trend, index=df.index).astype(int)
    out["Nexus_UT_Stop"] = pd.Series(ut_stop, index=df.index)
    out["Nexus_UT_Pos"] = pd.Series(ut_pos, index=df.index).astype(int)
    out["Nexus_StructState"] = pd.Series(struct_state, index=df.index).astype(int)

    out["Nexus_BOS_Bull"] = pd.Series(is_bos_bull, index=df.index)
    out["Nexus_BOS_Bear"] = pd.Series(is_bos_bear, index=df.index)
    out["Nexus_CHoCH_Bull"] = pd.Series(is_choch_bull, index=df.index)
    out["Nexus_CHoCH_Bear"] = pd.Series(is_choch_bear, index=df.index)

    out["Nexus_FullBull"] = pd.Series(is_full_bull, index=df.index)
    out["Nexus_FullBear"] = pd.Series(is_full_bear, index=df.index)

    out["Nexus_Signal_Buy"] = pd.Series(signal_buy, index=df.index)
    out["Nexus_Signal_Sell"] = pd.Series(signal_sell, index=df.index)

    return out, nexus_fvgs

# ==========================================
# 3C. PINE INDICATOR #3: APEX VECTOR v4.1 (FULL PORT)
# ==========================================
def compute_apex_vector(df: pd.DataFrame, cfg: dict):
    """
    Full port:
    - Efficiency (body/range) smoothed by EMA(len_vec)
    - Volume normalization (vol / SMA(vol, vol_norm)) if enabled
    - Vector raw: sign(close-open) * efficiency * vol_factor
    - Smoothing kernel: EMA/SMA/RMA/WMA/VWMA
    - Thresholds (super/resist) * strictness
    - State flags: super bull/bear, resistive, heat
    - Divergence engine (reg + hidden, bull/bear) using pivots on flux
    Outputs:
      Columns:
        Vector_Eff, Vector_VolFact, Vector_Flux,
        Vector_SuperBull, Vector_SuperBear, Vector_Resistive, Vector_Heat,
        Vector_Div_Bull_Reg, Vector_Div_Bull_Hid, Vector_Div_Bear_Reg, Vector_Div_Bear_Hid
    """
    eff_super = float(cfg["eff_super"])
    eff_resist = float(cfg["eff_resist"])
    vol_norm = int(cfg["vol_norm"])
    len_vec = int(cfg["len_vec"])
    sm_type = cfg["sm_type"]
    len_sm = int(cfg["len_sm"])
    use_vol = bool(cfg["use_vol"])
    strictness = float(cfg["strictness"])
    show_div = bool(cfg["show_div"])
    div_look = int(cfg["div_look"])
    show_reg = bool(cfg["show_reg"])
    show_hid = bool(cfg["show_hid"])

    range_abs = (df["High"] - df["Low"]).astype(float)
    body_abs = (df["Close"] - df["Open"]).abs().astype(float)
    raw_eff = np.where(range_abs.to_numpy() == 0, 0.0, (body_abs / range_abs).to_numpy())
    eff = pd.Series(raw_eff, index=df.index).ewm(span=len_vec, adjust=False).mean()

    vol_avg = df["Volume"].rolling(vol_norm).mean()
    vol_fact = pd.Series(1.0, index=df.index)
    if use_vol:
        denom = vol_avg.replace(0, np.nan)
        vol_fact = (df["Volume"] / denom).fillna(1.0)

    direction = np.sign((df["Close"] - df["Open"]).to_numpy())
    vector_raw = pd.Series(direction, index=df.index) * eff * vol_fact

    sm_type_u = (sm_type or "EMA").upper()
    if sm_type_u == "EMA":
        flux = vector_raw.ewm(span=len_sm, adjust=False).mean()
    elif sm_type_u == "SMA":
        flux = vector_raw.rolling(len_sm).mean()
    elif sm_type_u == "RMA":
        flux = calculate_rma(vector_raw, len_sm)
    elif sm_type_u == "WMA":
        flux = calculate_wma(vector_raw, len_sm)
    elif sm_type_u == "VWMA":
        flux = calculate_vwma(vector_raw, df["Volume"], len_sm)
    else:
        flux = vector_raw.ewm(span=len_sm, adjust=False).mean()

    th_super = eff_super * strictness
    th_resist = eff_resist * strictness

    is_super_bull = flux > th_super
    is_super_bear = flux < -th_super
    is_resistive = flux.abs() < th_resist
    is_heat = (~is_super_bull) & (~is_super_bear) & (~is_resistive)

    ph = pivothigh(flux, div_look, div_look)
    pl = pivotlow(flux, div_look, div_look)

    prev_pl_flux = np.nan
    prev_pl_price = np.nan
    prev_ph_flux = np.nan
    prev_ph_price = np.nan

    div_bull_reg = np.zeros(len(df), dtype=bool)
    div_bull_hid = np.zeros(len(df), dtype=bool)
    div_bear_reg = np.zeros(len(df), dtype=bool)
    div_bear_hid = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if not np.isnan(pl.iloc[i]):
            price_at_pivot = float(df["Low"].iloc[i])
            if show_div and np.isfinite(prev_pl_flux):
                if (price_at_pivot < prev_pl_price) and (pl.iloc[i] > prev_pl_flux):
                    div_bull_reg[i] = show_reg
                if (price_at_pivot > prev_pl_price) and (pl.iloc[i] < prev_pl_flux):
                    div_bull_hid[i] = show_hid
            prev_pl_flux = float(pl.iloc[i])
            prev_pl_price = price_at_pivot

        if not np.isnan(ph.iloc[i]):
            price_at_pivot = float(df["High"].iloc[i])
            if show_div and np.isfinite(prev_ph_flux):
                if (price_at_pivot > prev_ph_price) and (ph.iloc[i] < prev_ph_flux):
                    div_bear_reg[i] = show_reg
                if (price_at_pivot < prev_ph_price) and (ph.iloc[i] > prev_ph_flux):
                    div_bear_hid[i] = show_hid
            prev_ph_flux = float(ph.iloc[i])
            prev_ph_price = price_at_pivot

    out = df.copy()
    out["Vector_Eff"] = eff
    out["Vector_VolFact"] = vol_fact
    out["Vector_Flux"] = flux
    out["Vector_Th_Super"] = th_super
    out["Vector_Th_Resist"] = th_resist
    out["Vector_SuperBull"] = is_super_bull
    out["Vector_SuperBear"] = is_super_bear
    out["Vector_Resistive"] = is_resistive
    out["Vector_Heat"] = is_heat

    out["Vector_Div_Bull_Reg"] = pd.Series(div_bull_reg, index=df.index)
    out["Vector_Div_Bull_Hid"] = pd.Series(div_bull_hid, index=df.index)
    out["Vector_Div_Bear_Reg"] = pd.Series(div_bear_reg, index=df.index)
    out["Vector_Div_Bear_Hid"] = pd.Series(div_bear_hid, index=df.index)

    return out

# ==========================================
# 4. EXISTING INDICATORS + FULL INTEGRATION (NO REMOVALS)
# ==========================================
def calculate_linreg_mom(series, length=20):
    x = np.arange(length)
    return series.rolling(length).apply(lambda y: linregress(x, y)[0], raw=True)

def calc_indicators(df: pd.DataFrame, apex_cfg: dict, nexus_cfg: dict, vector_cfg: dict):
    """Calculates Base Indicators + existing GOD MODE + FULL Pine Ports (Apex Master, Nexus, Apex Vector)."""

    df = df.copy()

    # --- 0. Base Calcs (existing) ---
    df['HMA'] = calculate_hma(df['Close'], 55)
    df['ATR'] = calculate_atr(df, 14)
    df['Pivot_Resist'] = df['High'].rolling(20).max()
    df['Pivot_Support'] = df['Low'].rolling(20).min()
    df['MFI'] = (df['Close'].diff() * df['Volume']).rolling(14).mean()

    # --- 1) Apex Trend & Liquidity Master (FULL) ---
    df, apex_supply_zones, apex_demand_zones = compute_apex_trend_liquidity(df, apex_cfg)
    df["Apex_Supply_Zones_Count"] = len(apex_supply_zones)
    df["Apex_Demand_Zones_Count"] = len(apex_demand_zones)

    df["ApexMaster_Upper"] = df["Apex_Upper"]
    df["ApexMaster_Lower"] = df["Apex_Lower"]
    df["ApexMaster_Trend"] = df["Apex_Trend"]
    df["ApexMaster_Sig_Buy"] = df["Apex_Sig_Buy"]
    df["ApexMaster_Sig_Sell"] = df["Apex_Sig_Sell"]

    # --- 2) DarkPool Squeeze Momentum (existing) ---
    df['Sqz_Basis'] = df['Close'].rolling(20).mean()
    df['Sqz_Dev'] = df['Close'].rolling(20).std() * 2.0
    df['Sqz_Upper_BB'] = df['Sqz_Basis'] + df['Sqz_Dev']
    df['Sqz_Lower_BB'] = df['Sqz_Basis'] - df['Sqz_Dev']

    df['Sqz_Ma_KC'] = df['Close'].rolling(20).mean()
    df['Sqz_Range_MA'] = calculate_atr(df, 20)
    df['Sqz_Upper_KC'] = df['Sqz_Ma_KC'] + (df['Sqz_Range_MA'] * 1.5)
    df['Sqz_Lower_KC'] = df['Sqz_Ma_KC'] - (df['Sqz_Range_MA'] * 1.5)

    df['Squeeze_On'] = (df['Sqz_Lower_BB'] > df['Sqz_Lower_KC']) & (df['Sqz_Upper_BB'] < df['Sqz_Upper_KC'])

    highest = df['High'].rolling(20).max()
    lowest = df['Low'].rolling(20).min()
    avg_val = (highest + lowest + df['Sqz_Ma_KC']) / 3
    df['Sqz_Mom'] = (df['Close'] - avg_val).rolling(20).mean() * 100

    # --- 3) Money Flow Matrix (existing) ---
    up = df['Close'].diff().clip(lower=0)
    down = df['Close'].diff().clip(upper=0).abs()
    rs_raw = (up.rolling(14).mean() / down.rolling(14).mean().replace(0, np.nan))
    rsi_src = (100 - (100 / (1 + rs_raw))).fillna(50) - 50
    mf_vol = (df['Volume'] / df['Volume'].rolling(14).mean().replace(0, np.nan)).fillna(1.0)
    df['MF_Matrix'] = (rsi_src * mf_vol).ewm(span=3).mean()

    # --- 4) Dark Vector Scalping (existing) ---
    amp = 5
    df['VS_Low'] = df['Low'].rolling(amp).min()
    df['VS_High'] = df['High'].rolling(amp).max()
    df['VS_Trend'] = np.where(df['Close'] > df['VS_High'].shift(1), 1,
                              np.where(df['Close'] < df['VS_Low'].shift(1), -1, 0))
    df['VS_Trend'] = pd.Series(df['VS_Trend'], index=df.index).replace(to_replace=0, method='ffill').fillna(0).astype(int)

    # --- 5) Advanced Volume (existing) ---
    df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan)
    df['RVOL'] = df['RVOL'].fillna(1.0)

    # --- 6) EVWM (existing) ---
    ev_len = 21
    ev_base = calculate_hma(df['Close'], ev_len)
    ev_atr = calculate_atr(df, ev_len).replace(0, np.nan)
    ev_elast = (df['Close'] - ev_base) / ev_atr
    ev_force = np.sqrt(df['RVOL'].ewm(span=5).mean())
    df['EVWM'] = (ev_elast * ev_force).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- 8) Gann High Low Activator (existing simple) ---
    gann_len_simple = 3
    df['Gann_High'] = df['High'].rolling(gann_len_simple).mean()
    df['Gann_Low'] = df['Low'].rolling(gann_len_simple).mean()
    df['Gann_Trend'] = np.where(df['Close'] > df['Gann_High'].shift(1), 1,
                                np.where(df['Close'] < df['Gann_Low'].shift(1), -1, 0))
    df['Gann_Trend'] = pd.Series(df['Gann_Trend'], index=df.index).replace(to_replace=0, method='ffill').fillna(0).astype(int)

    # --- 9) Dark Vector (SuperTrend) (existing) ---
    st_val, st_dir = calculate_supertrend(df, 10, 4.0)
    df['DarkVector_Trend'] = st_dir.fillna(0).astype(int)

    # --- 10) Wyckoff VSA Trend Shield (existing) ---
    df['Trend_Shield_Bull'] = df['Close'] > df['Close'].rolling(200).mean()

    # --- 2) Nexus v8.2 (FULL) ---
    df, nexus_fvgs = compute_nexus(df, nexus_cfg)
    df["Nexus_FVG_Count"] = len(nexus_fvgs)

    # --- 3) Apex Vector v4.1 (FULL) ---
    df = compute_apex_vector(df, vector_cfg)

    # --- GOD MODE CONFLUENCE SIGNAL (existing GM_Score preserved) ---
    df['GM_Score'] = (
        df['Apex_Trend'].astype(int) +
        df['Gann_Trend'].astype(int) +
        df['DarkVector_Trend'].astype(int) +
        df['VS_Trend'].astype(int) +
        np.sign(df['Sqz_Mom']).fillna(0).astype(int)
    )

    # --- NEW: OMNI SCORE (adds Nexus + Vector without altering GM_Score) ---
    vector_bias = np.where(df["Vector_SuperBull"], 1, np.where(df["Vector_SuperBear"], -1, 0))
    nexus_bias = np.where(df["Nexus_FullBull"], 1, np.where(df["Nexus_FullBear"], -1, 0))
    struct_bias = np.where(df["Nexus_BOS_Bull"] | df["Nexus_CHoCH_Bull"], 1,
                           np.where(df["Nexus_BOS_Bear"] | df["Nexus_CHoCH_Bear"], -1, 0))

    df["Omni_Score"] = df["GM_Score"].astype(int) + vector_bias + nexus_bias + struct_bias
    df["Omni_Confidence"] = (50 + (df["Omni_Score"] * 10)).clip(0, 100).astype(int)

    # --- DASHBOARD V2 SPECIFIC CALCULATIONS (existing) ---
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']

    low_min = df['Low'].rolling(14).min()
    high_max = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    df['ROC'] = df['Close'].pct_change(14) * 100
    df['EMA_Fast'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr14 = df['ATR'].replace(0, np.nan)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / tr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / tr14)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
    df['ADX'] = dx.rolling(14).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = (100 - (100 / (1 + rs))).fillna(50)

    rsi_norm = (df['RSI'] - 50) * 2
    macd_norm = np.where(df['Hist'] > 0, np.minimum(df['Hist'] * 10, 100), np.maximum(df['Hist'] * 10, -100))
    stoch_norm = (df['Stoch_K'] - 50) * 2
    roc_norm = np.where(df['ROC'] > 0, np.minimum(df['ROC'] * 10, 100), np.maximum(df['ROC'] * 10, -100))

    df['Mom_Score'] = np.round((rsi_norm + macd_norm + stoch_norm + roc_norm) / 4)

    return df, apex_supply_zones, apex_demand_zones, nexus_fvgs

def calc_fear_greed_v4(df):
    """
    🔥 DarkPool's Fear & Greed v4 Port
    Calculates composite sentiment index, FOMO, and Panic states.
    """
    df = df.copy()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['FG_RSI'] = (100 - (100 / (1 + rs))).fillna(50)

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    df['FG_MACD'] = (50 + (hist * 10)).clip(0, 100)

    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper = sma20 + (std20 * 2)
    lower = sma20 - (std20 * 2)
    denom = (upper - lower).replace(0, np.nan)
    df['FG_BB'] = ((df['Close'] - lower) / denom * 100).clip(0, 100).fillna(50)

    sma50 = df['Close'].rolling(50).mean()
    sma200 = df['Close'].rolling(200).mean()
    conditions = [
        (df['Close'] > sma50) & (sma50 > sma200),
        (df['Close'] > sma50),
        (df['Close'] < sma50) & (sma50 < sma200)
    ]
    choices = [75, 60, 25]
    df['FG_MA'] = np.select(conditions, choices, default=40)

    df['FG_Raw'] = (df['FG_RSI'] * 0.30) + (df['FG_MACD'] * 0.25) + (df['FG_BB'] * 0.25) + (df['FG_MA'] * 0.20)
    df['FG_Index'] = df['FG_Raw'].rolling(5).mean().fillna(df['FG_Raw'])

    vol_ma = df['Volume'].rolling(20).mean()
    high_vol = df['Volume'] > (vol_ma * 2.5)
    high_rsi = df['FG_RSI'] > 70
    momentum = df['Close'] > df['Close'].shift(3) * 1.02
    above_bb = df['Close'] > (upper * 1.0)
    df['IS_FOMO'] = high_vol & high_rsi & momentum & above_bb

    daily_drop = df['Close'].pct_change() * 100
    sharp_drop = daily_drop < -3.0
    panic_vol = df['Volume'] > (vol_ma * 3.0)
    low_rsi = df['FG_RSI'] < 30
    df['IS_PANIC'] = sharp_drop & panic_vol & (low_rsi | (daily_drop < -5.0))

    return df

def run_monte_carlo(df, days=30, simulations=1000):
    last_price = df['Close'].iloc[-1]
    returns = df['Close'].pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()

    daily_returns_sim = np.random.normal(mu, sigma, (days, simulations))
    price_paths = np.zeros((days, simulations))
    price_paths[0] = last_price
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * (1 + daily_returns_sim[t])
    return price_paths

def calc_volume_profile(df, bins=50):
    price_min = df['Low'].min()
    price_max = df['High'].max()
    price_bins = np.linspace(price_min, price_max, bins)

    df = df.copy()
    df['Mid'] = (df['Close'] + df['Open']) / 2
    df['Bin'] = pd.cut(df['Mid'], bins=price_bins, labels=price_bins[:-1], include_lowest=True)

    vp = df.groupby('Bin')['Volume'].sum().reset_index()
    vp['Price'] = vp['Bin'].astype(float)
    poc_idx = vp['Volume'].idxmax()
    poc_price = vp.loc[poc_idx, 'Price']
    return vp, poc_price

def get_sr_channels(df, pivot_period=10, loopback=290, max_width_pct=5, min_strength=1):
    if len(df) < loopback:
        loopback = len(df)
    window = df.iloc[-loopback:].copy()

    window['Is_Pivot_H'] = window['High'] == window['High'].rolling(pivot_period*2+1, center=True).max()
    window['Is_Pivot_L'] = window['Low'] == window['Low'].rolling(pivot_period*2+1, center=True).min()

    pivot_vals = []
    pivot_vals.extend(window[window['Is_Pivot_H']]['High'].tolist())
    pivot_vals.extend(window[window['Is_Pivot_L']]['Low'].tolist())

    if not pivot_vals:
        return []
    pivot_vals.sort()

    price_range = window['High'].max() - window['Low'].min()
    max_width = price_range * (max_width_pct / 100)

    potential_zones = []
    for i in range(len(pivot_vals)):
        seed = pivot_vals[i]
        cluster_min = seed
        cluster_max = seed
        pivot_count = 1

        for j in range(i + 1, len(pivot_vals)):
            curr = pivot_vals[j]
            if (curr - seed) <= max_width:
                cluster_max = curr
                pivot_count += 1
            else:
                break

        touches = ((window['High'] >= cluster_min) & (window['Low'] <= cluster_max)).sum()
        score = (pivot_count * 20) + touches
        potential_zones.append({'min': cluster_min, 'max': cluster_max, 'score': score})

    potential_zones.sort(key=lambda x: x['score'], reverse=True)

    final_zones = []
    for zone in potential_zones:
        is_overlapping = False
        for existing in final_zones:
            if (zone['min'] < existing['max']) and (zone['max'] > existing['min']):
                is_overlapping = True
                break
        if not is_overlapping:
            final_zones.append(zone)
            if len(final_zones) >= 6:
                break

    return final_zones

def calculate_smc(df, swing_length=5):
    smc_data = {'structures': [], 'order_blocks': [], 'fvgs': []}

    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            smc_data['fvgs'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i], 'color': 'rgba(0, 255, 104, 0.3)'})
        if df['High'].iloc[i] < df['Low'].iloc[i-2]:
            smc_data['fvgs'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['Low'].iloc[i-2], 'y1': df['High'].iloc[i], 'color': 'rgba(255, 0, 8, 0.3)'})

    df = df.copy()
    df['Pivot_High'] = df['High'].rolling(window=swing_length*2+1, center=True).max() == df['High']
    df['Pivot_Low'] = df['Low'].rolling(window=swing_length*2+1, center=True).min() == df['Low']

    last_high = None
    last_low = None
    trend = 0

    for i in range(swing_length, len(df)):
        curr_idx = df.index[i]
        curr_close = df['Close'].iloc[i]

        if df['Pivot_High'].iloc[i-swing_length]:
            last_high = {'price': df['High'].iloc[i-swing_length], 'idx': df.index[i-swing_length], 'i': i-swing_length}
        if df['Pivot_Low'].iloc[i-swing_length]:
            last_low = {'price': df['Low'].iloc[i-swing_length], 'idx': df.index[i-swing_length], 'i': i-swing_length}

        if last_high and curr_close > last_high['price']:
            label = "CHoCH" if trend != 1 else "BOS"
            trend = 1
            smc_data['structures'].append({'x0': last_high['idx'], 'x1': curr_idx, 'y': last_high['price'], 'color': 'green', 'label': label})
            if last_low:
                subset = df.iloc[last_low['i']:i]
                if not subset.empty:
                    ob_idx = subset['Low'].idxmin()
                    ob_row = df.loc[ob_idx]
                    smc_data['order_blocks'].append({'x0': ob_idx, 'x1': df.index[-1], 'y0': ob_row['Low'], 'y1': ob_row['High'], 'color': 'rgba(33, 87, 243, 0.4)'})
            last_high = None

        elif last_low and curr_close < last_low['price']:
            label = "CHoCH" if trend != -1 else "BOS"
            trend = -1
            smc_data['structures'].append({'x0': last_low['idx'], 'x1': curr_idx, 'y': last_low['price'], 'color': 'red', 'label': label})
            if last_high:
                subset = df.iloc[last_high['i']:i]
                if not subset.empty:
                    ob_idx = subset['High'].idxmax()
                    ob_row = df.loc[ob_idx]
                    smc_data['order_blocks'].append({'x0': ob_idx, 'x1': df.index[-1], 'y0': ob_row['Low'], 'y1': ob_row['High'], 'color': 'rgba(255, 0, 0, 0.4)'})
            last_low = None

    return smc_data

def calc_correlations(ticker, lookback_days=180):
    macro_tickers = {
        "S&P 500": "SPY", "Bitcoin": "BTC-USD", "10Y Yield": "^TNX",
        "Dollar (DXY)": "DX-Y.NYB", "Gold": "GC=F", "Oil": "CL=F"
    }

    df_main = yf.download(ticker, period="1y", interval="1d", progress=False)['Close']
    df_macro = yf.download(list(macro_tickers.values()), period="1y", interval="1d", progress=False)['Close']

    combined = df_macro.copy()
    combined[ticker] = df_main
    corr_matrix = combined.iloc[-lookback_days:].corr()
    target_corr = corr_matrix[ticker].drop(ticker).sort_values(ascending=False)

    inv_map = {v: k for k, v in macro_tickers.items()}
    target_corr.index = [inv_map.get(x, x) for x in target_corr.index]

    return target_corr

def calc_mtf_trend(ticker):
    timeframes = {"1H": "1h", "4H": "1h", "Daily": "1d", "Weekly": "1wk"}
    trends = {}

    for tf_name, tf_code in timeframes.items():
        try:
            period = "1y" if tf_name in ["1H", "4H"] else "2y"
            df = yf.download(ticker, period=period, interval=tf_code, progress=False)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 50:
                trends[tf_name] = {"Trend": "N/A", "RSI": "N/A", "EMA Spread": "N/A"}
                continue

            if tf_name == "4H":
                df = df.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

            df['EMA20'] = df['Close'].ewm(span=20).mean()
            df['EMA50'] = df['Close'].ewm(span=50).mean()

            rsi = calculate_rsi(df["Close"], 14)

            last = df.iloc[-1]
            last_rsi = float(rsi.iloc[-1])

            trend = "BULLISH" if last['Close'] > last['EMA20'] and last['EMA20'] > last['EMA50'] else \
                    "BEARISH" if last['Close'] < last['EMA20'] and last['EMA20'] < last['EMA50'] else "NEUTRAL"

            trends[tf_name] = {
                "Trend": trend,
                "RSI": f"{last_rsi:.1f}",
                "EMA Spread": f"{(last['EMA20'] - last['EMA50']):.2f}"
            }
        except:
            trends[tf_name] = {"Trend": "N/A", "RSI": "N/A", "EMA Spread": "N/A"}

    return pd.DataFrame(trends).T

def calc_intraday_dna(ticker):
    try:
        df = yf.download(ticker, period="60d", interval="1h", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None
        df['Return'] = df['Close'].pct_change() * 100
        df['Hour'] = df.index.hour
        hourly_stats = df.groupby('Hour')['Return'].agg(['mean', 'sum', 'count', lambda x: (x > 0).mean() * 100])
        hourly_stats.columns = ['Avg Return', 'Total Return', 'Count', 'Win Rate']
        return hourly_stats
    except:
        return None

@st.cache_data(ttl=3600)
def get_seasonality_stats(ticker):
    try:
        df = yf.download(ticker, period="20y", interval="1mo", progress=False)
        if df.empty or len(df) < 12:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            else:
                return None
        df = df.dropna()
        df['Return'] = df['Close'].pct_change() * 100
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        heatmap_data = df.pivot_table(index='Year', columns='Month', values='Return')

        periods = [1, 3, 6, 12]
        hold_stats = {}
        for p in periods:
            rolling_ret = df['Close'].pct_change(periods=p) * 100
            rolling_ret = rolling_ret.dropna()
            win_count = (rolling_ret > 0).sum()
            total_count = len(rolling_ret)
            win_rate = (win_count / total_count * 100) if total_count > 0 else 0
            avg_ret = rolling_ret.mean()
            hold_stats[p] = {"Win Rate": win_rate, "Avg Return": avg_ret}

        month_stats = df.groupby('Month')['Return'].agg(['mean', lambda x: (x > 0).mean() * 100, 'count'])
        month_stats.columns = ['Avg Return', 'Win Rate', 'Count']

        return heatmap_data, hold_stats, month_stats
    except:
        return None

def calc_day_of_week_dna(ticker, lookback, calc_mode):
    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.iloc[-lookback:].copy()
        if calc_mode == "Close to Close (Total)":
            df['Day_Return'] = df['Close'].pct_change() * 100
        else:
            df['Day_Return'] = ((df['Close'] - df['Open']) / df['Open']) * 100

        df = df.dropna()
        df['Day_Name'] = df.index.day_name()

        pivot_ret = df.pivot(columns='Day_Name', values='Day_Return').fillna(0)
        cum_ret = pivot_ret.cumsum()

        stats = df.groupby('Day_Name')['Day_Return'].agg(['count', 'sum', 'mean', lambda x: (x > 0).mean() * 100])
        stats.columns = ['Count', 'Total Return', 'Avg Return', 'Win Rate']

        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        stats = stats.reindex([d for d in days_order if d in stats.index])

        return cum_ret, stats
    except:
        return None

# ==========================================
# 5. AI ANALYST (PRESERVED)
# ==========================================
def ask_ai_analyst(df, ticker, fundamentals, balance, risk_pct, timeframe):
    if not st.session_state.api_key:
        return "⚠️ Waiting for OpenAI API Key in the sidebar..."

    last = df.iloc[-1]
    trend = "BULLISH" if last['Close'] > last['HMA'] else "BEARISH"
    risk_dollars = balance * (risk_pct / 100)

    gm_score = last['GM_Score']
    gm_verdict = "STRONG BUY" if gm_score >= 3 else "STRONG SELL" if gm_score <= -3 else "NEUTRAL"

    if trend == "BULLISH":
        stop_level = last['Pivot_Support']
        direction = "LONG"
    else:
        stop_level = last['Pivot_Resist']
        direction = "SHORT"

    if pd.isna(stop_level) or abs(last['Close'] - stop_level) < (last['ATR']*0.5):
        stop_level = last['Close'] - (last['ATR']*2) if direction == "LONG" else last['Close'] + (last['ATR']*2)

    dist = abs(last['Close'] - stop_level)
    if dist == 0:
        dist = last['ATR']
    shares = risk_dollars / dist

    fund_text = "N/A"
    if fundamentals:
        fund_text = f"P/E: {fundamentals.get('P/E Ratio', 'N/A')}. Growth: {fundamentals.get('Rev Growth', 0)*100:.1f}%."

    fg_val = last.get('FG_Index', np.nan)
    fg_state = "N/A"
    if np.isfinite(fg_val):
        fg_state = "EXTREME GREED" if fg_val >= 80 else "GREED" if fg_val >= 60 else "NEUTRAL" if fg_val >= 40 else "FEAR" if fg_val >= 20 else "EXTREME FEAR"

    psych_alert = ""
    if bool(last.get('IS_FOMO', False)):
        psych_alert = "WARNING: ALGORITHMIC FOMO DETECTED."
    if bool(last.get('IS_PANIC', False)):
        psych_alert = "WARNING: PANIC SELLING DETECTED."

    prompt = f"""
    Act as a Senior Market Analyst. Analyze {ticker} on the **{timeframe} timeframe** at price ${last['Close']:.2f}.

    --- DATA FEED ---
    Technicals: Trend is {trend}. Volatility (ATR) is {last['ATR']:.2f}.
    RSI: {last['RSI']:.1f}.
    Volume (RVOL): {last['RVOL']:.1f}x.
    Titan Score: {gm_score} ({gm_verdict}).
    Momentum: {'Rising' if last['Sqz_Mom'] > 0 else 'Falling'}.
    Sentiment: {fg_state} ({fg_val if np.isfinite(fg_val) else 0:.1f}/100).
    {psych_alert}
    Fundamentals: {fund_text}

    --- MISSION ---
    Provide a concise, high-level overview of what is happening with this asset.
    1. Analyze the current market structure (Trend vs Chop).
    2. Explain the correlation between the technicals and sentiment.
    3. Provide a general outlook on potential direction.

    IMPORTANT:
    - Do NOT provide specific Entry prices, Exit prices, or Stop Loss numbers.
    - Do NOT give specific financial advice.
    - Keep it to a market situation overview only.
    - USE EMOJIS liberally (🚀, 📉, 🐂, 🐻, 🧠, ⚠️).
    """

    try:
        client = OpenAI(api_key=st.session_state.api_key)
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], max_tokens=2500)
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error: {e}"

# ==========================================
# 6. TELEGRAM SIGNALS (UPGRADED — MULTI-TYPE, DETAILED, NO REMOVALS)
# ==========================================
def _safe_float(x):
    try:
        return float(x)
    except:
        return float("nan")

def build_trade_levels(last_row: pd.Series, direction: str):
    """
    Deterministic levels derived from computed indicators only.
    - Entry: last close
    - Stop: prefer Nexus_UT_Stop if present and finite, else ATR-based
    - Targets: R-multiples (1R, 2R, 3R)
    """
    entry = _safe_float(last_row.get("Close", np.nan))
    atr = _safe_float(last_row.get("ATR", np.nan))
    ut = _safe_float(last_row.get("Nexus_UT_Stop", np.nan))
    if np.isfinite(ut):
        stop = ut
    else:
        if np.isfinite(entry) and np.isfinite(atr):
            stop = entry - (2 * atr) if direction == "LONG" else entry + (2 * atr)
        else:
            stop = float("nan")

    r = abs(entry - stop) if (np.isfinite(entry) and np.isfinite(stop) and entry != stop) else (atr if np.isfinite(atr) else float("nan"))
    tp1 = entry + (1 * r) if direction == "LONG" else entry - (1 * r)
    tp2 = entry + (2 * r) if direction == "LONG" else entry - (2 * r)
    tp3 = entry + (3 * r) if direction == "LONG" else entry - (3 * r)

    return {"entry": entry, "stop": stop, "r": r, "tp1": tp1, "tp2": tp2, "tp3": tp3}

def build_position_sizing(balance: float, risk_pct: float, levels: dict):
    risk_dollars = balance * (risk_pct / 100.0)
    r = levels.get("r", float("nan"))
    if not np.isfinite(r) or r <= 0:
        return {"risk_usd": risk_dollars, "size_units": float("nan"), "note": "Size unavailable (invalid risk distance)."}
    size = risk_dollars / r
    return {"risk_usd": risk_dollars, "size_units": size, "note": ""}

def format_signal_report(
    ticker: str,
    interval: str,
    df: pd.DataFrame,
    last_row: pd.Series,
    macro_text: str,
    ai_verdict: str,
    balance: float,
    risk_pct: float,
    report_type: str
):
    """
    Multi-type Telegram reports:
      - QUICK_PING
      - TRADE_SCALP
      - TRADE_SWING
      - FULL_ANALYSIS
    """
    ts = df.index[-1]
    ts_txt = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)

    apex_bias = "🐂 BULL" if int(last_row.get("Apex_Trend", 0)) == 1 else "🐻 BEAR" if int(last_row.get("Apex_Trend", 0)) == -1 else "⚪ CHOP"
    nexus_bias = "BUY" if bool(last_row.get("Nexus_Signal_Buy", False)) else "SELL" if bool(last_row.get("Nexus_Signal_Sell", False)) else "WAIT"
    vector_state = "SUPER BULL" if bool(last_row.get("Vector_SuperBull", False)) else \
                   "SUPER BEAR" if bool(last_row.get("Vector_SuperBear", False)) else \
                   "RESISTIVE" if bool(last_row.get("Vector_Resistive", False)) else "HEAT"
    squeeze = "💥 ON" if bool(last_row.get("Squeeze_On", False)) else "💤 OFF"
    rvol = _safe_float(last_row.get("RVOL", np.nan))
    rsi = _safe_float(last_row.get("RSI", np.nan))
    price = _safe_float(last_row.get("Close", np.nan))
    gm_score = int(last_row.get("GM_Score", 0)) if np.isfinite(_safe_float(last_row.get("GM_Score", 0))) else 0
    omni_score = int(last_row.get("Omni_Score", 0)) if np.isfinite(_safe_float(last_row.get("Omni_Score", 0))) else 0
    conf = int(last_row.get("Omni_Confidence", 50)) if np.isfinite(_safe_float(last_row.get("Omni_Confidence", 50))) else 50

    direction = "LONG" if omni_score > 0 else "SHORT" if omni_score < 0 else "NEUTRAL"
    levels = build_trade_levels(last_row, "LONG" if direction == "LONG" else "SHORT")
    sizing = build_position_sizing(balance, risk_pct, levels)

    struct = []
    if bool(last_row.get("Nexus_BOS_Bull", False)): struct.append("🟢 BOS(BULL)")
    if bool(last_row.get("Nexus_BOS_Bear", False)): struct.append("🔴 BOS(BEAR)")
    if bool(last_row.get("Nexus_CHoCH_Bull", False)): struct.append("🟡 CHoCH(BULL)")
    if bool(last_row.get("Nexus_CHoCH_Bear", False)): struct.append("🟡 CHoCH(BEAR)")
    struct_txt = " | ".join(struct) if struct else "None"

    div = []
    if bool(last_row.get("Vector_Div_Bull_Reg", False)): div.append("🔵 Bull Reg")
    if bool(last_row.get("Vector_Div_Bull_Hid", False)): div.append("🔵 Bull Hid")
    if bool(last_row.get("Vector_Div_Bear_Reg", False)): div.append("🩷 Bear Reg")
    if bool(last_row.get("Vector_Div_Bear_Hid", False)): div.append("🩷 Bear Hid")
    div_txt = " | ".join(div) if div else "None"

    if report_type == "QUICK_PING":
        msg = (
            f"🔥 {ticker} ({interval}) — TITAN PING\n"
            f"🕒 {ts_txt}\n\n"
            f"Price: ${price:.2f}\n"
            f"🧬 GM Score: {gm_score}/5 | 🧠 Omni: {omni_score} | 🎯 Conf: {conf}/100\n\n"
            f"Apex: {apex_bias}\n"
            f"Nexus: {nexus_bias}\n"
            f"Vector: {vector_state}\n"
            f"Squeeze: {squeeze}\n"
            f"📊 RSI: {rsi:.1f} | 🔋 RVOL: {rvol:.1f}x\n\n"
            f"{macro_text}\n"
            f"#DarkPool #Titan"
        )
        return [msg]

    if report_type in ["TRADE_SCALP", "TRADE_SWING"]:
        mode_tag = "⚡ SCALP SETUP" if report_type == "TRADE_SCALP" else "📈 SWING SETUP"
        risk_line = _safe_float(last_row.get("Nexus_UT_Stop", np.nan))
        risk_line_txt = f"{risk_line:.2f}" if np.isfinite(risk_line) else "N/A"

        # FIXED: avoid nested f-string quoting bug (engine correctness)
        if np.isfinite(risk_line):
            stop_ref_txt = f"Nexus UT {risk_line_txt}"
        else:
            stop_ref_txt = f"ATR Ref {levels['stop']:.2f}" if np.isfinite(levels.get("stop", np.nan)) else "N/A"

        msg = (
            f"{mode_tag} — {ticker} ({interval})\n"
            f"🕒 {ts_txt}\n\n"
            f"Bias: {direction} | 🎯 Conf: {conf}/100\n"
            f"Apex: {apex_bias} | Nexus: {nexus_bias} | Vector: {vector_state}\n"
            f"Structure: {struct_txt}\n"
            f"Divergence: {div_txt}\n"
            f"Squeeze: {squeeze}\n\n"
            f"📍 Entry(ref): ${levels['entry']:.2f}\n"
            f"🛡️ Stop(ref): {stop_ref_txt}\n"
            f"🎯 Targets(ref): TP1 {levels['tp1']:.2f} | TP2 {levels['tp2']:.2f} | TP3 {levels['tp3']:.2f}\n\n"
            f"💼 Risk Model:\n"
            f"- Capital: ${balance:,.0f}\n"
            f"- Risk: {risk_pct:.2f}% = ${sizing['risk_usd']:.2f}\n"
            f"- Position Size(ref units): {sizing['size_units']:.4f}\n\n"
            f"📊 RSI: {rsi:.1f} | 🔋 RVOL: {rvol:.1f}x\n"
            f"{macro_text}\n\n"
            f"⚠️ Not financial advice. Levels are indicator-derived references.\n"
            f"#Trading #DarkPool #Titan"
        )
        return [msg]

    msg = (
        f"🧠 FULL TITAN ANALYSIS — {ticker} ({interval})\n"
        f"🕒 {ts_txt}\n\n"
        f"1) REGIME & BIAS\n"
        f"- Price: ${price:.2f}\n"
        f"- GM Score: {gm_score}/5 | Omni Score: {omni_score} | Confidence: {conf}/100\n"
        f"- Direction Bias: {direction}\n\n"
        f"2) TREND STACK\n"
        f"- Apex Trend & Liquidity: {apex_bias} | Signals: "
        f"{'BUY' if bool(last_row.get('Apex_Sig_Buy', False)) else ('SELL' if bool(last_row.get('Apex_Sig_Sell', False)) else 'None')}\n"
        f"- Nexus Omni: {nexus_bias} | KernelTrend: {int(last_row.get('Nexus_KernelTrend', 0))} | "
        f"GannTrend: {int(last_row.get('Nexus_GannTrend', 0))} | UT Pos: {int(last_row.get('Nexus_UT_Pos', 0))}\n"
        f"- SuperTrend (DarkVector_Trend): {int(last_row.get('DarkVector_Trend', 0))}\n\n"
        f"3) MOMENTUM & FLOW\n"
        f"- Apex Vector State: {vector_state}\n"
        f"- Vector Flux: {_safe_float(last_row.get('Vector_Flux', np.nan)):.3f}\n"
        f"- Divergences: {div_txt}\n"
        f"- Squeeze Momentum: {'Rising' if _safe_float(last_row.get('Sqz_Mom', 0)) > 0 else 'Falling'} ({_safe_float(last_row.get('Sqz_Mom', 0)):.1f}) | {squeeze}\n"
        f"- Money Flow Matrix: {_safe_float(last_row.get('MF_Matrix', np.nan)):.2f}\n\n"
        f"4) STRUCTURE & LIQUIDITY\n"
        f"- Nexus Structure: {struct_txt}\n"
        f"- Apex Liquidity Zones (active snapshot): Supply {int(last_row.get('Apex_Supply_Zones_Count', 0))} | Demand {int(last_row.get('Apex_Demand_Zones_Count', 0))}\n"
        f"- Nexus FVGs (detected): {int(last_row.get('Nexus_FVG_Count', 0))}\n\n"
        f"5) RISK SNAPSHOT\n"
        f"- ATR(14): {_safe_float(last_row.get('ATR', np.nan)):.2f}\n"
        f"- Nexus UT Stop(ref): {_safe_float(last_row.get('Nexus_UT_Stop', np.nan)) if np.isfinite(_safe_float(last_row.get('Nexus_UT_Stop', np.nan))) else 'N/A'}\n"
        f"- Position Model (ref): Risk ${sizing['risk_usd']:.2f} | Size {sizing['size_units']:.4f}\n\n"
        f"6) MACRO CONTEXT\n{macro_text}\n\n"
        f"7) AI OUTLOOK (Grounded Summary)\n{ai_verdict}\n\n"
        f"⚠️ Not financial advice. This is a computed market intelligence report.\n"
        f"#DarkPool #Titan #Quant"
    )
    return [msg]

def send_telegram_messages(tg_token: str, tg_chat: str, messages: list[str], uploaded_file=None):
    """
    Preserves your original behavior:
    - Optional photo upload
    - Split long messages into safe chunks
    - No parse_mode to avoid Telegram cutoffs
    """
    if not tg_token or not tg_chat:
        raise ValueError("Missing Telegram token/chat id.")

    if uploaded_file:
        try:
            url_photo = f"https://api.telegram.org/bot{tg_token}/sendPhoto"
            data_photo = {'chat_id': tg_chat, 'caption': f"🔥 Analysis Upload"}
            files = {'photo': uploaded_file.getvalue()}
            requests.post(url_photo, data=data_photo, files=files, timeout=20)
        except Exception:
            pass

    url_msg = f"https://api.telegram.org/bot{tg_token}/sendMessage"
    max_length = 2000

    for m in messages:
        clean_msg = (m or "").replace("###", "")
        if len(clean_msg) <= max_length:
            requests.post(url_msg, data={"chat_id": tg_chat, "text": clean_msg}, timeout=20)
        else:
            for i in range(0, len(clean_msg), max_length):
                chunk = clean_msg[i:i+max_length]
                part_no = (i // max_length) + 1
                requests.post(url_msg, data={"chat_id": tg_chat, "text": f"(Part {part_no}) {chunk}"}, timeout=20)

# ==========================================
# 7. UI DASHBOARD LAYOUT (PRESERVED + NEW SETTINGS)
# ==========================================
st.sidebar.header("🎛️ Terminal Controls")
st.sidebar.subheader("📢 Social Broadcaster")

if 'tg_token' not in st.session_state:
    st.session_state.tg_token = ""
if 'tg_chat' not in st.session_state:
    st.session_state.tg_chat = ""

if "TELEGRAM_TOKEN" in st.secrets:
    st.session_state.tg_token = st.secrets["TELEGRAM_TOKEN"]
if "TELEGRAM_CHAT_ID" in st.secrets:
    st.session_state.tg_chat = st.secrets["TELEGRAM_CHAT_ID"]

tg_token = st.sidebar.text_input("Telegram Bot Token", value=st.session_state.tg_token, type="password", help="Enter your Telegram Bot Token")
tg_chat = st.sidebar.text_input("Telegram Chat ID", value=st.session_state.tg_chat, help="Enter your Telegram Chat ID")

# NEW: Multi-chat broadcasting (comma-separated)
tg_multi = st.sidebar.text_input("Telegram Multi-Chat IDs (comma-separated)", value="", help="Optional: broadcast to multiple chat IDs. If empty, uses Telegram Chat ID above.")

input_mode = st.sidebar.radio("Input Mode:", ["Curated Lists", "Manual Search (Global)", "Ticker Library (Extensive + Upload)"], index=2, help="Select input mode")

builtin_lib = build_builtin_ticker_library()
uploaded = None
uploaded_tickers = []
selected_category = None

if input_mode == "Curated Lists":
    assets = {
        "Indices": ["SPY", "QQQ", "DIA", "IWM", "VTI"],
        "Crypto (Top 20)": [
            "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
            "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "TRX-USD",
            "LINK-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "BCH-USD",
            "XLM-USD", "ALGO-USD", "ATOM-USD", "UNI-USD", "FIL-USD"
        ],
        "Tech Giants (Top 10)": [
            "NVDA", "TSLA", "AAPL", "MSFT", "GOOGL",
            "AMZN", "META", "AMD", "NFLX", "INTC"
        ],
        "Macro & Commodities": [
            "^TNX", "DX-Y.NYB", "GC=F", "SI=F",
            "CL=F", "NG=F", "^VIX", "TLT"
        ]
    }
    cat = st.sidebar.selectbox("Asset Class", list(assets.keys()), help="Select asset class")
    ticker = st.sidebar.selectbox("Ticker", assets[cat], help="Select ticker")
elif input_mode == "Manual Search (Global)":
    st.sidebar.info("Type any ticker (e.g. SSLN.L, BTC-USD)")
    ticker = st.sidebar.text_input("Search Ticker Symbol", value="BTC-USD", help="Enter ticker symbol").upper()
else:
    st.sidebar.markdown("### 📚 Ticker Library")
    selected_category = st.sidebar.selectbox("Library Category", list(builtin_lib.keys()), index=0)
    uploaded = st.sidebar.file_uploader("Upload tickers (CSV with 'ticker' column or TXT)", type=["csv", "txt"])
    uploaded_tickers = parse_uploaded_tickers(uploaded) if uploaded else []
    search_q = st.sidebar.text_input("Search within selected set", value="", help="Filter tickers by substring")

    universe = list(builtin_lib.get(selected_category, [])) + list(uploaded_tickers)
    universe = sorted(list(dict.fromkeys([x.strip() for x in universe if x and str(x).strip()])))

    if search_q.strip():
        sq = search_q.strip().upper()
        universe = [t for t in universe if sq in t.upper()]

    if not universe:
        ticker = st.sidebar.text_input("Fallback Manual Ticker", value="BTC-USD").upper()
    else:
        ticker = st.sidebar.selectbox("Ticker", universe, index=0)

interval = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=3, help="Select time interval")
st.sidebar.markdown("---")

balance = st.sidebar.number_input("Capital ($)", 1000, 1000000, 10000, help="Enter your capital")
risk_pct = st.sidebar.slider("Risk %", 0.5, 3.0, 1.0, help="Select risk percentage")

# ==============================
# NEW: FULL PINE INDICATOR CONFIG (NO REMOVALS)
# ==============================
st.sidebar.markdown("---")
with st.sidebar.expander("🌊 Apex Trend & Liquidity (Full)", expanded=False):
    apex_ma_type = st.selectbox("Trend Algorithm", ["EMA", "SMA", "HMA", "RMA"], index=2)
    apex_len_main = st.number_input("Trend Length", min_value=10, value=55, step=1)
    apex_mult = st.number_input("Volatility Multiplier", min_value=0.1, value=1.5, step=0.1)
    apex_src = st.selectbox("Source", ["Close", "Open", "High", "Low"], index=0)
    apex_show_liq = st.checkbox("Show Smart Liquidity Zones", value=True)
    apex_liq_len = st.number_input("Pivot Lookback", min_value=1, value=10, step=1)
    apex_zone_ext = st.number_input("Zone Extension", min_value=1, value=5, step=1)
    apex_use_vol = st.checkbox("Volume Filter", value=True)
    apex_use_rsi = st.checkbox("RSI Filter", value=False)

with st.sidebar.expander("🧠 Nexus v8.2 (Full)", expanded=False):
    nx_h = st.number_input("Kernel Lookback", min_value=10, value=50, step=1)
    nx_r = st.number_input("Kernel Smoothness", min_value=0.25, value=8.0, step=0.25)
    nx_gann = st.number_input("Gann Breakout Length", min_value=5, value=20, step=1)
    nx_tf = st.selectbox("Macro Trend Timeframe", ["", "4H", "1D", "1W"], index=0, help="Empty = current timeframe")
    nx_a = st.number_input("Stop Sensitivity", min_value=0.1, value=4.0, step=0.1)
    nx_atr = st.number_input("ATR Period", min_value=1, value=14, step=1)
    nx_liq = st.number_input("Pivot Length (Structure)", min_value=1, value=20, step=1)
    nx_show_fvg = st.checkbox("Show FVG", value=True)
    nx_filter_doji = st.checkbox("Filter Small FVGs", value=True)
    nx_strict = st.checkbox("Strict Mode (Filter Counter-Trend BOS)", value=True)

with st.sidebar.expander("⚛️ Apex Vector v4.1 (Full)", expanded=False):
    vx_eff_super = st.number_input("Superconductor Threshold", min_value=0.1, max_value=1.0, value=0.60, step=0.05)
    vx_eff_resist = st.number_input("Resistive Threshold", min_value=0.0, max_value=0.5, value=0.30, step=0.05)
    vx_vol_norm = st.number_input("Volume Normalization", min_value=10, value=55, step=1)
    vx_len_vec = st.number_input("Vector Length", min_value=2, value=14, step=1)
    vx_sm_type = st.selectbox("Smoothing Type", ["EMA", "SMA", "RMA", "WMA", "VWMA"], index=0)
    vx_len_sm = st.number_input("Smoothing Length", min_value=1, value=5, step=1)
    vx_use_vol = st.checkbox("Integrate Volume Flux", value=True)
    vx_strict = st.number_input("Global Strictness", min_value=0.1, value=1.0, step=0.1)
    vx_show_div = st.checkbox("Show Divergences", value=True)
    vx_div_look = st.number_input("Divergence Pivot Lookback", min_value=1, value=5, step=1)
    vx_show_reg = st.checkbox("Regular (Reversal)", value=True)
    vx_show_hid = st.checkbox("Hidden (Continuation)", value=False)

# Macro header (preserved)
macro_groups, m_price, m_chg = get_macro_data()

if m_price:
    group_names = list(macro_groups.keys())
    for i in range(0, len(group_names), 2):
        cols = st.columns(2)
        g1 = group_names[i]
        with cols[0].container(border=True):
            st.markdown(f"#### {g1}")
            sc = st.columns(4)
            for x, (n, s) in enumerate(macro_groups[g1].items()):
                fmt = "{:.3f}" if any(c in n for c in ["Yield", "GBP", "EUR", "JPY"]) else "{:,.2f}"
                sc[x].metric(n.split('(')[0], fmt.format(m_price.get(n, 0)), f"{m_chg.get(n, 0):.2f}%")
        if i + 1 < len(group_names):
            g2 = group_names[i+1]
            with cols[1].container(border=True):
                st.markdown(f"#### {g2}")
                sc = st.columns(4)
                for x, (n, s) in enumerate(macro_groups[g2].items()):
                    fmt = "{:.3f}" if any(c in n for c in ["Yield", "GBP", "EUR", "JPY"]) else "{:,.2f}"
                    sc[x].metric(n.split('(')[0], fmt.format(m_price.get(n, 0)), f"{m_chg.get(n, 0):.2f}%")
    st.markdown("---")

tab1, tab2, tab3, tab4, tab9, tab5, tab6, tab7, tab8, tab10 = st.tabs([
    "📊 God Mode Technicals",
    "🌍 Sector & Fundamentals",
    "📅 Monthly Seasonality",
    "📆 Day of Week DNA",
    "🧩 Correlation & MTF",
    "📟 DarkPool Dashboard",
    "🏦 Smart Money Concepts",
    "🔮 Quantitative Forecasting",
    "📊 Volume Profile",
    "📡 Broadcast & TradingView"
])

if st.button(f"Analyze {ticker}", help="Run Analysis"):
    st.session_state['run_analysis'] = True

if st.session_state.get('run_analysis'):
    with st.spinner(f"Analyzing {ticker} in God Mode..."):
        if interval in ["1m", "2m", "5m", "15m", "30m"]:
            fetch_period = "59d"
        elif interval in ["1h", "4h"]:
            fetch_period = "1y"
        else:
            fetch_period = "2y"

        df = safe_download(ticker, fetch_period, interval)

        if interval == "4h" and df is not None:
            agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            if 'Adj Close' in df.columns:
                agg_dict['Adj Close'] = 'last'
            df = df.resample('4h').agg(agg_dict).dropna()

        if df is not None:
            apex_cfg = {
                "ma_type": apex_ma_type,
                "len_main": apex_len_main,
                "mult": apex_mult,
                "src": apex_src,
                "show_liq": apex_show_liq,
                "liq_len": apex_liq_len,
                "zone_ext": apex_zone_ext,
                "use_vol": apex_use_vol,
                "use_rsi": apex_use_rsi,
            }
            nexus_cfg = {
                "h_val": nx_h,
                "r_val": nx_r,
                "gann_len": nx_gann,
                "tf_trend": nx_tf,
                "a_val": nx_a,
                "c_period": nx_atr,
                "liq_len": nx_liq,
                "show_fvg": nx_show_fvg,
                "filter_doji": nx_filter_doji,
                "strict_structure": nx_strict,
            }
            vector_cfg = {
                "eff_super": vx_eff_super,
                "eff_resist": vx_eff_resist,
                "vol_norm": vx_vol_norm,
                "len_vec": vx_len_vec,
                "sm_type": vx_sm_type,
                "len_sm": vx_len_sm,
                "use_vol": vx_use_vol,
                "strictness": vx_strict,
                "show_div": vx_show_div,
                "div_look": vx_div_look,
                "show_reg": vx_show_reg,
                "show_hid": vx_show_hid,
            }

            df, apex_supply_zones, apex_demand_zones, nexus_fvgs = calc_indicators(df, apex_cfg, nexus_cfg, vector_cfg)
            df = calc_fear_greed_v4(df)
            fund = get_fundamentals(ticker)
            sr_zones = get_sr_channels(df)

            # TAB 1: TECHNICALS
            with tab1:
                st.subheader(f"🎯 Apex God Mode: {ticker}")
                col_chart, col_gauge = st.columns([0.75, 0.25])

                with col_chart:
                    fig = make_subplots(
                        rows=5, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.15, 0.12, 0.10, 0.08],
                        vertical_spacing=0.02
                    )

                    fig.add_trace(go.Candlestick(
                        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.10)',
                                             line=dict(width=0), name="Apex Cloud"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='yellow', width=2), name="HMA Trend"), row=1, col=1)

                    fig.add_trace(go.Scatter(x=df.index, y=df["Nexus_Kernel"], line=dict(color='rgba(255,255,255,0.6)', width=2), name="Nexus Kernel"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df["Nexus_UT_Stop"], line=dict(color='rgba(255,165,0,0.9)', width=2), name="Nexus UT Stop"), row=1, col=1)

                    buy_signals = df[df['GM_Score'] >= 3]
                    sell_signals = df[df['GM_Score'] <= -3]
                    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low']*0.98, mode='markers',
                                             marker=dict(symbol='triangle-up', color='#00ff00', size=10), name="GM Buy"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High']*1.02, mode='markers',
                                             marker=dict(symbol='triangle-down', color='#ff0000', size=10), name="GM Sell"), row=1, col=1)

                    apex_buys = df[df["Apex_Sig_Buy"]]
                    apex_sells = df[df["Apex_Sig_Sell"]]
                    fig.add_trace(go.Scatter(x=apex_buys.index, y=apex_buys["Low"]*0.985, mode="markers",
                                             marker=dict(symbol="circle", color="rgba(0,230,118,0.9)", size=7),
                                             name="Apex BUY"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=apex_sells.index, y=apex_sells["High"]*1.015, mode="markers",
                                             marker=dict(symbol="circle", color="rgba(255,23,68,0.9)", size=7),
                                             name="Apex SELL"), row=1, col=1)

                    nx_buy = df[df["Nexus_Signal_Buy"]]
                    nx_sell = df[df["Nexus_Signal_Sell"]]
                    fig.add_trace(go.Scatter(x=nx_buy.index, y=nx_buy["Low"]*0.975, mode="markers",
                                             marker=dict(symbol="diamond", color="rgba(0,255,255,0.9)", size=8),
                                             name="Nexus BUY"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=nx_sell.index, y=nx_sell["High"]*1.025, mode="markers",
                                             marker=dict(symbol="diamond", color="rgba(255,0,255,0.9)", size=8),
                                             name="Nexus SELL"), row=1, col=1)

                    for z in sr_zones:
                        col = "rgba(0, 255, 0, 0.15)" if df['Close'].iloc[-1] > z['max'] else "rgba(255, 0, 0, 0.15)"
                        fig.add_shape(type="rect", x0=df.index[0], x1=df.index[-1], xref="x", yref="y",
                                      y0=z['min'], y1=z['max'], fillcolor=col, line=dict(width=0), row=1, col=1)

                    for z in apex_supply_zones:
                        fig.add_shape(type="rect", x0=z["idx"], x1=df.index[-1], y0=z["bot"], y1=z["top"],
                                      fillcolor="rgba(255,23,68,0.12)", line=dict(width=1, color="rgba(255,23,68,0.35)"),
                                      row=1, col=1)
                    for z in apex_demand_zones:
                        fig.add_shape(type="rect", x0=z["idx"], x1=df.index[-1], y0=z["bot"], y1=z["top"],
                                      fillcolor="rgba(0,230,118,0.12)", line=dict(width=1, color="rgba(0,230,118,0.35)"),
                                      row=1, col=1)

                    for fvg in nexus_fvgs[-80:]:
                        fc = "rgba(0,230,118,0.18)" if fvg["dir"] == "BULL" else "rgba(255,82,82,0.18)"
                        fig.add_shape(type="rect", x0=fvg["x0"], x1=fvg["x1"], y0=fvg["y0"], y1=fvg["y1"],
                                      fillcolor=fc, line=dict(width=0), row=1, col=1)

                    colors = ['#00E676' if v > 0 else '#FF5252' for v in df['Sqz_Mom'].fillna(0)]
                    fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=colors, name="Squeeze Mom"), row=2, col=1)

                    fig.add_trace(go.Scatter(x=df.index, y=df['MF_Matrix'], fill='tozeroy',
                                             line=dict(color='cyan', width=1), name="Money Flow"), row=3, col=1)

                    flux = df["Vector_Flux"].fillna(0)
                    flux_colors = np.where(df["Vector_SuperBull"], "rgba(0,230,118,0.85)",
                                           np.where(df["Vector_SuperBear"], "rgba(255,23,68,0.85)",
                                                    np.where(df["Vector_Resistive"], "rgba(84,110,122,0.85)", "rgba(255,214,0,0.85)")))
                    fig.add_trace(go.Bar(x=df.index, y=flux, marker_color=flux_colors, name="Vector Flux"), row=4, col=1)
                    ths = float(df["Vector_Th_Super"].iloc[-1]) if len(df) else 0.0
                    fig.add_hline(y=ths, line=dict(width=1, dash="dot"), row=4, col=1)
                    fig.add_hline(y=-ths, line=dict(width=1, dash="dot"), row=4, col=1)

                    fig.add_trace(go.Scatter(x=df.index, y=df["Nexus_UT_Pos"], mode="lines", name="UT Pos"), row=5, col=1)

                    fig.update_layout(height=950, template="plotly_dark", xaxis_rangeslider_visible=False, title_text="God Mode Technical Stack (Full Pine Ports)")
                    st.plotly_chart(fig, use_container_width=True)

                with col_gauge:
                    fg_val = df['FG_Index'].iloc[-1] if "FG_Index" in df.columns else 0
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=float(fg_val) if np.isfinite(fg_val) else 0,
                        title={'text': "Fear & Greed"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "white"},
                               'steps': [{'range': [0, 20], 'color': "#FF0000"},
                                         {'range': [80, 100], 'color': "#00FF00"}]}
                    ))
                    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    st.markdown("### 🧬 Indicator DNA")
                    last_row = df.iloc[-1]
                    st.metric("God Mode Score", f"{last_row['GM_Score']:.0f} / 5", delta="Bullish" if last_row['GM_Score'] > 0 else "Bearish")
                    st.metric("Omni Confidence", f"{int(last_row['Omni_Confidence'])}/100", delta="Aligned" if abs(int(last_row["Omni_Score"])) >= 3 else "Mixed")
                    st.metric("Apex Trend", "BULL" if int(last_row['Apex_Trend']) == 1 else "BEAR" if int(last_row['Apex_Trend']) == -1 else "CHOP")
                    st.metric("Nexus", "BUY" if bool(last_row["Nexus_Signal_Buy"]) else "SELL" if bool(last_row["Nexus_Signal_Sell"]) else "WAIT")
                    st.metric("Vector State", "SUPER BULL" if bool(last_row["Vector_SuperBull"]) else "SUPER BEAR" if bool(last_row["Vector_SuperBear"]) else "RESISTIVE" if bool(last_row["Vector_Resistive"]) else "HEAT")
                    st.metric("Squeeze", "ON" if bool(last_row['Squeeze_On']) else "OFF")
                    st.metric("Money Flow", f"{_safe_float(last_row['MF_Matrix']):.2f}")

                st.markdown("### 🤖 Strategy Briefing")
                ai_verdict = ask_ai_analyst(df, ticker, fund, balance, risk_pct, interval)
                st.info(ai_verdict)

            # TAB 2: FUNDAMENTALS
            with tab2:
                if fund:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("P/E Ratio", f"{fund.get('P/E Ratio', 'N/A')}")
                    c2.metric("Rev Growth", f"{fund.get('Rev Growth', 0)*100:.1f}%")
                    c3.metric("Debt/Equity", f"{fund.get('Debt/Equity', 'N/A')}")
                    st.write(f"**Summary:** {fund.get('Summary', 'No Data')}")
                st.subheader("🔥 Global Market Heatmap")
                s_data = get_global_performance()
                if s_data is not None:
                    fig_sector = go.Figure(go.Bar(
                        x=s_data.values, y=s_data.index, orientation='h',
                        marker_color=['#00ff00' if v >= 0 else '#ff0000' for v in s_data.values]
                    ))
                    fig_sector.update_layout(height=400, template="plotly_dark")
                    st.plotly_chart(fig_sector, use_container_width=True)

            # TAB 3: SEASONALITY
            with tab3:
                seas = get_seasonality_stats(ticker)
                if seas:
                    hm, hold, month = seas
                    fig_hm = px.imshow(hm, color_continuous_scale='RdYlGn', text_auto='.1f')
                    fig_hm.update_layout(template="plotly_dark", height=500)
                    st.plotly_chart(fig_hm, use_container_width=True)
                    c1, c2 = st.columns(2)
                    c1.dataframe(pd.DataFrame(hold).T.style.format("{:.1f}%").background_gradient(cmap="RdYlGn"))
                    curr_m = datetime.datetime.now().month
                    if curr_m in month.index:
                        c2.metric("Current Month Win Rate", f"{month.loc[curr_m, 'Win Rate']:.1f}%")

            # TAB 4: DNA
            with tab4:
                st.subheader("📆 Day & Hour DNA")
                c1, c2 = st.columns(2)

                dna_res = calc_day_of_week_dna(ticker, 250, "Close to Close (Total)")
                if dna_res:
                    cum, stats = dna_res
                    with c1:
                        st.markdown("**Day of Week Performance**")
                        fig_dna = go.Figure()
                        for c in cum.columns:
                            fig_dna.add_trace(go.Scatter(x=cum.index, y=cum[c], name=c))
                        fig_dna.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig_dna, use_container_width=True)
                        st.dataframe(stats.style.background_gradient(subset=['Win Rate'], cmap="RdYlGn"))

                hourly_res = calc_intraday_dna(ticker)
                if hourly_res is not None:
                    with c2:
                        st.markdown("**Intraday (Hourly) Performance**")
                        fig_hr = px.bar(hourly_res, x=hourly_res.index, y='Avg Return', color='Win Rate', color_continuous_scale='RdYlGn')
                        fig_hr.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig_hr, use_container_width=True)
                        st.dataframe(hourly_res.style.format("{:.2f}"))

            # TAB 9: CORRELATION & MTF
            with tab9:
                st.subheader("🧩 Cross-Asset Intelligence")
                c1, c2 = st.columns([0.4, 0.6])

                with c1:
                    st.markdown("**📡 Multi-Timeframe Radar**")
                    mtf_df = calc_mtf_trend(ticker)

                    def color_trend(val):
                        color = '#00ff00' if val == 'BULLISH' else '#ff0000' if val == 'BEARISH' else 'white'
                        return f'color: {color}; font-weight: bold'

                    st.dataframe(mtf_df.style.map(color_trend, subset=['Trend']), use_container_width=True)

                with c2:
                    st.markdown("**🔗 Macro Correlation Matrix (180 Days)**")
                    corr_data = calc_correlations(ticker)
                    fig_corr = px.bar(x=corr_data.values, y=corr_data.index, orientation='h', color=corr_data.values, color_continuous_scale='RdBu')
                    fig_corr.update_layout(template="plotly_dark", height=400, xaxis_title="Correlation Coefficient")
                    st.plotly_chart(fig_corr, use_container_width=True)

            # TAB 5: DASHBOARD
            with tab5:
                last = df.iloc[-1]
                dash_data = {
                    "Metric": [
                        "God Mode Score", "Omni Score", "Omni Confidence",
                        "Apex Trend", "Nexus Signal", "Vector State",
                        "Gann Trend (Nexus)", "Kernel Trend", "UT Pos", "EVWM Momentum", "RVOL"
                    ],
                    "Value": [
                        f"{int(last['GM_Score'])}",
                        f"{int(last['Omni_Score'])}",
                        f"{int(last['Omni_Confidence'])}/100",
                        "BULL" if int(last['Apex_Trend']) == 1 else "BEAR" if int(last['Apex_Trend']) == -1 else "CHOP",
                        "BUY" if bool(last["Nexus_Signal_Buy"]) else "SELL" if bool(last["Nexus_Signal_Sell"]) else "WAIT",
                        "SUPER BULL" if bool(last["Vector_SuperBull"]) else "SUPER BEAR" if bool(last["Vector_SuperBear"]) else "RESISTIVE" if bool(last["Vector_Resistive"]) else "HEAT",
                        "BULL" if int(last["Nexus_GannTrend"]) == 1 else "BEAR",
                        "BULL" if int(last["Nexus_KernelTrend"]) == 1 else "BEAR",
                        "SAFE" if int(last["Nexus_UT_Pos"]) == 1 else "RISK" if int(last["Nexus_UT_Pos"]) == -1 else "N/A",
                        f"{_safe_float(last['EVWM']):.2f}",
                        f"{_safe_float(last['RVOL']):.1f}x"
                    ]
                }
                st.dataframe(pd.DataFrame(dash_data), use_container_width=True)

            # TAB 6: SMC
            with tab6:
                smc = calculate_smc(df)
                fig_smc = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
                for ob in smc['order_blocks']:
                    fig_smc.add_shape(type="rect", x0=ob['x0'], x1=ob['x1'], y0=ob['y0'], y1=ob['y1'], fillcolor=ob['color'], opacity=0.5, line_width=0)
                for fvg in smc['fvgs']:
                    fig_smc.add_shape(type="rect", x0=fvg['x0'], x1=fvg['x1'], y0=fvg['y0'], y1=fvg['y1'], fillcolor=fvg['color'], opacity=0.5, line_width=0)
                for struct in smc['structures']:
                    fig_smc.add_shape(type="line", x0=struct['x0'], x1=struct['x1'], y0=struct['y'], y1=struct['y'], line=dict(color=struct['color'], width=1, dash="dot"))
                    fig_smc.add_annotation(x=struct['x1'], y=struct['y'], text=struct['label'], showarrow=False,
                                           yshift=10 if struct['color'] == 'green' else -10, font=dict(color=struct['color'], size=10))
                fig_smc.update_layout(height=600, template="plotly_dark", title="SMC Analysis")
                st.plotly_chart(fig_smc, use_container_width=True)

            # TAB 7: QUANT
            with tab7:
                mc = run_monte_carlo(df)
                fig_mc = go.Figure()
                for i in range(50):
                    fig_mc.add_trace(go.Scatter(y=mc[:, i], mode='lines', line=dict(color='rgba(255,255,255,0.05)'), showlegend=False))
                fig_mc.add_trace(go.Scatter(y=np.mean(mc, axis=1), mode='lines', name='Mean', line=dict(color='orange')))
                fig_mc.update_layout(height=500, template="plotly_dark", title="Monte Carlo Forecast (30 Days)")
                st.plotly_chart(fig_mc, use_container_width=True)

            # TAB 8: VOLUME PROFILE
            with tab8:
                vp, poc = calc_volume_profile(df)
                fig_vp = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.7, 0.3])
                fig_vp.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
                fig_vp.add_trace(go.Bar(x=vp['Volume'], y=vp['Price'], orientation='h', marker_color='rgba(0,200,255,0.3)'), row=1, col=2)
                fig_vp.add_hline(y=poc, line_color="yellow")
                fig_vp.update_layout(height=600, template="plotly_dark", title="Volume Profile (VPVR)")
                st.plotly_chart(fig_vp, use_container_width=True)

            # TAB 10: BROADCAST & TRADINGVIEW (COMPLETED + UPGRADED)
            with tab10:
                st.subheader("📡 Social Command Center")

                st.markdown("### 📺 TradingView Live Chart")
                tv_interval_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D", "1wk": "W"}
                tv_int = tv_interval_map.get(interval, "D")

                # Best-effort auto mapping, with override (no assumptions)
                tv_auto = ticker
                if tv_auto.endswith("-USD"):
                    tv_auto = tv_auto.replace("-", "")
                tv_override_mode = st.selectbox("TradingView Symbol Mode", ["Auto", "Manual Override"], index=0)
                tv_symbol = tv_auto
                if tv_override_mode == "Manual Override":
                    tv_symbol = st.text_input("TradingView Symbol (e.g., BINANCE:BTCUSDT, NASDAQ:NVDA, FX:EURUSD)", value=tv_auto)

                tv_widget_html = f"""
                <div class="tradingview-widget-container">
                    <div id="tradingview_widget"></div>
                    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                    <script type="text/javascript">
                    new TradingView.widget(
                    {{
                        "width": "100%",
                        "height": 520,
                        "symbol": "{tv_symbol}",
                        "interval": "{tv_int}",
                        "timezone": "Etc/UTC",
                        "theme": "dark",
                        "style": "1",
                        "locale": "en",
                        "toolbar_bg": "#0e1117",
                        "enable_publishing": false,
                        "hide_side_toolbar": false,
                        "allow_symbol_change": true,
                        "container_id": "tradingview_widget"
                    }}
                    );
                    </script>
                </div>
                """
                components.html(tv_widget_html, height=560, scrolling=False)

                st.markdown("### 🧾 TradingView Ticker Tape (Live Banner)")
                # Build a tape list from current category + a few macro staples
                tape_candidates = []
                try:
                    if selected_category and selected_category in builtin_lib:
                        tape_candidates.extend(builtin_lib[selected_category][:10])
                except Exception:
                    pass
                tape_candidates.extend(["SPY", "QQQ", "GC=F", "CL=F", "BTC-USD", "ETH-USD", "^VIX"])
                tape_candidates = sorted(list(dict.fromkeys([t for t in tape_candidates if t])))

                # TradingView ticker tape expects "proName" sometimes; we provide best-effort "symbol"
                # No assumptions: user can override with manual list in text area.
                tape_manual = st.text_area("Optional: Override tape symbols (comma-separated)", value="")

                tape_list = []
                if tape_manual.strip():
                    tape_list = [x.strip() for x in tape_manual.split(",") if x.strip()]
                else:
                    tape_list = tape_candidates

                tape_json_items = []
                for sym in tape_list[:20]:
                    if sym.endswith("-USD"):
                        sym_tv = sym.replace("-", "")
                    else:
                        sym_tv = sym
                    tape_json_items.append(f'{{"proName":"{sym_tv}","title":"{sym}"}}')

                ticker_tape_html = f"""
                <div class="tradingview-widget-container">
                  <div class="tradingview-widget-container__widget"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
                  {{
                    "symbols": [{",".join(tape_json_items)}],
                    "showSymbolLogo": true,
                    "colorTheme": "dark",
                    "isTransparent": true,
                    "displayMode": "adaptive",
                    "locale": "en"
                  }}
                  </script>
                </div>
                """
                components.html(ticker_tape_html, height=90, scrolling=False)

                st.markdown("---")
                st.markdown("### 📣 Broadcast Builder (Indicator-Driven)")

                last_row = df.iloc[-1]
                macro_text = "Macro Dashboard: Use top-of-app panels for cross-asset context."
                report_type_ui = st.selectbox(
                    "Report Type",
                    ["QUICK_PING", "TRADE_SCALP", "TRADE_SWING", "FULL_ANALYSIS"],
                    index=0
                )

                attach = st.file_uploader("Optional: Attach an image to Telegram (sent first)", type=["png", "jpg", "jpeg"])

                # Multi-target selection (for batch broadcast), does not break single-ticker flow
                st.markdown("#### 🎯 Batch Broadcast (Optional)")
                batch_list = []
                if input_mode == "Ticker Library (Extensive + Upload)":
                    batch_list = st.multiselect("Select multiple tickers for batch report", options=sorted(list(dict.fromkeys(builtin_lib.get(selected_category, []) + uploaded_tickers))), default=[])
                else:
                    batch_list = st.multiselect("Select multiple tickers for batch report", options=sorted(list(dict.fromkeys(sum(build_builtin_ticker_library().values(), [])))), default=[])

                generate_ai_for_broadcast = st.checkbox("Include AI Analyst section in FULL_ANALYSIS", value=True)

                def _resolve_targets(primary_chat: str, multi_field: str):
                    targets = []
                    if multi_field.strip():
                        parts = [x.strip() for x in multi_field.split(",") if x.strip()]
                        targets.extend(parts)
                    if primary_chat and primary_chat.strip():
                        if primary_chat.strip() not in targets:
                            targets.insert(0, primary_chat.strip())
                    return targets

                if st.button("🧾 Generate Report Preview"):
                    ai_text = ai_verdict if (report_type_ui == "FULL_ANALYSIS" and generate_ai_for_broadcast) else "AI section disabled for preview."
                    msgs = format_signal_report(
                        ticker=ticker,
                        interval=interval,
                        df=df,
                        last_row=last_row,
                        macro_text=macro_text,
                        ai_verdict=ai_text,
                        balance=balance,
                        risk_pct=risk_pct,
                        report_type=report_type_ui
                    )
                    st.text_area("Preview", value="\n\n---\n\n".join(msgs), height=350)

                if st.button("🚀 Send to Telegram Now"):
                    if not tg_token or not tg_chat:
                        st.error("Missing Telegram token/chat id.")
                    else:
                        try:
                            targets = _resolve_targets(tg_chat, tg_multi)
                            ai_text = ai_verdict if (report_type_ui == "FULL_ANALYSIS" and generate_ai_for_broadcast) else "AI section disabled."
                            # Single or batch
                            tickers_to_send = batch_list if batch_list else [ticker]
                            for tk in tickers_to_send:
                                df_local = df
                                last_local = last_row
                                if tk != ticker:
                                    df2 = safe_download(tk, fetch_period, interval)
                                    if interval == "4h" and df2 is not None:
                                        agg_dict2 = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
                                        if 'Adj Close' in df2.columns:
                                            agg_dict2['Adj Close'] = 'last'
                                        df2 = df2.resample('4h').agg(agg_dict2).dropna()
                                    if df2 is None or df2.empty:
                                        continue
                                    df2, apex_sz, apex_dz, nx_fv = calc_indicators(df2, apex_cfg, nexus_cfg, vector_cfg)
                                    df2 = calc_fear_greed_v4(df2)
                                    df_local = df2
                                    last_local = df2.iloc[-1]

                                msgs = format_signal_report(
                                    ticker=tk,
                                    interval=interval,
                                    df=df_local,
                                    last_row=last_local,
                                    macro_text=macro_text,
                                    ai_verdict=ai_text,
                                    balance=balance,
                                    risk_pct=risk_pct,
                                    report_type=report_type_ui
                                )
                                for target_chat in targets:
                                    send_telegram_messages(tg_token, target_chat, msgs, uploaded_file=attach)
                            st.success("✅ Telegram broadcast sent.")
                        except Exception as e:
                            st.error(f"Telegram error: {e}")

                st.markdown("### 🐦 X.com Post Generator (Copy/Paste)")
                x_mode = st.selectbox("X Post Style", ["Ultra Short", "Standard", "Thread Starter"], index=1)
                if st.button("✍️ Generate X Text"):
                    last_row = df.iloc[-1]
                    conf = int(last_row.get("Omni_Confidence", 50))
                    omni = int(last_row.get("Omni_Score", 0))
                    pxv = float(last_row.get("Close", np.nan)) if np.isfinite(_safe_float(last_row.get("Close", np.nan))) else np.nan
                    bias = "LONG" if omni > 0 else "SHORT" if omni < 0 else "NEUTRAL"
                    apex = "BULL" if int(last_row.get("Apex_Trend", 0)) == 1 else "BEAR" if int(last_row.get("Apex_Trend", 0)) == -1 else "CHOP"
                    nx = "BUY" if bool(last_row.get("Nexus_Signal_Buy", False)) else "SELL" if bool(last_row.get("Nexus_Signal_Sell", False)) else "WAIT"
                    vx = "SUPER BULL" if bool(last_row.get("Vector_SuperBull", False)) else "SUPER BEAR" if bool(last_row.get("Vector_SuperBear", False)) else "RESISTIVE" if bool(last_row.get("Vector_Resistive", False)) else "HEAT"

                    if x_mode == "Ultra Short":
                        txt = f"{ticker} ({interval}) | Bias: {bias} | Conf: {conf}/100 | Apex:{apex} Nexus:{nx} Vector:{vx} #Trading #DarkPool"
                    elif x_mode == "Thread Starter":
                        txt = (
                            f"🧠 {ticker} ({interval}) — Apex Trinity Readout\n"
                            f"Price: {pxv:.2f}\n"
                            f"Bias: {bias} | Omni: {omni} | Conf: {conf}/100\n"
                            f"Apex: {apex} | Nexus: {nx} | Vector: {vx}\n"
                            f"Next: structure/liquidity + risk map 👇\n"
                            f"#Trading #Quant"
                        )
                    else:
                        txt = (
                            f"📡 {ticker} ({interval})\n"
                            f"Bias: {bias} | Omni: {omni} | Conf: {conf}/100\n"
                            f"Apex:{apex} | Nexus:{nx} | Vector:{vx}\n"
                            f"#Trading #DarkPool"
                        )
                    st.text_area("X Post", value=txt, height=160)

        else:
            st.error("No data returned from YFinance for this ticker/interval.")
else:
    st.info("Click **Analyze** to run the full Apex Trinity terminal.")
