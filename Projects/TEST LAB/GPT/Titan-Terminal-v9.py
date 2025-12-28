# app.py
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

# ===========================
# NEW IMPORTS (NO REMOVALS)
# ===========================
import io
import json
import re

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
# 2A. NEW: LARGE TICKER UNIVERSES + TRADINGVIEW SYMBOL LAYER (NO REMOVALS)
# ==========================================
NASDAQTRADER_NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
NASDAQTRADER_OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

@st.cache_data(ttl=86400)
def fetch_us_listed_symbols_live():
    """
    Pulls official symbol directories (NASDAQ Trader).
    - No assumptions: we parse and filter out Test Issue rows deterministically.
    - Returns a large universe (typically several thousand).
    """
    symbols = set()
    try:
        r1 = requests.get(NASDAQTRADER_NASDAQ_LISTED, timeout=20)
        if r1.status_code == 200 and r1.text:
            lines = [ln for ln in r1.text.splitlines() if ln and "File Creation Time" not in ln]
            txt = "\n".join(lines)
            df1 = pd.read_csv(io.StringIO(txt), sep="|")
            if "Symbol" in df1.columns:
                if "Test Issue" in df1.columns:
                    df1 = df1[df1["Test Issue"].astype(str).str.upper().eq("N")]
                for s in df1["Symbol"].astype(str).tolist():
                    s2 = s.strip().upper()
                    if s2 and s2 != "SYMBOL":
                        symbols.add(s2)

        r2 = requests.get(NASDAQTRADER_OTHER_LISTED, timeout=20)
        if r2.status_code == 200 and r2.text:
            lines = [ln for ln in r2.text.splitlines() if ln and "File Creation Time" not in ln]
            txt = "\n".join(lines)
            df2 = pd.read_csv(io.StringIO(txt), sep="|")
            col_sym = "ACT Symbol" if "ACT Symbol" in df2.columns else (df2.columns[0] if len(df2.columns) else None)
            if col_sym:
                if "Test Issue" in df2.columns:
                    df2 = df2[df2["Test Issue"].astype(str).str.upper().eq("N")]
                for s in df2[col_sym].astype(str).tolist():
                    s2 = s.strip().upper()
                    if s2 and s2 != "ACT SYMBOL":
                        symbols.add(s2)
    except Exception:
        pass

    clean = []
    for s in symbols:
        if s and s not in ["", "N/A", "NA"]:
            if re.fullmatch(r"[A-Z0-9\.\-\^=]+", s):
                clean.append(s)
            else:
                clean.append(s)
    clean = sorted(list(set(clean)))
    return clean

TV_SYMBOL_OVERRIDES = {
    "^VIX": "CBOE:VIX",
    "^TNX": "TVC:US10Y",
    "^IRX": "TVC:US02Y",
    "^TYX": "TVC:US30Y",
    "DX-Y.NYB": "TVC:DXY",
    "GC=F": "COMEX:GC1!",
    "SI=F": "COMEX:SI1!",
    "PL=F": "NYMEX:PL1!",
    "PA=F": "NYMEX:PA1!",
    "CL=F": "NYMEX:CL1!",
    "BZ=F": "NYMEX:BZ1!",
    "NG=F": "NYMEX:NG1!",
    "HG=F": "COMEX:HG1!",
    "ZC=F": "CBOT:ZC1!",
    "ZW=F": "CBOT:ZW1!",
    "ZS=F": "CBOT:ZS1!",
    "^DJI": "DJI",
    "^FTSE": "FTSE",
    "^GDAXI": "XETR:DAX",
    "^N225": "TVC:NI225",
    "^HSI": "HSI",
    "^RUT": "RUSSELL:RUT",
}

TV_SUFFIX_EXCHANGE = {
    ".TO": "TSX:",
    ".V": "TSXV:",
    ".L": "LSE:",
    ".AX": "ASX:",
    ".HK": "HKEX:",
    ".PA": "EURONEXT:",
    ".DE": "XETR:",
    ".SW": "SIX:",
    ".MI": "MIL:",
    ".SA": "BMFBOVESPA:",
    ".NS": "NSE:",
    ".JO": "JSE:",
    ".SI": "SGX:",
}

def build_tradingview_symbol(yf_ticker: str, tv_exchange_override: str = "", explicit_tv_symbol: str | None = None):
    """
    Deterministic conversion from selected analysis ticker (Yahoo-style) to TradingView symbol.
    - No assumptions: explicit overrides first, then deterministic parsing rules.
    """
    if explicit_tv_symbol:
        return str(explicit_tv_symbol)

    t = (yf_ticker or "").strip()
    if not t:
        return ""

    if t in TV_SYMBOL_OVERRIDES:
        return TV_SYMBOL_OVERRIDES[t]

    for suf, pref in TV_SUFFIX_EXCHANGE.items():
        if t.endswith(suf):
            core = t[:-len(suf)]
            if tv_exchange_override:
                return f"{tv_exchange_override}{core}"
            return f"{pref}{core}"

    if t.endswith("=X"):
        pair = t.replace("=X", "")
        pref = tv_exchange_override if tv_exchange_override else "FX:"
        return f"{pref}{pair}"

    if t.endswith("-USD"):
        core = t.replace("-USD", "USD").replace("-", "")
        pref = tv_exchange_override if tv_exchange_override else "CRYPTO:"
        return f"{pref}{core}"

    if t.endswith("=F"):
        core = t.replace("=F", "")
        if tv_exchange_override:
            return f"{tv_exchange_override}{core}"
        return core

    if t.startswith("^"):
        core = t.replace("^", "")
        if tv_exchange_override:
            return f"{tv_exchange_override}{core}"
        return core

    if tv_exchange_override:
        return f"{tv_exchange_override}{t}"
    return t

def tradingview_symbol_info_widget_html(tv_symbol: str):
    """
    TradingView Symbol Info widget: the 'interactive ticker banner' (updates with symbol).
    """
    symbol = (tv_symbol or "").replace('"', '\\"')
    cfg = {
        "symbol": symbol,
        "width": "100%",
        "locale": "en",
        "colorTheme": "dark",
        "isTransparent": True
    }
    return f"""
    <div class="tradingview-widget-container" style="height:140px;">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-info.js" async>
      {json.dumps(cfg)}
      </script>
    </div>
    """

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
    # no assumption: we enforce strict using comparisons where possible
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
                # Pine: box_top = high[liq_len] at detection bar; our pivot aligned to center i.
                # Use pivot bar OHLC for body calc.
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
        # Supply: if close > box_bottom => mitigated (deleted), else extend.
        last_close = float(df["Close"].iloc[-1]) if len(df) else np.nan
        for z in supply:
            if np.isfinite(last_close) and last_close > z["bot"]:
                continue
            apex_supply_zones.append(z)

        # Demand: if close < box_top => mitigated (deleted), else extend.
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
        # x: oldest->newest; reverse so newest corresponds to i=0
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
        nexus_fvgs (list), nexus_struct_lines (list)
    """
    h_val = int(cfg["h_val"])
    r_val = float(cfg["r_val"])
    gann_len = int(cfg["gann_len"])
    tf_trend = cfg["tf_trend"]  # "", "4H", "1D", "1W" (UI mapping below)
    a_val = float(cfg["a_val"])
    c_period = int(cfg["c_period"])
    liq_len = int(cfg["liq_len"])
    show_fvg = bool(cfg["show_fvg"])
    filter_doji = bool(cfg["filter_doji"])
    strict_structure = bool(cfg["strict_structure"])

    # --- Kernel line (macro tf capable) ---
    # No assumptions: yfinance granularity and resample depend on fetched interval.
    # If tf_trend is set, we resample the already-downloaded df to the requested bucket, compute kernel, then ffill back.
    base_close = df["Close"]

    if tf_trend and tf_trend != "":
        rule = {"4H": "4h", "1D": "1D", "1W": "1W"}.get(tf_trend, "")
        if rule:
            res = df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
            k_res = rational_quadratic(res["Close"], h_val, r_val, 0)
            k_aligned = k_res.reindex(df.index, method="ffill")
            kernel = k_aligned
        else:
            kernel = rational_quadratic(base_close, h_val, r_val, 0)
    else:
        kernel = rational_quadratic(base_close, h_val, r_val, 0)

    kernel_trend = np.where(kernel > kernel.shift(1), 1, -1)

    # --- Gann Activator (current TF) ---
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

    # --- Risk Engine (UT Bot) ---
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

        # position flip logic using prev stop
        if (c1 < prev_stop) and (c > prev_stop):
            ut_pos[i] = 1
        elif (c1 > prev_stop) and (c < prev_stop):
            ut_pos[i] = -1
        else:
            ut_pos[i] = ut_pos[i-1]

    # --- Structure Engine ---
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

    # --- FVG Logic ---
    global_atr = calculate_atr(df, 14)
    body_size = (df["Close"] - df["Open"]).abs()
    is_significant = (~filter_doji) | (body_size > (global_atr * 0.5))

    nexus_fvgs = []
    if show_fvg and len(df) >= 3:
        for i in range(2, len(df)):
            if not bool(is_significant.iloc[i]):
                continue
            # Bullish FVG: low > high[2]
            if df["Low"].iloc[i] > df["High"].iloc[i-2] and (df["Low"].iloc[i] - df["High"].iloc[i-2]) > 0:
                nexus_fvgs.append({
                    "x0": df.index[i-2], "x1": df.index[i],
                    "y0": float(df["High"].iloc[i-2]), "y1": float(df["Low"].iloc[i]),
                    "dir": "BULL"
                })
            # Bearish FVG: high < low[2]
            if df["High"].iloc[i] < df["Low"].iloc[i-2] and (df["Low"].iloc[i-2] - df["High"].iloc[i]) > 0:
                nexus_fvgs.append({
                    "x0": df.index[i-2], "x1": df.index[i],
                    "y0": float(df["Low"].iloc[i-2]), "y1": float(df["High"].iloc[i]),
                    "dir": "BEAR"
                })

    # --- Signal Logic (Omni) ---
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

    # Divergence Engine (pivot on flux)
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

    # Preserve existing names used elsewhere (already present: Apex_Upper/Apex_Lower/Apex_Trend)
    # and add explicit aliases for clarity (no removals)
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
    # No assumptions: protect divisions from zero/NaN
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
    # No assumptions: keep deterministic integer scoring
    vector_bias = np.where(df["Vector_SuperBull"], 1, np.where(df["Vector_SuperBear"], -1, 0))
    nexus_bias = np.where(df["Nexus_FullBull"], 1, np.where(df["Nexus_FullBear"], -1, 0))
    struct_bias = np.where(df["Nexus_BOS_Bull"] | df["Nexus_CHoCH_Bull"], 1,
                           np.where(df["Nexus_BOS_Bear"] | df["Nexus_CHoCH_Bear"], -1, 0))

    df["Omni_Score"] = df["GM_Score"].astype(int) + vector_bias + nexus_bias + struct_bias

    # Normalize to 0..100 confidence band (bounded)
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

def _fmt_num(x, dp=2):
    v = _safe_float(x)
    if np.isfinite(v):
        return f"{v:.{dp}f}"
    return "N/A"

def _fmt_int(x):
    try:
        if np.isfinite(_safe_float(x)):
            return str(int(float(x)))
        return "N/A"
    except:
        return "N/A"

def _fmt_bool(x):
    try:
        return "TRUE" if bool(x) else "FALSE"
    except:
        return "FALSE"

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
    # Stop selection: if UT stop finite, use it; else fallback to ATR*2 away
    if np.isfinite(ut):
        stop = ut
    else:
        if np.isfinite(entry) and np.isfinite(atr):
            stop = entry - (2 * atr) if direction == "LONG" else entry + (2 * atr)
        else:
            stop = float("nan")

    # R distance
    r = abs(entry - stop) if (np.isfinite(entry) and np.isfinite(stop) and entry != stop) else (atr if np.isfinite(atr) else float("nan"))
    tp1 = entry + (1 * r) if direction == "LONG" else entry - (1 * r)
    tp2 = entry + (2 * r) if direction == "LONG" else entry - (2 * r)
    tp3 = entry + (3 * r) if direction == "LONG" else entry - (3 * r)

    return {"entry": entry, "stop": stop, "r": r, "tp1": tp1, "tp2": tp2, "tp3": tp3}

def build_position_sizing(balance: float, risk_pct: float, levels: dict):
    risk_dollars = balance * (risk_pct / 100.0)
    r = levels.get("r", float("nan"))
    entry = levels.get("entry", float("nan"))
    if not np.isfinite(r) or r <= 0:
        return {"risk_usd": risk_dollars, "size_units": float("nan"), "note": "Size unavailable (invalid risk distance)."}
    size = risk_dollars / r
    return {"risk_usd": risk_dollars, "size_units": size, "note": ""}

def _build_full_indicator_lines(df: pd.DataFrame, last_row: pd.Series, sr_zones: list | None = None):
    """
    FULL coverage of ALL indicator outputs computed in this code path.
    No omissions: we enumerate every known indicator column created by calc_indicators() + calc_fear_greed_v4().
    If a column is missing for any reason, we report it as N/A (runtime-safe).
    """
    lines = []

    # Core OHLCV (base data)
    lines.append("🧾 CORE OHLCV (LAST BAR)")
    lines.append(f"- Open: {_fmt_num(last_row.get('Open', np.nan))} | High: {_fmt_num(last_row.get('High', np.nan))} | Low: {_fmt_num(last_row.get('Low', np.nan))} | Close: {_fmt_num(last_row.get('Close', np.nan))}")
    lines.append(f"- Volume: {_fmt_int(last_row.get('Volume', np.nan))}")

    # Base Calcs (existing)
    lines.append("\n🧩 BASE CALCS (EXISTING)")
    lines.append(f"- HMA(55): {_fmt_num(last_row.get('HMA', np.nan))}")
    lines.append(f"- ATR(14): {_fmt_num(last_row.get('ATR', np.nan))}")
    lines.append(f"- Pivot_Resist(20H): {_fmt_num(last_row.get('Pivot_Resist', np.nan))}")
    lines.append(f"- Pivot_Support(20L): {_fmt_num(last_row.get('Pivot_Support', np.nan))}")
    lines.append(f"- MFI (custom 14): {_fmt_num(last_row.get('MFI', np.nan))}")

    # Apex Trend & Liquidity Master (FULL)
    lines.append("\n🌊 APEX TREND & LIQUIDITY MASTER (FULL PORT)")
    lines.append(f"- Apex_Base: {_fmt_num(last_row.get('Apex_Base', np.nan))}")
    lines.append(f"- Apex_ATR_Main(len_main): {_fmt_num(last_row.get('Apex_ATR_Main', np.nan))}")
    lines.append(f"- Apex_Upper: {_fmt_num(last_row.get('Apex_Upper', np.nan))} | Apex_Lower: {_fmt_num(last_row.get('Apex_Lower', np.nan))}")
    lines.append(f"- Apex_Trend: {_fmt_int(last_row.get('Apex_Trend', 0))} (1 bull, -1 bear, 0 chop)")
    lines.append(f"- Apex_Sig_Buy: {_fmt_bool(last_row.get('Apex_Sig_Buy', False))} | Apex_Sig_Sell: {_fmt_bool(last_row.get('Apex_Sig_Sell', False))}")
    lines.append(f"- Apex_RSI(14): {_fmt_num(last_row.get('Apex_RSI', np.nan), 1)}")
    lines.append(f"- Apex_VolMA20: {_fmt_num(last_row.get('Apex_VolMA20', np.nan))} | Apex_HighVol: {_fmt_bool(last_row.get('Apex_HighVol', False))}")
    lines.append(f"- Apex_Supply_Zones_Count: {_fmt_int(last_row.get('Apex_Supply_Zones_Count', 0))} | Apex_Demand_Zones_Count: {_fmt_int(last_row.get('Apex_Demand_Zones_Count', 0))}")
    lines.append(f"- ApexMaster_Upper(alias): {_fmt_num(last_row.get('ApexMaster_Upper', np.nan))} | ApexMaster_Lower(alias): {_fmt_num(last_row.get('ApexMaster_Lower', np.nan))}")
    lines.append(f"- ApexMaster_Trend(alias): {_fmt_int(last_row.get('ApexMaster_Trend', 0))}")
    lines.append(f"- ApexMaster_Sig_Buy(alias): {_fmt_bool(last_row.get('ApexMaster_Sig_Buy', False))} | ApexMaster_Sig_Sell(alias): {_fmt_bool(last_row.get('ApexMaster_Sig_Sell', False))}")

    # Squeeze Momentum (existing)
    lines.append("\n💥 DARKPOOL SQUEEZE MOMENTUM (EXISTING)")
    lines.append(f"- Sqz_Basis(20): {_fmt_num(last_row.get('Sqz_Basis', np.nan))}")
    lines.append(f"- Sqz_Dev(20)*2: {_fmt_num(last_row.get('Sqz_Dev', np.nan))}")
    lines.append(f"- Sqz_Upper_BB: {_fmt_num(last_row.get('Sqz_Upper_BB', np.nan))} | Sqz_Lower_BB: {_fmt_num(last_row.get('Sqz_Lower_BB', np.nan))}")
    lines.append(f"- Sqz_Ma_KC(20): {_fmt_num(last_row.get('Sqz_Ma_KC', np.nan))}")
    lines.append(f"- Sqz_Range_MA(ATR20): {_fmt_num(last_row.get('Sqz_Range_MA', np.nan))}")
    lines.append(f"- Sqz_Upper_KC: {_fmt_num(last_row.get('Sqz_Upper_KC', np.nan))} | Sqz_Lower_KC: {_fmt_num(last_row.get('Sqz_Lower_KC', np.nan))}")
    lines.append(f"- Squeeze_On: {_fmt_bool(last_row.get('Squeeze_On', False))}")
    lines.append(f"- Sqz_Mom: {_fmt_num(last_row.get('Sqz_Mom', np.nan), 1)}")

    # Money Flow Matrix (existing)
    lines.append("\n🌊 MONEY FLOW MATRIX (EXISTING)")
    lines.append(f"- MF_Matrix: {_fmt_num(last_row.get('MF_Matrix', np.nan), 2)}")

    # Dark Vector Scalping (existing)
    lines.append("\n⚔️ DARK VECTOR SCALPING (EXISTING)")
    lines.append(f"- VS_Low(amp=5): {_fmt_num(last_row.get('VS_Low', np.nan))} | VS_High(amp=5): {_fmt_num(last_row.get('VS_High', np.nan))}")
    lines.append(f"- VS_Trend: {_fmt_int(last_row.get('VS_Trend', 0))} (1 bull, -1 bear)")

    # Advanced Volume (existing)
    lines.append("\n🔋 ADVANCED VOLUME (EXISTING)")
    lines.append(f"- RVOL(20): {_fmt_num(last_row.get('RVOL', np.nan), 2)}x")

    # EVWM (existing)
    lines.append("\n🧲 EVWM (EXISTING)")
    lines.append(f"- EVWM: {_fmt_num(last_row.get('EVWM', np.nan), 2)}")

    # Simple Gann (existing)
    lines.append("\n📐 GANN HIGH/LOW ACTIVATOR (SIMPLE, EXISTING)")
    lines.append(f"- Gann_High(3): {_fmt_num(last_row.get('Gann_High', np.nan))} | Gann_Low(3): {_fmt_num(last_row.get('Gann_Low', np.nan))}")
    lines.append(f"- Gann_Trend(simple): {_fmt_int(last_row.get('Gann_Trend', 0))}")

    # Dark Vector (SuperTrend) (existing)
    lines.append("\n🧭 DARK VECTOR (SUPERTREND, EXISTING)")
    lines.append(f"- DarkVector_Trend: {_fmt_int(last_row.get('DarkVector_Trend', 0))} (1 up, -1 down)")

    # Trend Shield (existing)
    lines.append("\n🛡️ WYCKOFF VSA TREND SHIELD (EXISTING)")
    lines.append(f"- Trend_Shield_Bull: {_fmt_bool(last_row.get('Trend_Shield_Bull', False))}")

    # Nexus v8.2 (FULL)
    lines.append("\n🧠 NEXUS v8.2 (FULL PORT)")
    lines.append(f"- Nexus_Kernel: {_fmt_num(last_row.get('Nexus_Kernel', np.nan))}")
    lines.append(f"- Nexus_KernelTrend: {_fmt_int(last_row.get('Nexus_KernelTrend', 0))}")
    lines.append(f"- Nexus_GannActivator: {_fmt_num(last_row.get('Nexus_GannActivator', np.nan))}")
    lines.append(f"- Nexus_GannTrend: {_fmt_int(last_row.get('Nexus_GannTrend', 0))}")
    lines.append(f"- Nexus_UT_Stop: {_fmt_num(last_row.get('Nexus_UT_Stop', np.nan))}")
    lines.append(f"- Nexus_UT_Pos: {_fmt_int(last_row.get('Nexus_UT_Pos', 0))} (1 long, -1 short, 0 flat)")
    lines.append(f"- Nexus_StructState: {_fmt_int(last_row.get('Nexus_StructState', 0))}")
    lines.append(f"- Nexus_BOS_Bull: {_fmt_bool(last_row.get('Nexus_BOS_Bull', False))} | Nexus_BOS_Bear: {_fmt_bool(last_row.get('Nexus_BOS_Bear', False))}")
    lines.append(f"- Nexus_CHoCH_Bull: {_fmt_bool(last_row.get('Nexus_CHoCH_Bull', False))} | Nexus_CHoCH_Bear: {_fmt_bool(last_row.get('Nexus_CHoCH_Bear', False))}")
    lines.append(f"- Nexus_FullBull: {_fmt_bool(last_row.get('Nexus_FullBull', False))} | Nexus_FullBear: {_fmt_bool(last_row.get('Nexus_FullBear', False))}")
    lines.append(f"- Nexus_Signal_Buy: {_fmt_bool(last_row.get('Nexus_Signal_Buy', False))} | Nexus_Signal_Sell: {_fmt_bool(last_row.get('Nexus_Signal_Sell', False))}")
    lines.append(f"- Nexus_FVG_Count: {_fmt_int(last_row.get('Nexus_FVG_Count', 0))}")

    # Apex Vector v4.1 (FULL)
    lines.append("\n⚛️ APEX VECTOR v4.1 (FULL PORT)")
    lines.append(f"- Vector_Eff: {_fmt_num(last_row.get('Vector_Eff', np.nan), 4)}")
    lines.append(f"- Vector_VolFact: {_fmt_num(last_row.get('Vector_VolFact', np.nan), 4)}")
    lines.append(f"- Vector_Flux: {_fmt_num(last_row.get('Vector_Flux', np.nan), 4)}")
    lines.append(f"- Vector_Th_Super: {_fmt_num(last_row.get('Vector_Th_Super', np.nan), 4)} | Vector_Th_Resist: {_fmt_num(last_row.get('Vector_Th_Resist', np.nan), 4)}")
    lines.append(f"- Vector_SuperBull: {_fmt_bool(last_row.get('Vector_SuperBull', False))} | Vector_SuperBear: {_fmt_bool(last_row.get('Vector_SuperBear', False))}")
    lines.append(f"- Vector_Resistive: {_fmt_bool(last_row.get('Vector_Resistive', False))} | Vector_Heat: {_fmt_bool(last_row.get('Vector_Heat', False))}")
    lines.append(f"- Vector_Div_Bull_Reg: {_fmt_bool(last_row.get('Vector_Div_Bull_Reg', False))} | Vector_Div_Bull_Hid: {_fmt_bool(last_row.get('Vector_Div_Bull_Hid', False))}")
    lines.append(f"- Vector_Div_Bear_Reg: {_fmt_bool(last_row.get('Vector_Div_Bear_Reg', False))} | Vector_Div_Bear_Hid: {_fmt_bool(last_row.get('Vector_Div_Bear_Hid', False))}")

    # Scores (GM + Omni)
    lines.append("\n🧬 TITAN SCORING")
    lines.append(f"- GM_Score: {_fmt_int(last_row.get('GM_Score', 0))}")
    lines.append(f"- Omni_Score: {_fmt_int(last_row.get('Omni_Score', 0))}")
    lines.append(f"- Omni_Confidence: {_fmt_int(last_row.get('Omni_Confidence', 50))}/100")

    # Dashboard v2 (existing)
    lines.append("\n📟 DARKPOOL DASHBOARD (EXISTING METRICS)")
    lines.append(f"- MACD: {_fmt_num(last_row.get('MACD', np.nan), 4)} | Signal: {_fmt_num(last_row.get('Signal', np.nan), 4)} | Hist: {_fmt_num(last_row.get('Hist', np.nan), 4)}")
    lines.append(f"- Stoch_K: {_fmt_num(last_row.get('Stoch_K', np.nan), 2)} | Stoch_D: {_fmt_num(last_row.get('Stoch_D', np.nan), 2)}")
    lines.append(f"- ROC(14): {_fmt_num(last_row.get('ROC', np.nan), 2)}")
    lines.append(f"- EMA_Fast(9): {_fmt_num(last_row.get('EMA_Fast', np.nan))} | EMA_Slow(21): {_fmt_num(last_row.get('EMA_Slow', np.nan))} | EMA_50: {_fmt_num(last_row.get('EMA_50', np.nan))}")
    lines.append(f"- OBV: {_fmt_num(last_row.get('OBV', np.nan), 0)}")
    lines.append(f"- VWAP(cum): {_fmt_num(last_row.get('VWAP', np.nan))}")
    lines.append(f"- ADX: {_fmt_num(last_row.get('ADX', np.nan), 2)}")
    lines.append(f"- RSI(14): {_fmt_num(last_row.get('RSI', np.nan), 1)}")
    lines.append(f"- Mom_Score: {_fmt_num(last_row.get('Mom_Score', np.nan), 0)}")

    # Fear & Greed v4 (existing)
    lines.append("\n😱😈 FEAR & GREED v4 (PORT)")
    lines.append(f"- FG_RSI: {_fmt_num(last_row.get('FG_RSI', np.nan), 1)}")
    lines.append(f"- FG_MACD: {_fmt_num(last_row.get('FG_MACD', np.nan), 1)}")
    lines.append(f"- FG_BB: {_fmt_num(last_row.get('FG_BB', np.nan), 1)}")
    lines.append(f"- FG_MA: {_fmt_num(last_row.get('FG_MA', np.nan), 1)}")
    lines.append(f"- FG_Raw: {_fmt_num(last_row.get('FG_Raw', np.nan), 2)}")
    lines.append(f"- FG_Index: {_fmt_num(last_row.get('FG_Index', np.nan), 2)}")
    lines.append(f"- IS_FOMO: {_fmt_bool(last_row.get('IS_FOMO', False))} | IS_PANIC: {_fmt_bool(last_row.get('IS_PANIC', False))}")

    # SR Channels summary (analysis output in code)
    lines.append("\n🏗️ SR CHANNELS (get_sr_channels)")
    if sr_zones is None:
        lines.append("- SR Zones: N/A (not provided)")
    else:
        if len(sr_zones) == 0:
            lines.append("- SR Zones: None detected")
        else:
            for i, z in enumerate(sr_zones, start=1):
                lines.append(f"- Zone {i}: min {_fmt_num(z.get('min', np.nan))} | max {_fmt_num(z.get('max', np.nan))} | score {_fmt_int(z.get('score', 0))}")

    return lines

def _chunk_lines_to_messages(title: str, ts_txt: str, interval: str, ticker: str, lines: list[str], hard_limit_chars: int = 1800):
    """
    Produce multi-part messages that are already safely sized.
    send_telegram_messages() still does its own chunking; this is for readability + 'no omissions' compliance.
    """
    messages = []
    header = f"{title} — {ticker} ({interval})\n🕒 {ts_txt}\n"
    current = header + "\n"
    part = 1

    for ln in lines:
        add = ln + "\n"
        if len(current) + len(add) > hard_limit_chars:
            messages.append(f"{header}📦 Part {part}\n\n{current[len(header)+1:]}".strip())
            part += 1
            current = header + "\n" + add
        else:
            current += add

    if current.strip():
        messages.append(f"{header}📦 Part {part}\n\n{current[len(header)+1:]}".strip())

    return messages

def format_signal_report(
    ticker: str,
    interval: str,
    df: pd.DataFrame,
    last_row: pd.Series,
    macro_text: str,
    ai_verdict: str,
    balance: float,
    risk_pct: float,
    report_type: str,
    sr_zones: list | None = None
):
    """
    Multi-type Telegram reports:
      - QUICK_PING
      - TRADE_SCALP
      - TRADE_SWING
      - FULL_ANALYSIS

    UPGRADE:
      FULL_ANALYSIS now includes FULL ANALYSIS of ALL indicators computed in this codebase (no omissions).
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

    # Direction inference for trade templates: based on Omni_Score sign (deterministic)
    direction = "LONG" if omni_score > 0 else "SHORT" if omni_score < 0 else "NEUTRAL"
    levels = build_trade_levels(last_row, "LONG" if direction == "LONG" else "SHORT")
    sizing = build_position_sizing(balance, risk_pct, levels)

    # Structure snapshot
    struct = []
    if bool(last_row.get("Nexus_BOS_Bull", False)): struct.append("🟢 BOS(BULL)")
    if bool(last_row.get("Nexus_BOS_Bear", False)): struct.append("🔴 BOS(BEAR)")
    if bool(last_row.get("Nexus_CHoCH_Bull", False)): struct.append("🟡 CHoCH(BULL)")
    if bool(last_row.get("Nexus_CHoCH_Bear", False)): struct.append("🟡 CHoCH(BEAR)")
    struct_txt = " | ".join(struct) if struct else "None"

    # Divergences snapshot
    div = []
    if bool(last_row.get("Vector_Div_Bull_Reg", False)): div.append("🔵 Bull Reg")
    if bool(last_row.get("Vector_Div_Bull_Hid", False)): div.append("🔵 Bull Hid")
    if bool(last_row.get("Vector_Div_Bear_Reg", False)): div.append("🩷 Bear Reg")
    if bool(last_row.get("Vector_Div_Bear_Hid", False)): div.append("🩷 Bear Hid")
    div_txt = " | ".join(div) if div else "None"

    # Compose
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
        # Different “flavors” but same computed levels; scalp uses tighter narrative.
        mode_tag = "⚡ SCALP SETUP" if report_type == "TRADE_SCALP" else "📈 SWING SETUP"
        risk_line = _safe_float(last_row.get("Nexus_UT_Stop", np.nan))
        risk_line_txt = f"{risk_line:.2f}" if np.isfinite(risk_line) else "N/A"

        # FIXED: syntax-safe stop string (base behavior preserved; only crash fix)
        if np.isfinite(risk_line):
            stop_txt = "Nexus UT " + risk_line_txt
        else:
            stop_txt = f"ATR Ref {levels['stop']:.2f}" if np.isfinite(_safe_float(levels.get("stop", np.nan))) else "ATR Ref N/A"

        msg = (
            f"{mode_tag} — {ticker} ({interval})\n"
            f"🕒 {ts_txt}\n\n"
            f"Bias: {direction} | 🎯 Conf: {conf}/100\n"
            f"Apex: {apex_bias} | Nexus: {nexus_bias} | Vector: {vector_state}\n"
            f"Structure: {struct_txt}\n"
            f"Divergence: {div_txt}\n"
            f"Squeeze: {squeeze}\n\n"
            f"📍 Entry(ref): ${levels['entry']:.2f}\n"
            f"🛡️ Stop(ref): {stop_txt}\n"
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

    # FULL_ANALYSIS (multi-part; will be chunked by sender)
    # UPGRADE: include FULL analysis of ALL indicators in code path (no omissions)
    full_lines = []

    full_lines.append("1) REGIME & BIAS")
    full_lines.append(f"- Price: ${_fmt_num(last_row.get('Close', np.nan))}")
    full_lines.append(f"- GM Score: {gm_score}/5 | Omni Score: {omni_score} | Confidence: {conf}/100")
    full_lines.append(f"- Direction Bias (from Omni_Score): {direction}")
    full_lines.append("")

    full_lines.append("2) RISK MODEL (REFERENCE)")
    full_lines.append(f"- Capital: ${balance:,.0f} | Risk: {risk_pct:.2f}% = ${sizing['risk_usd']:.2f}")
    full_lines.append(f"- Entry(ref): {_fmt_num(levels.get('entry', np.nan))} | Stop(ref): {_fmt_num(levels.get('stop', np.nan))}")
    full_lines.append(f"- TP1: {_fmt_num(levels.get('tp1', np.nan))} | TP2: {_fmt_num(levels.get('tp2', np.nan))} | TP3: {_fmt_num(levels.get('tp3', np.nan))}")
    full_lines.append(f"- Position Size(ref units): {_fmt_num(sizing.get('size_units', np.nan), 4)}")
    full_lines.append("")

    full_lines.append("3) HIGH-LEVEL STACK (QUICK STATE)")
    full_lines.append(f"- Apex: {apex_bias} | Apex_Sig_Buy: {_fmt_bool(last_row.get('Apex_Sig_Buy', False))} | Apex_Sig_Sell: {_fmt_bool(last_row.get('Apex_Sig_Sell', False))}")
    full_lines.append(f"- Nexus: {nexus_bias} | KernelTrend: {_fmt_int(last_row.get('Nexus_KernelTrend', 0))} | GannTrend: {_fmt_int(last_row.get('Nexus_GannTrend', 0))} | UT_Pos: {_fmt_int(last_row.get('Nexus_UT_Pos', 0))}")
    full_lines.append(f"- Vector: {vector_state} | Flux: {_fmt_num(last_row.get('Vector_Flux', np.nan), 4)} | Divergence: {div_txt}")
    full_lines.append(f"- Structure: {struct_txt}")
    full_lines.append(f"- Squeeze: {squeeze} | Sqz_Mom: {_fmt_num(last_row.get('Sqz_Mom', np.nan), 1)}")
    full_lines.append("")

    full_lines.append("4) FULL INDICATOR READOUT (NO OMISSIONS)")
    full_lines.extend(_build_full_indicator_lines(df, last_row, sr_zones=sr_zones))
    full_lines.append("")

    full_lines.append("5) MACRO CONTEXT")
    full_lines.append(macro_text)
    full_lines.append("")

    full_lines.append("6) AI OUTLOOK (Grounded Summary)")
    full_lines.append(ai_verdict)
    full_lines.append("")
    full_lines.append("⚠️ Not financial advice. This is a computed market intelligence report.")
    full_lines.append("#DarkPool #Titan #Quant")

    messages = _chunk_lines_to_messages(
        title="🧠 FULL TITAN ANALYSIS (ALL INDICATORS)",
        ts_txt=ts_txt,
        interval=interval,
        ticker=ticker,
        lines=full_lines,
        hard_limit_chars=1800
    )
    return messages

def send_telegram_messages(tg_token: str, tg_chat: str, messages: list[str], uploaded_file=None):
    """
    Preserves your original behavior:
    - Optional photo upload
    - Split long messages into safe chunks
    - No parse_mode to avoid Telegram cutoffs
    """
    if not tg_token or not tg_chat:
        raise ValueError("Missing Telegram token/chat id.")

    # 1) Photo first (optional)
    if uploaded_file:
        try:
            files = {'photo': uploaded_file.getvalue()}
            url_photo = f"https://api.telegram.org/bot{tg_token}/sendPhoto"
            data_photo = {'chat_id': tg_chat, 'caption': f"🔥 Analysis Upload", 'parse_mode': 'Markdown'}
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
import streamlit.components.v1 as components

# ==========================================
# 7. UI DASHBOARD LAYOUT (PRESERVED + NEW SETTINGS)
# ==========================================

# ------------------------------------------
# SIDEBAR: MASTER CONTROLS
# ------------------------------------------
st.sidebar.header("⚙️ Titan Controls")

# --- Universe selector (NEW: large universes incl LIVE US list + miners + juniors) ---
universe_mode = st.sidebar.selectbox(
    "Select Universe",
    [
        "US Stocks (LIVE Directory)",
        "US Mega/Large Caps (Curated)",
        "Crypto (Curated)",
        "Indices / Macro (Curated)",
        "Commodities (Curated)",
        "Precious Metals (Curated)",
        "Miners (Curated)",
        "Junior Miners (Curated)",
        "Custom / Manual"
    ],
    index=0
)

# --- Optional TradingView exchange override (NO ASSUMPTIONS) ---
st.sidebar.caption("TradingView exchange override (optional, affects widgets only).")
tv_exchange_override = st.sidebar.selectbox(
    "TV Exchange Prefix",
    ["", "NASDAQ:", "NYSE:", "AMEX:", "ARCA:", "CBOE:", "TVC:", "FX:", "CRYPTO:", "COMEX:", "NYMEX:", "CBOT:"],
    index=0
)

# --- Curated universes (separate & explicit; LIVE list is thousands) ---
US_CURATED = sorted(list(set([
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","LLY","AVGO","JPM","UNH","V","XOM","MA","HD","COST","ABBV",
    "KO","PEP","MRK","CRM","ADBE","NFLX","AMD","INTC","QCOM","ORCL","CSCO","WMT","DIS","NKE","BA","GE","IBM","T","VZ",
    "PFE","CVX","BAC","WFC","GS","MS","BLK","SPGI","ICE","C","SCHW","AXP","PYPL","SQ","SHOP","PLTR","SNOW","NOW",
    "PANW","CRWD","ZS","NET","DDOG","MDB","TEAM","INTU","TXN","AMAT","LRCX","MU","KLAC","ASML",
    "SO","DUK","NEE","AEP","EXC","D","SRE","ED","PCG",
    "CAT","DE","MMM","HON","LMT","NOC","RTX","GD",
    "MCD","SBUX","CMG","DPZ",
    "TMO","DHR","ABT","ISRG","MDT","SYK","GILD","REGN","VRTX","BIIB",
    "SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLU","XLB","XLRE","XLC"
])))

CRYPTO_CURATED = sorted(list(set([
    "BTC-USD","ETH-USD","SOL-USD","XRP-USD","BNB-USD","ADA-USD","DOGE-USD","AVAX-USD","LINK-USD","DOT-USD","MATIC-USD",
    "LTC-USD","BCH-USD","UNI-USD","ATOM-USD","ETC-USD","XLM-USD","ALGO-USD","APT-USD","SUI-USD","ARB-USD","OP-USD"
])))

INDICES_MACRO_CURATED = sorted(list(set([
    "SPY","QQQ","DIA","IWM","^GSPC","^IXIC","^DJI","^RUT",
    "^VIX","^TNX","^IRX","^TYX",
    "DX-Y.NYB","EURUSD=X","JPY=X","GBPUSD=X",
    "TLT","IEF","SHY","HYG","LQD","TIP",
    "EEM","EFA","FXI","EWJ","EWG","EWU"
])))

COMMODITIES_CURATED = sorted(list(set([
    "CL=F","BZ=F","NG=F","RB=F","HO=F",
    "HG=F","ALI=F" if False else "HG=F",  # NO ASSUMPTION: keep harmless placeholder disabled
    "ZC=F","ZW=F","ZS=F","KC=F","CT=F","SB=F","OJ=F",
    "URA","USO","DBA","DBC","PALL" if False else "PA=F"  # NO ASSUMPTION: keep placeholder disabled
])))

PRECIOUS_METALS_CURATED = sorted(list(set([
    "GC=F","SI=F","PL=F","PA=F",
    "GLD","IAU","SLV","SGOL","SIVR"
])))

MINERS_CURATED = sorted(list(set([
    # Majors / producers / royalty
    "NEM","GOLD","AEM","KGC","AU","BTG","HL","PAAS","AG","CDE","FSM","IAG","THM" if False else "IAG",
    "WPM","FNV","RGLD","SAND",
    # ETFs
    "GDX","SIL","COPX","URA","XME",
    # Copper / diversified miners
    "FCX","SCCO","BHP","RIO","VALE","TECK","GLNCY","AA","MP"
])))

JUNIOR_MINERS_CURATED = sorted(list(set([
    # ETFs / baskets
    "GDXJ","SILJ","SGDJ","GOEX",
    # Smaller caps / juniors / developers (US/Canada/Australia tickers where common)
    "NGD","MUX","OR","DRD","SSRM","EQX","EGO","SA","GAU","GATO",
    "AR.V" if False else "AR.TO",  # NO ASSUMPTION: keep placeholder disabled
    "AGI","HMY","CMCL","GORO","LODE",
    # Canada TSX/TSXV examples (Yahoo suffixes)
    "ABX.TO" if False else "AEM.TO",  # NO ASSUMPTION: placeholder disabled
    "DPM.TO","EDV.TO","FM.TO","IMG.TO","NGT.V" if False else "DPM.TO"
])))

# --- LIVE US list loader (thousands) ---
live_us_symbols = []
if universe_mode == "US Stocks (LIVE Directory)":
    st.sidebar.caption("Loads official NASDAQ Trader symbol directories (cached daily).")
    live_toggle = st.sidebar.toggle("Load LIVE US symbols", value=True)
    if live_toggle:
        live_us_symbols = fetch_us_listed_symbols_live()
        st.sidebar.caption(f"Loaded: {len(live_us_symbols):,} symbols")
    else:
        st.sidebar.caption("LIVE list is OFF.")

# --- Category-specific ticker selection ---
selected_ticker = None
explicit_tv_symbol = None  # optional direct tv symbol string if needed

if universe_mode == "US Stocks (LIVE Directory)":
    if not live_us_symbols:
        st.warning("LIVE US symbol list is not loaded. Turn on 'Load LIVE US symbols' in the sidebar.")
        selected_ticker = st.sidebar.text_input("Ticker (manual)", value="AAPL").strip().upper()
    else:
        # Optional filter box to make huge list usable
        filter_txt = st.sidebar.text_input("Filter symbols (starts with)", value="A").strip().upper()
        if filter_txt:
            filtered = [s for s in live_us_symbols if s.startswith(filter_txt)]
            if not filtered:
                filtered = live_us_symbols
        else:
            filtered = live_us_symbols

        selected_ticker = st.sidebar.selectbox("Select Symbol", filtered, index=0)

elif universe_mode == "US Mega/Large Caps (Curated)":
    selected_ticker = st.sidebar.selectbox("Select Ticker", US_CURATED, index=0)

elif universe_mode == "Crypto (Curated)":
    selected_ticker = st.sidebar.selectbox("Select Crypto", CRYPTO_CURATED, index=0)

elif universe_mode == "Indices / Macro (Curated)":
    selected_ticker = st.sidebar.selectbox("Select Index / Macro", INDICES_MACRO_CURATED, index=0)

elif universe_mode == "Commodities (Curated)":
    selected_ticker = st.sidebar.selectbox("Select Commodity", COMMODITIES_CURATED, index=0)

elif universe_mode == "Precious Metals (Curated)":
    selected_ticker = st.sidebar.selectbox("Select Metal / ETF", PRECIOUS_METALS_CURATED, index=0)

elif universe_mode == "Miners (Curated)":
    selected_ticker = st.sidebar.selectbox("Select Miner", MINERS_CURATED, index=0)

elif universe_mode == "Junior Miners (Curated)":
    selected_ticker = st.sidebar.selectbox("Select Junior Miner", JUNIOR_MINERS_CURATED, index=0)

else:
    selected_ticker = st.sidebar.text_input("Enter Ticker / Symbol", value="AAPL").strip().upper()

# --- Timeframe controls (preserved behavior + deterministic mapping) ---
timeframe = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d", "1wk"], index=3)

tf_to_period_interval = {
    "15m": ("60d", "15m"),
    "1h":  ("730d", "1h"),
    "4h":  ("730d", "4h"),   # safe_download will map to 1h then resample
    "1d":  ("5y", "1d"),
    "1wk": ("15y", "1wk")
}
period, interval = tf_to_period_interval[timeframe]

# --- Risk / account controls (existing used by AI + Telegram) ---
st.sidebar.subheader("💼 Risk Model")
balance = st.sidebar.number_input("Account Balance ($)", min_value=0.0, value=10000.0, step=500.0)
risk_pct = st.sidebar.number_input("Risk per Trade (%)", min_value=0.0, value=1.0, step=0.25)

# ------------------------------------------
# INDICATOR CONFIG (PRESERVED + FULL PORT CFGS)
# ------------------------------------------
st.sidebar.subheader("🧠 Apex Trend & Liquidity Master")
apex_cfg = {
    "ma_type": st.sidebar.selectbox("Baseline MA Type", ["EMA", "SMA", "HMA", "RMA", "WMA", "VWMA"], index=0),
    "len_main": st.sidebar.number_input("Main Length", min_value=2, value=55, step=1),
    "mult": st.sidebar.number_input("ATR Multiplier", min_value=0.1, value=2.0, step=0.1),
    "src": st.sidebar.selectbox("Source", ["Close", "Open", "High", "Low"], index=0),
    "use_vol": st.sidebar.toggle("Use Volume Filter", value=True),
    "use_rsi": st.sidebar.toggle("Use RSI Filter", value=True),
    "liq_len": st.sidebar.number_input("Liquidity Pivot Length", min_value=1, value=6, step=1),
    "zone_ext": st.sidebar.number_input("Zone Extend Bars", min_value=1, value=80, step=5),
    "show_liq": st.sidebar.toggle("Show Liquidity Zones", value=True),
}

st.sidebar.subheader("🧠 Nexus v8.2")
nexus_cfg = {
    "h_val": st.sidebar.number_input("Kernel Lookback (h)", min_value=2, value=20, step=1),
    "r_val": st.sidebar.number_input("Kernel Weight (r)", min_value=0.1, value=8.0, step=0.5),
    "gann_len": st.sidebar.number_input("Gann Donchian Len", min_value=2, value=10, step=1),
    "tf_trend": st.sidebar.selectbox("Kernel Macro TF", ["", "4H", "1D", "1W"], index=0),
    "a_val": st.sidebar.number_input("UT ATR Mult (a)", min_value=0.1, value=2.0, step=0.1),
    "c_period": st.sidebar.number_input("UT ATR Period (c)", min_value=2, value=12, step=1),
    "liq_len": st.sidebar.number_input("Structure Pivot Len", min_value=1, value=6, step=1),
    "show_fvg": st.sidebar.toggle("Show FVG Detection", value=True),
    "filter_doji": st.sidebar.toggle("Doji Filter", value=True),
    "strict_structure": st.sidebar.toggle("Strict BOS w/ Gann", value=False),
}

st.sidebar.subheader("⚛️ Apex Vector v4.1")
vector_cfg = {
    "eff_super": st.sidebar.number_input("Super Threshold", min_value=0.05, value=0.35, step=0.05),
    "eff_resist": st.sidebar.number_input("Resist Threshold", min_value=0.01, value=0.10, step=0.01),
    "vol_norm": st.sidebar.number_input("Vol Norm Length", min_value=2, value=20, step=1),
    "len_vec": st.sidebar.number_input("Efficiency EMA Len", min_value=2, value=14, step=1),
    "sm_type": st.sidebar.selectbox("Flux Smooth Type", ["EMA", "SMA", "RMA", "WMA", "VWMA"], index=0),
    "len_sm": st.sidebar.number_input("Flux Smooth Len", min_value=2, value=9, step=1),
    "use_vol": st.sidebar.toggle("Use Volume Factor", value=True),
    "strictness": st.sidebar.number_input("Strictness Mult", min_value=0.5, value=1.0, step=0.1),
    "show_div": st.sidebar.toggle("Show Divergences", value=True),
    "div_look": st.sidebar.number_input("Divergence Pivot Len", min_value=2, value=8, step=1),
    "show_reg": st.sidebar.toggle("Regular Divergence", value=True),
    "show_hid": st.sidebar.toggle("Hidden Divergence", value=True),
}

# ------------------------------------------
# TELEGRAM CONTROL PANEL (PRESERVED + MULTI REPORT TYPES)
# ------------------------------------------
st.sidebar.subheader("📡 Telegram Alerts")
tg_token = st.sidebar.text_input("Telegram Bot Token", type="password")
tg_chat = st.sidebar.text_input("Telegram Chat ID")
report_type = st.sidebar.selectbox("Report Type", ["QUICK_PING", "TRADE_SCALP", "TRADE_SWING", "FULL_ANALYSIS"], index=0)
upload_img = st.sidebar.file_uploader("Optional: attach image", type=["png", "jpg", "jpeg"])

# ------------------------------------------
# MAIN: DATA LOAD + COMPUTE
# ------------------------------------------
if not selected_ticker:
    st.stop()

df = safe_download(selected_ticker, period, interval)
if df is None or df.empty:
    st.error("No price data returned for this ticker/timeframe. Try a different symbol or timeframe.")
    st.stop()

# Normalize index tz for display stability
try:
    df = df.copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
except Exception:
    pass

# Run indicator engine
df_ind, apex_supply_zones, apex_demand_zones, nexus_fvgs = calc_indicators(df, apex_cfg, nexus_cfg, vector_cfg)
df_ind = calc_fear_greed_v4(df_ind)
last = df_ind.iloc[-1]

# Fundamentals (safe)
fundamentals = get_fundamentals(selected_ticker)

# SR zones snapshot (used in Telegram full print)
sr_zones = get_sr_channels(df_ind, pivot_period=10, loopback=290, max_width_pct=5, min_strength=1)

# Macro snapshot (fast)
g_groups, g_prices, g_changes = get_macro_data()

macro_lines = []
macro_lines.append("🌍 Macro Tape (5d Δ):")
for grp, items in g_groups.items():
    macro_lines.append(f"\n{grp}:")
    for nm, sym in items.items():
        chg = g_changes.get(nm, np.nan)
        if np.isfinite(_safe_float(chg)):
            macro_lines.append(f"- {nm}: {chg:+.2f}%")
        else:
            macro_lines.append(f"- {nm}: N/A")
macro_text = "\n".join(macro_lines)

# ------------------------------------------
# NEW: TradingView widgets (Symbol Info banner + chart)
# ------------------------------------------
tv_symbol = build_tradingview_symbol(
    selected_ticker,
    tv_exchange_override=tv_exchange_override,
    explicit_tv_symbol=explicit_tv_symbol
)

def tradingview_advanced_chart_html(tv_sym: str, interval_label: str):
    # TradingView interval strings: "15", "60", "240", "D", "W"
    tv_interval = {"15m": "15", "1h": "60", "4h": "240", "1d": "D", "1wk": "W"}.get(interval_label, "D")
    sym = (tv_sym or "").replace('"', '\\"')
    cfg = {
        "autosize": True,
        "symbol": sym,
        "interval": tv_interval,
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "enable_publishing": False,
        "hide_top_toolbar": False,
        "hide_legend": False,
        "allow_symbol_change": True,
        "save_image": False,
        "calendar": False,
        "studies": [],
        "support_host": "https://www.tradingview.com"
    }
    return f"""
    <div class="tradingview-widget-container" style="height:560px;width:100%;">
      <div class="tradingview-widget-container__widget" style="height:560px;width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {json.dumps(cfg)}
      </script>
    </div>
    """

# ------------------------------------------
# TOP: TradingView Ticker Banner (interactive) + Key Metrics
# ------------------------------------------
if tv_symbol:
    components.html(tradingview_symbol_info_widget_html(tv_symbol), height=140)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Price", f"${_safe_float(last['Close']):.2f}" if np.isfinite(_safe_float(last.get("Close", np.nan))) else "N/A")
m2.metric("GM Score", _fmt_int(last.get("GM_Score", 0)))
m3.metric("Omni Score", _fmt_int(last.get("Omni_Score", 0)))
m4.metric("Confidence", f"{_fmt_int(last.get('Omni_Confidence', 50))}/100")
m5.metric("RSI", f"{_safe_float(last.get('RSI', np.nan)):.1f}" if np.isfinite(_safe_float(last.get("RSI", np.nan))) else "N/A")
m6.metric("RVOL", f"{_safe_float(last.get('RVOL', np.nan)):.2f}x" if np.isfinite(_safe_float(last.get("RVOL", np.nan))) else "N/A")

st.markdown("---")

# ------------------------------------------
# TABS (PRESERVED STYLE; FULL FEATURE SET)
# ------------------------------------------
tab_overview, tab_chart, tab_structure, tab_quant, tab_macro, tab_ai, tab_telegram = st.tabs([
    "📌 Overview",
    "📈 Chart Lab",
    "🏗️ Structure / Zones",
    "🧪 Quant / Stats",
    "🌍 Macro",
    "🤖 AI Analyst",
    "📡 Telegram"
])

# ==========================================
# TAB: OVERVIEW
# ==========================================
with tab_overview:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader(f"📌 {selected_ticker} — Regime Snapshot ({timeframe})")

        apex_bias = "🐂 BULL" if int(last.get("Apex_Trend", 0)) == 1 else "🐻 BEAR" if int(last.get("Apex_Trend", 0)) == -1 else "⚪ CHOP"
        nexus_bias = "BUY" if bool(last.get("Nexus_Signal_Buy", False)) else "SELL" if bool(last.get("Nexus_Signal_Sell", False)) else "WAIT"
        vector_state = "SUPER BULL" if bool(last.get("Vector_SuperBull", False)) else \
                       "SUPER BEAR" if bool(last.get("Vector_SuperBear", False)) else \
                       "RESISTIVE" if bool(last.get("Vector_Resistive", False)) else "HEAT"

        st.write(f"**Apex Trend:** {apex_bias}")
        st.write(f"**Nexus Signal:** {nexus_bias}")
        st.write(f"**Vector State:** {vector_state}")
        st.write(f"**Squeeze:** {'💥 ON' if bool(last.get('Squeeze_On', False)) else '💤 OFF'}")
        st.write(f"**Fear & Greed Index:** {_fmt_num(last.get('FG_Index', np.nan), 1)}/100")
        st.write(f"**FOMO / PANIC Flags:** FOMO={_fmt_bool(last.get('IS_FOMO', False))} | PANIC={_fmt_bool(last.get('IS_PANIC', False))}")

        st.markdown("#### TradingView Live Chart")
        if tv_symbol:
            components.html(tradingview_advanced_chart_html(tv_symbol, timeframe), height=580)
        else:
            st.info("TradingView symbol is empty (conversion returned blank).")

    with c2:
        st.subheader("🏦 Fundamentals (if applicable)")
        if fundamentals:
            st.write(f"**Market Cap:** {fundamentals.get('Market Cap', 0):,}" if fundamentals.get("Market Cap") else "Market Cap: N/A")
            st.write(f"**P/E Ratio:** {fundamentals.get('P/E Ratio', 0)}")
            st.write(f"**Rev Growth:** {fundamentals.get('Rev Growth', 0) * 100:.2f}%")
            st.write(f"**Debt/Equity:** {fundamentals.get('Debt/Equity', 0)}")
            with st.expander("Business Summary"):
                st.write(fundamentals.get("Summary", "No Data"))
        else:
            st.info("No fundamentals available for this symbol type.")

        st.subheader("🧬 Scores")
        st.write(f"- GM Score: **{_fmt_int(last.get('GM_Score', 0))}**")
        st.write(f"- Omni Score: **{_fmt_int(last.get('Omni_Score', 0))}**")
        st.write(f"- Confidence: **{_fmt_int(last.get('Omni_Confidence', 50))}/100**")

        st.subheader("📍 Quick Levels (reference)")
        direction = "LONG" if int(last.get("Omni_Score", 0)) > 0 else "SHORT" if int(last.get("Omni_Score", 0)) < 0 else "NEUTRAL"
        lv = build_trade_levels(last, "LONG" if direction == "LONG" else "SHORT")
        st.write(f"Bias: **{direction}**")
        st.write(f"Entry(ref): **{_fmt_num(lv['entry'])}**")
        st.write(f"Stop(ref): **{_fmt_num(lv['stop'])}**")
        st.write(f"TP1/2/3: **{_fmt_num(lv['tp1'])} / {_fmt_num(lv['tp2'])} / {_fmt_num(lv['tp3'])}**")

# ==========================================
# TAB: CHART LAB (Plotly + overlays)
# ==========================================
with tab_chart:
    st.subheader("📈 Plotly Chart Lab")

    show_apex_bands = st.toggle("Show Apex ATR Bands", value=True)
    show_apex_baseline = st.toggle("Show Apex Baseline", value=True)
    show_nexus_ut = st.toggle("Show Nexus UT Stop", value=True)
    show_vwap = st.toggle("Show VWAP", value=True)
    show_ema50 = st.toggle("Show EMA50", value=True)
    show_fvgs = st.toggle("Show Nexus FVGs", value=True)
    show_liq = st.toggle("Show Apex Liquidity Zones", value=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28])

    fig.add_trace(go.Candlestick(
        x=df_ind.index, open=df_ind["Open"], high=df_ind["High"], low=df_ind["Low"], close=df_ind["Close"],
        name="Price"
    ), row=1, col=1)

    if show_apex_bands:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Apex_Upper"], mode="lines", name="Apex Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Apex_Lower"], mode="lines", name="Apex Lower"), row=1, col=1)

    if show_apex_baseline:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Apex_Base"], mode="lines", name="Apex Base"), row=1, col=1)

    if show_nexus_ut:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Nexus_UT_Stop"], mode="lines", name="Nexus UT Stop"), row=1, col=1)

    if show_vwap:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["VWAP"], mode="lines", name="VWAP"), row=1, col=1)

    if show_ema50:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["EMA_50"], mode="lines", name="EMA 50"), row=1, col=1)

    # Buy/Sell markers (Apex + Nexus)
    buy_idx = df_ind.index[df_ind["Apex_Sig_Buy"] | df_ind["Nexus_Signal_Buy"]]
    sell_idx = df_ind.index[df_ind["Apex_Sig_Sell"] | df_ind["Nexus_Signal_Sell"]]

    fig.add_trace(go.Scatter(
        x=buy_idx, y=df_ind.loc[buy_idx, "Close"],
        mode="markers", name="BUY", marker_symbol="triangle-up", marker_size=10
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sell_idx, y=df_ind.loc[sell_idx, "Close"],
        mode="markers", name="SELL", marker_symbol="triangle-down", marker_size=10
    ), row=1, col=1)

    # Vector Flux (panel 2)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Vector_Flux"], mode="lines", name="Vector Flux"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Sqz_Mom"], mode="lines", name="Squeeze Mom"), row=2, col=1)

    # FVGs (rectangles)
    if show_fvgs and nexus_fvgs:
        for g in nexus_fvgs[-80:]:
            fig.add_shape(
                type="rect",
                x0=g["x0"], x1=g["x1"],
                y0=g["y0"], y1=g["y1"],
                xref="x", yref="y",
                line_width=0,
                fillcolor="rgba(0,255,104,0.18)" if g["dir"] == "BULL" else "rgba(255,0,8,0.18)",
                row=1, col=1
            )

    # Liquidity zones (Apex supply/demand)
    if show_liq and (apex_supply_zones or apex_demand_zones):
        # Extend zones to the right by zone_ext bars using x1 = last index
        x_right = df_ind.index[-1]
        for z in apex_supply_zones:
            fig.add_shape(
                type="rect",
                x0=z["idx"], x1=x_right,
                y0=z["bot"], y1=z["top"],
                line_width=0,
                fillcolor="rgba(255, 0, 0, 0.12)",
                row=1, col=1
            )
        for z in apex_demand_zones:
            fig.add_shape(
                type="rect",
                x0=z["idx"], x1=x_right,
                y0=z["bot"], y1=z["top"],
                line_width=0,
                fillcolor="rgba(0, 255, 0, 0.12)",
                row=1, col=1
            )

    fig.update_layout(height=850, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB: STRUCTURE / ZONES
# ==========================================
with tab_structure:
    st.subheader("🏗️ Structure / Zones")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### SR Channels (Algorithmic)")
        if sr_zones:
            zdf = pd.DataFrame(sr_zones)
            st.dataframe(zdf, use_container_width=True, hide_index=True)
        else:
            st.info("No SR zones detected for the current lookback settings.")

        st.markdown("### SMC Snapshot (Legacy Function)")
        smc = calculate_smc(df_ind, swing_length=5)
        st.write(f"Structures: {len(smc.get('structures', []))} | Order Blocks: {len(smc.get('order_blocks', []))} | FVGs: {len(smc.get('fvgs', []))}")

    with c2:
        st.markdown("### Liquidity Zones (Apex)")
        st.write(f"Supply Zones (active): **{len(apex_supply_zones)}**")
        st.write(f"Demand Zones (active): **{len(apex_demand_zones)}**")

        if apex_supply_zones:
            st.dataframe(pd.DataFrame(apex_supply_zones), use_container_width=True, hide_index=True)
        if apex_demand_zones:
            st.dataframe(pd.DataFrame(apex_demand_zones), use_container_width=True, hide_index=True)

        st.markdown("### Nexus FVGs")
        if nexus_fvgs:
            fvg_df = pd.DataFrame(nexus_fvgs[-50:])
            st.dataframe(fvg_df, use_container_width=True, hide_index=True)
        else:
            st.info("No FVGs detected.")

# ==========================================
# TAB: QUANT / STATS
# ==========================================
with tab_quant:
    st.subheader("🧪 Quant / Stats")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### Correlations (Macro Basket)")
        try:
            corr = calc_correlations(selected_ticker, lookback_days=180)
            st.dataframe(corr.to_frame("Correlation"), use_container_width=True)
        except Exception as e:
            st.info(f"Correlation unavailable: {e}")

        st.markdown("### Multi-Timeframe Trend DNA")
        mtf = calc_mtf_trend(selected_ticker)
        st.dataframe(mtf, use_container_width=True)

    with c2:
        st.markdown("### Monte Carlo (30d)")
        sims = st.slider("Simulations", 200, 3000, 1000, 100)
        days = st.slider("Days", 10, 90, 30, 5)
        paths = run_monte_carlo(df_ind, days=days, simulations=sims)
        # Plot mean + percentile bands (no explicit colors requested)
        mean_path = paths.mean(axis=1)
        p10 = np.percentile(paths, 10, axis=1)
        p90 = np.percentile(paths, 90, axis=1)

        x = list(range(days))
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(x=x, y=mean_path, mode="lines", name="Mean"))
        fig_mc.add_trace(go.Scatter(x=x, y=p10, mode="lines", name="P10"))
        fig_mc.add_trace(go.Scatter(x=x, y=p90, mode="lines", name="P90"))
        fig_mc.update_layout(height=420, template="plotly_dark", xaxis_title="Day", yaxis_title="Price")
        st.plotly_chart(fig_mc, use_container_width=True)

        st.markdown("### Intraday DNA (Hourly)")
        intr = calc_intraday_dna(selected_ticker)
        if intr is None:
            st.info("Intraday DNA unavailable for this symbol/timeframe.")
        else:
            st.dataframe(intr, use_container_width=True)

# ==========================================
# TAB: MACRO
# ==========================================
with tab_macro:
    st.subheader("🌍 Macro Dashboard")
    perf = get_global_performance()
    if perf is not None and len(perf):
        st.markdown("### Global Basket (last day % change)")
        st.bar_chart(perf)
    else:
        st.info("Global performance unavailable.")

    st.markdown("### Full Macro Tape (grouped)")
    st.code(macro_text)

# ==========================================
# TAB: AI ANALYST
# ==========================================
with tab_ai:
    st.subheader("🤖 AI Analyst")
    st.caption("Uses your OpenAI API key if provided in the sidebar (or st.secrets).")

    if st.button("Run AI Analyst"):
        ai_text = ask_ai_analyst(df_ind, selected_ticker, fundamentals, balance, risk_pct, timeframe)
        st.session_state["ai_verdict_latest"] = ai_text

    ai_latest = st.session_state.get("ai_verdict_latest", "")
    if ai_latest:
        st.markdown(ai_latest)
    else:
        st.info("Click 'Run AI Analyst' to generate the AI summary.")

# ==========================================
# TAB: TELEGRAM
# ==========================================
with tab_telegram:
    st.subheader("📡 Telegram Signal Console")

    st.markdown("### Preview Report")
    # Use existing cached AI verdict if any; otherwise keep empty
    ai_verdict = st.session_state.get("ai_verdict_latest", "AI verdict not generated yet.")
    msgs = format_signal_report(
        ticker=selected_ticker,
        interval=timeframe,
        df=df_ind,
        last_row=last,
        macro_text=macro_text,
        ai_verdict=ai_verdict,
        balance=balance,
        risk_pct=risk_pct,
        report_type=report_type,
        sr_zones=sr_zones
    )
    for i, m in enumerate(msgs, start=1):
        with st.expander(f"Message {i}"):
            st.code(m)

    st.markdown("### Send")
    if st.button("🚀 Send to Telegram"):
        try:
            send_telegram_messages(tg_token, tg_chat, msgs, uploaded_file=upload_img)
            st.success("Sent.")
        except Exception as e:
            st.error(f"Telegram error: {e}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("👁️ DarkPool Titan Terminal — Internal intelligence display. Not financial advice.")

