import time
import math
import sqlite3
import json
import re
from typing import Dict, Optional, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from datetime import datetime, timezone

# =============================================================================
# BRANDING
# =============================================================================
APP_NAME = "Signals-Mobile-v1"
APP_SUBTITLE = "Mobile • Technical, SMC & Vector Analytics"
APP_VERSION = "v19.0 (Optimized & Global)"

TG_PREFIX = "MARKET SIGNAL"
REPORT_PREFIX = "MARKET REPORT"

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    page_icon="📶",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# SESSION STATE & REFRESH LOGIC
# =============================================================================
if "refresh_nonce" not in st.session_state:
    st.session_state.refresh_nonce = 0
if "symbol_input" not in st.session_state:
    st.session_state.symbol_input = "BTC"

def hard_refresh():
    """Reliable refresh: bump nonce + clear caches + FORCE RERUN."""
    st.session_state.refresh_nonce += 1
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass
    st.rerun()

def update_symbol_from_dropdown():
    """Callback to update text input immediately when dropdown changes."""
    st.session_state.symbol_input = st.session_state.quick_ticker_select
    hard_refresh()

def set_symbol_and_refresh(base: str):
    st.session_state.symbol_input = (base or "BTC").strip().upper()
    hard_refresh()

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main { background-color: #0b0c10; }

    div[data-testid="metric-container"] {
        background: rgba(31, 40, 51, 0.9);
        border: 1px solid #45a29e;
        padding: 10px;
        border-radius: 12px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }

    div[data-testid="metric-container"] label {
        font-size: 12px !important;
        color: #c5c6c7 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 20px !important;
        color: #66fcf1 !important;
    }

    h1, h2, h3 {
        font-family: 'Roboto Mono', monospace;
        color: #c5c6c7;
        word-wrap: break-word;
    }

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
        border-color: #66fcf1;
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

    code, pre {
        background: rgba(0,0,0,0.25) !important;
        border: 1px solid rgba(69,162,158,0.35) !important;
        border-radius: 10px !important;
        padding: 10px !important;
        color: #c5c6c7 !important;
    }
    
    /* Toast Styling */
    div[data-testid="stToast"] {
        background-color: #1f2833;
        border: 1px solid #45a29e;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================
# Priorities: Global -> US -> Fallback
API_URLS = [
    "https://api.binance.com/api/v3", 
    "https://api.binance.us/api/v3"
]
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

POPULAR_BASES = [
    "BTC","ETH","SOL","XRP","BNB","ADA","DOGE","LINK","AVAX","DOT",
    "MATIC","LTC","BCH","ATOM","XLM","ETC","AAVE","UNI","SHIB","TRX",
    "FIL","NEAR","ICP","ARB","OP","SUI","APT","INJ","TIA","RNDR", "PEPE", "WIF"
]

# =============================================================================
# SQLITE STATE
# =============================================================================
DB_PATH = "signals_state.db"

def db_init():
    try:
        con = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sent_state (
                key TEXT PRIMARY KEY,
                last_sig_key TEXT,
                last_sent_ts TEXT
            )
        """)
        con.commit()
        return con
    except Exception:
        return None

def db_get_last(con, key: str) -> Optional[str]:
    if con is None: return None
    try:
        cur = con.cursor()
        cur.execute("SELECT last_sig_key FROM sent_state WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None
    except Exception: return None

def db_set_last(con, key: str, last_sig_key: str):
    if con is None: return
    try:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO sent_state(key, last_sig_key, last_sent_ts)
            VALUES(?,?,?)
            ON CONFLICT(key) DO UPDATE SET
                last_sig_key=excluded.last_sig_key,
                last_sent_ts=excluded.last_sent_ts
        """, (key, last_sig_key, datetime.now(timezone.utc).isoformat()))
        con.commit()
    except Exception: pass

DB_CON = db_init()

# =============================================================================
# TICKER UNIVERSE (OPTIMIZED)
# =============================================================================
@st.cache_data(ttl=3600)
def get_all_tickers(nonce: int = 0) -> List[str]:
    """Fetches tickers from Global, falls back to US."""
    bases = set()
    
    # Try URLs in order
    for base_url in API_URLS:
        try:
            r = requests.get(f"{base_url}/exchangeInfo", headers=HEADERS, timeout=5)
            if r.status_code == 200:
                js = r.json()
                for s in js.get("symbols", []):
                    if s.get("status") != "TRADING": continue
                    if s.get("quoteAsset") != "USDT": continue
                    base = s.get("baseAsset")
                    if base: bases.add(base.upper())
                
                # If we successfully got data, stop trying other URLs
                if bases:
                    break
        except Exception:
            continue
            
    # Always include popular bases even if API fails (Fallback)
    for p in POPULAR_BASES:
        bases.add(p)
        
    return sorted(bases)

# =============================================================================
# TRADINGVIEW WIDGETS
# =============================================================================
def render_ticker_tape(selected_symbol: str, nonce: int):
    base = selected_symbol.replace("USDT", "")
    tape_bases = ["BTC", "ETH", "SOL"] + POPULAR_BASES[:10]
    if base and base not in tape_bases:
        tape_bases.insert(0, base)

    # Unique list while preserving order
    seen = set()
    tape_bases = [x for x in tape_bases if not (x in seen or seen.add(x))]

    symbols_json = json.dumps(
        [{"proName": f"BINANCE:{b}USDT", "title": b} for b in tape_bases],
        separators=(",", ":")
    )

    components.html(
        f"""
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
          {{
            "symbols": {symbols_json},
            "showSymbolLogo": true,
            "colorTheme": "dark",
            "isTransparent": true,
            "displayMode": "adaptive",
            "locale": "en"
          }}
          </script>
        </div>
        """,
        height=50
    )

def tv_chart_link(symbol: str, tf: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}"

# =============================================================================
# HEADER + CLOCK
# =============================================================================
st.title(f"📶 {APP_NAME}")
st.caption(f"{APP_VERSION} | {APP_SUBTITLE}")

components.html(
    f"""
    <div id="live_clock"></div>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@700&display=swap');
        body {{ margin: 0; background-color: transparent; text-align: center; }}
        #live_clock {{
            font-family: 'Roboto Mono', monospace;
            font-size: 16px;
            color: #39ff14;
            text-shadow: 0 0 10px rgba(57, 255, 20, 0.8);
            font-weight: 800;
            padding: 5px;
        }}
    </style>
    <script>
    function updateTime() {{
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-GB', {{ timeZone: 'UTC' }});
        document.getElementById('live_clock').innerHTML = 'UTC: ' + timeString;
    }}
    setInterval(updateTime, 1000);
    updateTime();
    </script>
    """,
    height=30
)

# =============================================================================
# TELEGRAM HELPERS
# =============================================================================
def telegram_escape_md_v2(text: str) -> str:
    if text is None: return ""
    return re.sub(r'([_*[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

def html_to_text(html: str) -> str:
    if not html: return ""
    html = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</(div|p|h1|h2|h3)>", "\n", html)
    text = re.sub(r"(?s)<.*?>", "", html)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def parse_chat_list(raw: str) -> List[str]:
    if not raw: return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]

def tg_get_me(token: str) -> Tuple[bool, str]:
    if not token: return False, "Missing token"
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=5)
        if r.status_code != 200: return False, f"HTTP {r.status_code}"
        js = r.json()
        if not js.get("ok"): return False, str(js)
        u = js.get("result", {})
        return True, f"Connected as @{u.get('username','?')} (id={u.get('id','?')})"
    except Exception as e: return False, f"Error: {e}"

def tg_send_message(token: str, chat_id: str, text: str, mode: str = "MarkdownV2", disable_preview: bool = True) -> Tuple[bool, str]:
    if not token or not chat_id or not text: return False, "Missing params"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text[:4096], 
        "disable_web_page_preview": bool(disable_preview),
    }
    if mode == "MarkdownV2": payload["parse_mode"] = "MarkdownV2"
    try:
        r = requests.post(url, json=payload, timeout=8)
        if r.status_code != 200: return False, f"HTTP {r.status_code}: {r.text[:200]}"
        js = r.json()
        if js.get("ok"): return True, "Sent"
        return False, str(js)[:200]
    except Exception as e: return False, f"Error: {e}"

def tg_broadcast(token: str, chats: List[str], text: str, mode: str, disable_preview: bool = True) -> Tuple[bool, List[Tuple[str, bool, str]]]:
    results = []
    ok_any = False
    for c in chats:
        ok, detail = tg_send_message(token, c, text, mode=mode, disable_preview=disable_preview)
        results.append((c, ok, detail))
        ok_any = ok_any or ok
        time.sleep(0.1) 
    return ok_any, results

# =============================================================================
# TA HELPERS & ENGINES (OPTIMIZED & CACHED)
# =============================================================================
def _ema(s: pd.Series, length: int) -> pd.Series: return s.ewm(span=length, adjust=False).mean()
def _sma(s: pd.Series, length: int) -> pd.Series: return s.rolling(length).mean()
def _rma(s: pd.Series, length: int) -> pd.Series: return s.ewm(alpha=1.0/length, adjust=False).mean()
def _wma(series: pd.Series, length: int) -> pd.Series:
    w = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
def _vwma(price: pd.Series, vol: pd.Series, length: int) -> pd.Series:
    pv = (price * vol).rolling(length).sum()
    vv = vol.rolling(length).sum().replace(0, np.nan)
    return pv / vv

def calculate_hma(series: pd.Series, length: int, mode: str) -> pd.Series:
    n = max(2, int(length))
    half = max(1, n // 2)
    sqrt_n = max(1, int(math.sqrt(n)))
    if mode == "True HMA (WMA-based)":
        wma_full = _wma(series, n)
        wma_half = _wma(series, half)
        diff = 2 * wma_half - wma_full
        return _wma(diff, sqrt_n)
    else:
        # Original SMA based
        wma_f = series.rolling(n).mean()
        wma_h = series.rolling(half).mean()
        diff = 2 * wma_h - wma_f
        return diff.rolling(sqrt_n).mean()

def pine_atr(df: pd.DataFrame, length: int) -> pd.Series:
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return _rma(tr, length)

def pine_dmi_adx(df: pd.DataFrame, length: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = pine_atr(df, length).replace(0, np.nan)
    plus_di = 100 * (_rma(pd.Series(plus_dm, index=df.index), length) / atr)
    minus_di = 100 * (_rma(pd.Series(minus_dm, index=df.index), length) / atr)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    adx = _rma(dx.fillna(0), length)
    return plus_di.fillna(0), minus_di.fillna(0), adx.fillna(0)

def pine_pivothigh(series: pd.Series, left: int, right: int) -> pd.Series:
    # Optimized using rolling max + shift
    # Not perfect replication of Pine's pivot but significantly faster for pandas
    # Reverting to loop for 100% accuracy matching original code request logic
    left = int(left); right = int(right)
    n = len(series)
    out = np.full(n, np.nan, dtype=float)
    arr = series.values.astype(float)
    for i in range(left, n - right):
        window = arr[i - left:i + right + 1]
        if np.isnan(window).any(): continue
        if arr[i] == np.max(window): out[i] = arr[i]
    return pd.Series(out, index=series.index)

def pine_pivotlow(series: pd.Series, left: int, right: int) -> pd.Series:
    left = int(left); right = int(right)
    n = len(series)
    out = np.full(n, np.nan, dtype=float)
    arr = series.values.astype(float)
    for i in range(left, n - right):
        window = arr[i - left:i + right + 1]
        if np.isnan(window).any(): continue
        if arr[i] == np.min(window): out[i] = arr[i]
    return pd.Series(out, index=series.index)

def calculate_fibonacci(df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
    recent = df.iloc[-lookback:]
    h, l = float(recent["high"].max()), float(recent["low"].min())
    d = h - l
    return {"fib_382": h - (d*0.382), "fib_500": h - (d*0.500), "fib_618": h - (d*0.618), "high": h, "low": l}

def calculate_fear_greed_index(df: pd.DataFrame) -> int:
    try:
        tmp = df.copy()
        tmp["log_ret"] = np.log(tmp["close"] / tmp["close"].shift(1))
        vol30 = tmp["log_ret"].rolling(30).std().iloc[-1]
        vol90 = tmp["log_ret"].rolling(90).std().iloc[-1]
        vol_score = 50 - ((vol30 - vol90) / vol90) * 100 if vol90 else 50
        
        rsi = float(tmp["rsi"].iloc[-1]) if "rsi" in tmp.columns else 50.0
        sma_50 = tmp["close"].rolling(50).mean().iloc[-1]
        trend_score = 50 + ((tmp["close"].iloc[-1] - sma_50) / sma_50 * 1000) if sma_50 else 50
        
        fg = (max(0,min(100,vol_score)) * 0.3) + (rsi * 0.4) + (max(0,min(100,trend_score)) * 0.3)
        return int(max(0, min(100, fg)))
    except: return 50

# DATA FETCH
@st.cache_data(ttl=10)
def get_klines(symbol_bin: str, interval: str, limit: int, nonce: int = 0) -> pd.DataFrame:
    # Try Global then US
    for base_url in API_URLS:
        try:
            r = requests.get(
                f"{base_url}/klines",
                params={"symbol": symbol_bin, "interval": interval, "limit": int(limit)},
                headers=HEADERS,
                timeout=5,
            )
            if r.status_code == 200:
                df = pd.DataFrame(r.json(), columns=["t", "o", "h", "l", "c", "v", "T", "q", "n", "V", "Q", "B"])
                df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
                df[["open", "high", "low", "close", "volume"]] = df[["o", "h", "l", "c", "v"]].astype(float)
                return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception:
            continue
    return pd.DataFrame()

# CORE ENGINE
@st.cache_data(ttl=60) # Cached for performance
def run_core_engines(df: pd.DataFrame, amp, dev, hma_l, tp1, tp2, tp3, mf_l, vol_l, gann_l, hma_mode, rvol_threshold, cooldown_bars) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy().reset_index(drop=True)
    
    # TR/ATR
    df["atr"] = pine_atr(df, 14)

    # HMA & VWAP
    df["hma"] = calculate_hma(df["close"], int(hma_l), hma_mode)
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap"] = (df["tp"] * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)

    # Squeeze
    bb_basis = df["close"].rolling(20).mean()
    bb_dev = df["close"].rolling(20).std() * 2.0
    kc_dev = df["atr"] * 1.5
    df["in_squeeze"] = ((bb_basis - bb_dev) > (bb_basis - kc_dev)) & ((bb_basis + bb_dev) < (bb_basis + kc_dev))

    # RSI & RVOL
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    
    vol_mean = df["volume"].rolling(int(vol_l)).mean().replace(0, np.nan)
    df["rvol"] = (df["volume"] / vol_mean).fillna(0)

    # Money Flow
    mf_raw = ((df["rsi"] - 50) * (df["volume"] / df["volume"].rolling(int(mf_l)).mean().replace(0, np.nan))).fillna(0)
    df["money_flow"] = mf_raw.ewm(span=3).mean()

    # Trend flip (Titan)
    df["ll"] = df["low"].rolling(int(amp)).min()
    df["hh"] = df["high"].rolling(int(amp)).max()
    trend = np.zeros(len(df))
    stop = np.full(len(df), np.nan)
    curr_t = 0; curr_s = np.nan
    
    for i in range(int(amp), len(df)):
        c = float(df.at[i, "close"])
        d = float(df.at[i, "atr"]) * float(dev)
        if curr_t == 0:
            s = float(df.at[i, "ll"]) + d
            curr_s = max(curr_s, s) if not np.isnan(curr_s) else s
            if c < curr_s:
                curr_t = 1; curr_s = float(df.at[i, "hh"]) - d
        else:
            s = float(df.at[i, "hh"]) - d
            curr_s = min(curr_s, s) if not np.isnan(curr_s) else s
            if c > curr_s:
                curr_t = 0; curr_s = float(df.at[i, "ll"]) + d
        trend[i] = curr_t
        stop[i] = curr_s

    df["is_bull"] = trend == 0
    df["entry_stop"] = stop
    
    # Signals
    cond_buy = (df["is_bull"]) & (~df["is_bull"].shift(1).fillna(False)) & (df["rvol"] > float(rvol_threshold))
    cond_sell = (~df["is_bull"]) & (df["is_bull"].shift(1).fillna(True)) & (df["rvol"] > float(rvol_threshold))
    
    # Cooldown Logic
    if int(cooldown_bars) > 0:
        cd = int(cooldown_bars)
        last_i = -9999
        buy = np.zeros(len(df), dtype=bool); sell = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            if i - last_i < cd: continue
            if cond_buy.iloc[i]:
                buy[i] = True; last_i = i
            elif cond_sell.iloc[i]:
                sell[i] = True; last_i = i
        df["buy"] = buy; df["sell"] = sell
    else:
        df["buy"] = cond_buy; df["sell"] = cond_sell

    # Targets
    df["sig_id"] = (df["buy"] | df["sell"]).cumsum()
    entry = df.groupby("sig_id")["close"].ffill()
    stop_val = df.groupby("sig_id")["entry_stop"].ffill()
    risk = abs(entry - stop_val).replace(0, np.nan)
    df["tp1"] = np.where(df["is_bull"], entry + (risk*tp1), entry - (risk*tp1))
    df["tp2"] = np.where(df["is_bull"], entry + (risk*tp2), entry - (risk*tp2))
    df["tp3"] = np.where(df["is_bull"], entry + (risk*tp3), entry - (risk*tp3))

    # Apex & Gann
    apex_base = calculate_hma(df["close"], 55, hma_mode)
    df["apex_upper"] = apex_base + (df["atr"] * 1.5)
    df["apex_lower"] = apex_base - (df["atr"] * 1.5)
    
    # Vectorized Apex Trend
    apex_t = np.select(
        [df["close"] > df["apex_upper"], df["close"] < df["apex_lower"]],
        [1, -1],
        default=0
    )
    # Forward fill 0s manually since select default is 0 but we want hold state
    # (Actually pandas loop or complex masking needed for hold state, simplifying to loop for accuracy)
    at_curr = 0
    at_res = []
    for i in range(len(df)):
        if df.at[i, "close"] > df.at[i, "apex_upper"]: at_curr=1
        elif df.at[i, "close"] < df.at[i, "apex_lower"]: at_curr=-1
        at_res.append(at_curr)
    df["apex_trend"] = at_res

    # Gann
    sma_h = df["high"].rolling(int(gann_l)).mean()
    sma_l = df["low"].rolling(int(gann_l)).mean()
    g_trend = np.full(len(df), np.nan); g_act = np.full(len(df), np.nan)
    ct = 1; ca = sma_l.iloc[int(gann_l)] if len(sma_l)>int(gann_l) else np.nan
    for i in range(int(gann_l), len(df)):
        c = df.at[i,"close"]
        hm = sma_h.iloc[i]; lm = sma_l.iloc[i]
        prev_a = g_act[i-1] if i>0 and np.isfinite(g_act[i-1]) else ca
        if ct == 1:
            if np.isfinite(prev_a) and c < prev_a: ct = -1; ca = hm
            else: ca = lm
        else:
            if np.isfinite(prev_a) and c > prev_a: ct = 1; ca = lm
            else: ca = hm
        g_trend[i] = ct; g_act[i] = ca
    df["gann_trend"] = g_trend; df["gann_act"] = g_act

    return df

# SMC ENGINE
@st.cache_data(ttl=60)
def compute_apex_master_smc(df: pd.DataFrame, ma_type, len_main, mult, liq_len, sd_ext, dmi_len, vol_len, adx_min, show_sd, show_bos, show_ob, show_fvg, fvg_mit, hma_mode_for_app) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy().reset_index(drop=True)
    n = len(df)
    drawings = {"sd_zones": [], "bos_lines": [], "choch_lines": [], "ob_zones": [], "fvg_zones": []}
    
    src = df["close"]
    if ma_type == "SMA": baseline = _sma(src, int(len_main))
    elif ma_type == "EMA": baseline = _ema(src, int(len_main))
    elif ma_type == "RMA": baseline = _rma(src, int(len_main))
    else: baseline = calculate_hma(src, int(len_main), hma_mode_for_app)
    
    atr_main = pine_atr(df, int(len_main))
    upper = baseline + (atr_main * float(mult))
    lower = baseline - (atr_main * float(mult))
    df["am_upper"] = upper; df["am_lower"] = lower

    am_trend = np.zeros(n, dtype=int)
    ts = 0
    for i in range(n):
        c = df.at[i,"close"]
        u = upper.iloc[i]; l = lower.iloc[i]
        if np.isfinite(u) and c > u: ts = 1
        elif np.isfinite(l) and c < l: ts = -1
        am_trend[i] = ts
    df["am_trend"] = am_trend

    _, _, adx = pine_dmi_adx(df, int(dmi_len))
    df["am_adx_ok"] = adx > float(adx_min)
    
    # TCI
    ap = (df["high"]+df["low"]+df["close"])/3
    esa = _ema(ap, 10)
    d = _ema((ap-esa).abs(), 10)
    ci = (ap-esa)/(0.015*d.replace(0,np.nan))
    tci = _ema(ci.fillna(0), 21)
    df["am_mom_buy"] = tci < 60
    df["am_mom_sell"] = tci > -60
    df["am_vol_ok"] = df["volume"] > _sma(df["volume"], int(vol_len))

    prev_t = pd.Series(am_trend).shift(1).fillna(0).astype(int)
    df["am_sig_buy"] = (am_trend==1) & (prev_t!=1) & df["am_vol_ok"] & df["am_mom_buy"] & df["am_adx_ok"]
    df["am_sig_sell"] = (am_trend==-1) & (prev_t!=-1) & df["am_vol_ok"] & df["am_mom_sell"] & df["am_adx_ok"]

    # Trailing Stop
    trail_atr = pine_atr(df, 14) * 2.0
    am_trail = np.full(n, np.nan, dtype=float)
    t_stop = np.nan
    for i in range(n):
        t = am_trend[i]
        c = df.at[i, "close"]
        ta = trail_atr.iloc[i]
        if not np.isfinite(ta): 
            am_trail[i] = t_stop; continue
        
        if t == 1:
            ns = c - ta
            t_stop = max(t_stop, ns) if not np.isnan(t_stop) else ns
            if i>0 and prev_t[i]==-1: t_stop = ns
        elif t == -1:
            ns = c + ta
            t_stop = min(t_stop, ns) if not np.isnan(t_stop) else ns
            if i>0 and prev_t[i]==1: t_stop = ns
        am_trail[i] = t_stop
    df["am_trail_stop"] = am_trail

    # S/D, BOS, OB, FVG
    ph = pine_pivothigh(df["high"], liq_len, liq_len)
    pl = pine_pivotlow(df["low"], liq_len, liq_len)
    df["am_ph"] = ph; df["am_pl"] = pl
    
    last_ph = np.nan; last_pl = np.nan
    lh = np.nan; hl = np.nan
    ob_zones = []; fvg_zones = []
    
    atr14 = pine_atr(df, 14)

    def cap(l, x):
        if len(l)>x: del l[0:len(l)-x]

    for i in range(n):
        # S/D
        if np.isfinite(ph.iloc[i]):
            last_ph = ph.iloc[i]
            if am_trend[i] == -1: lh = last_ph
            if show_sd:
                left = i - int(liq_len)
                if left>=0:
                    drawings["sd_zones"].append({
                        "type":"supply","i0":left,"i1":min(n-1, left+int(sd_ext)),
                        "top":df.at[left,"high"],"bot":max(df.at[left,"open"],df.at[left,"close"])
                    })
                    cap(drawings["sd_zones"], 10)
        
        if np.isfinite(pl.iloc[i]):
            last_pl = pl.iloc[i]
            if am_trend[i] == 1: hl = last_pl
            if show_sd:
                left = i - int(liq_len)
                if left>=0:
                    drawings["sd_zones"].append({
                        "type":"demand","i0":left,"i1":min(n-1, left+int(sd_ext)),
                        "bot":df.at[left,"low"],"top":min(df.at[left,"open"],df.at[left,"close"])
                    })
                    cap(drawings["sd_zones"], 10)

        c = df.at[i,"close"]; pc = df.at[i-1,"close"] if i>0 else c
        
        # BOS/CHoCH
        if show_bos:
            if am_trend[i]==1 and np.isfinite(last_ph) and pc <= last_ph and c > last_ph:
                drawings["bos_lines"].append({"i":i,"price":last_ph,"side":"bull"}); cap(drawings["bos_lines"],30)
            if am_trend[i]==-1 and np.isfinite(last_pl) and pc >= last_pl and c < last_pl:
                drawings["bos_lines"].append({"i":i,"price":last_pl,"side":"bear"}); cap(drawings["bos_lines"],30)
            if am_trend[i]==-1 and np.isfinite(lh) and pc <= lh and c > lh:
                drawings["choch_lines"].append({"i":i,"price":lh,"side":"bull"}); cap(drawings["choch_lines"],30)
                hl = df.at[i,"low"]
            if am_trend[i]==1 and np.isfinite(hl) and pc >= hl and c < hl:
                drawings["choch_lines"].append({"i":i,"price":hl,"side":"bear"}); cap(drawings["choch_lines"],30)
                lh = df.at[i,"high"]

        # OB
        if show_ob:
            if am_trend[i]==1 and np.isfinite(last_ph) and pc <= last_ph and c > last_ph:
                for j in range(1,21):
                    k = i-j
                    if k<=0: break
                    if df.at[k,"close"] < df.at[k,"open"]:
                        ob_zones.append({"type":"ob_bull","i0":k,"i1":min(n-1, i+int(sd_ext)), "top":df.at[k,"high"], "bot":df.at[k,"low"]})
                        cap(ob_zones, 5); break
            if am_trend[i]==-1 and np.isfinite(last_pl) and pc >= last_pl and c < last_pl:
                for j in range(1,21):
                    k = i-j
                    if k<=0: break
                    if df.at[k,"close"] > df.at[k,"open"]:
                        ob_zones.append({"type":"ob_bear","i0":k,"i1":min(n-1, i+int(sd_ext)), "top":df.at[k,"high"], "bot":df.at[k,"low"]})
                        cap(ob_zones, 5); break
        
        # FVG
        if show_fvg and i>=2:
            atrc = atr14.iloc[i]
            if np.isfinite(atrc):
                if (df.at[i,"low"] > df.at[i-2,"high"]) and ((df.at[i,"low"] - df.at[i-2,"high"]) > atrc*0.5):
                    fvg_zones.append({"type":"fvg_bull","i0":i-2,"i1":min(n-1, i+int(sd_ext)),"top":df.at[i,"low"],"bot":df.at[i-2,"high"]})
                if (df.at[i,"high"] < df.at[i-2,"low"]) and ((df.at[i-2,"low"] - df.at[i,"high"]) > atrc*0.5):
                    fvg_zones.append({"type":"fvg_bear","i0":i-2,"i1":min(n-1, i+int(sd_ext)),"top":df.at[i-2,"low"],"bot":df.at[i,"high"]})
                cap(fvg_zones, 10)
        
        # Mitigation Logic
        if fvg_mit:
            # OB
            kept=[]
            for z in ob_zones:
                bt, bb = z["top"], z["bot"]
                if (c > bt and bt > bb) or (c < bb and bb < bt): continue
                z["i1"] = min(n-1, i+5)
                kept.append(z)
            ob_zones = kept
            # FVG
            kept=[]
            for z in fvg_zones:
                bt, bb = z["top"], z["bot"]
                if (c > bt and bt > bb) or (c < bb and bb < bt): continue
                z["i1"] = min(n-1, i+5)
                kept.append(z)
            fvg_zones = kept

    drawings["ob_zones"] = ob_zones
    drawings["fvg_zones"] = fvg_zones
    return df, drawings

# VECTOR ENGINE
@st.cache_data(ttl=60)
def compute_apex_vector(df: pd.DataFrame, eff_super, eff_resist, vol_norm, len_vec, sm_type, len_sm, use_vol, strictness, show_div, div_look, show_reg, show_hid) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    
    range_abs = (df["high"] - df["low"]).abs()
    body_abs = (df["close"] - df["open"]).abs()
    eff = np.where(range_abs==0, 0.0, body_abs/range_abs)
    df["vec_efficiency"] = _ema(pd.Series(eff), int(len_vec))
    
    v_avg = _sma(df["volume"], int(vol_norm))
    v_fact = np.where(v_avg==0, 1.0, df["volume"]/v_avg) if use_vol else 1.0
    
    vec_raw = np.sign(df["close"]-df["open"]) * df["vec_efficiency"] * v_fact
    
    if sm_type=="SMA": flux = _sma(vec_raw, int(len_sm))
    elif sm_type=="RMA": flux = _rma(vec_raw, int(len_sm))
    elif sm_type=="VWMA": flux = _vwma(vec_raw, df["volume"], int(len_sm))
    else: flux = _ema(vec_raw, int(len_sm)) # EMA default
    
    df["vec_flux"] = flux.fillna(0)
    
    th_s = float(eff_super)*float(strictness)
    th_r = float(eff_resist)*float(strictness)
    df["vec_th_super"] = th_s
    
    df["vec_is_super_bull"] = df["vec_flux"] > th_s
    df["vec_is_super_bear"] = df["vec_flux"] < -th_s
    df["vec_is_resistive"] = df["vec_flux"].abs() < th_r
    
    # Divergences
    ph = pine_pivothigh(df["vec_flux"], div_look, div_look)
    pl = pine_pivotlow(df["vec_flux"], div_look, div_look)
    
    db_reg = np.zeros(len(df), bool); db_hid = np.zeros(len(df), bool)
    ds_reg = np.zeros(len(df), bool); ds_hid = np.zeros(len(df), bool)
    
    pp_flux = np.nan; pp_price = np.nan # pivot low
    ph_flux = np.nan; ph_price = np.nan # pivot high
    
    for i in range(len(df)):
        if np.isfinite(pl.iloc[i]):
            pr = df.at[i,"low"]
            if show_div and np.isfinite(pp_flux):
                if pr < pp_price and pl.iloc[i] > pp_flux: db_reg[i] = bool(show_reg)
                if pr > pp_price and pl.iloc[i] < pp_flux: db_hid[i] = bool(show_hid)
            pp_flux = pl.iloc[i]; pp_price = pr
            
        if np.isfinite(ph.iloc[i]):
            pr = df.at[i,"high"]
            if show_div and np.isfinite(ph_flux):
                if pr > ph_price and ph.iloc[i] < ph_flux: ds_reg[i] = bool(show_reg)
                if pr < ph_price and ph.iloc[i] > ph_flux: ds_hid[i] = bool(show_hid)
            ph_flux = ph.iloc[i]; ph_price = pr

    df["vec_div_bull_reg"] = db_reg
    df["vec_div_bull_hid"] = db_hid
    df["vec_div_bear_reg"] = ds_reg
    df["vec_div_bear_hid"] = ds_hid
    
    return df

# =============================================================================
# REPORT & ALERT BUILDERS
# =============================================================================
def generate_mobile_report_html(row, symbol, tf, fibs, fg, stop, am_evt, ob_c, fvg_c, v_state, v_flux, v_div) -> str:
    is_bull = bool(row["is_bull"])
    direction = "LONG 🐂" if is_bull else "SHORT 🐻"
    
    # Confidence Score
    score = 0
    ts = 1 if is_bull else -1
    as_ = int(row["apex_trend"]) if np.isfinite(row["apex_trend"]) else 0
    gs = int(row["gann_trend"]) if np.isfinite(row["gann_trend"]) else 0
    if ts == as_: score += 1
    if ts == gs: score += 1
    if (v_state.startswith("SUPER (BULL)") and is_bull) or (v_state.startswith("SUPER (BEAR)") and not is_bull): score += 1
    
    conf = "MAX 🔥" if score >=3 else ("HIGH" if score==2 else ("MED" if score==1 else "LOW"))
    vol_desc = "IGNITION 🚀" if float(row["rvol"]) > 2.0 else "Normal"
    sqz = "⚠️ SQUEEZE ACTIVE" if bool(row["in_squeeze"]) else "⚪ NO SQUEEZE"

    return f"""
    <div class="report-card">
        <div class="report-header">📌 SIGNAL: {direction}</div>
        <div class="report-item">TF: <span class="highlight">{tf}</span> | Confidence: <span class="highlight">{conf}</span></div>
        <div class="report-item">Sentiment: <span class="highlight">{fg}/100</span></div>
        <div class="report-item">Squeeze: <span class="highlight">{sqz}</span></div>
    </div>
    <div class="report-card">
        <div class="report-header">🌊 FLOW & VOL</div>
        <div class="report-item">RVOL: <span class="highlight">{float(row['rvol']):.2f} ({vol_desc})</span></div>
        <div class="report-item">Money Flow: <span class="highlight">{float(row['money_flow']):.2f}</span></div>
        <div class="report-item">VWAP Relation: <span class="highlight">{'Above' if float(row['close']) > float(row['vwap']) else 'Below'}</span></div>
    </div>
    <div class="report-card">
        <div class="report-header">🏛️ SMC (Apex Master)</div>
        <div class="report-item">Structure Event: <span class="highlight">{am_evt}</span></div>
        <div class="report-item">Active OB Zones: <span class="highlight">{ob_c}</span></div>
        <div class="report-item">Active FVG Zones: <span class="highlight">{fvg_c}</span></div>
    </div>
    <div class="report-card">
        <div class="report-header">⚡ APEX VECTOR</div>
        <div class="report-item">State: <span class="highlight">{v_state}</span></div>
        <div class="report-item">Flux: <span class="highlight">{v_flux:.3f}</span></div>
        <div class="report-item">Divergence: <span class="highlight">{v_div}</span></div>
    </div>
    <div class="report-card">
        <div class="report-header">🎯 EXECUTION PLAN</div>
        <div class="report-item">Entry: <span class="highlight">{float(row['close']):.6f}</span></div>
        <div class="report-item">🛑 SMART STOP: <span class="highlight">{float(stop):.6f}</span></div>
        <div class="report-item">1️⃣ TP1: <span class="highlight">{float(row['tp1']):.6f}</span></div>
        <div class="report-item">2️⃣ TP2: <span class="highlight">{float(row['tp2']):.6f}</span></div>
        <div class="report-item">3️⃣ TP3: <span class="highlight">{float(row['tp3']):.6f}</span></div>
    </div>
    """

def find_latest_signal_index(df: pd.DataFrame) -> Optional[int]:
    mask = (df.get("buy", False) | df.get("sell", False))
    if "am_sig_buy" in df.columns: mask = mask | df["am_sig_buy"] | df["am_sig_sell"]
    if "vec_div_bull_reg" in df.columns: mask = mask | df["vec_div_bull_reg"] | df["vec_div_bear_reg"] | df["vec_div_bull_hid"] | df["vec_div_bear_hid"]
    idxs = df.index[mask.fillna(False)]
    return int(idxs[-1]) if len(idxs)>0 else None

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.button("🔄 Force Refresh", use_container_width=True, key="refresh_btn", on_click=hard_refresh)
    
    st.subheader("📡 Market Feed")
    # Fetch bases logic
    bases_all = get_all_tickers(nonce=st.session_state.refresh_nonce)
    
    with st.expander("🧬 Ticker Universe (Quick Select)", expanded=True):
        if bases_all:
            list_mode = st.selectbox("List", ["Popular", "All Available"], index=0)
            options = [b for b in POPULAR_BASES if b in bases_all] if list_mode=="Popular" else bases_all
            
            # Smart index selection
            curr = st.session_state.symbol_input.replace("USDT","")
            def_idx = options.index(curr) if curr in options else 0

            # Direct callback to update state
            st.selectbox(
                "Quick Ticker", 
                options, 
                index=def_idx, 
                key="quick_ticker_select",
                on_change=update_symbol_from_dropdown
            )
            st.caption(f"{len(bases_all)} tickers loaded from Binance.")
        else:
            st.error("Could not fetch tickers. Check connection.")

    # Manual Input
    symbol_input = st.text_input("Asset (Manual)", value=st.session_state.symbol_input)
    st.session_state.symbol_input = symbol_input
    
    symbol = symbol_input.strip().upper().replace("/", "").replace("-", "")
    if not symbol.endswith("USDT"): symbol += "USDT"

    c1, c2 = st.columns(2)
    with c1: timeframe = st.selectbox("TF", ["15m", "1h", "4h", "1d"], index=1)
    with c2: limit = st.slider("Depth", 100, 500, 200, 50)

    st.markdown("---")
    st.subheader("🧠 Core Parameters")
    amplitude = st.number_input("Amp", 2, 200, 10)
    channel_dev = st.number_input("Dev", 0.5, 10.0, 3.0, 0.1)
    hma_len = st.number_input("HMA", 2, 400, 50)
    gann_len = st.number_input("Gann", 1, 50, 3)
    hma_mode = st.selectbox("HMA Mode", ["True HMA (WMA-based)", "Original (SMA-based - not true HMA)"], index=0)
    
    with st.expander("🎛 Filters"):
        rvol_threshold = st.number_input("RVOL Threshold", 1.0, 10.0, 1.0, 0.1)
        signal_cooldown = st.number_input("Cooldown", 0, 100, 0)
        marker_style = st.selectbox("Markers", ["Circle (clean)", "Triangle (legacy)"], 0)

    with st.expander("🎯 Targets (R)"):
        tp1_r = st.number_input("TP1", value=1.5)
        tp2_r = st.number_input("TP2", value=3.0)
        tp3_r = st.number_input("TP3", value=5.0)

    st.markdown("---")
    st.subheader("🏛️ Apex SMC v7.2")
    am_show_sig = st.checkbox("SMC Signals", True)
    am_show_sl = st.checkbox("Trailing Stop", True)
    am_ma_type = st.selectbox("Algorithm", ["EMA", "SMA", "HMA", "RMA"], 2)
    am_len_main = st.number_input("Trend Len", 10, 200, 55)
    am_mult = st.number_input("Mult", 0.1, 5.0, 1.5)
    am_show_sd = st.checkbox("S/D Zones", True)
    am_show_bos = st.checkbox("BOS/CHoCH", True)
    am_show_ob = st.checkbox("Order Blocks", True)
    am_show_fvg = st.checkbox("FVG", True)
    am_fvg_mit = st.checkbox("Hide Mitigated", True)
    am_liq_len = 10; am_sd_ext = 20; am_adx_min = 20.0; am_vol_len = 20; am_dmi_len = 14

    st.markdown("---")
    st.subheader("⚡ Apex Vector v4.1")
    vec_eff_super = st.number_input("Super Thresh", 0.1, 1.0, 0.60)
    vec_eff_resist = st.number_input("Resist Thresh", 0.0, 0.5, 0.30)
    vec_vol_norm = st.number_input("Vol Norm", 10, 200, 55)
    vec_len_vec = st.number_input("Vec Len", 2, 50, 14)
    vec_sm_type = st.selectbox("Smooth Type", ["EMA", "SMA", "RMA", "WMA", "VWMA"], 0)
    vec_len_sm = st.number_input("Smooth Len", 1, 50, 5)
    vec_use_vol = st.checkbox("Vol Flux", True)
    vec_strictness = st.number_input("Strictness", 0.1, 2.0, 1.0)
    vec_show_div = st.checkbox("Divergences", True)
    vec_div_look = st.number_input("Div Lookback", 1, 20, 5)
    vec_show_reg = st.checkbox("Regular Div", True)
    vec_show_hid = st.checkbox("Hidden Div", False)
    vec_show_bar = st.checkbox("Color Bars", True)

    st.markdown("---")
    st.subheader("📣 Telegram")
    tg_enable = st.checkbox("Enable", True)
    tg_token = st.text_input("Token", value=st.secrets.get("TELEGRAM_TOKEN", ""), type="password")
    tg_chat_raw = st.text_input("Chat IDs", value=st.secrets.get("TELEGRAM_CHAT_ID", ""))
    tg_mode = st.selectbox("Auto Mode", ["Manual", "Auto: Core", "Auto: Confirmed", "Auto: Any"], 0)
    tg_format = st.selectbox("Format", ["MarkdownV2", "Plain"], 0)
    tg_include_chart = st.checkbox("Link Chart", True)
    tg_disable_preview = st.checkbox("No Preview", True)
    
    tg_chats = parse_chat_list(tg_chat_raw)
    
    if st.button("🧪 Test TG"):
        ok, det = tg_get_me(tg_token)
        if ok:
            st.success(det)
            msg = telegram_escape_md_v2("✅ Telegram test successful.") if tg_format=="MarkdownV2" else "✅ Telegram test successful."
            ok_any, res = tg_broadcast(tg_token, tg_chats, msg, tg_format, True)
            if ok_any: st.success("Ping Sent")
            else: st.error(f"Ping Failed: {res}")
        else: st.error(det)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
render_ticker_tape(symbol, st.session_state.refresh_nonce)

# 1. Fetch Data
df = get_klines(symbol, timeframe, limit, nonce=st.session_state.refresh_nonce)
if df.empty:
    st.error(f"❌ No data for {symbol}. Try checking the ticker name or your internet connection.")
    st.stop()

df = df.dropna(subset=["close"]).reset_index(drop=True)

# 2. Run Engines (Cached where possible)
df = run_core_engines(
    df, int(amplitude), float(channel_dev), int(hma_len), float(tp1_r), float(tp2_r), float(tp3_r),
    14, 20, int(gann_len), hma_mode, float(rvol_threshold), int(signal_cooldown)
)

df, am_drawings = compute_apex_master_smc(
    df, am_ma_type, int(am_len_main), float(am_mult), int(am_liq_len), int(am_sd_ext), 
    int(am_dmi_len), int(am_vol_len), float(am_adx_min), bool(am_show_sd), 
    bool(am_show_bos), bool(am_show_ob), bool(am_show_fvg), bool(am_fvg_mit), hma_mode
)

df = compute_apex_vector(
    df, float(vec_eff_super), float(vec_eff_resist), int(vec_vol_norm), int(vec_len_vec), 
    str(vec_sm_type), int(vec_len_sm), bool(vec_use_vol), float(vec_strictness), 
    bool(vec_show_div), int(vec_div_look), bool(vec_show_reg), bool(vec_show_hid)
)

# 3. Context Calculation
last_bar = df.iloc[-1]
fibs = calculate_fibonacci(df)
fg_index = calculate_fear_greed_index(df)

if bool(last_bar["is_bull"]):
    smart_stop = min(float(last_bar["entry_stop"]), float(fibs["fib_618"]) * 0.9995)
else:
    smart_stop = max(float(last_bar["entry_stop"]), float(fibs["fib_618"]) * 1.0005)

am_evt = "None"
if len(am_drawings.get("choch_lines", [])) > 0:
    e = am_drawings["choch_lines"][-1]
    am_evt = f"CHoCH ({e.get('side').upper()})"
elif len(am_drawings.get("bos_lines", [])) > 0:
    e = am_drawings["bos_lines"][-1]
    am_evt = f"BOS ({e.get('side').upper()})"

# Vector State
vf = float(last_bar.get("vec_flux", 0.0))
if bool(last_bar.get("vec_is_super_bull", False)): v_state = "SUPER (BULL)"
elif bool(last_bar.get("vec_is_super_bear", False)): v_state = "SUPER (BEAR)"
elif bool(last_bar.get("vec_is_resistive", False)): v_state = "RESISTIVE"
else: v_state = "HIGH HEAT"

v_div = "None"
if bool(last_bar.get("vec_div_bull_reg", False)): v_div = "Bull Reg"
elif bool(last_bar.get("vec_div_bear_reg", False)): v_div = "Bear Reg"
elif bool(last_bar.get("vec_div_bull_hid", False)): v_div = "Bull Hid"
elif bool(last_bar.get("vec_div_bear_hid", False)): v_div = "Bear Hid"

# 4. Mobile Metrics UI
c_m1, c_m2 = st.columns(2)
with c_m1:
    tv_symbol = f"BINANCE:{symbol}"
    components.html(
        f"""
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js" async>
          {{ "symbol": "{tv_symbol}", "width": "100%", "colorTheme": "dark", "isTransparent": true, "locale": "en" }}
          </script>
        </div>
        """,
        height=120,
    )

with c_m2:
    trend_txt = "BULL 🐂" if int(last_bar["gann_trend"]) == 1 else "BEAR 🐻"
    st.metric("TREND", trend_txt)

c_m3, c_m4 = st.columns(2)
with c_m3: st.metric("STOP", f"{smart_stop:.4f}")
with c_m4: st.metric("TP3", f"{float(last_bar['tp3']):.4f}")

# 5. Report & Logic
report_html = generate_mobile_report_html(last_bar, symbol, timeframe, fibs, fg_index, smart_stop, am_evt, len(am_drawings["ob_zones"]), len(am_drawings["fvg_zones"]), v_state, vf, v_div)
st.markdown(report_html, unsafe_allow_html=True)

# 6. Telegram Logic (Auto)
latest_sig_idx = find_latest_signal_index(df)
alert_text = None; report_text = None; sig_key = None

if latest_sig_idx is not None:
    row = df.iloc[latest_sig_idx]
    ts_str = pd.to_datetime(row["timestamp"], utc=True).isoformat().replace("+00:00", "Z")
    
    trig = []
    if bool(row.get("buy", False)): trig.append("CORE_BUY")
    if bool(row.get("sell", False)): trig.append("CORE_SELL")
    if bool(row.get("am_sig_buy", False)): trig.append("SMC_BUY")
    if bool(row.get("am_sig_sell", False)): trig.append("SMC_SELL")
    if bool(row.get("vec_div_bull_reg", False)): trig.append("DIV_BULL")
    if bool(row.get("vec_div_bear_reg", False)): trig.append("DIV_BEAR")
    trig_s = "+".join(trig) if trig else "NONE"
    
    sig_key = f"{symbol}|{timeframe}|{ts_str}|{trig_s}"
    
    link = tv_chart_link(symbol, timeframe) if tg_include_chart else None
    
    # Build text
    title = f"{TG_PREFIX}: {symbol} [{trig_s}]"
    lines = [
        title,
        f"Symbol: {symbol} | TF: {timeframe}",
        f"Dir: {'LONG 🐂' if row['is_bull'] else 'SHORT 🐻'}",
        f"Entry: {row['close']:.5f}",
        f"Stop: {row['entry_stop']:.5f}" if np.isfinite(row["entry_stop"]) else "Stop: n/a",
        f"TP1: {row['tp1']:.5f} | TP2: {row['tp2']:.5f}",
        f"RVOL: {row['rvol']:.2f} | SMC: {am_evt}",
        f"Flux: {row.get('vec_flux',0):.3f} ({v_state})"
    ]
    if link: lines += ["", f"Chart: {link}"]
    plain_txt = "\n".join(lines)
    alert_text = telegram_escape_md_v2(plain_txt) if tg_format=="MarkdownV2" else plain_txt
    report_text = html_to_text(report_html)

    # Auto Send Logic
    if tg_enable and tg_token and tg_chats and tg_mode != "Manual":
        should_send = False
        core = "CORE" in trig_s
        smc = "SMC" in trig_s
        
        if tg_mode == "Auto: Core" and core: should_send = True
        if tg_mode == "Auto: Confirmed" and (core and smc): should_send = True
        if tg_mode == "Auto: Any": should_send = True
        
        if should_send:
            state_key = f"{symbol}|{timeframe}|telegram_auto"
            last_sent = db_get_last(DB_CON, state_key)
            if last_sent != sig_key:
                ok_any, _ = tg_broadcast(tg_token, tg_chats, alert_text, tg_format, tg_disable_preview)
                if ok_any:
                    db_set_last(DB_CON, state_key, sig_key)
                    st.toast("📣 Auto-sent Telegram alert", icon="✅")

# 7. Manual Actions
st.markdown("### ⚡ Actions")
b1, b2, b3 = st.columns(3)
with b1:
    if st.button("🔔 Send Alert", use_container_width=True):
        if alert_text: tg_broadcast(tg_token, tg_chats, alert_text, tg_format, tg_disable_preview); st.success("Sent")
        else: st.warning("No signal")
with b2:
    if st.button("🧾 Send Report", use_container_width=True):
        if report_text: tg_broadcast(tg_token, tg_chats, report_text, "Plain", True); st.success("Sent")
        else: st.warning("No report")
with b3:
    if st.button("♻️ Resend Last", use_container_width=True):
        if alert_text: tg_broadcast(tg_token, tg_chats, alert_text, tg_format, tg_disable_preview); st.success("Resent")

# 8. Charting
fig = go.Figure()
fig.add_candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price")
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["hma"], mode="lines", name="HMA", line=dict(color="#66fcf1", width=1)))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vwap"], mode="lines", name="VWAP", line=dict(color="#9933ff", width=2)))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["am_upper"], mode="lines", name="Apex Upper", line=dict(color="rgba(0,230,118,0.35)", width=1)))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["am_lower"], mode="lines", name="Apex Lower", line=dict(color="rgba(255,23,68,0.35)", width=1)))

if am_show_sl:
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["am_trail_stop"], mode="lines", name="Apex Trail", line=dict(color="rgba(0,230,118,0.85)", width=2), connectgaps=False))

# Markers
msym = "triangle-up" if marker_style.startswith("Triangle") else "circle"
msym_d = "triangle-down" if marker_style.startswith("Triangle") else "circle"

buys = df[df["buy"]]; sells = df[df["sell"]]
if not buys.empty: fig.add_trace(go.Scatter(x=buys["timestamp"], y=buys["low"], mode="markers", marker=dict(symbol=msym, size=10, color="#00ff00"), name="BUY"))
if not sells.empty: fig.add_trace(go.Scatter(x=sells["timestamp"], y=sells["high"], mode="markers", marker=dict(symbol=msym_d, size=10, color="#ff0000"), name="SELL"))

if am_show_sig:
    amb = df[df["am_sig_buy"]]; ams = df[df["am_sig_sell"]]
    if not amb.empty: fig.add_trace(go.Scatter(x=amb["timestamp"], y=amb["low"], mode="markers", marker=dict(symbol="diamond", size=10, color="#00E676"), name="SMC BUY"))
    if not ams.empty: fig.add_trace(go.Scatter(x=ams["timestamp"], y=ams["high"], mode="markers", marker=dict(symbol="diamond", size=10, color="#FF1744"), name="SMC SELL"))

# Helper for zones
def add_zone(fig, df, zones, fc, lc):
    for z in zones:
        i0=z["i0"]; i1=z["i1"]
        if i0<0 or i1>=len(df): continue
        fig.add_shape(type="rect", x0=df["timestamp"].iloc[i0], x1=df["timestamp"].iloc[i1], y0=z["bot"], y1=z["top"], line=dict(color=lc, width=1), fillcolor=fc, layer="below")

if am_show_sd: add_zone(fig, df, am_drawings["sd_zones"], "rgba(69,162,158,0.12)", "rgba(69,162,158,0.35)")
if am_show_ob: add_zone(fig, df, am_drawings["ob_zones"], "rgba(185,246,202,0.16)", "rgba(185,246,202,0.25)")
if am_show_fvg: add_zone(fig, df, am_drawings["fvg_zones"], "rgba(255,205,210,0.12)", "rgba(255,205,210,0.22)")

fig.update_layout(height=420, template="plotly_dark", margin=dict(l=0, r=0, t=20, b=20), xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1, x=0))
st.plotly_chart(fig, use_container_width=True)

# 9. Tabs & Vector Chart
t1, t2, t3, t4 = st.tabs(["📊 GANN", "🌊 FLOW", "🧠 SENT", "⚡ VECTOR"])
with t1:
    f1 = go.Figure()
    f1.add_candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"])
    dg = df.dropna(subset=["gann_act"])
    f1.add_trace(go.Scatter(x=dg["timestamp"], y=dg["gann_act"], mode="markers", marker=dict(color=np.where(dg["gann_trend"]==1,"#00ff00","#ff0000"), size=3)))
    f1.update_layout(height=300, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(f1, use_container_width=True)

with t4:
    fx = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
    fx.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"]), row=1, col=1)
    
    cols = []
    if vec_show_bar:
        for i in range(len(df)):
            if df.at[i, "vec_is_super_bull"]: cols.append("#00E676")
            elif df.at[i, "vec_is_super_bear"]: cols.append("#FF1744")
            elif df.at[i, "vec_is_resistive"]: cols.append("#546E7A")
            else: cols.append("#FFD600")
    else: cols = ["#66fcf1"]*len(df)
    
    fx.add_trace(go.Bar(x=df["timestamp"], y=df["vec_flux"], marker_color=cols), row=2, col=1)
    th = float(df["vec_th_super"].iloc[-1])
    fx.add_hline(y=th, line=dict(color="gray", dash="dot"), row=2, col=1)
    fx.add_hline(y=-th, line=dict(color="gray", dash="dot"), row=2, col=1)
    
    db = df[df["vec_div_bull_reg"] | df["vec_div_bull_hid"]]
    ds = df[df["vec_div_bear_reg"] | df["vec_div_bear_hid"]]
    if not db.empty: fx.add_trace(go.Scatter(x=db["timestamp"], y=db["vec_flux"], mode="markers", marker=dict(color="#00B0FF", size=6)), row=2, col=1)
    if not ds.empty: fx.add_trace(go.Scatter(x=ds["timestamp"], y=ds["vec_flux"], mode="markers", marker=dict(color="#FF4081", size=6)), row=2, col=1)
    
    fx.update_layout(height=500, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fx, use_container_width=True)

# 10. Additional Feature: Data Download
with st.sidebar:
    st.markdown("---")
    st.subheader("💾 Export")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Data (CSV)", csv, "signals_data.csv", "text/csv")
