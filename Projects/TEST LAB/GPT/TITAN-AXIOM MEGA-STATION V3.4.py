Without omitting anything from the first code. I share. -------------# =============================================================================
# TITAN-AXIOM MEGA-STATION V3.4 (DIAMOND-VISUAL) — INTEGRATED BUILD
# -----------------------------------------------------------------------------
# No omissions. No assumptions. Base preserved.
#
# INTEGRATION GOALS (100% Feature Parity):
# - MODE 1: TITAN MOBILE (Binance.US | Scalping | SMC | Gann | Simple Clock | Report Cards)
# - MODE 2: AXIOM QUANT (YFinance | Swing | Physics | Macro | World Clock | AI | Monte Carlo | VP | DNA)
# - PLUS: Terminal-v2 Features Integrated:
#   * Macro 40-panel dashboard (batch yfinance)
#   * God Mode indicators + GM Score + Fear/Greed v4
#   * Seasonality (monthly), Day-of-week DNA (advanced), Intraday hour DNA
#   * Correlation + Multi-timeframe trend radar
#   * Smart Money Concepts overlay
#   * Volume Profile (VPVR) and POC
#   * TradingView advanced chart embed
#   * Telegram advanced broadcaster (multi-chat, MarkdownV2 escape, photo upload, chunking)
#   * TradingView Alerts Inbox (SQLite): view + manual ingest + send selected
#
# CONFLICTS (explicit):
# - Streamlit only allows ONE st.set_page_config -> single call used.
# - CSS merged: DIAMOND-VISUAL is base, Terminal-v2 styles added (non-destructive).
# =============================================================================

import time
import math
import sqlite3
import random
import json
import re
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import streamlit.components.v1 as components
from openai import OpenAI
from scipy.stats import linregress

# =============================================================================
# 1) PAGE CONFIG (SINGLE CALL — STREAMLIT LIMITATION)
# =============================================================================
st.set_page_config(
    page_title="Titan-Axiom Omega",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# 2) CSS (DIAMOND-VISUAL BASE + TERMINAL-v2 ADDITIONS)
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=SF+Pro+Display:wght@300;500;700&display=swap');

    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'SF Pro Display', sans-serif; }

    /* TITAN STYLE ELEMENTS */
    .titan-metric { background: rgba(31, 40, 51, 0.9); border: 1px solid #45a29e; padding: 10px; border-radius: 8px; }

    /* AXIOM NEON METRICS */
    div[data-testid="metric-container"] {
        background: rgba(22, 27, 34, 0.9);
        border-left: 4px solid #00F0FF;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 10px;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] { font-size: 14px !important; color: #8b949e !important; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 24px !important; color: #f0f6fc !important; font-weight: 300; }

    /* UNIVERSAL HEADERS & BUTTONS */
    h1, h2, h3 { font-family: 'Roboto Mono', monospace; color: #58a6ff; }
    .stButton > button {
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        border: 1px solid #238636; color: #ffffff;
        font-weight: bold; height: 3.5em; font-size: 16px !important;
        border-radius: 6px;
    }
    .stButton > button:hover { background: #238636; color: #ffffff; }

    /* AXIOM TICKER MARQUEE */
    .ticker-wrap {
        width: 100%; overflow: hidden; background-color: #0d1117; border-bottom: 1px solid #30363d;
        height: 40px; display: flex; align-items: center; margin-bottom: 15px;
    }
    .ticker { display: inline-block; animation: marquee 45s linear infinite; white-space: nowrap; }
    @keyframes marquee { 0% { transform: translate(100%, 0); } 100% { transform: translate(-100%, 0); } }
    .ticker-item { padding: 0 2rem; font-family: 'Roboto Mono'; font-size: 0.85rem; color: #58a6ff; text-shadow: 0 0 5px rgba(88, 166, 255, 0.3); }

    /* TITAN MOBILE REPORT CARDS (MATCHING SCREENSHOT) */
    .report-card {
        background-color: #161b22;
        border-left: 4px solid #3fb950;
        padding: 16px;
        border-radius: 6px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.5);
        border: 1px solid #30363d;
    }
    .report-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 12px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'Roboto Mono', monospace;
    }
    .report-item {
        margin-bottom: 8px;
        font-size: 0.95rem;
        color: #8b949e;
        display: flex;
        align-items: center;
        gap: 8px;
        justify-content: flex-start;
    }
    .value-cyan { color: #38bdf8; font-weight: 600; font-family: 'Roboto Mono'; }
    .value-green { color: #3fb950; font-weight: 600; font-family: 'Roboto Mono'; }
    .value-red { color: #ff6b6b; font-weight: 600; font-family: 'Roboto Mono'; }
    .value-yellow { color: #e3b341; font-weight: 600; font-family: 'Roboto Mono'; }
    .strategy-tag { background-color: #1f6feb; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-right: 5px; }

    /* Terminal-v2 style additions (non-destructive) */
    .title-glow {
        font-size: 2.0em;
        font-weight: bold;
        color: #ffffff;
        text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 40px #00ff00;
        margin-bottom: 10px;
        font-family: 'Roboto Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3) SECRETS + BASIC UTILITIES
# =============================================================================
class SecretsManager:
    @staticmethod
    def get(key: str, default: str = "") -> str:
        try:
            return st.secrets.get(key, default)
        except Exception:
            return default

def timeframe_to_min(tf: str) -> int:
    if tf == "15m": return 15
    if tf == "1h": return 60
    if tf == "4h": return 240
    if tf == "1d": return 1440
    if tf == "1wk": return 10080
    return 60

# =============================================================================
# 4) TELEGRAM ADVANCED BROADCASTER (Terminal-v2 parity)
# =============================================================================
def _parse_chat_ids(chat_id_field: str) -> List[str]:
    if not isinstance(chat_id_field, str):
        return []
    parts = [p.strip() for p in chat_id_field.split(",")]
    return [p for p in parts if p]

def tg_escape_markdown_v2(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r'([_\*\[\]\(\)~`>#+\-=|{}\.!])', r'\\\1', str(text))

def tg_send_message(token: str, chat_id: str, text: str, parse_mode: Optional[str] = None) -> Tuple[bool, str]:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        r = requests.post(url, data=payload, timeout=20)
        ok = (r.status_code == 200)
        return ok, r.text
    except Exception as e:
        return False, str(e)

def tg_send_photo(token: str, chat_id: str, photo_bytes: bytes, caption: Optional[str] = None, parse_mode: Optional[str] = None) -> Tuple[bool, str]:
    try:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        if parse_mode:
            data["parse_mode"] = parse_mode
        files = {"photo": photo_bytes}
        r = requests.post(url, data=data, files=files, timeout=30)
        ok = (r.status_code == 200)
        return ok, r.text
    except Exception as e:
        return False, str(e)

def send_telegram_simple(token: str, chat: str, msg: str) -> bool:
    # Base behavior preserved (Mega-Station original)
    if not token or not chat:
        return False
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"},
            timeout=3
        )
        return True
    except Exception:
        return False

# =============================================================================
# 5) TRADINGVIEW ALERTS INBOX (SQLite) — Terminal-v2 parity
# =============================================================================
TV_DB_PATH = "tv_alerts.db"

def init_tv_db(db_path: str = TV_DB_PATH) -> None:
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                dedup_key TEXT PRIMARY KEY,
                ts INTEGER,
                src TEXT,
                symbol TEXT,
                tf TEXT,
                side TEXT,
                confidence INTEGER,
                entry REAL,
                stop REAL,
                tp1 REAL,
                tp2 REAL,
                reason TEXT,
                raw_json TEXT,
                created_at REAL
            )
        """)
        conn.commit()
        conn.close()
    except Exception:
        pass

@dataclass(frozen=True)
class TVAlert:
    v: int
    src: str
    symbol: str
    tf: str
    side: str
    confidence: int
    entry: float
    stop: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    dedup_key: str
    reason: str
    ts: int

def parse_tv_alert(body: str) -> TVAlert:
    try:
        obj = json.loads(body)
    except Exception as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    def get(k: str, default: Any = None) -> Any:
        return obj.get(k, default)

    return TVAlert(
        v=int(get("v", 1)),
        src=str(get("src", "tv")),
        symbol=str(get("symbol", "")).strip(),
        tf=str(get("tf", "")).strip(),
        side=str(get("side", "")).strip().upper(),
        confidence=int(get("confidence", 0)),
        entry=float(get("entry", 0.0)),
        stop=float(get("stop")) if get("stop") not in (None, "", False) else None,
        tp1=float(get("tp1")) if get("tp1") not in (None, "", False) else None,
        tp2=float(get("tp2")) if get("tp2") not in (None, "", False) else None,
        dedup_key=str(get("dedup_key", "")).strip(),
        reason=str(get("reason", "")).strip(),
        ts=int(get("ts", 0))
    )

def upsert_tv_alert(alert: TVAlert, raw_body: str, db_path: str = TV_DB_PATH) -> None:
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO alerts
            (dedup_key, ts, src, symbol, tf, side, confidence, entry, stop, tp1, tp2, reason, raw_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.dedup_key, alert.ts, alert.src, alert.symbol, alert.tf, alert.side,
            alert.confidence, alert.entry,
            alert.stop, alert.tp1, alert.tp2,
            alert.reason, raw_body, time.time()
        ))
        conn.commit()
        conn.close()
    except Exception:
        pass

def fetch_recent_tv_alerts(limit: int = 50, db_path: str = TV_DB_PATH) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM alerts ORDER BY ts DESC LIMIT {int(limit)}", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# =============================================================================
# 6) VISUALS ENGINE (Mega-Station base)
# =============================================================================
class Visuals:
    @staticmethod
    def render_titan_clock():
        html = """
        <div style="display:flex; justify-content:center; font-family:'Roboto Mono'; color:#39ff14; text-shadow:0 0 10px rgba(57,255,20,0.8); font-weight:bold;">
            <span id="clock">--:--:-- UTC</span>
        </div>
        <script>
        setInterval(() => {
            document.getElementById('clock').innerText = new Date().toLocaleTimeString('en-GB', {timeZone:'UTC'}) + ' UTC';
        }, 1000);
        </script>
        """
        components.html(html, height=30)

    @staticmethod
    def render_axiom_clock():
        html = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@500&display=swap');
            body { margin: 0; background: transparent; font-family: 'Roboto Mono', monospace; color: #fff; overflow: hidden; }
            .clock-container {
                display: flex; justify-content: space-between; align-items: center;
                background: rgba(0, 0, 0, 0.3); border: 1px solid #222;
                padding: 8px 15px; border-radius: 4px; backdrop-filter: blur(5px);
            }
            .clock-box { text-align: center; width: 32%; }
            .city { font-size: 0.65rem; color: #666; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 2px; }
            .time { font-size: 1.1rem; font-weight: bold; color: #e0e0e0; text-shadow: 0 0 8px rgba(255, 255, 255, 0.1); }
            .accent-ny { border-bottom: 2px solid #00F0FF; }
            .accent-lon { border-bottom: 2px solid #FF0055; }
            .accent-tok { border-bottom: 2px solid #D500F9; }
        </style>
        </head>
        <body>
            <div class="clock-container">
                <div class="clock-box accent-ny"><div class="city">NEW YORK</div><div class="time" id="ny">--:--:--</div></div>
                <div class="clock-box accent-lon"><div class="city">LONDON</div><div class="time" id="lon">--:--:--</div></div>
                <div class="clock-box accent-tok"><div class="city">TOKYO</div><div class="time" id="tok">--:--:--</div></div>
            </div>
            <script>
                function update() {
                    const now = new Date();
                    const fmt = (tz) => new Intl.DateTimeFormat('en-US', {timeZone: tz, hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false}).format(now);
                    document.getElementById('ny').innerText = fmt('America/New_York');
                    document.getElementById('lon').innerText = fmt('Europe/London');
                    document.getElementById('tok').innerText = fmt('Asia/Tokyo');
                }
                setInterval(update, 1000); update();
            </script>
        </body>
        </html>
        """
        components.html(html, height=80)

    @staticmethod
    def render_titan_tape(selected_symbol: str):
        base = selected_symbol.replace("USDT", "")
        tape_bases = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE"]
        if base not in tape_bases:
            tape_bases.insert(0, base)
        symbols_json = json.dumps([{"proName": f"BINANCE:{b}USDT", "title": b} for b in tape_bases], separators=(",", ":"))
        components.html(
            f"""<div class="tradingview-widget-container">
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
            </div>""",
            height=50
        )

    @staticmethod
    def render_axiom_banner():
        html = """
        <div class="ticker-wrap">
            <div class="ticker">
                <span class="ticker-item">💠 AXIOM QUANTITATIVE SYSTEM ONLINE</span>
                <span class="ticker-item">BTC-USD: LIVE</span>
                <span class="ticker-item">SPY: LIVE</span>
                <span class="ticker-item">VIX: LIVE</span>
                <span class="ticker-item">GOLD: LIVE</span>
                <span class="ticker-item">NVDA: LIVE</span>
                <span class="ticker-item">EUR/USD: LIVE</span>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    @staticmethod
    def render_tradingview_advanced_chart(symbol: str, interval: str):
        tv_interval_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D", "1wk": "W"}
        tv_int = tv_interval_map.get(interval, "D")
        tv_symbol = symbol.replace("-", "") if "BTC" in symbol else symbol
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
                "toolbar_bg": "#0d1117",
                "enable_publishing": false,
                "hide_side_toolbar": false,
                "allow_symbol_change": true,
                "container_id": "tradingview_widget"
            }}
            );
            </script>
        </div>
        """
        components.html(tv_widget_html, height=520)
        st.caption("TradingView advanced chart with drawing tools enabled (left toolbar).")

# =============================================================================
# 7) TERMINAL-v2 MACRO ENGINE (Batch 40 indicators)
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_macro_data_40():
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
        "📈 Growth & Real Assets": {"Emerging Mkts": "EEM", "China (FXI)": "FXI", "Real Estate": "VNQ", "Soybeans": "ZS=F"},
    }

    all_tickers_list = []
    ticker_to_name_map = {}
    for g_name, g_dict in groups.items():
        for t_name, t_sym in g_dict.items():
            all_tickers_list.append(t_sym)
            ticker_to_name_map[t_sym] = t_name

    try:
        data_batch = yf.download(all_tickers_list, period="5d", interval="1d", group_by="ticker", progress=False)
        prices: Dict[str, float] = {}
        changes: Dict[str, float] = {}

        for sym in all_tickers_list:
            try:
                df = data_batch[sym] if len(all_tickers_list) > 1 else data_batch
                if df is None or df.empty:
                    continue
                df = df.dropna(how="all")
                if len(df) >= 2:
                    col = "Close" if "Close" in df.columns else "Adj Close"
                    curr = float(df[col].iloc[-1])
                    prev = float(df[col].iloc[-2])
                    chg = ((curr - prev) / prev) * 100.0

                    name = ticker_to_name_map.get(sym, sym)
                    prices[name] = curr
                    changes[name] = float(chg)
            except Exception:
                continue

        return groups, prices, changes
    except Exception:
        return groups, {}, {}

# =============================================================================
# 8) TERMINAL-v2 DATA HELPERS (YFinance)
# =============================================================================
def safe_download(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    try:
        dl_interval = "1h" if interval == "4h" else interval
        df = yf.download(ticker, period=period, interval=dl_interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None
        if "Close" not in df.columns:
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            else:
                return None
        return df
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentals(ticker: str) -> Optional[Dict[str, Any]]:
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
    except Exception:
        return None

# =============================================================================
# 9) TERMINAL-v2 INDICATOR STACK (God Mode + Fear/Greed v4 + SMC + VPVR + etc.)
# =============================================================================
def calculate_wma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).apply(
        lambda x: np.dot(x, np.arange(1, length + 1)) / (length * (length + 1) / 2),
        raw=True
    )

def calculate_hma(series: pd.Series, length: int) -> pd.Series:
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = calculate_wma(series, half_length)
    wma_full = calculate_wma(series, length)
    diff = 2 * wma_half - wma_full
    return calculate_wma(diff, sqrt_length)

def calculate_atr_ohlc(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    atr = calculate_atr_ohlc(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    final_upperband = hl2 + (multiplier * atr)
    final_lowerband = hl2 - (multiplier * atr)

    close = df["Close"].values
    upper = final_upperband.values
    lower = final_lowerband.values
    st_vals = np.zeros(len(df))
    trend = np.zeros(len(df))

    st_vals[0] = lower[0]
    trend[0] = 1

    for i in range(1, len(df)):
        if close[i - 1] > st_vals[i - 1]:
            st_vals[i] = max(lower[i], st_vals[i - 1]) if close[i] > st_vals[i - 1] else upper[i]
            trend[i] = 1 if close[i] > st_vals[i - 1] else -1
            if close[i] < lower[i] and trend[i - 1] == 1:
                st_vals[i] = upper[i]
                trend[i] = -1
        else:
            st_vals[i] = min(upper[i], st_vals[i - 1]) if close[i] < st_vals[i - 1] else lower[i]
            trend[i] = -1 if close[i] < st_vals[i - 1] else 1
            if close[i] > upper[i] and trend[i - 1] == -1:
                st_vals[i] = lower[i]
                trend[i] = 1

    return pd.Series(st_vals, index=df.index), pd.Series(trend, index=df.index)

def calc_indicators_godmode(df: pd.DataFrame) -> pd.DataFrame:
    # Base
    df = df.copy()
    df["HMA"] = calculate_hma(df["Close"], 55)
    df["ATR"] = calculate_atr_ohlc(df, 14)
    df["Pivot_Resist"] = df["High"].rolling(20).max()
    df["Pivot_Support"] = df["Low"].rolling(20).min()
    df["MFI"] = (df["Close"].diff() * df["Volume"]).rolling(14).mean()

    # 1) Apex Trend & Liquidity Master (HMA + ATR Bands)
    apex_mult = 1.5
    df["Apex_Base"] = df["HMA"]
    df["Apex_ATR"] = calculate_atr_ohlc(df, 55)
    df["Apex_Upper"] = df["Apex_Base"] + (df["Apex_ATR"] * apex_mult)
    df["Apex_Lower"] = df["Apex_Base"] - (df["Apex_ATR"] * apex_mult)
    df["Apex_Trend"] = np.where(df["Close"] > df["Apex_Upper"], 1, np.where(df["Close"] < df["Apex_Lower"], -1, 0))
    df["Apex_Trend"] = df["Apex_Trend"].replace(to_replace=0, method="ffill")

    # 2) Squeeze Momentum (BB/KC)
    df["Sqz_Basis"] = df["Close"].rolling(20).mean()
    df["Sqz_Dev"] = df["Close"].rolling(20).std() * 2.0
    df["Sqz_Upper_BB"] = df["Sqz_Basis"] + df["Sqz_Dev"]
    df["Sqz_Lower_BB"] = df["Sqz_Basis"] - df["Sqz_Dev"]

    df["Sqz_Ma_KC"] = df["Close"].rolling(20).mean()
    df["Sqz_Range_MA"] = calculate_atr_ohlc(df, 20)
    df["Sqz_Upper_KC"] = df["Sqz_Ma_KC"] + (df["Sqz_Range_MA"] * 1.5)
    df["Sqz_Lower_KC"] = df["Sqz_Ma_KC"] - (df["Sqz_Range_MA"] * 1.5)

    df["Squeeze_On"] = (df["Sqz_Lower_BB"] > df["Sqz_Lower_KC"]) & (df["Sqz_Upper_BB"] < df["Sqz_Upper_KC"])

    highest = df["High"].rolling(20).max()
    lowest = df["Low"].rolling(20).min()
    avg_val = (highest + lowest + df["Sqz_Ma_KC"]) / 3
    df["Sqz_Mom"] = (df["Close"] - avg_val).rolling(20).mean() * 100

    # 3) Money Flow Matrix
    up = df["Close"].diff().clip(lower=0)
    dn = df["Close"].diff().clip(upper=0).abs()
    rs_like = (up.rolling(14).mean() / (dn.rolling(14).mean() + 1e-9))
    rsi_src = (100 - (100 / (1 + rs_like))) - 50
    mf_vol = df["Volume"] / (df["Volume"].rolling(14).mean() + 1e-9)
    df["MF_Matrix"] = (rsi_src * mf_vol).ewm(span=3).mean()

    # 4) Dark Vector Scalping (Donchian staircase proxy)
    amp = 5
    df["VS_Low"] = df["Low"].rolling(amp).min()
    df["VS_High"] = df["High"].rolling(amp).max()
    df["VS_Trend"] = np.where(df["Close"] > df["VS_High"].shift(1), 1, np.where(df["Close"] < df["VS_Low"].shift(1), -1, 0))
    df["VS_Trend"] = df["VS_Trend"].replace(to_replace=0, method="ffill")

    # 5) RVOL
    df["RVOL"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)

    # 6) EVWM
    ev_len = 21
    ev_base = calculate_hma(df["Close"], ev_len)
    ev_atr = calculate_atr_ohlc(df, ev_len)
    ev_elast = (df["Close"] - ev_base) / (ev_atr + 1e-9)
    ev_force = np.sqrt(df["RVOL"].ewm(span=5).mean())
    df["EVWM"] = ev_elast * ev_force

    # 8) Gann HL Activator (simple)
    gann_len = 3
    df["Gann_High"] = df["High"].rolling(gann_len).mean()
    df["Gann_Low"] = df["Low"].rolling(gann_len).mean()
    df["Gann_Trend"] = np.where(df["Close"] > df["Gann_High"].shift(1), 1, np.where(df["Close"] < df["Gann_Low"].shift(1), -1, 0))
    df["Gann_Trend"] = df["Gann_Trend"].replace(to_replace=0, method="ffill")

    # 9) DarkVector Trend (SuperTrend dir)
    _, st_dir = calculate_supertrend(df, 10, 4.0)
    df["DarkVector_Trend"] = st_dir

    # 10) Trend Shield (SMA200)
    df["Trend_Shield_Bull"] = df["Close"] > df["Close"].rolling(200).mean()

    # GOD MODE SCORE
    df["GM_Score"] = (
        df["Apex_Trend"] +
        df["Gann_Trend"] +
        df["DarkVector_Trend"] +
        df["VS_Trend"] +
        np.sign(df["Sqz_Mom"]).replace({0: 0})
    )

    # Dashboard classics (RSI etc.)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    return df

def calc_fear_greed_v4(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # RSI component
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["FG_RSI"] = 100 - (100 / (1 + rs))

    # MACD component
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    df["FG_MACD"] = (50 + (hist * 10)).clip(0, 100)

    # BB component
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    upper = sma20 + (std20 * 2)
    lower = sma20 - (std20 * 2)
    df["FG_BB"] = ((df["Close"] - lower) / (upper - lower + 1e-9) * 100).clip(0, 100)

    # MA trend component
    sma50 = df["Close"].rolling(50).mean()
    sma200 = df["Close"].rolling(200).mean()
    conditions = [
        (df["Close"] > sma50) & (sma50 > sma200),
        (df["Close"] > sma50),
        (df["Close"] < sma50) & (sma50 < sma200)
    ]
    choices = [75, 60, 25]
    df["FG_MA"] = np.select(conditions, choices, default=40)

    df["FG_Raw"] = (df["FG_RSI"] * 0.30) + (df["FG_MACD"] * 0.25) + (df["FG_BB"] * 0.25) + (df["FG_MA"] * 0.20)
    df["FG_Index"] = df["FG_Raw"].rolling(5).mean()

    # FOMO/PANIC
    vol_ma = df["Volume"].rolling(20).mean()
    high_vol = df["Volume"] > (vol_ma * 2.5)
    high_rsi = df["FG_RSI"] > 70
    momentum = df["Close"] > df["Close"].shift(3) * 1.02
    above_bb = df["Close"] > (upper * 1.0)
    df["IS_FOMO"] = high_vol & high_rsi & momentum & above_bb

    daily_drop = df["Close"].pct_change() * 100
    sharp_drop = daily_drop < -3.0
    panic_vol = df["Volume"] > (vol_ma * 3.0)
    low_rsi = df["FG_RSI"] < 30
    df["IS_PANIC"] = sharp_drop & panic_vol & (low_rsi | (daily_drop < -5.0))

    return df

def run_monte_carlo(df: pd.DataFrame, days: int = 30, simulations: int = 300) -> np.ndarray:
    last_price = float(df["Close"].iloc[-1])
    returns = df["Close"].pct_change().dropna()
    mu = float(returns.mean())
    sigma = float(returns.std())
    daily_returns_sim = np.random.normal(mu, sigma, (days, simulations))
    price_paths = np.zeros((days, simulations))
    price_paths[0] = last_price
    for t in range(1, days):
        price_paths[t] = price_paths[t - 1] * (1 + daily_returns_sim[t])
    return price_paths

def calc_volume_profile(df: pd.DataFrame, bins: int = 50) -> Tuple[pd.DataFrame, float]:
    w = df.copy()
    price_min = float(w["Low"].min())
    price_max = float(w["High"].max())
    price_bins = np.linspace(price_min, price_max, bins)
    w["Mid"] = (w["Close"] + w["Open"]) / 2
    w["Bin"] = pd.cut(w["Mid"], bins=price_bins, labels=price_bins[:-1], include_lowest=True)
    vp = w.groupby("Bin", observed=False)["Volume"].sum().reset_index()
    vp["Price"] = vp["Bin"].astype(float)
    poc_idx = int(vp["Volume"].idxmax())
    poc_price = float(vp.loc[poc_idx, "Price"])
    return vp, poc_price

def calculate_smc_overlay(df: pd.DataFrame, swing_length: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    smc_data = {"structures": [], "order_blocks": [], "fvgs": []}
    w = df.copy()

    # FVGs
    for i in range(2, len(w)):
        if w["Low"].iloc[i] > w["High"].iloc[i - 2]:
            smc_data["fvgs"].append({
                "x0": w.index[i - 2], "x1": w.index[i],
                "y0": float(w["High"].iloc[i - 2]), "y1": float(w["Low"].iloc[i]),
                "color": "rgba(0, 255, 104, 0.3)"
            })
        if w["High"].iloc[i] < w["Low"].iloc[i - 2]:
            smc_data["fvgs"].append({
                "x0": w.index[i - 2], "x1": w.index[i],
                "y0": float(w["Low"].iloc[i - 2]), "y1": float(w["High"].iloc[i]),
                "color": "rgba(255, 0, 8, 0.3)"
            })

    # Pivots
    w["Pivot_High"] = w["High"].rolling(window=swing_length * 2 + 1, center=True).max() == w["High"]
    w["Pivot_Low"] = w["Low"].rolling(window=swing_length * 2 + 1, center=True).min() == w["Low"]

    last_high = None
    last_low = None
    trend = 0

    for i in range(swing_length, len(w)):
        curr_idx = w.index[i]
        curr_close = float(w["Close"].iloc[i])

        if bool(w["Pivot_High"].iloc[i - swing_length]):
            last_high = {"price": float(w["High"].iloc[i - swing_length]), "idx": w.index[i - swing_length], "i": i - swing_length}
        if bool(w["Pivot_Low"].iloc[i - swing_length]):
            last_low = {"price": float(w["Low"].iloc[i - swing_length]), "idx": w.index[i - swing_length], "i": i - swing_length}

        if last_high and curr_close > last_high["price"]:
            label = "CHoCH" if trend != 1 else "BOS"
            trend = 1
            smc_data["structures"].append({"x0": last_high["idx"], "x1": curr_idx, "y": last_high["price"], "color": "green", "label": label})
            if last_low:
                subset = w.iloc[last_low["i"]:i]
                if not subset.empty:
                    ob_idx = subset["Low"].idxmin()
                    ob_row = w.loc[ob_idx]
                    smc_data["order_blocks"].append({
                        "x0": ob_idx, "x1": w.index[-1],
                        "y0": float(ob_row["Low"]), "y1": float(ob_row["High"]),
                        "color": "rgba(33, 87, 243, 0.4)"
                    })
            last_high = None

        elif last_low and curr_close < last_low["price"]:
            label = "CHoCH" if trend != -1 else "BOS"
            trend = -1
            smc_data["structures"].append({"x0": last_low["idx"], "x1": curr_idx, "y": last_low["price"], "color": "red", "label": label})
            if last_high:
                subset = w.iloc[last_high["i"]:i]
                if not subset.empty:
                    ob_idx = subset["High"].idxmax()
                    ob_row = w.loc[ob_idx]
                    smc_data["order_blocks"].append({
                        "x0": ob_idx, "x1": w.index[-1],
                        "y0": float(ob_row["Low"]), "y1": float(ob_row["High"]),
                        "color": "rgba(255, 0, 0, 0.4)"
                    })
            last_low = None

    return smc_data

@st.cache_data(ttl=3600, show_spinner=False)
def get_seasonality_stats(ticker: str):
    try:
        df = yf.download(ticker, period="20y", interval="1mo", progress=False)
        if df.empty or len(df) < 12:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Close" not in df.columns:
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            else:
                return None
        df = df.dropna()
        df["Return"] = df["Close"].pct_change() * 100
        df["Year"] = df.index.year
        df["Month"] = df.index.month
        heatmap_data = df.pivot_table(index="Year", columns="Month", values="Return")
        periods = [1, 3, 6, 12]
        hold_stats = {}
        for p in periods:
            rolling_ret = df["Close"].pct_change(periods=p) * 100
            rolling_ret = rolling_ret.dropna()
            win_rate = float((rolling_ret > 0).mean() * 100) if len(rolling_ret) else 0.0
            avg_ret = float(rolling_ret.mean()) if len(rolling_ret) else 0.0
            hold_stats[p] = {"Win Rate": win_rate, "Avg Return": avg_ret}
        month_stats = df.groupby("Month")["Return"].agg(["mean", lambda x: (x > 0).mean() * 100, "count"])
        month_stats.columns = ["Avg Return", "Win Rate", "Count"]
        return heatmap_data, hold_stats, month_stats
    except Exception:
        return None

def calc_day_of_week_dna(ticker: str, lookback: int, calc_mode: str):
    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.iloc[-lookback:].copy()
        if calc_mode == "Close to Close (Total)":
            df["Day_Return"] = df["Close"].pct_change() * 100
        else:
            df["Day_Return"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100
        df = df.dropna()
        df["Day_Name"] = df.index.day_name()
        pivot_ret = df.pivot(columns="Day_Name", values="Day_Return").fillna(0)
        cum_ret = pivot_ret.cumsum()
        stats = df.groupby("Day_Name")["Day_Return"].agg(["count", "sum", "mean", lambda x: (x > 0).mean() * 100])
        stats.columns = ["Count", "Total Return", "Avg Return", "Win Rate"]
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        stats = stats.reindex([d for d in days_order if d in stats.index])
        return cum_ret, stats
    except Exception:
        return None

def calc_intraday_dna(ticker: str):
    try:
        df = yf.download(ticker, period="60d", interval="1h", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None
        df["Return"] = df["Close"].pct_change() * 100
        df["Hour"] = df.index.hour
        hourly_stats = df.groupby("Hour")["Return"].agg(["mean", "sum", "count", lambda x: (x > 0).mean() * 100])
        hourly_stats.columns = ["Avg Return", "Total Return", "Count", "Win Rate"]
        return hourly_stats
    except Exception:
        return None

def calc_correlations(ticker: str, lookback_days: int = 180) -> Optional[pd.Series]:
    macro_tickers = {
        "S&P 500": "SPY",
        "Bitcoin": "BTC-USD",
        "10Y Yield": "^TNX",
        "Dollar (DXY)": "DX-Y.NYB",
        "Gold": "GC=F",
        "Oil": "CL=F",
    }
    try:
        df_main = yf.download(ticker, period="1y", interval="1d", progress=False)["Close"]
        df_macro = yf.download(list(macro_tickers.values()), period="1y", interval="1d", progress=False)["Close"]
        combined = df_macro.copy()
        combined[ticker] = df_main
        corr_matrix = combined.iloc[-lookback_days:].corr()
        target_corr = corr_matrix[ticker].drop(ticker).sort_values(ascending=False)
        inv_map = {v: k for k, v in macro_tickers.items()}
        target_corr.index = [inv_map.get(x, x) for x in target_corr.index]
        return target_corr
    except Exception:
        return None

def calc_mtf_trend(ticker: str) -> pd.DataFrame:
    timeframes = {"1H": "1h", "4H": "1h", "Daily": "1d", "Weekly": "1wk"}
    trends = {}
    for tf_name, tf_code in timeframes.items():
        try:
            period = "1y" if tf_name in ("1H", "4H") else "2y"
            df = yf.download(ticker, period=period, interval=tf_code, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 50:
                trends[tf_name] = {"Trend": "N/A", "RSI": "N/A", "EMA Spread": "N/A"}
                continue
            if tf_name == "4H":
                df = df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
            df["EMA20"] = df["Close"].ewm(span=20).mean()
            df["EMA50"] = df["Close"].ewm(span=50).mean()
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            df["RSI"] = 100 - (100 / (1 + rs))
            last = df.iloc[-1]
            trend = (
                "BULLISH" if last["Close"] > last["EMA20"] and last["EMA20"] > last["EMA50"]
                else "BEARISH" if last["Close"] < last["EMA20"] and last["EMA20"] < last["EMA50"]
                else "NEUTRAL"
            )
            trends[tf_name] = {"Trend": trend, "RSI": f"{float(last['RSI']):.1f}", "EMA Spread": f"{float(last['EMA20'] - last['EMA50']):.2f}"}
        except Exception:
            trends[tf_name] = {"Trend": "N/A", "RSI": "N/A", "EMA Spread": "N/A"}
    return pd.DataFrame(trends).T

# =============================================================================
# 10) AI ANALYST (Terminal-v2 style, grounded)
# =============================================================================
def ask_ai_analyst_godmode(df: pd.DataFrame, ticker: str, fundamentals: Optional[Dict[str, Any]], timeframe: str, api_key: str) -> str:
    if not api_key:
        return "⚠️ Missing OpenAI API Key."

    last = df.iloc[-1]
    trend = "BULLISH" if float(last["Close"]) > float(last.get("HMA", last["Close"])) else "BEARISH"

    gm_score = float(last.get("GM_Score", 0))
    gm_verdict = "STRONG BUY" if gm_score >= 3 else "STRONG SELL" if gm_score <= -3 else "NEUTRAL"

    fg_val = float(last.get("FG_Index", np.nan))
    fg_state = "N/A"
    if not np.isnan(fg_val):
        fg_state = "EXTREME GREED" if fg_val >= 80 else "GREED" if fg_val >= 60 else "NEUTRAL" if fg_val >= 40 else "FEAR" if fg_val >= 20 else "EXTREME FEAR"

    psych_alert = ""
    if bool(last.get("IS_FOMO", False)):
        psych_alert = "WARNING: ALGORITHMIC FOMO DETECTED."
    if bool(last.get("IS_PANIC", False)):
        psych_alert = "WARNING: PANIC SELLING DETECTED."

    fund_text = "N/A"
    if fundamentals:
        pe = fundamentals.get("P/E Ratio", "N/A")
        growth = fundamentals.get("Rev Growth", 0)
        try:
            growth_txt = f"{float(growth) * 100:.1f}%"
        except Exception:
            growth_txt = "N/A"
        fund_text = f"P/E: {pe}. Growth: {growth_txt}."

    prompt = f"""
Act as a Senior Market Analyst. Analyze {ticker} on the **{timeframe} timeframe** at price {float(last['Close']):.4f}.

--- DATA FEED ---
Technicals: Trend is {trend}. Volatility (ATR) is {float(last.get('ATR', 0)):.4f}.
RSI: {float(last.get('RSI', 0)):.1f}.
Volume (RVOL): {float(last.get('RVOL', 0)):.2f}x.
Titan Score: {gm_score:.0f} ({gm_verdict}).
Momentum: {"Rising" if float(last.get("Sqz_Mom", 0)) > 0 else "Falling"}.
Sentiment: {fg_state} ({fg_val if not np.isnan(fg_val) else 0:.1f}/100).
{psych_alert}
Fundamentals: {fund_text}

--- MISSION ---
Provide a concise, high-level overview of what is happening with this asset.
1) Analyze current market structure (Trend vs Chop).
2) Explain correlation between technicals and sentiment.
3) Provide general outlook on potential direction.

IMPORTANT:
- Do NOT provide specific Entry/Exit/Stop numbers.
- Do NOT give financial advice.
- Use emojis liberally 🚀📉🐂🐻🧠⚠️
"""

    try:
        client = OpenAI(api_key=api_key)
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error: {e}"

# =============================================================================
# 11) TITAN ENGINE (BINANCE / SCALPING) — Mega-Station base preserved
# =============================================================================
BINANCE_API_BASE = "https://api.binance.us/api/v3"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

class TitanEngine:
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_binance_bases() -> List[str]:
        try:
            r = requests.get(f"{BINANCE_API_BASE}/exchangeInfo", headers=HEADERS, timeout=5)
            if r.status_code != 200:
                return []
            js = r.json()
            bases = set()
            for s in js.get("symbols", []):
                if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT":
                    bases.add(s.get("baseAsset").upper())
            return sorted(list(bases))
        except Exception:
            return []

    @staticmethod
    @st.cache_data(ttl=5, show_spinner=False)
    def get_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
        try:
            r = requests.get(
                f"{BINANCE_API_BASE}/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                headers=HEADERS,
                timeout=5
            )
            if r.status_code == 200:
                df = pd.DataFrame(r.json(), columns=["t","o","h","l","c","v","T","q","n","V","Q","B"])
                df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
                df[["open","high","low","close","volume"]] = df[["o","h","l","c","v"]].astype(float)
                return df[["timestamp","open","high","low","close","volume"]]
        except Exception:
            pass
        return pd.DataFrame()

    @staticmethod
    def get_ma(series: pd.Series, length: int, ma_type: str) -> pd.Series:
        if ma_type == "SMA":
            return series.rolling(length).mean()
        elif ma_type == "EMA":
            return series.ewm(span=length, adjust=False).mean()
        elif ma_type == "RMA":
            return series.ewm(alpha=1/length, adjust=False).mean()
        else:  # HMA
            half_len = int(length / 2)
            sqrt_len = int(math.sqrt(length))
            wma_f = series.rolling(length).mean()
            wma_h = series.rolling(half_len).mean()
            diff = 2 * wma_h - wma_f
            return diff.rolling(sqrt_len).mean()

    @staticmethod
    def calculate_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
        df = df.copy()
        df["up_move"] = df["high"] - df["high"].shift(1)
        df["down_move"] = df["low"].shift(1) - df["low"]
        df["pdm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
        df["ndm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1)))
        )
        df["tr_s"] = df["tr"].rolling(length).sum()
        df["pdm_s"] = df["pdm"].rolling(length).sum()
        df["ndm_s"] = df["ndm"].rolling(length).sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            df["pdi"] = 100 * (df["pdm_s"] / (df["tr_s"] + 1e-9))
            df["ndi"] = 100 * (df["ndm_s"] / (df["tr_s"] + 1e-9))
            df["dx"] = 100 * abs(df["pdi"] - df["ndi"]) / (df["pdi"] + df["ndi"] + 1e-9)
        return df["dx"].rolling(length).mean()

    @staticmethod
    def calculate_wavetrend(df: pd.DataFrame, chlen: int = 10, avg: int = 21) -> pd.Series:
        ap = (df["high"] + df["low"] + df["close"]) / 3
        esa = ap.ewm(span=chlen, adjust=False).mean()
        d = (ap - esa).abs().ewm(span=chlen, adjust=False).mean()
        ci = (ap - esa) / (0.015 * d + 1e-9)
        tci = ci.ewm(span=avg, adjust=False).mean()
        return tci

    @staticmethod
    def calculate_fear_greed(df: pd.DataFrame) -> int:
        try:
            df = df.copy()
            df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
            if len(df) < 90:
                return 50
            vol_score = 50 - ((df["log_ret"].rolling(30).std().iloc[-1] - df["log_ret"].rolling(90).std().iloc[-1]) / (df["log_ret"].rolling(90).std().iloc[-1] + 1e-9)) * 100
            vol_score = max(0, min(100, float(vol_score)))
            rsi = float(df["rsi"].iloc[-1])
            sma_50 = float(df["close"].rolling(50).mean().iloc[-1])
            dist = (float(df["close"].iloc[-1]) - sma_50) / (sma_50 + 1e-9)
            trend_score = 50 + (dist * 1000)
            return int((vol_score * 0.3) + (rsi * 0.4) + (max(0, min(100, float(trend_score))) * 0.3))
        except Exception:
            return 50

    @staticmethod
    @st.cache_data(show_spinner=True)
    def run_engine(df: pd.DataFrame, amp: int, dev: float, hma_l: int, gann_l: int, apex_len: int, apex_mult: float, liq_len: int):
        if df.empty:
            return df, []

        df = df.copy().reset_index(drop=True)

        # Basics
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1)))
        )
        df["atr"] = df["tr"].ewm(alpha=1/14, adjust=False).mean()
        df["hma"] = TitanEngine.get_ma(df["close"], hma_l, "HMA")

        # Momentum
        delta = df["close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
        df["rsi"] = 100 - (100 / (1 + (gain/(loss + 1e-9))))
        df["rvol"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)

        # Money Flow
        rsi_source = df["rsi"] - 50
        vol_sma = df["volume"].rolling(14).mean()
        df["money_flow"] = (rsi_source * (df["volume"] / (vol_sma + 1e-9))).ewm(span=3).mean()

        # Squeeze
        bb_basis = df["close"].rolling(20).mean()
        bb_dev = df["close"].rolling(20).std() * 2.0
        kc_basis = df["close"].rolling(20).mean()
        kc_dev = df["atr"] * 1.5
        df["in_squeeze"] = ((bb_basis - bb_dev) > (kc_basis - kc_dev)) & ((bb_basis + bb_dev) < (kc_basis + kc_dev))

        # Titan Trend (trail stop logic)
        df["ll"] = df["low"].rolling(amp).min()
        df["hh"] = df["high"].rolling(amp).max()
        trend = np.zeros(len(df))
        stop = np.full(len(df), np.nan)
        curr_t = 0
        curr_s = np.nan

        for i in range(amp, len(df)):
            c = float(df.at[i, "close"])
            d = float(df.at[i, "atr"]) * dev
            if curr_t == 0:
                s = float(df.at[i, "ll"]) + d
                curr_s = max(curr_s, s) if not np.isnan(curr_s) else s
                if c < curr_s:
                    curr_t = 1
                    curr_s = float(df.at[i, "hh"]) - d
            else:
                s = float(df.at[i, "hh"]) - d
                curr_s = min(curr_s, s) if not np.isnan(curr_s) else s
                if c > curr_s:
                    curr_t = 0
                    curr_s = float(df.at[i, "ll"]) + d
            trend[i] = curr_t
            stop[i] = curr_s

        df["is_bull"] = trend == 0
        df["entry_stop"] = stop

        # Apex SMC
        df["apex_base"] = TitanEngine.get_ma(df["close"], apex_len, "HMA")
        df["apex_upper"] = df["apex_base"] + (df["atr"] * apex_mult)
        df["apex_lower"] = df["apex_base"] - (df["atr"] * apex_mult)
        df["apex_adx"] = TitanEngine.calculate_adx(df)
        df["apex_tci"] = TitanEngine.calculate_wavetrend(df)

        apex_trend = np.zeros(len(df))
        apex_trail = np.full(len(df), np.nan)
        visual_zones: List[Dict[str, Any]] = []
        curr_at = 0
        curr_tr = np.nan

        for i in range(max(apex_len, liq_len, 20), len(df)):
            c = float(df.at[i, "close"])
            if c > float(df.at[i, "apex_upper"]):
                curr_at = 1
            elif c < float(df.at[i, "apex_lower"]):
                curr_at = -1
            apex_trend[i] = curr_at

            atr2 = float(df.at[i, "atr"]) * 2.0
            if curr_at == 1:
                val = c - atr2
                curr_tr = max(curr_tr, val) if not np.isnan(curr_tr) else val
                if apex_trend[i-1] == -1:
                    curr_tr = val
            elif curr_at == -1:
                val = c + atr2
                curr_tr = min(curr_tr, val) if not np.isnan(curr_tr) else val
                if apex_trend[i-1] == 1:
                    curr_tr = val
            apex_trail[i] = curr_tr

            # Pivot zones
            p_idx = i - liq_len
            is_ph = True
            for k in range(1, liq_len + 1):
                if float(df.at[p_idx, "high"]) <= float(df.at[p_idx-k, "high"]) or float(df.at[p_idx, "high"]) <= float(df.at[p_idx+k, "high"]):
                    is_ph = False
                    break
            if is_ph:
                visual_zones.append({
                    "type": "SUPPLY",
                    "x0": df.at[p_idx, "timestamp"],
                    "x1": df.at[i, "timestamp"],
                    "y0": float(df.at[p_idx, "high"]),
                    "y1": float(df.at[p_idx, "close"]),
                    "color": "rgba(229, 57, 53, 0.3)"
                })

            is_pl = True
            for k in range(1, liq_len + 1):
                if float(df.at[p_idx, "low"]) >= float(df.at[p_idx-k, "low"]) or float(df.at[p_idx, "low"]) >= float(df.at[p_idx+k, "low"]):
                    is_pl = False
                    break
            if is_pl:
                visual_zones.append({
                    "type": "DEMAND",
                    "x0": df.at[p_idx, "timestamp"],
                    "x1": df.at[i, "timestamp"],
                    "y0": float(df.at[p_idx, "low"]),
                    "y1": float(df.at[p_idx, "close"]),
                    "color": "rgba(67, 160, 71, 0.3)"
                })

        df["apex_trend"] = apex_trend
        df["apex_trail"] = apex_trail

        # Flux (missing flux fix preserved)
        rg = df["high"] - df["low"]
        body = np.abs(df["close"] - df["open"])
        eff_raw = np.where(rg == 0, 0, body / (rg + 1e-9))
        eff_sm = pd.Series(eff_raw, index=df.index).ewm(span=14).mean()
        vol_avg = df["volume"].rolling(55).mean()
        v_rat = np.where(vol_avg == 0, 1, df["volume"] / (vol_avg + 1e-9))
        direction = np.sign(df["close"] - df["open"])
        raw = direction * eff_sm * pd.Series(v_rat, index=df.index)
        df["Apex_Flux"] = raw.ewm(span=5).mean()

        # Gann
        sma_h = df["high"].rolling(gann_l).mean()
        sma_l = df["low"].rolling(gann_l).mean()
        g_trend = np.zeros(len(df))
        curr_g = 1
        for i in range(gann_l, len(df)):
            if curr_g == 1:
                if float(df.at[i, "close"]) < float(sma_l.iloc[i-1]):
                    curr_g = -1
            else:
                if float(df.at[i, "close"]) > float(sma_h.iloc[i-1]):
                    curr_g = 1
            g_trend[i] = curr_g
        df["gann_trend"] = g_trend

        # Targets
        risk = abs(df["close"] - df["entry_stop"])
        risk = risk.mask(risk == 0, df["close"] * 0.01)
        df["tp1"] = np.where(df["is_bull"], df["close"] + 1.5 * risk, df["close"] - 1.5 * risk)
        df["tp2"] = np.where(df["is_bull"], df["close"] + 3.0 * risk, df["close"] - 3.0 * risk)
        df["tp3"] = np.where(df["is_bull"], df["close"] + 5.0 * risk, df["close"] - 5.0 * risk)

        # VWAP
        df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
        df["vol_tp"] = df["tp"] * df["volume"]
        df["vwap"] = df["vol_tp"].cumsum() / (df["volume"].cumsum() + 1e-9)

        if len(visual_zones) > 20:
            visual_zones = visual_zones[-20:]

        return df, visual_zones

    @staticmethod
    def detect_special_setups(df: pd.DataFrame) -> Dict[str, bool]:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        setups = {"squeeze_breakout": False, "gann_reversal": False, "rvol_ignition": False}
        if bool(prev["in_squeeze"]) and not bool(last["in_squeeze"]):
            setups["squeeze_breakout"] = True
        if float(last["gann_trend"]) != float(prev["gann_trend"]):
            setups["gann_reversal"] = True
        if float(last["rvol"]) > 3.0:
            setups["rvol_ignition"] = True
        return setups

    @staticmethod
    def generate_mobile_report(row: pd.Series, fg_index: int, special_setups: Dict[str, bool]) -> str:
        is_bull = bool(row["is_bull"])
        direction_text = "LONG 🐂" if is_bull else "SHORT 🐻"

        titan_s = 1 if is_bull else -1
        apex_s = int(row["apex_trend"]) if not np.isnan(row["apex_trend"]) else 0
        gann_s = int(row["gann_trend"]) if not np.isnan(row["gann_trend"]) else 0
        score = 0
        if titan_s == apex_s:
            score += 1
        if titan_s == gann_s:
            score += 1
        conf_text = "LOW"
        if score == 2:
            conf_text = "HIGH"
        elif score == 1:
            conf_text = "MEDIUM"

        squeeze_active = bool(row["in_squeeze"])
        sqz_text = "SQUEEZE ACTIVE" if squeeze_active else "NO SQUEEZE"
        sqz_icon = "⚠️" if squeeze_active else "⚪"
        sqz_class = "value-yellow" if squeeze_active else "value-cyan"

        rvol_val = float(row["rvol"])
        rvol_desc = "(Normal)"
        if rvol_val > 2.0:
            rvol_desc = "(Ignition)"
        elif rvol_val < 0.5:
            rvol_desc = "(Low)"

        vwap_rel = "Above" if float(row["close"]) > float(row["vwap"]) else "Below"
        vwap_color_class = "value-cyan"

        # Special setup badges (preserved feature intent)
        setup_badges = []
        if special_setups.get("squeeze_breakout"):
            setup_badges.append("💥 SQUEEZE BREAKOUT")
        if special_setups.get("gann_reversal"):
            setup_badges.append("🔁 GANN REVERSAL")
        if special_setups.get("rvol_ignition"):
            setup_badges.append("🚀 RVOL IGNITION")
        setup_line = " | ".join(setup_badges) if setup_badges else "—"

        return f"""
        <div class="report-card" style="border-left: 4px solid #38bdf8;">
            <div class="report-header">
                <span>💠 SIGNAL: {direction_text}</span>
            </div>
            <div class="report-item">Confidence: <span class="value-cyan">{conf_text}</span></div>
            <div class="report-item">Sentiment: <span class="value-cyan">{fg_index}/100</span></div>
            <div class="report-item">Squeeze: {sqz_icon} <span class="{sqz_class}">{sqz_text}</span></div>
            <div class="report-item">Setups: <span class="value-cyan">{setup_line}</span></div>
        </div>

        <div class="report-card" style="border-left: 4px solid #38bdf8;">
            <div class="report-header">
                <span>🌊 FLOW & VOL</span>
            </div>
            <div class="report-item">RVOL: <span class="value-cyan">{rvol_val:.2f} {rvol_desc}</span></div>
            <div class="report-item">Money Flow: <span class="value-cyan">{float(row['money_flow']):.2f}</span></div>
            <div class="report-item">VWAP Relation: <span class="{vwap_color_class}">{vwap_rel}</span></div>
        </div>

        <div class="report-card" style="border-left: 4px solid #38bdf8;">
            <div class="report-header">
                <span>🎯 EXECUTION PLAN</span>
            </div>
            <div class="report-item">Entry: <span class="value-cyan">{float(row['close']):.4f}</span></div>
            <div class="report-item">🛑 SMART STOP: <span class="value-cyan">{float(row['entry_stop']):.4f}</span></div>
            <div class="report-item">1️⃣ TP1 (1.5R): <span class="value-cyan">{float(row['tp1']):.4f}</span></div>
            <div class="report-item">2️⃣ TP2 (3.0R): <span class="value-cyan">{float(row['tp2']):.4f}</span></div>
            <div class="report-item">3️⃣ TP3 (5.0R): <span class="value-cyan">{float(row['tp3']):.4f}</span></div>
        </div>
        """

# =============================================================================
# 12) AXIOM ENGINE (STOCKS / PHYSICS) — Mega-Station base preserved + upgraded
# =============================================================================
class AxiomEngine:
    @staticmethod
    def get_asset_classes():
        return [
            "Crypto (Major)", "Crypto (Alt/Meme)", "Indices & ETFs",
            "US Tech (Mag 7+)", "US Large Cap", "High Volatility",
            "Commodities & Forex"
        ]

    @staticmethod
    def get_tickers_by_class(asset_class):
        universe = {
            "Crypto (Major)": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD"],
            "Crypto (Alt/Meme)": ["SHIB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FLOKI-USD", "DOGE-USD"],
            "Indices & ETFs": ["SPY", "QQQ", "IWM", "DIA", "TLT", "VXX", "UVXY", "SQQQ", "TQQQ", "SOXL"],
            "US Tech (Mag 7+)": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD"],
            "US Large Cap": ["JPM", "BAC", "WFC", "JNJ", "LLY", "PFE", "PG", "KO", "PEP", "COST", "WMT"],
            "High Volatility": ["MSTR", "COIN", "MARA", "RIOT", "CLSK", "GME", "AMC", "HOOD", "DKNG"],
            "Commodities & Forex": ["GLD", "SLV", "USO", "EURUSD=X", "GBPUSD=X", "JPY=X", "GC=F", "CL=F"]
        }
        return universe.get(asset_class, [])

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_data(ticker, timeframe, limit=500):
        tf_map = {"15m": "1mo", "1h": "6mo", "4h": "1y", "1d": "2y", "1wk": "5y"}
        period = tf_map.get(timeframe, "1y")
        try:
            df = yf.download(ticker, period=period, interval=("1h" if timeframe == "4h" else timeframe), progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df.xs(ticker, axis=1, level=0)
                except Exception:
                    df.columns = df.columns.get_level_values(0)

            # 4h resample fix (Terminal-v2 parity)
            if timeframe == "4h" and not df.empty:
                df = df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()

            df = df.rename(columns={c: c.capitalize() for c in df.columns})
            if "Adj close" in df.columns:
                df["Close"] = df["Adj close"]
            return df.dropna().tail(limit)
        except Exception:
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_fundamentals(ticker):
        if "-" in ticker or "=" in ticker:
            return None
        try:
            stock = yf.Ticker(ticker)
            return {"Market Cap": stock.info.get("marketCap", "N/A"), "P/E Ratio": stock.info.get("trailingPE", "N/A")}
        except Exception:
            return None

    @staticmethod
    def tanh(x):
        return np.tanh(np.clip(x, -20, 20))

    @staticmethod
    def calc_chedo(df, length=50):
        c = df["Close"].values
        log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
        mu = pd.Series(log_ret).rolling(length).mean().values
        sigma = pd.Series(log_ret).rolling(length).std().values
        v = sigma / (np.abs(mu) + 1e-9)
        abs_ret_v = np.abs(log_ret) * v
        hyper_dist = np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))
        kappa_h = AxiomEngine.tanh(pd.Series(hyper_dist).rolling(length).mean().values)
        diff_ret = np.diff(log_ret, prepend=0)
        lyap = np.log(np.abs(diff_ret) + 1e-9)
        lambda_n = AxiomEngine.tanh((pd.Series(lyap).rolling(length).mean().values + 5) / 7)
        ent = pd.Series(log_ret**2).rolling(length).sum().values
        ent_n = AxiomEngine.tanh(ent * 10)
        raw = (0.4 * kappa_h) + (0.3 * lambda_n) + (0.3 * ent_n)
        df["CHEDO"] = 2 / (1 + np.exp(-raw * 4)) - 1
        return df

    @staticmethod
    def calc_rqzo(df, harmonics=25):
        src = df["Close"]
        mn, mx = src.rolling(100).min(), src.rolling(100).max()
        norm = (src - mn) / (mx - mn + 1e-9)
        v = np.abs(norm.diff())
        c_limit = 0.05
        gamma = 1 / np.sqrt(1 - (np.minimum(v, c_limit*0.99)/c_limit)**2)
        idx = np.arange(len(df))
        tau = (idx % 100) / gamma.fillna(1.0)
        zeta = np.zeros(len(df))
        for n in range(1, harmonics + 1):
            amp = n ** -0.5
            theta = tau * np.log(n)
            zeta += amp * np.sin(theta)
        df["RQZO"] = pd.Series(zeta, index=df.index).fillna(0)
        return df

    @staticmethod
    def calc_apex_flux(df, length=14):
        rg = df["High"] - df["Low"]
        body = np.abs(df["Close"] - df["Open"])
        eff_raw = np.where(rg == 0, 0, body / (rg + 1e-9))
        eff_sm = pd.Series(eff_raw, index=df.index).ewm(span=length).mean()
        vol_avg = df["Volume"].rolling(55).mean()
        v_rat = np.where(vol_avg == 0, 1, df["Volume"] / (vol_avg + 1e-9))
        direction = np.sign(df["Close"] - df["Open"])
        raw = direction * eff_sm * pd.Series(v_rat, index=df.index)
        df["Apex_Flux"] = raw.ewm(span=5).mean()
        df["Apex_State"] = np.select(
            [df["Apex_Flux"] > 0.6, df["Apex_Flux"] < -0.6],
            ["Super Bull", "Super Bear"],
            default="Neutral"
        )
        return df

    @staticmethod
    def calc_smc(df, length=55):
        def wma(s, l):
            w = np.arange(1, l+1)
            return s.rolling(l).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)

        close = df["Close"]
        half = int(length/2)
        sqrt = int(np.sqrt(length))
        wma_f = wma(close, length)
        wma_h = wma(close, half)
        df["HMA_Trend"] = wma(2*wma_h - wma_f, sqrt)
        df["Trend_Dir"] = np.where(close > df["HMA_Trend"], 1, -1)
        return df

    @staticmethod
    @st.cache_data(ttl=600, show_spinner=False)
    def run_monte_carlo(df, days=30, sims=150):
        last_price = float(df["Close"].iloc[-1])
        returns = df["Close"].pct_change().dropna()
        mu = float(returns.mean())
        sigma = float(returns.std())
        sim_rets = np.random.normal(mu, sigma, (days, sims))
        paths = np.zeros((days, sims))
        paths[0] = last_price
        for t in range(1, days):
            paths[t] = paths[t-1] * (1 + sim_rets[t])
        return paths

    @staticmethod
    def calc_volume_profile(df, bins=50):
        return calc_volume_profile(df, bins=bins)

    @staticmethod
    def calc_day_of_week_dna_simple(ticker):
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            df["Day"] = df.index.day_name()
            df["Ret"] = df["Close"].pct_change() * 100
            return df.groupby("Day")["Ret"].mean()
        except Exception:
            return None

    @staticmethod
    def analyze_ai(ticker, price, chedo, rqzo, flux, api_key):
        if not api_key:
            return "❌ Missing OpenAI API Key."
        prompt = (
            f"Analyze {ticker} at {price}. Metrics: Entropy (CHEDO) {chedo:.2f}, "
            f"Relativity (RQZO) {rqzo:.2f}, Flux {flux:.2f}. "
            f">0.8 Entropy is chaos. >0.6 Flux is breakout. Brief strategy."
        )
        try:
            client = OpenAI(api_key=api_key)
            return client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
        except Exception as e:
            return f"AI Error: {e}"

# =============================================================================
# 13) UI PANELS: Macro Dashboard + TV Inbox + Setup
# =============================================================================
def render_macro_40_panel():
    macro_groups, m_price, m_chg = get_macro_data_40()
    st.subheader("🌍 Global Macro Dashboard (40-panel)")
    if not m_price:
        st.info("Macro feed unavailable (yfinance returned empty).")
        return

    group_names = list(macro_groups.keys())
    for i in range(0, len(group_names), 2):
        cols = st.columns(2)
        g1 = group_names[i]
        with cols[0].container(border=True):
            st.markdown(f"#### {g1}")
            sc = st.columns(4)
            for x, (n, sym) in enumerate(macro_groups[g1].items()):
                fmt = "{:.3f}" if any(k in n for k in ["Yield", "GBP", "EUR", "JPY"]) else "{:,.2f}"
                sc[x].metric(n.split("(")[0], fmt.format(m_price.get(n, 0)), f"{m_chg.get(n, 0):.2f}%")
        if i + 1 < len(group_names):
            g2 = group_names[i + 1]
            with cols[1].container(border=True):
                st.markdown(f"#### {g2}")
                sc = st.columns(4)
                for x, (n, sym) in enumerate(macro_groups[g2].items()):
                    fmt = "{:.3f}" if any(k in n for k in ["Yield", "GBP", "EUR", "JPY"]) else "{:,.2f}"
                    sc[x].metric(n.split("(")[0], fmt.format(m_price.get(n, 0)), f"{m_chg.get(n, 0):.2f}%")

def render_tv_inbox_panel(tg_token: str, tg_chat: str, tg_parse_mode_label: str):
    init_tv_db()

    st.markdown("---")
    st.subheader("🛰️ TradingView Alerts Inbox (SQLite)")
    st.caption("Reads alerts stored in local SQLite. External webhook receiver recommended (FastAPI).")

    cols_inbox = st.columns([0.35, 0.65])
    with cols_inbox[0]:
        inbox_limit = st.slider("Rows", 10, 200, 50)
        _ = st.button("Refresh Inbox")
        send_selected = st.button("Send Selected Alert to Telegram")
        if "last_broadcast_dedup" not in st.session_state:
            st.session_state.last_broadcast_dedup = ""
        st.write("**Last Broadcast dedup_key:**")
        st.code(st.session_state.last_broadcast_dedup if st.session_state.last_broadcast_dedup else "(none)")

    tv_df = fetch_recent_tv_alerts(limit=inbox_limit, db_path=TV_DB_PATH)
    if tv_df is None or tv_df.empty:
        st.info("No stored alerts found. If you have a receiver, ensure it writes into tv_alerts.db using the same schema.")
    else:
        tv_df_display = tv_df.copy()
        try:
            tv_df_display["ts_readable"] = tv_df_display["ts"].apply(
                lambda x: datetime.utcfromtimestamp(int(x)/1000).strftime("%Y-%m-%d %H:%M:%S UTC") if pd.notna(x) else ""
            )
        except Exception:
            tv_df_display["ts_readable"] = ""

        with cols_inbox[1]:
            st.caption("Select a row index to broadcast that alert.")
            selected_idx = st.number_input(
                "Selected row index (0 = top)",
                min_value=0,
                max_value=max(0, len(tv_df_display) - 1),
                value=0,
                step=1
            )

        st.dataframe(tv_df_display, use_container_width=True)

        if send_selected:
            if not tg_token or not tg_chat:
                st.warning("⚠️ Enter Telegram keys in the sidebar.")
            else:
                try:
                    chat_ids = _parse_chat_ids(tg_chat)
                    row = tv_df_display.iloc[int(selected_idx)]
                    text_lines = [
                        "🛰️ TradingView Alert",
                        f"Symbol: {row.get('symbol','')}",
                        f"TF: {row.get('tf','')}",
                        f"Side: {row.get('side','')}",
                        f"Confidence: {row.get('confidence','')}",
                        f"Entry: {row.get('entry','')}",
                        f"Stop: {row.get('stop','')}",
                        f"TP1: {row.get('tp1','')}",
                        f"TP2: {row.get('tp2','')}",
                        f"Reason: {row.get('reason','')}",
                        f"Time: {row.get('ts_readable','')}",
                        f"dedup_key: {row.get('dedup_key','')}",
                    ]
                    out = "\n".join(text_lines)

                    parse_mode = None
                    outbound_text = out
                    if tg_parse_mode_label == "MarkdownV2 (Escaped)":
                        parse_mode = "MarkdownV2"
                        outbound_text = tg_escape_markdown_v2(out)

                    for cid in chat_ids:
                        tg_send_message(tg_token, cid, outbound_text, parse_mode=parse_mode)

                    st.session_state.last_broadcast_dedup = str(row.get("dedup_key", ""))
                    st.success("✅ Alert sent.")
                except Exception as e:
                    st.error(f"Failed to send alert: {e}")

    with st.expander("🧾 Manual Ingest (Paste JSON) — Optional", expanded=False):
        st.caption("Test the inbox without a receiver by pasting a TradingView-style JSON payload here.")
        raw_json = st.text_area(
            "Raw JSON",
            value='{"v":1,"src":"tv","symbol":"BINANCE:BTCUSDT","tf":"60","side":"BUY","confidence":65,"entry":99999.0,"stop":99000.0,"tp1":101000.0,"tp2":102000.0,"dedup_key":"BTCUSDT_60_BUY_1","reason":"Example","ts":1700000000000}',
            height=120
        )
        if st.button("Ingest JSON into SQLite"):
            try:
                alert = parse_tv_alert(raw_json)
                if not alert.dedup_key:
                    st.warning("dedup_key is missing. Provide one to store.")
                else:
                    upsert_tv_alert(alert, raw_json, db_path=TV_DB_PATH)
                    st.success("✅ Stored alert in SQLite.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

def render_setup_panel():
    st.subheader("🧪 Diagnostics & Setup")
    st.write("**Secrets status** (recommended):")
    try:
        st.write(f"- OPENAI_API_KEY in secrets: {'✅' if 'OPENAI_API_KEY' in st.secrets else '❌'}")
        st.write(f"- TELEGRAM_TOKEN in secrets: {'✅' if 'TELEGRAM_TOKEN' in st.secrets else '❌'}")
        st.write(f"- TELEGRAM_CHAT_ID in secrets: {'✅' if 'TELEGRAM_CHAT_ID' in st.secrets else '❌'}")
    except Exception:
        st.write("- Secrets not available in this environment.")
    st.write("**SQLite Alerts DB:**")
    st.code(TV_DB_PATH)
    st.write("**TradingView Webhook note:** Streamlit typically cannot receive webhooks directly. Use an external receiver and point it at this DB.")
    st.write("**External receiver skeleton (FastAPI):**")
    st.code(
        "from fastapi import FastAPI, Request, HTTPException\n"
        "import uvicorn, sqlite3, time, json\n"
        "from dataclasses import dataclass\n"
        "app = FastAPI()\n"
        "# parse json, upsert into sqlite (same schema), run uvicorn.\n",
        language="python"
    )

# =============================================================================
# 14) MAIN CONTROLLER (Mega-Station base preserved + integrated panels)
# =============================================================================
def main():
    init_tv_db()

    # Header
    st.markdown('<div class="title-glow">💠 TITAN-AXIOM MEGA-STATION</div>', unsafe_allow_html=True)
    st.caption("Diamond-Visual • Titan Mobile (Binance) + Axiom Quant (YFinance) + Terminal-v2 Parity Add-ons")

    # Sidebar Controller (preserved base)
    st.sidebar.header("💠 MEGA-STATION")
    mode = st.sidebar.radio("ENGINE MODE", ["TITAN MOBILE (Crypto)", "AXIOM QUANT (Stocks)"])

    with st.sidebar.expander("🔐 KEYS"):
        tg_token = st.text_input("Telegram Bot Token", value=SecretsManager.get("TELEGRAM_TOKEN"), type="password")
        tg_chat = st.text_input("Telegram Chat ID(s) (comma-separated supported)", value=SecretsManager.get("TELEGRAM_CHAT_ID"))
        ai_key = st.text_input("OpenAI API Key", value=SecretsManager.get("OPENAI_API_KEY"), type="password")

        st.markdown("**Telegram Parse Mode**")
        tg_parse_mode_label = st.selectbox("Parse Mode", ["None (Plain Text)", "MarkdownV2 (Escaped)"], index=0)

    st.sidebar.markdown("---")
    with st.sidebar.expander("🧪 Setup / Diagnostics", expanded=False):
        render_setup_panel()

    # -------------------------------------------------------------------------
    # MODE 1: TITAN MOBILE (Binance)
    # -------------------------------------------------------------------------
    if mode == "TITAN MOBILE (Crypto)":
        st.sidebar.subheader("📡 BINANCE FEED")
        bases = TitanEngine.get_binance_bases()
        idx = bases.index("BTC") if "BTC" in bases else 0
        base = st.sidebar.selectbox("Asset", bases if bases else ["BTC"], index=min(idx, max(0, len(bases)-1)))
        ticker = f"{base}USDT"

        c1, c2 = st.sidebar.columns(2)
        with c1:
            timeframe = st.selectbox("TF", ["15m", "1h", "4h", "1d"], index=1)
        with c2:
            limit = st.slider("Depth", 100, 500, 200, 50)

        st.sidebar.markdown("---")
        st.sidebar.subheader("🧠 LOGIC")
        amp = st.sidebar.number_input("Amplitude", 2, 100, 10)
        dev = st.sidebar.number_input("Deviation", 0.5, 5.0, 3.0)
        hma_len = st.sidebar.number_input("HMA Len", 10, 200, 50)
        gann_len = st.sidebar.number_input("Gann Len", 2, 50, 3)

        st.title(f"💠 TITAN: {base}")
        Visuals.render_titan_clock()
        Visuals.render_titan_tape(ticker)

        # Tabs: Titan + TradingView + Inbox
        t_tabs = st.tabs(["📊 Titan Mobile", "📺 TradingView", "🛰️ TV Alerts Inbox", "🌍 Macro 40", "⚙️ Broadcast Center"])

        with t_tabs[0]:
            with st.spinner("Connecting to Binance..."):
                df = TitanEngine.get_klines(ticker, timeframe, int(limit))

            if not df.empty:
                df, zones = TitanEngine.run_engine(df, int(amp), float(dev), int(hma_len), int(gann_len), 55, 1.5, 10)
                last = df.iloc[-1]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("TREND", "BULL 🟢" if bool(last["is_bull"]) else "BEAR 🔴")
                c2.metric("FLUX", f"{float(last['Apex_Flux']):.2f}")
                c3.metric("STOP", f"{float(last['entry_stop']):.4f}")
                c4.metric("TP3", f"{float(last['tp3']):.4f}")

                fg = TitanEngine.calculate_fear_greed(df)
                spec = TitanEngine.detect_special_setups(df)
                st.markdown(TitanEngine.generate_mobile_report(last, fg, spec), unsafe_allow_html=True)

                st.markdown("### 📈 Chart")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(
                    x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["hma"], line=dict(color="#00F0FF", width=1), name="HMA"
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["apex_trail"], line=dict(color="orange", width=1, dash="dot"), name="Trail"
                ), row=1, col=1)

                for z in zones:
                    fig.add_shape(type="rect", x0=z["x0"], x1=z["x1"], y0=z["y0"], y1=z["y1"], fillcolor=z["color"], line_width=0, row=1, col=1)

                colors = np.where(df["Apex_Flux"] > 0, "#00E676", "#FF1744")
                fig.add_trace(go.Bar(
                    x=df["timestamp"], y=df["Apex_Flux"], marker_color=colors, name="Flux"
                ), row=2, col=1)

                fig.update_layout(
                    height=560, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0),
                    hovermode="x unified", xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Binance connection failed or returned empty data.")

        with t_tabs[1]:
            st.subheader("📺 TradingView Advanced Chart")
            Visuals.render_tradingview_advanced_chart(f"BINANCE:{ticker}", timeframe)

        with t_tabs[2]:
            render_tv_inbox_panel(tg_token=tg_token, tg_chat=tg_chat, tg_parse_mode_label=tg_parse_mode_label)

        with t_tabs[3]:
            render_macro_40_panel()

        with t_tabs[4]:
            st.subheader("📢 Broadcast Center (Titan Mobile)")
            st.caption("Enhanced Telegram broadcaster (multi-chat, optional screenshot, safe chunking).")

            uploaded_file = st.file_uploader("Upload Chart Screenshot (Optional)", type=["png", "jpg", "jpeg"])
            msg_default = f"🚀 TITAN SIGNAL\nSymbol: {ticker}\nTF: {timeframe}\n(Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')})"
            msg = st.text_area("Message Preview", value=msg_default, height=140)

            if st.button("Send to Telegram 🚀"):
                if not tg_token or not tg_chat:
                    st.warning("⚠️ Enter Telegram keys in sidebar.")
                else:
                    chat_ids = _parse_chat_ids(tg_chat)
                    parse_mode = None
                    outbound_text = msg
                    if tg_parse_mode_label == "MarkdownV2 (Escaped)":
                        parse_mode = "MarkdownV2"
                        outbound_text = tg_escape_markdown_v2(msg)

                    # Photo first
                    if uploaded_file:
                        for cid in chat_ids:
                            tg_send_photo(
                                token=tg_token,
                                chat_id=cid,
                                photo_bytes=uploaded_file.getvalue(),
                                caption=(tg_escape_markdown_v2(f"🔥 TITAN: {ticker}") if parse_mode == "MarkdownV2" else f"🔥 TITAN: {ticker}"),
                                parse_mode=parse_mode
                            )

                    # Chunked message
                    max_length = 2000
                    for cid in chat_ids:
                        if len(outbound_text) <= max_length:
                            tg_send_message(tg_token, cid, outbound_text, parse_mode=parse_mode)
                        else:
                            part = 1
                            for i in range(0, len(outbound_text), max_length):
                                chunk = outbound_text[i:i+max_length]
                                prefix = f"(Part {part}) "
                                tg_send_message(tg_token, cid, prefix + chunk, parse_mode=parse_mode)
                                part += 1

                    st.success("✅ Sent to Telegram (chunk-safe).")

    # -------------------------------------------------------------------------
    # MODE 2: AXIOM QUANT (Stocks/YFinance) — upgraded with Terminal-v2 parity
    # -------------------------------------------------------------------------
    else:
        st.sidebar.subheader("📡 MARKET DATA")
        ac_list = AxiomEngine.get_asset_classes()
        ac = st.sidebar.selectbox("Sector", ac_list)
        ticks = AxiomEngine.get_tickers_by_class(ac)
        ticker = st.sidebar.selectbox("Ticker", ticks if ticks else ["SPY"])
        tf = st.sidebar.selectbox("TF", ["15m", "1h", "4h", "1d", "1wk"], index=3)

        st.title(f"💠 AXIOM: {ticker}")
        Visuals.render_axiom_clock()
        Visuals.render_axiom_banner()

        # Macro strip (quick)
        with st.expander("🌍 Macro 40 (Quick View)", expanded=False):
            render_macro_40_panel()

        with st.spinner("Crunching Physics + God Mode..."):
            df = AxiomEngine.fetch_data(ticker, tf)

        if df.empty:
            st.error("No data returned from yfinance.")
            return

        # Physics
        df = AxiomEngine.calc_chedo(df)
        df = AxiomEngine.calc_rqzo(df)
        df = AxiomEngine.calc_apex_flux(df)
        df = AxiomEngine.calc_smc(df)

        # God Mode overlay (Terminal-v2 parity)
        # We compute on the same OHLCV columns already present in df.
        df_gm = calc_fear_greed_v4(calc_indicators_godmode(df[["Open","High","Low","Close","Volume"]].copy()))
        # Merge GM columns into main df by index
        for col in df_gm.columns:
            if col not in df.columns:
                df[col] = df_gm[col]

        last = df.iloc[-1]
        fund = AxiomEngine.get_fundamentals(ticker)

        # Dashboard Metrics (expanded)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("PRICE", f"{float(last['Close']):.4f}")
        c2.metric("ENTROPY (CHEDO)", f"{float(last['CHEDO']):.2f}", delta="Risk" if abs(float(last["CHEDO"])) > 0.8 else "Stable")
        c3.metric("FLUX", f"{float(last['Apex_Flux']):.2f}", delta=str(last.get("Apex_State", "")))
        c4.metric("TREND (Physics)", "BULL" if int(last["Trend_Dir"]) == 1 else "BEAR")
        c5.metric("GM SCORE", f"{float(last.get('GM_Score', 0)):.0f}/5", delta="Bullish" if float(last.get("GM_Score", 0)) > 0 else "Bearish")

        # Tabs (Axiom upgraded to Terminal-v2 parity)
        tabs = st.tabs([
            "📉 TECH (Physics+GM)",
            "🤖 AI (God Mode)",
            "🧩 Correlation & MTF",
            "📅 Monthly Seasonality",
            "📆 Day & Hour DNA",
            "🏦 Smart Money Concepts",
            "📊 Volume Profile (VPVR)",
            "🔮 Monte Carlo",
            "📡 TradingView",
            "🛰️ TV Alerts Inbox",
            "🧪 Scanner",
            "⚙️ Broadcast Center"
        ])

        with tabs[0]:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.48, 0.17, 0.17, 0.18], vertical_spacing=0.02)
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["HMA_Trend"], line=dict(color="#fff", width=1, dash="dot"), name="HMA_Trend (Physics)"), row=1, col=1)

            # Apex cloud (GM)
            if "Apex_Upper" in df.columns and "Apex_Lower" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["Apex_Upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["Apex_Lower"], fill="tonexty", fillcolor="rgba(0,230,118,0.10)", line=dict(width=0), name="Apex Cloud (GM)"), row=1, col=1)

            # CHEDO
            fig.add_trace(go.Scatter(
                x=df.index, y=df["CHEDO"], line=dict(color="#00F0FF", width=2),
                fill="tozeroy", fillcolor="rgba(0,240,255,0.1)", name="Entropy (CHEDO)"
            ), row=2, col=1)

            # Squeeze Mom
            if "Sqz_Mom" in df.columns:
                sqz_colors = np.where(df["Sqz_Mom"] > 0, "#00E676", "#FF5252")
                fig.add_trace(go.Bar(x=df.index, y=df["Sqz_Mom"], marker_color=sqz_colors, name="Squeeze Mom"), row=3, col=1)

            # Flux
            flux_colors = np.where(df["Apex_Flux"] > 0.6, "#00E676", np.where(df["Apex_Flux"] < -0.6, "#FF1744", "#2979FF"))
            fig.add_trace(go.Bar(x=df.index, y=df["Apex_Flux"], marker_color=flux_colors, name="Flux"), row=4, col=1)

            fig.update_layout(height=820, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📌 Fundamentals", expanded=False):
                if fund:
                    st.write(f"Market Cap: {fund.get('Market Cap')} | P/E: {fund.get('P/E Ratio')}")
                else:
                    st.info("Fundamentals unavailable for this ticker.")

        with tabs[1]:
            st.subheader("🤖 AI Analyst (Grounded)")
            st.caption("Uses computed indicators only. No entries/exits/stops produced.")
            if st.button("RUN AI BRIEFING"):
                ai_text = ask_ai_analyst_godmode(df, ticker, get_fundamentals(ticker), tf, ai_key)
                st.info(ai_text)

        with tabs[2]:
            st.subheader("🧩 Correlation & Multi-Timeframe Radar")
            cA, cB = st.columns([0.4, 0.6])

            with cA:
                st.markdown("**📡 Multi-Timeframe Radar**")
                mtf_df = calc_mtf_trend(ticker)

                def color_trend(val):
                    color = "#00ff00" if val == "BULLISH" else "#ff0000" if val == "BEARISH" else "white"
                    return f"color: {color}; font-weight: bold"
                if "Trend" in mtf_df.columns:
                    st.dataframe(mtf_df.style.map(color_trend, subset=["Trend"]), use_container_width=True)
                else:
                    st.dataframe(mtf_df, use_container_width=True)

            with cB:
                st.markdown("**🔗 Macro Correlation Matrix (180 Days)**")
                corr = calc_correlations(ticker)
                if corr is None or corr.empty:
                    st.info("Correlation unavailable.")
                else:
                    fig_corr = px.bar(x=corr.values, y=corr.index, orientation="h", color=corr.values, color_continuous_scale="RdBu")
                    fig_corr.update_layout(template="plotly_dark", height=420, xaxis_title="Correlation Coefficient")
                    st.plotly_chart(fig_corr, use_container_width=True)

        with tabs[3]:
            st.subheader("📅 Monthly Seasonality")
            seas = get_seasonality_stats(ticker)
            if seas:
                hm, hold, month = seas
                fig_hm = px.imshow(hm, color_continuous_scale="RdYlGn", text_auto=".1f")
                fig_hm.update_layout(template="plotly_dark", height=520)
                st.plotly_chart(fig_hm, use_container_width=True)
                c1, c2 = st.columns(2)
                c1.dataframe(pd.DataFrame(hold).T.style.format("{:.1f}%").background_gradient(cmap="RdYlGn"), use_container_width=True)
                curr_m = datetime.now().month
                if curr_m in month.index:
                    c2.metric("Current Month Win Rate", f"{float(month.loc[curr_m, 'Win Rate']):.1f}%")
            else:
                st.info("Seasonality unavailable (insufficient monthly data).")

        with tabs[4]:
            st.subheader("📆 Day & Hour DNA")
            c1, c2 = st.columns(2)
            dna_res = calc_day_of_week_dna(ticker, 250, "Close to Close (Total)")
            if dna_res:
                cum, stats = dna_res
                with c1:
                    st.markdown("**Day of Week Performance**")
                    fig_dna = go.Figure()
                    for col in cum.columns:
                        fig_dna.add_trace(go.Scatter(x=cum.index, y=cum[col], name=col))
                    fig_dna.update_layout(template="plotly_dark", height=420)
                    st.plotly_chart(fig_dna, use_container_width=True)
                    st.dataframe(stats.style.background_gradient(subset=["Win Rate"], cmap="RdYlGn"), use_container_width=True)

            hourly_res = calc_intraday_dna(ticker)
            if hourly_res is not None:
                with c2:
                    st.markdown("**Intraday (Hourly) Performance**")
                    fig_hr = px.bar(hourly_res, x=hourly_res.index, y="Avg Return", color="Win Rate", color_continuous_scale="RdYlGn")
                    fig_hr.update_layout(template="plotly_dark", height=420)
                    st.plotly_chart(fig_hr, use_container_width=True)
                    st.dataframe(hourly_res.style.format("{:.2f}"), use_container_width=True)

        with tabs[5]:
            st.subheader("🏦 Smart Money Concepts (Overlay)")
            smc = calculate_smc_overlay(df[["Open","High","Low","Close","Volume"]].copy(), swing_length=5)
            fig_smc = go.Figure(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]))
            for ob in smc["order_blocks"]:
                fig_smc.add_shape(type="rect", x0=ob["x0"], x1=ob["x1"], y0=ob["y0"], y1=ob["y1"], fillcolor=ob["color"], opacity=0.5, line_width=0)
            for fvg in smc["fvgs"]:
                fig_smc.add_shape(type="rect", x0=fvg["x0"], x1=fvg["x1"], y0=fvg["y0"], y1=fvg["y1"], fillcolor=fvg["color"], opacity=0.5, line_width=0)
            for struct in smc["structures"]:
                fig_smc.add_shape(type="line", x0=struct["x0"], x1=struct["x1"], y0=struct["y"], y1=struct["y"], line=dict(color=struct["color"], width=1, dash="dot"))
                fig_smc.add_annotation(x=struct["x1"], y=struct["y"], text=struct["label"], showarrow=False, yshift=10 if struct["color"] == "green" else -10, font=dict(color=struct["color"], size=10))
            fig_smc.update_layout(height=650, template="plotly_dark", title="SMC Analysis")
            st.plotly_chart(fig_smc, use_container_width=True)

        with tabs[6]:
            st.subheader("📊 Volume Profile (VPVR)")
            vp, poc = calc_volume_profile(df[["Open","High","Low","Close","Volume"]].copy(), bins=50)
            fig_vp = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.7, 0.3])
            fig_vp.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]), row=1, col=1)
            fig_vp.add_trace(go.Bar(x=vp["Volume"], y=vp["Price"], orientation="h", marker_color="rgba(0,200,255,0.3)"), row=1, col=2)
            fig_vp.add_hline(y=poc, line_color="yellow")
            fig_vp.update_layout(height=650, template="plotly_dark", title="Volume Profile (VPVR)")
            st.plotly_chart(fig_vp, use_container_width=True)
            st.caption(f"POC: {poc:.4f}")

        with tabs[7]:
            st.subheader("🔮 Monte Carlo Forecast")
            mc = AxiomEngine.run_monte_carlo(df, days=30, sims=200)
            fig_mc = go.Figure()
            for i in range(min(60, mc.shape[1])):
                fig_mc.add_trace(go.Scatter(y=mc[:, i], mode="lines", line=dict(color="rgba(255,255,255,0.05)"), showlegend=False))
            fig_mc.add_trace(go.Scatter(y=np.mean(mc, axis=1), mode="lines", name="Mean", line=dict(color="orange")))
            fig_mc.update_layout(height=520, template="plotly_dark", title="Monte Carlo Forecast (30 Days)")
            st.plotly_chart(fig_mc, use_container_width=True)

        with tabs[8]:
            st.subheader("📡 TradingView Advanced Chart")
            Visuals.render_tradingview_advanced_chart(ticker, tf)

        with tabs[9]:
            render_tv_inbox_panel(tg_token=tg_token, tg_chat=tg_chat, tg_parse_mode_label=tg_parse_mode_label)

        with tabs[10]:
            st.subheader("🧪 Multi-Asset Scanner (God Mode Score)")
            st.caption("Scans a small universe using the God Mode indicator engine (yfinance).")

            scan_universe = [
                "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD",
                "SPY", "QQQ", "IWM", "TLT", "GLD", "CL=F", "^VIX", "NVDA", "TSLA", "AAPL", "MSFT"
            ]
            scan_limit = st.slider("Max assets to scan", 5, len(scan_universe), 10)
            scan_interval = st.selectbox("Scanner interval", ["1h", "1d", "1wk"], index=1)

            if st.button("Run Scanner"):
                rows = []
                for sym in scan_universe[:scan_limit]:
                    period = "6mo" if scan_interval != "1wk" else "3y"
                    sdf = safe_download(sym, period, scan_interval)
                    if sdf is None or sdf.empty:
                        rows.append({"Symbol": sym, "Status": "No Data"})
                        continue
                    try:
                        sdf = calc_fear_greed_v4(calc_indicators_godmode(sdf))
                        last_s = sdf.iloc[-1]
                        rows.append({
                            "Symbol": sym,
                            "Price": float(last_s["Close"]),
                            "GM_Score": float(last_s["GM_Score"]),
                            "Apex": "BULL" if int(last_s["Apex_Trend"]) == 1 else "BEAR",
                            "Vector": "BULL" if int(last_s["DarkVector_Trend"]) == 1 else "BEAR",
                            "Squeeze": bool(last_s["Squeeze_On"]),
                            "RSI": float(last_s["RSI"]),
                            "RVOL": float(last_s["RVOL"]),
                            "Status": "OK"
                        })
                    except Exception as e:
                        rows.append({"Symbol": sym, "Status": f"Err: {e}"})
                scan_df = pd.DataFrame(rows)
                st.dataframe(scan_df, use_container_width=True)

        with tabs[11]:
            st.subheader("📢 Broadcast Center (Axiom)")
            st.caption("Enhanced Telegram broadcaster (multi-chat, optional screenshot, safe chunking).")
            uploaded_file = st.file_uploader("Upload Chart Screenshot (Optional)", type=["png", "jpg", "jpeg"])
            msg_default = f"🔥 AXIOM REPORT\nTicker: {ticker}\nTF: {tf}\nPrice: {float(last['Close']):.4f}\nGM Score: {float(last.get('GM_Score',0)):.0f}/5\n(Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')})"
            msg = st.text_area("Message Preview", value=msg_default, height=160)

            if st.button("Send to Telegram 🚀"):
                if not tg_token or not tg_chat:
                    st.warning("⚠️ Enter Telegram keys in sidebar.")
                else:
                    chat_ids = _parse_chat_ids(tg_chat)
                    parse_mode = None
                    outbound_text = msg
                    if tg_parse_mode_label == "MarkdownV2 (Escaped)":
                        parse_mode = "MarkdownV2"
                        outbound_text = tg_escape_markdown_v2(msg)

                    if uploaded_file:
                        for cid in chat_ids:
                            tg_send_photo(
                                token=tg_token,
                                chat_id=cid,
                                photo_bytes=uploaded_file.getvalue(),
                                caption=(tg_escape_markdown_v2(f"🔥 AXIOM: {ticker}") if parse_mode == "MarkdownV2" else f"🔥 AXIOM: {ticker}"),
                                parse_mode=parse_mode
                            )

                    max_length = 2000
                    for cid in chat_ids:
                        if len(outbound_text) <= max_length:
                            tg_send_message(tg_token, cid, outbound_text, parse_mode=parse_mode)
                        else:
                            part = 1
                            for i in range(0, len(outbound_text), max_length):
                                chunk = outbound_text[i:i+max_length]
                                prefix = f"(Part {part}) "
                                tg_send_message(tg_token, cid, prefix + chunk, parse_mode=parse_mode)
                                part += 1

                    st.success("✅ Sent to Telegram (chunk-safe).")

if __name__ == "__main__":
    main()




# =============================================================================
# PATCH UPGRADE — DETAILED TELEGRAM SIGNALS + INTERNAL STRATEGY REPORTS + AI BRAIN
# -----------------------------------------------------------------------------
# ✅ No omissions from your base file required: this is an ADDITIVE PATCH.
# ✅ Keeps 100% of your current code intact.
# ✅ Implements:
#   - Detailed Telegram signal reports (Titan + Axiom) using INTERNAL indicators/strategies
#   - Optional AI Brain briefing that is grounded + contradiction-aware
#   - Robust multi-chat broadcast with MarkdownV2 escaping + safe chunking
#   - Optional auto-generated chart image (Plotly->PNG) with graceful fallback
#
# IMPORTANT:
# - Your existing “send_telegram_simple” stays untouched (base preserved).
# - AI briefings: NO specific entry/exit/stop numbers (already your constraint).
# - Strategy report: uses your internal plan numbers only (Titan already produces them).
#
# HOW TO APPLY:
# 1) Paste SECTION A into your existing file inside section (4) TELEGRAM ADVANCED BROADCASTER
#    (right after tg_send_photo / send_telegram_simple).
# 2) Paste SECTION B into your file near your AI analyst section (10) (anywhere above main()).
# 3) Apply SECTION C modifications inside BOTH Broadcast Centers (Titan tab[4] and Axiom tab[11]).
# =============================================================================


# =========================
# SECTION A — TELEGRAM UPGRADE HELPERS (ADD AFTER send_telegram_simple)
# =========================

def tg_chunk_text_safe(text: str, max_len: int = 2000) -> List[str]:
    """
    Chunk text safely for Telegram.
    - Prefers splitting on double-newlines, then newlines, then hard split.
    - Avoids ending a chunk with a trailing backslash (MarkdownV2 escape char).
    """
    if not text:
        return [""]

    chunks: List[str] = []
    s = str(text)

    while len(s) > max_len:
        cut = s.rfind("\n\n", 0, max_len)
        if cut == -1:
            cut = s.rfind("\n", 0, max_len)
        if cut == -1 or cut < int(max_len * 0.4):
            cut = max_len

        part = s[:cut].rstrip()
        s = s[cut:].lstrip()

        # Avoid ending on a single "\" which would break MarkdownV2 escaping
        if part.endswith("\\"):
            part = part[:-1]
            s = "\\" + s

        if part:
            chunks.append(part)

    if s.strip():
        # final chunk
        last = s.strip()
        if last.endswith("\\"):
            last = last[:-1]  # best-effort
        chunks.append(last)

    return chunks if chunks else [""]


def tg_send_broadcast(
    token: str,
    chat_ids_csv: str,
    text: str,
    parse_mode_label: str = "None (Plain Text)",
    photo_bytes: Optional[bytes] = None,
    photo_caption: Optional[str] = None,
    max_len: int = 2000
) -> Dict[str, Any]:
    """
    Multi-chat Telegram broadcaster with:
      - optional MarkdownV2 escaping
      - safe chunking
      - optional photo upload first
    Returns per-chat results.
    """
    results: Dict[str, Any] = {"ok": False, "sent": []}

    if not token or not chat_ids_csv:
        results["error"] = "Missing token or chat id(s)."
        return results

    chat_ids = _parse_chat_ids(chat_ids_csv)
    if not chat_ids:
        results["error"] = "No valid chat IDs found."
        return results

    parse_mode = None
    outbound_text = text if text is not None else ""

    if parse_mode_label == "MarkdownV2 (Escaped)":
        parse_mode = "MarkdownV2"
        outbound_text = tg_escape_markdown_v2(outbound_text)

    chunks = tg_chunk_text_safe(outbound_text, max_len=max_len)

    for cid in chat_ids:
        chat_res = {"chat_id": cid, "photo": None, "chunks": []}

        # Photo first (optional)
        if photo_bytes:
            cap = photo_caption or ""
            if parse_mode == "MarkdownV2":
                cap = tg_escape_markdown_v2(cap)
            okp, resp = tg_send_photo(
                token=token,
                chat_id=cid,
                photo_bytes=photo_bytes,
                caption=cap if cap else None,
                parse_mode=parse_mode
            )
            chat_res["photo"] = {"ok": okp, "resp": resp}

        # Then chunks
        if len(chunks) <= 1:
            okm, resp = tg_send_message(token, cid, chunks[0], parse_mode=parse_mode)
            chat_res["chunks"].append({"part": 1, "ok": okm, "resp": resp})
        else:
            part_idx = 1
            for ch in chunks:
                prefix = f"(Part {part_idx}/{len(chunks)}) "
                okm, resp = tg_send_message(token, cid, prefix + ch, parse_mode=parse_mode)
                chat_res["chunks"].append({"part": part_idx, "ok": okm, "resp": resp})
                part_idx += 1

        results["sent"].append(chat_res)

    results["ok"] = True
    return results


def _fmt_pct(x: Any, digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}%"
    except Exception:
        return "N/A"


def _fmt_num(x: Any, digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def _fmt_price(x: Any, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def titan_direction_from_row(last_row: pd.Series) -> str:
    try:
        return "LONG 🐂" if bool(last_row.get("is_bull", False)) else "SHORT 🐻"
    except Exception:
        return "N/A"


def titan_confidence_from_row(last_row: pd.Series) -> str:
    """
    Mirrors your mobile report alignment scoring, but returns text only.
    """
    try:
        is_bull = bool(last_row["is_bull"])
        titan_s = 1 if is_bull else -1
        apex_s = int(last_row["apex_trend"]) if pd.notna(last_row.get("apex_trend", np.nan)) else 0
        gann_s = int(last_row["gann_trend"]) if pd.notna(last_row.get("gann_trend", np.nan)) else 0
        score = 0
        if titan_s == apex_s:
            score += 1
        if titan_s == gann_s:
            score += 1
        if score == 2:
            return "HIGH"
        if score == 1:
            return "MEDIUM"
        return "LOW"
    except Exception:
        return "N/A"


def build_titan_telegram_report(
    ticker: str,
    timeframe: str,
    df: pd.DataFrame,
    last_row: pd.Series,
    fg_index: int,
    special_setups: Dict[str, bool],
    include_numbers: bool = True
) -> str:
    """
    Detailed Titan signal report from INTERNAL indicators/strategy.
    include_numbers=True includes your internal stop/TP plan (already in your strategy engine).
    """
    direction = titan_direction_from_row(last_row)
    conf = titan_confidence_from_row(last_row)

    squeeze_active = bool(last_row.get("in_squeeze", False))
    sqz = "SQUEEZE ACTIVE ⚠️" if squeeze_active else "NO SQUEEZE ⚪"

    rvol_val = _fmt_num(last_row.get("rvol", np.nan), 2)
    mf_val = _fmt_num(last_row.get("money_flow", np.nan), 2)
    flux_val = _fmt_num(last_row.get("Apex_Flux", np.nan), 2)

    vwap_rel = "Above VWAP" if float(last_row.get("close", 0)) > float(last_row.get("vwap", 0)) else "Below VWAP"

    setups = []
    if special_setups.get("squeeze_breakout"):
        setups.append("💥 Squeeze Breakout")
    if special_setups.get("gann_reversal"):
        setups.append("🔁 Gann Reversal")
    if special_setups.get("rvol_ignition"):
        setups.append("🚀 RVOL Ignition")
    setups_line = " | ".join(setups) if setups else "—"

    ts_txt = "N/A"
    try:
        ts_txt = str(last_row.get("timestamp", ""))[:19]
    except Exception:
        pass

    lines = [
        "💠 TITAN SIGNAL REPORT",
        f"Symbol: {ticker}",
        f"TF: {timeframe}",
        f"Time: {ts_txt}",
        "—",
        f"Direction: {direction}",
        f"Confidence: {conf}",
        f"Sentiment (FG): {fg_index}/100",
        f"Flux: {flux_val}",
        f"RVOL: {rvol_val}x",
        f"Money Flow: {mf_val}",
        f"VWAP: {vwap_rel}",
        f"Squeeze: {sqz}",
        f"Setups: {setups_line}",
    ]

    if include_numbers:
        lines += [
            "—",
            "🎯 INTERNAL EXECUTION PLAN (Strategy Engine)",
            f"Price: {_fmt_price(last_row.get('close', np.nan), 4)}",
            f"Smart Stop: {_fmt_price(last_row.get('entry_stop', np.nan), 4)}",
            f"TP1 (1.5R): {_fmt_price(last_row.get('tp1', np.nan), 4)}",
            f"TP2 (3.0R): {_fmt_price(last_row.get('tp2', np.nan), 4)}",
            f"TP3 (5.0R): {_fmt_price(last_row.get('tp3', np.nan), 4)}",
        ]

    # Guardrails
    lines += [
        "—",
        "⚠️ System Note: This is an automated internal indicator report (not financial advice)."
    ]

    return "\n".join(lines)


def axiom_gm_verdict(gm_score: float) -> str:
    if gm_score >= 3:
        return "STRONG BUY 🐂"
    if gm_score <= -3:
        return "STRONG SELL 🐻"
    return "NEUTRAL ⚖️"


def fg_state_from_value(fg_val: float) -> str:
    try:
        if np.isnan(fg_val):
            return "N/A"
        if fg_val >= 80:
            return "EXTREME GREED 😈"
        if fg_val >= 60:
            return "GREED 🙂"
        if fg_val >= 40:
            return "NEUTRAL 😐"
        if fg_val >= 20:
            return "FEAR 😟"
        return "EXTREME FEAR 😱"
    except Exception:
        return "N/A"


def build_axiom_telegram_report(
    ticker: str,
    timeframe: str,
    df: pd.DataFrame,
    last_row: pd.Series,
    fundamentals: Optional[Dict[str, Any]] = None,
    include_physics: bool = True,
    include_godmode: bool = True
) -> str:
    """
    Detailed Axiom report using INTERNAL computed indicators (Physics + God Mode).
    """
    price = _fmt_price(last_row.get("Close", np.nan), 4)
    chedo = _fmt_num(last_row.get("CHEDO", np.nan), 2)
    rqzo = _fmt_num(last_row.get("RQZO", np.nan), 2)
    flux = _fmt_num(last_row.get("Apex_Flux", np.nan), 2)
    flux_state = str(last_row.get("Apex_State", ""))
    trend_dir = "BULL 🐂" if int(last_row.get("Trend_Dir", -1)) == 1 else "BEAR 🐻"

    gm_score = float(last_row.get("GM_Score", 0))
    verdict = axiom_gm_verdict(gm_score)

    fg_val = float(last_row.get("FG_Index", np.nan))
    fg_state = fg_state_from_value(fg_val)
    rsi = _fmt_num(last_row.get("RSI", np.nan), 1)
    rvol = _fmt_num(last_row.get("RVOL", np.nan), 2)

    fund_line = "Fundamentals: N/A"
    if fundamentals:
        fund_line = f"Fundamentals: P/E={fundamentals.get('P/E Ratio','N/A')} | MCap={fundamentals.get('Market Cap','N/A')}"

    lines = [
        "💠 AXIOM QUANT REPORT",
        f"Ticker: {ticker}",
        f"TF: {timeframe}",
        f"Price: {price}",
        "—",
    ]

    if include_physics:
        lines += [
            "🧬 PHYSICS LAYER",
            f"Trend (Physics): {trend_dir}",
            f"Entropy (CHEDO): {chedo}",
            f"Relativity (RQZO): {rqzo}",
            f"Flux: {flux} ({flux_state})",
            "—",
        ]

    if include_godmode:
        lines += [
            "🧠 GOD MODE LAYER",
            f"GM Score: {gm_score:.0f}/5 → {verdict}",
            f"RSI: {rsi}",
            f"RVOL: {rvol}x",
            f"Sentiment: {fg_state} ({_fmt_num(fg_val, 1)}/100)",
            "—",
        ]

    lines += [
        fund_line,
        "—",
        "⚠️ System Note: Automated indicator synthesis (not financial advice)."
    ]

    return "\n".join(lines)


# Optional: attempt to render Plotly figure to PNG bytes for Telegram
def plotly_fig_to_png_bytes(fig: go.Figure, scale: int = 2) -> Optional[bytes]:
    """
    Requires kaleido in environment. If not available, returns None.
    """
    try:
        import plotly.io as pio
        return pio.to_image(fig, format="png", scale=scale)
    except Exception:
        return None


# =========================
# SECTION B — CONTRADICTION-AWARE AI BRAIN (ADD NEAR SECTION 10 AI ANALYST)
# =========================

def ask_ai_brain_titan(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    fg_index: int,
    last_row: pd.Series,
    special_setups: Dict[str, bool],
    api_key: str
) -> str:
    """
    Titan AI Brain briefing:
    - grounded in INTERNAL Titan indicators only
    - NO specific entry/exit/stop numbers
    - contradiction-aware: aligns with the internal signal direction unless it flags divergence
    """
    if not api_key:
        return "⚠️ Missing OpenAI API Key."

    direction = titan_direction_from_row(last_row)
    conf = titan_confidence_from_row(last_row)

    squeeze_active = bool(last_row.get("in_squeeze", False))
    sqz = "ON" if squeeze_active else "OFF"

    setups = []
    if special_setups.get("squeeze_breakout"):
        setups.append("SQUEEZE_BREAKOUT")
    if special_setups.get("gann_reversal"):
        setups.append("GANN_REVERSAL")
    if special_setups.get("rvol_ignition"):
        setups.append("RVOL_IGNITION")
    setups_line = ", ".join(setups) if setups else "NONE"

    # Compute a quick internal snapshot (no new indicators)
    try:
        rsi = float(last_row.get("rsi", np.nan))
    except Exception:
        rsi = np.nan

    prompt = f"""
You are the TITAN AI Brain, a disciplined market analyst.

Asset: {ticker}
Timeframe: {timeframe}

INTERNAL SIGNAL:
- Direction: {direction}
- Confidence: {conf}
- Fear/Greed (internal): {fg_index}/100
- Squeeze: {sqz}
- RVOL: {float(last_row.get("rvol", 0)):.2f}x
- Money Flow: {float(last_row.get("money_flow", 0)):.2f}
- Flux: {float(last_row.get("Apex_Flux", 0)):.2f}
- RSI: {rsi if not np.isnan(rsi) else 0:.1f}
- Setups: {setups_line}

MISSION:
1) Explain what the signal means in plain language.
2) Provide a "playbook" using IF/THEN logic (scenario-based), but:
   - DO NOT provide specific entry/exit/stop/take-profit numbers.
3) Contradiction handling:
   - If your analysis disagrees with the internal direction, clearly label it as "DIVERGENCE"
     and explain why (without numbers).

RULES:
- Not financial advice.
- Emojis ok.
- Keep it concise, high signal.
"""

    try:
        client = OpenAI(api_key=api_key)
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error: {e}"


def ask_ai_brain_axiom(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    fundamentals: Optional[Dict[str, Any]],
    api_key: str
) -> str:
    """
    Axiom AI Brain briefing:
    - grounded in INTERNAL Physics + GM indicators already computed on df
    - NO specific entry/exit/stop numbers
    - contradiction-aware vs GM verdict
    """
    if not api_key:
        return "⚠️ Missing OpenAI API Key."

    last = df.iloc[-1]
    gm_score = float(last.get("GM_Score", 0))
    gm_verdict = axiom_gm_verdict(gm_score)
    price = float(last.get("Close", 0))

    fg_val = float(last.get("FG_Index", np.nan))
    fg_state = fg_state_from_value(fg_val)

    fund_text = "N/A"
    if fundamentals:
        pe = fundamentals.get("P/E Ratio", "N/A")
        mc = fundamentals.get("Market Cap", "N/A")
        fund_text = f"P/E: {pe} | MCap: {mc}"

    prompt = f"""
You are the AXIOM AI Brain, a disciplined quantitative market analyst.

Asset: {ticker}
Timeframe: {timeframe}
Price: {price:.4f}

INTERNAL METRICS (GROUND TRUTH):
- Physics Trend Dir: {"BULL" if int(last.get("Trend_Dir",-1))==1 else "BEAR"}
- CHEDO (Entropy): {float(last.get("CHEDO", 0)):.2f}
- RQZO (Relativity): {float(last.get("RQZO", 0)):.2f}
- Flux: {float(last.get("Apex_Flux", 0)):.2f} ({str(last.get("Apex_State",""))})
- GM Score: {gm_score:.0f}/5 → {gm_verdict}
- RSI: {float(last.get("RSI", 0)):.1f}
- RVOL: {float(last.get("RVOL", 0)):.2f}x
- Sentiment: {fg_state} ({(0 if np.isnan(fg_val) else fg_val):.1f}/100)
- Fundamentals: {fund_text}

MISSION:
1) Explain current regime (trend vs chop / risk-on vs risk-off) using the metrics above.
2) Provide a scenario-based playbook using IF/THEN logic.
3) Contradiction handling:
   - If your outlook conflicts with the GM verdict, label "DIVERGENCE" and explain.

RULES:
- DO NOT provide specific entry/exit/stop/take-profit numbers.
- Not financial advice.
- Keep it concise; use emojis sparingly but clearly.
"""

    try:
        client = OpenAI(api_key=api_key)
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error: {e}"


# =========================
# SECTION C — UI INTEGRATION (MODIFY Broadcast Centers, KEEP EVERYTHING ELSE)
# =========================
# Apply these modifications inside:
#   - TITAN mode tab t_tabs[4] “⚙️ Broadcast Center”
#   - AXIOM mode tab tabs[11] “⚙️ Broadcast Center”
#
# You are NOT deleting your existing logic; you are upgrading it to generate
# detailed messages and optionally append AI Brain analysis.
#
# --- TITAN Broadcast Center REPLACE the content inside `with t_tabs[4]:` with: ---
"""
        with t_tabs[4]:
            st.subheader("📢 Broadcast Center (Titan Mobile)")
            st.caption("Enhanced Telegram broadcaster: detailed internal report + optional AI Brain (contradiction-aware).")

            uploaded_file = st.file_uploader("Upload Chart Screenshot (Optional)", type=["png", "jpg", "jpeg"])

            include_numbers = st.checkbox("Include Internal Execution Plan Numbers (Stop/TPs)", value=True)
            include_ai = st.checkbox("Append AI Brain Briefing (No numbers)", value=False)

            # Build a rich default report from internal indicators
            msg_default = ""
            try:
                # df/last/spec/fg exist in Titan tab[0] scope; if not, we rebuild lightly.
                if "df" in locals() and df is not None and not df.empty:
                    _df_for_msg = df
                else:
                    _df_for_msg = TitanEngine.get_klines(ticker, timeframe, int(limit))
                    if not _df_for_msg.empty:
                        _df_for_msg, _zones = TitanEngine.run_engine(_df_for_msg, int(amp), float(dev), int(hma_len), int(gann_len), 55, 1.5, 10)

                if _df_for_msg is not None and not _df_for_msg.empty:
                    _last = _df_for_msg.iloc[-1]
                    _fg = TitanEngine.calculate_fear_greed(_df_for_msg)
                    _spec = TitanEngine.detect_special_setups(_df_for_msg)
                    msg_default = build_titan_telegram_report(
                        ticker=ticker,
                        timeframe=timeframe,
                        df=_df_for_msg,
                        last_row=_last,
                        fg_index=int(_fg),
                        special_setups=_spec,
                        include_numbers=bool(include_numbers),
                    )
                else:
                    msg_default = f"🚀 TITAN SIGNAL\nSymbol: {ticker}\nTF: {timeframe}\n(Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')})"
            except Exception:
                msg_default = f"🚀 TITAN SIGNAL\nSymbol: {ticker}\nTF: {timeframe}\n(Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')})"

            msg = st.text_area("Message Preview", value=msg_default, height=260)

            if include_ai:
                if st.button("Generate AI Brain Briefing (Titan)"):
                    try:
                        # use current computed df if available
                        _df_ai = df if "df" in locals() and df is not None and not df.empty else TitanEngine.get_klines(ticker, timeframe, int(limit))
                        if _df_ai is not None and not _df_ai.empty:
                            _df_ai, _zones_ai = TitanEngine.run_engine(_df_ai, int(amp), float(dev), int(hma_len), int(gann_len), 55, 1.5, 10)
                            _last_ai = _df_ai.iloc[-1]
                            _fg_ai = TitanEngine.calculate_fear_greed(_df_ai)
                            _spec_ai = TitanEngine.detect_special_setups(_df_ai)

                            ai_text = ask_ai_brain_titan(
                                df=_df_ai,
                                ticker=ticker,
                                timeframe=timeframe,
                                fg_index=int(_fg_ai),
                                last_row=_last_ai,
                                special_setups=_spec_ai,
                                api_key=ai_key
                            )
                            msg = msg + "\n\n—\n🤖 AI BRAIN BRIEFING\n" + str(ai_text)
                            st.success("AI briefing appended to preview.")
                        else:
                            st.warning("No data available to generate AI briefing.")
                    except Exception as e:
                        st.error(f"AI briefing failed: {e}")

            if st.button("Send to Telegram 🚀"):
                if not tg_token or not tg_chat:
                    st.warning("⚠️ Enter Telegram keys in sidebar.")
                else:
                    photo_bytes = uploaded_file.getvalue() if uploaded_file else None
                    caption = f"🔥 TITAN: {ticker} ({timeframe})"

                    res = tg_send_broadcast(
                        token=tg_token,
                        chat_ids_csv=tg_chat,
                        text=msg,
                        parse_mode_label=tg_parse_mode_label,
                        photo_bytes=photo_bytes,
                        photo_caption=caption,
                        max_len=2000
                    )
                    if res.get("ok"):
                        st.success("✅ Sent to Telegram (detailed report + safe chunking).")
                    else:
                        st.error(f"Send failed: {res.get('error','Unknown error')}")
"""

# --- AXIOM Broadcast Center REPLACE the content inside `with tabs[11]:` with: ---
"""
        with tabs[11]:
            st.subheader("📢 Broadcast Center (Axiom)")
            st.caption("Enhanced Telegram broadcaster: detailed internal report + optional AI Brain (contradiction-aware).")

            uploaded_file = st.file_uploader("Upload Chart Screenshot (Optional)", type=["png", "jpg", "jpeg"])

            include_physics = st.checkbox("Include Physics Layer (CHEDO/RQZO/Flux)", value=True)
            include_gm = st.checkbox("Include God Mode Layer (GM/FG/RSI/RVOL)", value=True)
            include_ai = st.checkbox("Append AI Brain Briefing (No numbers)", value=False)

            # Fundamentals already fetched as `fund` above in your Axiom flow
            try:
                _fund_for_msg = fund if "fund" in locals() else AxiomEngine.get_fundamentals(ticker)
            except Exception:
                _fund_for_msg = None

            msg_default = build_axiom_telegram_report(
                ticker=ticker,
                timeframe=tf,
                df=df,
                last_row=last,
                fundamentals=_fund_for_msg,
                include_physics=bool(include_physics),
                include_godmode=bool(include_gm)
            )

            msg = st.text_area("Message Preview", value=msg_default, height=260)

            if include_ai:
                if st.button("Generate AI Brain Briefing (Axiom)"):
                    try:
                        ai_text = ask_ai_brain_axiom(
                            df=df,
                            ticker=ticker,
                            timeframe=tf,
                            fundamentals=_fund_for_msg,
                            api_key=ai_key
                        )
                        msg = msg + "\n\n—\n🤖 AI BRAIN BRIEFING\n" + str(ai_text)
                        st.success("AI briefing appended to preview.")
                    except Exception as e:
                        st.error(f"AI briefing failed: {e}")

            if st.button("Send to Telegram 🚀"):
                if not tg_token or not tg_chat:
                    st.warning("⚠️ Enter Telegram keys in sidebar.")
                else:
                    photo_bytes = uploaded_file.getvalue() if uploaded_file else None
                    caption = f"🔥 AXIOM: {ticker} ({tf})"

                    res = tg_send_broadcast(
                        token=tg_token,
                        chat_ids_csv=tg_chat,
                        text=msg,
                        parse_mode_label=tg_parse_mode_label,
                        photo_bytes=photo_bytes,
                        photo_caption=caption,
                        max_len=2000
                    )
                    if res.get("ok"):
                        st.success("✅ Sent to Telegram (detailed report + safe chunking).")
                    else:
                        st.error(f"Send failed: {res.get('error','Unknown error')}")
"""
