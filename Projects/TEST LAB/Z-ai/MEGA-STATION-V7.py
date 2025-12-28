# =============================================================================
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
        df["tp3"] = np.where(df["is_bull"], df["close"] + 5.0 * risk, df["
