# app.py
# Streamlit Telegram Signals Commander (Signal-only, monetizable-ready)
# - Generates rule-based signals from OHLCV (via ccxt) for crypto
# - Applies regime filter (trend vs chop) using EMAs + ATR
# - Produces clean signal cards + posts to Telegram (channel/group)
# - Logs signals + outcomes locally (SQLite) for transparent performance tracking
#
# ✅ Secrets: put these in .streamlit/secrets.toml (Streamlit Cloud) or set env vars.
#   TELEGRAM_BOT_TOKEN="123:abc..."
#   TELEGRAM_CHAT_ID="-1001234567890"   # channel/group id (negative for channels)
#
# Optional:
#   DEFAULT_EXCHANGE="mexc"
#   OPENAI_API_KEY="..."  # if you want AI commentary (disabled by default)
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py

import os
import json
import time
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional dependencies
try:
    import ccxt
    CCXT_OK = True
except Exception:
    CCXT_OK = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# =========================
# CONFIG
# =========================
APP_TITLE = "Signal-Only Telegram Engine"
DB_PATH = "signals.db"

DEFAULT_TIMEFRAMES = ["15m", "1h", "4h", "1d"]
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "BNB/USDT"]

# Strategy defaults (tuned for "clean, low-noise")
DEFAULT_FAST_EMA = 50
DEFAULT_SLOW_EMA = 200
DEFAULT_ATR_LEN = 14
DEFAULT_ATR_MULT_STOP = 1.8  # stop distance = ATR * mult
DEFAULT_MIN_RR = 1.5         # minimum planned R:R
DEFAULT_PULLBACK_DEPTH_ATR = 0.6  # pullback proximity to fast EMA in ATR units

# Signal throttle
DEFAULT_MAX_SIGNALS_PER_SCAN = 3
DEFAULT_COOLDOWN_MINUTES_PER_SYMBOL = 90


# =========================
# UI THEME
# =========================
def inject_css():
    st.markdown(
        """
<style>
:root{
  --bg:#070a0f;
  --card:#0e1320;
  --border:#1a2336;
  --text:#e9eefc;
  --muted:#92a0bf;
  --bull:#00f5d4;
  --bear:#ff2e63;
  --warn:#ffd166;
  --ok:#7CFF6B;
}
.stApp{
  background: radial-gradient(1200px 900px at 25% 10%, rgba(0,245,212,0.07), transparent 60%),
              radial-gradient(1200px 900px at 80% 30%, rgba(255,46,99,0.06), transparent 55%),
              var(--bg);
  color:var(--text);
}
div[data-testid="stMetric"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border:1px solid var(--border);
  padding:14px 14px;
  border-radius:18px;
}
.block-container{ padding-top: 1.2rem; }
hr{ border: none; height:1px; background: rgba(255,255,255,0.08); margin: 12px 0; }

.card{
  background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012));
  border:1px solid var(--border);
  border-radius:18px;
  padding:16px 16px;
}
.k{
  color: var(--muted);
  font-size: 12px;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.v{
  font-size: 14px;
}
.badge{
  display:inline-block;
  padding:4px 10px;
  border-radius: 999px;
  border:1px solid rgba(255,255,255,0.12);
  font-size: 12px;
  margin-right: 6px;
}
.badge-bull{ border-color: rgba(0,245,212,0.35); color: var(--bull); }
.badge-bear{ border-color: rgba(255,46,99,0.35); color: var(--bear); }
.badge-chop{ border-color: rgba(146,160,191,0.35); color: var(--muted); }
.badge-warn{ border-color: rgba(255,209,102,0.45); color: var(--warn); }
.small{ color: var(--muted); font-size: 12px; }
.mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
</style>
""",
        unsafe_allow_html=True,
    )


# =========================
# UTIL
# =========================
def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def fmt_price(x: float, decimals: Optional[int] = None) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if decimals is None:
        # auto decimals for crypto
        if x >= 1000:
            decimals = 1
        elif x >= 100:
            decimals = 2
        elif x >= 1:
            decimals = 4
        else:
            decimals = 6
    return f"{x:.{decimals}f}"


def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x*100:.2f}%"


# =========================
# DB (Signals + Cooldowns + Outcomes)
# =========================
def db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def db_init(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_utc TEXT NOT NULL,
          exchange TEXT NOT NULL,
          symbol TEXT NOT NULL,
          timeframe TEXT NOT NULL,
          bias TEXT NOT NULL,
          setup TEXT NOT NULL,
          entry REAL,
          stop REAL,
          t1 REAL,
          t2 REAL,
          t3 REAL,
          rr REAL,
          leverage_max REAL,
          risk_note TEXT,
          invalidation TEXT,
          notes TEXT,
          sent_to_telegram INTEGER DEFAULT 0,
          message_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cooldowns (
          key TEXT PRIMARY KEY,
          last_signal_utc TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS outcomes (
          signal_id INTEGER PRIMARY KEY,
          outcome TEXT,
          closed_utc TEXT,
          r_multiple REAL,
          notes TEXT,
          FOREIGN KEY(signal_id) REFERENCES signals(id)
        )
        """
    )
    conn.commit()


def cooldown_key(exchange: str, symbol: str, timeframe: str) -> str:
    return f"{exchange}::{symbol}::{timeframe}"


def is_in_cooldown(conn: sqlite3.Connection, key: str, cooldown_minutes: int) -> Tuple[bool, Optional[str]]:
    row = conn.execute("SELECT last_signal_utc FROM cooldowns WHERE key=?", (key,)).fetchone()
    if not row:
        return False, None
    last_str = row[0]
    try:
        last_dt = datetime.strptime(last_str, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        delta_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60.0
        if delta_min < cooldown_minutes:
            return True, f"{cooldown_minutes - int(delta_min)}m remaining"
    except Exception:
        pass
    return False, None


def set_cooldown(conn: sqlite3.Connection, key: str):
    t = now_utc()
    conn.execute(
        "INSERT INTO cooldowns(key,last_signal_utc) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET last_signal_utc=excluded.last_signal_utc",
        (key, t),
    )
    conn.commit()


def save_signal(conn: sqlite3.Connection, s: Dict) -> int:
    cols = [
        "created_utc","exchange","symbol","timeframe","bias","setup",
        "entry","stop","t1","t2","t3","rr","leverage_max","risk_note",
        "invalidation","notes","sent_to_telegram","message_id"
    ]
    vals = [s.get(c) for c in cols]
    q = f"INSERT INTO signals({','.join(cols)}) VALUES({','.join(['?']*len(cols))})"
    cur = conn.execute(q, vals)
    conn.commit()
    return int(cur.lastrowid)


def mark_sent(conn: sqlite3.Connection, signal_id: int, message_id: str):
    conn.execute("UPDATE signals SET sent_to_telegram=1, message_id=? WHERE id=?", (message_id, signal_id))
    conn.commit()


def load_recent_signals(conn: sqlite3.Connection, limit: int = 50) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
        conn,
        params=(limit,),
    )
    return df


def load_outcomes(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT s.id, s.created_utc, s.exchange, s.symbol, s.timeframe, s.bias, s.setup, s.rr,
               o.outcome, o.closed_utc, o.r_multiple, o.notes as outcome_notes
        FROM signals s
        LEFT JOIN outcomes o ON o.signal_id = s.id
        ORDER BY s.id DESC
        """,
        conn,
    )
    return df


def upsert_outcome(conn: sqlite3.Connection, signal_id: int, outcome: str, r_multiple: float, notes: str):
    conn.execute(
        """
        INSERT INTO outcomes(signal_id, outcome, closed_utc, r_multiple, notes)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(signal_id) DO UPDATE SET
          outcome=excluded.outcome,
          closed_utc=excluded.closed_utc,
          r_multiple=excluded.r_multiple,
          notes=excluded.notes
        """,
        (signal_id, outcome, now_utc(), r_multiple, notes),
    )
    conn.commit()


# =========================
# DATA (CCXT)
# =========================
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    # Streamlit secrets first, then env vars
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


def ccxt_exchange(name: str):
    if not CCXT_OK:
        raise RuntimeError("ccxt not installed. Add ccxt to requirements.")
    if not hasattr(ccxt, name):
        raise ValueError(f"Exchange '{name}' not supported by ccxt.")
    ex = getattr(ccxt, name)({"enableRateLimit": True})
    # Public mode (no keys needed for OHLCV)
    return ex


def fetch_ohlcv(ex_name: str, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    ex = ccxt_exchange(ex_name)
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt").drop(columns=["ts"])
    return df


# =========================
# INDICATORS
# =========================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(df).ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).ewm(span=length, adjust=False).mean()
    roll_down = pd.Series(loss, index=series.index).ewm(span=length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


# =========================
# SIGNAL LOGIC (Rule-based, low-noise)
# =========================
@dataclass
class Signal:
    exchange: str
    symbol: str
    timeframe: str
    bias: str              # Bull / Bear / Neutral
    setup: str             # Pullback Continuation / Reversal etc.
    entry: float
    stop: float
    t1: float
    t2: float
    t3: float
    rr: float
    leverage_max: float
    risk_note: str
    invalidation: str
    notes: str


def determine_regime(df: pd.DataFrame, fast_len: int, slow_len: int, atr_len: int) -> Dict:
    close = df["close"]
    e_fast = ema(close, fast_len)
    e_slow = ema(close, slow_len)
    a = atr(df, atr_len)

    # Basic trend slope & separation
    sep = (e_fast - e_slow) / (a + 1e-12)

    # Chop heuristic: low separation + mean reversion conditions
    # - if sep small OR fast ema oscillates around slow
    sep_now = float(sep.iloc[-1])
    slope_fast = float((e_fast.iloc[-1] - e_fast.iloc[-6]) / (a.iloc[-1] + 1e-12))

    trend = "Trend" if abs(sep_now) >= 0.6 and abs(slope_fast) >= 0.10 else "Chop"
    bias = "Bull" if e_fast.iloc[-1] > e_slow.iloc[-1] else "Bear"

    return {
        "trend": trend,     # Trend / Chop
        "bias": bias,       # Bull / Bear
        "ema_fast": e_fast,
        "ema_slow": e_slow,
        "atr": a,
        "sep": sep,
        "sep_now": sep_now,
        "slope_fast": slope_fast,
    }


def make_pullback_signal(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
    fast_len: int,
    slow_len: int,
    atr_len: int,
    atr_mult_stop: float,
    min_rr: float,
    pullback_depth_atr: float,
    leverage_max: float,
) -> Optional[Signal]:

    reg = determine_regime(df, fast_len, slow_len, atr_len)
    if reg["trend"] != "Trend":
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    e_fast = reg["ema_fast"]
    e_slow = reg["ema_slow"]
    a = reg["atr"]

    # Use last closed candle (avoid using the forming candle)
    i = -2
    px = float(close.iloc[i])
    atr_now = float(a.iloc[i])
    if atr_now <= 0:
        return None

    bias = reg["bias"]  # Bull / Bear

    # Pullback definition:
    # - In bull trend: price pulled back near fast EMA, but still above slow EMA
    # - Trigger: close back above fast EMA (bull) / below fast EMA (bear) on last closed candle
    near_fast = abs(px - float(e_fast.iloc[i])) <= (pullback_depth_atr * atr_now)

    if bias == "Bull":
        trend_ok = px > float(e_slow.iloc[i])
        trigger = px > float(e_fast.iloc[i])  # reclaim
        setup_name = "Pullback Continuation"
        if not (trend_ok and near_fast and trigger):
            return None

        entry = px
        stop = entry - (atr_mult_stop * atr_now)
        # Targets based on recent swing high + R multiples
        recent_high = float(high.iloc[-30:-2].max())
        r = entry - stop
        t1 = entry + 1.0 * r
        t2 = entry + 1.7 * r
        t3 = max(entry + 2.5 * r, recent_high)  # at least near swing high
        rr = (t2 - entry) / (entry - stop + 1e-12)

        if rr < min_rr:
            return None

        invalidation = f"Invalid if 1H close < {fmt_price(stop)} (stop) or trend flips (EMA{fast_len} < EMA{slow_len})."
        notes = f"Regime: Trend | EMA{fast_len}>{slow_len} | Pullback near EMA{fast_len} within {pullback_depth_atr} ATR."

        return Signal(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            bias="Bull",
            setup=setup_name,
            entry=entry,
            stop=stop,
            t1=t1,
            t2=t2,
            t3=t3,
            rr=float(rr),
            leverage_max=leverage_max,
            risk_note="Risk 0.5–1.0% max. Leverage ≤ 5x.",
            invalidation=invalidation,
            notes=notes,
        )

    else:
        trend_ok = px < float(e_slow.iloc[i])
        trigger = px < float(e_fast.iloc[i])  # reject
        setup_name = "Pullback Continuation"
        if not (trend_ok and near_fast and trigger):
            return None

        entry = px
        stop = entry + (atr_mult_stop * atr_now)
        recent_low = float(low.iloc[-30:-2].min())
        r = stop - entry
        t1 = entry - 1.0 * r
        t2 = entry - 1.7 * r
        t3 = min(entry - 2.5 * r, recent_low)
        rr = (entry - t2) / (stop - entry + 1e-12)

        if rr < min_rr:
            return None

        invalidation = f"Invalid if 1H close > {fmt_price(stop)} (stop) or trend flips (EMA{fast_len} > EMA{slow_len})."
        notes = f"Regime: Trend | EMA{fast_len}<{slow_len} | Pullback near EMA{fast_len} within {pullback_depth_atr} ATR."

        return Signal(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            bias="Bear",
            setup=setup_name,
            entry=entry,
            stop=stop,
            t1=t1,
            t2=t2,
            t3=t3,
            rr=float(rr),
            leverage_max=leverage_max,
            risk_note="Risk 0.5–1.0% max. Leverage ≤ 5x.",
            invalidation=invalidation,
            notes=notes,
        )


def make_range_reversion_signal(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
    fast_len: int,
    slow_len: int,
    atr_len: int,
    atr_mult_stop: float,
    min_rr: float,
    leverage_max: float,
) -> Optional[Signal]:

    reg = determine_regime(df, fast_len, slow_len, atr_len)
    if reg["trend"] != "Chop":
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    e_fast = reg["ema_fast"]
    a = reg["atr"]

    i = -2
    px = float(close.iloc[i])
    atr_now = float(a.iloc[i])
    if atr_now <= 0:
        return None

    # Range proxy: use rolling bounds
    look = 40
    lo = float(low.iloc[-look:-2].min())
    hi = float(high.iloc[-look:-2].max())
    mid = (hi + lo) / 2.0

    # If price at extremes, mean reversion
    dist_to_lo = (px - lo) / (atr_now + 1e-12)
    dist_to_hi = (hi - px) / (atr_now + 1e-12)

    # Light RSI confirmation (optional but useful in chop)
    r = rsi(close, 14)
    rsi_now = float(r.iloc[i])

    # Buy near range low
    if dist_to_lo <= 0.8 and rsi_now <= 38:
        entry = px
        stop = entry - (atr_mult_stop * atr_now)
        t1 = mid
        t2 = hi
        t3 = hi + 0.5 * (hi - lo)
        rr = (t1 - entry) / (entry - stop + 1e-12)
        if rr < min_rr:
            return None
        return Signal(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            bias="Bull",
            setup="Range Reversion",
            entry=entry,
            stop=stop,
            t1=t1,
            t2=t2,
            t3=t3,
            rr=float(rr),
            leverage_max=leverage_max,
            risk_note="Chop-mode only. Small size. Risk 0.25–0.75%. Leverage ≤ 3–5x.",
            invalidation=f"Invalid if close < {fmt_price(stop)} or range breaks down.",
            notes=f"Regime: Chop | Buy near range low ({fmt_price(lo)}–{fmt_price(hi)}). RSI={rsi_now:.1f}.",
        )

    # Sell near range high
    if dist_to_hi <= 0.8 and rsi_now >= 62:
        entry = px
        stop = entry + (atr_mult_stop * atr_now)
        t1 = mid
        t2 = lo
        t3 = lo - 0.5 * (hi - lo)
        rr = (entry - t1) / (stop - entry + 1e-12)
        if rr < min_rr:
            return None
        return Signal(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            bias="Bear",
            setup="Range Reversion",
            entry=entry,
            stop=stop,
            t1=t1,
            t2=t2,
            t3=t3,
            rr=float(rr),
            leverage_max=leverage_max,
            risk_note="Chop-mode only. Small size. Risk 0.25–0.75%. Leverage ≤ 3–5x.",
            invalidation=f"Invalid if close > {fmt_price(stop)} or range breaks out.",
            notes=f"Regime: Chop | Sell near range high ({fmt_price(lo)}–{fmt_price(hi)}). RSI={rsi_now:.1f}.",
        )

    return None


def generate_signal(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
    fast_len: int,
    slow_len: int,
    atr_len: int,
    atr_mult_stop: float,
    min_rr: float,
    pullback_depth_atr: float,
    leverage_max: float,
    allow_chop_setups: bool,
) -> Optional[Signal]:

    # 1) Prefer Trend pullback signals
    s1 = make_pullback_signal(
        df, exchange, symbol, timeframe,
        fast_len, slow_len, atr_len,
        atr_mult_stop, min_rr, pullback_depth_atr, leverage_max
    )
    if s1:
        return s1

    # 2) Optional Chop signals
    if allow_chop_setups:
        s2 = make_range_reversion_signal(
            df, exchange, symbol, timeframe,
            fast_len, slow_len, atr_len,
            atr_mult_stop, min_rr, leverage_max
        )
        if s2:
            return s2

    return None


# =========================
# TELEGRAM
# =========================
def telegram_send_message(token: str, chat_id: str, text: str, parse_mode: str = "HTML") -> Tuple[bool, str]:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code == 200:
            data = r.json()
            msg_id = str(data.get("result", {}).get("message_id", ""))
            return True, msg_id
        return False, f"{r.status_code}: {r.text}"
    except Exception as e:
        return False, str(e)


def format_signal_message(sig: Signal) -> str:
    # HTML formatted for Telegram
    direction_emoji = "🟢" if sig.bias == "Bull" else "🔴" if sig.bias == "Bear" else "⚪️"
    setup_emoji = "⚡️"
    tf = sig.timeframe.upper()
    header = f"<b>{direction_emoji} {sig.symbol} — {sig.bias} ({tf})</b>\n"
    setup = f"{setup_emoji} <b>Setup:</b> {sig.setup}\n"
    levels = (
        f"🎯 <b>Entry:</b> <code>{fmt_price(sig.entry)}</code>\n"
        f"🛑 <b>Stop:</b> <code>{fmt_price(sig.stop)}</code>\n"
        f"✅ <b>Targets:</b> "
        f"<code>{fmt_price(sig.t1)}</code> / <code>{fmt_price(sig.t2)}</code> / <code>{fmt_price(sig.t3)}</code>\n"
    )
    risk = (
        f"📌 <b>Leverage:</b> ≤ <code>{sig.leverage_max:.0f}x</code>\n"
        f"📉 <b>Risk:</b> {sig.risk_note}\n"
        f"📏 <b>Planned RR:</b> <code>{sig.rr:.2f}R</code>\n"
    )
    invalid = f"⛔️ <b>Invalidation:</b> {sig.invalidation}\n"
    notes = f"🧠 <b>Notes:</b> {sig.notes}\n"
    footer = f"<i>{sig.exchange} • {now_utc()}</i>"
    return header + setup + "\n" + levels + "\n" + risk + "\n" + invalid + "\n" + notes + "\n" + footer


# =========================
# PLOT
# =========================
def make_chart(df: pd.DataFrame, fast: int, slow: int, atr_len: int, title: str) -> "go.Figure":
    if not PLOTLY_OK:
        raise RuntimeError("Plotly missing.")
    close = df["close"]
    e_fast = ema(close, fast)
    e_slow = ema(close, slow)
    a = atr(df, atr_len)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price",
        ),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(x=df.index, y=e_fast, name=f"EMA {fast}", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=e_slow, name=f"EMA {slow}", mode="lines"), row=1, col=1)

    # ATR panel
    fig.add_trace(go.Scatter(x=df.index, y=a, name=f"ATR {atr_len}", mode="lines"), row=2, col=1)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# =========================
# STREAMLIT APP
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📡")
    inject_css()

    st.title("📡 Signal-Only Telegram Engine")
    st.caption("Rule-based crypto signals → clean templates → Telegram broadcast → outcome tracking.")

    if not CCXT_OK:
        st.error("ccxt is not installed. Add `ccxt` to your requirements.txt.")
        st.stop()
    if not PLOTLY_OK:
        st.warning("Plotly not installed — charts disabled. Add `plotly` to requirements.txt.")

    # DB init
    conn = db_connect()
    db_init(conn)

    # Secrets
    tg_token = get_secret("TELEGRAM_BOT_TOKEN", "")
    tg_chat_id = get_secret("TELEGRAM_CHAT_ID", "")
    default_exchange = get_secret("DEFAULT_EXCHANGE", "mexc")

    # Sidebar: Engine controls
    st.sidebar.header("⚙️ Engine Controls")

    exchange = st.sidebar.text_input("Exchange (ccxt id)", value=default_exchange, help="Example: mexc, binance, bybit")
    timeframe = st.sidebar.selectbox("Timeframe", DEFAULT_TIMEFRAMES, index=1)

    symbols = st.sidebar.multiselect("Universe (symbols)", DEFAULT_SYMBOLS, default=["BTC/USDT", "ETH/USDT"])
    custom_symbol = st.sidebar.text_input("Add symbol (optional)", value="")
    if custom_symbol.strip():
        if custom_symbol.strip() not in symbols:
            symbols = symbols + [custom_symbol.strip()]

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Strategy Parameters")
    fast_len = st.sidebar.number_input("Fast EMA", min_value=5, max_value=200, value=DEFAULT_FAST_EMA, step=1)
    slow_len = st.sidebar.number_input("Slow EMA", min_value=20, max_value=400, value=DEFAULT_SLOW_EMA, step=1)
    atr_len = st.sidebar.number_input("ATR Length", min_value=5, max_value=50, value=DEFAULT_ATR_LEN, step=1)
    atr_mult_stop = st.sidebar.number_input("Stop ATR Mult", min_value=0.8, max_value=4.0, value=DEFAULT_ATR_MULT_STOP, step=0.1)
    pullback_depth_atr = st.sidebar.number_input("Pullback Near EMA (ATR)", min_value=0.2, max_value=2.0, value=DEFAULT_PULLBACK_DEPTH_ATR, step=0.1)
    min_rr = st.sidebar.number_input("Minimum planned R:R", min_value=1.0, max_value=5.0, value=DEFAULT_MIN_RR, step=0.1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧯 Risk / Throttle")
    leverage_max = st.sidebar.number_input("Max Leverage", min_value=1, max_value=10, value=5, step=1)
    max_signals = st.sidebar.number_input("Max signals per scan", min_value=1, max_value=10, value=DEFAULT_MAX_SIGNALS_PER_SCAN, step=1)
    cooldown_min = st.sidebar.number_input("Cooldown per symbol (min)", min_value=0, max_value=720, value=DEFAULT_COOLDOWN_MINUTES_PER_SYMBOL, step=5)
    allow_chop = st.sidebar.checkbox("Allow chop setups (range reversion)", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📣 Telegram")
    st.sidebar.write("Loaded via secrets/env (recommended).")
    show_tg = st.sidebar.checkbox("Show Telegram config status", value=False)
    if show_tg:
        st.sidebar.write("Token:", "✅" if tg_token else "❌")
        st.sidebar.write("Chat ID:", "✅" if tg_chat_id else "❌")

    send_live = st.sidebar.checkbox("Send LIVE to Telegram", value=False, help="If OFF: you can preview messages safely.")
    st.sidebar.caption("Safety tip: test with a private group first.")

    # Top KPIs
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.metric("Exchange", exchange)
    with colB:
        st.metric("Timeframe", timeframe.upper())
    with colC:
        st.metric("Universe", f"{len(symbols)} symbols")
    with colD:
        st.metric("DB", DB_PATH)

    tabs = st.tabs(["🚨 Scanner", "📊 Charts", "🧾 Signal Log", "🏁 Outcomes"])

    # =========================
    # Scanner
    # =========================
    with tabs[0]:
        st.subheader("🚨 Scan → Generate signals")
        st.write("Produces low-noise signals using regime filter (Trend vs Chop).")
        scan_btn = st.button("Scan now", type="primary", use_container_width=True)

        if scan_btn:
            found: List[Signal] = []
            errors: List[str] = []

            with st.spinner("Fetching data + scanning..."):
                for sym in symbols:
                    if len(found) >= int(max_signals):
                        break

                    key = cooldown_key(exchange, sym, timeframe)
                    in_cd, remaining = is_in_cooldown(conn, key, int(cooldown_min))
                    if in_cd:
                        continue

                    try:
                        df = fetch_ohlcv(exchange, sym, timeframe, limit=320)
                        if len(df) < 220:
                            continue

                        sig = generate_signal(
                            df=df,
                            exchange=exchange,
                            symbol=sym,
                            timeframe=timeframe,
                            fast_len=int(fast_len),
                            slow_len=int(slow_len),
                            atr_len=int(atr_len),
                            atr_mult_stop=float(atr_mult_stop),
                            min_rr=float(min_rr),
                            pullback_depth_atr=float(pullback_depth_atr),
                            leverage_max=float(leverage_max),
                            allow_chop_setups=allow_chop,
                        )
                        if sig:
                            found.append(sig)
                            set_cooldown(conn, key)

                    except Exception as e:
                        errors.append(f"{sym}: {e}")

            if errors:
                with st.expander("Errors (non-fatal)"):
                    for e in errors[:30]:
                        st.write("•", e)

            if not found:
                st.info("No signals found this scan. (This is good: low noise.)")

            for sig in found:
                msg = format_signal_message(sig)
                # Save to DB
                row = dict(
                    created_utc=now_utc(),
                    exchange=sig.exchange,
                    symbol=sig.symbol,
                    timeframe=sig.timeframe,
                    bias=sig.bias,
                    setup=sig.setup,
                    entry=float(sig.entry),
                    stop=float(sig.stop),
                    t1=float(sig.t1),
                    t2=float(sig.t2),
                    t3=float(sig.t3),
                    rr=float(sig.rr),
                    leverage_max=float(sig.leverage_max),
                    risk_note=sig.risk_note,
                    invalidation=sig.invalidation,
                    notes=sig.notes,
                    sent_to_telegram=0,
                    message_id="",
                )
                signal_id = save_signal(conn, row)

                st.markdown(
                    f"""
<div class="card">
  <div>
    <span class="badge {'badge-bull' if sig.bias=='Bull' else 'badge-bear' if sig.bias=='Bear' else 'badge-chop'}">{sig.bias}</span>
    <span class="badge">{sig.symbol}</span>
    <span class="badge">{sig.timeframe.upper()}</span>
    <span class="badge badge-warn">RR {sig.rr:.2f}R</span>
  </div>
  <div style="margin-top:10px;" class="mono">{msg}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    st.caption("Telegram")
                    if send_live:
                        if not tg_token or not tg_chat_id:
                            st.error("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in secrets/env.")
                        else:
                            ok, info = telegram_send_message(tg_token, tg_chat_id, msg)
                            if ok:
                                mark_sent(conn, signal_id, info)
                                st.success("Sent ✅")
                            else:
                                st.error(f"Failed: {info}")
                    else:
                        st.info("Preview mode (not sent).")

                with c2:
                    st.caption("Copy")
                    st.code(msg, language="html")

                with c3:
                    st.caption("Quick outcome (optional)")
                    outcome = st.selectbox(
                        f"Outcome for #{signal_id} ({sig.symbol})",
                        ["(none)", "Win", "Loss", "BE", "Manual close"],
                        key=f"out_{signal_id}",
                    )
                    r_mult = st.number_input(f"R-multiple for #{signal_id}", value=0.0, step=0.1, key=f"r_{signal_id}")
                    note = st.text_input(f"Outcome note #{signal_id}", value="", key=f"note_{signal_id}")
                    if st.button("Save outcome", key=f"save_{signal_id}"):
                        if outcome != "(none)":
                            upsert_outcome(conn, signal_id, outcome, float(r_mult), note)
                            st.success("Outcome saved.")

                st.markdown("<hr/>", unsafe_allow_html=True)

    # =========================
    # Charts
    # =========================
    with tabs[1]:
        st.subheader("📊 Visual validation")
        if not PLOTLY_OK:
            st.info("Install plotly to enable charts.")
        else:
            sym2 = st.selectbox("Symbol", symbols if symbols else DEFAULT_SYMBOLS)
            tf2 = st.selectbox("Timeframe", DEFAULT_TIMEFRAMES, index=1, key="tf_chart")
            limit = st.slider("Bars", 200, 600, 350, 10)

            if st.button("Load chart", use_container_width=True):
                with st.spinner("Fetching..."):
                    df = fetch_ohlcv(exchange, sym2, tf2, limit=int(limit))
                    reg = determine_regime(df, int(fast_len), int(slow_len), int(atr_len))
                    title = f"{sym2} • {tf2.upper()} • Regime: {reg['trend']} • Bias: {reg['bias']}"

                    fig = make_chart(df, int(fast_len), int(slow_len), int(atr_len), title)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(
                        f"""
<div class="card">
  <div class="k">Regime diagnostics</div>
  <div class="v">Trend: <b>{reg['trend']}</b> | Bias: <b>{reg['bias']}</b> | Separation(EMA): <b>{reg['sep_now']:.2f}</b> | SlopeFast: <b>{reg['slope_fast']:.2f}</b></div>
  <div class="small">Separation and slope are ATR-normalized. Chop tends to print low absolute values.</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

    # =========================
    # Signal log
    # =========================
    with tabs[2]:
        st.subheader("🧾 Signal log")
        df = load_recent_signals(conn, limit=100)
        if df.empty:
            st.info("No signals logged yet.")
        else:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download signals JSON",
                data=df.to_json(orient="records", date_format="iso"),
                file_name="signals_log.json",
                mime="application/json",
                use_container_width=True,
            )

    # =========================
    # Outcomes
    # =========================
    with tabs[3]:
        st.subheader("🏁 Outcomes & performance")
        out = load_outcomes(conn)
        if out.empty:
            st.info("No outcomes yet.")
        else:
            # KPI
            closed = out.dropna(subset=["outcome"])
            total = len(out)
            closed_n = len(closed)

            win_n = int((closed["outcome"] == "Win").sum())
            loss_n = int((closed["outcome"] == "Loss").sum())
            be_n = int((closed["outcome"] == "BE").sum())

            avg_r = float(closed["r_multiple"].fillna(0).mean()) if closed_n else 0.0
            sum_r = float(closed["r_multiple"].fillna(0).sum()) if closed_n else 0.0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signals", total)
            c2.metric("Closed", closed_n)
            c3.metric("Win / Loss / BE", f"{win_n} / {loss_n} / {be_n}")
            c4.metric("Avg R / Total R", f"{avg_r:.2f} / {sum_r:.2f}")

            st.dataframe(out, use_container_width=True, hide_index=True)

            st.download_button(
                "Download outcomes CSV",
                data=out.to_csv(index=False),
                file_name="outcomes.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("---")
    st.caption("⚠️ Trading is risky. This tool is for educational/informational use. Use your own judgment and risk controls.")


if __name__ == "__main__":
    main()
