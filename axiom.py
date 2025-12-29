# -*- coding: utf-8 -*-
"""
AXIOM — TITAN EDITION (Streamlit Fixed Build)
------------------------------------------------
FIX APPLIED:
- Removed React/JSX from Python. (Your SyntaxError happened because JSX like <span className="...">💠</span>
  was pasted into a .py file, which Python cannot parse.)
- Rebuilt the same UI/flow natively in Streamlit (sidebar selectors, tabs, metrics, charts, report generator,
  Telegram broadcast + log).

Run:
  streamlit run "00 02 ai-studio.py"
"""

import math
import time
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(layout="wide", page_title="💠 AXIOM — TITAN EDITION", page_icon="💠")

# Basic theme CSS (keeps your “darkpool” vibe, safely inside HTML strings)
st.markdown(
    """
    <style>
      :root { --axiom: #00F0FF; --bg: #050505; --panel: #0a0a0a; --border:#222; --muted:#6b7280; }
      html, body, [class*="css"] { background: var(--bg) !important; }
      .axiom-title { font-size: 28px; font-weight: 800; color: #fff; letter-spacing: -0.02em; margin:0; }
      .axiom-sub { color: var(--muted); font-size: 11px; letter-spacing: 0.25em; margin-left: 34px; margin-top:-6px;}
      .glass { background: rgba(255,255,255,0.03); border:1px solid #111; border-radius: 14px; padding: 14px; }
      .metric { background: rgba(255,255,255,0.03); border-left: 2px solid #333; border-radius: 12px; padding: 14px; }
      .metric:hover { border-left-color: var(--axiom); background: rgba(255,255,255,0.06); }
      .metric-label { color:#6b7280; font-size: 10px; letter-spacing: .22em; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;}
      .metric-value { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; font-size: 22px; font-weight: 300; }
      .metric-sub { color:#4b5563; font-size: 12px; }
      .pill { display:inline-block; font-size:10px; letter-spacing:.12em; font-weight:800; padding:4px 8px; border-radius:999px; background:#111; border:1px solid #222; }
      .pill-long { color:#00E676; }
      .pill-short { color:#FF1744; }
      .pill-wait { color:#9CA3AF; }
      .hr { height:1px; background:#111; margin: 10px 0 12px 0; }
      .small { font-size: 11px; color:#9CA3AF; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; }
      .btn-primary button { background: linear-gradient(90deg, #00F0FF, #0055FF) !important; color:#fff !important; font-weight:800 !important; }
      .btn-ghost button { background: #111 !important; color:#fff !important; border:1px solid #333 !important; }
      .btn-ghost button:hover { border-color: var(--axiom) !important; }
      .note { font-size: 10px; color:#6b7280; line-height:1.35; }
      .banner { background:#020202; border:1px solid #111; border-radius:14px; padding:10px 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# DATA / QUANT (Pure Python replacements for your React services)
# =============================================================================
ASSET_CLASSES: Dict[str, List[str]] = {
    "Crypto (Major)": ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "BNB-USD"],
    "US Indices": ["^GSPC", "^NDX", "^DJI", "^RUT"],
    "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F"],
    "FX Majors": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
    "Stocks (Mega)": ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"],
}

TIMEFRAMES = ["15m", "1h", "4h", "1d"]


def _tf_to_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    return 240


def generate_data(ticker: str, timeframe: str, bars: int = 240, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Deterministic-ish synthetic OHLCV + derived fields to match your UI expectations.
    (Replace with real feeds later; this is a working drop-in.)
    """
    if seed is None:
        seed = abs(hash((ticker, timeframe))) % (2**32)
    rng = np.random.default_rng(seed)

    tf_min = _tf_to_minutes(timeframe)
    end = dt.datetime.utcnow().replace(second=0, microsecond=0)
    idx = pd.date_range(end=end, periods=bars, freq=f"{tf_min}min")

    # price random walk + volatility regime
    base = 100.0 + (abs(hash(ticker)) % 5000) / 10.0
    vol = 0.0025 + (abs(hash(timeframe)) % 10) / 5000.0
    rets = rng.normal(0, vol, size=bars).cumsum()
    close = base * (1.0 + rets)

    # build OHLC
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, size=bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, size=bars))
    volume = (rng.lognormal(mean=10, sigma=0.25, size=bars)).astype(float)

    df = pd.DataFrame({"time": idx, "open": open_, "high": high, "low": low, "close": close, "volume": volume})

    # indicators to mimic your fields
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["atr14"] = (df["high"] - df["low"]).rolling(14).mean().fillna(method="bfill")

    # pseudo GM base line
    df["gm_apex_base"] = df["ema50"]

    # trend and signal
    df["nexus_trend"] = np.where(df["ema50"] >= df["ema200"], 1, -1)

    # signal: ema50 slope + distance from ema50 + trend agreement
    slope = df["ema50"].diff().fillna(0)
    dist = (df["close"] - df["ema50"]) / (df["atr14"].replace(0, np.nan)).fillna(1.0)
    raw = (np.sign(slope) + np.sign(dist)).clip(-1, 1)

    df["nexus_signal"] = np.where((raw > 0) & (df["nexus_trend"] == 1), 1,
                          np.where((raw < 0) & (df["nexus_trend"] == -1), -1, 0))

    # vector flux/state (toy)
    df["vector_flux"] = ((df["close"].pct_change().fillna(0)).rolling(14).mean() * 100).fillna(0)
    df["vector_state"] = np.where(df["vector_flux"] > 0.10, "Bull",
                           np.where(df["vector_flux"] < -0.10, "Bear", "Neutral"))

    # entropy proxy (chedo)
    volat = df["close"].pct_change().rolling(20).std().fillna(0)
    df["chedo"] = (volat / (volat.rolling(200).mean().replace(0, np.nan))).fillna(0).clip(0, 3)
    df["chedo"] = (df["chedo"] - 1.0)  # center-ish

    # risk line (trail)
    df["nexus_risk"] = np.where(df["nexus_trend"] == 1,
                         df["close"] - (df["atr14"] * 2.2),
                         df["close"] + (df["atr14"] * 2.2))

    return df


def calculate_fibs(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"smart_stop": 0.0, "tp1": 0.0, "tp2": 0.0, "tp3": 0.0}
    swing = df.tail(120)
    hi = float(swing["high"].max())
    lo = float(swing["low"].min())
    last = float(df["close"].iloc[-1])
    rng = max(hi - lo, 1e-9)

    # smart stop: beyond 0.786 from last direction
    smart_stop = last - 0.786 * rng if last >= (hi + lo) / 2 else last + 0.786 * rng
    tp1 = last + 0.236 * rng if last >= (hi + lo) / 2 else last - 0.236 * rng
    tp2 = last + 0.382 * rng if last >= (hi + lo) / 2 else last - 0.382 * rng
    tp3 = last + 0.618 * rng if last >= (hi + lo) / 2 else last - 0.618 * rng

    return {"smart_stop": smart_stop, "tp1": tp1, "tp2": tp2, "tp3": tp3}


def calculate_volume_profile(df: pd.DataFrame, bins: int = 30) -> Tuple[pd.DataFrame, float]:
    if df.empty:
        return pd.DataFrame({"price": [], "volume": []}), 0.0
    prices = df["close"].to_numpy()
    vols = df["volume"].to_numpy()
    lo, hi = float(np.min(prices)), float(np.max(prices))
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.digitize(prices, edges) - 1
    idx = np.clip(idx, 0, bins - 1)

    vol_by_bin = np.zeros(bins, dtype=float)
    for i, v in zip(idx, vols):
        vol_by_bin[i] += float(v)

    centers = (edges[:-1] + edges[1:]) / 2
    vp = pd.DataFrame({"price": centers, "volume": vol_by_bin})
    poc = float(vp.loc[vp["volume"].idxmax(), "price"]) if len(vp) else float(prices[-1])
    return vp, poc


def get_fundamentals_stub(ticker: str) -> Dict[str, str]:
    # Replace with real API when ready
    mk = f"{(abs(hash(ticker)) % 5000) / 10:.1f}B"
    pe = f"{(abs(hash(ticker)) % 40) + 8:.1f}"
    rg = f"+{(abs(hash(ticker)) % 35) / 10:.1f}%"
    summ = f"{ticker}: Synthetic fundamentals stub. Wire to a real feed when integrating market data."
    return {"marketCap": mk, "peRatio": pe, "revGrowth": rg, "summary": summ}


REPORT_TYPES = ["Quick Signal", "Full Report", "Risk Focus"]


def generate_report(report_type: str, ticker: str, df: pd.DataFrame, fibs: Dict[str, float], fundamentals: Dict[str, str]) -> str:
    last = df.iloc[-1]
    sig = "BUY" if int(last["nexus_signal"]) == 1 else "SELL" if int(last["nexus_signal"]) == -1 else "WAIT"
    bias = "LONG" if int(last["nexus_signal"]) == 1 else "SHORT" if int(last["nexus_signal"]) == -1 else "NEUTRAL"

    lines = []
    lines.append(f"💠 AXIOM | {report_type}")
    lines.append(f"Ticker: {ticker} | TF: {st.session_state.get('timeframe','')} | Signal: {sig} ({bias})")
    lines.append(f"Price: {last['close']:.2f} | Risk Line: {last['nexus_risk']:.2f}")
    lines.append(f"Vector: {last['vector_state']} | Flux: {last['vector_flux']:.2f}")
    lines.append(f"Entropy (CHEDO): {last['chedo']:.2f}")

    if report_type in ("Full Report", "Risk Focus"):
        lines.append("")
        lines.append("— Fibonacci Targets —")
        lines.append(f"Smart Stop: {fibs['smart_stop']:.2f}")
        lines.append(f"TP1: {fibs['tp1']:.2f} | TP2: {fibs['tp2']:.2f} | TP3: {fibs['tp3']:.2f}")

    if report_type == "Full Report":
        lines.append("")
        lines.append("— Fundamentals —")
        lines.append(f"Market Cap: {fundamentals['marketCap']} | P/E: {fundamentals['peRatio']} | Rev Growth: {fundamentals['revGrowth']}")
        lines.append(f"Summary: {fundamentals['summary']}")

    if report_type == "Risk Focus":
        lines.append("")
        lines.append("— Risk Notes —")
        risk = "HIGH" if abs(float(last["chedo"])) > 0.7 else "MODERATE"
        lines.append(f"Risk Regime: {risk} | Trail: nexus_risk active")

    return "\n".join(lines)


def telegram_send(token: str, chat_id: str, message: str) -> Tuple[bool, str]:
    if not token or not chat_id:
        return False, "Missing Telegram token or chat_id."
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": message, "disable_web_page_preview": True}, timeout=12)
        j = r.json()
        if r.status_code != 200 or not j.get("ok", False):
            return False, j.get("description", f"HTTP {r.status_code}")
        return True, "sent"
    except Exception as e:
        return False, str(e)


# =============================================================================
# SESSION STATE INIT
# =============================================================================
if "selectedClass" not in st.session_state:
    st.session_state.selectedClass = "Crypto (Major)"
if "selectedTicker" not in st.session_state:
    st.session_state.selectedTicker = "BTC-USD"
if "timeframe" not in st.session_state:
    st.session_state.timeframe = "4h"
if "isMobile" not in st.session_state:
    st.session_state.isMobile = False
if "broadcastMsg" not in st.session_state:
    st.session_state.broadcastMsg = ""
if "reportType" not in st.session_state:
    st.session_state.reportType = "Quick Signal"
if "broadcastLog" not in st.session_state:
    st.session_state.broadcastLog = []  # list of dicts
if "tickerSearch" not in st.session_state:
    st.session_state.tickerSearch = ""
if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""
if "telegram_token" not in st.session_state:
    st.session_state.telegram_token = ""
if "telegram_chat_id" not in st.session_state:
    st.session_state.telegram_chat_id = ""


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown('<p class="axiom-title"><span style="color:#00F0FF;">💠</span> AXIOM</p>', unsafe_allow_html=True)
    st.markdown('<div class="axiom-sub">TITAN EDITION</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.session_state.isMobile = st.toggle("MOBILE OPTIMIZED", value=st.session_state.isMobile)

    st.session_state.selectedClass = st.selectbox(
        "Sector",
        options=list(ASSET_CLASSES.keys()),
        index=list(ASSET_CLASSES.keys()).index(st.session_state.selectedClass) if st.session_state.selectedClass in ASSET_CLASSES else 0
    )

    st.session_state.tickerSearch = st.text_input("Ticker Search", value=st.session_state.tickerSearch, placeholder="Type to filter tickers...")

    tickers = ASSET_CLASSES.get(st.session_state.selectedClass, [])
    q = st.session_state.tickerSearch.strip().lower()
    filtered = [t for t in tickers if q in t.lower()] if q else tickers
    if not filtered:
        filtered = tickers

    # ensure selectedTicker still valid
    if st.session_state.selectedTicker not in tickers:
        st.session_state.selectedTicker = tickers[0] if tickers else ""

    st.session_state.selectedTicker = st.selectbox(
        "Ticker",
        options=filtered,
        index=filtered.index(st.session_state.selectedTicker) if st.session_state.selectedTicker in filtered else 0
    )

    st.session_state.timeframe = st.selectbox(
        "Interval",
        options=TIMEFRAMES,
        index=TIMEFRAMES.index(st.session_state.timeframe) if st.session_state.timeframe in TIMEFRAMES else 2
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.caption("API Credentials")

    st.session_state.openai_key = st.text_input("OpenAI Key", value=st.session_state.openai_key, type="password")
    st.session_state.telegram_token = st.text_input("Telegram Token", value=st.session_state.telegram_token, type="password")
    st.session_state.telegram_chat_id = st.text_input("Chat ID", value=st.session_state.telegram_chat_id)

    st.markdown(
        '<div class="note">Telegram is sent server-side from Streamlit (no browser CORS). '
        'If you deploy, keep secrets in Streamlit Secrets.</div>',
        unsafe_allow_html=True
    )


# =============================================================================
# MAIN HEADER: “TickerMarquee” + “LiveClock” equivalents
# =============================================================================
colA, colB = st.columns([3, 1], vertical_alignment="center")
with colA:
    st.markdown(
        f"""
        <div class="banner">
          <div class="mono" style="color:#6b7280; font-size:10px; letter-spacing:.22em; text-transform:uppercase;">
            Live Ticker Banner
          </div>
          <div class="mono" style="color:#00F0FF; font-weight:800; font-size:14px; margin-top:4px;">
            {st.session_state.selectedTicker} • {st.session_state.selectedClass} • {st.session_state.timeframe}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with colB:
    now = dt.datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f"""
        <div class="banner" style="text-align:right;">
          <div class="mono" style="color:#6b7280; font-size:10px; letter-spacing:.22em; text-transform:uppercase;">
            Live Clock
          </div>
          <div class="mono" style="color:#fff; font-weight:800; font-size:18px; margin-top:2px;">
            {now}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# =============================================================================
# LOAD DATA
# =============================================================================
df = generate_data(st.session_state.selectedTicker, st.session_state.timeframe)
fibs = calculate_fibs(df)
vp, poc = calculate_volume_profile(df)
fundamentals = get_fundamentals_stub(st.session_state.selectedTicker)
last = df.iloc[-1] if not df.empty else None

signal_bias = "NEUTRAL"
if last is not None:
    if int(last["nexus_signal"]) == 1:
        signal_bias = "LONG"
    elif int(last["nexus_signal"]) == -1:
        signal_bias = "SHORT"
    else:
        signal_bias = "WAIT"

confidence = 0
if last is not None:
    is_bull = float(last["gm_apex_base"]) < float(last["close"])
    if int(last["nexus_signal"]) == (1 if is_bull else -1):
        confidence += 50
    if abs(float(last["vector_flux"])) > 0.5:
        confidence += 25
    if (int(last["nexus_trend"]) == 1 and is_bull) or (int(last["nexus_trend"]) == -1 and not is_bull):
        confidence += 25


# =============================================================================
# METRICS ROW
# =============================================================================
def metric_card(label: str, value: str, sub: str, color: str = "#00F0FF"):
    st.markdown(
        f"""
        <div class="metric">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color};">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if st.session_state.isMobile:
    mcols = st.columns(1)
else:
    mcols = st.columns(4)

with mcols[0]:
    v = "WAIT"
    c = "#9CA3AF"
    if last is not None:
        if int(last["nexus_signal"]) == 1:
            v, c = "BUY", "#00E676"
        elif int(last["nexus_signal"]) == -1:
            v, c = "SELL", "#FF1744"
        else:
            v, c = "WAIT", "#9CA3AF"
    metric_card("Nexus Signal", v, "TRINITY SYSTEM", c)

if not st.session_state.isMobile:
    with mcols[1]:
        vs = str(last["vector_state"]) if last is not None else "Neutral"
        vf = float(last["vector_flux"]) if last is not None else 0.0
        metric_card("Vector State", vs, f"Flux: {vf:.2f}", "#00F0FF")

    with mcols[2]:
        ch = float(last["chedo"]) if last is not None else 0.0
        sub = "RISK" if abs(ch) > 0.7 else "STABLE"
        metric_card("Entropy (CHEDO)", f"{ch:.2f}", sub, "#D500F9")

    with mcols[3]:
        rl = float(last["nexus_risk"]) if last is not None else 0.0
        cl = float(last["close"]) if last is not None else 0.0
        col = "#00E676" if cl > rl else "#FF1744"
        metric_card("Risk Line", f"${rl:.2f}", "UT BOT TRAIL", col)
else:
    # mobile: show remaining in a second row
    m2 = st.columns(1)
    with m2[0]:
        vs = str(last["vector_state"]) if last is not None else "Neutral"
        vf = float(last["vector_flux"]) if last is not None else 0.0
        metric_card("Vector State", vs, f"Flux: {vf:.2f}", "#00F0FF")
    with m2[0]:
        ch = float(last["chedo"]) if last is not None else 0.0
        sub = "RISK" if abs(ch) > 0.7 else "STABLE"
        metric_card("Entropy (CHEDO)", f"{ch:.2f}", sub, "#D500F9")
    with m2[0]:
        rl = float(last["nexus_risk"]) if last is not None else 0.0
        cl = float(last["close"]) if last is not None else 0.0
        col = "#00E676" if cl > rl else "#FF1744"
        metric_card("Risk Line", f"${rl:.2f}", "UT BOT TRAIL", col)

st.write("")


# =============================================================================
# TABS (Tech / Macro / AI / Broadcast)
# =============================================================================
tab_tech, tab_macro, tab_ai, tab_broadcast = st.tabs(["Titan Tech", "Macro", "Intelligence", "Signals & Broadcast"])


# ------------------------------
# TECH TAB (Price, Entropy, Flux)
# ------------------------------
with tab_tech:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], name="EMA 50"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema200"], name="EMA 200"))
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=30, b=10),
        title=f"{st.session_state.selectedTicker} — Price",
        template="plotly_dark",
        xaxis_title="", yaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    # Entropy chart
    with c1:
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=df["time"], y=df["chedo"], name="CHEDO"))
        fig_e.update_layout(
            height=280, margin=dict(l=10, r=10, t=30, b=10),
            title="Entropy (CHEDO)",
            template="plotly_dark",
        )
        st.plotly_chart(fig_e, use_container_width=True)

    # Flux chart
    with c2:
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=df["time"], y=df["vector_flux"], name="Flux"))
        fig_f.update_layout(
            height=280, margin=dict(l=10, r=10, t=30, b=10),
            title="Flux Vector",
            template="plotly_dark",
        )
        st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------
# MACRO TAB (Fundamentals + Fibs)
# ------------------------------
with tab_macro:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#00F0FF; margin:0 0 10px 0;">Fundamentals</h3>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="mono" style="font-size:13px;">
              <div style="display:flex; justify-content:space-between;"><span>Market Cap</span><span style="color:#fff;">{fundamentals['marketCap']}</span></div>
              <div style="display:flex; justify-content:space-between;"><span>P/E Ratio</span><span style="color:#fff;">{fundamentals['peRatio']}</span></div>
              <div style="display:flex; justify-content:space-between;"><span>Rev Growth</span><span style="color:#00E676;">{fundamentals['revGrowth']}</span></div>
              <div style="margin-top:10px; border-top:1px solid #222; padding-top:10px; color:#9CA3AF; font-size:12px; font-style:italic;">
                {fundamentals['summary']}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#D500F9; margin:0 0 10px 0;">Fibonacci Targets</h3>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="mono" style="font-size:13px;">
              <div style="display:flex; justify-content:space-between; color:#f87171;"><span>Smart Stop</span><span>${fibs['smart_stop']:.2f}</span></div>
              <div style="display:flex; justify-content:space-between; color:#facc15;"><span>TP1</span><span>${fibs['tp1']:.2f}</span></div>
              <div style="display:flex; justify-content:space-between; color:#4ade80;"><span>TP2</span><span>${fibs['tp2']:.2f}</span></div>
              <div style="display:flex; justify-content:space-between; color:#00E676;"><span>TP3</span><span>${fibs['tp3']:.2f}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------
# AI TAB (UI placeholder like yours)
# ------------------------------
with tab_ai:
    st.markdown('<div class="glass" style="max-width:860px; margin:0 auto;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#fff; margin:0 0 10px 0;">🧠 AI Intelligence</h3>', unsafe_allow_html=True)

    mode = st.selectbox("Mode", ["Comprehensive Analysis", "Technical Breakdown", "Risk Assessment"])
    run = st.button("RUN ANALYSIS (OPENAI)")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    if run:
        # This is a safe placeholder; wire to OpenAI/Gemini using secrets when you’re ready.
        st.code(
            f"""Analyzing {st.session_state.selectedTicker} using God Mode algorithms...
> Mode: {mode}
> Checking CHEDO Entropy... {"Stable" if abs(float(last["chedo"])) <= 0.7 else "Risk"}
> Flux Vector... {"Bullish" if float(last["vector_flux"]) > 0 else "Bearish"} bias detected.
> Calculating SMC Order Blocks... Done.""",
            language="text",
        )
    else:
        st.code(
            f"""Analyzing {st.session_state.selectedTicker} using God Mode algorithms...
> Checking CHEDO Entropy... Stable.
> Flux Vector... Bullish Divergence detected.
> Calculating SMC Order Blocks... Done.""",
            language="text",
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------
# BROADCAST TAB (VP + Generator + Log)
# ------------------------------
with tab_broadcast:
    # Volume profile chart
    fig_vp = go.Figure()
    fig_vp.add_trace(go.Bar(x=vp["volume"], y=vp["price"], orientation="h", name="Volume"))
    fig_vp.add_vline(x=float(vp["volume"].max()) * 0.0, line_width=0)  # no-op but keeps layout consistent
    fig_vp.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        title=f"Volume Profile (POC: {poc:.2f})",
        template="plotly_dark",
        yaxis_title="Price",
        xaxis_title="Volume"
    )
    st.plotly_chart(fig_vp, use_container_width=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        # Bias pill + confidence meter
        pill_class = "pill-wait" if signal_bias == "WAIT" else "pill-long" if signal_bias == "LONG" else "pill-short"
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
              <div>
                <div style="color:#fff; font-weight:800; margin-bottom:6px;">📡 Signal Generator</div>
                <span class="pill {pill_class}">{signal_bias} BIAS</span>
              </div>
              <div style="text-align:right;">
                <div class="small">CONFIDENCE</div>
                <div class="mono" style="color:#00F0FF; font-weight:800;">{confidence}%</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(confidence / 100)

        st.session_state.reportType = st.selectbox("Report Type", REPORT_TYPES, index=REPORT_TYPES.index(st.session_state.reportType))

        gen = st.button("⚡ Generate Signal Report", key="gen_report")

        if gen:
            st.session_state.broadcastMsg = generate_report(
                st.session_state.reportType,
                st.session_state.selectedTicker,
                df,
                fibs,
                fundamentals
            )

        st.session_state.broadcastMsg = st.text_area(
            "Signal / Report",
            value=st.session_state.broadcastMsg,
            height=260,
            placeholder="Generate a report above or type a custom signal..."
        )

        bc_cols = st.columns([1, 1, 2])
        with bc_cols[0]:
            st.download_button(
                "COPY (download txt)",
                data=st.session_state.broadcastMsg or "",
                file_name=f"{st.session_state.selectedTicker}_signal.txt",
                mime="text/plain"
            )
        with bc_cols[1]:
            st.caption("")

        with bc_cols[2]:
            st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
            do_send = st.button("🚀 BROADCAST TO TELEGRAM", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        send_status = ""
        send_ok = None
        if do_send:
            if not st.session_state.broadcastMsg.strip():
                send_ok = False
                send_status = "No message to send."
            else:
                ok, msg = telegram_send(
                    st.session_state.telegram_token.strip(),
                    st.session_state.telegram_chat_id.strip(),
                    st.session_state.broadcastMsg
                )
                send_ok = ok
                send_status = msg

                # log (kept simple)
                st.session_state.broadcastLog.insert(
                    0,
                    {
                        "name": f"{st.session_state.selectedTicker} {st.session_state.reportType}",
                        "message": st.session_state.broadcastMsg,
                        "scheduleTime": dt.datetime.now().strftime("%H:%M:%S"),
                        "status": "sent" if ok else "error",
                    }
                )
                if ok:
                    st.session_state.broadcastMsg = ""

        if send_ok is True:
            st.success("SENT ✅")
        elif send_ok is False and send_status:
            st.error(f"ERROR ❌: {send_status}")

        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #222; padding-bottom:8px; margin-bottom:10px;">
              <div style="color:#9CA3AF; font-weight:800; font-size:12px; letter-spacing:.22em; text-transform:uppercase;">
                Broadcast Log
              </div>
              <div class="mono" style="color:#00F0FF; font-weight:800;">
                {len(st.session_state.broadcastLog)} SENT
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if len(st.session_state.broadcastLog) == 0:
            st.info("No signals transmitted.")
        else:
            for i, item in enumerate(st.session_state.broadcastLog[:25]):
                msg = item.get("message", "")
                name = item.get("name", "Broadcast")
                tm = item.get("scheduleTime", "")
                status = item.get("status", "sent")
                is_buy = ("BUY" in msg) or ("LONG" in msg) or ("🟢" in msg)
                is_sell = ("SELL" in msg) or ("SHORT" in msg) or ("🔴" in msg)

                border = "#00E67655" if is_buy else "#FF174455" if is_sell else "#333"
                st.markdown(
                    f"""
                    <div style="background:#0a0a0a; border:1px solid {border}; border-radius:12px; padding:10px; margin-bottom:10px;">
                      <div class="mono" style="display:flex; justify-content:space-between; color:#6b7280; font-size:10px; letter-spacing:.12em; text-transform:uppercase;">
                        <span style="color:#d1d5db; font-weight:800;">{name}</span>
                        <span>{tm} • {status}</span>
                      </div>
                      <div class="mono" style="color:#9CA3AF; font-size:12px; margin-top:6px; white-space:pre-wrap;">
                        {msg[:180]}{"..." if len(msg) > 180 else ""}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)
