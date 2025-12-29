# =============================================================================
# TRADECAST PRO — TELEGRAM SIGNAL BROADCASTER (STREAMLIT .PY PORT)
# -----------------------------------------------------------------------------
# Motto / Non-Negotiables:
#   "No omissions. No assumptions. Base preserved."
#
# What this file is:
#   A full Streamlit single-file Python implementation of the uploaded Tradecast
#   React app features:
#     - Signal Center (validation + optional AI intel injection + Telegram broadcast)
#     - AI Reports Lab (AI report generation + Telegram broadcast)
#     - Script Forge (Pine Script generation)
#     - Gateways (Telegram + AI keys management via st.secrets + UI override)
#     - Plotly chart panel (mock market data like the React app)
#     - Persistent session logs + error dashboard
#
# Secrets (auto-load from Streamlit Secrets first, then env vars, then UI inputs):
#   - GEMINI_API_KEY
#   - OPENAI_API_KEY
#   - TELEGRAM_BOT_TOKEN
#   - TELEGRAM_CHAT_ID
#
# Notes / Constraints:
#   - This port intentionally preserves the React app’s “mock chart data” behavior.
#   - AI generation supports Gemini if available; otherwise falls back to OpenAI if present.
#   - Telegram sending is server-side (Streamlit) so no browser CORS issues.
# =============================================================================

from __future__ import annotations

import os
import json
import uuid
import time
import math
import random
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# PAGE CONFIG + THEME
# =============================================================================
st.set_page_config(
    page_title="TRADECAST PRO — Telegram Signal Broadcaster",
    page_icon="🧠",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    :root {
      --bg: #020617;
      --panel: #0f172a;
      --panel2: #071024;
      --border: rgba(148, 163, 184, 0.15);
      --text: #e2e8f0;
      --muted: rgba(226,232,240,0.6);
      --accent: #4f46e5;
      --ok: #10b981;
      --bad: #ef4444;
      --warn: #f59e0b;
    }

    .block-container { padding-top: 1.25rem; padding-bottom: 3rem; }
    body, .stApp { background: var(--bg); color: var(--text); }

    /* Tighten inputs */
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background: rgba(2,6,23,0.95) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        color: var(--text) !important;
    }

    /* Panels */
    .tc-panel {
        background: rgba(15,23,42,0.92);
        border: 1px solid var(--border);
        border-radius: 26px;
        padding: 22px 22px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.35);
    }

    .tc-subpanel {
        background: rgba(2,6,23,0.65);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 22px;
        padding: 18px 18px;
    }

    .tc-title {
        font-size: 26px;
        font-weight: 900;
        letter-spacing: -0.04em;
        margin: 0;
        padding: 0;
        text-transform: uppercase;
    }
    .tc-kicker {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.35em;
        text-transform: uppercase;
        color: var(--muted);
        margin-top: 4px;
        text-decoration: underline;
        text-decoration-color: rgba(148,163,184,0.25);
        text-decoration-thickness: 2px;
        text-underline-offset: 6px;
    }

    .tc-pill {
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(2,6,23,0.6);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: rgba(226,232,240,0.72);
    }

    .tc-pill-ok {
        border-color: rgba(16,185,129,0.25);
        color: rgba(16,185,129,0.95);
        background: rgba(16,185,129,0.08);
    }
    .tc-pill-bad {
        border-color: rgba(239,68,68,0.25);
        color: rgba(239,68,68,0.95);
        background: rgba(239,68,68,0.08);
    }
    .tc-pill-warn {
        border-color: rgba(245,158,11,0.25);
        color: rgba(245,158,11,0.95);
        background: rgba(245,158,11,0.08);
    }

    /* Log console */
    .tc-console {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 11px;
        line-height: 1.45;
        background: rgba(2,6,23,0.75);
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 18px;
        padding: 14px 14px;
        max-height: 280px;
        overflow: auto;
        white-space: pre-wrap;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# DATA MODELS (mirrors the TS types)
# =============================================================================
class SignalDirection:
    BUY = "BUY"
    SELL = "SELL"


class AnalysisDepth:
    QUICK = "QUICK"
    DETAILED = "DETAILED"
    QUANT = "QUANT"


@dataclass
class TradingSignal:
    id: str
    symbol: str
    direction: str
    entry: float
    tp1: float
    tp2: float
    tp3: float
    sl: float
    timeframe: str
    strategy: str
    timestamp: str  # ISO string


@dataclass
class GroundingSource:
    title: str
    uri: str


@dataclass
class AnalysisReport:
    title: str
    summary: str
    technicalDetails: str
    outlook: str  # BULLISH/BEARISH/NEUTRAL
    sources: Optional[List[GroundingSource]] = None
    depth: Optional[str] = None


@dataclass
class TelegramConfig:
    botToken: str
    chatId: str


# =============================================================================
# SESSION STATE INIT
# =============================================================================
def _ss_init() -> None:
    if "logs" not in st.session_state:
        st.session_state.logs = [
            "[SYSTEM] Core Engine: Gemini/OpenAI / Pine V6.0 Active",
            "[INFO] Neural reasoning pipelines established...",
        ]
    if "errors" not in st.session_state:
        st.session_state.errors = []
    if "telegram" not in st.session_state:
        st.session_state.telegram = {"botToken": "", "chatId": ""}
    if "signal" not in st.session_state:
        st.session_state.signal = {
            "id": "",
            "symbol": "BTC/USDT",
            "direction": SignalDirection.BUY,
            "entry": 64250.0,
            "tp1": 65500.0,
            "tp2": 67000.0,
            "tp3": 69000.0,
            "sl": 63000.0,
            "timeframe": "1H",
            "strategy": "Neural Momentum + RSI 6.0",
        }
    if "signalIntelMode" not in st.session_state:
        st.session_state.signalIntelMode = "NONE"  # NONE/QUICK/DEEP
    if "intelPreview" not in st.session_state:
        st.session_state.intelPreview = None
    if "analysisContext" not in st.session_state:
        st.session_state.analysisContext = ""
    if "reportDepth" not in st.session_state:
        st.session_state.reportDepth = AnalysisDepth.DETAILED
    if "currentReport" not in st.session_state:
        st.session_state.currentReport = None
    if "pinePrompt" not in st.session_state:
        st.session_state.pinePrompt = ""
    if "pineScript" not in st.session_state:
        st.session_state.pineScript = ""
    if "broadcastType" not in st.session_state:
        st.session_state.broadcastType = "STRICT SIGNAL"  # STRICT SIGNAL / AI RISK ANALYSIS / MARKET SUMMARY


_ss_init()


# =============================================================================
# UTILITIES: LOGS / SECRETS
# =============================================================================
def add_log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {msg}")


def add_error(err: str) -> None:
    st.session_state.errors.append(err)


def clear_errors() -> None:
    st.session_state.errors = []


def _read_secret(key: str) -> str:
    # Priority: st.secrets -> env var -> empty
    try:
        if key in st.secrets and st.secrets[key]:
            return str(st.secrets[key])
    except Exception:
        pass
    return str(os.environ.get(key, "") or "")


def load_config_from_secrets() -> Dict[str, str]:
    return {
        "GEMINI_API_KEY": _read_secret("GEMINI_API_KEY"),
        "OPENAI_API_KEY": _read_secret("OPENAI_API_KEY"),
        "TELEGRAM_BOT_TOKEN": _read_secret("TELEGRAM_BOT_TOKEN"),
        "TELEGRAM_CHAT_ID": _read_secret("TELEGRAM_CHAT_ID"),
    }


SECRETS = load_config_from_secrets()


# =============================================================================
# MOCK MARKET DATA (mirrors MarketChart.tsx behavior but vectorized with pandas)
# =============================================================================
TIMEFRAME_TO_MINUTES: Dict[str, int] = {
    "1M": 1,
    "5M": 5,
    "15M": 15,
    "1H": 60,
    "4H": 240,
    "1D": 1440,
    "1W": 10080,
}


def _symbol_seed(symbol: str) -> int:
    return sum(ord(c) for c in symbol)


def generate_mock_series(symbol: str, timeframe: str, n: int = 30) -> pd.DataFrame:
    """
    Vectorized mock series generator to emulate the React app:
      - seeded by symbol
      - timeframe impacts volatility scaling
      - BTC/ETH have different base prices
    """
    seed = _symbol_seed(symbol)
    tf = timeframe.upper().strip()
    minutes = TIMEFRAME_TO_MINUTES.get(tf, 15)

    # React heuristic: M -> 1, H -> 10, D -> 50, else 100 (approx)
    if "M" in tf:
        tf_mult = 1
    elif "H" in tf:
        tf_mult = 10
    elif "D" in tf:
        tf_mult = 50
    else:
        tf_mult = 100

    base_price = 100.0
    if "BTC" in symbol.upper():
        base_price = 64000.0
    elif "ETH" in symbol.upper():
        base_price = 3500.0

    # Deterministic RNG but still "alive": combine seed with date bucket
    # so it changes during the day while staying stable across reruns for a short window.
    day_bucket = int(time.time() // (60 * 15))  # 15-minute bucket
    rng = np.random.default_rng(seed + day_bucket)

    idx = pd.date_range(
        end=pd.Timestamp.utcnow(),
        periods=n,
        freq=pd.Timedelta(minutes=minutes),
    )

    i = np.arange(n, dtype=float)
    noise = (np.sin(seed + i * 0.4) * tf_mult * 3.0) + ((rng.random(n) - 0.5) * tf_mult)
    trend = i * tf_mult * 0.1
    price = base_price + noise + trend

    df = pd.DataFrame({"ts": idx, "price": price}).set_index("ts")
    # React code uses a 6-ish slice (i-5 to i) but names it ma20. We'll preserve output as "ma20".
    df["ma20"] = df["price"].rolling(6, min_periods=6).mean()
    return df


def build_price_chart(df: pd.DataFrame, symbol: str, timeframe: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["price"],
            mode="lines",
            name="Price",
            line=dict(width=3),
            fill="tozeroy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ma20"],
            mode="lines",
            name="MA20",
            line=dict(width=1.5, dash="dot"),
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(t=20, r=20, b=30, l=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(30,41,59,0.8)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(30,41,59,0.8)", zeroline=False, side="right")
    return fig


# =============================================================================
# TELEGRAM SENDER (server-side)
# =============================================================================
def telegram_send_message(config: TelegramConfig, text: str, parse_mode: str = "Markdown") -> Dict[str, Any]:
    if not config.botToken or not config.chatId:
        raise ValueError("Telegram Bot Token or Chat ID missing.")
    url = f"https://api.telegram.org/bot{config.botToken}/sendMessage"
    payload = {
        "chat_id": config.chatId,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    resp = requests.post(url, json=payload, timeout=30)
    if not resp.ok:
        try:
            data = resp.json()
            raise RuntimeError(data.get("description") or "Failed to send Telegram message")
        except Exception as e:
            raise RuntimeError(f"Failed to send Telegram message (HTTP {resp.status_code}): {resp.text[:500]}") from e
    return resp.json()


def format_signal_message(sig: TradingSignal) -> str:
    direction_emoji = "🟢 BUY" if sig.direction == SignalDirection.BUY else "🔴 SELL"
    return (
        f"🚀 *TRADING SIGNAL: {sig.symbol}* 🚀\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Direction: {direction_emoji}\n"
        f"Timeframe: {sig.timeframe}\n"
        f"Strategy: {sig.strategy}\n\n"
        f"📍 *ENTRY:* {sig.entry}\n"
        f"🎯 *TP1:* {sig.tp1}\n"
        f"🎯 *TP2:* {sig.tp2}\n"
        f"🎯 *TP3:* {sig.tp3}\n"
        f"🛑 *SL:* {sig.sl}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ Generated: {dt.datetime.now().strftime('%H:%M:%S')}\n"
    )


def format_report_message(rep: AnalysisReport) -> str:
    outlook = (rep.outlook or "NEUTRAL").upper()
    outlook_emoji = "📈" if outlook == "BULLISH" else "📉" if outlook == "BEARISH" else "↔️"
    msg = (
        f"📊 *TECHNICAL ANALYSIS: {rep.title}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{outlook_emoji} *Outlook:* {outlook}\n\n"
        f"📝 *Summary:*\n{rep.summary}\n\n"
        f"🔬 *Technical Details:*\n{rep.technicalDetails}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
    )
    if rep.sources:
        src_lines = [f"• [{s.title}]({s.uri})" for s in rep.sources if s.title and s.uri]
        if src_lines:
            msg += "\n🌐 *Sources:*\n" + "\n".join(src_lines)
    return msg


# =============================================================================
# AI ENGINE (Gemini preferred, OpenAI fallback)
# =============================================================================
def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Attempts to parse JSON even if the model wraps it in code fences or extra text.
    """
    if not text:
        return {}
    t = text.strip()
    # Strip triple backticks if present
    if t.startswith("```"):
        t = t.strip("`")
        # After stripping, it may still contain language hint
        t = t.replace("json", "", 1).strip()
    # Find first { ... } block
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end >= 0 and end > start:
        t = t[start : end + 1]
    return json.loads(t)


def _try_gemini_report(symbol: str, timeframe: str, context: str, depth: str) -> Optional[AnalysisReport]:
    """
    Gemini via HTTP REST (no dependency assumptions).
    Uses "gemini-3-pro-preview" for QUANT and "gemini-3-flash-preview" otherwise,
    matching the React app naming.
    """
    api_key = SECRETS.get("GEMINI_API_KEY") or ""
    if not api_key:
        return None

    depth_instructions = {
        AnalysisDepth.QUICK: "Executive summary and key levels only. Ultra-concise.",
        AnalysisDepth.DETAILED: "Full technical breakdown, sentiment analysis, and price action synthesis.",
        AnalysisDepth.QUANT: "Statistical probabilities, correlation matrices, and algorithmic order flow logic. Use deep math.",
    }

    model = "gemini-3-pro-preview" if depth == AnalysisDepth.QUANT else "gemini-3-flash-preview"

    # NOTE: Model availability can differ by account/region.
    # If this fails, we fall back to OpenAI below.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    system_task = (
        f"Task: Produce a {depth} Technical Intelligence Report for {symbol} ({timeframe}).\n"
        f"Context: {context}\n"
        f"Constraints: {depth_instructions.get(depth, depth_instructions[AnalysisDepth.DETAILED])}\n"
        f"Return ONLY valid JSON with keys: title, summary, technicalDetails, outlook.\n"
        f"outlook must be one of: BULLISH, BEARISH, NEUTRAL.\n"
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": system_task}]}],
        # We keep it simple: JSON-only instruction rather than strict schema.
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2048,
        },
    }

    resp = requests.post(url, json=payload, timeout=60)
    if not resp.ok:
        return None

    data = resp.json()
    # Extract text
    try:
        cand = data.get("candidates", [])[0]
        parts = cand.get("content", {}).get("parts", [])
        text = "".join([p.get("text", "") for p in parts])
        obj = _safe_json_loads(text)
    except Exception:
        return None

    title = str(obj.get("title") or f"{symbol} ({timeframe})")
    summary = str(obj.get("summary") or "")
    technical = str(obj.get("technicalDetails") or "")
    outlook = str(obj.get("outlook") or "NEUTRAL").upper()
    if outlook not in {"BULLISH", "BEARISH", "NEUTRAL"}:
        outlook = "NEUTRAL"

    return AnalysisReport(
        title=title,
        summary=summary,
        technicalDetails=technical,
        outlook=outlook,
        sources=None,  # REST response grounding differs; keep None to avoid false claims.
        depth=depth,
    )


def _try_openai_report(symbol: str, timeframe: str, context: str, depth: str) -> Optional[AnalysisReport]:
    api_key = SECRETS.get("OPENAI_API_KEY") or ""
    if not api_key:
        return None

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    depth_instructions = {
        AnalysisDepth.QUICK: "Executive summary and key levels only. Ultra-concise.",
        AnalysisDepth.DETAILED: "Full technical breakdown, sentiment analysis, and price action synthesis.",
        AnalysisDepth.QUANT: "Statistical probabilities, correlation matrices, and algorithmic order flow logic. Use deep math.",
    }

    client = OpenAI(api_key=api_key)

    prompt = (
        f"Produce a {depth} technical intelligence report for {symbol} ({timeframe}).\n"
        f"Context: {context}\n"
        f"Constraints: {depth_instructions.get(depth, depth_instructions[AnalysisDepth.DETAILED])}\n\n"
        f"Return ONLY valid JSON with keys:\n"
        f'  "title", "summary", "technicalDetails", "outlook"\n'
        f'Where outlook is one of: "BULLISH", "BEARISH", "NEUTRAL".\n'
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",  # pragmatic default; change in Secrets/ENV if desired
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = resp.choices[0].message.content or ""
        obj = _safe_json_loads(text)
    except Exception:
        return None

    title = str(obj.get("title") or f"{symbol} ({timeframe})")
    summary = str(obj.get("summary") or "")
    technical = str(obj.get("technicalDetails") or "")
    outlook = str(obj.get("outlook") or "NEUTRAL").upper()
    if outlook not in {"BULLISH", "BEARISH", "NEUTRAL"}:
        outlook = "NEUTRAL"

    return AnalysisReport(
        title=title,
        summary=summary,
        technicalDetails=technical,
        outlook=outlook,
        sources=None,
        depth=depth,
    )


def generate_technical_report(symbol: str, timeframe: str, context: str, depth: str) -> AnalysisReport:
    rep = _try_gemini_report(symbol, timeframe, context, depth)
    if rep:
        return rep
    rep = _try_openai_report(symbol, timeframe, context, depth)
    if rep:
        return rep
    raise RuntimeError("AI synthesis failed: No working GEMINI_API_KEY or OPENAI_API_KEY (or provider call failed).")


def _try_gemini_pine(indicator_name: str, logic: str) -> Optional[str]:
    api_key = SECRETS.get("GEMINI_API_KEY") or ""
    if not api_key:
        return None
    model = "gemini-3-pro-preview"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    prompt = (
        "Develop a high-performance TradingView Pine Script v6.0 indicator/strategy.\n"
        f'Name: "{indicator_name}"\n'
        f"Logic: {logic}\n"
        "Requirements: Strictly v6 syntax, optimized ta.* calls, proper namespace usage, professional styling.\n"
        "Return ONLY code.\n"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.25, "maxOutputTokens": 4096},
    }
    resp = requests.post(url, json=payload, timeout=90)
    if not resp.ok:
        return None
    data = resp.json()
    try:
        cand = data.get("candidates", [])[0]
        parts = cand.get("content", {}).get("parts", [])
        return "".join([p.get("text", "") for p in parts]).strip()
    except Exception:
        return None


def _try_openai_pine(indicator_name: str, logic: str) -> Optional[str]:
    api_key = SECRETS.get("OPENAI_API_KEY") or ""
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    client = OpenAI(api_key=api_key)
    prompt = (
        "Develop a high-performance TradingView Pine Script v6.0 indicator/strategy.\n"
        f'Name: "{indicator_name}"\n'
        f"Logic: {logic}\n"
        "Requirements: Strict v6 syntax, professional visuals, return ONLY code.\n"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None


def generate_pine_script(indicator_name: str, logic: str) -> str:
    code = _try_gemini_pine(indicator_name, logic)
    if code:
        return code
    code = _try_openai_pine(indicator_name, logic)
    if code:
        return code
    raise RuntimeError("Pine generation failed: No working GEMINI_API_KEY or OPENAI_API_KEY (or provider call failed).")


# =============================================================================
# SIGNAL VALIDATION (mirrors React chained logic)
# =============================================================================
def validate_signal(sig: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    symbol = str(sig.get("symbol", "")).strip()
    direction = str(sig.get("direction", SignalDirection.BUY))
    entry = float(sig.get("entry", 0) or 0)
    tp1 = float(sig.get("tp1", 0) or 0)
    tp2 = float(sig.get("tp2", 0) or 0)
    tp3 = float(sig.get("tp3", 0) or 0)
    sl = float(sig.get("sl", 0) or 0)

    if not symbol:
        errs.append("Critical Error: Market Symbol identification missing.")
    if entry <= 0:
        errs.append(f"Structural Error: Entry price ({entry}) must be a positive market value.")

    if direction == SignalDirection.BUY:
        if sl >= entry:
            errs.append(f"BUY Logic Error: Stop Loss ({sl}) must be strictly lower than Entry ({entry}).")
        if entry >= tp1:
            errs.append(f"BUY Logic Error: Take Profit 1 ({tp1}) must be strictly higher than Entry ({entry}).")
        if tp1 >= tp2:
            errs.append(f"BUY Logic Error: Take Profit 2 ({tp2}) must be strictly higher than Take Profit 1 ({tp1}).")
        if tp2 >= tp3:
            errs.append(f"BUY Logic Error: Take Profit 3 ({tp3}) must be strictly higher than Take Profit 2 ({tp2}).")
    else:
        if sl <= entry:
            errs.append(f"SELL Logic Error: Stop Loss ({sl}) must be strictly higher than Entry ({entry}).")
        if entry <= tp1:
            errs.append(f"SELL Logic Error: Take Profit 1 ({tp1}) must be strictly lower than Entry ({entry}).")
        if tp1 <= tp2:
            errs.append(f"SELL Logic Error: Take Profit 2 ({tp2}) must be strictly lower than Take Profit 1 ({tp1}).")
        if tp2 <= tp3:
            errs.append(f"SELL Logic Error: Take Profit 3 ({tp3}) must be strictly lower than Take Profit 2 ({tp2}).")

    return (len(errs) == 0, errs)


# =============================================================================
# BROADCAST ENGINE (Paramount Broadcast Types)
# =============================================================================
def broadcast_strict_signal(tg: TelegramConfig, sig: TradingSignal) -> None:
    telegram_send_message(tg, format_signal_message(sig), parse_mode="Markdown")


def broadcast_ai_risk_analysis(tg: TelegramConfig, sig: TradingSignal, rep: AnalysisReport) -> None:
    # A dedicated “risk” wrapper, while still respecting the React-style report fields.
    risk_header = (
        f"🧨 *AI RISK ANALYSIS* 🧨\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Symbol: *{sig.symbol}*\n"
        f"Direction: *{sig.direction}*\n"
        f"Timeframe: *{sig.timeframe}*\n"
        f"Entry: *{sig.entry}* | SL: *{sig.sl}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
    )
    body = (
        f"🧠 *Model Depth:* {rep.depth or 'DETAILED'}\n"
        f"📍 *Outlook:* {rep.outlook}\n\n"
        f"⚠️ *Risk Summary:*\n{rep.summary}\n\n"
        f"🔬 *Key Risk Factors / Notes:*\n{rep.technicalDetails}\n"
    )
    telegram_send_message(tg, risk_header + body, parse_mode="Markdown")


def broadcast_market_summary(tg: TelegramConfig, symbol: str, timeframe: str, rep: AnalysisReport) -> None:
    header = (
        f"🛰️ *MARKET SUMMARY* 🛰️\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Asset: *{symbol}* ({timeframe})\n"
        f"Outlook: *{rep.outlook}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
    )
    body = f"📝 *Summary:*\n{rep.summary}\n\n🔬 *Details:*\n{rep.technicalDetails}\n"
    telegram_send_message(tg, header + body, parse_mode="Markdown")


# =============================================================================
# UI — HEADER
# =============================================================================
st.markdown(
    f"""
<div class="tc-panel">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:16px;">
    <div>
      <div class="tc-title">TRADECAST PRO</div>
      <div class="tc-kicker">v3.0 NEURAL — Streamlit Port</div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end;">
      <span class="tc-pill">Search Grounding: ON*</span>
      <span class="tc-pill {'tc-pill-ok' if SECRETS.get('TELEGRAM_BOT_TOKEN') else ''}">Telegram: {'SECRETS' if SECRETS.get('TELEGRAM_BOT_TOKEN') else 'UI/EMPTY'}</span>
      <span class="tc-pill {'tc-pill-ok' if SECRETS.get('GEMINI_API_KEY') else 'tc-pill-warn' if SECRETS.get('OPENAI_API_KEY') else 'tc-pill-bad'}">AI: {'GEMINI' if SECRETS.get('GEMINI_API_KEY') else ('OPENAI' if SECRETS.get('OPENAI_API_KEY') else 'OFF')}</span>
    </div>
  </div>
  <div style="margin-top:10px;color:rgba(226,232,240,0.55);font-size:12px;font-weight:700;">
    *If your Gemini REST endpoint/model isn’t enabled for your key/account, the app automatically falls back to OpenAI when available.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


# =============================================================================
# SIDEBAR — QUICK GATEWAYS (non-destructive overrides)
# =============================================================================
with st.sidebar:
    st.markdown("### 🔐 Gateways (Quick Override)")
    st.caption("Secrets load automatically. These fields only override for this session.")

    # Default from secrets; allow override in session state
    if not st.session_state.telegram["botToken"]:
        st.session_state.telegram["botToken"] = SECRETS.get("TELEGRAM_BOT_TOKEN", "")
    if not st.session_state.telegram["chatId"]:
        st.session_state.telegram["chatId"] = SECRETS.get("TELEGRAM_CHAT_ID", "")

    st.session_state.telegram["botToken"] = st.text_input(
        "Telegram Bot Token",
        value=st.session_state.telegram["botToken"],
        type="password",
        placeholder="0000000:AAxxxx...",
    )
    st.session_state.telegram["chatId"] = st.text_input(
        "Telegram Chat ID",
        value=st.session_state.telegram["chatId"],
        placeholder="-100xxxx...",
    )

    st.divider()
    st.markdown("### 📡 Paramount Broadcast Type")
    st.session_state.broadcastType = st.radio(
        "Telegram Report Type",
        options=["STRICT SIGNAL", "AI RISK ANALYSIS", "MARKET SUMMARY"],
        index=["STRICT SIGNAL", "AI RISK ANALYSIS", "MARKET SUMMARY"].index(st.session_state.broadcastType),
    )

    st.divider()
    if st.button("🧪 Send Test Ping", use_container_width=True):
        try:
            tg = TelegramConfig(
                botToken=st.session_state.telegram["botToken"],
                chatId=st.session_state.telegram["chatId"],
            )
            telegram_send_message(tg, "✅ TRADECAST PRO ping OK.", parse_mode="Markdown")
            add_log("[SUCCESS] Telegram test ping delivered.")
        except Exception as e:
            add_log(f"[ERROR] Telegram test ping failed: {e}")

    st.divider()
    st.markdown("### 🧾 Logs")
    if st.button("Clear Logs", use_container_width=True):
        st.session_state.logs = ["[SYSTEM] Log buffer cleared."]

    st.markdown(
        f'<div class="tc-console">{st.session_state.logs[-120:] and "".join([line + "\\n" for line in st.session_state.logs[-120:]])}</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN TABS
# =============================================================================
tab_signal, tab_reports, tab_pine, tab_settings = st.tabs(
    ["📍 Signal Center", "📊 AI Reports Lab", "🧩 Script Forge", "⚙️ Gateways"]
)

# =============================================================================
# TAB 1: SIGNAL CENTER
# =============================================================================
with tab_signal:
    colA, colB = st.columns([1.55, 1.0], gap="large")

    # --- CHART PANEL
    with colA:
        st.markdown('<div class="tc-panel">', unsafe_allow_html=True)
        sym = st.session_state.signal["symbol"]
        tf = st.session_state.signal["timeframe"]
        df = generate_mock_series(sym, tf, n=30)

        current_price = float(df["price"].iloc[-1])
        prev_price = float(df["price"].iloc[-2]) if len(df) > 1 else current_price
        is_up = current_price >= prev_price
        pct = 0.0 if prev_price == 0 else abs(current_price - prev_price) / prev_price * 100.0

        st.markdown(
            f"""
            <div style="display:flex;align-items:flex-end;justify-content:space-between;gap:12px;flex-wrap:wrap;">
              <div>
                <div style="font-weight:900;font-size:20px;letter-spacing:-0.02em;">
                  {sym} <span class="tc-pill" style="margin-left:8px;">{tf}</span>
                </div>
                <div style="display:flex;align-items:center;gap:10px;margin-top:6px;">
                  <div style="font-family:ui-monospace,Menlo,Monaco,Consolas,'Courier New',monospace;
                              font-size:28px;font-weight:900;color:{'#10b981' if is_up else '#ef4444'};">
                    ${current_price:,.2f}
                  </div>
                  <div class="tc-pill {'tc-pill-ok' if is_up else 'tc-pill-bad'}">
                    {"▲" if is_up else "▼"} {pct:.2f}%
                  </div>
                </div>
              </div>
              <div style="display:flex;gap:14px;">
                <span class="tc-pill">Price</span>
                <span class="tc-pill">MA20*</span>
              </div>
            </div>
            <div style="margin-top:8px;color:rgba(226,232,240,0.55);font-size:12px;font-weight:700;">
              *MA20 label preserved from React app; computed as rolling(6) to match its behavior.
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig = build_price_chart(df, sym, tf)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- SIGNAL ARCHITECT PANEL
    with colB:
        st.markdown('<div class="tc-panel">', unsafe_allow_html=True)
        st.markdown("#### 📐 Signal Architect")

        # Validation errors
        ok, errs = validate_signal(st.session_state.signal)
        if errs:
            st.error("Protocol Validation Breach\n\n" + "\n".join([f"- {e}" for e in errs]))

        # Inputs
        st.session_state.signal["symbol"] = st.text_input(
            "Market Pair",
            value=st.session_state.signal["symbol"],
            placeholder="BTC/USDT",
        )

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.session_state.signal["direction"] = st.selectbox(
                "Position",
                options=[SignalDirection.BUY, SignalDirection.SELL],
                index=0 if st.session_state.signal["direction"] == SignalDirection.BUY else 1,
            )
        with dcol2:
            st.session_state.signal["timeframe"] = st.selectbox(
                "Interval",
                options=list(TIMEFRAME_TO_MINUTES.keys()),
                index=list(TIMEFRAME_TO_MINUTES.keys()).index(st.session_state.signal["timeframe"])
                if st.session_state.signal["timeframe"] in TIMEFRAME_TO_MINUTES
                else list(TIMEFRAME_TO_MINUTES.keys()).index("1H"),
            )

        ncol1, ncol2 = st.columns(2)
        with ncol1:
            st.session_state.signal["entry"] = st.number_input(
                "Entry Target",
                value=float(st.session_state.signal["entry"]),
                step=1.0,
            )
        with ncol2:
            st.session_state.signal["sl"] = st.number_input(
                "Stop Loss",
                value=float(st.session_state.signal["sl"]),
                step=1.0,
            )

        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            st.session_state.signal["tp1"] = st.number_input(
                "TP 1",
                value=float(st.session_state.signal["tp1"]),
                step=1.0,
            )
        with tcol2:
            st.session_state.signal["tp2"] = st.number_input(
                "TP 2",
                value=float(st.session_state.signal["tp2"]),
                step=1.0,
            )
        with tcol3:
            st.session_state.signal["tp3"] = st.number_input(
                "TP 3",
                value=float(st.session_state.signal["tp3"]),
                step=1.0,
            )

        st.session_state.signal["strategy"] = st.text_area(
            "Strategy / Notes",
            value=st.session_state.signal["strategy"],
            height=120,
        )

        st.divider()

        st.markdown("##### 🤖 Signal Intel Mode")
        st.session_state.signalIntelMode = st.selectbox(
            "Intel Injection",
            options=["NONE", "QUICK", "DEEP"],
            index=["NONE", "QUICK", "DEEP"].index(st.session_state.signalIntelMode),
            help="NONE = pure signal. QUICK/DEEP will call AI and append summary into the Strategy field before broadcast.",
        )

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("🧠 Generate Intel Preview", use_container_width=True, disabled=(st.session_state.signalIntelMode == "NONE")):
                try:
                    add_log(f"[AI] Invoking reasoning for {st.session_state.signal['symbol']}...")
                    depth = AnalysisDepth.QUICK if st.session_state.signalIntelMode == "QUICK" else AnalysisDepth.DETAILED
                    rep = generate_technical_report(
                        st.session_state.signal["symbol"],
                        st.session_state.signal["timeframe"],
                        f"Analyze: {st.session_state.signal['direction']} {st.session_state.signal['symbol']}. "
                        f"Strategy: {st.session_state.signal['strategy']}",
                        depth,
                    )
                    st.session_state.intelPreview = rep.summary
                    add_log(f"[AI] Intelligence preview complete. Outlook: {rep.outlook}")
                except Exception as e:
                    add_log(f"[ERROR] AI Synthesis failed: {e}")
                    st.session_state.intelPreview = None

        with btn2:
            if st.button("📡 Broadcast Signal", use_container_width=True):
                # Validate
                clear_errors()
                ok2, errs2 = validate_signal(st.session_state.signal)
                if not ok2:
                    add_log(f"[ERROR] Protocol Violation Detected: {len(errs2)} mathematical inconsistencies in signal.")
                else:
                    # Credentials
                    tg = TelegramConfig(
                        botToken=st.session_state.telegram["botToken"],
                        chatId=st.session_state.telegram["chatId"],
                    )
                    if not tg.botToken or not tg.chatId:
                        add_log("[ERROR] Gateway Auth Error: Credentials undefined.")
                        st.warning("Telegram credentials missing. Go to Gateways tab / sidebar.")
                    else:
                        try:
                            add_log(f"[ACTION] Deploying neural broadcast: {st.session_state.signal['symbol']}")
                            final_strategy = st.session_state.signal["strategy"]

                            rep_for_broadcast: Optional[AnalysisReport] = None
                            if st.session_state.signalIntelMode != "NONE":
                                depth = AnalysisDepth.QUICK if st.session_state.signalIntelMode == "QUICK" else AnalysisDepth.DETAILED
                                rep_for_broadcast = generate_technical_report(
                                    st.session_state.signal["symbol"],
                                    st.session_state.signal["timeframe"],
                                    st.session_state.signal["strategy"],
                                    depth,
                                )
                                final_strategy = f"{final_strategy}\n\n🤖 AI INTEL:\n{rep_for_broadcast.summary}"

                            sig_obj = TradingSignal(
                                id=str(uuid.uuid4()),
                                symbol=st.session_state.signal["symbol"],
                                direction=st.session_state.signal["direction"],
                                entry=float(st.session_state.signal["entry"]),
                                tp1=float(st.session_state.signal["tp1"]),
                                tp2=float(st.session_state.signal["tp2"]),
                                tp3=float(st.session_state.signal["tp3"]),
                                sl=float(st.session_state.signal["sl"]),
                                timeframe=st.session_state.signal["timeframe"],
                                strategy=final_strategy,
                                timestamp=dt.datetime.now().isoformat(timespec="seconds"),
                            )

                            # Paramount broadcast routing
                            btype = st.session_state.broadcastType
                            if btype == "STRICT SIGNAL":
                                broadcast_strict_signal(tg, sig_obj)
                            elif btype == "AI RISK ANALYSIS":
                                if rep_for_broadcast is None:
                                    rep_for_broadcast = generate_technical_report(
                                        sig_obj.symbol, sig_obj.timeframe, sig_obj.strategy, AnalysisDepth.DETAILED
                                    )
                                broadcast_ai_risk_analysis(tg, sig_obj, rep_for_broadcast)
                            else:  # MARKET SUMMARY
                                if rep_for_broadcast is None:
                                    rep_for_broadcast = generate_technical_report(
                                        sig_obj.symbol, sig_obj.timeframe, sig_obj.strategy, AnalysisDepth.QUICK
                                    )
                                broadcast_market_summary(tg, sig_obj.symbol, sig_obj.timeframe, rep_for_broadcast)

                            add_log("[SUCCESS] Broadcast delivered via Secure Telegram Gateway.")
                            st.success("Broadcast sent.")
                        except Exception as e:
                            add_log(f"[ERROR] Protocol Error: {e}")
                            st.error(str(e))

        if st.session_state.intelPreview:
            st.info(f"**Intel Preview:**\n\n{st.session_state.intelPreview}")

        st.divider()
        st.markdown("##### 🧾 Signal Preview")
        verified = "VERIFIED" if ok else "INVALID"
        st.markdown(
            f"""
            <div class="tc-subpanel">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span class="tc-pill">Logic Status</span>
                <span class="tc-pill {'tc-pill-ok' if ok else 'tc-pill-bad'}">{verified}</span>
              </div>
              <div style="margin-top:12px;font-family:ui-monospace,Menlo,Monaco,Consolas,'Courier New',monospace;font-weight:900;">
                DIRECTION: <span style="color:{'#10b981' if st.session_state.signal['direction']==SignalDirection.BUY else '#ef4444'};">
                  {'🟢 BUY' if st.session_state.signal['direction']==SignalDirection.BUY else '🔴 SELL'}
                </span><br/>
                TF: <span style="color:#fff;">{st.session_state.signal['timeframe']}</span>
                <div style="height:1px;background:rgba(148,163,184,0.15);margin:12px 0;"></div>
                📍 ENTRY: <span style="color:#fff;text-decoration:underline;text-decoration-color:rgba(79,70,229,0.55);">{float(st.session_state.signal['entry']):,.2f}</span><br/>
                🎯 TP1: <span style="color:#10b981;">{float(st.session_state.signal['tp1']):,.2f}</span><br/>
                🎯 TP2: <span style="color:#34d399;">{float(st.session_state.signal['tp2']):,.2f}</span><br/>
                🎯 TP3: <span style="color:#6ee7b7;">{float(st.session_state.signal['tp3']):,.2f}</span><br/>
                🛑 SL: <span style="color:#ef4444;">{float(st.session_state.signal['sl']):,.2f}</span><br/>
                <div style="margin-top:10px;color:rgba(79,70,229,0.85);font-size:11px;font-style:italic;font-weight:800;">
                  {st.session_state.signal['strategy']}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# TAB 2: AI REPORTS LAB
# =============================================================================
with tab_reports:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="tc-panel">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Neural Analytics")

        st.session_state.reportDepth = st.selectbox(
            "Reasoning Depth",
            options=[AnalysisDepth.QUICK, AnalysisDepth.DETAILED, AnalysisDepth.QUANT],
            index=[AnalysisDepth.QUICK, AnalysisDepth.DETAILED, AnalysisDepth.QUANT].index(st.session_state.reportDepth),
        )

        st.session_state.analysisContext = st.text_area(
            "Provide market vectors for deep synthesis…",
            value=st.session_state.analysisContext,
            height=260,
            placeholder="Example: Key levels, macro drivers, catalyst risk, funding/positioning notes…",
        )

        if st.button("⚡ Initialize Analysis", use_container_width=True, disabled=(not st.session_state.analysisContext.strip())):
            try:
                add_log(f"[AI] Initializing {st.session_state.reportDepth} analysis for {st.session_state.signal['symbol']}...")
                rep = generate_technical_report(
                    st.session_state.signal["symbol"],
                    st.session_state.signal["timeframe"],
                    st.session_state.analysisContext,
                    st.session_state.reportDepth,
                )
                st.session_state.currentReport = rep
                add_log("[AI] Synthesis complete.")
            except Exception as e:
                add_log(f"[ERROR] Neural engine mismatch: {e}")
                st.session_state.currentReport = None
                st.error(str(e))

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="tc-panel">', unsafe_allow_html=True)
        rep: Optional[AnalysisReport] = st.session_state.currentReport

        if rep is None:
            st.markdown(
                """
                <div style="opacity:0.5;text-align:center;padding:60px 10px;">
                  <div style="font-size:54px;">📄</div>
                  <div style="font-size:18px;font-weight:900;text-transform:uppercase;">Engine Standby</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            outlook = rep.outlook.upper()
            pill_class = "tc-pill-ok" if outlook == "BULLISH" else "tc-pill-bad" if outlook == "BEARISH" else "tc-pill-warn"
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap;">
                  <div style="font-size:22px;font-weight:900;text-transform:uppercase;text-decoration:underline;
                              text-decoration-color:rgba(79,70,229,0.85);text-decoration-thickness:4px;text-underline-offset:10px;">
                    {rep.title}
                  </div>
                  <div class="tc-pill {pill_class}">{outlook}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown('<div class="tc-subpanel">', unsafe_allow_html=True)
            st.markdown("**AI Executive Summary**")
            st.write(f"“{rep.summary}”")
            st.markdown("</div>", unsafe_allow_html=True)
            st.write("")
            st.markdown('<div class="tc-subpanel">', unsafe_allow_html=True)
            st.markdown("**Technical Details**")
            st.write(rep.technicalDetails)
            st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            if st.button("📡 Transmit Intelligence", use_container_width=True):
                tg = TelegramConfig(
                    botToken=st.session_state.telegram["botToken"],
                    chatId=st.session_state.telegram["chatId"],
                )
                if not tg.botToken or not tg.chatId:
                    add_log("[ERROR] Broadcast failed: Missing Telegram credentials.")
                    st.warning("Telegram credentials missing. Go to Gateways tab / sidebar.")
                else:
                    try:
                        add_log(f"[ACTION] Deploying intelligence broadcast: {rep.title}")
                        # Use the chosen Paramount broadcast type
                        btype = st.session_state.broadcastType
                        if btype == "STRICT SIGNAL":
                            # Strict signal doesn't apply directly to a report; we send the report in strict report format.
                            telegram_send_message(tg, format_report_message(rep), parse_mode="Markdown")
                        elif btype == "AI RISK ANALYSIS":
                            # Use current signal scaffold for risk framing
                            sig_obj = TradingSignal(
                                id=str(uuid.uuid4()),
                                symbol=st.session_state.signal["symbol"],
                                direction=st.session_state.signal["direction"],
                                entry=float(st.session_state.signal["entry"]),
                                tp1=float(st.session_state.signal["tp1"]),
                                tp2=float(st.session_state.signal["tp2"]),
                                tp3=float(st.session_state.signal["tp3"]),
                                sl=float(st.session_state.signal["sl"]),
                                timeframe=st.session_state.signal["timeframe"],
                                strategy=st.session_state.signal["strategy"],
                                timestamp=dt.datetime.now().isoformat(timespec="seconds"),
                            )
                            broadcast_ai_risk_analysis(tg, sig_obj, rep)
                        else:
                            broadcast_market_summary(tg, st.session_state.signal["symbol"], st.session_state.signal["timeframe"], rep)

                        add_log("[SUCCESS] Intelligence synthesis delivered via Secure Telegram Gateway.")
                        st.success("Report sent.")
                    except Exception as e:
                        add_log(f"[ERROR] Protocol Error: {e}")
                        st.error(str(e))

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# TAB 3: SCRIPT FORGE (Pine v6 generator)
# =============================================================================
with tab_pine:
    st.markdown('<div class="tc-panel">', unsafe_allow_html=True)
    st.markdown("#### 🧩 Pine Forge 6.0")

    st.session_state.pinePrompt = st.text_input(
        "Describe trading logic (e.g. RSI Divergence + Volume Profile)…",
        value=st.session_state.pinePrompt,
        placeholder="Example: EMA ribbon trend filter + volatility stop + alerts…",
    )

    if st.button("⚙️ Forge V6 Source", use_container_width=True, disabled=(not st.session_state.pinePrompt.strip())):
        try:
            add_log("[AI] Compiling logic to Pine Script v6.0 Standard...")
            code = generate_pine_script("Dark Singularity Engine", st.session_state.pinePrompt)
            st.session_state.pineScript = code
            add_log("[SUCCESS] Pine source generated.")
        except Exception as e:
            add_log(f"[ERROR] Compilation failed: {e}")
            st.session_state.pineScript = ""
            st.error(str(e))

    if st.session_state.pineScript:
        st.write("")
        st.markdown("##### ✅ Generated Pine Script (v6)")
        st.code(st.session_state.pineScript, language="pascal")

        st.download_button(
            "⬇️ Download Pine Script",
            data=st.session_state.pineScript.encode("utf-8"),
            file_name="tradecast_pine_v6.pine",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# TAB 4: SETTINGS / GATEWAYS (Secrets status + overrides)
# =============================================================================
with tab_settings:
    st.markdown('<div class="tc-panel">', unsafe_allow_html=True)
    st.markdown("#### ⚙️ System Gateways")

    st.markdown("**Secrets detected (read-only):**")
    s_cols = st.columns(2)
    with s_cols[0]:
        st.markdown(
            f"- TELEGRAM_BOT_TOKEN: {'✅' if SECRETS.get('TELEGRAM_BOT_TOKEN') else '❌'}\n"
            f"- TELEGRAM_CHAT_ID: {'✅' if SECRETS.get('TELEGRAM_CHAT_ID') else '❌'}"
        )
    with s_cols[1]:
        st.markdown(
            f"- GEMINI_API_KEY: {'✅' if SECRETS.get('GEMINI_API_KEY') else '❌'}\n"
            f"- OPENAI_API_KEY: {'✅' if SECRETS.get('OPENAI_API_KEY') else '❌'}"
        )

    st.write("")
    st.markdown("**Session overrides (editable):**")
    st.info("These override only for this Streamlit session. Secrets are not modified.")

    st.session_state.telegram["botToken"] = st.text_input(
        "Telegram API Key (Bot Token)",
        value=st.session_state.telegram["botToken"],
        type="password",
        placeholder="0000000:AAxxxx...",
        key="tg_token_settings",
    )
    st.session_state.telegram["chatId"] = st.text_input(
        "Destination Chat ID",
        value=st.session_state.telegram["chatId"],
        placeholder="-100xxxx...",
        key="tg_chat_settings",
    )

    st.write("")
    if st.button("🧪 Test Telegram Gateway", use_container_width=True):
        try:
            tg = TelegramConfig(
                botToken=st.session_state.telegram["botToken"],
                chatId=st.session_state.telegram["chatId"],
            )
            telegram_send_message(tg, "✅ TRADECAST PRO gateway test OK.", parse_mode="Markdown")
            add_log("[SUCCESS] Telegram gateway test delivered.")
            st.success("Telegram test message sent.")
        except Exception as e:
            add_log(f"[ERROR] Telegram gateway test failed: {e}")
            st.error(str(e))

    st.write("")
    st.markdown("**AI Provider Notes:**")
    st.markdown(
        "- If you provide **GEMINI_API_KEY**, the app will attempt Gemini REST first.\n"
        "- If Gemini fails (model not enabled / endpoint changes), it auto-falls back to **OPENAI_API_KEY** if present.\n"
        "- If neither works, AI actions will error cleanly in the UI/logs."
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# FOOTER — FULL LOG CONSOLE (redundant to sidebar, preserved)
# =============================================================================
st.write("")
with st.expander("🖥️ Neural Override Terminal (Logs)", expanded=False):
    st.markdown(
        f'<div class="tc-console">{st.session_state.logs[-240:] and "".join([line + "\\n" for line in st.session_state.logs[-240:]])}</div>',
        unsafe_allow_html=True,
    )
