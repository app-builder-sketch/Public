# streamlit_app.py
# =============================================================================
# STRICT EXPLICIT PROTOCOL — LEAD QUANTITATIVE ARCHITECT
# -----------------------------------------------------------------------------
# CRITICAL CONSTRAINTS (ENFORCED):
# 1) EXPLICIT INDICATOR LOCK:
#    - Forbidden to use/enable any technical indicator (RSI, MACD, EMA, etc.)
#      unless the *mathematical formula* has been explicitly provided by the user
#      in the current session and registered in the "User-Approved Registry".
# 2) NO ASSUMPTIONS:
#    - If an indicator is not in the User-Approved Registry, it does not exist.
# 3) NO OMISSIONS:
#    - This file is complete. Every iteration must output the full file.
# 4) PARAMOUNT BROADCAST ENGINE:
#    - Must support multiple distinct Telegram report types:
#      STRICT SIGNAL, AI RISK ANALYSIS, MARKET SUMMARY.
# 5) PURE PANDAS:
#    - Any calculations (if/when introduced) must use native Pandas vectorization.
# 6) SECRETS:
#    - Auto-load all API keys from Streamlit Secrets. Never hardcode credentials.
#
# NOTE:
# - This app does NOT compute indicators. It enforces the lock by blocking
#   unapproved indicator mentions/usages in user prompts and AI prompts.
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import json
import requests
import google.generativeai as genai
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import re

# ──────────────────────────────────────────────────────────────
# PAGE CONFIGURATION (MUST BE FIRST STREAMLIT CALL)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TRADECAST v3.0 NEURAL (Strict Explicit)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CONFIGURATION & SECRETS
# ──────────────────────────────────────────────────────────────
# Add to .streamlit/secrets.toml:
# [telegram]
# bot_token = "YOUR_BOT_TOKEN"
# chat_id = "YOUR_CHAT_ID"
# [gemini]
# api_key = "YOUR_GEMINI_API_KEY"
# model = "gemini-pro"  # optional

class Config:
    TELEGRAM_BOT_TOKEN = st.secrets.get("telegram", {}).get("bot_token", "")
    TELEGRAM_CHAT_ID = st.secrets.get("telegram", {}).get("chat_id", "")
    GEMINI_API_KEY = st.secrets.get("gemini", {}).get("api_key", "")
    GEMINI_MODEL = st.secrets.get("gemini", {}).get("model", "gemini-pro")

# ──────────────────────────────────────────────────────────────
# TYPE DEFINITIONS
# ──────────────────────────────────────────────────────────────
class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"

class AnalysisDepth(Enum):
    QUICK = "QUICK"
    DETAILED = "DETAILED"
    QUANT = "QUANT"

class ReportType(Enum):
    STRICT_SIGNAL = "STRICT SIGNAL"
    AI_RISK_ANALYSIS = "AI RISK ANALYSIS"
    MARKET_SUMMARY = "MARKET SUMMARY"

# ──────────────────────────────────────────────────────────────
# EXPLICIT INDICATOR LOCK — USER-APPROVED REGISTRY
# ──────────────────────────────────────────────────────────────
# We detect common indicator keywords. If any are present in user inputs
# (strategy text, Pine forge prompt, etc.) and NOT registered with a formula,
# we block the action and log a protocol violation.
#
# IMPORTANT:
# - Registering a formula does NOT automatically implement it. It only permits
#   the system to *reference* it in AI/Pine prompts under strict explicit rules.
#
# If you want actual indicator computation later, formulas must be provided and
# implemented with PURE Pandas vectorization (Constraint #5).

INDICATOR_KEYWORDS = [
    # moving averages
    "SMA", "EMA", "WMA", "HMA", "VWMA", "DEMA", "TEMA",
    # momentum/oscillators
    "RSI", "MACD", "STOCH", "STOCHASTIC", "CCI", "ADX", "DMI",
    "MOMENTUM", "ROC",
    # volatility/bands
    "ATR", "BOLLINGER", "BBANDS", "KELTNER", "DONCHIAN",
    # volume/orderflow
    "OBV", "VWAP", "VOLUME PROFILE", "VPVR", "DELTA",
    # trend systems
    "SUPERTREND", "PARABOLIC SAR", "PSAR", "ICHIMOKU",
    # misc
    "FIB", "FIBONACCI", "PIVOT", "PIVOT POINTS",
]

def _normalize_indicator_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().upper())

def _find_indicator_mentions(text: str) -> List[str]:
    if not text:
        return []
    t = text.upper()
    found = []
    for kw in INDICATOR_KEYWORDS:
        # keyword boundary-ish matching; allow spaces in keywords
        pattern = r"(?<![A-Z0-9])" + re.escape(kw) + r"(?![A-Z0-9])"
        if re.search(pattern, t):
            found.append(_normalize_indicator_name(kw))
    # Also catch "EMA200" / "RSI14" style mentions
    compact_patterns = {
        "EMA": r"\bEMA\s*\d+\b|\bEMA\d+\b",
        "SMA": r"\bSMA\s*\d+\b|\bSMA\d+\b",
        "RSI": r"\bRSI\s*\d+\b|\bRSI\d+\b",
        "ATR": r"\bATR\s*\d+\b|\bATR\d+\b",
        "MACD": r"\bMACD\b",
    }
    for name, pat in compact_patterns.items():
        if re.search(pat, t):
            found.append(_normalize_indicator_name(name))
    # unique
    return sorted(set(found))

def _unapproved_indicator_mentions(text: str, registry: Dict[str, str]) -> List[str]:
    mentions = _find_indicator_mentions(text)
    if not mentions:
        return []
    approved = { _normalize_indicator_name(k): v for k, v in registry.items() }
    unapproved = [m for m in mentions if m not in approved]
    return unapproved

def _extract_json_from_text(s: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extraction:
    - strips code fences
    - attempts direct json.loads
    - if fails, extracts the first {...} block and retries
    """
    if not s:
        return None
    cleaned = s.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Extract first JSON object block
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

# ──────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ──────────────────────────────────────────────────────────────
def init_session_state():
    if "logs" not in st.session_state:
        st.session_state.logs = [
            "[SYSTEM] Core Engine: Gemini / Pine Forge Active (Strict Explicit)",
            "[INFO] Neural reasoning pipelines established...",
            "[GUARD] EXPLICIT INDICATOR LOCK: ENABLED",
        ]
    if "signal" not in st.session_state:
        st.session_state.signal = {
            "symbol": "BTC/USDT",
            "direction": SignalDirection.BUY.value,
            "entry": 64250.0,
            "tp1": 65500.0,
            "tp2": 67000.0,
            "tp3": 69000.0,
            "sl": 63000.0,
            "timeframe": "1H",
            # IMPORTANT: Removed indicator reference to comply with lock.
            "strategy": "Price action + key levels + risk-defined invalidation",
        }
    if "errors" not in st.session_state:
        st.session_state.errors = []
    if "current_report" not in st.session_state:
        st.session_state.current_report = None
    if "pine_script" not in st.session_state:
        st.session_state.pine_script = ""
    if "indicator_registry" not in st.session_state:
        # { "RSI": "RSI = 100 - 100/(1+RS) ..." }
        st.session_state.indicator_registry = {}
    if "broadcast_type" not in st.session_state:
        st.session_state.broadcast_type = ReportType.STRICT_SIGNAL.value

def add_log(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")

# ──────────────────────────────────────────────────────────────
# VALIDATION ENGINE
# ──────────────────────────────────────────────────────────────
def validate_signal(signal: Dict[str, Any]) -> List[str]:
    errors = []
    direction = signal["direction"]

    if not signal["symbol"].strip():
        errors.append("Critical Error: Market Symbol identification missing.")

    if signal["entry"] <= 0:
        errors.append(f"Structural Error: Entry price ({signal['entry']}) must be positive.")

    # EXPLICIT INDICATOR LOCK (strategy text)
    unapproved = _unapproved_indicator_mentions(
        signal.get("strategy", ""),
        st.session_state.indicator_registry
    )
    if unapproved:
        errors.append(
            "EXPLICIT INDICATOR LOCK: Unapproved indicator mention(s) in Strategy: "
            + ", ".join(unapproved)
            + ". Register formulas in the Indicator Registry before referencing."
        )

    # Detailed Chained Validation
    if direction == SignalDirection.BUY.value:
        if signal["sl"] >= signal["entry"]:
            errors.append(f"BUY Logic Error: Stop Loss ({signal['sl']}) must be strictly lower than Entry ({signal['entry']}).")
        if signal["entry"] >= signal["tp1"]:
            errors.append(f"BUY Logic Error: Take Profit 1 ({signal['tp1']}) must be strictly higher than Entry ({signal['entry']}).")
        if signal["tp1"] >= signal["tp2"]:
            errors.append(f"BUY Logic Error: Take Profit 2 ({signal['tp2']}) must be strictly higher than Take Profit 1 ({signal['tp1']}).")
        if signal["tp2"] >= signal["tp3"]:
            errors.append(f"BUY Logic Error: Take Profit 3 ({signal['tp3']}) must be strictly higher than Take Profit 2 ({signal['tp2']}).")
    else:
        if signal["sl"] <= signal["entry"]:
            errors.append(f"SELL Logic Error: Stop Loss ({signal['sl']}) must be strictly higher than Entry ({signal['entry']}).")
        if signal["entry"] <= signal["tp1"]:
            errors.append(f"SELL Logic Error: Take Profit 1 ({signal['tp1']}) must be strictly lower than Entry ({signal['entry']}).")
        if signal["tp1"] <= signal["tp2"]:
            errors.append(f"SELL Logic Error: Take Profit 2 ({signal['tp2']}) must be strictly lower than Take Profit 1 ({signal['tp1']}).")
        if signal["tp2"] <= signal["tp3"]:
            errors.append(f"SELL Logic Error: Take Profit 3 ({signal['tp3']}) must be strictly lower than Take Profit 2 ({signal['tp2']}).")

    if errors:
        add_log(f"[ERROR] Protocol Violation Detected: {len(errors)} issue(s).")

    return errors

# ──────────────────────────────────────────────────────────────
# AI SERVICES (STRICT EXPLICIT SAFE PROMPTS)
# ──────────────────────────────────────────────────────────────
def initialize_gemini() -> bool:
    try:
        genai.configure(api_key=Config.GEMINI_API_KEY)
        return True
    except Exception as e:
        add_log(f"[ERROR] Gemini init failed: {str(e)}")
        return False

def _build_indicator_registry_block(registry: Dict[str, str]) -> str:
    if not registry:
        return (
            "INDICATOR POLICY:\n"
            "- You MUST NOT use or reference any technical indicators.\n"
            "- Focus strictly on price action, structure, support/resistance, risk/invalidation, and scenario analysis.\n"
        )
    lines = [
        "INDICATOR POLICY:",
        "- You MAY ONLY reference the following user-approved indicators, exactly as defined by the provided formulas.",
        "- Do NOT introduce any other indicators.",
        "",
        "USER-APPROVED INDICATOR REGISTRY (Name -> Formula):",
    ]
    for k, v in registry.items():
        kk = _normalize_indicator_name(k)
        vv = v.strip()
        lines.append(f"- {kk}: {vv}")
    return "\n".join(lines)

def generate_technical_report(
    symbol: str,
    timeframe: str,
    context: str,
    depth: AnalysisDepth
) -> Optional[Dict[str, Any]]:
    if not Config.GEMINI_API_KEY:
        add_log("[ERROR] Gemini API key not configured in secrets")
        return None

    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        temperature = 0.1 if depth == AnalysisDepth.QUANT else 0.3

        policy_block = _build_indicator_registry_block(st.session_state.indicator_registry)

        prompt = f"""
You are operating under STRICT EXPLICIT protocol.

{policy_block}

Task:
Conduct a {depth.value} market analysis for {symbol} on {timeframe} timeframe.
Use ONLY the provided context. If context lacks data, state limitations clearly.

Context:
{context}

Return JSON ONLY in this exact schema:
{{
  "title": "Asset Analysis",
  "outlook": "BULLISH|BEARISH|NEUTRAL",
  "summary": "Executive summary in 2-3 sentences",
  "technicalDetails": "Detailed analysis with key levels, invalidation, scenarios, and risk notes",
  "sources": [{{"title": "Source Name", "uri": "https://..."}}]
}}
""".strip()

        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        parsed = _extract_json_from_text(getattr(response, "text", "") or "")
        if not parsed:
            add_log("[ERROR] AI Synthesis failed: JSON parse error")
            return None
        return parsed

    except Exception as e:
        add_log(f"[ERROR] AI Synthesis failed: {str(e)}")
        return None

def generate_ai_risk_analysis(
    signal: Dict[str, Any],
    context: str,
    depth: AnalysisDepth
) -> Optional[Dict[str, Any]]:
    if not Config.GEMINI_API_KEY:
        add_log("[ERROR] Gemini API key not configured in secrets")
        return None

    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        temperature = 0.1 if depth == AnalysisDepth.QUANT else 0.25

        policy_block = _build_indicator_registry_block(st.session_state.indicator_registry)

        prompt = f"""
You are operating under STRICT EXPLICIT protocol.

{policy_block}

Task:
Produce an AI RISK ANALYSIS for the following trade idea. Emphasize risk, invalidation, and execution hazards.
Do NOT invent data. Use only given inputs. If missing, say so.

Trade:
- Symbol: {signal.get("symbol")}
- Direction: {signal.get("direction")}
- Timeframe: {signal.get("timeframe")}
- Entry: {signal.get("entry")}
- Stop Loss: {signal.get("sl")}
- TP1: {signal.get("tp1")}
- TP2: {signal.get("tp2")}
- TP3: {signal.get("tp3")}
- Strategy basis: {signal.get("strategy")}

Context:
{context}

Return JSON ONLY in this schema:
{{
  "title": "AI Risk Analysis",
  "riskScore": 0,
  "riskLabel": "LOW|MEDIUM|HIGH",
  "keyRisks": ["..."],
  "invalidation": "Clear invalidation conditions",
  "executionPlan": "Execution notes: entries, partials, stops",
  "positionSizing": "Sizing guidance (conceptual, non-financial advice)",
  "notes": "Any constraints/limitations",
  "sources": [{{"title": "Source Name", "uri": "https://..."}}]
}}
""".strip()

        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        parsed = _extract_json_from_text(getattr(response, "text", "") or "")
        if not parsed:
            add_log("[ERROR] AI Risk failed: JSON parse error")
            return None
        return parsed

    except Exception as e:
        add_log(f"[ERROR] AI Risk failed: {str(e)}")
        return None

def generate_market_summary(
    context: str,
    depth: AnalysisDepth
) -> Optional[Dict[str, Any]]:
    if not Config.GEMINI_API_KEY:
        add_log("[ERROR] Gemini API key not configured in secrets")
        return None

    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        temperature = 0.15 if depth == AnalysisDepth.QUANT else 0.35

        policy_block = _build_indicator_registry_block(st.session_state.indicator_registry)

        prompt = f"""
You are operating under STRICT EXPLICIT protocol.

{policy_block}

Task:
Create a MARKET SUMMARY (macro + risk-on/off tone + key levels overview) based ONLY on the provided context.
Do NOT introduce technical indicators unless they are explicitly allowed in the registry above.

Context:
{context}

Return JSON ONLY in this schema:
{{
  "title": "Market Summary",
  "tone": "RISK-ON|RISK-OFF|MIXED|UNCLEAR",
  "headline": "One-line headline",
  "summary": "5-8 bullet points (as a single string with newlines)",
  "watchLevels": "Key levels to watch (text)",
  "riskNotes": "Main risks (text)",
  "sources": [{{"title": "Source Name", "uri": "https://..."}}]
}}
""".strip()

        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        parsed = _extract_json_from_text(getattr(response, "text", "") or "")
        if not parsed:
            add_log("[ERROR] Market Summary failed: JSON parse error")
            return None
        return parsed

    except Exception as e:
        add_log(f"[ERROR] Market Summary failed: {str(e)}")
        return None

def generate_pine_script(name: str, logic: str) -> str:
    # Enforce indicator lock BEFORE calling AI
    unapproved = _unapproved_indicator_mentions(logic, st.session_state.indicator_registry)
    if unapproved:
        add_log(f"[ERROR] Pine Forge blocked: Unapproved indicator(s) referenced: {', '.join(unapproved)}")
        return (
            "// ERROR: EXPLICIT INDICATOR LOCK\n"
            "// Unapproved indicator references detected in your logic prompt:\n"
            f"// {', '.join(unapproved)}\n"
            "// Register formulas in the Indicator Registry before requesting Pine generation.\n"
        )

    if not Config.GEMINI_API_KEY:
        add_log("[ERROR] Gemini API key not configured in secrets")
        return "// Error: API key missing"

    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        policy_block = _build_indicator_registry_block(st.session_state.indicator_registry)

        prompt = f"""
You are generating TradingView Pine Script under STRICT EXPLICIT protocol.

{policy_block}

Generate a complete Pine Script v6 indicator named: {name}
Based on the following logic description (must not introduce unapproved indicators):
{logic}

Requirements:
- Use //@version=6
- Include proper input() declarations
- Add alertcondition() for signals
- Use explicit variable typing
- Follow TradingView style guide
- Do NOT include explanations; return ONLY code.
""".strip()

        response = model.generate_content(prompt, generation_config={"temperature": 0.2})
        return (getattr(response, "text", "") or "").replace("```pinescript", "").replace("```", "").strip()

    except Exception as e:
        add_log(f"[ERROR] Compilation failed: {str(e)}")
        return f"// Error: {str(e)}"

# ──────────────────────────────────────────────────────────────
# TELEGRAM SERVICES
# ──────────────────────────────────────────────────────────────
def send_to_telegram(message: str, parse_mode: str = "HTML") -> bool:
    if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
        add_log("[ERROR] Gateway Auth Error: Telegram credentials undefined")
        return False

    try:
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": Config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": parse_mode}
        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            add_log("[SUCCESS] Broadcast delivered via Secure Telegram Gateway")
            return True
        add_log(f"[ERROR] Telegram API: {response.text}")
        return False

    except Exception as e:
        add_log(f"[ERROR] Protocol Error: {str(e)}")
        return False

def format_signal_message(signal: Dict[str, Any]) -> str:
    emoji = "🟢" if signal["direction"] == SignalDirection.BUY.value else "🔴"
    return f"""🚀 <b>{ReportType.STRICT_SIGNAL.value}: {signal['symbol']} 🚀</b>

📊 <b>Direction:</b> {emoji} {signal['direction']}
⏱ <b>Timeframe:</b> {signal['timeframe']}

📍 <b>Entry:</b> <code>{signal['entry']}</code>
🎯 <b>TP1:</b> <code>{signal['tp1']}</code>
🎯 <b>TP2:</b> <code>{signal['tp2']}</code>
🎯 <b>TP3:</b> <code>{signal['tp3']}</code>
🛑 <b>SL:</b> <code>{signal['sl']}</code>

💡 <b>Basis:</b> {signal['strategy']}

⚡ <b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

def format_ai_risk_message(report: Dict[str, Any], symbol: str) -> str:
    return f"""🧠 <b>{ReportType.AI_RISK_ANALYSIS.value}: {symbol}</b>

⚠️ <b>Risk Score:</b> <code>{report.get('riskScore')}</code> / 10
🏷 <b>Label:</b> {report.get('riskLabel')}

🧩 <b>Key Risks:</b>
{_to_bullets(report.get('keyRisks', []))}

🧨 <b>Invalidation:</b> {report.get('invalidation')}

🛠 <b>Execution Plan:</b> {report.get('executionPlan')}

📏 <b>Position Sizing:</b> {report.get('positionSizing')}

📝 <b>Notes:</b> {report.get('notes')}

🔗 <b>Sources:</b> {len(report.get('sources', []))} references
"""

def format_market_summary_message(report: Dict[str, Any]) -> str:
    return f"""🌐 <b>{ReportType.MARKET_SUMMARY.value}</b>

🎛 <b>Tone:</b> {report.get('tone')}
🗞 <b>Headline:</b> {report.get('headline')}

🧾 <b>Summary:</b>
{report.get('summary')}

🎯 <b>Watch Levels:</b> {report.get('watchLevels')}

⚠️ <b>Risk Notes:</b> {report.get('riskNotes')}

🔗 <b>Sources:</b> {len(report.get('sources', []))} references
"""

def _to_bullets(items: Any) -> str:
    if not items:
        return "• (none)"
    if isinstance(items, list):
        return "\n".join([f"• {str(x)}" for x in items])
    return f"• {str(items)}"

def format_report_message(report: Dict[str, Any]) -> str:
    # Backward compatible generic AI report format
    return f"""📊 <b>{report.get('title', 'Report')}</b>

🎯 <b>Market Outlook:</b> {report.get('outlook', 'N/A')}

📝 <b>Summary:</b> {report.get('summary', '')}

📈 <b>Technical Analysis:</b> {report.get('technicalDetails', '')}

🔗 <b>Sources:</b> {len(report.get('sources', []))} references
"""

# ──────────────────────────────────────────────────────────────
# UI COMPONENTS
# ──────────────────────────────────────────────────────────────
def render_chart():
    # Mock chart - replace with real data fetch in a future iteration.
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=["00:00", "01:00", "02:00", "03:00", "04:00"],
            open=[64000, 64200, 64100, 64300, 64400],
            high=[64200, 64300, 64200, 64500, 64600],
            low=[63900, 64000, 64000, 64100, 64300],
            close=[64100, 64250, 64150, 64400, 64500],
            name="MOCK",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#020617",
        paper_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=10),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def render_logs():
    with st.sidebar:
        st.markdown("### 🖥️ NEURAL TERMINAL")
        with st.container():
            for log in st.session_state.logs[-20:]:
                st.code(log, language=None)

def render_indicator_registry():
    st.header("🧾 USER-APPROVED INDICATOR REGISTRY")
    st.warning(
        "EXPLICIT INDICATOR LOCK is ACTIVE. "
        "If you mention indicators in Strategy/AI/Pine prompts, you MUST register the mathematical formula here first. "
        "Registering does not auto-implement calculations; it only permits referencing under strict explicit rules."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Add / Update Formula")
        name = st.text_input("Indicator Name (e.g., RSI, EMA)", value="")
        formula = st.text_area("Mathematical Formula (as provided by you)", height=180, placeholder="Paste the exact math definition here.")
        if st.button("➕ Register Formula", type="primary", use_container_width=True):
            if not name.strip() or not formula.strip():
                st.error("Name and formula are required.")
            else:
                k = _normalize_indicator_name(name)
                st.session_state.indicator_registry[k] = formula.strip()
                add_log(f"[REGISTRY] Added/Updated indicator formula: {k}")
                st.success(f"Registered: {k}")

    with col2:
        st.subheader("Current Registry")
        reg = st.session_state.indicator_registry
        if not reg:
            st.info("Registry is empty. All indicator references are blocked.")
        else:
            for k in sorted(reg.keys()):
                with st.expander(f"{k}", expanded=False):
                    st.code(reg[k], language=None)
                    if st.button(f"🗑 Remove {k}", key=f"rm_{k}"):
                        st.session_state.indicator_registry.pop(k, None)
                        add_log(f"[REGISTRY] Removed indicator formula: {k}")
                        st.success(f"Removed: {k}")

# ──────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────
def main():
    # Custom CSS
    st.markdown(
        """
        <style>
        .stButton>button {
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .block-container {
            padding-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()

    # Sidebar Navigation
    with st.sidebar:
        st.title("🧠 TRADECAST")
        st.markdown("### v3.0 NEURAL — Strict Explicit")

        tab = st.radio(
            "Navigation",
            ["📊 Signal Center", "📈 AI Reports Lab", "⚡ Script Forge", "🧾 Indicator Registry", "🔐 Gateways"],
            label_visibility="collapsed",
        )

        # API Status Indicators
        st.markdown("---")
        st.markdown("**API Status**")
        if Config.GEMINI_API_KEY:
            st.success("✅ Gemini Active")
        else:
            st.error("❌ Gemini Inactive")

        if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
            st.success("✅ Telegram Active")
        else:
            st.error("❌ Telegram Inactive")

        st.markdown("---")
        st.caption("Guardrails: EXPLICIT INDICATOR LOCK is ON.")

    # Main Content
    if tab == "📊 Signal Center":
        render_signal_center()
    elif tab == "📈 AI Reports Lab":
        render_reports_lab()
    elif tab == "⚡ Script Forge":
        render_script_forge()
    elif tab == "🧾 Indicator Registry":
        render_indicator_registry()
    elif tab == "🔐 Gateways":
        render_gateways()

    # Always render terminal at bottom
    render_logs()

def render_signal_center():
    st.header("📡 SIGNAL ARCHITECT")

    # Error Display
    if st.session_state.errors:
        st.error("### 🔴 PROTOCOL VALIDATION BREACH")
        for error in st.session_state.errors:
            st.markdown(f"- {error}")

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.form("signal_form"):
            col_left, col_right = st.columns(2)

            with col_left:
                symbol = st.text_input("Market Pair", value=st.session_state.signal["symbol"])
                direction = st.radio(
                    "Position",
                    ["BUY", "SELL"],
                    index=0 if st.session_state.signal["direction"] == "BUY" else 1,
                )
                entry = st.number_input("Entry Target", value=float(st.session_state.signal["entry"]), format="%.2f")
                sl = st.number_input("Stop Loss", value=float(st.session_state.signal["sl"]), format="%.2f")

            with col_right:
                timeframe_options = ["1M", "5M", "15M", "1H", "4H", "1D", "1W"]
                timeframe = st.selectbox(
                    "Interval",
                    timeframe_options,
                    index=timeframe_options.index(st.session_state.signal["timeframe"]),
                )
                tp1 = st.number_input("TP 1", value=float(st.session_state.signal["tp1"]), format="%.2f")
                tp2 = st.number_input("TP 2", value=float(st.session_state.signal["tp2"]), format="%.2f")
                tp3 = st.number_input("TP 3", value=float(st.session_state.signal["tp3"]), format="%.2f")

            strategy = st.text_area("Reasoning Basis", value=st.session_state.signal["strategy"], height=120)

            # Paramount Broadcast Engine: choose report type
            broadcast_type = st.selectbox(
                "Broadcast Type",
                [rt.value for rt in ReportType],
                index=[rt.value for rt in ReportType].index(st.session_state.broadcast_type),
                help="STRICT SIGNAL sends the structured trade. AI RISK ANALYSIS and MARKET SUMMARY require Gemini key.",
            )

            submitted = st.form_submit_button("🚀 DEPLOY ENCRYPTED BROADCAST", type="primary", use_container_width=True)

            if submitted:
                st.session_state.broadcast_type = broadcast_type

                # Update signal
                st.session_state.signal.update(
                    {
                        "symbol": symbol,
                        "direction": direction,
                        "entry": entry,
                        "sl": sl,
                        "timeframe": timeframe,
                        "tp1": tp1,
                        "tp2": tp2,
                        "tp3": tp3,
                        "strategy": strategy,
                    }
                )

                # Validate (includes indicator lock)
                errors = validate_signal(st.session_state.signal)
                st.session_state.errors = errors

                if errors:
                    st.stop()

                # Dispatch based on report type
                if broadcast_type == ReportType.STRICT_SIGNAL.value:
                    message = format_signal_message(st.session_state.signal)
                    if send_to_telegram(message):
                        st.success("✅ Strict Signal broadcast successfully!")
                    else:
                        st.error("❌ Broadcast failed. Check Telegram Gateway status.")
                else:
                    if not Config.GEMINI_API_KEY:
                        st.error("Gemini is inactive. Add gemini.api_key in secrets to use AI report types.")
                        add_log("[ERROR] Broadcast blocked: Gemini inactive for AI report type")
                        st.stop()

                    ai_context = st.text_area(
                        "AI Context (required for AI reports)",
                        placeholder="Provide only the context you want AI to use (no invented data).",
                        height=160,
                    )

                    if not ai_context.strip():
                        st.warning("AI report type selected. Provide context, then click Deploy again.")
                        add_log("[WARN] AI broadcast requested without context")
                        st.stop()

                    depth = st.selectbox(
                        "AI Reasoning Depth",
                        ["QUICK", "DETAILED", "QUANT"],
                        index=0,
                        help="QUANT uses lower temperature and expects more rigorous structure.",
                    )

                    with st.spinner("Synthesizing report..."):
                        if broadcast_type == ReportType.AI_RISK_ANALYSIS.value:
                            report = generate_ai_risk_analysis(st.session_state.signal, ai_context, AnalysisDepth[depth])
                            if report:
                                msg = format_ai_risk_message(report, st.session_state.signal["symbol"])
                                if send_to_telegram(msg):
                                    st.success("✅ AI Risk Analysis transmitted!")
                                else:
                                    st.error("❌ Transmission failed.")
                            else:
                                st.error("❌ AI Risk Analysis generation failed (see terminal logs).")

                        elif broadcast_type == ReportType.MARKET_SUMMARY.value:
                            report = generate_market_summary(ai_context, AnalysisDepth[depth])
                            if report:
                                msg = format_market_summary_message(report)
                                if send_to_telegram(msg):
                                    st.success("✅ Market Summary transmitted!")
                                else:
                                    st.error("❌ Transmission failed.")
                            else:
                                st.error("❌ Market Summary generation failed (see terminal logs).")

        st.markdown("---")
        render_chart()

    with col2:
        st.subheader("🧬 Current Signal Snapshot")
        s = st.session_state.signal
        st.markdown(f"**Symbol:** {s['symbol']}")
        st.markdown(f"**Direction:** {s['direction']}")
        st.markdown(f"**Timeframe:** {s['timeframe']}")
        st.markdown(f"**Entry:** `{s['entry']}`")
        st.markdown(f"**SL:** `{s['sl']}`")
        st.markdown(f"**TP1/TP2/TP3:** `{s['tp1']}` / `{s['tp2']}` / `{s['tp3']}`")
        st.markdown("**Basis:**")
        st.write(s["strategy"])

def render_reports_lab():
    st.header("🧪 NEURAL ANALYTICS LAB")

    if not Config.GEMINI_API_KEY:
        st.warning("Gemini inactive. Add gemini.api_key in secrets to enable AI reports.")
        return

    col1, col2 = st.columns(2)

    with col1:
        depth = st.radio(
            "Reasoning Depth",
            ["QUICK", "DETAILED", "QUANT"],
            format_func=lambda x: f"{x} Analysis",
        )
        context = st.text_area(
            "Analysis Context",
            placeholder="Provide market vectors for synthesis (STRICT: no invented data).",
            height=300,
        )

        if st.button("🔬 INITIALIZE ANALYSIS", type="primary", use_container_width=True):
            if not context.strip():
                st.warning("Please provide analysis context")
                return

            with st.spinner("Synthesizing neural vectors..."):
                report = generate_technical_report(
                    st.session_state.signal["symbol"],
                    st.session_state.signal["timeframe"],
                    context,
                    AnalysisDepth[depth],
                )

                if report:
                    st.session_state.current_report = report
                    st.success("Analysis complete!")
                else:
                    st.error("Analysis failed (see terminal logs).")

    with col2:
        if st.session_state.current_report:
            report = st.session_state.current_report
            st.markdown(f"### {report.get('title', 'Asset Analysis')}")

            outlook_color = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(report.get("outlook", "NEUTRAL"), "⚪")
            st.markdown(f"**Market Outlook:** {outlook_color} {report.get('outlook', 'N/A')}")
            st.info(f"**Summary:** {report.get('summary', '')}")

            with st.expander("📊 Technical Details", expanded=True):
                st.markdown(report.get("technicalDetails", ""))

            if report.get("sources"):
                st.markdown("**Sources:**")
                for source in report["sources"]:
                    title = source.get("title", "Source")
                    uri = source.get("uri", "")
                    if uri:
                        st.markdown(f"- [{title}]({uri})")
                    else:
                        st.markdown(f"- {title}")

            if st.button("📡 TRANSMIT INTELLIGENCE", use_container_width=True):
                message = format_report_message(report)
                if send_to_telegram(message):
                    st.success("Report transmitted!")
                else:
                    st.error("Transmission failed. Check Telegram Gateway status.")
        else:
            st.info("Generate a report to see results here")

def render_script_forge():
    st.header("⚡ PINE FORGE 6.0 (Strict Explicit)")

    st.info(
        "EXPLICIT INDICATOR LOCK applies here too. "
        "If your prompt references indicators (e.g., RSI/EMA/MACD), you must register formulas first."
    )

    prompt = st.text_input(
        "Trading Logic Description",
        placeholder="e.g., Break of structure + retest + risk-defined invalidation (no unapproved indicators)",
    )

    if st.button("🔨 FORGE V6 SOURCE", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Please describe your trading logic")
            return

        with st.spinner("Compiling to Pine Script v6.0..."):
            code = generate_pine_script("StrictExplicitEngine", prompt)
            st.session_state.pine_script = code
            if code.startswith("// ERROR:"):
                st.error("Pine Forge blocked by guardrails (see output).")
            else:
                st.success("Compilation complete!")

    if st.session_state.pine_script:
        st.markdown("#### 📄 Generated Pine Script v6.0")

        col_copy, col_download = st.columns([1, 4])
        with col_copy:
            st.button("📋 Copy", help="Use the code block below to copy manually (Streamlit clipboard varies by browser).")

        with col_download:
            st.download_button(
                label="💾 Download .pine",
                data=st.session_state.pine_script,
                file_name="strategy.pine",
                mime="text/plain",
            )

        st.code(st.session_state.pine_script, language="javascript", line_numbers=True)

def render_gateways():
    st.header("🔐 SYSTEM GATEWAYS")
    st.markdown("### 🔑 API Configuration")

    st.info("API keys are loaded securely from `.streamlit/secrets.toml`")

    with st.expander("View Current Configuration"):
        if Config.GEMINI_API_KEY:
            st.success(f"Gemini API Key: {Config.GEMINI_API_KEY[:8]}...{Config.GEMINI_API_KEY[-4:]}")
            st.caption(f"Gemini Model: {Config.GEMINI_MODEL}")
        else:
            st.error("Gemini API Key: Not configured")

        if Config.TELEGRAM_BOT_TOKEN:
            st.success(f"Telegram Bot Token: {Config.TELEGRAM_BOT_TOKEN[:8]}...{Config.TELEGRAM_BOT_TOKEN[-4:]}")
        else:
            st.error("Telegram Bot Token: Not configured")

        if Config.TELEGRAM_CHAT_ID:
            st.success(f"Telegram Chat ID: {Config.TELEGRAM_CHAT_ID}")
        else:
            st.error("Telegram Chat ID: Not configured")

    st.markdown("---")
    st.markdown(
        """
### Setup Instructions

1. Create a file named `.streamlit/secrets.toml` in your project root
2. Add your API keys:

```toml
[telegram]
bot_token = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
chat_id = "-1001234567890"

[gemini]
api_key = "YOUR_GEMINI_API_KEY"
model = "gemini-pro"
