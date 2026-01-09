# app.py
# =============================================================================
# CommodityLens Pro — Precious Metals & Energy Portfolio Analyzer (Streamlit)
# Features added:
#   ✅ Custom portfolio builder (paste tickers/weights)
#   ✅ Export report to Markdown + PDF
#   ✅ Rebalance suggestions + scenario stress tests (oil crash, reflation, recession, USD spike, etc.)
#
# Notes:
# - This app does NOT fetch live prices by default (keeps it deterministic + fast).
# - Scenario stress tests use configurable "shock assumptions" by category.
# - Optional Gemini AI narrative is supported if you provide a key (st.secrets or UI input).
#
# Security note:
# - For Streamlit Cloud, set secrets in .streamlit/secrets.toml (recommended).
# - If you input the key in the UI, it is stored in session only (not persisted).
#
# DISCLAIMER:
# This tool is for research/education and scenario exploration — not financial advice.
# =============================================================================

from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# PDF (ReportLab is installed in this environment)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# Optional: Gemini (google-generativeai may or may not be installed in your env)
try:
    import google.generativeai as genai  # type: ignore
    _GENAI_AVAILABLE = True
except Exception:
    _GENAI_AVAILABLE = False


# =============================================================================
# UI CONFIG
# =============================================================================

st.set_page_config(
    page_title="CommodityLens Pro — Portfolio Analyzer",
    page_icon="🧭",
    layout="wide",
)

st.markdown(
    """
<style>
.small-muted { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
.kpi { padding: 14px 14px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.02); }
.hr { height: 1px; background: rgba(255,255,255,0.08); margin: 10px 0 16px; }
.badge { display:inline-block; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.03); font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Scenario:
    name: str
    description: str
    shocks_by_category: Dict[str, float]  # percent shock, e.g. -30.0 for -30%


# =============================================================================
# DEFAULT PORTFOLIOS + CATEGORY MAP
# =============================================================================

CATEGORY_MAP: Dict[str, str] = {
    # Physical metals
    "SGLN": "Gold (physical)",
    "SSLN": "Silver (physical)",
    "SPLT": "Platinum (physical)",
    "SPDM": "Palladium (physical)",

    # Energy
    "ESIE": "Energy equities (Europe)",
    "IESU": "Energy equities (US)",
    "SPOG": "Oil & gas E&P",

    # Miners (equity beta)
    "SILG": "Silver miners",
    "COPP": "Copper miners",
    "URJP": "Uranium miners",
    "GJGB": "Junior gold miners",
}

NAME_MAP: Dict[str, str] = {
    "SGLN": "iShares Physical Gold ETC",
    "SSLN": "iShares Physical Silver ETC",
    "SPLT": "iShares Physical Platinum ETC",
    "SPDM": "iShares Physical Palladium ETC",
    "ESIE": "iShares MSCI Europe Energy Sector UCITS ETF",
    "IESU": "iShares S&P 500 Energy Sector UCITS ETF",
    "SPOG": "iShares Oil & Gas Exploration & Production UCITS ETF",
    "SILG": "Global X Silver Miners",
    "COPP": "Sprott Pure Play Copper Miners",
    "URJP": "Sprott Junior Uranium Miners",
    "GJGB": "VanEck Junior Gold Miners",
}

def default_portfolios() -> Dict[str, pd.DataFrame]:
    p1 = pd.DataFrame(
        [
            ("SSLN", 40.0),
            ("ESIE", 15.0),
            ("SPOG", 15.0),
            ("SGLN", 15.0),
            ("IESU", 15.0),
        ],
        columns=["Ticker", "Weight"],
    )
    p2 = pd.DataFrame(
        [
            ("SILG", 34.0),
            ("COPP", 22.0),
            ("URJP", 22.0),
            ("GJGB", 22.0),
        ],
        columns=["Ticker", "Weight"],
    )
    p3 = pd.DataFrame(
        [
            ("SSLN", 35.0),
            ("SPDM", 25.0),
            ("SPLT", 25.0),
            ("SGLN", 15.0),
        ],
        columns=["Ticker", "Weight"],
    )

    for df in (p1, p2, p3):
        df["Name"] = df["Ticker"].map(NAME_MAP).fillna("")
        df["Category"] = df["Ticker"].map(CATEGORY_MAP).fillna("Unknown")
        df["Notes"] = ""

    return {
        "Portfolio 1 — Metals + Energy core": p1,
        "Portfolio 2 — Miners basket": p2,
        "Portfolio 3 — PGM + Gold/Silver": p3,
    }


# =============================================================================
# PARSING + VALIDATION
# =============================================================================

def _coerce_weight(x: str) -> Optional[float]:
    x = x.strip()
    if not x:
        return None
    x = x.replace("%", "").replace(",", ".")
    try:
        return float(x)
    except Exception:
        return None

def parse_portfolio_text(raw: str) -> pd.DataFrame:
    """
    Accepts paste formats like:
      SSLN 40
      SGLN,15
      "SPOG\t15"
    Also supports lines with name: 'SSLN 40 iShares Physical Silver' -> ignores extras.
    """
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    rows: List[Tuple[str, float]] = []

    for ln in lines:
        # split on comma, tab, multiple spaces
        parts = re.split(r"[,\t]+|\s{2,}|\s+", ln.strip())
        parts = [p for p in parts if p.strip()]
        if len(parts) < 2:
            continue
        ticker = parts[0].upper().strip()
        w = _coerce_weight(parts[1])
        if w is None:
            continue
        rows.append((ticker, w))

    df = pd.DataFrame(rows, columns=["Ticker", "Weight"])
    if df.empty:
        return df

    # If weights sum to ~1, treat as fractions and scale to %
    s = df["Weight"].sum()
    if 0.9 <= s <= 1.1:
        df["Weight"] = df["Weight"] * 100.0

    df["Name"] = df["Ticker"].map(NAME_MAP).fillna("")
    df["Category"] = df["Ticker"].map(CATEGORY_MAP).fillna("Unknown")
    df["Notes"] = ""
    return df

def normalize_and_validate(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    if df.empty:
        warnings.append("No holdings parsed yet.")
        return df, warnings

    df = df.copy()
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df = df.dropna(subset=["Ticker", "Weight"]).reset_index(drop=True)

    # consolidate duplicates
    df = (
        df.groupby(["Ticker"], as_index=False)
        .agg({"Weight": "sum"})
        .merge(pd.DataFrame({"Ticker": list(df["Ticker"].unique())}), on="Ticker", how="right")
    )

    # Restore metadata
    df["Name"] = df["Ticker"].map(NAME_MAP).fillna("")
    df["Category"] = df["Ticker"].map(CATEGORY_MAP).fillna("Unknown")
    df["Notes"] = ""

    total = float(df["Weight"].sum())
    if total <= 0:
        warnings.append("Total weight is 0 or invalid.")
        return df, warnings

    # Keep as-is (don’t auto-normalize silently), but flag
    if abs(total - 100.0) > 0.5:
        warnings.append(f"Total weight sums to {total:.2f}%. Consider normalizing to 100%.")

    if (df["Weight"] < 0).any():
        warnings.append("Negative weights detected (shorts). This app assumes long-only by default.")

    return df, warnings

def normalize_to_100(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total = float(df["Weight"].sum())
    if total == 0:
        return df
    df["Weight"] = df["Weight"] / total * 100.0
    return df


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(df: pd.DataFrame) -> Dict[str, float | str]:
    total = float(df["Weight"].sum())
    w = (df["Weight"] / 100.0).fillna(0.0)
    hhi = float((w ** 2).sum())  # 0..1
    eff_n = float(1.0 / hhi) if hhi > 0 else float("nan")
    largest = df.loc[df["Weight"].idxmax(), "Ticker"] if not df.empty else ""
    largest_w = float(df["Weight"].max()) if not df.empty else 0.0
    return {
        "total_weight_pct": total,
        "hhi": hhi,
        "effective_holdings": eff_n,
        "largest_holding": str(largest),
        "largest_weight_pct": largest_w,
    }

def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("Category", as_index=False)["Weight"]
        .sum()
        .sort_values("Weight", ascending=False)
        .reset_index(drop=True)
    )
    return out

def pretty_hhi(hhi: float) -> str:
    if math.isnan(hhi):
        return "—"
    # interpret roughly
    if hhi >= 0.25:
        label = "High concentration"
    elif hhi >= 0.15:
        label = "Moderate concentration"
    else:
        label = "Lower concentration"
    return f"{hhi:.3f} ({label})"


# =============================================================================
# SCENARIOS
# =============================================================================

DEFAULT_SCENARIOS: List[Scenario] = [
    Scenario(
        name="Oil crash",
        description="A fast oil drawdown (demand shock / oversupply). Energy equities hit hardest; precious metals mixed.",
        shocks_by_category={
            "Oil & gas E&P": -35.0,
            "Energy equities (US)": -30.0,
            "Energy equities (Europe)": -28.0,
            "Gold (physical)": +4.0,
            "Silver (physical)": -8.0,
            "Platinum (physical)": -10.0,
            "Palladium (physical)": -12.0,
            "Silver miners": -18.0,
            "Copper miners": -25.0,
            "Uranium miners": -10.0,
            "Junior gold miners": -12.0,
            "Unknown": -10.0,
        },
    ),
    Scenario(
        name="Reflation",
        description="Growth + inflation impulse. Commodities and cyclicals benefit; miners often amplify the move.",
        shocks_by_category={
            "Oil & gas E&P": +25.0,
            "Energy equities (US)": +18.0,
            "Energy equities (Europe)": +16.0,
            "Gold (physical)": +6.0,
            "Silver (physical)": +12.0,
            "Platinum (physical)": +10.0,
            "Palladium (physical)": +8.0,
            "Silver miners": +20.0,
            "Copper miners": +24.0,
            "Uranium miners": +14.0,
            "Junior gold miners": +15.0,
            "Unknown": +8.0,
        },
    ),
    Scenario(
        name="Recession",
        description="Risk-off + slowing demand. Energy/industrial metals suffer; gold may act defensively.",
        shocks_by_category={
            "Oil & gas E&P": -30.0,
            "Energy equities (US)": -22.0,
            "Energy equities (Europe)": -20.0,
            "Gold (physical)": +8.0,
            "Silver (physical)": -10.0,
            "Platinum (physical)": -18.0,
            "Palladium (physical)": -20.0,
            "Silver miners": -22.0,
            "Copper miners": -35.0,
            "Uranium miners": -15.0,
            "Junior gold miners": -10.0,
            "Unknown": -12.0,
        },
    ),
    Scenario(
        name="USD spike",
        description="Broad USD strength / tightening shock. Dollar-priced commodities often face headwinds.",
        shocks_by_category={
            "Oil & gas E&P": -12.0,
            "Energy equities (US)": -8.0,
            "Energy equities (Europe)": -10.0,
            "Gold (physical)": -8.0,
            "Silver (physical)": -12.0,
            "Platinum (physical)": -10.0,
            "Palladium (physical)": -12.0,
            "Silver miners": -15.0,
            "Copper miners": -18.0,
            "Uranium miners": -10.0,
            "Junior gold miners": -14.0,
            "Unknown": -10.0,
        },
    ),
]

def run_scenario(df: pd.DataFrame, shocks_by_category: Dict[str, float]) -> float:
    """
    Returns portfolio impact in % under the shock assumptions.
    """
    if df.empty:
        return 0.0
    tmp = df.copy()
    tmp["Shock"] = tmp["Category"].map(shocks_by_category).fillna(shocks_by_category.get("Unknown", 0.0))
    # Impact = sum(w% * shock%) / 100
    impact = float((tmp["Weight"] * tmp["Shock"]).sum() / 100.0)
    return impact

def scenario_table(df: pd.DataFrame, scenarios: List[Scenario]) -> pd.DataFrame:
    rows = []
    for sc in scenarios:
        impact = run_scenario(df, sc.shocks_by_category)
        rows.append({"Scenario": sc.name, "Impact (%)": impact, "Description": sc.description})
    out = pd.DataFrame(rows).sort_values("Impact (%)").reset_index(drop=True)
    return out


# =============================================================================
# REBALANCING
# =============================================================================

def propose_rebalance_to_category_targets(
    df: pd.DataFrame,
    category_targets: Dict[str, float],
    within_category: str = "proportional",  # or "equal"
) -> pd.DataFrame:
    """
    Returns a table with current weights, target weights, and deltas per holding.
    """
    if df.empty:
        return pd.DataFrame()

    df = normalize_to_100(df.copy())
    cats = df["Category"].unique().tolist()

    # ensure all existing categories have a target (default to current)
    current_cat = category_breakdown(df).set_index("Category")["Weight"].to_dict()
    targets = dict(current_cat)
    for c in cats:
        if c in category_targets:
            targets[c] = float(category_targets[c])

    # normalize targets to 100
    t_sum = sum(targets.values())
    if t_sum <= 0:
        targets = current_cat
        t_sum = sum(targets.values())
    targets = {k: v / t_sum * 100.0 for k, v in targets.items()}

    out = df.copy()
    out["Current %"] = out["Weight"]

    # compute per-holding target inside each category
    targets_by_holding = []
    for cat, g in out.groupby("Category", as_index=False):
        cat_name = cat
        cat_target = targets.get(cat_name, float(g["Current %"].sum()))

        if within_category == "equal":
            # equal weight within category
            each = cat_target / len(g)
            for _, r in g.iterrows():
                targets_by_holding.append((r["Ticker"], each))
        else:
            # proportional to current weights in category
            cat_current_sum = float(g["Current %"].sum())
            if cat_current_sum <= 0:
                each = cat_target / len(g)
                for _, r in g.iterrows():
                    targets_by_holding.append((r["Ticker"], each))
            else:
                for _, r in g.iterrows():
                    prop = float(r["Current %"]) / cat_current_sum
                    targets_by_holding.append((r["Ticker"], cat_target * prop))

    tgt = pd.DataFrame(targets_by_holding, columns=["Ticker", "Target %"])
    out = out.merge(tgt, on="Ticker", how="left")
    out["Delta %"] = out["Target %"] - out["Current %"]
    out["Action"] = out["Delta %"].apply(lambda x: "BUY" if x > 0.05 else ("SELL" if x < -0.05 else "HOLD"))

    out = out[["Ticker", "Name", "Category", "Current %", "Target %", "Delta %", "Action"]].sort_values(
        "Delta %", ascending=False
    )
    return out.reset_index(drop=True)

def propose_simple_equal_weight(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = normalize_to_100(df.copy())
    n = len(df)
    target = 100.0 / n
    out = df.copy()
    out["Current %"] = out["Weight"]
    out["Target %"] = target
    out["Delta %"] = out["Target %"] - out["Current %"]
    out["Action"] = out["Delta %"].apply(lambda x: "BUY" if x > 0.05 else ("SELL" if x < -0.05 else "HOLD"))
    out = out[["Ticker", "Name", "Category", "Current %", "Target %", "Delta %", "Action"]].sort_values("Delta %", ascending=False)
    return out.reset_index(drop=True)


# =============================================================================
# REPORT EXPORT
# =============================================================================

def build_markdown_report(
    title: str,
    df: pd.DataFrame,
    metrics: Dict[str, float | str],
    cat_df: pd.DataFrame,
    scen_df: pd.DataFrame,
    rebalance_df: Optional[pd.DataFrame],
    assumptions_note: str,
    ai_text: Optional[str] = None,
) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"*Generated:* {now}")
    lines.append("")
    lines.append("> **Disclaimer:** Educational research only — not financial advice.")
    lines.append("")
    lines.append("## Holdings")
    lines.append("")
    if df.empty:
        lines.append("_No holdings._")
    else:
        tmp = df.copy()
        tmp = normalize_to_100(tmp)
        tmp = tmp[["Ticker", "Name", "Category", "Weight"]].rename(columns={"Weight": "Weight %"})
        lines.append(tmp.to_markdown(index=False))
    lines.append("")
    lines.append("## Quick Metrics")
    lines.append("")
    lines.append(f"- Total weight: **{float(metrics['total_weight_pct']):.2f}%**")
    lines.append(f"- Largest holding: **{metrics['largest_holding']} ({float(metrics['largest_weight_pct']):.2f}%)**")
    lines.append(f"- HHI concentration: **{float(metrics['hhi']):.3f}** (higher = more concentrated)")
    lines.append(f"- Effective holdings (1/HHI): **{float(metrics['effective_holdings']):.2f}**")
    lines.append("")
    lines.append("## Category Breakdown")
    lines.append("")
    if cat_df.empty:
        lines.append("_No categories._")
    else:
        lines.append(cat_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Scenario Stress Tests (assumption-based)")
    lines.append("")
    lines.append(scen_df[["Scenario", "Impact (%)", "Description"]].to_markdown(index=False))
    lines.append("")
    lines.append("### Assumption note")
    lines.append("")
    lines.append(assumptions_note.strip())
    lines.append("")
    lines.append("## Rebalance Suggestions")
    lines.append("")
    if rebalance_df is None or rebalance_df.empty:
        lines.append("_No rebalance plan generated._")
    else:
        lines.append(rebalance_df.to_markdown(index=False))
    lines.append("")
    if ai_text:
        lines.append("## AI Narrative (optional)")
        lines.append("")
        lines.append(ai_text.strip())
        lines.append("")
    return "\n".join(lines)

def markdown_to_pdf_bytes(md: str, title: str) -> bytes:
    """
    Simple Markdown-ish to PDF:
    - Converts lines starting with #/##/### into headings
    - Bullet lines rendered as paragraphs
    - Tables are not parsed as tables (kept as monospace-ish paragraphs)
    """
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 10))

    for ln in md.splitlines():
        ln = ln.rstrip()
        if not ln:
            story.append(Spacer(1, 8))
            continue

        if ln.startswith("# "):
            story.append(Paragraph(f"<b>{ln[2:]}</b>", styles["Heading1"]))
        elif ln.startswith("## "):
            story.append(Paragraph(f"<b>{ln[3:]}</b>", styles["Heading2"]))
        elif ln.startswith("### "):
            story.append(Paragraph(f"<b>{ln[4:]}</b>", styles["Heading3"]))
        elif ln.startswith("> "):
            story.append(Paragraph(f"<i>{ln[2:]}</i>", styles["BodyText"]))
        else:
            # Escape minimal HTML specials
            safe = (ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            # keep code/table formatting readable
            if safe.startswith("```") or safe.startswith("|") or safe.startswith("-|") or safe.startswith("|-"):
                story.append(Paragraph(f"<font face='Courier'>{safe}</font>", styles["BodyText"]))
            else:
                story.append(Paragraph(safe, styles["BodyText"]))

    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, title=title)
    doc.build(story)
    return buff.getvalue()


# =============================================================================
# OPTIONAL GEMINI AI
# =============================================================================

def gemini_generate_text(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 1200,
) -> str:
    if not _GENAI_AVAILABLE:
        raise RuntimeError(
            "google-generativeai is not installed in this environment. "
            "Install it with: pip install google-generativeai"
        )
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model_name=model)
    resp = m.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        },
    )
    text = getattr(resp, "text", None)
    if not text:
        # Some responses store candidates differently; keep it safe
        return str(resp)
    return text


# =============================================================================
# PLOTLY CHARTS
# =============================================================================

def pie_holdings(df: pd.DataFrame, title: str) -> go.Figure:
    d = normalize_to_100(df.copy()) if not df.empty else df.copy()
    labels = d["Ticker"].tolist() if not d.empty else []
    values = d["Weight"].tolist() if not d.empty else []
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.35)])
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10), height=360)
    return fig

def bar_categories(cat_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if not cat_df.empty:
        fig.add_trace(go.Bar(x=cat_df["Category"], y=cat_df["Weight"]))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10), height=360)
    fig.update_xaxes(tickangle=25)
    return fig

def bar_scenarios(scen_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if not scen_df.empty:
        fig.add_trace(go.Bar(x=scen_df["Scenario"], y=scen_df["Impact (%)"]))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10), height=320)
    return fig


# =============================================================================
# APP STATE HELPERS
# =============================================================================

def get_secret_api_key() -> str:
    # Common patterns:
    # st.secrets["GEMINI_API_KEY"] or st.secrets["GOOGLE_API_KEY"]
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        if k in st.secrets and str(st.secrets[k]).strip():
            return str(st.secrets[k]).strip()
    return ""

def get_portfolio_from_state() -> pd.DataFrame:
    return st.session_state.get("portfolio_df", pd.DataFrame()).copy()

def set_portfolio_in_state(df: pd.DataFrame) -> None:
    st.session_state["portfolio_df"] = df.copy()


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🧭 CommodityLens Pro")

with st.sidebar.expander("About", expanded=False):
    st.markdown(
        """
- Build or select a metals/energy portfolio
- Analyze concentration + category exposure
- Run assumption-based scenario shocks
- Generate rebalance suggestions
- Export report to Markdown/PDF
- Optional Gemini AI narrative (requires key)
"""
    )

PORTS = default_portfolios()
mode = st.sidebar.radio(
    "Portfolio source",
    ["Built-in", "Custom"],
    index=0,
)

if mode == "Built-in":
    sel = st.sidebar.selectbox("Choose a portfolio", list(PORTS.keys()), index=0)
    df0 = PORTS[sel].copy()
    set_portfolio_in_state(df0)
    portfolio_title = sel
else:
    portfolio_title = st.sidebar.text_input("Custom portfolio name", value="Custom Portfolio")
    st.sidebar.markdown("Paste holdings (one per line): `TICKER weight` or `TICKER,weight`")
    example = "SSLN 40\nSGLN 15\nSPOG 15\nESIE 15\nIESU 15"
    raw = st.sidebar.text_area("Holdings paste box", value=st.session_state.get("custom_raw", example), height=180)
    st.session_state["custom_raw"] = raw

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Parse holdings", use_container_width=True):
        parsed = parse_portfolio_text(raw)
        parsed, warns = normalize_and_validate(parsed)
        set_portfolio_in_state(parsed)
        if warns:
            st.sidebar.warning("\n".join(warns))

    if col_b.button("Normalize to 100%", use_container_width=True):
        df_tmp = get_portfolio_from_state()
        if not df_tmp.empty:
            set_portfolio_in_state(normalize_to_100(df_tmp))

st.sidebar.markdown('<div class="hr"></div>', unsafe_allow_html=True)

with st.sidebar.expander("Scenario settings", expanded=False):
    st.markdown("Edit scenario shock assumptions by category (in %).")
    # store scenarios in session so user edits persist
    if "scenarios" not in st.session_state:
        st.session_state["scenarios"] = DEFAULT_SCENARIOS

    scenario_names = [s.name for s in st.session_state["scenarios"]]
    sc_name = st.selectbox("Select scenario", scenario_names, index=0)
    sc = next(s for s in st.session_state["scenarios"] if s.name == sc_name)

    st.caption(sc.description)
    # Shocks editor
    shock_df = pd.DataFrame(
        [{"Category": k, "Shock (%)": float(v)} for k, v in sc.shocks_by_category.items()]
    ).sort_values("Category").reset_index(drop=True)

    edited = st.data_editor(
        shock_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Shock (%)": st.column_config.NumberColumn(format="%.2f"),
        },
        key=f"shock_editor_{sc_name}",
    )

    if st.button("Save scenario edits"):
        shocks_new = {}
        for _, r in edited.iterrows():
            cat = str(r["Category"]).strip()
            if not cat:
                continue
            shocks_new[cat] = float(r["Shock (%)"])
        # update scenario object
        for i, s in enumerate(st.session_state["scenarios"]):
            if s.name == sc_name:
                st.session_state["scenarios"][i] = Scenario(name=s.name, description=s.description, shocks_by_category=shocks_new)
                break
        st.success("Scenario updated.")

with st.sidebar.expander("AI (optional)", expanded=False):
    st.caption("Add a Gemini API key via secrets or paste it here for this session.")
    secret_key = get_secret_api_key()
    if secret_key:
        st.markdown("✅ Using API key from **st.secrets**.")
    api_key_ui = st.text_input("Gemini API key (session only)", type="password", value=st.session_state.get("api_key_ui", ""))
    st.session_state["api_key_ui"] = api_key_ui

    model = st.text_input("Model name", value=st.session_state.get("gem_model", "gemini-1.5-pro"))
    st.session_state["gem_model"] = model
    temperature = st.slider("Temperature", 0.0, 1.0, float(st.session_state.get("gem_temp", 0.2)), 0.05)
    st.session_state["gem_temp"] = temperature

with st.sidebar.expander("Rebalance controls", expanded=False):
    reb_mode = st.selectbox("Rebalance method", ["Category targets", "Equal weight"], index=0)
    st.session_state["reb_mode"] = reb_mode
    within = st.selectbox("Within category distribution", ["proportional", "equal"], index=0)
    st.session_state["within_category"] = within


# =============================================================================
# MAIN
# =============================================================================

st.title("🧭 CommodityLens Pro — Portfolio Analyzer")
st.markdown(
    '<span class="small-muted">Custom portfolio builder • Markdown/PDF export • Rebalance plans • Scenario stress tests</span>',
    unsafe_allow_html=True,
)

df = get_portfolio_from_state()
df, warns = normalize_and_validate(df)

if warns:
    st.warning("\n".join(warns))

if df.empty:
    st.info("Add holdings (Built-in or Custom) to begin.")
    st.stop()

# Allow editing Category/Name/Notes
st.subheader("Holdings")
st.caption("You can edit names/categories/notes. Weight edits are supported too.")
df_edit = df.copy()
df_edit = st.data_editor(
    df_edit,
    use_container_width=True,
    hide_index=True,
    num_rows="dynamic",
    column_config={
        "Weight": st.column_config.NumberColumn("Weight (%)", format="%.2f"),
    },
    key="holdings_editor",
)

# Persist edits
set_portfolio_in_state(df_edit)
df = df_edit.copy()

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    if st.button("Normalize weights to 100%", use_container_width=True):
        df = normalize_to_100(df)
        set_portfolio_in_state(df)
        st.rerun()
with col2:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download holdings CSV", csv_bytes, file_name="holdings.csv", mime="text/csv", use_container_width=True)
with col3:
    st.write("")
with col4:
    st.write("")

metrics = compute_metrics(df)
cat_df = category_breakdown(df)
scen_df = scenario_table(df, st.session_state.get("scenarios", DEFAULT_SCENARIOS))

# KPI row
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric("Total weight", f"{metrics['total_weight_pct']:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric("Largest holding", f"{metrics['largest_holding']}", f"{metrics['largest_weight_pct']:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric("HHI (concentration)", pretty_hhi(float(metrics["hhi"])))
    st.markdown("</div>", unsafe_allow_html=True)
with k4:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric("Effective holdings", f"{float(metrics['effective_holdings']):.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

# Charts
cA, cB = st.columns(2)
with cA:
    st.plotly_chart(pie_holdings(df, "Holdings composition"), use_container_width=True)
with cB:
    st.plotly_chart(bar_categories(cat_df, "Category weights"), use_container_width=True)

st.subheader("Scenario stress tests (assumption-based)")
st.caption("These are not forecasts — they are *what-if* impacts based on your shock assumptions per category.")
s1, s2 = st.columns([2, 1])
with s1:
    st.dataframe(scen_df, use_container_width=True, hide_index=True)
with s2:
    st.plotly_chart(bar_scenarios(scen_df, "Scenario impacts"), use_container_width=True)

# =============================================================================
# REBALANCE SECTION
# =============================================================================

st.subheader("Rebalance suggestions")

rebalance_df: Optional[pd.DataFrame] = None
reb_mode = st.session_state.get("reb_mode", "Category targets")
within_category = st.session_state.get("within_category", "proportional")

if reb_mode == "Equal weight":
    rebalance_df = propose_simple_equal_weight(df)
    st.caption("Equal-weight target across holdings (simple baseline).")
else:
    st.caption("Set category targets. The app will distribute target weights within each category.")
    current_cat = cat_df.set_index("Category")["Weight"].to_dict()

    # Build target inputs
    target_inputs: Dict[str, float] = {}
    tgt_cols = st.columns(2)
    cats_sorted = list(current_cat.keys())

    # Render sliders (split into two columns)
    half = (len(cats_sorted) + 1) // 2
    left = cats_sorted[:half]
    right = cats_sorted[half:]

    with tgt_cols[0]:
        for cat in left:
            target_inputs[cat] = st.slider(
                f"{cat} target (%)",
                0.0, 100.0,
                float(st.session_state.get(f"tgt_{cat}", current_cat[cat])),
                0.5,
                key=f"tgt_{cat}",
            )
    with tgt_cols[1]:
        for cat in right:
            target_inputs[cat] = st.slider(
                f"{cat} target (%)",
                0.0, 100.0,
                float(st.session_state.get(f"tgt_{cat}", current_cat[cat])),
                0.5,
                key=f"tgt_{cat}",
            )

    rebalance_df = propose_rebalance_to_category_targets(df, target_inputs, within_category=within_category)

if rebalance_df is None or rebalance_df.empty:
    st.info("No rebalance plan generated.")
else:
    st.dataframe(rebalance_df, use_container_width=True, hide_index=True)
    buys = rebalance_df[rebalance_df["Action"] == "BUY"]
    sells = rebalance_df[rebalance_df["Action"] == "SELL"]
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Top buys (by % delta)**")
        st.dataframe(buys.head(10), use_container_width=True, hide_index=True)
    with b2:
        st.markdown("**Top sells (by % delta)**")
        st.dataframe(sells.head(10), use_container_width=True, hide_index=True)

# =============================================================================
# REPORT GENERATION + EXPORT
# =============================================================================

st.subheader("Report export (Markdown / PDF)")

assumptions_note = """
**Scenario assumptions** are user-editable and category-based.  
These shocks are *not predictions*. They’re intended to help you reason about:
- Concentration risk (what dominates outcomes)
- Exposure directionality (which scenarios hurt/help most)
- Rebalance trade-offs (what you’d need to buy/sell to change scenario sensitivity)
"""

with st.expander("Optional: Generate AI narrative (Gemini)", expanded=False):
    st.markdown(
        """
If you provide a Gemini API key, the app can generate a narrative that explains:
- key exposures and concentration
- scenario vulnerabilities and hedges
- rebalance rationale and trade-offs

If `google-generativeai` is not installed in your environment, you’ll see an error.
"""
    )
    run_ai = st.checkbox("Generate AI narrative", value=False)
    ai_text: Optional[str] = None

    if run_ai:
        api_key = (st.session_state.get("api_key_ui", "") or get_secret_api_key()).strip()
        if not api_key:
            st.error("No API key found. Add GEMINI_API_KEY to st.secrets or paste it in the sidebar.")
        else:
            prompt = f"""
You are an expert commodities + metals portfolio analyst.
Write a concise but high-value narrative report.

Portfolio title: {portfolio_title}

Holdings (ticker, weight%, category):
{normalize_to_100(df)[["Ticker","Weight","Category"]].to_string(index=False)}

Metrics:
- Total weight: {metrics['total_weight_pct']:.2f}%
- Largest holding: {metrics['largest_holding']} ({metrics['largest_weight_pct']:.2f}%)
- HHI concentration: {float(metrics['hhi']):.3f}
- Effective holdings: {float(metrics['effective_holdings']):.2f}

Scenario impacts (assumption-based):
{scen_df[["Scenario","Impact (%)"]].to_string(index=False)}

Rebalance plan (if present):
{("" if rebalance_df is None else rebalance_df.head(20).to_string(index=False))}

Write sections:
1) What this portfolio is really exposed to (drivers + hidden correlations)
2) Key risks (concentration, equity beta vs metals, regime dependence)
3) Scenario notes (biggest downside, biggest upside; what dominates the outcome)
4) Rebalance critique (what changes, what you sacrifice, what improves)
5) Practical operating plan (when to add, when to trim, guardrails)

Keep it practical. Avoid generic disclaimers.
"""
            with st.spinner("Generating AI narrative..."):
                try:
                    ai_text = gemini_generate_text(
                        api_key=api_key,
                        model=st.session_state.get("gem_model", "gemini-1.5-pro"),
                        prompt=prompt,
                        temperature=float(st.session_state.get("gem_temp", 0.2)),
                        max_output_tokens=1400,
                    )
                    st.success("AI narrative generated.")
                    st.text_area("AI Narrative (editable)", value=ai_text, height=260)
                except Exception as e:
                    st.error(f"AI generation failed: {e}")

# Build the report (use AI text if generated above)
# We read ai_text from the expander scope if it exists; else None
ai_text_final = None
if "holdings_editor" in st.session_state:
    # If user generated ai_text, they'd see it; we keep a stable approach:
    # allow user to paste/edit narrative here for inclusion.
    ai_text_final = st.text_area("Optional narrative to include in report", value="", height=140)

md = build_markdown_report(
    title=portfolio_title,
    df=df,
    metrics=metrics,
    cat_df=cat_df,
    scen_df=scen_df,
    rebalance_df=rebalance_df,
    assumptions_note=assumptions_note,
    ai_text=ai_text_final if ai_text_final.strip() else None,
)

st.markdown("**Report preview (Markdown):**")
st.code(md, language="markdown")

# Export buttons
md_bytes = md.encode("utf-8")
st.download_button(
    "Download report.md",
    md_bytes,
    file_name=f"{re.sub(r'[^a-zA-Z0-9_-]+','_', portfolio_title).strip('_')}.md",
    mime="text/markdown",
    use_container_width=True,
)

try:
    pdf_bytes = markdown_to_pdf_bytes(md, title=portfolio_title)
    st.download_button(
        "Download report.pdf",
        pdf_bytes,
        file_name=f"{re.sub(r'[^a-zA-Z0-9_-]+','_', portfolio_title).strip('_')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
except Exception as e:
    st.error(f"PDF export failed: {e}")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown(
    """
<span class="small-muted">
Tip: For Streamlit secrets, create <code>.streamlit/secrets.toml</code>:
<br/><code>GEMINI_API_KEY = "YOUR_KEY"</code>
</span>
""",
    unsafe_allow_html=True,
)
