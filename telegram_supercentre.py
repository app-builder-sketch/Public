# telegram_supercentre/app.py
from __future__ import annotations

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd

# Make local src importable when running from this folder
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.supercentre.config import load_config
from src.supercentre.storage import (
    init_db,
    list_signals,
    get_signal_by_id,
    list_deliveries,
    log_delivery,
    mark_signal_status,
)
from src.supercentre.telegram import TelegramClient, format_signal_html
from src.supercentre.tv_alerts import parse_tv_alert


st.set_page_config(page_title="Signals Super Centre", page_icon="🛰️", layout="wide")

GENERAL, TELEGRAM = load_config()
init_db(GENERAL.db_path)

st.title("🛰️ Telegram Signals Super Centre")
st.caption("Ingest TradingView alerts → store → review → broadcast to Telegram (with delivery logs).")

with st.sidebar:
    st.subheader("⚙️ Settings")
    st.write("Config is loaded from environment variables or Streamlit secrets.")
    st.code(
        "\n".join(
            [
                "DB_PATH (default: ./signals.db)",
                "TELEGRAM_BOT_TOKEN",
                "TELEGRAM_CHAT_ID (single) OR TELEGRAM_CHAT_IDS (comma-separated)",
                "ALLOW_HTML (default: true)",
                "RATE_LIMIT_SECONDS (default: 2)",
                "AUTO_BROADCAST (default: false)",
            ]
        ),
        language="text",
    )

    st.divider()
    st.subheader("🔌 Webhook Receiver")
    st.write("Run the FastAPI receiver to accept TradingView webhooks:")
    st.code("python -m services.webhook_receiver", language="bash")

    st.divider()
    st.subheader("🧪 Quick Parse Test")
    sample = st.text_area(
        "Paste a TradingView alert JSON (or your alert text) to preview parsing:",
        value='{"symbol":"BTCUSDT","timeframe":"15","side":"BUY","price":42500.5,"strategy":"Nexus","confidence":0.74,"reason":"Confluence aligned"}',
        height=160,
    )
    if st.button("Parse sample"):
        try:
            parsed = parse_tv_alert(sample)
            st.success("Parsed OK")
            st.json(parsed)
        except Exception as e:
            st.error(f"Parse failed: {e}")


def _fmt_ts(ts: Optional[float]) -> str:
    if not ts:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


tab_inbox, tab_signal, tab_deliveries, tab_broadcast = st.tabs(
    ["📥 Inbox", "📄 Signal Detail", "📦 Deliveries", "📣 Broadcast"]
)

with tab_inbox:
    st.subheader("📥 Signals Inbox")
    colA, colB, colC = st.columns([1, 1, 2])

    with colA:
        limit = st.number_input("Rows", min_value=10, max_value=5000, value=200, step=10)
    with colB:
        status = st.selectbox("Status", ["all", "new", "broadcasted", "failed"], index=0)
    with colC:
        q = st.text_input("Search (symbol/strategy/reason/id)", value="").strip()

    signals = list_signals(GENERAL.db_path, limit=int(limit), status=status, query=q)
    if not signals:
        st.info("No signals found yet. Send a webhook to populate the inbox.")
    else:
        df = pd.DataFrame(signals)
        # friendlier ordering
        cols = [c for c in ["id", "ts", "symbol", "timeframe", "side", "price", "strategy", "confidence", "status", "reason"] if c in df.columns]
        df = df[cols]
        if "ts" in df.columns:
            df["ts"] = df["ts"].apply(_fmt_ts)

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.caption("Tip: click a signal ID below to load it in the Signal Detail tab.")
        chosen = st.text_input("Load by Signal ID:", value="")
        if chosen:
            st.session_state["selected_signal_id"] = chosen
            st.success("Saved selection. Open **Signal Detail** tab.")


with tab_signal:
    st.subheader("📄 Signal Detail")

    sid = st.session_state.get("selected_signal_id") or st.text_input("Signal ID:", value="").strip()
    if not sid:
        st.info("Select a signal from Inbox, or paste a Signal ID here.")
    else:
        sig = get_signal_by_id(GENERAL.db_path, sid)
        if not sig:
            st.error("Signal not found.")
        else:
            left, right = st.columns([2, 1])
            with left:
                st.json(sig)
            with right:
                st.metric("Symbol", sig.get("symbol", "-"))
                st.metric("Side", sig.get("side", "-"))
                st.metric("Timeframe", sig.get("timeframe", "-"))
                st.metric("Price", sig.get("price", "-"))
                st.metric("Confidence", sig.get("confidence", "-"))
                st.metric("Status", sig.get("status", "-"))

            st.divider()
            st.subheader("🧾 Rendered Telegram Message")
            msg_html = format_signal_html(sig)
            if TELEGRAM.allow_html:
                st.markdown(msg_html, unsafe_allow_html=True)
            else:
                st.code(msg_html)

            st.divider()
            st.subheader("✅ Actions")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Mark NEW"):
                    mark_signal_status(GENERAL.db_path, sid, "new")
                    st.success("Updated.")
                    st.rerun()
            with col2:
                if st.button("Mark BROADCASTED"):
                    mark_signal_status(GENERAL.db_path, sid, "broadcasted")
                    st.success("Updated.")
                    st.rerun()
            with col3:
                if st.button("Mark FAILED"):
                    mark_signal_status(GENERAL.db_path, sid, "failed")
                    st.success("Updated.")
                    st.rerun()


with tab_deliveries:
    st.subheader("📦 Delivery Log")
    col1, col2 = st.columns([1, 2])
    with col1:
        dlimit = st.number_input("Rows", min_value=10, max_value=5000, value=200, step=10, key="dlimit")
    with col2:
        dq = st.text_input("Search (signal_id/chat_id/error)", value="", key="dq").strip()

    deliveries = list_deliveries(GENERAL.db_path, limit=int(dlimit), query=dq)
    if not deliveries:
        st.info("No deliveries logged yet.")
    else:
        ddf = pd.DataFrame(deliveries)
        cols = [c for c in ["id", "ts", "signal_id", "chat_id", "ok", "status_code", "error"] if c in ddf.columns]
        ddf = ddf[cols]
        if "ts" in ddf.columns:
            ddf["ts"] = ddf["ts"].apply(_fmt_ts)
        st.dataframe(ddf, use_container_width=True, hide_index=True)


with tab_broadcast:
    st.subheader("📣 Broadcast Controls")

    st.write("This tab sends messages to Telegram using your configured bot + chat IDs.")
    if not TELEGRAM.bot_token:
        st.warning("Missing TELEGRAM_BOT_TOKEN. Add it to Streamlit secrets or environment variables to broadcast.")
    if not TELEGRAM.chat_ids:
        st.warning("Missing TELEGRAM_CHAT_ID or TELEGRAM_CHAT_IDS. Add at least one chat id to broadcast.")

    st.divider()
    sid = st.session_state.get("selected_signal_id") or st.text_input("Signal ID to broadcast:", value="").strip()

    sig = get_signal_by_id(GENERAL.db_path, sid) if sid else None
    if sig:
        st.success("Signal loaded.")
        st.json(sig)
    else:
        st.info("Load a signal by ID to broadcast it.")

    st.divider()
    manual_text = st.text_area(
        "Or send a manual message (optional):",
        value="",
        height=140,
        help="If provided, this will be sent instead of the formatted signal message.",
    )

    auto_broadcast = st.toggle("Auto-broadcast on NEW signals (receiver-side)", value=GENERAL.auto_broadcast)
    st.caption(
        "Note: Auto-broadcast is enforced by the receiver service. Toggle here is informational unless you restart receiver with updated env/secrets."
    )

    if st.button("Send to Telegram", type="primary", disabled=not (TELEGRAM.bot_token and TELEGRAM.chat_ids)):
        if not manual_text and not sig:
            st.error("Provide a manual message or a valid Signal ID.")
        else:
            client = TelegramClient(
                bot_token=TELEGRAM.bot_token,
                chat_ids=TELEGRAM.chat_ids,
                allow_html=TELEGRAM.allow_html,
                rate_limit_seconds=TELEGRAM.rate_limit_seconds,
                timeout_seconds=TELEGRAM.timeout_seconds,
            )
            payload = manual_text if manual_text else format_signal_html(sig)

            results = []
            for cid in TELEGRAM.chat_ids:
                try:
                    ok, status_code, err = client.send_message(cid, payload)
                    results.append({"chat_id": cid, "ok": ok, "status_code": status_code, "error": err or ""})
                    log_delivery(GENERAL.db_path, signal_id=sid or "-", chat_id=str(cid), ok=ok, status_code=status_code, error=err)
                    if ok and sig:
                        mark_signal_status(GENERAL.db_path, sid, "broadcasted")
                except Exception as e:
                    results.append({"chat_id": cid, "ok": False, "status_code": 0, "error": str(e)})
                    log_delivery(GENERAL.db_path, signal_id=sid or "-", chat_id=str(cid), ok=False, status_code=0, error=str(e))
                    if sig:
                        mark_signal_status(GENERAL.db_path, sid, "failed")

            st.subheader("Results")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            st.success("Done.")
            st.rerun()


st.divider()
st.caption("Signals Super Centre • built for reliable alert ingestion + Telegram distribution.")
