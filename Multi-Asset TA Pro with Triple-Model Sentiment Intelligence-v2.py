# =========================
# app.py (PART 1/6)
# =========================
"""
Multi-Asset TA Pro with Triple-Model Sentiment Intelligence
===============================================================

A comprehensive financial analysis application combining:
- Advanced Technical Analysis indicators
- Triple-Model Sentiment Analysis (Transformer + VADER + TextBlob)
- Interactive Plotly visualizations
- Professional-grade broadcasting system

Author: Advanced Financial AI System
Version: 1.0.0
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta, timezone
import requests
import json
import re
from typing import Dict, List, Tuple, Optional, Any
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# IMPORT STATUS (NO Streamlit calls before set_page_config)
# ==========================================
IMPORT_WARNINGS: List[str] = []

# ==========================================
# SENTIMENT ANALYSIS IMPORTS
# ==========================================
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    IMPORT_WARNINGS.append("⚠️ Transformers not installed. Install with: pip install transformers torch")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    IMPORT_WARNINGS.append("⚠️ VADER not installed. Install with: pip install vaderSentiment")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    IMPORT_WARNINGS.append("⚠️ TextBlob not installed. Install with: pip install textblob")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

# ==========================================
# TECHNICAL ANALYSIS IMPORTS
# ==========================================
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    IMPORT_WARNINGS.append("⚠️ TA-Lib not installed. Using pure Python fallback implementations.")

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Multi-Asset TA Pro with Sentiment Intelligence",
    page_icon="📊",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Multi-Asset TA Pro - Advanced Financial Analysis with Triple-Model Sentiment Intelligence"
    }
)



# ==========================================
# SECRETS + TELEGRAM HELPERS
# ==========================================
def get_secret_str(*keys: str, default: str = "") -> str:
    """Return the first matching Streamlit secret value as a string.

    Accepts multiple possible keys (top-level or nested via dot-notation).
    Examples:
        get_secret_str("TELEGRAM_TOKEN", "telegram_token")
        get_secret_str("telegram.bot_token", "telegram_token")
    """
    try:
        secrets = st.secrets
    except Exception:
        secrets = {}

    def _get_by_path(obj: Any, path: str) -> Optional[Any]:
        cur = obj
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    for k in keys:
        if not k:
            continue
        val = _get_by_path(secrets, k)
        if val is None and isinstance(secrets, dict):
            val = secrets.get(k)
        if val is None:
            continue
        try:
            s = str(val).strip()
        except Exception:
            continue
        if s:
            return s
    return default


def send_telegram_message(bot_token: str, chat_id: str, text: str, timeout_s: int = 12) -> Dict[str, Any]:
    """Send a Telegram message with reliability: timeouts, status checks, and safe logging."""
    t0 = time.time()
    bot_token = (bot_token or "").strip()
    chat_id = (chat_id or "").strip()
    text = text if text is not None else ""

    if not bot_token:
        return {
            "ok": False,
            "status_code": None,
            "response_snippet": "",
            "error": "Missing Telegram bot token",
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
    if not chat_id:
        return {
            "ok": False,
            "status_code": None,
            "response_snippet": "",
            "error": "Missing Telegram chat id",
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        elapsed_ms = int((time.time() - t0) * 1000)
        snippet = (resp.text or "")[:1000]

        ok = False
        err = ""
        try:
            data = resp.json()
            ok = bool(data.get("ok")) and resp.status_code == 200
            if not ok:
                err = str(data.get("description") or "")[:300]
        except Exception:
            ok = resp.status_code == 200
            if not ok:
                err = "Non-JSON Telegram response"

        if resp.status_code != 200 and not err:
            err = f"HTTP {resp.status_code}"

        return {
            "ok": ok,
            "status_code": resp.status_code,
            "response_snippet": snippet,
            "error": err,
            "elapsed_ms": elapsed_ms,
        }

    except requests.Timeout:
        return {
            "ok": False,
            "status_code": None,
            "response_snippet": "",
            "error": f"Timeout after {timeout_s}s",
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
    except requests.RequestException as e:
        return {
            "ok": False,
            "status_code": None,
            "response_snippet": "",
            "error": str(e)[:300],
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
# ==========================================
# STYLING
# ==========================================
def inject_custom_css():
    """Enhanced professional styling for the application"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
            color: #e0e0e0;
            font-family: 'Inter', sans-serif;
        }
        
        /* Sentiment Gauge Styling */
        .sentiment-gauge {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(0, 240, 255, 0.2);
            backdrop-filter: blur(10px);
        }
        
        /* Metric Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 16px;
            border-left: 4px solid #00F0FF;
            backdrop-filter: blur(20px);
            margin-bottom: 12px;
        }
        
        /* Progress indicators */
        .progress-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 4px;
            margin: 8px 0;
        }
        
        .progress-bar {
            height: 8px;
            border-radius: 6px;
            background: linear-gradient(90deg, #00F0FF, #00FF88);
            transition: width 0.3s ease;
        }
        
        /* Alert styling */
        .sentiment-alert {
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            border-left: 4px solid;
        }
        
        .alert-positive {
            background: rgba(0, 255, 136, 0.1);
            border-color: #00FF88;
        }
        
        .alert-negative {
            background: rgba(255, 0, 68, 0.1);
            border-color: #FF0044;
        }
        
        .alert-neutral {
            background: rgba(255, 165, 0, 0.1);
            border-color: #FFA500;
        }
        
        /* Table styling */
        .dataframe {
            background-color: rgba(0, 0, 0, 0.3) !important;
            color: #e0e0e0 !important;
            border: 1px solid #333;
            border-radius: 8px;
        }
        
        /* Button enhancements */
        .stButton > button {
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 240, 255, 0.3);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: rgba(0, 0, 0, 0.3) !important;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent;
            border-bottom: 2px solid #222;
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 12px 20px;
            border-radius: 6px 6px 0 0;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            color: #00F0FF !important;
            border-bottom: 3px solid #00F0FF !important;
            background: rgba(0, 240, 255, 0.1) !important;
        }
    </style>
    """, unsafe_allow_html=True)
