import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import requests
import json
import urllib.parse
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. PAGE CONFIG & DARKPOOL STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN AI MACRO", page_icon="👁️")

def inject_terminal_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@300;400;700&display=swap');
        .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Inter', sans-serif; }
        
        /* Glassmorphism Metric Cards */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.02);
            border-left: 3px solid #00F0FF;
            backdrop-filter: blur(15px);
            border-radius: 4px; padding: 20px;
        }
        
        /* Side-Car Analysis Styling */
        .analysis-box {
            background: rgba(0, 240, 255, 0.03);
            border-radius: 8px;
            padding: 25px;
            border: 1px solid rgba(0, 240, 255, 0.1);
            height: 100%;
        }
        .analysis-title { color: #00F0FF; font-family: 'Roboto Mono'; font-size: 1.2rem; margin-bottom: 15px; font-weight: bold; }
        .analysis-text { color: #aaa; line-height: 1.6; font-size: 0.95rem; }
        .highlight { color: #fff; font-weight: bold; }
        
        /* Buttons & Tabs */
        .stTabs [data-baseweb="tab-list"] { background-color: transparent; border-bottom: 1px solid #222; }
        .stTabs [aria-selected="true"] { color: #00F0FF; border-bottom: 2px solid #00F0FF; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA SERVICES (200+ TICKERS)
# ==========================================
class TickerService:
    @staticmethod
    def get_universe():
        crypto = [f"{c}-USD" for c in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "MATIC", "LTC", "BCH", "UNI", "SHIB", "NEAR", "ICP", "STX", "AAVE", "MKR", "LDO", "RNDR", "INJ", "TIA", "SUI", "APT"]]
        stocks = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "INTC", "ADBE", "CRM", "AVGO", "QCOM", "COST", "PYPL", "SBUX", "DIS", "BA", "NKE", "LLY", "V", "MA", "JPM", "WMT", "UNH", "PG", "HD", "ORCL", "MRK", "ABBV", "BAC", "CVX", "KO", "PEP", "TMO", "PFE"]
        indices = ["SPY", "QQQ", "IWM", "DIA", "TLT", "VXX", "GLD", "SLV", "USO", "UNG", "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "ARKK", "SMH"]
        forex = ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "USDCAD=X", "DX-Y.NYB"]
        return sorted(list(set(crypto + stocks + indices + forex)))

    @staticmethod
    def classify(ticker: str) -> str:
        # Asset-class mapping rules
        if ticker.endswith("-USD"):
            return "CRYPTO"
        if ticker.endswith("=X") or ticker == "DX-Y.NYB":
            return "FX"
        # Use the same lists already present (no assumptions beyond this script)
        if ticker in ["SPY", "QQQ", "IWM", "DIA", "TLT", "VXX", "GLD", "SLV", "USO", "UNG", "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "ARKK", "SMH"]:
            return "ETF/INDEX"
        return "STOCK"

@st.cache_data(ttl=300)
def fetch_data(ticker, timeframe):
    p_map = {"15m": "30d", "1h": "90d", "4h": "1y", "1d": "2y", "1wk": "5y"}
    try:
        df = yf.download(ticker, period=p_map.get(timeframe, "1y"), interval=timeframe, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if df.empty:
            return pd.DataFrame(), f"No data returned for {ticker} ({timeframe}). This can happen with unsupported intervals or invalid symbols."
        return df, ""
    except Exception as e:
        return pd.DataFrame(), f"Data fetch error for {ticker} ({timeframe}): {type(e).__name__}: {str(e)}"

# ==========================================
# 3. LIVE ADAPTABLE TRADINGVIEW HEADER
# ==========================================
def _default_tv_symbol(selected_ticker: str) -> str:
    # Special case: Yahoo's Dollar Index ticker
    if selected_ticker == "DX-Y.NYB":
        return "TVC:DXY"

    asset_class = TickerService.classify(selected_ticker)

    clean = selected_ticker.replace("-USD", "").replace("=X", "").upper()

    if asset_class == "CRYPTO":
        # Keep original BINANCE mapping style
        return f"BINANCE:{clean}USDT"
    if asset_class == "FX":
        # Use FX:<PAIR> style for typical ticker tape widgets
        return f"FX:{clean}"
    if asset_class == "ETF/INDEX":
        # Default to NYSEARCA for ETFs (user can override)
        return f"NYSEARCA:{clean}"
    # STOCK
    return f"NASDAQ:{clean}"

def render_tv_header(selected_ticker, tv_exchange_prefix_override: str = "", tv_symbol_override_full: str = ""):
    # If user provides a full override like "NASDAQ:AAPL", use it directly.
    if tv_symbol_override_full.strip():
        selected_symbol = tv_symbol_override_full.strip().upper()
        selected_title = selected_ticker.replace("-USD", "").replace("=X", "").upper()
    else:
        # Default symbol by asset class
        default_symbol = _default_tv_symbol(selected_ticker)
        selected_title = selected_ticker.replace("-USD", "").replace("=X", "").upper()

        # If user overrides exchange prefix only, rebuild symbol as PREFIX:<SYMBOL_PART>
        if tv_exchange_prefix_override.strip():
            prefix = tv_exchange_prefix_override.strip().upper()
            if ":" in default_symbol:
                _, sym = default_symbol.split(":", 1)
                selected_symbol = f"{prefix}:{sym}"
            else:
                selected_symbol = default_symbol
        else:
            selected_symbol = default_symbol

    symbols = [
        {"proName": selected_symbol, "title": selected_title},
        {"proName": "BITSTAMP:BTCUSD", "title": "BTC"},
        {"proName": "CME_MINI:ES1!", "title": "S&P 500"},
    ]
    config = {"symbols": symbols, "colorTheme": "dark", "isTransparent": True, "displayMode": "adaptive", "locale": "en"}
    encoded = urllib.parse.quote(json.dumps(config))
    url = f"https://s.tradingview.com/embed-widget/ticker-tape/?locale=en#{encoded}"
    components.html(f'<iframe src="{url}" style="width:100%; height:46px; border:none; overflow:hidden;"></iframe>', height=50)

# ==========================================
# 4. ENGINES: FLOW & AXIOM
# ==========================================
class FlowEngine:
    @staticmethod
    def calculate(df, p):
        df['FastMA'] = df['Close'].ewm(span=p['len_f']).mean()
        df['SlowMA'] = df['Close'].ewm(span=p['len_s']).mean()
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
        )
        df['ATR'] = df['TR'].rolling(14).mean()
        df['is_uptrend'] = df['FastMA'] > df['SlowMA']
        df['long_trigger'] = (df['is_uptrend']) & (df['is_uptrend'].shift(1) == False)
        return df

class AxiomEngine:
    @staticmethod
    def calculate(df, length=50):
        # CHEDO Entropy
        log_ret = np.diff(np.log(df['Close'].values), prepend=np.log(df['Close'].iloc[0]))
        ent = pd.Series(log_ret**2).rolling(length).sum().values
        df['CHEDO'] = np.tanh(ent * 10)

        # Apex Flux Efficiency (HARDENED: guard rg == 0)
        rg = (df['High'] - df['Low']).astype(float)
        body = np.abs(df['Close'] - df['Open']).astype(float)

        safe_rg = rg.replace(0.0, np.nan)
        ratio = (body / safe_rg).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df['Flux'] = ratio.ewm(span=14).mean() * np.sign(df['Close'] - df['Open'])

        # RQZO Relativity
        df['RQZO'] = np.abs(df['Close'].pct_change()).rolling(length).mean() * 1000
        return df

# ==========================================
# 4B. AI PARAMETER OPTIMIZER (APO)
# ==========================================
def ai_optimize_params(api_key: str, model_name: str, ticker: str, timeframe: str, df: pd.DataFrame, default_len_f: int = 21, default_len_s: int = 50):
    """
    APO returns a dict: {"len_f": int, "len_s": int, "raw": str}
    Safe fallback to defaults on any failure.
    """
    out = {"len_f": default_len_f, "len_s": default_len_s, "raw": ""}

    if not api_key:
        out["raw"] = "APO skipped: missing OpenAI key."
        return out

    if df is None or df.empty or len(df) < 60:
        out["raw"] = "APO skipped: insufficient data."
        return out

    # Compute compact summary (no assumptions beyond what's in the DF)
    recent = df.tail(200).copy()
    close = recent["Close"]
    returns = close.pct_change().dropna()
    vol = float(returns.std() * np.sqrt(252)) if len(returns) > 2 else 0.0
    last_close = float(close.iloc[-1])
    last_ret = float(returns.iloc[-1]) if len(returns) > 0 else 0.0
    range_pct = float(((recent["High"] - recent["Low"]) / recent["Close"]).tail(50).mean()) if len(recent) >= 50 else 0.0

    prompt = f"""
You are an optimizer for a fast/slow EMA trend regime used for signal broadcasting.
Return ONLY strict JSON with keys: len_f, len_s.

Constraints:
- len_f integer in [5, 100]
- len_s integer in [10, 200]
- len_s must be > len_f

Context:
- Ticker: {ticker}
- Timeframe: {timeframe}
- Last Close: {last_close}
- Last Return: {last_ret}
- Annualized Vol (approx): {vol}
- Avg Range% (last ~50 bars): {range_pct}

Goal:
- Choose len_f, len_s that balance responsiveness and noise for trend-following signals.
- Prefer stable parameters when volatility is high.
"""

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You return only JSON. No prose."},
                {"role": "user", "content": prompt.strip()},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        out["raw"] = raw

        # Parse JSON safely
        data = json.loads(raw)
        len_f = int(data.get("len_f", default_len_f))
        len_s = int(data.get("len_s", default_len_s))

        # Enforce constraints
        len_f = max(5, min(100, len_f))
        len_s = max(10, min(200, len_s))
        if len_s <= len_f:
            len_s = min(200, len_f + 20)

        out["len_f"] = len_f
        out["len_s"] = len_s
        return out

    except Exception as e:
        out["raw"] = f"APO error: {type(e).__name__}: {str(e)}"
        return out

# ==========================================
# 4C. TELEGRAM SEND + DELIVERY LOG
# ==========================================
def _init_delivery_log():
    if "delivery_log" not in st.session_state:
        st.session_state.delivery_log = []

def telegram_send(bot_token: str, chat_id: str, text: str, timeout_sec: int = 10):
    """
    Returns dict with: ok(bool), status_code(int|None), response_text(str), response_snippet(str)
    """
    result = {"ok": False, "status_code": None, "response_text": "", "response_snippet": ""}
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=timeout_sec
        )
        result["status_code"] = r.status_code
        result["response_text"] = r.text or ""
        snippet = (result["response_text"][:220] + "…") if len(result["response_text"]) > 220 else result["response_text"]
        result["response_snippet"] = snippet

        if r.status_code >= 200 and r.status_code < 300:
            # Telegram responses usually include {"ok": true/false, ...}
            try:
                payload = r.json()
                result["ok"] = bool(payload.get("ok", True))
            except Exception:
                result["ok"] = True
        else:
            result["ok"] = False

        return result
    except Exception as e:
        result["ok"] = False
        result["response_text"] = f"{type(e).__name__}: {str(e)}"
        result["response_snippet"] = result["response_text"][:220]
        return result

def append_delivery_log(entry: dict):
    _init_delivery_log()
    st.session_state.delivery_log.insert(0, entry)
    # Keep log small
    st.session_state.delivery_log = st.session_state.delivery_log[:50]

def build_message(report_type: str, ticker: str, tf: str, last: pd.Series) -> str:
    # STRICT SIGNAL (original style preserved, expanded with timeframe)
    if report_type == "STRICT SIGNAL":
        return (
            f"🚀 TITAN SIGNAL: {ticker}\n"
            f"Timeframe: {tf}\n"
            f"Trend: {'BULL' if last['is_uptrend'] else 'BEAR'}\n"
            f"Price: ${last['Close']:.2f}\n"
            f"Entropy: {last['CHEDO']:.2f}"
        )

    # RISK template
    if report_type == "RISK":
        atr = last.get("ATR", np.nan)
        rqzo = last.get("RQZO", np.nan)
        flux = last.get("Flux", np.nan)
        chedo = last.get("CHEDO", np.nan)
        return (
            f"⚠️ TITAN RISK REPORT: {ticker}\n"
            f"Timeframe: {tf}\n"
            f"Trend Regime: {'BULLISH' if last['is_uptrend'] else 'BEARISH'}\n"
            f"Price: ${last['Close']:.2f}\n"
            f"CHEDO (Entropy): {chedo:.2f}\n"
            f"Flux (Efficiency): {flux:.2f}\n"
            f"RQZO (Relativity): {rqzo:.2f}\n"
            f"ATR(14): {atr:.4f}\n"
            f"Risk Note: Entropy > 0.80 = Critical Instability; Flux magnitude > 0.60 = High-efficiency impulse."
        )

    # MARKET SUMMARY template
    if report_type == "MARKET SUMMARY":
        spread = abs(last["FastMA"] - last["SlowMA"])
        return (
            f"📊 TITAN MARKET SUMMARY: {ticker}\n"
            f"Timeframe: {tf}\n"
            f"Trend: {'BULLISH' if last['is_uptrend'] else 'BEARISH'}\n"
            f"Last Close: ${last['Close']:.2f}\n"
            f"Fast/Slow Spread: {spread:.2f}\n"
            f"CHEDO: {last['CHEDO']:.2f} | Flux: {last['Flux']:.2f}\n"
            f"Signal: {'Trend Flip Long Trigger' if bool(last.get('long_trigger', False)) else 'No Fresh Trigger'}"
        )

    # Fallback (should not happen)
    return (
        f"🚀 TITAN SIGNAL: {ticker}\n"
        f"Timeframe: {tf}\n"
        f"Trend: {'BULL' if last['is_uptrend'] else 'BEAR'}\n"
        f"Price: ${last['Close']:.2f}\n"
        f"Entropy: {last['CHEDO']:.2f}"
    )

# ==========================================
# 5. MAIN TERMINAL UI
# ==========================================
def main():
    inject_terminal_css()
    _init_delivery_log()

    # --- SECRETS MODE (optional) ---
    def _get_secret(path_a: str, path_b: str, default=""):
        # Supports st.secrets["a"]["b"] access
        try:
            return st.secrets[path_a][path_b]
        except Exception:
            return default

    with st.sidebar:
        st.header("⚙️ TERMINAL CONTROL")

        use_secrets = st.toggle("Use st.secrets (recommended)", value=False)

        # OpenAI Key + APO
        if use_secrets:
            api_key = _get_secret("openai", "api_key", "")
            if not api_key:
                st.warning("st.secrets missing: openai.api_key — you can enter it manually below.")
                api_key = st.text_input("OpenAI Key", type="password")
        else:
            api_key = st.text_input("OpenAI Key", type="password")

        model_name = st.text_input("OpenAI Model (APO)", value="gpt-4o-mini")

        st.divider()
        ticker = st.selectbox("Global Asset Selection", TickerService.get_universe())
        tf = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d", "1wk"], index=3)
        use_ai = st.toggle("AI Parameter Optimizer (APO)", value=True)

        st.divider()

        # TradingView header controls
        st.subheader("📺 TradingView Header")
        asset_class = TickerService.classify(ticker)
        st.caption(f"Asset Class: {asset_class}")

        default_symbol = _default_tv_symbol(ticker)
        default_prefix = default_symbol.split(":", 1)[0] if ":" in default_symbol else ""

        tv_exchange_prefix_override = st.text_input(
            "TV Exchange Prefix Override (optional)",
            value=default_prefix,
            help="Example: BINANCE, NASDAQ, NYSEARCA, FX, TVC. This only changes the prefix for the selected asset."
        )
        tv_symbol_override_full = st.text_input(
            "TV Full Symbol Override (optional)",
            value="",
            help='If set, use a complete symbol like "NASDAQ:AAPL" or "BINANCE:BTCUSDT". Takes precedence over prefix override.'
        )

        st.divider()

        # Telegram credentials
        if use_secrets:
            tg_token = _get_secret("telegram", "bot_token", "")
            tg_chat = _get_secret("telegram", "chat_id", "")
            if not tg_token or not tg_chat:
                st.warning("st.secrets missing: telegram.bot_token and/or telegram.chat_id — you can enter them manually below.")
                tg_token = st.text_input("Telegram Bot Token", type="password", value=tg_token)
                tg_chat = st.text_input("Telegram Chat ID", value=tg_chat)
        else:
            tg_token = st.text_input("Telegram Bot Token", type="password")
            tg_chat = st.text_input("Telegram Chat ID")

    render_tv_header(ticker, tv_exchange_prefix_override=tv_exchange_prefix_override, tv_symbol_override_full=tv_symbol_override_full)

    df, data_err = fetch_data(ticker, tf)

    if data_err:
        st.warning(data_err)

    if not df.empty:
        df = AxiomEngine.calculate(df)

        # APO param selection
        len_f, len_s = 21, 50
        apo_debug = ""
        if use_ai:
            apo = ai_optimize_params(api_key=api_key, model_name=model_name, ticker=ticker, timeframe=tf, df=df, default_len_f=21, default_len_s=50)
            len_f, len_s = apo["len_f"], apo["len_s"]
            apo_debug = apo.get("raw", "")

        df = FlowEngine.calculate(df, {"len_f": len_f, "len_s": len_s})
        last = df.iloc[-1]

        tab1, tab2, tab3 = st.tabs(["📊 Macro Intelligence", "⚛️ Quantitative Physics", "📡 Global Broadcast"])

        # --- TAB 1: MACRO INTELLIGENCE ---
        with tab1:
            col_chart, col_text = st.columns([0.7, 0.3])

            with col_chart:
                fig_macro = make_subplots(rows=1, cols=1)
                fig_macro.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Price"))
                fig_macro.add_trace(go.Scatter(x=df.index, y=df['FastMA'], line=dict(color='cyan', width=1.5), name="Fast MA"))
                fig_macro.add_trace(go.Scatter(x=df.index, y=df['SlowMA'], fill='tonexty', fillcolor='rgba(0,255,104,0.05)', line=dict(color='yellow', width=1), name="Macro Cloud"))
                fig_macro.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_macro, use_container_width=True)

            with col_text:
                st.markdown(f"""
                <div class="analysis-box">
                    <div class="analysis-title">🌊 DARKPOOL FLOW ANALYSIS</div>
                    <div class="analysis-text">
                        The current trend for <span class="highlight">{ticker}</span> is <span class="highlight">{"BULLISH" if last['is_uptrend'] else "BEARISH"}</span>. 
                        <br><br>
                        The <span class="highlight">Macro Cloud</span> (Fast/Slow EMA spread) indicates institutional accumulation zones. 
                        When price holds above the cloud, it suggests a "Buy the Dip" regime. 
                        <br><br>
                        <span class="highlight">Reading:</span> The spread is currently {abs(last['FastMA'] - last['SlowMA']):.2f}. 
                        A widening spread confirms trend acceleration, while a narrowing spread warns of a potential crossover reversal.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if use_ai and apo_debug:
                    with st.expander("APO Debug (raw model output)", expanded=False):
                        st.code(apo_debug)

        # --- TAB 2: QUANTUM PHYSICS ---
        with tab2:
            st.subheader("⚛️ Axiom Physics Suite")

            # Row 1: CHEDO Entropy
            c1_plot, c1_text = st.columns([0.7, 0.3])
            with c1_plot:
                fig_chedo = go.Figure(go.Scatter(x=df.index, y=df['CHEDO'], fill='tozeroy', line=dict(color='#00F0FF')))
                fig_chedo.update_layout(height=250, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0), yaxis_title="CHEDO")
                st.plotly_chart(fig_chedo, use_container_width=True)
            with c1_text:
                st.markdown(f"""<div class="analysis-box"><div class="analysis-title">Entropy (CHEDO)</div><div class="analysis-text">
                Current: <span class="highlight">{last['CHEDO']:.2f}</span>. <br>Entropy measures market chaos. 
                Values > 0.8 indicate <span class="highlight">Critical Instability</span>, suggesting an imminent snap-back or trend exhaustion.
                </div></div>""", unsafe_allow_html=True)

            # Row 2: Apex Flux
            c2_plot, c2_text = st.columns([0.7, 0.3])
            with c2_plot:
                colors = ['#00E676' if x > 0 else '#FF1744' for x in df['Flux']]
                fig_flux = go.Figure(go.Bar(x=df.index, y=df['Flux'], marker_color=colors))
                fig_flux.update_layout(height=250, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0), yaxis_title="Flux")
                st.plotly_chart(fig_flux, use_container_width=True)
            with c2_text:
                st.markdown(f"""<div class="analysis-box"><div class="analysis-title">Efficiency (Flux)</div><div class="analysis-text">
                Current: <span class="highlight">{last['Flux']:.2f}</span>. <br>Flux measures movement efficiency. 
                Scores > 0.6 indicate <span class="highlight">Superconductor State</span> where price moves with zero friction.
                </div></div>""", unsafe_allow_html=True)

        # --- TAB 3: BROADCAST ---
        with tab3:
            st.header("📡 Global Broadcaster")

            report_type = st.selectbox(
                "Report Template",
                ["STRICT SIGNAL", "RISK", "MARKET SUMMARY"],
                index=0
            )

            msg = build_message(report_type, ticker, tf, last)
            st.code(msg)

            col_send, col_log = st.columns([0.35, 0.65])

            with col_send:
                if st.button("Send to Telegram"):
                    if not tg_token or not tg_chat:
                        st.error("Missing Telegram credentials: provide Bot Token and Chat ID (or enable st.secrets mode with telegram.bot_token and telegram.chat_id).")
                    else:
                        result = telegram_send(tg_token, tg_chat, msg, timeout_sec=10)

                        entry = {
                            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "ticker": ticker,
                            "timeframe": tf,
                            "template": report_type,
                            "status_code": result["status_code"],
                            "ok": result["ok"],
                            "snippet": result["response_snippet"],
                        }
                        append_delivery_log(entry)

                        if result["ok"]:
                            st.success(f"Dispatched. HTTP {result['status_code']}.")
                        else:
                            st.error(f"Telegram dispatch failed. HTTP {result['status_code']}. Response: {result['response_snippet']}")

            with col_log:
                st.subheader("📬 Delivery Log")
                if st.session_state.delivery_log:
                    log_df = pd.DataFrame(st.session_state.delivery_log)
                    st.dataframe(log_df, use_container_width=True, height=260)
                else:
                    st.info("No messages sent yet.")

if __name__ == "__main__":
    main()
