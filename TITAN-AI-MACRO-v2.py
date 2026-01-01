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
import time

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
            min-height: 400px;
        }
        .analysis-title { color: #00F0FF; font-family: 'Roboto Mono'; font-size: 1.2rem; margin-bottom: 15px; font-weight: bold; }
        .analysis-text { color: #aaa; line-height: 1.6; font-size: 0.95rem; white-space: pre-wrap; }
        .highlight { color: #fff; font-weight: bold; }
        
        /* Buttons & Tabs */
        .stTabs [data-baseweb="tab-list"] { background-color: transparent; border-bottom: 1px solid #222; }
        .stTabs [aria-selected="true"] { color: #00F0FF; border-bottom: 2px solid #00F0FF; }
        
        /* Toast Styling */
        div[data-baseweb="toast"] { background-color: #111; border: 1px solid #00F0FF; color: #fff; }
        
        /* Text Area for Review */
        textarea { background-color: #0a0a0a !important; color: #00F0FF !important; border: 1px solid #333 !important; font-family: 'Roboto Mono' !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA SERVICES & INPUT HANDLING
# ==========================================
class TickerService:
    @staticmethod
    def get_universe():
        crypto = [f"{c}-USD" for c in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "MATIC", "LTC", "BCH", "UNI", "SHIB", "NEAR", "ICP", "STX", "AAVE", "MKR", "LDO", "RNDR", "INJ", "TIA", "SUI", "APT"]]
        stocks = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "INTC", "ADBE", "CRM", "AVGO", "QCOM", "COST", "PYPL", "SBUX", "DIS", "BA", "NKE", "LLY", "V", "MA", "JPM", "WMT", "UNH", "PG", "HD", "ORCL", "MRK", "ABBV", "BAC", "CVX", "KO", "PEP", "TMO", "PFE"]
        indices = ["SPY", "QQQ", "IWM", "DIA", "TLT", "VXX", "GLD", "SLV", "USO", "UNG", "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "ARKK", "SMH"]
        forex = ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "USDCAD=X", "DX-Y.NYB"]
        return sorted(list(set(crypto + stocks + indices + forex)))

@st.cache_data(ttl=300)
def fetch_data(ticker, timeframe):
    p_map = {"15m": "30d", "1h": "90d", "4h": "1y", "1d": "2y", "1wk": "5y"}
    try:
        # Progress=False suppresses the yfinance stdout print
        df = yf.download(ticker, period=p_map.get(timeframe, "1y"), interval=timeframe, progress=False)
        
        # CRITICAL FIX: Handle MultiIndex columns (common in new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Ensure we have data
        if df.empty: return pd.DataFrame()
        
        return df.dropna()
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# 3. TRADINGVIEW HEADER
# ==========================================
def render_tv_header(selected_ticker):
    clean = selected_ticker.replace("-USD", "").replace("=X", "").upper()
    
    # Try to guess the TradingView symbol format
    tv_symbol = f"BINANCE:{clean}USDT" if "USD" in selected_ticker else f"NASDAQ:{clean}"
    
    symbols = [{"proName": tv_symbol, "title": clean},
               {"proName": "BITSTAMP:BTCUSD", "title": "BTC"},
               {"proName": "CME_MINI:ES1!", "title": "S&P 500"}]
               
    config = {"symbols": symbols, "colorTheme": "dark", "isTransparent": True, "displayMode": "adaptive", "locale": "en"}
    encoded = urllib.parse.quote(json.dumps(config))
    url = f"https://s.tradingview.com/embed-widget/ticker-tape/?locale=en#{encoded}"
    
    components.html(f'<iframe src="{url}" style="width:100%; height:46px; border:none; overflow:hidden;"></iframe>', height=50)

# ==========================================
# 4. ENGINES: FLOW, AXIOM, & AI
# ==========================================
class FlowEngine:
    @staticmethod
    def calculate(df, p):
        df['FastMA'] = df['Close'].ewm(span=p['len_f']).mean()
        df['SlowMA'] = df['Close'].ewm(span=p['len_s']).mean()
        
        # True Range Calculation
        df['TR'] = np.maximum(df['High']-df['Low'], np.maximum(abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(14).mean()
        
        # Trend Logic
        df['is_uptrend'] = df['FastMA'] > df['SlowMA']
        df['long_trigger'] = (df['is_uptrend']) & (df['is_uptrend'].shift(1) == False)
        return df

class AxiomEngine:
    @staticmethod
    def calculate(df, length=50):
        # CHEDO Entropy (Chaos Metric)
        log_ret = np.diff(np.log(df['Close'].values), prepend=np.log(df['Close'].iloc[0]))
        ent = pd.Series(log_ret**2).rolling(length).sum().values
        df['CHEDO'] = np.tanh(ent * 10)
        
        # Apex Flux Efficiency (Movement Directness)
        rg = df['High'] - df['Low']
        rg = rg.replace(0, np.nan) # Avoid div by zero
        body = np.abs(df['Close'] - df['Open'])
        df['Flux'] = (body / rg).ewm(span=14).mean() * np.sign(df['Close'] - df['Open'])
        
        # RQZO Relativity
        df['RQZO'] = np.abs(df['Close'].pct_change()).rolling(length).mean() * 1000
        return df

class AIEngine:
    @staticmethod
    def generate_insight(api_key, ticker, df_row):
        """Generates a short cyberpunk-style insight."""
        if not api_key: return "⚠️ AI Neural Link Disconnected."
        return AIEngine._call_gpt(api_key, ticker, df_row, "short")

    @staticmethod
    def generate_full_report(api_key, ticker, df_row):
        """Generates a detailed detailed broadcast report."""
        if not api_key: return "⚠️ AI Link Missing. Please enter API Key in sidebar."
        return AIEngine._call_gpt(api_key, ticker, df_row, "detailed")

    @staticmethod
    def _call_gpt(api_key, ticker, df_row, mode):
        client = OpenAI(api_key=api_key)
        
        price = df_row['Close']
        trend = "BULLISH" if df_row['is_uptrend'] else "BEARISH"
        flux = df_row['Flux']
        entropy = df_row['CHEDO']
        
        if mode == "short":
            prompt = f"""
            Analyze {ticker} (Price: {price:.2f}, Trend: {trend}).
            Flux: {flux:.2f}, Entropy: {entropy:.2f}.
            Write 3 concise sentences using cyberpunk trading terminology. No headers.
            """
        else:
            prompt = f"""
            You are TITAN, an elite quant AI. Generate a FULL BROADCAST REPORT for {ticker}.
            
            MARKET DATA:
            - Price: {price:.2f}
            - Structure: {trend}
            - Flux Efficiency: {flux:.2f} (-1.0 to 1.0)
            - Entropy/Chaos: {entropy:.2f} (0.0 to 1.0)
            
            OUTPUT FORMAT:
            1. HEADLINE: [Emoji] Ticker - Action (e.g. ACCUMULATE / WATCH / SHORT)
            2. ANALYSIS: A paragraph analyzing the momentum and entropy.
            3. LEVELS: Identify 2 key psychological levels based on the price.
            4. VERDICT: Final tactical instruction.
            
            Tone: Professional, institutional, slightly futuristic.
            """
            
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a financial trading AI."}, 
                          {"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ AI Transmission Failed: {str(e)}"

# ==========================================
# 5. MAIN TERMINAL UI
# ==========================================
def main():
    inject_terminal_css()
    
    # Initialize Session State for Broadcast Review
    if "broadcast_msg" not in st.session_state:
        st.session_state.broadcast_msg = ""
    
    with st.sidebar:
        st.header("⚙️ TERMINAL CONTROL")
        
        # API Keys
        api_key = st.text_input("OpenAI Key", type="password", help="Required for AI Analysis")
        tg_token = st.text_input("Telegram Bot Token", type="password")
        tg_chat = st.text_input("Telegram Chat ID")
        st.divider()
        
        # Dynamic Ticker Selection
        mode = st.radio("Target Selection", ["Universe", "Custom"], horizontal=True)
        if mode == "Universe":
            ticker = st.selectbox("Asset Class", TickerService.get_universe())
        else:
            ticker = st.text_input("Custom Ticker", value="GME", help="Enter any Yahoo Finance ticker").upper()
            
        tf = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d", "1wk"], index=3)
        use_ai = st.toggle("AI Parameter Optimizer (APO)", value=True)
        
        st.info("System Ready. Waiting for signals.")

    # Render Header
    render_tv_header(ticker)
    
    # Fetch & Calc
    df = fetch_data(ticker, tf)
    
    if df.empty:
        st.error(f"❌ Failed to fetch data for **{ticker}**. It may be delisted or the API is rate-limited.")
        return

    # Run Engines
    df = AxiomEngine.calculate(df)
    df = FlowEngine.calculate(df, {"len_f": 21, "len_s": 50})
    last = df.iloc[-1]
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Macro Intelligence", "⚛️ Quantitative Physics", "📡 Global Broadcast"])
    
    # --- TAB 1: MACRO INTELLIGENCE ---
    with tab1:
        col_chart, col_text = st.columns([0.7, 0.3])
        
        with col_chart:
            # Main Candlestick Chart with MA Cloud
            fig_macro = make_subplots(rows=1, cols=1)
            fig_macro.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Price"))
            fig_macro.add_trace(go.Scatter(x=df.index, y=df['FastMA'], line=dict(color='cyan', width=1.5), name="Fast MA"))
            fig_macro.add_trace(go.Scatter(x=df.index, y=df['SlowMA'], fill='tonexty', fillcolor='rgba(0,255,104,0.05)', line=dict(color='yellow', width=1), name="Macro Cloud"))
            
            fig_macro.update_layout(
                height=650, 
                template="plotly_dark", 
                xaxis_rangeslider_visible=False, 
                margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_macro, use_container_width=True)
        
        with col_text:
            # Dynamic Analysis Box
            st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
            st.markdown('<div class="analysis-title">🌊 DARKPOOL FLOW ANALYSIS</div>', unsafe_allow_html=True)
            
            if use_ai and api_key:
                # Use Live AI
                with st.spinner("Decrypting Market Data..."):
                    ai_text = AIEngine.generate_insight(api_key, ticker, last)
                st.markdown(f'<div class="analysis-text">{ai_text}</div>', unsafe_allow_html=True)
            else:
                # Fallback Static Logic
                spread = abs(last['FastMA'] - last['SlowMA'])
                trend_status = "BULLISH" if last['is_uptrend'] else "BEARISH"
                
                static_html = f"""
                <div class="analysis-text">
                    The current trend for <span class="highlight">{ticker}</span> is <span class="highlight">{trend_status}</span>. 
                    <br><br>
                    The <span class="highlight">Macro Cloud</span> (Fast/Slow EMA spread) indicates institutional accumulation zones. 
                    When price holds above the cloud, it suggests a "Buy the Dip" regime. 
                    <br><br>
                    <span class="highlight">Reading:</span> The spread is currently {spread:.2f}. 
                    A widening spread confirms trend acceleration, while a narrowing spread warns of a potential crossover reversal.
                    <br><br>
                    <i>Enable AI switch + API Key for neural analysis.</i>
                </div>
                """
                st.markdown(static_html, unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: QUANTUM PHYSICS ---
    with tab2:
        st.subheader("⚛️ Axiom Physics Suite")
        
        # Row 1: CHEDO Entropy
        c1_plot, c1_text = st.columns([0.7, 0.3])
        with c1_plot:
            fig_chedo = go.Figure(go.Scatter(x=df.index, y=df['CHEDO'], fill='tozeroy', line=dict(color='#00F0FF')))
            fig_chedo.update_layout(height=250, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), yaxis_title="CHEDO", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_chedo, use_container_width=True)
        with c1_text:
            st.markdown(f"""<div class="analysis-box" style="min-height:200px"><div class="analysis-title">Entropy (CHEDO)</div><div class="analysis-text">
            Current: <span class="highlight">{last['CHEDO']:.2f}</span>. <br>Entropy measures market chaos. 
            Values > 0.8 indicate <span class="highlight">Critical Instability</span>, suggesting an imminent snap-back or trend exhaustion.
            </div></div>""", unsafe_allow_html=True)

    # --- TAB 3: GLOBAL BROADCAST (UPDATED) ---
    with tab3:
        st.header("📡 Global Broadcaster")
        st.markdown("Use this module to generate, review, and transmit signals to the Darkpool channel.")
        
        col_gen, col_preview = st.columns([0.4, 0.6])
        
        with col_gen:
            st.subheader("1. Signal Generation")
            include_ai_report = st.checkbox("Include Detailed AI Report", value=True)
            
            if st.button("⚡ Generate Signal Draft", use_container_width=True):
                # Basic Data
                basic_msg = f"🚀 TITAN SIGNAL: {ticker}\nTrend: {'BULL 🟢' if last['is_uptrend'] else 'BEAR 🔴'}\nPrice: {last['Close']:.2f}\nEntropy: {last['CHEDO']:.2f}\nFlux: {last['Flux']:.2f}"
                
                # Append AI Report if requested
                if include_ai_report and api_key:
                    with st.spinner("Synthesizing Neural Report..."):
                        report = AIEngine.generate_full_report(api_key, ticker, last)
                        final_msg = f"{basic_msg}\n\n{report}\n\n---\nGenerated: {datetime.now().strftime('%H:%M:%S UTC')}"
                else:
                    final_msg = f"{basic_msg}\n\n---\nGenerated: {datetime.now().strftime('%H:%M:%S UTC')}"
                
                # Store in Session State
                st.session_state.broadcast_msg = final_msg
                st.success("Draft Generated. Please Review.")
        
        with col_preview:
            st.subheader("2. Review & Transmit")
            
            # Text Area bound to Session State for editing
            edited_msg = st.text_area(
                "Review Dispatch (Editable)", 
                value=st.session_state.broadcast_msg, 
                height=300,
                help="Edit the message here before sending."
            )
            
            # Confirm Send Button
            if st.button("📡 Confirm & Transmit", type="primary", use_container_width=True):
                if not edited_msg:
                    st.warning("⚠️ Draft is empty. Please generate a signal first.")
                elif not tg_token or not tg_chat:
                    st.error("❌ Transmission Aborted: Missing Telegram Token or Chat ID.")
                else:
                    with st.spinner("Encrypting and Broadcasting..."):
                        try:
                            url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
                            payload = {"chat_id": tg_chat, "text": edited_msg}
                            res = requests.post(url, json=payload, timeout=5)
                            
                            if res.status_code == 200:
                                st.balloons()
                                st.toast("Signal Broadcasted Successfully!", icon="📡")
                                st.success("✅ Message delivered to Darkpool channel.")
                            else:
                                st.error(f"❌ Telegram Error: {res.text}")
                        except Exception as e:
                            st.error(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    main()
