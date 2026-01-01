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

@st.cache_data(ttl=300)
def fetch_data(ticker, timeframe):
    p_map = {"15m": "30d", "1h": "90d", "4h": "1y", "1d": "2y", "1wk": "5y"}
    try:
        df = yf.download(ticker, period=p_map.get(timeframe, "1y"), interval=timeframe, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except: return pd.DataFrame()

# ==========================================
# 3. LIVE ADAPTABLE TRADINGVIEW HEADER
# ==========================================
def render_tv_header(selected_ticker):
    clean = selected_ticker.replace("-USD", "").replace("=X", "").upper()
    symbols = [{"proName": f"BINANCE:{clean}USDT" if "USD" in selected_ticker else f"NASDAQ:{clean}", "title": clean},
               {"proName": "BITSTAMP:BTCUSD", "title": "BTC"},
               {"proName": "CME_MINI:ES1!", "title": "S&P 500"}]
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
        df['TR'] = np.maximum(df['High']-df['Low'], np.maximum(abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))))
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
        # Apex Flux Efficiency
        rg = df['High'] - df['Low']
        body = np.abs(df['Close'] - df['Open'])
        df['Flux'] = (body / rg).ewm(span=14).mean() * np.sign(df['Close'] - df['Open'])
        # RQZO Relativity
        df['RQZO'] = np.abs(df['Close'].pct_change()).rolling(length).mean() * 1000
        return df

# ==========================================
# 5. MAIN TERMINAL UI
# ==========================================
def main():
    inject_terminal_css()
    
    with st.sidebar:
        st.header("⚙️ TERMINAL CONTROL")
        api_key = st.text_input("OpenAI Key", type="password")
        st.divider()
        ticker = st.selectbox("Global Asset Selection", TickerService.get_universe())
        tf = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d", "1wk"], index=3)
        use_ai = st.toggle("AI Parameter Optimizer (APO)", value=True)
        st.divider()
        tg_token = st.text_input("Telegram Bot Token", type="password")
        tg_chat = st.text_input("Telegram Chat ID")

    render_tv_header(ticker)
    df = fetch_data(ticker, tf)
    
    if not df.empty:
        df = AxiomEngine.calculate(df)
        df = FlowEngine.calculate(df, {"len_f": 21, "len_s": 50})
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
                fig_macro.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
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

        # --- TAB 2: QUANTUM PHYSICS ---
        with tab2:
            st.subheader("⚛️ Axiom Physics Suite")
            
            # Row 1: CHEDO Entropy
            c1_plot, c1_text = st.columns([0.7, 0.3])
            with c1_plot:
                fig_chedo = go.Figure(go.Scatter(x=df.index, y=df['CHEDO'], fill='tozeroy', line=dict(color='#00F0FF')))
                fig_chedo.update_layout(height=250, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), yaxis_title="CHEDO")
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
                fig_flux.update_layout(height=250, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), yaxis_title="Flux")
                st.plotly_chart(fig_flux, use_container_width=True)
            with c2_text:
                st.markdown(f"""<div class="analysis-box"><div class="analysis-title">Efficiency (Flux)</div><div class="analysis-text">
                Current: <span class="highlight">{last['Flux']:.2f}</span>. <br>Flux measures movement efficiency. 
                Scores > 0.6 indicate <span class="highlight">Superconductor State</span> where price moves with zero friction.
                </div></div>""", unsafe_allow_html=True)

        # --- TAB 3: BROADCAST ---
        with tab3:
            st.header("📡 Global Broadcaster")
            msg = f"🚀 TITAN SIGNAL: {ticker}\nTrend: {'BULL' if last['is_uptrend'] else 'BEAR'}\nPrice: ${last['Close']:.2f}\nEntropy: {last['CHEDO']:.2f}"
            st.code(msg)
            if st.button("Send to Telegram"):
                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", json={"chat_id": tg_chat, "text": msg})
                st.success("Dispatched.")

if __name__ == "__main__":
    main()
