
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from openai import OpenAI
from datetime import datetime, timedelta

# ==========================================
# 1. CORE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Axiom Quantitative",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. UI/UX ENGINE (MOBILE/DESKTOP SWITCHING)
# ==========================================
def inject_axiom_css(is_mobile):
    base_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=SF+Pro+Display:wght@300;500;700&display=swap');
        .stApp { background-color: #050505; color: #e0e0e0; font-family: 'SF Pro Display', sans-serif; }
        
        /* METRICS */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.03);
            border-left: 2px solid #333;
            backdrop-filter: blur(10px);
            border-radius: 0px;
            padding: 15px;
        }
        div[data-testid="stMetricLabel"] { color: #888; letter-spacing: 1px; text-transform: uppercase; }
        div[data-testid="stMetricValue"] { color: #fff; font-weight: 300; }
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] { gap: 15px; background-color: transparent; border-bottom: 1px solid #222; }
        .stTabs [data-baseweb="tab"] { background-color: transparent; border: none; color: #666; }
        .stTabs [aria-selected="true"] { color: #fff; border-bottom: 2px solid #00F0FF; }
    </style>
    """
    
    desktop_css = """
    <style>
        div[data-testid="stMetricLabel"] { font-size: 0.75rem; }
        .js-plotly-plot .plotly .modebar { orientation: v; left: 0; }
    </style>
    """
    
    mobile_css = """
    <style>
        div[data-testid="stMetric"] {
            margin-bottom: 10px;
            border-left: 4px solid #00F0FF; 
            padding: 20px; 
        }
        div[data-testid="stMetricLabel"] { font-size: 0.9rem; font-weight: 700; }
        div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
        .js-plotly-plot { height: 100% !important; }
        button { min-height: 50px !important; margin-top: 10px !important; }
    </style>
    """
    
    st.markdown(base_css, unsafe_allow_html=True)
    if is_mobile:
        st.markdown(mobile_css, unsafe_allow_html=True)
    else:
        st.markdown(desktop_css, unsafe_allow_html=True)

# ==========================================
# 3. DATA SERVICE
# ==========================================
class DataService:
    @staticmethod
    def get_asset_classes():
        return [
            "Crypto (Major)", "Crypto (Alt/Meme)", 
            "Indices & ETFs", "US Tech (Mag 7+)", 
            "US Large Cap", "High Volatility", 
            "Commodities & Forex"
        ]

    @staticmethod
    def get_tickers_by_class(asset_class):
        universe = {
            "Crypto (Major)": [
                "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "AVAX-USD", 
                "DOGE-USD", "DOT-USD", "TRX-USD", "LINK-USD", "MATIC-USD", "LTC-USD", "BCH-USD", 
                "UNI-USD", "XLM-USD", "ATOM-USD", "XMR-USD", "ETC-USD", "FIL-USD"
            ],
            "Crypto (Alt/Meme)": [
                "SHIB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FLOKI-USD", "NEAR-USD", "APT-USD", 
                "ARB-USD", "OP-USD", "RNDR-USD", "INJ-USD", "STX-USD", "IMX-USD", "VET-USD", 
                "MKR-USD", "AAVE-USD", "GRT-USD", "ALGO-USD", "QNT-USD", "FLOW-USD", "SAND-USD", 
                "MANA-USD", "AXS-USD", "THETA-USD", "EOS-USD", "XTZ-USD", "KAVA-USD", "NEO-USD"
            ],
            "Indices & ETFs": [
                "SPY", "QQQ", "IWM", "DIA", "TLT", "VXX", "UVXY", "SQQQ", "TQQQ", "SOXL", "SOXS", 
                "XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "ARKK", "SMH", "EEM", 
                "FXI", "EWZ", "GDX", "GDXJ", "JETS", "TAN", "URA", "HYG", "LQD", "TIP"
            ],
            "US Tech (Mag 7+)": [
                "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "INTC", "AVGO", 
                "QCOM", "TXN", "MU", "AMAT", "LRCX", "ADI", "KLAC", "CRM", "ADBE", "ORCL", 
                "CSCO", "NFLX", "IBM", "NOW", "UBER", "ABNB", "PANW", "SNOW", "PLTR", "CRWD"
            ],
            "US Large Cap": [
                "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "V", "MA", "AXP", 
                "JNJ", "UNH", "LLY", "PFE", "MRK", "ABBV", "TMO", "DHR", "BMY", "CVS",
                "PG", "KO", "PEP", "COST", "WMT", "TGT", "HD", "MCD", "NKE", "SBUX",
                "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "PSX", "VLO", "KMI",
                "BA", "CAT", "DE", "GE", "HON", "LMT", "RTX", "MMM", "UPS", "UNP"
            ],
            "High Volatility": [
                "MSTR", "COIN", "MARA", "RIOT", "CLSK", "HUT", "GME", "AMC", "HOOD", "DKNG", 
                "ROKU", "TDOC", "ZM", "PTON", "CVNA", "UPST", "AFRM", "SOFI", "LCID", "RIVN", 
                "NIO", "XPEV", "LI", "BABA", "PDD", "JD", "BIDU", "TCEHY", "BILI", "FUTU"
            ],
            "Commodities & Forex": [
                "GLD", "SLV", "IAU", "PPLT", "PALL", "USO", "UNG", "DBC", "DBA", "CORN", "WEAT",
                "EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
                "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "BTC=F", "ETH=F"
            ]
        }
        return universe.get(asset_class, [])

    @staticmethod
    def fetch_data(ticker, timeframe, limit=500):
        tf_map = {"15m": "1mo", "1h": "6mo", "4h": "1y", "1d": "2y", "1wk": "5y"}
        period = tf_map.get(timeframe, "1y")
        
        try:
            df = yf.download(ticker, period=period, interval=timeframe, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df.xs(ticker, axis=1, level=0)
                except:
                    df.columns = df.columns.get_level_values(0)
            
            cols = {c: c.capitalize() for c in df.columns}
            df = df.rename(columns=cols)
            if 'Adj close' in df.columns: df['Close'] = df['Adj close']
            
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(c in df.columns for c in required):
                return pd.DataFrame()
            
            return df.dropna().tail(limit)
        except Exception:
            return pd.DataFrame()

# ==========================================
# 4. QUANTITATIVE ENGINE
# ==========================================
class QuantEngine:
    @staticmethod
    def tanh(x): return np.tanh(np.clip(x, -20, 20))

    @staticmethod
    def calc_chedo(df, length=50):
        c = df['Close'].values
        log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
        
        mu = pd.Series(log_ret).rolling(length).mean().values
        sigma = pd.Series(log_ret).rolling(length).std().values
        v = sigma / (np.abs(mu) + 1e-9)
        abs_ret_v = np.abs(log_ret) * v
        hyper_dist = np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))
        kappa_h = QuantEngine.tanh(pd.Series(hyper_dist).rolling(length).mean().values)
        
        diff_ret = np.diff(log_ret, prepend=0)
        lyap = np.log(np.abs(diff_ret) + 1e-9)
        lambda_n = QuantEngine.tanh((pd.Series(lyap).rolling(length).mean().values + 5) / 7)
        
        ent = pd.Series(log_ret**2).rolling(length).sum().values
        ent_n = QuantEngine.tanh(ent * 10)
        
        raw = (0.4 * kappa_h) + (0.3 * lambda_n) + (0.3 * ent_n)
        df['CHEDO'] = 2 / (1 + np.exp(-raw * 4)) - 1
        return df

    @staticmethod
    def calc_rqzo(df, harmonics=25):
        src = df['Close']
        mn, mx = src.rolling(100).min(), src.rolling(100).max()
        norm = (src - mn) / (mx - mn + 1e-9)
        
        v = np.abs(norm.diff())
        c_limit = 0.05
        gamma = 1 / np.sqrt(1 - (np.minimum(v, c_limit*0.99)/c_limit)**2)
        
        idx = np.arange(len(df))
        tau = (idx % 100) / gamma.fillna(1.0)
        zeta = np.zeros(len(df))
        
        for n in range(1, harmonics + 1):
            amp = n ** -0.5
            theta = tau * np.log(n)
            zeta += amp * np.sin(theta)
            
        df['RQZO'] = pd.Series(zeta).fillna(0)
        return df

    @staticmethod
    def calc_apex_flux(df, length=14):
        # 1. Efficiency
        rg = df['High'] - df['Low']
        body = np.abs(df['Close'] - df['Open'])
        eff_raw = np.where(rg == 0, 0, body / rg)
        eff_series = pd.Series(eff_raw, index=df.index) 
        eff_sm = eff_series.ewm(span=length).mean()
        
        # 2. Volume Flux
        vol_avg = df['Volume'].rolling(55).mean()
        v_rat_raw = np.where(vol_avg == 0, 1, df['Volume'] / vol_avg)
        v_rat_series = pd.Series(v_rat_raw, index=df.index)
        
        # 3. Vector Direction
        direction = np.sign(df['Close'] - df['Open'])
        
        # 4. Calculation
        raw = direction * eff_sm * v_rat_series
        df['Apex_Flux'] = raw.ewm(span=5).mean()
        
        # 5. State Classification
        df['Apex_State'] = np.select(
            [df['Apex_Flux'] > 0.6, df['Apex_Flux'] < -0.6],
            ["Super Bull", "Super Bear"], default="Neutral"
        )
        return df

    @staticmethod
    def calc_smc(df, length=55):
        def wma(s, l):
            w = np.arange(1, l+1)
            return s.rolling(l).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
            
        close = df['Close']
        half = int(length/2)
        sqrt = int(np.sqrt(length))
        
        wma_f = wma(close, length)
        wma_h = wma(close, half)
        df['HMA_Trend'] = wma(2*wma_h - wma_f, sqrt)
        df['Trend_Dir'] = np.where(close > df['HMA_Trend'], 1, -1)
        
        p = 5
        df['Pivot_H'] = df['High'].rolling(p*2+1, center=True).max() == df['High']
        df['Pivot_L'] = df['Low'].rolling(p*2+1, center=True).min() == df['Low']
        return df

# ==========================================
# 5. INTELLIGENCE (AI & BROADCAST)
# ==========================================
class Intelligence:
    @staticmethod
    def check_ai_key(manual_key=None):
        """Returns valid key or None."""
        if manual_key and manual_key.startswith("sk-"):
            return manual_key
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
        return None

    @staticmethod
    def analyze(df, ticker, api_key):
        if not api_key:
            return "❌ Missing OpenAI API Key. Please provide it in the sidebar."
        
        last = df.iloc[-1]
        system_prompt = "You are Axiom, a physics-based quantitative trading assistant."
        user_prompt = f"Analyze {ticker}. Price: {last['Close']:.2f}. CHEDO: {last['CHEDO']:.2f}. RQZO: {last['RQZO']:.2f}. Flux: {last['Apex_Flux']:.2f}."
        
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ AI Error: {str(e)}"

    @staticmethod
    def broadcast_telegram(message, bot_token, chat_id):
        if not bot_token or not chat_id:
            return False, "❌ Missing Telegram Credentials"
            
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        try:
            # Chunking logic for safety
            max_len = 2000
            if len(message) <= max_len:
                requests.post(url, json={"chat_id": chat_id, "text": message})
            else:
                for i in range(0, len(message), max_len):
                    chunk = message[i:i+max_len]
                    requests.post(url, json={"chat_id": chat_id, "text": f"(Part {i//max_len+1}) {chunk}"})
            return True, "✅ Broadcast Sent"
        except Exception as e:
            return False, f"❌ Broadcast Failed: {str(e)}"

# ==========================================
# 6. GRAPHICS
# ==========================================
class Graphics:
    @staticmethod
    def render_dashboard(df, ticker, is_mobile):
        row_heights = [0.45, 0.15, 0.15, 0.25] if not is_mobile else [0.5, 0.15, 0.15, 0.2]
        total_height = 850 if not is_mobile else 1200
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03 if is_mobile else 0.015,
            row_heights=row_heights,
            specs=[[{"secondary_y": False}], [{}], [{}], [{}]],
            subplot_titles=("", "", "", "")
        )

        # Panel 1: Price
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA_Trend'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], fill='tonexty', fillcolor='rgba(255, 255, 255, 0.02)', mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price", increasing_line_color='#00F0FF', increasing_fillcolor='rgba(0,240,255,0.1)', decreasing_line_color='#FF0055', decreasing_fillcolor='rgba(255,0,85,0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA_Trend'], line=dict(color='#FFFFFF', width=1, dash='dot'), name="SMC Base"), row=1, col=1)
        bos_h = df[df['Pivot_H']]
        bos_l = df[df['Pivot_L']]
        fig.add_trace(go.Scatter(x=bos_h.index, y=bos_h['High'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='#FFD700', line=dict(width=1, color='black')), name="Struct High"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bos_l.index, y=bos_l['Low'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#FFD700', line=dict(width=1, color='black')), name="Struct Low"), row=1, col=1)

        # Panel 2: CHEDO
        fig.add_hrect(y0=0.8, y1=1.2, row=2, col=1, fillcolor="#FF0055", opacity=0.15, line_width=0)
        fig.add_hrect(y0=-1.2, y1=-0.8, row=2, col=1, fillcolor="#00F0FF", opacity=0.15, line_width=0)
        fig.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], line=dict(color='#00F0FF', width=2), fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.05)', name="Entropy"), row=2, col=1)

        # Panel 3: RQZO
        fig.add_trace(go.Scatter(x=df.index, y=df['RQZO'], line=dict(color='#D500F9', width=1.5), fill='tozeroy', fillcolor='rgba(213, 0, 249, 0.05)', name="Relativity"), row=3, col=1)
        fig.add_hline(y=0, line_color="#333", line_width=1, row=3, col=1)

        # Panel 4: Apex Flux
        colors = np.where(df['Apex_Flux'] > 0.6, '#00E676', np.where(df['Apex_Flux'] < -0.6, '#FF1744', np.where(df['Apex_Flux'] > 0, '#2979FF', '#B0BEC5')))
        fig.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], marker_color=colors, marker_line_width=0, name="Flux"), row=4, col=1)
        fig.add_hrect(y0=0.6, y1=1.0, row=4, col=1, fillcolor="#00E676", opacity=0.05, line_width=0)
        fig.add_hrect(y0=-1.0, y1=-0.6, row=4, col=1, fillcolor="#FF1744", opacity=0.05, line_width=0)

        fig.update_layout(height=total_height, template="plotly_dark", paper_bgcolor="#050505", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=40 if not is_mobile else 0, t=20, b=0), hovermode="x unified", showlegend=False, font=dict(family="SF Pro Display, Roboto Mono, sans-serif", size=12))
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)', zeroline=False)
        fig.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 7. MAIN APP
# ==========================================
def main():
    st.sidebar.markdown("## 💠 AXIOM")
    is_mobile = st.sidebar.toggle("📱 Mobile Optimized", value=False)
    inject_axiom_css(is_mobile)
    
    # --- CREDENTIALS MANAGEMENT ---
    with st.sidebar.expander("🔐 API Credentials", expanded=False):
        openai_key = st.text_input("OpenAI Key", type="password", help="For AI Analysis")
        tg_token = st.text_input("Telegram Token", type="password", help="For Broadcast")
        tg_chat = st.text_input("Telegram Chat ID", help="For Broadcast")
    
    # Resolving Keys (Secrets vs Manual)
    final_openai_key = Intelligence.check_ai_key(openai_key)
    final_tg_token = tg_token if tg_token else st.secrets.get("TELEGRAM_TOKEN", None)
    final_tg_chat = tg_chat if tg_chat else st.secrets.get("TELEGRAM_CHAT_ID", None)

    # --- ASSETS ---
    st.sidebar.markdown("### Market Data")
    asset_class = st.sidebar.selectbox("Sector", DataService.get_asset_classes())
    ticker = st.sidebar.selectbox("Ticker", DataService.get_tickers_by_class(asset_class))
    tf = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=2)
    
    with st.spinner("Calculating Quantum States..."):
        df = DataService.fetch_data(ticker, tf)
        if df.empty:
            st.error("Market Data Unavailable.")
            return
        try:
            df = QuantEngine.calc_chedo(df)
            df = QuantEngine.calc_rqzo(df)
            df = QuantEngine.calc_apex_flux(df)
            df = QuantEngine.calc_smc(df)
            last = df.iloc[-1]
        except Exception as e:
            st.error(f"Engine Failure: {e}")
            return

    # --- DASHBOARD ---
    if is_mobile:
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
    else:
        c1, c2, c3, c4 = st.columns(4)

    c1.metric("Price", f"{last['Close']:.2f}")
    c2.metric("CHEDO", f"{last['CHEDO']:.2f}", delta="Risk" if abs(last['CHEDO'])>0.7 else "Stable")
    c3.metric("RQZO", f"{last['RQZO']:.2f}")
    c4.metric("Flux", f"{last['Apex_Flux']:.2f}", delta=last['Apex_State'])

    t1, t2, t3, t4 = st.tabs(["📉 Tech", "🧠 AI", "📈 View", "📡 Msg"])
    
    with t1:
        Graphics.render_dashboard(df, ticker, is_mobile)
        
    with t2:
        if st.button("Run Intelligence", use_container_width=is_mobile):
            res = Intelligence.analyze(df, ticker, final_openai_key)
            st.markdown(res)
            
    with t3:
        # TradingView with Drawing Tools
        h = 500 if is_mobile else 650
        tv_sym = ticker.replace("-", "").replace("=X", "").replace("=F", "")
        html = f"""
        <div class="tradingview-widget-container">
          <div id="tradingview_widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
            "width": "100%", "height": {h}, "symbol": "{tv_sym}",
            "interval": "D", "timezone": "Etc/UTC", "theme": "dark",
            "style": "1", "locale": "en", "toolbar_bg": "#f1f3f6",
            "enable_publishing": false, "hide_side_toolbar": false,
            "allow_symbol_change": true, "details": true,
            "container_id": "tradingview_widget"
          }}
          );
          </script>
        </div>
        """
        st.components.v1.html(html, height=h)

    with t4:
        msg = st.text_area("Broadcast Payload", f"AXIOM SIGNAL: {ticker}\nFlux: {last['Apex_State']}\nEntropy: {last['CHEDO']:.2f}")
        if st.button("🚀 Broadcast to Telegram", use_container_width=is_mobile):
            success, info = Intelligence.broadcast_telegram(msg, final_tg_token, final_tg_chat)
            if success: st.success(info)
            else: st.error(info)

if __name__ == "__main__":
    main()
