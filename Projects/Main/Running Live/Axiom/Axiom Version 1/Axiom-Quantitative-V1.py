
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
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
    """
    Injects dynamic CSS based on the Mobile Toggle state.
    """
    base_css = """
    <style>
        /* GLOBAL THEME */
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=SF+Pro+Display:wght@300;500;700&display=swap');
        
        .stApp { background-color: #050505; color: #e0e0e0; font-family: 'SF Pro Display', sans-serif; }
        
        /* METRICS - GLASSMORPHISM */
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
        /* FORCE TOUCH OPTIMIZATION */
        div[data-testid="stMetric"] {
            margin-bottom: 10px;
            border-left: 4px solid #00F0FF; /* Thicker accent for visibility */
            padding: 20px; /* Larger touch target */
        }
        div[data-testid="stMetricLabel"] { font-size: 0.9rem; font-weight: 700; }
        div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
        
        /* FORCE SINGLE COLUMN BEHAVIOR FOR PLOTLY IF NEEDED */
        .js-plotly-plot { height: 100% !important; }
        
        /* INCREASE BUTTON SIZES */
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
    @st.cache_data(ttl=300)
    def get_universe():
        """Institutional Ticker Universe."""
        return [
            "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD",  # Digital Assets
            "SPY", "QQQ", "IWM", "NVDA", "MSFT", "TSLA", # Equities
            "GLD", "SLV", "USO", "TLT", "VIX"            # Macro/Rates
        ]

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
# 4. QUANTITATIVE ENGINE (PHYSICS)
# ==========================================
class QuantEngine:
    """Core Physics Layer: CHEDO, RQZO, Apex, SMC."""
    
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
        """
        Apex Vector v4.1 (Efficiency * Volume Flux)
        FIX: Explicit Index Alignment enforced to prevent pandas broadcasting errors.
        """
        # 1. Efficiency
        rg = df['High'] - df['Low']
        body = np.abs(df['Close'] - df['Open'])
        eff_raw = np.where(rg == 0, 0, body / rg)
        
        # Enforce Index Alignment with df
        eff_series = pd.Series(eff_raw, index=df.index) 
        eff_sm = eff_series.ewm(span=length).mean()
        
        # 2. Volume Flux
        vol_avg = df['Volume'].rolling(55).mean()
        v_rat_raw = np.where(vol_avg == 0, 1, df['Volume'] / vol_avg)
        
        # Enforce Index Alignment with df
        v_rat_series = pd.Series(v_rat_raw, index=df.index)
        
        # 3. Vector Direction
        direction = np.sign(df['Close'] - df['Open'])
        
        # 4. Calculation (Now safe with aligned indices)
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
# 5. INTELLIGENCE (AI)
# ==========================================
class Intelligence:
    @staticmethod
    def analyze(df, ticker):
        if "OPENAI_API_KEY" not in st.secrets:
            return "⚠️ OpenAI Key Missing. Configure secrets."
        
        last = df.iloc[-1]
        system_prompt = "You are Axiom, a physics-based quantitative trading assistant."
        user_prompt = f"Analyze {ticker}. Price: {last['Close']:.2f}. CHEDO: {last['CHEDO']:.2f}. RQZO: {last['RQZO']:.2f}. Flux: {last['Apex_Flux']:.2f}."
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# ==========================================
# 6. GRAPHICS (PLOTLY)
# ==========================================
class Graphics:
    @staticmethod
    def render_dashboard(df, ticker, is_mobile):
        # Adjust Heights/Spacing for Mobile
        row_heights = [0.4, 0.2, 0.2, 0.2] if not is_mobile else [0.5, 0.15, 0.15, 0.2]
        total_height = 800 if not is_mobile else 1200 # Taller on mobile for scrolling
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05 if is_mobile else 0.02,
            row_heights=row_heights,
            specs=[[{"secondary_y": False}], [{}], [{}], [{}]],
            subplot_titles=("", "Entropy (CHEDO)", "Relativity (RQZO)", "Flux Vector")
        )

        # 1. Price + SMC
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Price",
            increasing_line_color='#00F0FF', decreasing_line_color='#FF0055'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA_Trend'], line=dict(color='white', width=1), name="SMC"), row=1, col=1)
        bos = df[df['Pivot_H']]
        fig.add_trace(go.Scatter(x=bos.index, y=bos['High'], mode='markers', marker=dict(symbol='triangle-down', size=8, color='#FFD700'), name="BOS"), row=1, col=1)

        # 2. CHEDO
        fig.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], line=dict(color='#00F0FF'), fill='tozeroy', fillcolor='rgba(0,240,255,0.1)', name="CHEDO"), row=2, col=1)
        fig.add_hrect(y0=0.8, y1=1.0, row=2, col=1, fillcolor="#FF0055", opacity=0.1, line_width=0)

        # 3. RQZO
        fig.add_trace(go.Scatter(x=df.index, y=df['RQZO'], line=dict(color='#9C27B0'), name="RQZO"), row=3, col=1)

        # 4. Flux
        colors = np.where(df['Apex_Flux'] > 0.6, '#00F0FF', np.where(df['Apex_Flux'] < -0.6, '#FF0055', '#333333'))
        fig.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], marker_color=colors, name="Flux"), row=4, col=1)

        # Layout
        fig.update_layout(
            height=total_height,
            template="plotly_dark",
            paper_bgcolor="#050505",
            plot_bgcolor="rgba(255,255,255,0.02)",
            margin=dict(l=0, r=40 if not is_mobile else 10, t=20, b=0), # Tighter margins on mobile
            xaxis_rangeslider_visible=False,
            showlegend=False,
            font=dict(family="SF Pro Display, sans-serif", size=14 if is_mobile else 12)
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 7. APP ORCHESTRATION
# ==========================================
def main():
    # --- BRAND SIDEBAR ---
    st.sidebar.markdown("## 💠 AXIOM")
    st.sidebar.markdown("---")
    
    # --- TOGGLE: MOBILE INDUCTION ---
    # This forces the layout changes
    is_mobile = st.sidebar.toggle("📱 Mobile Optimized", value=False)
    
    # Inject CSS based on toggle
    inject_axiom_css(is_mobile)
    
    # Controls
    ticker = st.sidebar.selectbox("Asset", DataService.get_universe())
    tf = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=2)
    
    # Processing
    with st.spinner("Processing..."):
        df = DataService.fetch_data(ticker, tf)
        if df.empty:
            st.error("Signal Lost.")
            return
        
        # Calculations (Order matters for safety)
        try:
            df = QuantEngine.calc_chedo(df)
            df = QuantEngine.calc_rqzo(df)
            df = QuantEngine.calc_apex_flux(df)
            df = QuantEngine.calc_smc(df)
            last = df.iloc[-1]
        except Exception as e:
            st.error(f"Computation Error: {e}")
            return

    # --- RESPONSIVE LAYOUT LOGIC ---
    if is_mobile:
        # 2x2 Grid for Mobile (Larger Cards)
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
    else:
        # 4x1 Row for Desktop
        c1, c2, c3, c4 = st.columns(4)

    c1.metric("Price", f"{last['Close']:.2f}")
    c2.metric("CHEDO", f"{last['CHEDO']:.2f}", delta="Risk" if abs(last['CHEDO'])>0.7 else "Stable")
    c3.metric("RQZO", f"{last['RQZO']:.2f}")
    c4.metric("Flux", f"{last['Apex_Flux']:.2f}", delta=last['Apex_State'])

    # Tabs
    t1, t2, t3, t4 = st.tabs(["📉 Tech", "🧠 AI", "📈 View", "📡 Msg"])
    
    with t1:
        Graphics.render_dashboard(df, ticker, is_mobile)
        
    with t2:
        if st.button("Run Intelligence", use_container_width=is_mobile):
            res = Intelligence.analyze(df, ticker)
            st.markdown(res)
            
    with t3:
        # TradingView Embed (Responsive Height)
        h = 400 if is_mobile else 600
        tv_sym = ticker.replace("-", "")
        html = f"""
        <div class="tradingview-widget-container">
          <div id="tradingview_widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{ "width": "100%", "height": {h}, "symbol": "{tv_sym}", "interval": "D", "theme": "dark", "container_id": "tradingview_widget" }}
          );
          </script>
        </div>
        """
        st.components.v1.html(html, height=h)

    with t4:
        msg = st.text_area("Payload", f"AXIOM: {ticker} | Flux: {last['Apex_State']}")
        if st.button("Broadcast", use_container_width=is_mobile):
            st.success("Sent.")

if __name__ == "__main__":
    main()
