import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from openai import OpenAI
import streamlit.components.v1 as components
from datetime import datetime

# ==========================================
# 1. CORE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AXIOM OMNI: Unified Field",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. UI/UX ENGINE (DarkPool/Neon Aesthetic)
# ==========================================
def inject_omni_css(is_mobile):
    base_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=SF+Pro+Display:wght@300;500;700&display=swap');
        
        .stApp { background-color: #050505; color: #e0e0e0; font-family: 'SF Pro Display', sans-serif; }
        
        /* METRICS: Glassmorphism & Neon Borders */
        div[data-testid="stMetric"] {
            background: rgba(10, 10, 10, 0.8);
            border-left: 2px solid #333;
            backdrop-filter: blur(10px);
            padding: 15px;
            transition: all 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-left: 2px solid #00F0FF;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.1);
        }
        div[data-testid="stMetricLabel"] { color: #888; letter-spacing: 1px; text-transform: uppercase; font-size: 0.75rem; }
        div[data-testid="stMetricValue"] { color: #fff; font-weight: 300; text-shadow: 0 0 5px rgba(255,255,255,0.2); }
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] { gap: 5px; background-color: transparent; border-bottom: 1px solid #222; }
        .stTabs [data-baseweb="tab"] { background-color: transparent; border: none; color: #666; font-size: 0.9rem; }
        .stTabs [aria-selected="true"] { color: #00F0FF; border-bottom: 2px solid #00F0FF; }
        
        /* INPUTS */
        .stTextInput > div > div > input { color: #00F0FF; background-color: #0a0a0a; border: 1px solid #333; }
        
        /* REPORT CARDS */
        .omni-card {
            background: #0e0e0e;
            border: 1px solid #222;
            border-left: 3px solid #7C4DFF;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 0 5px 5px 0;
        }
        .omni-header { color: #fff; font-weight: bold; font-size: 1.1em; margin-bottom: 5px; border-bottom: 1px solid #333; padding-bottom: 5px; }
        .omni-row { display: flex; justify-content: space-between; font-size: 0.9em; margin-bottom: 3px; color: #aaa; }
        .omni-val { color: #00F0FF; font-family: 'Roboto Mono'; }
        .omni-val-bear { color: #FF1744; font-family: 'Roboto Mono'; }
        
        /* CONFIDENCE BAR */
        .conf-bar-bg { width: 100%; background: #222; height: 6px; border-radius: 3px; margin-top: 5px; }
        .conf-bar-fill { height: 100%; border-radius: 3px; box-shadow: 0 0 5px currentColor; }
    </style>
    """
    st.markdown(base_css, unsafe_allow_html=True)

def render_clock():
    html = """
    <div style="display: flex; gap: 20px; font-family: 'Roboto Mono'; font-size: 0.8rem; color: #666; margin-bottom: 10px;">
        <div>NY: <span id="ny" style="color: #00F0FF;">--:--</span></div>
        <div>LON: <span id="lon" style="color: #FF0055;">--:--</span></div>
        <div>TOK: <span id="tok" style="color: #D500F9;">--:--</span></div>
    </div>
    <script>
    function u() {
        const d = new Date();
        const f = (z) => d.toLocaleTimeString('en-US', {timeZone: z, hour12: false, hour:'2-digit', minute:'2-digit'});
        document.getElementById('ny').innerText = f('America/New_York');
        document.getElementById('lon').innerText = f('Europe/London');
        document.getElementById('tok').innerText = f('Asia/Tokyo');
    }
    setInterval(u, 1000); u();
    </script>
    """
    components.html(html, height=30)

# ==========================================
# 3. DATA & UNIFIED PHYSICS ENGINE
# ==========================================
class OmniMath:
    @staticmethod
    def get_data(ticker, interval, period="1y"):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df.empty: return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs(ticker, axis=1, level=0)
                except: df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={c: c.capitalize() for c in df.columns})
            if 'Adj close' in df.columns: df['Close'] = df['Adj close']
            return df.dropna()
        except Exception as e:
            st.error(f"Data Feed Error: {e}")
            return pd.DataFrame()

    # --- FROM NEXUS: RATIONAL QUADRATIC KERNEL ---
    @staticmethod
    def calc_kernel_regression(df, lookback=50, weight=8.0):
        # Python implementation of the Pine Script Rational Quadratic Kernel
        # Uses a windowed approach for performance approximation
        src = df['Close'].values
        n = len(src)
        kernel_line = np.zeros(n)
        
        # Optimization: Use indices to simulate the loop
        # We calculate the last 'lookback' points more precisely
        for i in range(lookback, n):
            current_weight = 0.0
            cumulative_weight = 0.0
            for j in range(lookback + 1):
                y_val = src[i - j]
                # Formula: (1 + (j^2 / (2 * weight * lookback^2))) ^ -weight
                w_val = np.power(1 + (np.power(j, 2) / (2 * weight * lookback * lookback)), -weight)
                current_weight += y_val * w_val
                cumulative_weight += w_val
            kernel_line[i] = current_weight / cumulative_weight
            
        df['Kernel'] = kernel_line
        df['Kernel_Slope'] = np.where(df['Kernel'] > df['Kernel'].shift(1), 1, -1)
        return df

    # --- FROM NEXUS: GANN BREAKOUT ---
    @staticmethod
    def calc_gann_activator(df, length=20):
        df['Donch_High'] = df['High'].rolling(length).max()
        df['Donch_Low'] = df['Low'].rolling(length).min()
        
        # Iterative logic required for state persistence
        trend = np.zeros(len(df))
        activator = np.zeros(len(df))
        close = df['Close'].values
        d_high = df['Donch_High'].values
        d_low = df['Donch_Low'].values
        
        curr_trend = 1
        curr_act = d_low[0]
        
        for i in range(1, len(df)):
            if curr_trend == 1:
                if close[i] < curr_act:
                    curr_trend = -1
                    curr_act = d_high[i]
                else:
                    curr_act = d_low[i]
            else:
                if close[i] > curr_act:
                    curr_trend = 1
                    curr_act = d_low[i]
                else:
                    curr_act = d_high[i]
            trend[i] = curr_trend
            activator[i] = curr_act
            
        df['Gann_Trend'] = trend
        df['Gann_Activator'] = activator
        return df

    # --- FROM QUANTUM APEX V7.4: PHYSICS VECTOR ---
    @staticmethod
    def calc_apex_vector(df, length=14):
        # Efficiency: Body / Range
        rg = df['High'] - df['Low']
        body = np.abs(df['Close'] - df['Open'])
        eff = np.where(rg == 0, 0, body / rg)
        eff_sm = pd.Series(eff).ewm(span=length).mean()
        
        # Volume Factor: Vol / AvgVol
        vol_avg = df['Volume'].rolling(55).mean()
        vol_fact = np.where(vol_avg == 0, 1.0, df['Volume'] / vol_avg)
        
        # Direction
        direction = np.sign(df['Close'] - df['Open'])
        
        # Vector = Direction * Efficiency * VolumeFactor
        raw_vec = direction * eff_sm * vol_fact
        df['Apex_Vector'] = pd.Series(raw_vec).ewm(span=5).mean()
        
        # States
        df['Apex_State'] = np.select(
            [df['Apex_Vector'] > 0.6, df['Apex_Vector'] < -0.6, abs(df['Apex_Vector']) < 0.3],
            [2, -2, 0], default=1 # 2=SuperBull, -2=SuperBear, 0=Resistive, 1=Heat
        )
        return df

    # --- FROM AXIOM: CHEDO (ENTROPY) ---
    @staticmethod
    def calc_chedo(df, length=50):
        c = df['Close'].values
        log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
        mu = pd.Series(log_ret).rolling(length).mean().values
        sigma = pd.Series(log_ret).rolling(length).std().values
        v = sigma / (np.abs(mu) + 1e-9)
        abs_ret_v = np.abs(log_ret) * v
        hyper_dist = np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))
        kappa_h = np.tanh(pd.Series(hyper_dist).rolling(length).mean().values)
        raw = (0.4 * kappa_h) # Simplified for speed
        df['CHEDO'] = 2 / (1 + np.exp(-raw * 4)) - 1
        return df

    # --- UNIFIED: SMC (WORMHOLES & EVENT HORIZONS) ---
    @staticmethod
    def calc_smc_unified(df):
        # FVGs (Wormholes)
        df['FVG_Bull'] = (df['Low'] > df['High'].shift(2)) & (df['Close'] > df['Open'])
        df['FVG_Bear'] = (df['High'] < df['Low'].shift(2)) & (df['Close'] < df['Open'])
        
        # Pivots (for BOS)
        window = 5
        df['Pivot_H'] = df['High'].rolling(window*2+1, center=True).max() == df['High']
        df['Pivot_L'] = df['Low'].rolling(window*2+1, center=True).min() == df['Low']
        
        return df

    # --- THE OMNI BRAIN (CONFIDENCE SCORING) ---
    @staticmethod
    def calc_omni_brain(df):
        # Weights
        w_trend = 0.30 # Nexus Kernel + Gann
        w_phys = 0.30  # Apex Vector
        w_entr = 0.20  # Axiom Entropy
        w_smc = 0.20   # SMC
        
        last = df.iloc[-1]
        
        # 1. Trend Score
        score_trend = 0
        if last['Kernel_Slope'] == 1: score_trend += 0.5
        if last['Gann_Trend'] == 1: score_trend += 0.5
        if last['Kernel_Slope'] == -1: score_trend -= 0.5
        if last['Gann_Trend'] == -1: score_trend -= 0.5
        
        # 2. Physics Score
        score_phys = np.clip(last['Apex_Vector'], -1, 1)
        
        # 3. Entropy Score (Stability favors trend)
        score_entr = 0
        if abs(last['CHEDO']) < 0.7: # Stable regime
             score_entr = 1 if score_trend > 0 else -1
        else: # Chaos regime
             score_entr = 0 
             
        # 4. SMC Score (Proximity to breaks)
        score_smc = 0 # Simplified: Assuming trend following
        if last['Close'] > df['High'].rolling(20).max().iloc[-2]: score_smc = 1
        if last['Close'] < df['Low'].rolling(20).min().iloc[-2]: score_smc = -1
        
        # Aggregate
        total_bull = (max(0, score_trend)*w_trend) + (max(0, score_phys)*w_phys) + (max(0, score_entr)*w_entr) + (max(0, score_smc)*w_smc)
        total_bear = (max(0, -score_trend)*w_trend) + (max(0, -score_phys)*w_phys) + (max(0, -score_entr)*w_entr) + (max(0, -score_smc)*w_smc)
        
        # Normalize to 0-100
        df['Conf_Bull'] = total_bull * 100
        df['Conf_Bear'] = total_bear * 100
        
        return df

# ==========================================
# 4. INTELLIGENCE & REPORTING
# ==========================================
class OmniIntel:
    @staticmethod
    def generate_signal_card(df, ticker):
        last = df.iloc[-1]
        conf_b = last['Conf_Bull']
        conf_s = last['Conf_Bear']
        
        # Determine State
        if conf_b > 65 and conf_b > conf_s: status, color = "OMNI BUY", "#00E676"
        elif conf_s > 65 and conf_s > conf_b: status, color = "OMNI SELL", "#FF1744"
        else: status, color = "NEUTRAL / CHOP", "#777"
        
        # Physics Description
        vec = last['Apex_Vector']
        if vec > 0.6: phys_desc = "Superconductor (Low Resistance)"
        elif vec < -0.6: phys_desc = "Superconductor (Free Fall)"
        elif abs(vec) < 0.3: phys_desc = "Resistive (High Friction)"
        else: phys_desc = "Standard Flow"
        
        html = f"""
        <div class="omni-card" style="border-left: 4px solid {color};">
            <div class="omni-header">{status} <span style="float:right; font-size:0.8em;">{ticker}</span></div>
            <div class="omni-row">
                <span>Confidence (Bull)</span>
                <span class="omni-val">{conf_b:.1f}%</span>
            </div>
            <div class="omni-row">
                <span>Confidence (Bear)</span>
                <span class="omni-val-bear">{conf_s:.1f}%</span>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width: {max(conf_b, conf_s)}%; background: {color};"></div>
            </div>
            <hr style="border-color: #333; margin: 10px 0;">
            <div class="omni-row"><span>Physics:</span> <span style="color:#fff;">{phys_desc}</span></div>
            <div class="omni-row"><span>Entropy:</span> <span style="color:{'#FF1744' if abs(last['CHEDO'])>0.8 else '#00E676'};">{last['CHEDO']:.2f}</span></div>
            <div class="omni-row"><span>Kernel Trend:</span> <span style="color:{'#00E676' if last['Kernel_Slope']>0 else '#FF1744'};">{'RISING' if last['Kernel_Slope']>0 else 'FALLING'}</span></div>
        </div>
        """
        return html

    @staticmethod
    def get_ai_analysis(df, ticker, api_key):
        if not api_key: return "⚠️ API Key Required"
        last = df.iloc[-1]
        prompt = f"""
        Act as 'Axiom-Omni', a hedge fund algo. Analyze {ticker}.
        Data:
        - Price: {last['Close']}
        - Apex Vector (Momentum): {last['Apex_Vector']:.2f} (Abs >0.6 is Superconductor)
        - Nexus Kernel (Trend): {'Bull' if last['Kernel_Slope']>0 else 'Bear'}
        - CHEDO (Entropy): {last['CHEDO']:.2f} (Abs >0.8 is Chaos)
        - Unified Confidence: Bull {last['Conf_Bull']:.0f}% / Bear {last['Conf_Bear']:.0f}%
        
        Output formatted markdown:
        1. **Regime Identification**: (Trending vs Chopping vs Reversing)
        2. **Physics Check**: Is price moving efficiently?
        3. **Strategic Action**: Recommendation based on Confidence scores.
        """
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content
        except Exception as e: return f"AI Error: {e}"

# ==========================================
# 5. VISUALIZATION (PLOTLY)
# ==========================================
def render_omni_chart(df, ticker, show_fvg, show_kernel):
    # 4 Rows: Price, Apex Vector, Entropy, Confidence
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02, 
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # 1. Price Chart (Candles + Kernel + Gann + SMC)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price",
        increasing_line_color='#00F0FF', decreasing_line_color='#FF1744'
    ), row=1, col=1)
    
    if show_kernel:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Kernel'], 
            line=dict(color='#FFD700', width=2), name="Nexus Kernel"
        ), row=1, col=1)
        
    # Gann Staircase
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Gann_Activator'],
        line=dict(color='white', width=1, dash='dot'), name="Gann Risk Line"
    ), row=1, col=1)

    # FVGs (Wormholes) - Plotting rectangles as shapes is heavy, using markers for speed
    if show_fvg:
        fvg_b = df[df['FVG_Bull']]
        fig.add_trace(go.Scatter(
            x=fvg_b.index, y=fvg_b['Low'], mode='markers',
            marker=dict(symbol='line-ew', line_width=15, line_color='rgba(0, 230, 118, 0.3)', size=10),
            name="Bull FVG"
        ), row=1, col=1)

    # 2. Apex Vector
    colors = np.where(df['Apex_Vector'] > 0.6, '#00E676', 
             np.where(df['Apex_Vector'] < -0.6, '#FF1744', 
             np.where(abs(df['Apex_Vector']) < 0.3, '#546E7A', '#FFD600')))
    fig.add_trace(go.Bar(
        x=df.index, y=df['Apex_Vector'], 
        marker_color=colors, name="Apex Vector"
    ), row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dot", row=2, col=1, line_color="#333")
    fig.add_hline(y=-0.6, line_dash="dot", row=2, col=1, line_color="#333")

    # 3. CHEDO (Entropy)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['CHEDO'], 
        line=dict(color='#D500F9', width=1.5), fill='tozeroy', 
        fillcolor='rgba(213, 0, 249, 0.1)', name="Entropy"
    ), row=3, col=1)
    
    # 4. Omni Confidence
    fig.add_trace(go.Scatter(x=df.index, y=df['Conf_Bull'], line=dict(color='#00E676', width=1), name="Bull Conf"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Conf_Bear'], line=dict(color='#FF1744', width=1), name="Bear Conf"), row=4, col=1)
    fig.add_hline(y=70, line_color="white", line_dash="dot", opacity=0.3, row=4, col=1)

    fig.update_layout(
        height=900, 
        paper_bgcolor='#050505', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="SF Pro Display, sans-serif", size=11, color="#aaa"),
        margin=dict(l=10, r=40, t=10, b=10),
        showlegend=False,
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=False, rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. MAIN APP LOGIC
# ==========================================
def main():
    st.sidebar.title("💠 AXIOM OMNI")
    st.sidebar.caption("Unified Field: Nexus x Apex x Axiom")
    
    # Settings
    is_mobile = st.sidebar.checkbox("Mobile Optimized", value=False)
    inject_omni_css(is_mobile)
    
    # Ticker Selection
    asset_class = st.sidebar.selectbox("Market", ["Crypto", "Indices", "Tech", "Forex"])
    
    tickers = {
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"],
        "Indices": ["SPY", "QQQ", "IWM", "UVXY"],
        "Tech": ["NVDA", "TSLA", "AAPL", "MSFT"],
        "Forex": ["EURUSD=X", "JPY=X", "GBPUSD=X"]
    }
    
    ticker = st.sidebar.selectbox("Asset", tickers[asset_class])
    tf = st.sidebar.select_slider("Timeframe", options=["15m", "1h", "4h", "1d"], value="4h")
    
    # Parameters
    with st.sidebar.expander("⚙️ Physics Parameters"):
        k_look = st.slider("Nexus Kernel Lookback", 20, 100, 50)
        v_look = st.slider("Apex Vector Lookback", 5, 50, 14)
        show_fvg = st.toggle("Show Wormholes (FVG)", True)
        show_kernel = st.toggle("Show Kernel Line", True)
        
    api_key = st.sidebar.text_input("OpenAI Key", type="password")

    # Header
    c1, c2 = st.columns([3, 1])
    with c1: st.title(f"{ticker} [{tf}]")
    with c2: render_clock()

    # Data Processing
    with st.spinner("Calculating Unified Field Physics..."):
        df = OmniMath.get_data(ticker, tf)
        if not df.empty:
            # 1. Nexus Math
            df = OmniMath.calc_kernel_regression(df, lookback=k_look)
            df = OmniMath.calc_gann_activator(df)
            # 2. Apex Math
            df = OmniMath.calc_apex_vector(df, length=v_look)
            df = OmniMath.calc_smc_unified(df)
            # 3. Axiom Math
            df = OmniMath.calc_chedo(df)
            # 4. Integration
            df = OmniMath.calc_omni_brain(df)
        else:
            st.error("Data Unavailable")
            st.stop()

    # Dashboard Layout
    col_l, col_r = st.columns([1, 3] if not is_mobile else [1, 1])
    
    with col_l:
        # Signal Card
        st.markdown(OmniIntel.generate_signal_card(df, ticker), unsafe_allow_html=True)
        
        # Metrics
        last = df.iloc[-1]
        c_trend = "#00E676" if last['Gann_Trend'] == 1 else "#FF1744"
        st.metric("Gann Trend", "BULL" if last['Gann_Trend']==1 else "BEAR")
        st.metric("Flux Vector", f"{last['Apex_Vector']:.2f}", delta="Superconductor" if abs(last['Apex_Vector'])>0.6 else "Normal")
        
        if st.button("🧠 AI Analysis"):
            analysis = OmniIntel.get_ai_analysis(df, ticker, api_key)
            st.info(analysis)

    with col_r:
        # Chart
        render_omni_chart(df, ticker, show_fvg, show_kernel)

    # Raw Data Expander
    with st.expander("View Quantum Dataframe"):
        st.dataframe(df.tail(10).style.format("{:.2f}"))

if __name__ == "__main__":
    main()
