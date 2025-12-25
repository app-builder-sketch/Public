import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
import requests
from datetime import datetime

# ==========================================
# 1. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="TITAN ZEN", page_icon="🧿", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;900&display=swap');
    
    .stApp { background-color: #030303; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    
    /* MINIMAL HEADER */
    .titan-header {
        font-family: 'Rajdhani', sans-serif; font-size: 2rem; font-weight: 700;
        color: #fff; letter-spacing: 2px;
        border-bottom: 1px solid #222; padding-bottom: 10px; margin-bottom: 15px;
    }
    .titan-accent { color: #00f260; }
    
    /* HUD METRICS */
    div[data-testid="stMetric"] {
        background: #080808; border: 1px solid #222; 
        border-left: 3px solid #00f260; border-radius: 4px; 
    }
    div[data-testid="stMetricValue"] { font-family: 'Rajdhani'; font-size: 1.4rem !important; color: #fff; }
    div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #555; }

    /* CLEAN TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; padding: 5px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #555; font-size: 0.75rem; font-weight: 700; border: none; }
    .stTabs [aria-selected="true"] { color: #00f260; border-bottom: 1px solid #00f260; }
    
    /* INPUTS */
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        background-color: #080808; color: #fff; border: 1px solid #222;
    }
    button[kind="primary"] { background: #00f260; border: none; color: black; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# SECRETS
api_key = st.secrets.get("OPENAI_API_KEY")
tg_token = st.secrets.get("TELEGRAM_TOKEN")
tg_chat = st.secrets.get("TELEGRAM_CHAT_ID")

# ==========================================
# 2. PHYSICS ENGINE (OPTIMIZED)
# ==========================================
class TitanEngine:
    @staticmethod
    def _hma(series, length):
        half = int(length / 2); sqrt = int(np.sqrt(length))
        wma_half = series.rolling(half).apply(lambda x: np.dot(x, np.arange(1, half+1)) / (half*(half+1)/2), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / (length*(length+1)/2), raw=True)
        return (2 * wma_half - wma_full).rolling(sqrt).apply(lambda x: np.dot(x, np.arange(1, sqrt+1)) / (sqrt*(sqrt+1)/2), raw=True)

    @staticmethod
    def process_data(df):
        df = df.copy()
        # Physics
        vel = df['Close'].pct_change().abs(); c = 0.05
        gamma = 1 / np.sqrt(1 - (np.minimum(vel, c*0.99)/c)**2)
        entropy = (df['Close'].pct_change().rolling(20).std() - 0) / 0.05 
        zeta = np.sin((np.arange(len(df))%100)/gamma) * np.exp(-2*np.abs(entropy-0.6))*10
        df['RQZO'] = zeta.fillna(0); df['Entropy'] = entropy.fillna(0.5)

        # Flux
        eff = ((df['Close']-df['Open']).abs()/(df['High']-df['Low']).replace(0,1)).ewm(span=14).mean()
        flux = (np.sign(df['Close']-df['Open']) * eff * (df['Volume']/df['Volume'].rolling(55).mean())).ewm(span=5).mean()
        df['Flux'] = flux; df['State'] = np.select([flux>0.6, flux<-0.6, flux.abs()<0.3], ['Super_Bull', 'Super_Bear', 'Resistive'], 'Heat')

        # Trend
        hma = TitanEngine._hma(df['Close'], 55); atr = (df['High']-df['Low']).rolling(55).mean()
        df['Cloud_Upper'] = hma + atr*1.5; df['Cloud_Lower'] = hma - atr*1.5
        df['Trend'] = np.where(df['Close'] > df['Cloud_Upper'], 1, np.where(df['Close'] < df['Cloud_Lower'], -1, 0))
        df['Trend'] = df['Trend'].replace(0, method='ffill')
        
        # Pattern Detection (Simplified for Visuals)
        body = (df['Close'] - df['Open']).abs()
        wick_up = df['High'] - df[['Open', 'Close']].max(axis=1)
        wick_dn = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Pat_Bull'] = (wick_dn > body*2) & (wick_up < body) # Hammer
        df['Pat_Bear'] = (wick_up > body*2) & (wick_dn < body) # Shooting Star
        return df

    @staticmethod
    def scan_structures(df):
        # LIMIT STRUCTURES TO REDUCE CLUTTER
        structs = {'OB': [], 'FVG': []}; pivot = 10
        for i in range(pivot, len(df)):
            # OB
            if df['Trend'].iloc[i] == 1 and df['Close'].iloc[i] > df['High'].iloc[i-pivot]:
                ob = df.iloc[i-15:i][df.iloc[i-15:i]['Close'] < df.iloc[i-15:i]['Open']]
                if not ob.empty: structs['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'col': 'rgba(0, 242, 96, 0.2)', 'bor': '#00f260'})
            if df['Trend'].iloc[i] == -1 and df['Close'].iloc[i] < df['Low'].iloc[i-pivot]:
                ob = df.iloc[i-15:i][df.iloc[i-15:i]['Close'] > df.iloc[i-15:i]['Open']]
                if not ob.empty: structs['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'col': 'rgba(255, 0, 60, 0.2)', 'bor': '#ff003c'})
            # FVG
            if i > 2:
                if df['Low'].iloc[i-2] > df['High'].iloc[i]: 
                    structs['FVG'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i], 'y1': df['Low'].iloc[i-2], 'col': 'rgba(255, 204, 0, 0.1)', 'bor': '#ffcc00'})
                if df['High'].iloc[i-2] < df['Low'].iloc[i]:
                    structs['FVG'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i], 'col': 'rgba(255, 204, 0, 0.1)', 'bor': '#ffcc00'})
        return structs

    @staticmethod
    def get_vpvr(df):
        try:
            bins = np.linspace(df['Low'].min(), df['High'].max(), 50)
            df['Bin'] = pd.cut(df['Close'], bins=bins, include_lowest=True)
            vp = df.groupby('Bin', observed=True)['Volume'].sum().reset_index()
            return vp, vp.loc[vp['Volume'].idxmax(), df['Bin'].apply(lambda x: x.mid).astype(float)]
        except: return pd.DataFrame(), 0.0

    @staticmethod
    def run_monte_carlo(df, days=30, sims=50): # Reduced sims for speed
        last = df['Close'].iloc[-1]; vol = df['Close'].pct_change().std()
        sim_df = pd.DataFrame()
        for x in range(sims): sim_df[x] = [last * (1 + np.random.normal(0, vol))**d for d in range(days)]
        return sim_df

# ==========================================
# 3. ZEN RENDERER (CLEAN CHARTS)
# ==========================================
def render_zen_chart(df, structs, vp):
    last = df.iloc[-1]
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
        row_heights=[0.8, 0.2], specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # 1. VOLUME WATERMARK (Layer 0 - Back)
    # Very low opacity, scaled to secondary Y axis to sit at bottom
    vol_c = ['rgba(0, 242, 96, 0.15)' if c >= o else 'rgba(255, 0, 60, 0.15)' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_c, showlegend=False, hoverinfo='skip'), row=1, col=1, secondary_y=True)

    # 2. TREND CLOUD (Layer 1)
    c_fill = 'rgba(0, 242, 96, 0.05)' if last['Trend']==1 else 'rgba(255, 0, 60, 0.05)'
    fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Upper'], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Lower'], fill='tonexty', fillcolor=c_fill, line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)

    # 3. SMC SHAPES (Layer 2 - Only Last 5)
    # We slice [-5:] to prevent cluttering the chart with old history
    for ob in structs['OB'][-5:]: 
        fig.add_shape(type="rect", x0=ob['x0'], x1=df.index[-1], y0=ob['y0'], y1=ob['y1'], fillcolor=ob['col'], line=dict(color=ob['bor'], width=1), row=1, col=1)
    for fvg in structs['FVG'][-5:]:
        fig.add_shape(type="rect", x0=fvg['x0'], x1=fvg['x1'], y0=fvg['y0'], y1=fvg['y1'], fillcolor=fvg['col'], line=dict(color=fvg['bor'], width=0, dash='dot'), row=1, col=1)

    # 4. PRICE (Layer 3 - Front)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color='#00f260', increasing_fillcolor='#00f260',
        decreasing_line_color='#ff003c', decreasing_fillcolor='#ff003c', name="Price"
    ), row=1, col=1)

    # 5. PATTERN MARKERS (Discrete)
    pat_bull = df[df['Pat_Bull']]
    pat_bear = df[df['Pat_Bear']]
    fig.add_trace(go.Scatter(x=pat_bull.index, y=pat_bull['Low']*0.999, mode='markers', marker=dict(symbol='triangle-up', size=8, color='#00f260'), name="Hammer"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pat_bear.index, y=pat_bear['High']*1.001, mode='markers', marker=dict(symbol='triangle-down', size=8, color='#ff003c'), name="Shooting Star"), row=1, col=1)

    # 6. PHYSICS (Bottom Panel)
    fig.add_trace(go.Scatter(x=df.index, y=df['RQZO'], line=dict(color='#2979FF', width=1.5), name="RQZO"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Entropy'], line=dict(color='#651FFF', width=1), name="Entropy"), row=2, col=1)
    fig.add_hline(y=0.8, line_color="#ff003c", line_dash="dot", row=2, col=1)

    # 7. CLEAN LAYOUT
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#030303", plot_bgcolor="#080808", height=550, 
        margin=dict(l=0, r=40, t=10, b=10), 
        hovermode="x unified", 
        showlegend=False,
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False)), 
        yaxis=dict(showgrid=True, gridcolor="#222", side="right"), # Price on Right
        yaxis2=dict(showgrid=False, range=[0, df['Volume'].max()*6], visible=False), # Vol scale hidden
        yaxis3=dict(showgrid=True, gridcolor="#222", side="right") # Physics on Right
    )
    return fig

# ==========================================
# 4. MAIN APPLICATION
# ==========================================
def main():
    # HEADER
    st.markdown('<div class="titan-header">TITAN <span class="titan-accent">ZEN</span></div>', unsafe_allow_html=True)

    # CONFIG
    assets = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "GOLD": "GC=F", "SPX": "^SPX", "NVDA": "NVDA"}
    c1, c2 = st.columns([3, 1])
    sel_ast = c1.selectbox("ASSET", list(assets.keys()), label_visibility="collapsed")
    ticker = assets[sel_ast]
    interval = c2.selectbox("TF", ["15m", "1h", "4h", "1d"], index=1, label_visibility="collapsed")
    
    per_map = {"15m": "59d", "1h": "1y", "4h": "1y", "1d": "2y"}
    
    try:
        with st.spinner("CALCULATING VECTORS..."):
            df = yf.download(ticker, interval=interval, period=per_map[interval], progress=False)
            if df.empty: st.error("NO CONNECTION"); return
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # CORE LOGIC
            df = TitanEngine.process_data(df)
            structs = TitanEngine.scan_structures(df)
            vp, poc = TitanEngine.get_vpvr(df)
            mc_sim = TitanEngine.run_monte_carlo(df)
            last = df.iloc[-1]

        # HUD
        m1, m2 = st.columns(2)
        with m1:
            st.metric("STRUCTURE", "BULL" if last['Trend']==1 else "BEAR", f"{last['Cloud_Lower']:.2f}")
            st.metric("FLUX", last['State'].replace('_',' '), f"{last['Flux']:.2f}")
        with m2:
            st.metric("ENTROPY", "CHAOS" if last['Entropy']>0.8 else "STABLE", f"{last['Entropy']:.2f}")
            st.metric("POC", f"{poc:.2f}", "VOLUME")

        # TABS
        t1, t2, t3, t4 = st.tabs(["👁️ CHART", "🔬 INTEL", "🧠 AI", "📡 UPLINK"])

        with t1: 
            st.plotly_chart(render_zen_chart(df, structs, vp), use_container_width=True, config={'displayModeBar': False})

        with t2:
            st.caption("MONTE CARLO (30D)")
            fig_mc = go.Figure()
            for i in range(min(15, len(mc_sim.columns))):
                fig_mc.add_trace(go.Scatter(y=mc_sim[i], mode='lines', line=dict(color='rgba(255,255,255,0.05)'), showlegend=False))
            fig_mc.add_trace(go.Scatter(y=mc_sim.mean(axis=1), mode='lines', line=dict(color='#00f260', width=2), name="Mean"))
            fig_mc.update_layout(template="plotly_dark", height=200, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="#030303")
            st.plotly_chart(fig_mc, use_container_width=True)

        with t3:
            st.subheader("TACTICAL AI")
            if not api_key: st.warning("NO API KEY"); api_in = st.text_input("Key", type="password")
            else: api_in = api_key
            persona = st.selectbox("MODE", ["SCALPER", "SWING"])
            if st.button("ANALYZE", use_container_width=True):
                if api_in:
                    p = f"Role: {persona}. Asset: {ticker}. Trend: {last['Trend']}. Entropy: {last['Entropy']:.2f}. Flux: {last['State']}. Verdict?"
                    try:
                        client = OpenAI(api_key=api_in)
                        res = client.chat.completions.create(model="gpt-4", messages=[{"role":"user", "content":p}])
                        st.info(res.choices[0].message.content)
                    except Exception as e: st.error(str(e))

        with t4:
            st.subheader("SIGNAL UPLINK")
            if not tg_token: st.info("NO TELEGRAM CONFIG"); u_tok = st.text_input("Token"); u_chat = st.text_input("Chat")
            else: u_tok, u_chat = tg_token, tg_chat
            msg = f"{'🟢' if last['Trend']==1 else '🔴'} **TITAN:** {ticker}\nFlux: {last['State']}\nPrice: {last['Close']:.2f}"
            if st.button("SEND", type="primary", use_container_width=True):
                if u_tok and u_chat:
                    requests.post(f"https://api.telegram.org/bot{u_tok}/sendMessage", data={"chat_id": u_chat, "text": msg})
                    st.success("SENT")

    except Exception as e: st.error(f"HALT: {e}")

if __name__ == "__main__":
    main()
