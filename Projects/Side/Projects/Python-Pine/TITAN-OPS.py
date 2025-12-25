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
# 1. SINGULARITY UI CONFIG
# ==========================================
st.set_page_config(page_title="TITAN SINGULARITY", page_icon="🧿", layout="wide", initial_sidebar_state="collapsed")

# --- HOLOGRAPHIC CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;900&display=swap');
    
    .stApp { background-color: #030303; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    
    /* LIVE HEADER */
    .titan-header {
        font-family: 'Rajdhani', sans-serif; font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #00f260, #0575e6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        border-bottom: 2px solid #111; padding-bottom: 10px; margin-bottom: 15px;
    }
    
    /* MTF BIAS PILL */
    .mtf-pill {
        font-size: 0.8rem; padding: 4px 12px; border-radius: 12px; font-weight: bold;
        display: inline-block; margin-left: 10px; vertical-align: middle;
    }
    
    /* HUD METRICS */
    div[data-testid="stMetric"] {
        background: rgba(10, 15, 20, 0.95); border: 1px solid #1f2937; 
        border-left: 3px solid #0575e6; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    }
    div[data-testid="stMetricValue"] { font-family: 'Rajdhani'; font-size: 1.4rem !important; color: #fff; }
    div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #6b7280; letter-spacing: 1px; }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: #080808; padding: 5px; border-radius: 8px; border: 1px solid #222; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #666; font-size: 0.75rem; font-weight: 700; flex: 1; border: none; }
    .stTabs [aria-selected="true"] { background: #111; color: #00f260; border-radius: 4px; border-bottom: 2px solid #00f260; }
    
    /* BUTTONS */
    button[kind="primary"] { background: linear-gradient(90deg, #00f260, #0575e6); border: none; color: black; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# SECRETS
api_key = st.secrets.get("OPENAI_API_KEY")
tg_token = st.secrets.get("TELEGRAM_TOKEN")
tg_chat = st.secrets.get("TELEGRAM_CHAT_ID")

# ==========================================
# 2. HYPER-PHYSICS ENGINE
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
        # 1. Physics: RQZO & Entropy
        vel = df['Close'].pct_change().abs(); c = 0.05
        gamma = 1 / np.sqrt(1 - (np.minimum(vel, c*0.99)/c)**2)
        entropy = (df['Close'].pct_change().rolling(20).std() - 0) / 0.05 
        zeta = np.sin((np.arange(len(df))%100)/gamma) * np.exp(-2*np.abs(entropy-0.6))*10
        df['RQZO'] = zeta.fillna(0); df['Entropy'] = entropy.fillna(0.5)

        # 2. Vector Flux
        eff = ((df['Close']-df['Open']).abs()/(df['High']-df['Low']).replace(0,1)).ewm(span=14).mean()
        flux = (np.sign(df['Close']-df['Open']) * eff * (df['Volume']/df['Volume'].rolling(55).mean())).ewm(span=5).mean()
        df['Flux'] = flux; df['State'] = np.select([flux>0.6, flux<-0.6, flux.abs()<0.3], ['Super_Bull', 'Super_Bear', 'Resistive'], 'Heat')

        # 3. SMC Trend Cloud
        hma = TitanEngine._hma(df['Close'], 55); atr = (df['High']-df['Low']).rolling(55).mean()
        df['Cloud_Upper'] = hma + atr*1.5; df['Cloud_Lower'] = hma - atr*1.5
        df['Trend'] = np.where(df['Close'] > df['Cloud_Upper'], 1, np.where(df['Close'] < df['Cloud_Lower'], -1, 0))
        df['Trend'] = df['Trend'].replace(0, method='ffill')
        
        # 4. Tactical Patterns (NEW)
        body = (df['Close'] - df['Open']).abs()
        wick_up = df['High'] - df[['Open', 'Close']].max(axis=1)
        wick_dn = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Pattern'] = np.select(
            [(wick_dn > body*2) & (wick_up < body), (wick_up > body*2) & (wick_dn < body)],
            ['🔨 HAMMER', '☄️ SHOOTING STAR'], default=None
        )
        return df

    @staticmethod
    def get_mtf_bias(ticker):
        # Fetches Daily trend for context
        try:
            df_d = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if df_d.empty: return 0
            if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)
            hma = TitanEngine._hma(df_d['Close'], 55).iloc[-1]
            return 1 if df_d['Close'].iloc[-1] > hma else -1
        except: return 0

    @staticmethod
    def scan_structures(df):
        structs = {'OB': [], 'FVG': []}; pivot = 10
        for i in range(pivot, len(df)):
            # Order Blocks
            if df['Trend'].iloc[i] == 1 and df['Close'].iloc[i] > df['High'].iloc[i-pivot]:
                ob = df.iloc[i-15:i][df.iloc[i-15:i]['Close'] < df.iloc[i-15:i]['Open']]
                if not ob.empty: structs['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'col': 'rgba(0, 242, 96, 0.15)', 'bor': '#00f260'})
            if df['Trend'].iloc[i] == -1 and df['Close'].iloc[i] < df['Low'].iloc[i-pivot]:
                ob = df.iloc[i-15:i][df.iloc[i-15:i]['Close'] > df.iloc[i-15:i]['Open']]
                if not ob.empty: structs['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'col': 'rgba(255, 0, 60, 0.15)', 'bor': '#ff003c'})
            # FVGs
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
    def run_monte_carlo(df, days=30, sims=100):
        last = df['Close'].iloc[-1]; vol = df['Close'].pct_change().std()
        sim_df = pd.DataFrame()
        for x in range(sims): sim_df[x] = [last * (1 + np.random.normal(0, vol))**d for d in range(days)]
        return sim_df

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_fundamentals(ticker):
        if any(x in ticker for x in ["-", "=", "^"]): return None
        try:
            i = yf.Ticker(ticker).info
            return {"MktCap": i.get("marketCap", 0), "PE": i.get("trailingPE", 0), "Growth": i.get("revenueGrowth", 0)}
        except: return None

    @staticmethod
    def calc_correlations(ticker):
        try:
            a = {"SPX": "^SPX", "GOLD": "GC=F", "BTC": "BTC-USD"}
            df = yf.download(list(a.values()) + [ticker], period="3mo", progress=False)['Close']
            return df.corr()[ticker].drop(ticker).sort_values(ascending=False)
        except: return None

# ==========================================
# 3. SINGULARITY VISUALS
# ==========================================
def render_singularity_chart(df, structs, vp):
    last = df.iloc[-1]
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
        row_heights=[0.75, 0.25], specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Volume (Back)
    vol_c = ['rgba(0, 242, 96, 0.15)' if c >= o else 'rgba(255, 0, 60, 0.15)' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_c, showlegend=False), row=1, col=1, secondary_y=True)

    # Price (Front)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color='#00f260', increasing_fillcolor='#00f260',
        decreasing_line_color='#ff003c', decreasing_fillcolor='#ff003c', name="Price"
    ), row=1, col=1)

    # Patterns
    pats = df.dropna(subset=['Pattern'])
    if not pats.empty:
        fig.add_trace(go.Scatter(
            x=pats.index, y=pats['High']*1.001, mode='text', text=pats['Pattern'],
            textposition='top center', textfont=dict(color='#fff', size=10)
        ), row=1, col=1)

    # SMC Overlays
    for ob in structs['OB'][-20:]: 
        fig.add_shape(type="rect", x0=ob['x0'], x1=df.index[-1], y0=ob['y0'], y1=ob['y1'], fillcolor=ob['col'], line=dict(color=ob['bor'], width=1), row=1, col=1)
    for fvg in structs['FVG'][-20:]:
        fig.add_shape(type="rect", x0=fvg['x0'], x1=fvg['x1'], y0=fvg['y0'], y1=fvg['y1'], fillcolor=fvg['col'], line=dict(color=fvg['bor'], width=0, dash='dot'), row=1, col=1)

    # Cloud
    c_fill = 'rgba(0, 242, 96, 0.05)' if last['Trend']==1 else 'rgba(255, 0, 60, 0.05)'
    fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Lower'], fill='tonexty', fillcolor=c_fill, line=dict(width=0), name="Cloud"), row=1, col=1)

    # Physics Panel
    fig.add_trace(go.Scatter(x=df.index, y=df['RQZO'], line=dict(color='#0575e6', width=1.5), name="RQZO"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Entropy'], line=dict(color='#9c27b0', width=1), name="Entropy"), row=2, col=1)
    fig.add_hline(y=0.8, line_color="#ff003c", line_dash="dot", row=2, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#030303", plot_bgcolor="#080808", height=500, 
        margin=dict(l=0, r=0, t=10, b=10), hovermode="x unified", showlegend=False,
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False)), yaxis=dict(showgrid=True, gridcolor="#222"),
        yaxis2=dict(showgrid=False, range=[0, df['Volume'].max()*5], visible=False)
    )
    return fig

# ==========================================
# 4. MAIN OPS
# ==========================================
def main():
    # ASSET DOCK
    assets = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "GOLD": "GC=F", "SPX": "^SPX", "NVDA": "NVDA", "TSLA": "TSLA"}
    c1, c2 = st.columns([3, 1])
    sel_ast = c1.selectbox("ASSET", list(assets.keys()), label_visibility="collapsed")
    ticker = assets[sel_ast]
    interval = c2.selectbox("TF", ["15m", "1h", "4h", "1d"], index=1, label_visibility="collapsed")
    
    per_map = {"15m": "59d", "1h": "1y", "4h": "1y", "1d": "2y"}
    
    try:
        with st.spinner("SYNCING SINGULARITY NODES..."):
            # 1. MTF Context
            mtf_bias = TitanEngine.get_mtf_bias(ticker)
            bias_col = "#00f260" if mtf_bias == 1 else "#ff003c" if mtf_bias == -1 else "#888"
            bias_txt = "DAILY: BULL" if mtf_bias == 1 else "DAILY: BEAR" if mtf_bias == -1 else "DAILY: FLAT"

            # 2. Main Data
            df = yf.download(ticker, interval=interval, period=per_map[interval], progress=False)
            if df.empty: st.error("NO DATA"); return
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # 3. Engines
            df = TitanEngine.process_data(df)
            structs = TitanEngine.scan_structures(df)
            vp, poc = TitanEngine.get_vpvr(df)
            mc_sim = TitanEngine.run_monte_carlo(df)
            fund = TitanEngine.get_fundamentals(ticker)
            corr = TitanEngine.calc_correlations(ticker)
            last = df.iloc[-1]

        # HEADER
        st.markdown(f"""
        <div class="titan-header">
            TITAN // SINGULARITY
            <span class="mtf-pill" style="background:{bias_col}; color:#000;">{bias_txt}</span>
        </div>
        """, unsafe_allow_html=True)

        # HUD
        m1, m2 = st.columns(2)
        with m1:
            st.metric("STRUCTURE", "BULL" if last['Trend']==1 else "BEAR", f"{last['Cloud_Lower']:.2f}")
            st.metric("FLUX", last['State'].replace('_',' '), f"{last['Flux']:.2f}")
        with m2:
            st.metric("ENTROPY", "CHAOS" if last['Entropy']>0.8 else "STABLE", f"{last['Entropy']:.2f}")
            st.metric("POC", f"{poc:.2f}", "VOLUME")

        # TABS
        t1, t2, t3, t4 = st.tabs(["👁️ VISION", "🔬 DEEP INTEL", "🧠 TACTICAL AI", "📡 UPLINK"])

        with t1: 
            st.plotly_chart(render_singularity_chart(df, structs, vp), use_container_width=True, config={'displayModeBar': False})
            if last['Pattern']: st.info(f"DETECTED: {last['Pattern']}")

        with t2:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("30D FORECAST")
                fig_mc = go.Figure()
                for i in range(min(20, len(mc_sim.columns))):
                    fig_mc.add_trace(go.Scatter(y=mc_sim[i], mode='lines', line=dict(color='rgba(255,255,255,0.05)'), showlegend=False))
                fig_mc.add_trace(go.Scatter(y=mc_sim.mean(axis=1), mode='lines', line=dict(color='#0575e6', width=2), name="Mean"))
                fig_mc.update_layout(template="plotly_dark", height=200, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="#000000")
                st.plotly_chart(fig_mc, use_container_width=True)
            with c2:
                if corr is not None:
                    st.caption("CORRELATION")
                    st.dataframe(corr.to_frame().T, use_container_width=True)
            if fund:
                st.caption(f"MKT CAP: ${fund['MktCap']/1e9:.1f}B | P/E: {fund['PE']:.1f}")

        with t3:
            st.subheader("TACTICAL AI")
            if not api_key: st.warning("NO API KEY"); api_in = st.text_input("Key", type="password")
            else: api_in = api_key
            persona = st.selectbox("MODE", ["SCALPER", "SWING", "MACRO"])
            if st.button("EXECUTE ANALYSIS", use_container_width=True):
                if api_in:
                    p = f"Role: {persona}. Asset: {ticker}. TF: {interval}. MTF Bias: {bias_txt}. Trend: {last['Trend']}. Entropy: {last['Entropy']:.2f}. Pattern: {last['Pattern']}. Verdict?"
                    try:
                        client = OpenAI(api_key=api_in)
                        res = client.chat.completions.create(model="gpt-4", messages=[{"role":"user", "content":p}])
                        st.info(res.choices[0].message.content)
                    except Exception as e: st.error(str(e))

        with t4:
            st.subheader("UPLINK")
            if not tg_token or not tg_chat: st.info("NO CONFIG"); u_tok = st.text_input("Token"); u_chat = st.text_input("Chat")
            else: u_tok, u_chat = tg_token, tg_chat
            sig = "🟢" if last['Trend']==1 else "🔴"
            msg = f"{sig} **TITAN: {ticker}**\n• BIAS: {bias_txt}\n• FLUX: {last['State']}\n• PRICE: {last['Close']:.2f}"
            if st.button("BROADCAST", type="primary", use_container_width=True):
                if u_tok and u_chat:
                    try: requests.post(f"https://api.telegram.org/bot{u_tok}/sendMessage", data={"chat_id": u_chat, "text": msg}); st.success("SENT")
                    except: st.error("FAIL")

    except Exception as e: st.error(f"HALT: {e}")

if __name__ == "__main__":
    main()
