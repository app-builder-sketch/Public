import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
import requests
import urllib.parse
from datetime import datetime

# ==========================================
# 1. TITAN UI & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TITAN OPS",
    page_icon="☢️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- NEON / GLASSMORPHISM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;900&display=swap');
    
    .stApp { background-color: #000000; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    
    /* LIVE STATUS BAR */
    .status-bar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 12px; border-bottom: 1px solid #222; margin-bottom: 15px;
        background: #050505;
    }
    .status-live { color: #00ff88; font-weight: 900; animation: blink 2s infinite; font-family: 'Roboto Mono'; }
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

    /* BATTLE GRID METRICS */
    div[data-testid="stMetric"] {
        background: rgba(10, 10, 10, 0.9); 
        border: 1px solid #222; 
        border-left: 3px solid #00b8ff;
        padding: 10px; border-radius: 6px; 
    }
    div[data-testid="stMetricValue"] { font-family: 'Roboto Mono'; font-size: 1.1rem !important; color: #fff; }
    div[data-testid="stMetricLabel"] { font-size: 0.65rem !important; color: #888; letter-spacing: 1px; }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 4px; background: #050505; padding: 4px; 
        border-radius: 8px; border: 1px solid #222; 
    }
    .stTabs [data-baseweb="tab"] { 
        background: transparent; color: #666; 
        font-size: 0.7rem; font-weight: 700; flex: 1; border: none; 
    }
    .stTabs [aria-selected="true"] { background: #1a1a1a; color: #00ff88; border-radius: 4px; }
    
    /* BUTTONS */
    button[kind="primary"] { 
        background: linear-gradient(90deg, #00b8ff, #0055ff); 
        border: none; color: white; font-weight: bold; border-radius: 6px; 
    }
</style>
""", unsafe_allow_html=True)

# --- AUTO-LOAD SECRETS ---
api_key = st.secrets.get("OPENAI_API_KEY")
tg_token = st.secrets.get("TELEGRAM_TOKEN")
tg_chat = st.secrets.get("TELEGRAM_CHAT_ID")

# ==========================================
# 2. MASTER DATA ENGINE (FULL SUITE)
# ==========================================
class TitanEngine:
    @staticmethod
    def _hma(series, length):
        half = int(length / 2); sqrt = int(np.sqrt(length))
        wma_half = series.rolling(half).apply(lambda x: np.dot(x, np.arange(1, half+1)) / (half*(half+1)/2), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / (length*(length+1)/2), raw=True)
        return (2 * wma_half - wma_full).rolling(sqrt).apply(lambda x: np.dot(x, np.arange(1, sqrt+1)) / (sqrt*(sqrt+1)/2), raw=True)

    @staticmethod
    def run_simulation(df):
        df = df.copy()
        # Physics: RQZO & Entropy
        vel = df['Close'].pct_change().abs(); c = 0.05
        gamma = 1 / np.sqrt(1 - (np.minimum(vel, c*0.99)/c)**2)
        entropy = (df['Close'].pct_change().rolling(20).std() - 0) / 0.05 
        zeta = np.sin((np.arange(len(df))%100)/gamma) * np.exp(-2*np.abs(entropy-0.6))*10
        df['RQZO'] = zeta.fillna(0); df['Entropy'] = entropy.fillna(0.5)

        # Vector Flux
        eff = ((df['Close']-df['Open']).abs()/(df['High']-df['Low']).replace(0,1)).ewm(span=14).mean()
        flux = (np.sign(df['Close']-df['Open']) * eff * (df['Volume']/df['Volume'].rolling(55).mean())).ewm(span=5).mean()
        df['Flux'] = flux; df['State'] = np.select([flux>0.6, flux<-0.6, flux.abs()<0.3], ['Super_Bull', 'Super_Bear', 'Resistive'], 'Heat')

        # SMC Trend
        hma = TitanEngine._hma(df['Close'], 55); atr = (df['High']-df['Low']).rolling(55).mean()
        df['Cloud_Upper'] = hma + atr*1.5; df['Cloud_Lower'] = hma - atr*1.5
        df['Trend'] = np.where(df['Close'] > df['Cloud_Upper'], 1, np.where(df['Close'] < df['Cloud_Lower'], -1, 0))
        df['Trend'] = df['Trend'].replace(0, method='ffill')
        return df

    @staticmethod
    def scan_structures(df):
        structs = {'OB': [], 'FVG': []}; pivot = 10
        for i in range(pivot, len(df)):
            # Order Blocks
            if df['Trend'].iloc[i] == 1 and df['Close'].iloc[i] > df['High'].iloc[i-pivot]:
                ob = df.iloc[i-15:i][df.iloc[i-15:i]['Close'] < df.iloc[i-15:i]['Open']]
                if not ob.empty: structs['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'col': 'rgba(0, 255, 136, 0.2)', 'bor': '#00ff88'})
            
            if df['Trend'].iloc[i] == -1 and df['Close'].iloc[i] < df['Low'].iloc[i-pivot]:
                ob = df.iloc[i-15:i][df.iloc[i-15:i]['Close'] > df.iloc[i-15:i]['Open']]
                if not ob.empty: structs['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'col': 'rgba(255, 0, 85, 0.2)', 'bor': '#ff0055'})
            
            # FVGs
            if i > 2:
                if df['Low'].iloc[i-2] > df['High'].iloc[i]: 
                    structs['FVG'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i], 'y1': df['Low'].iloc[i-2], 'col': 'rgba(255, 200, 0, 0.15)', 'bor': '#ffcc00'})
                if df['High'].iloc[i-2] < df['Low'].iloc[i]:
                    structs['FVG'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i], 'col': 'rgba(255, 200, 0, 0.15)', 'bor': '#ffcc00'})
        return structs

    @staticmethod
    def get_vpvr(df):
        try:
            bins = np.linspace(df['Low'].min(), df['High'].max(), 50)
            df['Bin'] = pd.cut(df['Close'], bins=bins, include_lowest=True)
            vp = df.groupby('Bin', observed=True)['Volume'].sum().reset_index()
            vp['Price'] = vp['Bin'].apply(lambda x: x.mid).astype(float)
            return vp, vp.loc[vp['Volume'].idxmax(), 'Price']
        except: return pd.DataFrame(), 0.0

    @staticmethod
    def run_monte_carlo(df, days=30, sims=100):
        last = df['Close'].iloc[-1]; vol = df['Close'].pct_change().std()
        sim_df = pd.DataFrame()
        for x in range(sims): sim_df[x] = [last * (1 + np.random.normal(0, vol))**d for d in range(days)]
        return sim_df

    @staticmethod
    def get_seasonality(df):
        try:
            df = df.copy(); df['Ret'] = df['Close'].pct_change() * 100
            df['Month'] = df.index.month
            return df.groupby('Month')['Ret'].mean()
        except: return None

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_fundamentals(ticker):
        if any(x in ticker for x in ["-", "=", "^"]): return None # Skip fundamentals for crypto/forex/indices
        try:
            info = yf.Ticker(ticker).info
            return {"MktCap": info.get("marketCap", 0), "PE": info.get("trailingPE", 0), "Growth": info.get("revenueGrowth", 0)}
        except: return None

    @staticmethod
    def calc_correlations(ticker):
        assets = {"SPX": "^SPX", "GOLD": "GC=F", "BTC": "BTC-USD"}
        try:
            df_m = yf.download(list(assets.values()) + [ticker], period="3mo", progress=False)['Close']
            return df_m.corr()[ticker].drop(ticker).sort_values(ascending=False)
        except: return None

# ==========================================
# 3. EXPERT VISUAL RENDERER (PLOTLY PRO)
# ==========================================
def render_pro_chart(df, structs, vp):
    last = df.iloc[-1]
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
        row_heights=[0.75, 0.25], specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Volume Underlay
    vol_c = ['rgba(0, 255, 136, 0.2)' if c >= o else 'rgba(255, 0, 85, 0.2)' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_c, showlegend=False), row=1, col=1, secondary_y=True)

    # Price Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color='#00ff88', increasing_fillcolor='#00ff88',
        decreasing_line_color='#ff0055', decreasing_fillcolor='#ff0055', name="Price"
    ), row=1, col=1)

    # SMC Overlays
    for ob in structs['OB'][-25:]: 
        fig.add_shape(type="rect", x0=ob['x0'], x1=df.index[-1], y0=ob['y0'], y1=ob['y1'], 
                      fillcolor=ob['col'], line=dict(color=ob['bor'], width=1), row=1, col=1)
    for fvg in structs['FVG'][-25:]:
        fig.add_shape(type="rect", x0=fvg['x0'], x1=fvg['x1'], y0=fvg['y0'], y1=fvg['y1'],
                      fillcolor=fvg['col'], line=dict(color=fvg['bor'], width=0, dash='dot'), row=1, col=1)

    # Cloud
    c_fill = 'rgba(0, 255, 136, 0.05)' if last['Trend']==1 else 'rgba(255, 0, 85, 0.05)'
    fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Lower'], fill='tonexty', fillcolor=c_fill, line=dict(width=0), name="Cloud"), row=1, col=1)

    # Physics Panel
    fig.add_trace(go.Scatter(x=df.index, y=df['RQZO'], line=dict(color='#00b8ff', width=1.5), name="RQZO"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Entropy'], line=dict(color='#9c27b0', width=1), name="Entropy"), row=2, col=1)
    fig.add_hline(y=0.8, line_color="#ff0055", line_dash="dot", row=2, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#000000", plot_bgcolor="#050505", height=550, 
        margin=dict(l=0, r=0, t=10, b=10), hovermode="x unified", showlegend=False,
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor="#222"),
        yaxis2=dict(showgrid=False, range=[0, df['Volume'].max()*5], visible=False)
    )
    return fig

# ==========================================
# 4. MAIN OPS LOOP
# ==========================================
def main():
    st.markdown("""<div class="status-bar"><span style="font-weight:800; color:#fff;">TITAN // OPS</span><span class="status-live">● ONLINE</span></div>""", unsafe_allow_html=True)

    # ASSET DOCK
    assets = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "GOLD": "GC=F", "SPX": "^SPX", "NVDA": "NVDA", "TSLA": "TSLA"}
    c1, c2 = st.columns([3, 1])
    sel_ast = c1.selectbox("ASSET", list(assets.keys()), label_visibility="collapsed")
    ticker = assets[sel_ast]
    interval = c2.selectbox("TF", ["15m", "1h", "4h", "1d"], index=1, label_visibility="collapsed")

    per_map = {"15m": "59d", "1h": "1y", "4h": "1y", "1d": "2y"}
    
    try:
        with st.spinner("SYNCING..."):
            df = yf.download(ticker, interval=interval, period=per_map[interval], progress=False)
            if df.empty: st.error("NO DATA"); return
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # CORE CALCS
            df = TitanEngine.run_simulation(df)
            structs = TitanEngine.scan_structures(df)
            vp, poc = TitanEngine.get_vpvr(df)
            seas = TitanEngine.get_seasonality(df)
            mc_sim = TitanEngine.run_monte_carlo(df)
            fund = TitanEngine.get_fundamentals(ticker)
            corr = TitanEngine.calc_correlations(ticker)
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
        t1, t2, t3, t4 = st.tabs(["👁️ PRO CHART", "🔬 DEEP INTEL", "🧠 TACTICAL AI", "📡 UPLINK"])

        with t1: # Visuals
            st.plotly_chart(render_pro_chart(df, structs, vp), use_container_width=True, config={'displayModeBar': False})

        with t2: # Deep Intel (Restored & Optimized)
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                st.caption("30D FORECAST")
                fig_mc = go.Figure()
                for i in range(min(20, len(mc_sim.columns))):
                    fig_mc.add_trace(go.Scatter(y=mc_sim[i], mode='lines', line=dict(color='rgba(255,255,255,0.05)'), showlegend=False))
                fig_mc.add_trace(go.Scatter(y=mc_sim.mean(axis=1), mode='lines', line=dict(color='#00b8ff', width=2), name="Mean"))
                fig_mc.update_layout(template="plotly_dark", height=200, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="#000000")
                st.plotly_chart(fig_mc, use_container_width=True)
            with c_d2:
                if seas is not None:
                    st.caption("SEASONALITY")
                    fig_s = go.Figure(go.Bar(x=seas.index, y=seas.values, marker_color=seas.values, marker_colorscale="RdBu"))
                    fig_s.update_layout(template="plotly_dark", height=200, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="#000000")
                    st.plotly_chart(fig_s, use_container_width=True)

            # RESTORED FUNDAMENTALS & CORRELATIONS
            if fund: 
                st.caption("FUNDAMENTALS")
                c_f1, c_f2, c_f3 = st.columns(3)
                c_f1.metric("MKT CAP", f"${fund['MktCap']/1e9:.1f}B")
                c_f2.metric("P/E", f"{fund['PE']:.1f}")
                c_f3.metric("GROWTH", f"{fund['Growth']*100:.1f}%")
            
            if corr is not None:
                st.caption("CORRELATIONS (3M)")
                st.dataframe(corr.to_frame().T, use_container_width=True)

        with t3: # AI
            st.subheader("AI OPS")
            if not api_key: st.warning("ENTER API KEY IN SECRETS"); api_input = st.text_input("API Key", type="password")
            else: api_input = api_key
            
            persona = st.selectbox("OPERATOR", ["SCALPER", "SWING", "MACRO"])
            if st.button("RUN ANALYSIS", use_container_width=True):
                if api_input:
                    prompt = f"Role: {persona}. Asset: {ticker}. Trend: {last['Trend']}. Entropy: {last['Entropy']:.2f}. Flux: {last['State']}. Verdict?"
                    try:
                        client = OpenAI(api_key=api_input)
                        res = client.chat.completions.create(model="gpt-4", messages=[{"role":"user", "content":prompt}])
                        st.info(res.choices[0].message.content)
                    except Exception as e: st.error(str(e))

        with t4: # Broadcast
            st.subheader("SIGNAL UPLINK")
            if not tg_token or not tg_chat: st.info("ENTER TELEGRAM DETAILS"); u_token = st.text_input("Token", type="password"); u_chat = st.text_input("Chat ID")
            else: u_token, u_chat = tg_token, tg_chat

            sig = "🟢" if last['Trend']==1 else "🔴"
            def_msg = f"{sig} **TITAN: {ticker}**\n• STRUCT: {'BULL' if last['Trend']==1 else 'BEAR'}\n• FLUX: {last['State']}\n• PRICE: {last['Close']:.2f}"
            msg = st.text_area("MSG", value=def_msg)
            if st.button("SEND", type="primary", use_container_width=True):
                if u_token and u_chat:
                    try: requests.post(f"https://api.telegram.org/bot{u_token}/sendMessage", data={"chat_id": u_chat, "text": msg}); st.success("SENT")
                    except: st.error("FAIL")

    except Exception as e: st.error(f"HALT: {e}")

if __name__ == "__main__":
    main()
