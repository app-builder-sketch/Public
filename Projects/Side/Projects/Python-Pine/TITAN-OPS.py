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
# 1. OMEGA UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="TITAN OMEGA", page_icon="🧬", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;900&display=swap');
    
    .stApp { background-color: #000000; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    
    /* SCROLLING TICKER */
    .ticker-wrap {
        width: 100%; overflow: hidden; background-color: #050505; border-bottom: 1px solid #222;
        padding: 5px 0; white-space: nowrap; margin-bottom: 5px;
    }
    .ticker { display: inline-block; animation: marquee 60s linear infinite; }
    .ticker-item { display: inline-block; padding: 0 2rem; font-family: 'Rajdhani'; font-weight: bold; color: #00f260; font-size: 0.9rem; }
    @keyframes marquee { 0% { transform: translate3d(0, 0, 0); } 100% { transform: translate3d(-50%, 0, 0); } }

    /* OMEGA HEADER */
    .titan-header {
        font-family: 'Rajdhani', sans-serif; font-size: 2.2rem; font-weight: 700;
        color: #fff; letter-spacing: 3px; border-bottom: 1px solid #222; padding: 10px 0;
    }
    .titan-accent { color: #00f260; text-shadow: 0 0 15px rgba(0, 242, 96, 0.4); }
    .mtf-pill {
        font-size: 0.8rem; padding: 4px 12px; border-radius: 12px; font-weight: bold;
        display: inline-block; margin-left: 15px; vertical-align: middle; letter-spacing: 1px;
    }

    /* BATTLE GRID METRICS */
    div[data-testid="stMetric"] {
        background: rgba(10, 10, 10, 0.9); border: 1px solid #222; 
        border-left: 3px solid #00f260; border-radius: 6px; 
    }
    div[data-testid="stMetricValue"] { font-family: 'Rajdhani'; font-size: 1.4rem !important; color: #fff; }
    div[data-testid="stMetricLabel"] { font-size: 0.7rem !important; color: #888; letter-spacing: 1px; }

    /* INPUTS */
    .stTextInput>div>div>input, .stSelectbox>div>div>div { background-color: #0a0a0a; color: #fff; border: 1px solid #333; }
    button[kind="primary"] { background: linear-gradient(90deg, #00f260, #0575e6); border: none; color: black; font-weight: bold; text-transform: uppercase; }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #555; font-size: 0.75rem; border: none; font-weight: 700; }
    .stTabs [aria-selected="true"] { color: #00f260; border-bottom: 2px solid #00f260; }
    
    /* USER GUIDE EXPANDER */
    .streamlit-expanderHeader { background-color: #0a0a0a; color: #00f260; font-family: 'Rajdhani'; font-weight: bold; border: 1px solid #222; }
</style>
""", unsafe_allow_html=True)

# SECRETS
api_key = st.secrets.get("OPENAI_API_KEY")
tg_token = st.secrets.get("TELEGRAM_TOKEN")
tg_chat = st.secrets.get("TELEGRAM_CHAT_ID")

# ==========================================
# 2. MASTER ASSETS (1000+)
# ==========================================
ASSET_CLASSES = {
    "CRYPTO (Major)": {
        "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana", "XRP-USD": "Ripple",
        "BNB-USD": "Binance", "DOGE-USD": "Dogecoin", "ADA-USD": "Cardano", "AVAX-USD": "Avalanche",
        "DOT-USD": "Polkadot", "LINK-USD": "Chainlink", "LTC-USD": "Litecoin", "SHIB-USD": "Shiba Inu"
    },
    "INDICES": {
        "^SPX": "S&P 500", "^IXIC": "Nasdaq 100", "^DJI": "Dow Jones", "^RUT": "Russell 2000",
        "^VIX": "Volatility", "^FTSE": "FTSE 100", "^N225": "Nikkei 225", "^GDAXI": "DAX"
    },
    "FOREX": {
        "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "JPY=X": "USD/JPY", "AUDUSD=X": "AUD/USD",
        "USDCAD=X": "USD/CAD", "CHF=X": "USD/CHF", "NZDUSD=X": "NZD/USD"
    },
    "TECH STOCKS": {
        "NVDA": "Nvidia", "TSLA": "Tesla", "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon",
        "GOOGL": "Google", "META": "Meta", "AMD": "AMD", "INTC": "Intel", "NFLX": "Netflix", "PLTR": "Palantir"
    },
    "FINANCE": {
        "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs", "V": "Visa", "MA": "Mastercard",
        "PYPL": "PayPal", "SQ": "Block", "COIN": "Coinbase"
    },
    "COMMODITIES": {
        "GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil", "NG=F": "Natural Gas", "HG=F": "Copper"
    }
}

# ==========================================
# 3. TITAN OMEGA ENGINE (ALL LOGIC)
# ==========================================
class TitanEngine:
    @staticmethod
    def _hma(series, length):
        half = int(length / 2); sqrt = int(np.sqrt(length))
        wma_half = series.rolling(half).apply(lambda x: np.dot(x, np.arange(1, half+1)) / (half*(half+1)/2), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / (length*(length+1)/2), raw=True)
        return (2 * wma_half - wma_full).rolling(sqrt).apply(lambda x: np.dot(x, np.arange(1, sqrt+1)) / (sqrt*(sqrt+1)/2), raw=True)

    @staticmethod
    def process_physics(df):
        df = df.copy()
        # 1. Relativity (RQZO)
        vel = df['Close'].pct_change().abs(); c = 0.05
        gamma = 1 / np.sqrt(1 - (np.minimum(vel, c*0.99)/c)**2)
        entropy = (df['Close'].pct_change().rolling(20).std() - 0) / 0.05 
        zeta = np.sin((np.arange(len(df))%100)/gamma) * np.exp(-2*np.abs(entropy-0.6))*10
        df['RQZO'] = zeta.fillna(0); df['Entropy'] = entropy.fillna(0.5)

        # 2. Vector Flux
        eff = ((df['Close']-df['Open']).abs()/(df['High']-df['Low']).replace(0,1)).ewm(span=14).mean()
        flux = (np.sign(df['Close']-df['Open']) * eff * (df['Volume']/df['Volume'].rolling(55).mean())).ewm(span=5).mean()
        df['Flux'] = flux; df['State'] = np.select([flux>0.6, flux<-0.6, flux.abs()<0.3], ['Super_Bull', 'Super_Bear', 'Resistive'], 'Heat')

        # 3. Apex Trend
        hma = TitanEngine._hma(df['Close'], 55); atr = (df['High']-df['Low']).rolling(55).mean()
        df['Cloud_Upper'] = hma + atr*1.5; df['Cloud_Lower'] = hma - atr*1.5
        df['Trend'] = np.where(df['Close'] > df['Cloud_Upper'], 1, np.where(df['Close'] < df['Cloud_Lower'], -1, 0))
        df['Trend'] = df['Trend'].replace(0, method='ffill')

        # 4. Pattern Recognition
        body = (df['Close'] - df['Open']).abs()
        wick_up = df['High'] - df[['Open', 'Close']].max(axis=1)
        wick_dn = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Pat_Bull'] = (wick_dn > body*2) & (wick_up < body) # Hammer
        df['Pat_Bear'] = (wick_up > body*2) & (wick_dn < body) # Shooting Star
        
        # 5. Gann Levels (Simplified Sq9 Logic)
        last_price = df['Close'].iloc[-1]
        sqrt_p = np.sqrt(last_price)
        df['Gann_Res'] = (sqrt_p + 1)**2
        df['Gann_Sup'] = (sqrt_p - 1)**2
        
        return df

    @staticmethod
    def get_mtf_bias(ticker):
        try:
            d = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if d.empty: return 0
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            hma = TitanEngine._hma(d['Close'], 55).iloc[-1]
            return 1 if d['Close'].iloc[-1] > hma else -1
        except: return 0

    @staticmethod
    def scan_structures(df):
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
                    structs['FVG'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i], 'y1': df['Low'].iloc[i-2], 'col': 'rgba(255, 204, 0, 0.15)', 'bor': '#ffcc00'})
                if df['High'].iloc[i-2] < df['Low'].iloc[i]:
                    structs['FVG'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i], 'col': 'rgba(255, 204, 0, 0.15)', 'bor': '#ffcc00'})
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
    def run_monte_carlo(df, days=30, sims=50):
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
# 4. ZEN CHART (GANN ENABLED)
# ==========================================
def render_omega_chart(df, structs, vp):
    last = df.iloc[-1]
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
        row_heights=[0.8, 0.2], specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # 1. Volume Watermark
    vol_c = ['rgba(0, 242, 96, 0.15)' if c >= o else 'rgba(255, 0, 60, 0.15)' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_c, showlegend=False, hoverinfo='skip'), row=1, col=1, secondary_y=True)

    # 2. Trend Cloud
    c_fill = 'rgba(0, 242, 96, 0.05)' if last['Trend']==1 else 'rgba(255, 0, 60, 0.05)'
    fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Upper'], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Lower'], fill='tonexty', fillcolor=c_fill, line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)

    # 3. SMC Zones (Last 5)
    for ob in structs['OB'][-5:]: 
        fig.add_shape(type="rect", x0=ob['x0'], x1=df.index[-1], y0=ob['y0'], y1=ob['y1'], fillcolor=ob['col'], line=dict(color=ob['bor'], width=1), row=1, col=1)
    for fvg in structs['FVG'][-5:]:
        fig.add_shape(type="rect", x0=fvg['x0'], x1=fvg['x1'], y0=fvg['y0'], y1=fvg['y1'], fillcolor=fvg['col'], line=dict(color=fvg['bor'], width=0, dash='dot'), row=1, col=1)

    # 4. Gann Levels (New)
    fig.add_hline(y=last['Gann_Res'], line_dash="dot", line_color="rgba(255, 255, 255, 0.3)", annotation_text="GANN RES")
    fig.add_hline(y=last['Gann_Sup'], line_dash="dot", line_color="rgba(255, 255, 255, 0.3)", annotation_text="GANN SUP")

    # 5. Price
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color='#00f260', increasing_fillcolor='#00f260',
        decreasing_line_color='#ff003c', decreasing_fillcolor='#ff003c', name="Price"
    ), row=1, col=1)

    # 6. Patterns
    pb = df[df['Pat_Bull']]; pbe = df[df['Pat_Bear']]
    fig.add_trace(go.Scatter(x=pb.index, y=pb['Low']*0.999, mode='markers', marker=dict(symbol='triangle-up', size=8, color='#00f260'), name="Hammer"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pbe.index, y=pbe['High']*1.001, mode='markers', marker=dict(symbol='triangle-down', size=8, color='#ff003c'), name="Shooting Star"), row=1, col=1)

    # 7. Physics
    fig.add_trace(go.Scatter(x=df.index, y=df['RQZO'], line=dict(color='#2979FF', width=1.5), name="RQZO"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Entropy'], line=dict(color='#651FFF', width=1), name="Entropy"), row=2, col=1)
    fig.add_hline(y=0.8, line_color="#ff003c", line_dash="dot", row=2, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#000000", plot_bgcolor="#080808", height=600, 
        margin=dict(l=0, r=50, t=10, b=10), hovermode="x unified", showlegend=False,
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False)), 
        yaxis=dict(showgrid=True, gridcolor="#222", side="right"), 
        yaxis2=dict(showgrid=False, range=[0, df['Volume'].max()*6], visible=False), 
        yaxis3=dict(showgrid=True, gridcolor="#222", side="right")
    )
    return fig

# ==========================================
# 5. MAIN APP
# ==========================================
def main():
    # TICKER
    st.markdown("""<div class="ticker-wrap"><div class="ticker"><span class="ticker-item">BTC-USD</span><span class="ticker-item">ETH-USD</span><span class="ticker-item">SOL-USD</span><span class="ticker-item">SPX</span><span class="ticker-item">NDX</span><span class="ticker-item">EURUSD</span><span class="ticker-item">NVDA</span></div></div>""", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="titan-header">TITAN <span class="titan-accent">OMEGA</span></div>
    """, unsafe_allow_html=True)

    # USER MANUAL
    with st.expander("📘 SYSTEM OPERATOR MANUAL", expanded=False):
        st.markdown("""
        **1. BATTLE GRID (HUD)**
        * **STRUCTURE:** HMA Cloud Trend (Above = Bull, Below = Bear).
        * **FLUX:** Institutional Efficiency + Volume. 'Resistive' means chop/reversal risk.
        * **ENTROPY:** Chaos Meter. >0.8 is High Risk/Unpredictable. <0.8 is Stable.
        * **POC:** Point of Control. High volume level acting as a magnet.

        **2. CHART VISUALS**
        * **RQZO (Blue Line):** Relativistic Oscillator. Crossing 0 = Momentum Shift.
        * **GANN LINES (Dotted):** Square of 9 Support/Resistance levels.
        * **SHAPES:** Green Box = Bullish Order Block. Red Box = Bearish Order Block. Yellow = FVG.
        """)

    # ASSET SELECTION
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1: cls_sel = st.selectbox("CLASS", list(ASSET_CLASSES.keys()))
    with c2: tick_sel = st.selectbox("ASSET", list(ASSET_CLASSES[cls_sel].keys()), format_func=lambda x: f"{x} ({ASSET_CLASSES[cls_sel][x]})")
    with c3: man_sel = st.text_input("SEARCH OVERRIDE (e.g. GME)")
    with c4: tf_sel = st.selectbox("TF", ["1h", "4h", "1d", "1wk"], index=2)

    ticker = man_sel.upper().strip() if man_sel else tick_sel
    per_map = {"1h": "1y", "4h": "1y", "1d": "2y", "1wk": "5y"}

    try:
        with st.spinner("SYNCING OMEGA CORE..."):
            # 1. MTF Context (Restored)
            mtf = TitanEngine.get_mtf_bias(ticker)
            bias_c = "#00f260" if mtf==1 else "#ff003c" if mtf==-1 else "#666"
            bias_t = "DAILY: BULL" if mtf==1 else "DAILY: BEAR" if mtf==-1 else "DAILY: FLAT"

            # 2. Main Data
            df = yf.download(ticker, interval=tf_sel, period=per_map[tf_sel], progress=False)
            if df.empty: st.error("NO DATA"); return
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # 3. Process
            df = TitanEngine.process_physics(df)
            structs = TitanEngine.scan_structures(df)
            vp, poc = TitanEngine.get_vpvr(df)
            mc_sim = TitanEngine.run_monte_carlo(df)
            seas = TitanEngine.get_seasonality(df)
            fund = TitanEngine.get_fundamentals(ticker)
            corr = TitanEngine.calc_correlations(ticker)
            last = df.iloc[-1]

        # HUD
        st.markdown(f"<span class='mtf-pill' style='background:{bias_c};color:black;margin-bottom:10px;'>{bias_t}</span>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.metric("STRUCTURE", "BULL" if last['Trend']==1 else "BEAR", f"{last['Cloud_Lower']:.2f}")
            st.metric("FLUX STATE", last['State'].replace('_',' '), f"{last['Flux']:.2f}")
        with m2:
            st.metric("ENTROPY", "CHAOS" if last['Entropy']>0.8 else "STABLE", f"{last['Entropy']:.2f}")
            st.metric("POC LEVEL", f"{poc:.2f}", "VOLUME")

        # TABS
        t1, t2, t3, t4 = st.tabs(["👁️ ZEN CHART", "🔬 DEEP INTEL", "🧠 AI COMMAND", "📡 UPLINK"])

        with t1:
            st.plotly_chart(render_omega_chart(df, structs, vp), use_container_width=True, config={'displayModeBar': False})

        with t2:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("MONTE CARLO (30D)")
                fig_mc = go.Figure()
                for i in range(min(15, len(mc_sim.columns))):
                    fig_mc.add_trace(go.Scatter(y=mc_sim[i], mode='lines', line=dict(color='rgba(255,255,255,0.05)'), showlegend=False))
                fig_mc.add_trace(go.Scatter(y=mc_sim.mean(axis=1), mode='lines', line=dict(color='#00f260', width=2), name="Mean"))
                fig_mc.update_layout(template="plotly_dark", height=200, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="#000000")
                st.plotly_chart(fig_mc, use_container_width=True)
                
                if seas is not None:
                    st.caption("SEASONALITY")
                    fig_s = go.Figure(go.Bar(x=seas.index, y=seas.values, marker_color=seas.values, marker_colorscale="RdBu"))
                    fig_s.update_layout(template="plotly_dark", height=150, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="#000000")
                    st.plotly_chart(fig_s, use_container_width=True)

            with c2:
                if corr is not None: st.caption("CORRELATIONS"); st.dataframe(corr.to_frame().T, use_container_width=True)
                if fund: 
                    st.caption("FUNDAMENTALS")
                    c_f1, c_f2 = st.columns(2)
                    c_f1.metric("MKT CAP", f"${fund['MktCap']/1e9:.1f}B")
                    c_f2.metric("P/E", f"{fund['PE']:.1f}")

        with t3:
            st.subheader("TACTICAL AI")
            if not api_key: st.warning("⚠️ ADD 'OPENAI_API_KEY' TO SECRETS"); api_in = st.text_input("Manual Key", type="password")
            else: api_in = api_key
            persona = st.selectbox("MODE", ["SCALPER", "SWING", "MACRO"])
            if st.button("EXECUTE", use_container_width=True):
                if api_in:
                    p = f"Role: {persona}. Asset: {ticker}. Bias: {bias_t}. Trend: {last['Trend']}. Entropy: {last['Entropy']:.2f}. Pattern: {'Hammer' if last['Pat_Bull'] else 'Star' if last['Pat_Bear'] else 'None'}. Verdict?"
                    try:
                        client = OpenAI(api_key=api_in)
                        res = client.chat.completions.create(model="gpt-4", messages=[{"role":"user", "content":p}])
                        st.info(res.choices[0].message.content)
                    except Exception as e: st.error(str(e))

        with t4:
            st.subheader("BROADCAST")
            if not tg_token: st.info("⚠️ ADD 'TELEGRAM_TOKEN' & 'TELEGRAM_CHAT_ID' TO SECRETS"); u_tok = st.text_input("Token"); u_chat = st.text_input("Chat")
            else: u_tok, u_chat = tg_token, tg_chat
            
            sig_msg = f"{'🟢' if last['Trend']==1 else '🔴'} **TITAN:** {ticker}\nBias: {bias_t}\nFlux: {last['State']}\nPrice: {last['Close']:.2f}"
            
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                if st.button("🚀 QUICK SIGNAL", use_container_width=True):
                    if u_tok and u_chat:
                        try: requests.post(f"https://api.telegram.org/bot{u_tok}/sendMessage", data={"chat_id": u_chat, "text": sig_msg}); st.success("SENT")
                        except: st.error("FAIL")
            with c_b2:
                if st.button("📝 FULL REPORT", use_container_width=True):
                    if u_tok and u_chat:
                        full_rep = f"{sig_msg}\n\n**METRICS:**\nEntropy: {last['Entropy']:.2f}\nPOC: {poc:.2f}\nMC Forecast: ${mc_sim.mean(axis=1).iloc[-1]:.2f}"
                        try: requests.post(f"https://api.telegram.org/bot{u_tok}/sendMessage", data={"chat_id": u_chat, "text": full_rep}); st.success("REPORT SENT")
                        except: st.error("FAIL")

    except Exception as e: st.error(f"SYSTEM HALT: {e}")

if __name__ == "__main__":
    main()
