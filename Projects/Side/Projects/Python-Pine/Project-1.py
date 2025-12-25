import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
import requests
import urllib.parse

# ==========================================
# 1. SYSTEM CONFIGURATION & MOBILE UI
# ==========================================
st.set_page_config(
    layout="wide", 
    page_title="Quantum Market Terminal", 
    page_icon="⚛️",
    initial_sidebar_state="collapsed"
)

# --- PROFESSIONAL MOBILE-FIRST CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    .stApp { background-color: #050505; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    
    /* Header & Typography */
    .header-container { padding: 1rem 0; border-bottom: 1px solid #333; margin-bottom: 1rem; }
    .header-main { 
        font-size: 1.8rem; font-weight: 800; 
        background: -webkit-linear-gradient(45deg, #00E5FF, #2979FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
    }
    .header-sub { font-size: 0.8rem; color: #888; margin-top: 5px; }

    /* Glassmorphism Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; color: #FFF; }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem; color: #9E9E9E; }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; overflow-x: auto; white-space: nowrap; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border: none; color: #666; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #00E5FF; background-color: rgba(0, 229, 255, 0.1); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- SECRETS ---
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key: api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# ==========================================
# 2. DATA ENGINE (Fundamentals, Macro, Correlations)
# ==========================================
class DataEngine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_fundamentals(ticker):
        if "-" in ticker or "=" in ticker: return None 
        try:
            info = yf.Ticker(ticker).info
            return {
                "Market Cap": info.get("marketCap", 0),
                "P/E Ratio": info.get("trailingPE", 0),
                "Rev Growth": info.get("revenueGrowth", 0),
                "Summary": info.get("longBusinessSummary", "N/A")
            }
        except: return None

    @staticmethod
    @st.cache_data(ttl=300)
    def get_macro_data():
        tickers = {"S&P 500": "SPY", "Gold": "GC=F", "DXY": "DX-Y.NYB", "10Y Yield": "^TNX", "Bitcoin": "BTC-USD", "VIX": "^VIX"}
        try:
            data = yf.download(list(tickers.values()), period="5d", interval="1d", progress=False)['Close']
            results = {}
            for name, sym in tickers.items():
                if sym in data.columns and len(data) >= 2:
                    curr, prev = data[sym].iloc[-1], data[sym].iloc[-2]
                    results[name] = {"price": curr, "chg": ((curr - prev) / prev) * 100}
            return results
        except: return {}

    @staticmethod
    def get_seasonality(df):
        try:
            df = df.copy()
            df['Ret'] = df['Close'].pct_change() * 100
            df['Month'] = df.index.month
            df['Day'] = df.index.day_name()
            monthly = df.groupby('Month')['Ret'].mean()
            daily = df.groupby('Day')['Ret'].agg(['mean', 'count', lambda x: (x>0).mean()*100])
            daily.columns = ['Avg Return', 'Count', 'Win Rate']
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            daily = daily.reindex([d for d in day_order if d in daily.index])
            return monthly, daily
        except: return None, None

    @staticmethod
    def calc_volume_profile(df, bins=50):
        try:
            price_bins = np.linspace(df['Low'].min(), df['High'].max(), bins)
            df_calc = df.copy()
            df_calc['Bin'] = pd.cut(df_calc['Close'], bins=price_bins, include_lowest=True)
            vp = df_calc.groupby('Bin', observed=True)['Volume'].sum().reset_index()
            vp['Price'] = vp['Bin'].apply(lambda x: x.mid).astype(float)
            return (vp, vp.loc[vp['Volume'].idxmax(), 'Price']) if not vp.empty else (pd.DataFrame(), 0.0)
        except: return pd.DataFrame(), 0.0

    @staticmethod
    def calc_correlations(ticker):
        assets = {"S&P 500": "SPY", "Gold": "GC=F", "Bitcoin": "BTC-USD", "10Y": "^TNX"}
        try:
            df_main = yf.download(ticker, period="6mo", progress=False)['Close']
            df_macro = yf.download(list(assets.values()), period="6mo", progress=False)['Close']
            combined = df_macro.copy(); combined[ticker] = df_main
            corr = combined.corr()[ticker].drop(ticker, errors='ignore').sort_values(ascending=False)
            inv_map = {v: k for k, v in assets.items()}
            corr.index = [inv_map.get(x, x) for x in corr.index]
            return corr
        except: return None

# ==========================================
# 3. QUANTUM PHYSICS ENGINE (RQZO, Vector, SMC)
# ==========================================
class QuantumEngine:
    @staticmethod
    def _hma(series, length):
        half = int(length / 2); sqrt = int(np.sqrt(length))
        wma_half = series.rolling(half).apply(lambda x: np.dot(x, np.arange(1, half+1)) / (half*(half+1)/2), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / (length*(length+1)/2), raw=True)
        return (2 * wma_half - wma_full).rolling(sqrt).apply(lambda x: np.dot(x, np.arange(1, sqrt+1)) / (sqrt*(sqrt+1)/2), raw=True)

    @staticmethod
    def process_physics(df):
        df = df.copy()
        # RQZO [cite: 1-12]
        vel = df['Close'].pct_change().abs(); c = 0.05
        gamma = 1 / np.sqrt(1 - (np.minimum(vel, c*0.99)/c)**2)
        entropy = (df['Close'].pct_change().rolling(20).std() - 0) / 0.05 # Simplified norm
        zeta = np.sin((np.arange(len(df))%100)/gamma) * np.exp(-2*np.abs(entropy-0.6))*10
        df['RQZO'] = zeta.fillna(0); df['Entropy'] = entropy.fillna(0.5)

        # Apex Vector [cite: 13-36]
        eff = ((df['Close']-df['Open']).abs()/(df['High']-df['Low']).replace(0,1)).ewm(span=14).mean()
        flux = (np.sign(df['Close']-df['Open']) * eff * (df['Volume']/df['Volume'].rolling(55).mean())).ewm(span=5).mean()
        df['Flux'] = flux; df['State'] = np.select([flux>0.6, flux<-0.6, flux.abs()<0.3], ['Super_Bull', 'Super_Bear', 'Resistive'], 'Heat')

        # Trend Cloud [cite: 37-45]
        hma = QuantumEngine._hma(df['Close'], 55); atr = (df['High']-df['Low']).rolling(55).mean()
        df['Cloud_Upper'] = hma + atr*1.5; df['Cloud_Lower'] = hma - atr*1.5
        df['Trend'] = np.where(df['Close'] > df['Cloud_Upper'], 1, np.where(df['Close'] < df['Cloud_Lower'], -1, 0))
        df['Trend'] = df['Trend'].replace(0, method='ffill')
        return df

    @staticmethod
    def get_smc_structures(df):
        # SMC: BOS, OB, FVG [cite: 46-54]
        structures = {'BOS': [], 'OB': [], 'FVG': []}
        pivot = 10
        df['PH'] = df['High'].rolling(pivot*2+1, center=True).max() == df['High']
        df['PL'] = df['Low'].rolling(pivot*2+1, center=True).min() == df['Low']
        last_ph, last_pl = np.nan, np.nan

        for i in range(pivot, len(df)):
            if df['PH'].iloc[i-pivot]: last_ph = df['High'].iloc[i-pivot]
            if df['PL'].iloc[i-pivot]: last_pl = df['Low'].iloc[i-pivot]
            
            # BOS & Order Blocks
            if df['Trend'].iloc[i] == 1 and df['Close'].iloc[i] > last_ph and df['Close'].iloc[i-1] <= last_ph:
                structures['BOS'].append({'idx': df.index[i], 'price': last_ph, 'type': 'Bull'})
                ob = df.iloc[i-15:i][df.iloc[i-15:i]['Close'] < df.iloc[i-15:i]['Open']]
                if not ob.empty: structures['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'col': 'rgba(0,255,128,0.3)'})
            
            if df['Trend'].iloc[i] == -1 and df['Close'].iloc[i] < last_pl and df['Close'].iloc[i-1] >= last_pl:
                structures['BOS'].append({'idx': df.index[i], 'price': last_pl, 'type': 'Bear'})
                ob = df.iloc[i-15:i][df.iloc[i-15:i]['Close'] > df.iloc[i-15:i]['Open']]
                if not ob.empty: structures['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'col': 'rgba(255,0,80,0.3)'})
            
            # FVG Detection 
            if i > 2:
                if df['Low'].iloc[i-2] > df['High'].iloc[i]: # Bearish FVG
                    structures['FVG'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i], 'y1': df['Low'].iloc[i-2], 'col': 'rgba(255,200,0,0.2)'})
                if df['High'].iloc[i-2] < df['Low'].iloc[i]: # Bullish FVG
                    structures['FVG'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i], 'col': 'rgba(255,200,0,0.2)'})
        return structures

    @staticmethod
    def run_monte_carlo(df, days=30, sims=100):
        last = df['Close'].iloc[-1]; vol = df['Close'].pct_change().std()
        sim_df = pd.DataFrame()
        for x in range(sims):
            sim_df[x] = [last * (1 + np.random.normal(0, vol))**d for d in range(days)]
        return sim_df

# ==========================================
# 4. MAIN UI LOGIC
# ==========================================
def main():
    st.markdown('<div class="header-container"><h1 class="header-main">Quantum Market Terminal</h1><p class="header-sub">Full-Spectrum Institutional Analytics</p></div>', unsafe_allow_html=True)

    with st.expander("📡 Control Center", expanded=True):
        c1, c2 = st.columns([2, 1])
        ticker = c1.text_input("Ticker", "BTC-USD").upper()
        interval = c2.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=3)
        if st.button("🚀 Initialize System", type="primary", use_container_width=True): st.session_state.run = True

    # Macro Sidebar
    macro = DataEngine.get_macro_data()
    if macro: 
        st.sidebar.markdown("### 🌍 Pulse")
        for k, v in macro.items(): st.sidebar.metric(k, f"{v['price']:.2f}", f"{v['chg']:.2f}%")

    if st.session_state.get("run"):
        per_map = {"15m": "59d", "1h": "1y", "4h": "1y", "1d": "2y", "1wk": "5y"}
        with st.spinner("Processing Quantum Mechanics..."):
            df = yf.download(ticker, interval=interval, period=per_map[interval], progress=False)
            if df.empty: st.error("No Data"); return
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            df = QuantumEngine.process_physics(df)
            structs = QuantumEngine.get_smc_structures(df)
            vp, poc = DataEngine.calc_volume_profile(df)
            seas_m, seas_d = DataEngine.get_seasonality(df)
            corr = DataEngine.calc_correlations(ticker)
            mc_sim = QuantumEngine.run_monte_carlo(df)

        last = df.iloc[-1]
        
        # METRICS GRID
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Structure", "Bullish" if last['Trend']==1 else "Bearish", f"Cloud: {last['Cloud_Upper']:.2f}", delta_color="normal" if last['Trend']==1 else "inverse")
        m2.metric("Entropy", f"{last['Entropy']:.2f}", "Chaotic" if last['Entropy']>0.8 else "Stable", delta_color="inverse" if last['Entropy']>0.8 else "off")
        m3.metric("Flux State", last['State'].replace('_', ' '), f"{last['Flux']:.2f}")
        m4.metric("POC Level", f"{poc:.2f}", "Vol Control", delta_color="normal" if last['Close']>poc else "inverse")

        # TABS
        tabs = st.tabs(["📊 Structure", "⚛️ Physics", "🔬 Deep Dive", "📢 Share"])

        with tabs[0]: # Chart
            fig = make_subplots(rows=1, cols=1)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
            
            # Cloud
            c_fill = 'rgba(0, 230, 118, 0.1)' if last['Trend']==1 else 'rgba(255, 23, 68, 0.1)'
            fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Upper'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Lower'], fill='tonexty', fillcolor=c_fill, line=dict(width=0), name="Cloud"))
            
            # SMC Overlay (OB + FVG)
            for ob in structs['OB'][-15:]: fig.add_shape(type="rect", x0=ob['x0'], x1=df.index[-1], y0=ob['y0'], y1=ob['y1'], fillcolor=ob['col'], line_width=0)
            for fvg in structs['FVG'][-15:]: fig.add_shape(type="rect", x0=fvg['x0'], x1=fvg['x1'], y0=fvg['y0'], y1=fvg['y1'], fillcolor=fvg['col'], line_width=0)
            
            if not vp.empty:
                fig.add_trace(go.Bar(x=vp['Volume'], y=vp['Price'], orientation='h', marker_color='rgba(255,255,255,0.1)', name="VPVR", xaxis='x2'))
                fig.update_layout(xaxis2=dict(overlaying='x', side='top', range=[0, vp['Volume'].max()*4], showticklabels=False))
            
            fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]: # Physics
            fig_q = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig_q.add_trace(go.Scatter(x=df.index, y=df['RQZO'], line=dict(color='#00E5FF'), name="RQZO"), row=1, col=1)
            fig_q.add_trace(go.Scatter(x=df.index, y=df['Entropy'], line=dict(color='#AB47BC'), name="Entropy"), row=2, col=1)
            fig_q.add_hline(y=0.8, line_color="red", line_dash="dash", row=2, col=1)
            fig_q.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=10,b=10))
            st.plotly_chart(fig_q, use_container_width=True)
            
            fig_f = go.Figure()
            colors = ['#00E676' if s=='Super_Bull' else '#FF1744' if s=='Super_Bear' else '#546E7A' for s in df['State']]
            fig_f.add_trace(go.Bar(x=df.index, y=df['Flux'], marker_color=colors, name="Flux"))
            fig_f.update_layout(height=250, template="plotly_dark", margin=dict(l=0,r=0,t=10,b=10))
            st.plotly_chart(fig_f, use_container_width=True)

        with tabs[2]: # Deep Dive
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Seasonality")
                if seas_m is not None: st.plotly_chart(px.bar(x=seas_m.index, y=seas_m.values, color=seas_m.values, color_continuous_scale="RdBu"), use_container_width=True)
                if seas_d is not None: 
                    try: import matplotlib; st.dataframe(seas_d.style.background_gradient(cmap="Greens"), use_container_width=True)
                    except: st.dataframe(seas_d, use_container_width=True)
            with c2:
                st.markdown("##### Forecast")
                fig_mc = go.Figure()
                for i in range(min(50, len(mc_sim.columns))): fig_mc.add_trace(go.Scatter(y=mc_sim[i], mode='lines', line=dict(color='rgba(255,255,255,0.05)'), showlegend=False))
                fig_mc.add_trace(go.Scatter(y=mc_sim.mean(axis=1), mode='lines', line=dict(color='#00E5FF', width=2), name="Mean"))
                st.plotly_chart(fig_mc, use_container_width=True)
                
            if corr is not None:
                st.markdown("##### Correlations")
                try: import matplotlib; st.dataframe(corr.to_frame().T.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1), use_container_width=True)
                except: st.dataframe(corr.to_frame().T, use_container_width=True)

        with tabs[3]: # Broadcaster (Original Feature Restored)
            st.markdown("#### 📡 Broadcast Signal")
            signal_text = f"🔥 {ticker} ({interval})\nStruct: {'Bull' if last['Trend']==1 else 'Bear'}\nFlux: {last['State']}\nEntropy: {last['Entropy']:.2f}\n#QuantumTerminal"
            msg = st.text_area("Draft Message", value=signal_text)
            if st.button("🐦 Launch Tweet"):
                st.link_button("Post to X", f"https://twitter.com/intent/tweet?text={urllib.parse.quote(msg)}")
            
            if api_key:
                if st.button("🤖 AI Analysis"):
                    with st.spinner("Analyzing..."):
                        prompt = f"Analyze {ticker}. Trend: {last['Trend']}, Entropy: {last['Entropy']:.2f}, Flux: {last['Flux']:.2f}. Concise."
                        try:
                            client = OpenAI(api_key=api_key)
                            st.info(client.chat.completions.create(model="gpt-4", messages=[{"role":"user", "content":prompt}]).choices[0].message.content)
                        except Exception as e: st.error(str(e))

if __name__ == "__main__":
    main()
