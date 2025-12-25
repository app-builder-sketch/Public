import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
from scipy.stats import linregress
import datetime

# ==========================================
# 1. SYSTEM CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    layout="wide", 
    page_title="Quantum Market Terminal", 
    page_icon="⚛️",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL INSTITUTIONAL THEME ---
st.markdown("""
<style>
    /* Global Container */
    .stApp { background-color: #0E1117; color: #C9D1D9; font-family: 'Inter', sans-serif; }
    
    /* Typography */
    .header-main { font-size: 2.2rem; font-weight: 700; color: #F0F6FC; letter-spacing: -1px; margin-bottom: 0; }
    .header-sub { font-size: 1.0rem; color: #8B949E; margin-bottom: 25px; font-weight: 400; }
    .section-title { font-size: 1.2rem; font-weight: 600; color: #58A6FF; margin-top: 20px; border-bottom: 1px solid #30363D; padding-bottom: 5px; }

    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 6px;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover { border-color: #58A6FF; transform: translateY(-2px); }
    div[data-testid="stMetricLabel"] { color: #8B949E; font-size: 0.85rem; }
    div[data-testid="stMetricValue"] { color: #F0F6FC; font-size: 1.5rem; font-weight: 600; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #30363D; gap: 20px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border: none; color: #8B949E; font-size: 0.95rem; }
    .stTabs [aria-selected="true"] { color: #58A6FF; border-bottom: 2px solid #58A6FF; font-weight: 600; }
    
    /* Expanders */
    .streamlit-expanderHeader { background-color: #0d1117; border: 1px solid #30363D; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# --- SECRETS & API ---
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("OpenAI API Key (Analyst)", type="password")

# ==========================================
# 2. DATA ENGINE: MACRO & FUNDAMENTALS (FROM TITAN)
# ==========================================
class DataEngine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_fundamentals(ticker):
        """Fetches key financial metrics."""
        if "-" in ticker or "=" in ticker or "^" in ticker: return None 
        try:
            info = yf.Ticker(ticker).info
            return {
                "Market Cap": info.get("marketCap", 0),
                "P/E Ratio": info.get("trailingPE", 0),
                "Rev Growth": info.get("revenueGrowth", 0),
                "Debt/Equity": info.get("debtToEquity", 0),
                "Summary": info.get("longBusinessSummary", "No Data Available")
            }
        except: return None

    @staticmethod
    @st.cache_data(ttl=300)
    def get_macro_data():
        """Fetches 40+ global macro indicators."""
        tickers = {
            "🇺🇸 S&P 500": "SPY", "🌍 World": "URTH", "🏆 Gold": "GC=F", "🛢️ Oil": "CL=F",
            "💵 DXY": "DX-Y.NYB", "yield 10Y": "^TNX", "₿ Bitcoin": "BTC-USD", "fear VIX": "^VIX"
        }
        try:
            data = yf.download(list(tickers.values()), period="5d", interval="1d", progress=False)['Close']
            results = {}
            for name, sym in tickers.items():
                if sym in data.columns and len(data) >= 2:
                    curr, prev = data[sym].iloc[-1], data[sym].iloc[-2]
                    chg = ((curr - prev) / prev) * 100
                    results[name] = {"price": curr, "chg": chg}
            return results
        except: return {}

    @staticmethod
    def get_seasonality(df):
        """Calculates Monthly and Day-of-Week Seasonality."""
        try:
            df['Ret'] = df['Close'].pct_change() * 100
            df['Month'] = df.index.month
            df['Day'] = df.index.day_name()
            
            # Monthly Heatmap
            monthly = df.groupby('Month')['Ret'].mean()
            
            # Day of Week DNA
            daily = df.groupby('Day')['Ret'].agg(['mean', 'count', lambda x: (x>0).mean()*100])
            daily.columns = ['Avg Return', 'Count', 'Win Rate']
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            daily = daily.reindex([d for d in day_order if d in daily.index])
            
            return monthly, daily
        except: return None, None

    @staticmethod
    def calc_volume_profile(df, bins=50):
        """Calculates VPVR (Volume Profile)."""
        price_min = df['Low'].min()
        price_max = df['High'].max()
        price_bins = np.linspace(price_min, price_max, bins)
        df['Bin'] = pd.cut(df['Close'], bins=price_bins, include_lowest=True)
        vp = df.groupby('Bin', observed=True)['Volume'].sum().reset_index()
        vp['Price'] = [i.mid for i in vp['Bin']]
        return vp, vp.loc[vp['Volume'].idxmax(), 'Price']

    @staticmethod
    def calc_correlations(ticker):
        """Cross-Asset Correlation Matrix."""
        assets = {"S&P 500": "SPY", "Gold": "GC=F", "Bitcoin": "BTC-USD", "10Y Yield": "^TNX", "Dollar": "DX-Y.NYB"}
        try:
            df_main = yf.download(ticker, period="6mo", progress=False)['Close']
            df_macro = yf.download(list(assets.values()), period="6mo", progress=False)['Close']
            combined = df_macro.copy()
            combined[ticker] = df_main
            corr = combined.corr()[ticker].drop(ticker).sort_values(ascending=False)
            # Map symbols back to names
            inv_map = {v: k for k, v in assets.items()}
            corr.index = [inv_map.get(x, x) for x in corr.index]
            return corr
        except: return None

# ==========================================
# 3. QUANTUM PHYSICS ENGINE (FROM PINE)
# ==========================================
class QuantumEngine:
    @staticmethod
    def _hma(series, length):
        """Hull Moving Average."""
        half_len = int(length / 2)
        sqrt_len = int(np.sqrt(length))
        wma_half = series.rolling(half_len).apply(lambda x: np.dot(x, np.arange(1, half_len+1)) / (half_len*(half_len+1)/2), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / (length*(length+1)/2), raw=True)
        diff = 2 * wma_half - wma_full
        return diff.rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(1, sqrt_len+1)) / (sqrt_len*(sqrt_len+1)/2), raw=True)

    @staticmethod
    def process_full_physics(df):
        """Integrates RQZO, Apex Vector, and SMC in one pass."""
        # --- 1. RQZO (Relativity) ---
        # Lorentz Factor
        velocity = df['Close'].pct_change().abs()
        c = 0.05 # Theoretical terminal volatility limit (5%)
        v_clamped = np.minimum(velocity, c * 0.99)
        gamma = 1 / np.sqrt(1 - (v_clamped / c)**2)
        
        # Entropy
        roll_std = df['Close'].pct_change().rolling(20).std()
        entropy = (roll_std - roll_std.rolling(100).min()) / (roll_std.rolling(100).max() - roll_std.rolling(100).min())
        
        # Zeta Oscillator (Simplified Vectorized)
        tau = (np.arange(len(df)) % 100) / gamma
        zeta = np.sin(tau) * np.exp(-2.0 * np.abs(entropy - 0.6)) * 10
        
        df['RQZO'] = zeta.fillna(0)
        df['Entropy'] = entropy.fillna(0.5)

        # --- 2. Apex Vector (Flux) ---
        eff = ( (df['Close'] - df['Open']).abs() / (df['High'] - df['Low']).replace(0, 1) ).ewm(span=14).mean()
        vol_flux = (df['Volume'] / df['Volume'].rolling(55).mean())
        df['Flux'] = (np.sign(df['Close'] - df['Open']) * eff * vol_flux).ewm(span=5).mean()
        df['State'] = np.select(
            [df['Flux'] > 0.6, df['Flux'] < -0.6, df['Flux'].abs() < 0.3],
            ['Super_Bull', 'Super_Bear', 'Resistive'], default='Heat'
        )

        # --- 3. SMC Trend ---
        hma = QuantumEngine._hma(df['Close'], 55)
        atr = (df['High'] - df['Low']).rolling(55).mean()
        df['Cloud_Upper'] = hma + (atr * 1.5)
        df['Cloud_Lower'] = hma - (atr * 1.5)
        
        df['Trend'] = 0
        df.loc[df['Close'] > df['Cloud_Upper'], 'Trend'] = 1
        df.loc[df['Close'] < df['Cloud_Lower'], 'Trend'] = -1
        df['Trend'] = df['Trend'].replace(0, method='ffill')

        return df

    @staticmethod
    def get_structures(df):
        """Extracts Order Blocks and BOS."""
        pivot_len = 10
        structures = {'BOS': [], 'OB': []}
        
        df['PH'] = df['High'].rolling(pivot_len*2+1, center=True).max() == df['High']
        df['PL'] = df['Low'].rolling(pivot_len*2+1, center=True).min() == df['Low']
        
        last_ph, last_pl = np.nan, np.nan
        
        for i in range(pivot_len, len(df)):
            if df['PH'].iloc[i-pivot_len]: last_ph = df['High'].iloc[i-pivot_len]
            if df['PL'].iloc[i-pivot_len]: last_pl = df['Low'].iloc[i-pivot_len]
            
            # BOS Bull
            if df['Trend'].iloc[i] == 1 and df['Close'].iloc[i] > last_ph and df['Close'].iloc[i-1] <= last_ph:
                structures['BOS'].append({'idx': df.index[i], 'price': last_ph, 'type': 'Bull'})
                # Order Block
                obs = df.iloc[i-15:i]
                ob = obs[obs['Close'] < obs['Open']]
                if not ob.empty:
                    structures['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'type': 'Demand'})

            # BOS Bear
            if df['Trend'].iloc[i] == -1 and df['Close'].iloc[i] < last_pl and df['Close'].iloc[i-1] >= last_pl:
                structures['BOS'].append({'idx': df.index[i], 'price': last_pl, 'type': 'Bear'})
                obs = df.iloc[i-15:i]
                ob = obs[obs['Close'] > obs['Open']]
                if not ob.empty:
                    structures['OB'].append({'x0': ob.index[-1], 'y0': ob['Low'].iloc[-1], 'y1': ob['High'].iloc[-1], 'type': 'Supply'})
                    
        return structures

    @staticmethod
    def run_monte_carlo(df, days=30, sims=200):
        """Runs Monte Carlo Simulation."""
        last_price = df['Close'].iloc[-1]
        daily_vol = df['Close'].pct_change().std()
        
        simulation_df = pd.DataFrame()
        for x in range(sims):
            price_series = [last_price]
            for _ in range(days):
                price_series.append(price_series[-1] * (1 + np.random.normal(0, daily_vol)))
            simulation_df[x] = price_series
        return simulation_df

# ==========================================
# 4. MAIN APPLICATION
# ==========================================
def main():
    # --- SIDEBAR: CONTROLS & MACRO ---
    st.sidebar.markdown("### 📡 Asset Config")
    ticker = st.sidebar.text_input("Ticker Symbol", value="BTC-USD").upper()
    interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d", "1wk"], index=3)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌍 Global Pulse")
    macro = DataEngine.get_macro_data()
    if macro:
        cols = st.sidebar.columns(2)
        for i, (k, v) in enumerate(macro.items()):
            cols[i % 2].metric(k, f"{v['price']:.2f}", f"{v['chg']:.2f}%")
    
    if st.sidebar.button("Initialize Quantum Engine", type="primary"):
        st.session_state.run = True

    # --- MAIN HEADER ---
    st.markdown('<div class="header-main">Quantum Market Terminal</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="header-sub">Integrated Physics & Institutional Analytics • {ticker} • {interval}</div>', unsafe_allow_html=True)

    if st.session_state.get("run"):
        # Timeframe Mapping
        per_map = {"15m": "59d", "1h": "1y", "4h": "1y", "1d": "2y", "1wk": "5y"}
        
        with st.spinner("Crunching vectors..."):
            df = yf.download(ticker, interval=interval, period=per_map[interval], progress=False)
            if df.empty:
                st.error("No Data Found.")
                return
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # RUN ALL ENGINES
            df = QuantumEngine.process_full_physics(df)
            structs = QuantumEngine.get_structures(df)
            vp, poc = DataEngine.calc_volume_profile(df)
            seas_m, seas_d = DataEngine.get_seasonality(df)
            fund = DataEngine.get_fundamentals(ticker)
            corr = DataEngine.calc_correlations(ticker)
            mc_sim = QuantumEngine.run_monte_carlo(df)

        last = df.iloc[-1]

        # --- LAYER 1: HEADS-UP DISPLAY (METRICS) ---
        c1, c2, c3, c4 = st.columns(4)
        
        # Structure
        struct_col = "normal" if last['Trend'] == 1 else "inverse"
        c1.metric("Market Structure", "Bullish" if last['Trend'] == 1 else "Bearish", f"Cloud: {last['Cloud_Upper']:.2f}", delta_color=struct_col)
        
        # Physics
        ent_label = "Chaotic" if last['Entropy'] > 0.8 else "Stable"
        ent_col = "inverse" if last['Entropy'] > 0.8 else "off"
        c2.metric("Quantum Entropy", f"{last['Entropy']:.2f}", ent_label, delta_color=ent_col)
        
        # Flux
        flux_state_map = {"Super_Bull": "Superconductor 🟢", "Super_Bear": "Superconductor 🔴", "Resistive": "Resistive ⚪", "Heat": "High Heat 🟡"}
        c3.metric("Vector Flux", f"{last['Flux']:.2f}", flux_state_map.get(last['State'], "Neutral"))
        
        # Volume
        vol_col = "normal" if last['Close'] > poc else "inverse"
        c4.metric("Volume POC", f"{poc:.2f}", "Control Price", delta_color=vol_col)

        st.markdown("---")

        # --- LAYER 2: ANALYTICAL TABS ---
        tab_chart, tab_phys, tab_deep, tab_fund, tab_ai = st.tabs([
            "📊 Structural Chart", "⚛️ Physics Lab", "🔬 Deep Analytics", "🏢 Fundamentals", "🤖 AI Analyst"
        ])

        # TAB 1: STRUCTURE & SMC
        with tab_chart:
            fig = make_subplots(rows=1, cols=1)
            # Price
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
            # Cloud
            cloud_col = 'rgba(0, 230, 118, 0.1)' if last['Trend'] == 1 else 'rgba(255, 23, 68, 0.1)'
            fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Upper'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df.index, y=df['Cloud_Lower'], fill='tonexty', fillcolor=cloud_col, line=dict(width=0), name="Trend Cloud"))
            # Order Blocks
            for ob in structs['OB'][-10:]:
                col = 'rgba(0, 255, 128, 0.3)' if ob['type'] == 'Demand' else 'rgba(255, 0, 80, 0.3)'
                fig.add_shape(type="rect", x0=ob['x0'], x1=df.index[-1], y0=ob['y0'], y1=ob['y1'], fillcolor=col, line_width=0)
            
            # VPVR Sidebar
            fig.add_trace(go.Bar(x=vp['Volume'], y=vp['Price'], orientation='h', marker_color='rgba(255,255,255,0.1)', name="Volume Profile", xaxis='x2'))
            
            fig.update_layout(
                height=650, template="plotly_dark", xaxis_rangeslider_visible=False,
                xaxis2=dict(overlaying='x', side='top', range=[0, vp['Volume'].max()*5], showticklabels=False)
            )
            st.plotly_chart(fig, use_container_width=True)

        # TAB 2: PHYSICS (RQZO & VECTOR)
        with tab_phys:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("##### Relativistic Oscillator (RQZO)")
                fig_q = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig_q.add_trace(go.Scatter(x=df.index, y=df['RQZO'], line=dict(color='#00E5FF'), name="RQZO"), row=1, col=1)
                fig_q.add_trace(go.Scatter(x=df.index, y=df['Entropy'], line=dict(color='#AB47BC'), name="Entropy"), row=2, col=1)
                fig_q.add_hline(y=0.8, line_color="red", line_dash="dash", row=2, col=1)
                fig_q.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_q, use_container_width=True)
            
            with col_p2:
                st.markdown("##### Vector Flux")
                fig_f = go.Figure()
                colors = ['#00E676' if s == 'Super_Bull' else '#FF1744' if s == 'Super_Bear' else '#546E7A' for s in df['State']]
                fig_f.add_trace(go.Bar(x=df.index, y=df['Flux'], marker_color=colors, name="Flux"))
                fig_f.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_f, use_container_width=True)

        # TAB 3: DEEP ANALYTICS (SEASONALITY, CORRELATION, MONTE CARLO)
        with tab_deep:
            c_d1, c_d2 = st.columns(2)
            
            with c_d1:
                st.markdown("##### 🧬 Temporal DNA (Seasonality)")
                if seas_m is not None:
                    fig_seas = px.bar(x=seas_m.index, y=seas_m.values, labels={'x':'Month', 'y':'Avg Return %'}, color=seas_m.values, color_continuous_scale="RdBu")
                    st.plotly_chart(fig_seas, use_container_width=True)
                if seas_d is not None:
                    st.dataframe(seas_d.style.background_gradient(cmap="Greens"), use_container_width=True)
            
            with c_d2:
                st.markdown("##### 🔮 Monte Carlo Forecast (30 Day)")
                fig_mc = go.Figure()
                for i in range(min(50, len(mc_sim.columns))):
                    fig_mc.add_trace(go.Scatter(y=mc_sim[i], mode='lines', line=dict(color='rgba(255,255,255,0.05)'), showlegend=False))
                fig_mc.add_trace(go.Scatter(y=mc_sim.mean(axis=1), mode='lines', line=dict(color='#00E5FF', width=2), name="Mean Path"))
                st.plotly_chart(fig_mc, use_container_width=True)
                
            st.markdown("##### 🔗 Asset Correlations")
            if corr is not None:
                st.dataframe(corr.to_frame().T.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1), use_container_width=True)

        # TAB 4: FUNDAMENTALS
        with tab_fund:
            if fund:
                c_f1, c_f2, c_f3 = st.columns(3)
                c_f1.metric("Market Cap", f"${fund['Market Cap']:,}")
                c_f2.metric("P/E Ratio", f"{fund['P/E Ratio']}")
                c_f3.metric("Revenue Growth", f"{fund['Rev Growth']*100:.1f}%")
                st.caption(f"**Business Summary:** {fund['Summary']}")
            else:
                st.info("Fundamental data not available for this asset class.")

        # TAB 5: AI ANALYST
        with tab_ai:
            if st.button("Generate Intelligence Report", type="primary"):
                if api_key:
                    with st.spinner("Analyzing market physics..."):
                        prompt = f"""
                        Analyze {ticker} ({interval}).
                        Technical Data:
                        - Structure: {"Bullish" if last['Trend'] == 1 else "Bearish"} (Cloud Support: {last['Cloud_Lower']:.2f})
                        - Physics: Entropy {last['Entropy']:.2f} ({ent_label}), RQZO {last['RQZO']:.2f}.
                        - Flux State: {last['State']} (Value: {last['Flux']:.2f}).
                        - Volume Control: {poc:.2f}.
                        
                        Task: Provide a professional trading assessment.
                        1. Is the trend supported by Physics (Flux)?
                        2. Is the market chaotic (High Entropy)?
                        3. Where is the key liquidity (POC/Order Blocks)?
                        Keep it concise.
                        """
                        try:
                            client = OpenAI(api_key=api_key)
                            resp = client.chat.completions.create(model="gpt-4", messages=[{"role":"user", "content":prompt}])
                            st.success(resp.choices[0].message.content)
                        except Exception as e:
                            st.error(f"AI Error: {e}")
                else:
                    st.warning("Please enter OpenAI API Key in Sidebar.")

if __name__ == "__main__":
    main()
