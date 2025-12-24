# app.py
# ⚖️ DARKPOOL BALANCE COMMANDER
# The Grand Unified Theory (Apex + RQZO + SMC)
#
# ARCHITECTURE:
# - Consensus Engine: Weighted voting mechanism (Trend + Cycle + Structure + Entropy).
# - "No One Indicator Rules": Signals require a System Score > Threshold.
# - Vis: Triple-Pane Expert Plotly (Price, Vector, Oscillator).
#
# SETUP:
# pip install streamlit ccxt pandas numpy plotly scipy aiohttp
#
# SECRETS (.streamlit/secrets.toml):
# [general]
# APP_PASSWORD = "..."

import asyncio
import time
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import ccxt.async_support as ccxt

# =========================
# 🎨 UI ARCHITECTURE
# =========================
def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
        
        :root{
            --bg: #000000;
            --card: #0a0a0a;
            --border: #222;
            --neon-green: #00ff41;
            --neon-red: #ff0055;
            --neon-blue: #00f3ff;
            --neon-gold: #ffd700;
            --text-dim: #666;
        }
        .stApp { background-color: var(--bg); font-family: 'Roboto Mono', monospace; color: #e0e0e0; }
        
        /* METRIC CARDS */
        .metric-container {
            background: rgba(10, 10, 10, 0.8);
            border: 1px solid var(--border);
            border-left: 3px solid #444;
            padding: 10px 15px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .metric-label { font-size: 0.8em; color: var(--text-dim); letter-spacing: 1px; text-transform: uppercase; }
        .metric-val { font-size: 1.1em; font-weight: bold; color: white; }
        
        /* BALANCE GAUGE */
        .balance-bar-bg { width: 100%; height: 8px; background: #222; border-radius: 4px; overflow: hidden; margin-top: 5px; }
        .balance-bar-fill { height: 100%; transition: width 0.5s; }
        
        /* SIGNAL CARD */
        .signal-card {
            background: #080808;
            border: 1px solid #333;
            padding: 20px;
            margin-bottom: 20px;
            position: relative;
        }
        .signal-card.BUY { border-top: 4px solid var(--neon-green); box-shadow: 0 -10px 30px -10px rgba(0,255,65,0.1); }
        .signal-card.SELL { border-top: 4px solid var(--neon-red); box-shadow: 0 -10px 30px -10px rgba(255,0,85,0.1); }
        .signal-card.NEUTRAL { border-top: 4px solid #444; }
        
        .card-title { font-size: 1.5em; font-weight: bold; display: flex; justify-content: space-between; }
        .confluence-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 15px; }
        .conf-item { background: #111; padding: 8px; text-align: center; border-radius: 4px; font-size: 0.8em; }
        
        /* GLOW TEXT */
        .glow { text-shadow: 0 0 10px rgba(255,255,255,0.2); }
        .bull-text { color: var(--neon-green); text-shadow: 0 0 10px rgba(0,255,65,0.4); }
        .bear-text { color: var(--neon-red); text-shadow: 0 0 10px rgba(255,0,85,0.4); }
    </style>
    """, unsafe_allow_html=True)

# =========================
# 🔐 SECURITY GATE
# =========================
def check_password():
    if "APP_PASSWORD" not in st.secrets: return True
    if "password_correct" not in st.session_state: st.session_state.password_correct = False
    
    if not st.session_state.password_correct:
        pwd = st.text_input("ENTER SYSTEM KEY:", type="password")
        if pwd == st.secrets["APP_PASSWORD"]:
            st.session_state.password_correct = True
            st.rerun()
        elif pwd:
            st.error("ACCESS DENIED")
        return False
    return True

# =========================
# 🧮 MATH ENGINE: THE UNIFIED FIELD
# =========================
def calc_rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def calc_fractal_dim(series, length=20):
    highest = series.rolling(length).max()
    lowest = series.rolling(length).min()
    range_len = (highest - lowest) / length
    path_len = series.diff().abs().rolling(length).sum() / length
    return (1 + (np.log10(range_len.replace(0, np.nan)) / np.log10(path_len.replace(0, np.nan)))).fillna(1.5)

def calc_entropy(series, length=20):
    returns = series.pct_change().fillna(0)
    def ent(w):
        c, _ = np.histogram(w, bins=5)
        p = c[c>0]/length
        return -np.sum(p * np.log(p))
    return returns.rolling(length).apply(ent, raw=True).fillna(0) / np.log(5)

def run_unified_math(df, params):
    """
    Calculates Apex (Vector), RQZO (Relativity), and SMC (Structure).
    Normalizes outputs for the Consensus Engine.
    """
    src = df['close']
    h, l = df['high'], df['low']
    
    # --- 1. QUANTUM APEX (Vector Physics) ---
    range_abs = h - l
    body_abs = (src - df['open']).abs()
    eff = (body_abs / range_abs.replace(0, 1)).ewm(span=params['apex_len'], adjust=False).mean()
    vol_fact = df['volume'] / df['volume'].rolling(55).mean().replace(0, 1)
    direction = np.sign(src - df['open'])
    
    # Raw Vector
    df['Apex_Flux'] = (direction * eff * vol_fact).ewm(span=5, adjust=False).mean()
    
    # --- 2. RQZO (Relativistic Oscillator) ---
    # Velocity & Gamma
    norm_price = (src - src.rolling(100).min()) / (src.rolling(100).max() - src.rolling(100).min() + 1e-10)
    velocity = norm_price.diff().abs()
    gamma = 1 / np.sqrt(1 - np.minimum(velocity, 0.99)**2).fillna(1.0)
    
    # Fractal & Entropy
    df['FDI'] = calc_fractal_dim(src, 20)
    df['Entropy'] = calc_entropy(src, 20)
    entropy_gate = np.exp(-2.0 * (df['Entropy'] - 0.6).abs())
    
    # Zeta (Simplified vectorized approx for speed)
    # We use a 3-harmonic approx instead of full loop for the Balance Engine speed
    tau = (np.arange(len(df)) % 100) / gamma
    zeta = np.zeros(len(df))
    for n in range(1, 4):
        zeta += (n**-0.5) * np.sin(tau * np.log(n))
    
    df['RQZO'] = zeta * entropy_gate * 10
    
    # --- 3. SMC (Structure) ---
    n = 5
    ph = h.rolling(n*2+1, center=True).max() == h
    pl = l.rolling(n*2+1, center=True).min() == l
    df['Last_PH'] = h.where(ph).ffill()
    df['Last_PL'] = l.where(pl).ffill()
    
    # Break of Structure
    df['BOS_Bull'] = (src > df['Last_PH'].shift(1)) & (h.shift(1) <= df['Last_PH'].shift(1))
    df['BOS_Bear'] = (src < df['Last_PL'].shift(1)) & (l.shift(1) >= df['Last_PL'].shift(1))
    
    return df

# =========================
# ⚖️ CONSENSUS ENGINE (BALANCE LOGIC)
# =========================
def calculate_balance_score(df, params):
    """
    "No One Indicator Rules"
    Normalizes all signals and calculates a weighted consensus.
    """
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    scores = {}
    
    # 1. Apex Score (-100 to 100) | Weight 30%
    # Thresholds: Super Bull > 0.6, Super Bear < -0.6
    apex_raw = curr['Apex_Flux']
    # Normalize approx -1.0 to 1.0 -> -100 to 100
    scores['Apex'] = np.clip(apex_raw * 150, -100, 100) 
    
    # 2. RQZO Score (-100 to 100) | Weight 30%
    # Oscillates approx -5 to 5
    scores['RQZO'] = np.clip(curr['RQZO'] * 20, -100, 100)
    
    # 3. SMC Score (Binary Step) | Weight 20%
    # Bull Break: +100, Bear Break: -100, Holding: Trend direction
    smc_score = 0
    if curr['BOS_Bull'] or prev['BOS_Bull']: smc_score = 100
    elif curr['BOS_Bear'] or prev['BOS_Bear']: smc_score = -100
    else:
        # Hold previous trend logic
        if curr['close'] > curr['Last_PH']: smc_score = 50
        elif curr['close'] < curr['Last_PL']: smc_score = -50
    scores['SMC'] = smc_score
    
    # 4. Entropy Penalty (Filter) | Weight 20% (Negative Weight)
    # High Entropy reduces the confidence of the entire system
    chaos = curr['Entropy'] # 0.0 to 1.0 (usually)
    # If entropy > 0.8, we penalize the score towards 0
    entropy_factor = 1.0
    if chaos > 0.8: entropy_factor = 0.2 # 80% penalty
    elif chaos > 0.6: entropy_factor = 0.5 # 50% penalty
    
    # --- WEIGHTED SUM ---
    # Raw Balance
    raw_balance = (scores['Apex'] * 0.30) + (scores['RQZO'] * 0.30) + (scores['SMC'] * 0.20)
    
    # Apply Momentum Bias (remaining 20%)
    mom_bias = 20 if curr['close'] > df['close'].iloc[-20] else -20
    raw_balance += mom_bias
    
    # Apply Entropy Gate
    final_balance = raw_balance * entropy_factor
    
    return final_balance, scores, entropy_factor

# =========================
# 📊 DATA LAYER
# =========================
async def run_analysis(ex_id, symbol, tf, params):
    try:
        ex = getattr(ccxt, ex_id)()
        ohlcv = await ex.fetch_ohlcv(symbol, tf, limit=300)
        await ex.close()
        
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        
        df = run_unified_math(df, params)
        score, components, ent_factor = calculate_balance_score(df, params)
        
        return {
            "symbol": symbol, "df": df, "score": score,
            "components": components, "entropy_factor": ent_factor
        }
    except Exception as e:
        return None

# =========================
# 📈 EXPERT PLOTLY
# =========================
def plot_consensus_chart(res):
    df = res['df']
    sym = res['symbol']
    
    # Create 3-Pane Chart
    # 1. Price + SMC
    # 2. Apex Vector (Trend)
    # 3. RQZO (Momentum/Cycle)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02, 
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{sym} [Structure]", "Apex Vector [Trend]", "RQZO [Relativity]")
    )
    
    # Pane 1: Price
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="Price", increasing_line_color='#00ff41', decreasing_line_color='#ff0055'
    ), row=1, col=1)
    
    # Plot SMC Pivots
    fig.add_trace(go.Scatter(x=df['time'], y=df['Last_PH'], mode='lines', line=dict(color='rgba(255,0,85,0.5)', width=1, dash='dot'), name="Structure High"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['Last_PL'], mode='lines', line=dict(color='rgba(0,255,65,0.5)', width=1, dash='dot'), name="Structure Low"), row=1, col=1)
    
    # Pane 2: Apex Vector
    colors = ['#00ff41' if v > 0.6 else '#ff0055' if v < -0.6 else '#444' for v in df['Apex_Flux']]
    fig.add_trace(go.Bar(x=df['time'], y=df['Apex_Flux'], marker_color=colors, name="Apex Flux"), row=2, col=1)
    fig.add_hline(y=0.6, line_color="#333", line_dash="dot", row=2, col=1)
    fig.add_hline(y=-0.6, line_color="#333", line_dash="dot", row=2, col=1)
    
    # Pane 3: RQZO
    fig.add_trace(go.Scatter(x=df['time'], y=df['RQZO'], mode='lines', line=dict(color='#00f3ff', width=2), name="RQZO"), row=3, col=1)
    fig.add_hline(y=0, line_color="#333", row=3, col=1)
    
    # Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000",
        plot_bgcolor="#0a0a0a",
        height=800,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        xaxis_rangeslider_visible=False
    )
    return fig

# =========================
# 🖥️ MAIN DASHBOARD
# =========================
def main():
    st.set_page_config(page_title="DarkPool Balance Commander", layout="wide", page_icon="⚖️")
    inject_css()
    if not check_password(): st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ BALANCE LOGIC")
        ex_id = st.selectbox("Exchange", ["binance", "bybit", "mexc"], index=0)
        symbol = st.text_input("Asset", "BTC/USDT").upper()
        tf = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
        
        st.markdown("---")
        st.markdown("**Thresholds**")
        bal_thresh = st.slider("Confluence Threshold", 50, 90, 70, 5)
        
        params = {"apex_len": 14}
        
        scan_btn = st.button("⚖️ EVALUATE BALANCE", type="primary")
        
    # Main
    st.markdown('<div class="glow" style="font-size:2em; font-weight:bold;">⚖️ BALANCE COMMANDER</div>', unsafe_allow_html=True)
    st.caption("Grand Unified Theory: Apex + RQZO + SMC")
    
    if scan_btn:
        with st.spinner("Synthesizing Market Forces..."):
            res = asyncio.run(run_analysis(ex_id, symbol, tf, params))
            
            if res:
                score = res['score']
                comps = res['components']
                
                # Signal Logic
                signal_type = "NEUTRAL"
                if score > bal_thresh: signal_type = "BUY"
                elif score < -bal_thresh: signal_type = "SELL"
                
                # Colors
                sig_col = "#00ff41" if signal_type == "BUY" else "#ff0055" if signal_type == "SELL" else "#666"
                
                # --- SIGNAL CARD ---
                st.markdown(f"""
                <div class="signal-card {signal_type}">
                    <div class="card-title">
                        <span>{symbol} <span style="font-size:0.6em; color:#888;">{tf}</span></span>
                        <span style="color:{sig_col}">{signal_type}</span>
                    </div>
                    
                    <div class="metric-container" style="margin-top:20px; border-left-color:{sig_col};">
                        <span class="metric-label">SYSTEM BALANCE SCORE</span>
                        <span class="metric-val">{score:.1f} / 100</span>
                    </div>
                    <div class="balance-bar-bg">
                        <div class="balance-bar-fill" style="width:{abs(score)}%; background:{sig_col};"></div>
                    </div>
                    
                    <div class="confluence-grid">
                        <div class="conf-item" style="border-bottom: 2px solid {'#00ff41' if comps['Apex']>0 else '#ff0055'}">
                            <div style="color:#888">APEX</div>
                            <div style="font-weight:bold">{comps['Apex']:.0f}</div>
                        </div>
                        <div class="conf-item" style="border-bottom: 2px solid {'#00f3ff' if comps['RQZO']>0 else '#ff0055'}">
                            <div style="color:#888">RQZO</div>
                            <div style="font-weight:bold">{comps['RQZO']:.0f}</div>
                        </div>
                        <div class="conf-item" style="border-bottom: 2px solid {'#fff' if comps['SMC']==0 else '#00ff41' if comps['SMC']>0 else '#ff0055'}">
                            <div style="color:#888">SMC</div>
                            <div style="font-weight:bold">{comps['SMC']}</div>
                        </div>
                        <div class="conf-item" style="border-bottom: 2px solid {'#ff0055' if res['entropy_factor']<1 else '#00ff41'}">
                            <div style="color:#888">ENTROPY</div>
                            <div style="font-weight:bold">{res['entropy_factor']*100:.0f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # --- CHART ---
                fig = plot_consensus_chart(res)
                st.plotly_chart(fig, use_container_width=True)
                
                # --- RAW DATA ---
                with st.expander("Consensus Matrix"):
                    st.dataframe(res['df'].tail(20))
            
            else:
                st.error("Connection Failed.")

if __name__ == "__main__":
    main()
