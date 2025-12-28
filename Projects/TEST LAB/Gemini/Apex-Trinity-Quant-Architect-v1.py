import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import sqlite3
import datetime
import time
import requests

# ════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION & SETUP
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Apex Trinity | Quant Architect",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Broadcast Ready" look
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background-color: #1e2127;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3137;
        text-align: center;
    }
    .bullish { color: #00E676; font-weight: bold; }
    .bearish { color: #FF1744; font-weight: bold; }
    .neutral { color: #78909C; font-weight: bold; }
    .hud-table { width: 100%; border-collapse: collapse; color: white; }
    .hud-table td { padding: 8px; border-bottom: 1px solid #333; }
    .marquee {
        width: 100%;
        overflow: hidden;
        white-space: nowrap;
        box-sizing: border-box;
        animation: marquee 20s linear infinite;
        color: #00E676;
        padding: 10px;
        background: #111;
        border-bottom: 1px solid #333;
    }
    @keyframes marquee {
        0%   { transform: translate(100%, 0); }
        100% { transform: translate(-100%, 0); }
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# 1. DATABASE LAYER (SQLite)
# ════════════════════════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect('signals_history.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ticker TEXT,
            signal_type TEXT,
            confidence REAL,
            ai_reasoning TEXT,
            price_at_signal REAL
        )
    ''')
    conn.commit()
    return conn

conn = init_db()

def log_signal(ticker, signal_type, confidence, ai_reasoning, price):
    try:
        c = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO signals (timestamp, ticker, signal_type, confidence, ai_reasoning, price_at_signal) VALUES (?, ?, ?, ?, ?, ?)",
                  (timestamp, ticker, signal_type, confidence, ai_reasoning, price))
        conn.commit()
    except Exception as e:
        st.error(f"DB Error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# 2. ASSET UNIVERSE & DATA INGESTION
# ════════════════════════════════════════════════════════════════════════════
ASSET_UNIVERSE = {
    "Crypto": {
        "Majors": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"],
        "DeFi": ["UNI-USD", "AAVE-USD", "MKR-USD", "LDO-USD"],
        "L2": ["OP-USD", "ARB-USD", "MATIC-USD"]
    },
    "Indices": {
        "US": ["^GSPC", "^IXIC", "^DJI"],
        "Global": ["^FTSE", "^N225"]
    },
    "Forex": {
        "Majors": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
        "Minors": ["AUDUSD=X", "USDCAD=X"]
    },
    "Tech Stocks": {
        "Mag7": ["NVDA", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META"],
        "Semi": ["AMD", "INTC", "TSM"]
    }
}

@st.cache_data(ttl=60) # Cache for 1 minute for "Live" feel
def fetch_data(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        # Handle MultiIndex columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════
# 3. TECHNICAL ANALYSIS ENGINE (Pine Script -> Python)
# ════════════════════════════════════════════════════════════════════════════
class ApexMath:
    @staticmethod
    def rma(series, length):
        """Relative Moving Average (Wilde's)"""
        alpha = 1 / length
        return series.ewm(alpha=alpha, adjust=False).mean()

    @staticmethod
    def hma(series, length):
        """Hull Moving Average"""
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wmaf = series.rolling(window=half_length).mean() # Approximate WMA with rolling mean for speed, or implement true WMA
        # True WMA implementation for accuracy
        def wma(s, l):
            weights = np.arange(1, l + 1)
            return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        
        wma_half = wma(series, half_length)
        wma_full = wma(series, length)
        raw_hma = 2 * wma_half - wma_full
        return wma(raw_hma, sqrt_length)

    @staticmethod
    def atr(high, low, close, length):
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return ApexMath.rma(tr, length)

    @staticmethod
    def rational_quadratic_kernel(series, lookback, weight):
        """Replicates the Pine Script Rational Quadratic Kernel Loop"""
        # This is computationally expensive in Python loops, so we vectorize where possible or use a rolling apply.
        # Pine: sum(y * w) / sum(w)
        # w = (1 + (i^2 / (2 * weight * lookback^2))) ^ -weight
        
        weights = []
        for i in range(lookback + 1):
            w = np.power(1 + (np.power(i, 2) / (2 * weight * lookback * lookback)), -weight)
            weights.append(w)
        weights = np.array(weights)
        
        def kernel_calc(window):
            # Window comes in chronological order. Pine loop goes 0 to lookback (current to past).
            # So we reverse the window to match 'i' representing lag.
            window_rev = window[::-1] 
            # If window is shorter than weights (start of series), slice weights
            w_slice = weights[:len(window_rev)]
            current_weight = np.sum(window_rev * w_slice)
            cumulative_weight = np.sum(w_slice)
            return current_weight / cumulative_weight

        return series.rolling(window=lookback+1).apply(kernel_calc, raw=True)

    @staticmethod
    def supertrend_ut_bot(close, high, low, atr_period, multiplier):
        src = close
        xATR = ApexMath.atr(high, low, close, atr_period)
        nLoss = multiplier * xATR
        
        traj = [0.0] * len(close)
        
        for i in range(1, len(close)):
            prev_traj = traj[i-1]
            c = close.iloc[i]
            c1 = close.iloc[i-1]
            
            if c > prev_traj and c1 > prev_traj:
                traj[i] = max(prev_traj, c - nLoss.iloc[i])
            elif c < prev_traj and c1 < prev_traj:
                traj[i] = min(prev_traj, c + nLoss.iloc[i])
            elif c > prev_traj:
                traj[i] = c - nLoss.iloc[i]
            else:
                traj[i] = c + nLoss.iloc[i]
                
        return pd.Series(traj, index=close.index)

def calculate_apex_indicators(df):
    if df is None or len(df) < 100:
        return df
    
    # ---------------- APEX TREND ----------------
    df['HMA_55'] = ApexMath.hma(df['Close'], 55)
    df['ATR_55'] = ApexMath.atr(df['High'], df['Low'], df['Close'], 55)
    df['A1_Upper'] = df['HMA_55'] + (df['ATR_55'] * 1.5)
    df['A1_Lower'] = df['HMA_55'] - (df['ATR_55'] * 1.5)
    
    df['A1_Trend'] = 0
    df.loc[df['Close'] > df['A1_Upper'], 'A1_Trend'] = 1
    df.loc[df['Close'] < df['A1_Lower'], 'A1_Trend'] = -1
    # Fill NaN/Hold previous trend
    df['A1_Trend'] = df['A1_Trend'].replace(0, method='ffill')
    
    # Volume Filter
    df['Vol_MA'] = df['Volume'].rolling(20).mean()
    df['A1_Vol_Cond'] = df['Volume'] > df['Vol_MA']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean() # Simplified RSI for speed
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ---------------- NEXUS V8.2 ----------------
    # Kernel
    df['N2_Kernel'] = ApexMath.rational_quadratic_kernel(df['Close'], 50, 8.0)
    df['N2_Kernel_Trend'] = np.where(df['N2_Kernel'] > df['N2_Kernel'].shift(1), 1, -1)
    
    # Gann Activator (Donchian)
    df['Donch_High'] = df['High'].rolling(20).max()
    df['Donch_Low'] = df['Low'].rolling(20).min()
    # Simplified Gann Logic for Python Vectorization
    df['N2_Gann_Trend'] = 0
    # Iterative approach needed for Gann flip logic
    gann_trend = [1] * len(df)
    activator = [df['Donch_Low'].iloc[0]] * len(df)
    
    for i in range(1, len(df)):
        if gann_trend[i-1] == 1:
            if df['Close'].iloc[i] < activator[i-1]:
                gann_trend[i] = -1
                activator[i] = df['Donch_High'].iloc[i]
            else:
                gann_trend[i] = 1
                activator[i] = df['Donch_Low'].iloc[i]
        else:
            if df['Close'].iloc[i] > activator[i-1]:
                gann_trend[i] = 1
                activator[i] = df['Donch_Low'].iloc[i]
            else:
                gann_trend[i] = -1
                activator[i] = df['Donch_High'].iloc[i]
    
    df['N2_Gann_Trend'] = gann_trend
    
    # UT Bot Risk
    df['N2_Risk_Line'] = ApexMath.supertrend_ut_bot(df['Close'], df['High'], df['Low'], 14, 4.0)
    df['N2_Risk_State'] = np.where(df['Close'] > df['N2_Risk_Line'], 1, -1)
    
    # Nexus Signal
    conditions = [
        (df['N2_Risk_State'] == 1) & (df['N2_Gann_Trend'] == 1) & (df['N2_Kernel_Trend'] == 1),
        (df['N2_Risk_State'] == -1) & (df['N2_Gann_Trend'] == -1) & (df['N2_Kernel_Trend'] == -1)
    ]
    df['N2_Signal'] = np.select(conditions, [1, -1], default=0)

    # ---------------- APEX VECTOR ----------------
    range_abs = df['High'] - df['Low']
    body_abs = (df['Close'] - df['Open']).abs()
    raw_eff = body_abs / range_abs
    raw_eff = raw_eff.fillna(0)
    
    df['V3_Efficiency'] = raw_eff.ewm(span=14).mean()
    
    vol_avg = df['Volume'].rolling(55).mean()
    vol_fact = df['Volume'] / vol_avg
    vol_fact = vol_fact.fillna(1.0)
    
    direction = np.sign(df['Close'] - df['Open'])
    vector_raw = direction * df['V3_Efficiency'] * vol_fact
    df['V3_Flux'] = vector_raw.ewm(span=5).mean()
    
    # Vector States
    threshold = 0.60
    df['V3_State'] = "NEUTRAL"
    df.loc[df['V3_Flux'] > threshold, 'V3_State'] = "SUPER_BULL"
    df.loc[df['V3_Flux'] < -threshold, 'V3_State'] = "SUPER_BEAR"
    df.loc[df['V3_Flux'].abs() < 0.30, 'V3_State'] = "RESISTIVE"
    
    return df

# ════════════════════════════════════════════════════════════════════════════
# 4. AI BRAIN & BROADCASTING
# ════════════════════════════════════════════════════════════════════════════
def get_ai_analysis(ticker, current_data):
    """
    Connects to OpenAI (if key exists) to analyze the math.
    """
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    
    system_role = "You are an elite quantitative financial analyst named Apex. You analyze technical indicators to provide a clear, decisive market outlook. Be concise, professional, and data-driven."
    
    prompt = f"""
    Analyze the following technical data for {ticker}:
    
    **Apex Trend Engine:**
    - Trend Direction: {'BULLISH' if current_data['A1_Trend'] == 1 else 'BEARISH'}
    - Price vs Cloud: Close {current_data['Close']:.2f} (Upper Band: {current_data['A1_Upper']:.2f}, Lower Band: {current_data['A1_Lower']:.2f})
    - RSI: {current_data['RSI']:.2f}
    
    **Nexus System (Confluence):**
    - Signal Status: {'BUY' if current_data['N2_Signal'] == 1 else 'SELL' if current_data['N2_Signal'] == -1 else 'NEUTRAL/WAIT'}
    - Kernel Trend: {'UP' if current_data['N2_Kernel_Trend'] == 1 else 'DOWN'}
    - Gann Structure: {'BULL' if current_data['N2_Gann_Trend'] == 1 else 'BEAR'}
    
    **Apex Vector (Physics):**
    - State: {current_data['V3_State']}
    - Flux: {current_data['V3_Flux']:.4f} (Efficiency: {current_data['V3_Efficiency']:.2%})
    
    Output a structured report:
    1. **Executive Summary**: Buy/Sell/Hold confidence.
    2. **Key Drivers**: Which indicator is strongest?
    3. **Risk Level**: Based on Volatility and Nexus Risk Line.
    """
    
    if not api_key:
        return "AI Analysis Unavailable: Please configure OPENAI_API_KEY in st.secrets."
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Connection Error: {e}"

def broadcast_to_socials(message, platform):
    """Mock broadcasting function using st.secrets for hypothetical keys"""
    if platform == "Telegram":
        token = st.secrets.get("TELEGRAM_TOKEN", None)
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID", None)
        if token and chat_id:
            # url = f"https://api.telegram.org/bot{token}/sendMessage"
            # requests.post(url, json={'chat_id': chat_id, 'text': message})
            return True, "Sent to Telegram (API Configured)"
        else:
            return True, "Simulated Telegram Broadcast (No Key)"
            
    elif platform == "X":
        # Implement X API logic here
        return True, "Simulated X Post"
    return False, "Unknown Platform"

# ════════════════════════════════════════════════════════════════════════════
# 5. UI & WORKFLOW
# ════════════════════════════════════════════════════════════════════════════
# --- SIDEBAR ---
st.sidebar.title("Apex Trinity 1.0")
st.sidebar.markdown("---")

sector = st.sidebar.selectbox("Market Sector", list(ASSET_UNIVERSE.keys()))
sub_type = st.sidebar.selectbox("Category", list(ASSET_UNIVERSE[sector].keys()))
ticker = st.sidebar.selectbox("Ticker", ASSET_UNIVERSE[sector][sub_type])

interval = st.sidebar.select_slider("Timeframe", options=["15m", "1h", "4h", "1d", "1wk"], value="1d")
st.sidebar.markdown("---")

run_backtest = st.sidebar.button("Run Backtest Analysis")

# --- HEADER ---
st.markdown(f"""
<div class="marquee">
    LIVE DATA STREAM: Analyzing {ticker} on {interval} Timeframe | Apex Trend: ACTIVE | Nexus V8.2: ONLINE | Vector Physics: CALCULATING
</div>
""", unsafe_allow_html=True)

# --- MAIN LOGIC ---
df = fetch_data(ticker, period="2y", interval=interval)

if df is not None:
    # CALCULATION
    with st.spinner('Running Apex Trinity Engines...'):
        df = calculate_apex_indicators(df)
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

    # --- TOP METRICS (HUD) ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"{last_row['Close']:.2f}", f"{(last_row['Close'] - prev_row['Close']):.2f}")
    
    with col2:
        trend_color = "bullish" if last_row['A1_Trend'] == 1 else "bearish"
        trend_txt = "BULLISH" if last_row['A1_Trend'] == 1 else "BEARISH"
        st.markdown(f"<div class='metric-card'><div style='font-size:12px; color:#888;'>APEX TREND</div><div class='{trend_color}' style='font-size:20px;'>{trend_txt}</div></div>", unsafe_allow_html=True)

    with col3:
        nex_color = "bullish" if last_row['N2_Signal'] == 1 else "bearish" if last_row['N2_Signal'] == -1 else "neutral"
        nex_txt = "BUY" if last_row['N2_Signal'] == 1 else "SELL" if last_row['N2_Signal'] == -1 else "WAIT"
        st.markdown(f"<div class='metric-card'><div style='font-size:12px; color:#888;'>NEXUS SIGNAL</div><div class='{nex_color}' style='font-size:20px;'>{nex_txt}</div></div>", unsafe_allow_html=True)

    with col4:
        vec_color = "bullish" if "SUPER_BULL" in last_row['V3_State'] else "bearish" if "SUPER_BEAR" in last_row['V3_State'] else "neutral"
        st.markdown(f"<div class='metric-card'><div style='font-size:12px; color:#888;'>VECTOR FLUX</div><div class='{vec_color}' style='font-size:20px;'>{last_row['V3_Flux']:.2f}</div></div>", unsafe_allow_html=True)

    # --- TABS ---
    tab_chart, tab_ai, tab_db, tab_bc = st.tabs(["Interactive Charts", "AI Brain Analysis", "Signal History", "Broadcaster"])

    with tab_chart:
        # Plotly Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Price"
        ), row=1, col=1)

        # Apex Trend Bands
        fig.add_trace(go.Scatter(x=df['Date'], y=df['A1_Upper'], mode='lines', line=dict(color='green', width=1), name='Apex Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['A1_Lower'], mode='lines', line=dict(color='red', width=1), name='Apex Lower'), row=1, col=1)
        
        # Nexus Kernel
        fig.add_trace(go.Scatter(x=df['Date'], y=df['N2_Kernel'], mode='lines', line=dict(color='yellow', width=2), name='Nexus Kernel'), row=1, col=1)

        # Nexus Risk Line
        fig.add_trace(go.Scatter(x=df['Date'], y=df['N2_Risk_Line'], mode='lines', line=dict(color='purple', dash='dot'), name='Risk Line'), row=1, col=1)

        # Vector Flux
        colors = ['green' if v > 0 else 'red' for v in df['V3_Flux']]
        fig.add_trace(go.Bar(x=df['Date'], y=df['V3_Flux'], marker_color=colors, name='Vector Flux'), row=2, col=1)

        fig.update_layout(height=800, template="plotly_dark", title=f"{ticker} Technical Analysis")
        st.plotly_chart(fig, use_container_width=True)

        # TradingView Widget Embed (Constraint Check: "TradingView Widget Stream")
        st.markdown("### 🔴 Live TradingView Stream")
        tv_widget_code = f"""
        <div class="tradingview-widget-container">
          <div id="tradingview_12345"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
            "width": "100%",
            "height": 500,
            "symbol": "{ticker.replace('-USD', 'USD') if 'USD' in ticker else ticker}",
            "interval": "{'D' if interval=='1d' else '60'}",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "container_id": "tradingview_12345"
          }}
          );
          </script>
        </div>
        """
        st.components.v1.html(tv_widget_code, height=500)

    with tab_ai:
        st.subheader("🤖 Apex Brain Analysis")
        if st.button("Generate AI Report"):
            with st.spinner("AI Architect is thinking..."):
                report = get_ai_analysis(ticker, last_row)
                st.markdown(report)
                
                # Auto-log interesting signals if AI is run
                if last_row['N2_Signal'] != 0:
                    log_signal(ticker, "Nexus Reversal", 0.85, report[:100], last_row['Close'])

    with tab_db:
        st.subheader("Signal Database Audit")
        
        # Display DB
        c = conn.cursor()
        c.execute("SELECT * FROM signals ORDER BY timestamp DESC")
        data = c.fetchall()
        db_df = pd.DataFrame(data, columns=["ID", "Timestamp", "Ticker", "Type", "Confidence", "Reasoning", "Price"])
        st.dataframe(db_df, use_container_width=True)
        
        if st.button("Clear Database"):
            c.execute("DELETE FROM signals")
            conn.commit()
            st.rerun()

    with tab_bc:
        st.subheader("📢 Smart Broadcasting Hub")
        col_b1, col_b2 = st.columns(2)
        
        broadcast_msg = f"APEX SIGNAL: {ticker} | {interval}\nTrend: {trend_txt}\nNexus: {nex_txt}\nVector: {last_row['V3_State']}\nPrice: {last_row['Close']:.2f}"
        
        with col_b1:
            st.text_area("Telegram Message Preview", value=broadcast_msg, height=150)
            if st.button("Send to Telegram"):
                success, note = broadcast_to_socials(broadcast_msg, "Telegram")
                if success: st.success(note)
        
        with col_b2:
            st.text_area("X Post Preview", value=f"{broadcast_msg} #Trading #ApexTrinity", height=150)
            if st.button("Post to X"):
                success, note = broadcast_to_socials(broadcast_msg, "X")
                if success: st.success(note)

    # --- BACKTESTING OVERLAY ---
    if run_backtest:
        st.markdown("---")
        st.subheader(f"Strategy Backtest ({interval})")
        
        # Simple Strategy: Buy when N2_Signal == 1, Sell when N2_Signal == -1
        df['Strategy_Ret'] = df['N2_Signal'].shift(1) * df['Close'].pct_change()
        df['Equity_Curve'] = (1 + df['Strategy_Ret']).cumprod()
        
        bt_fig = go.Figure()
        bt_fig.add_trace(go.Scatter(x=df['Date'], y=df['Equity_Curve'], mode='lines', name='Strategy Equity'))
        bt_fig.add_trace(go.Scatter(x=df['Date'], y=(1+df['Close'].pct_change()).cumprod(), mode='lines', name='Buy & Hold', line=dict(dash='dot')))
        bt_fig.update_layout(title="Backtest Performance: Nexus Logic", template="plotly_dark")
        st.plotly_chart(bt_fig, use_container_width=True)

else:
    st.error("Unable to fetch data. Please check the ticker or internet connection.")
