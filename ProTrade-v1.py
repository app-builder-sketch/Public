import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="ProTrade Mobile Analyst",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed" # Better for mobile default
)

# Custom CSS for Mobile Optimization
st.markdown("""
    <style>
    .stApp { margin-top: -50px; }
    .stMetric { background-color: #1e1e1e; padding: 10px; border-radius: 5px; text-align: center; }
    div[data-testid="stExpander"] details summary { font-weight: bold; font-size: 1.1em; }
    </style>
""", unsafe_allow_html=True)

# --- 2. CORE DATA & INDICATOR ENGINE (NO OMISSIONS) ---
@st.cache_data(ttl=300) # Cache for 5 mins to save mobile data
def fetch_and_calculate(ticker, period, interval):
    # Fetch Data
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception:
        return None

    # --- PURE PANDAS INDICATORS ---
    
    # 1. Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # 2. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD (Moving Average Convergence Divergence)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 4. Bollinger Bands
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (std_dev * 2)
    df['BB_Lower'] = df['SMA_20'] - (std_dev * 2)

    # 5. ATR (Average True Range)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift()).abs()
    tr3 = (df['Low'] - df['Close'].shift()).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    return df.dropna()

# --- 3. STRATEGY LOGIC (10 MODES) ---
def get_signal(row, prev_row, strategy_mode):
    """
    Returns: 1 (Buy), -1 (Sell), 0 (Neutral)
    """
    # 1. Golden Cross (Trend)
    if strategy_mode == "Golden Cross":
        if prev_row['EMA_50'] < prev_row['EMA_200'] and row['EMA_50'] > row['EMA_200']: return 1
        if prev_row['EMA_50'] > prev_row['EMA_200'] and row['EMA_50'] < row['EMA_200']: return -1 # Death Cross

    # 2. Death Cross (Specific Bearish Focus)
    elif strategy_mode == "Death Cross":
        if prev_row['EMA_50'] > prev_row['EMA_200'] and row['EMA_50'] < row['EMA_200']: return -1
    
    # 3. RSI Oversold (Reversion Long)
    elif strategy_mode == "RSI Oversold":
        if row['RSI'] < 30: return 1
        if row['RSI'] > 70: return -1 # Exit/Flip

    # 4. RSI Overbought (Reversion Short)
    elif strategy_mode == "RSI Overbought":
        if row['RSI'] > 70: return -1
        if row['RSI'] < 30: return 1

    # 5. MACD Bullish (Momentum)
    elif strategy_mode == "MACD Bullish":
        if prev_row['MACD'] < prev_row['MACD_Signal'] and row['MACD'] > row['MACD_Signal']: return 1
    
    # 6. MACD Bearish (Momentum)
    elif strategy_mode == "MACD Bearish":
        if prev_row['MACD'] > prev_row['MACD_Signal'] and row['MACD'] < row['MACD_Signal']: return -1

    # 7. Bollinger Bounce (Long)
    elif strategy_mode == "Bollinger Bounce":
        if row['Low'] < row['BB_Lower'] and row['Close'] > row['BB_Lower']: return 1
    
    # 8. Bollinger Rejection (Short)
    elif strategy_mode == "Bollinger Reject":
        if row['High'] > row['BB_Upper'] and row['Close'] < row['BB_Upper']: return -1

    # 9. EMA 20 Pullback (Trend Continuation)
    elif strategy_mode == "EMA 20 Pullback":
        # Only trade if trend is UP (Price > 200 EMA)
        if row['Close'] > row['EMA_200']:
            if row['Low'] <= row['SMA_20'] and row['Close'] > row['SMA_20']: return 1

    # 10. ATR Breakout (Volatility)
    elif strategy_mode == "ATR Breakout":
        # Price breaks out of Bollinger AND Volatility is high
        if row['Close'] > row['BB_Upper'] and row['TR'] > row['ATR']: return 1
        if row['Close'] < row['BB_Lower'] and row['TR'] > row['ATR']: return -1

    return 0

# --- 4. BACKTESTING ENGINE ---
def run_backtest(df, strategy_mode, initial_capital=10000):
    balance = initial_capital
    position = 0 # 0 = Flat, >0 = Units held
    equity_curve = []
    trades = []
    wins = 0
    losses = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        signal = get_signal(row, prev, strategy_mode)
        price = row['Close']

        # BUY Logic
        if position == 0 and signal == 1:
            position = balance / price
            balance = 0
            trades.append({'Date': df.index[i], 'Type': 'BUY', 'Price': price, 'PnL': 0})
        
        # SELL Logic (Signal Flip or simplified exit for backtest)
        elif position > 0 and signal == -1:
            proceeds = position * price
            pnl = proceeds - (trades[-1]['Price'] * position)
            
            if pnl > 0: wins += 1
            else: losses += 1
            
            balance = proceeds
            position = 0
            trades.append({'Date': df.index[i], 'Type': 'SELL', 'Price': price, 'PnL': pnl})

        # Calculate Equity
        current_val = balance + (position * price)
        equity_curve.append(current_val)

    # Close final position
    if position > 0:
        proceeds = position * df['Close'].iloc[-1]
        balance = proceeds
        equity_curve.append(balance)
    
    # Metrics
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    return pd.DataFrame(trades), equity_curve, balance, win_rate

# --- 5. TELEGRAM BROADCASTER ---
def broadcast(token, chat_id, message):
    if not token or not chat_id:
        return False, "Missing API Credentials"
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload)
        if r.status_code == 200:
            return True, "Success"
        else:
            return False, f"API Error: {r.text}"
    except Exception as e:
        return False, str(e)

# --- 6. MAIN UI LAYOUT ---

# A. SIDEBAR / SETTINGS (Mobile: This is in the hamburger menu)
with st.sidebar:
    st.header("⚙️ Config")
    ticker = st.text_input("Asset", "BTC-USD")
    timeframe = st.selectbox("Interval", ["15m", "1h", "4h", "1d"])
    lookback = "1mo" if timeframe in ["15m", "1h"] else "2y"
    
    st.divider()
    
    strategy_mode = st.selectbox("Strategy Mode", [
        "Golden Cross", "Death Cross", 
        "RSI Oversold", "RSI Overbought", 
        "MACD Bullish", "MACD Bearish",
        "Bollinger Bounce", "Bollinger Reject",
        "EMA 20 Pullback", "ATR Breakout"
    ])
    
    st.divider()
    st.subheader("📢 Telegram")
    tg_token = st.text_input("Bot Token", type="password")
    tg_chat = st.text_input("Chat ID")

# B. MAIN CONTENT AREA
st.title("📱 Algo-Analyst")

# Fetch Data
data = fetch_and_calculate(ticker, lookback, timeframe)

if data is None:
    st.error(f"Could not fetch data for {ticker}. Try a different asset.")
else:
    # --- LIVE ANALYSIS TAB ---
    tab_live, tab_backtest, tab_report = st.tabs(["📊 Live", "🛠️ Backtest", "📝 Report"])

    with tab_live:
        # 1. Latest Data Processing
        last_row = data.iloc[-1]
        prev_row = data.iloc[-2]
        
        # Calculate Signal
        current_signal = get_signal(last_row, prev_row, strategy_mode)
        
        # Calculate Targets (Laddered)
        atr = last_row['ATR']
        entry_price = last_row['Close']
        is_long = current_signal == 1
        is_short = current_signal == -1
        
        # Determine targets based on direction (Simplified for display)
        # Defaulting to Long logic for visualization if neutral, or Short if Short signal
        direction_mult = -1 if is_short else 1 
        
        tp1 = entry_price + (atr * 1.5 * direction_mult)
        tp2 = entry_price + (atr * 3.0 * direction_mult)
        tp3 = entry_price + (atr * 5.0 * direction_mult)
        sl = entry_price - (atr * 2.0 * direction_mult)

        # 2. Status Indicators (Mobile Friendly Metrics)
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Price", f"{entry_price:.2f}", f"{(entry_price - prev_row['Close']):.2f}")
        with m2:
            sig_text = "🟢 BUY" if is_long else ("🔴 SELL" if is_short else "⚪ WAIT")
            st.metric("Signal", sig_text, strategy_mode)

        # 3. Mobile Optimized Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Price
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], line=dict(color='orange', width=1), name="50 EMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name="BB Up"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name="BB Low"), row=1, col=1)

        # Lower Indicator (Dynamic)
        if "RSI" in strategy_mode:
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", row=2, col=1)
        elif "MACD" in strategy_mode:
            fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name="MACD Hist"), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(x=data.index, y=data['ATR'], name="ATR"), row=2, col=1)

        # Mobile Chart Layout: No range slider, fixed height
        fig.update_layout(
            height=500, 
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. Action Buttons (Laddered Targets)
        with st.expander("🎯 Active Trade Setup (Click to Expand)", expanded=True):
            c1, c2 = st.columns(2)
            c1.info(f"TP1: {tp1:.2f}")
            c2.info(f"TP2: {tp2:.2f}")
            c1.success(f"TP3: {tp3:.2f}")
            c2.error(f"SL: {sl:.2f}")
            
            if st.button("🚀 Broadcast Signal", type="primary", use_container_width=True):
                msg = (
                    f"🚨 *{strategy_mode.upper()} SIGNAL: {ticker}*\n"
                    f"Action: {sig_text}\n"
                    f"Entry: {entry_price:.2f}\n\n"
                    f"🎯 TP1: {tp1:.2f}\n"
                    f"🎯 TP2: {tp2:.2f}\n"
                    f"🚀 TP3: {tp3:.2f}\n"
                    f"🛑 SL: {sl:.2f} (Trailing)\n"
                )
                success, reason = broadcast(tg_token, tg_chat, msg)
                if success: st.toast("Signal Sent!", icon="✅")
                else: st.toast(f"Failed: {reason}", icon="❌")

    # --- BACKTEST TAB ---
    with tab_backtest:
        st.subheader(f"Test: {strategy_mode}")
        
        if st.button("▶️ Run Backtest", use_container_width=True):
            trades, equity, final_bal, win_rate = run_backtest(data, strategy_mode)
            
            # KPI Cards
            k1, k2, k3 = st.columns(3)
            k1.metric("Final Balance", f"${final_bal:.0f}")
            k2.metric("Win Rate", f"{win_rate:.1f}%")
            k3.metric("Total Trades", len(trades)//2)
            
            # Equity Chart
            st.area_chart(equity, height=250)
            
            # Trade Log
            st.dataframe(trades, use_container_width=True)

    # --- REPORT TAB ---
    with tab_report:
        st.subheader(f"📋 Analyst Report: {ticker}")
        st.markdown(f"""
        **Technical Summary:**
        The asset is currently trading at **{entry_price:.2f}**. 
        
        * **Trend:** {"Bullish" if last_row['EMA_50'] > last_row['EMA_200'] else "Bearish"} (based on EMA alignment).
        * **Volatility:** ATR is {atr:.2f}, indicating {"High" if atr > data['ATR'].mean() else "Normal"} volatility.
        * **RSI:** {last_row['RSI']:.1f} - {"Overbought" if last_row['RSI']>70 else ("Oversold" if last_row['RSI']<30 else "Neutral")}.
        
        **Strategy Note:**
        Using the **{strategy_mode}** strategy, we look for high-probability setups confirmed by momentum. 
        Always adhere to the trailing Stop Loss calculated at **{sl:.2f}**.
        """)
