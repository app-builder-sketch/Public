import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Momentum Structural Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to clean up the UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@st.cache_data
def fetch_data(ticker, period="10y"):
    """
    Fetches historical data from Yahoo Finance.
    """
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty:
            return None
        return df
    except Exception as e:
        return None

def resample_data(df, timeframe):
    """
    Resamples daily data to Weekly (W) or Monthly (M) to avoid 'daily zig-zags'.
    """
    if timeframe == "Weekly":
        # Resample to Weekly, taking the last Close, but Max High and Min Low
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        df_resampled = df.resample('W-FRI').agg(agg_dict).dropna()
    elif timeframe == "Monthly":
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        df_resampled = df.resample('ME').agg(agg_dict).dropna()
    else:
        df_resampled = df
    
    return df_resampled

def calculate_indicators(df, roc_period, smooth_period):
    """
    Calculates Momentum (Rate of Change) and Moving Averages.
    """
    data = df.copy()
    
    # Calculate Rate of Change (Momentum)
    # ROC = ((Close - Close_n) / Close_n) * 100
    data['ROC'] = ((data['Close'] - data['Close'].shift(roc_period)) / data['Close'].shift(roc_period)) * 100
    
    # Calculate a Signal Line (SMA of ROC) to smooth out noise
    data['ROC_Signal'] = data['ROC'].rolling(window=smooth_period).mean()
    
    # Calculate Long Term Trend Baseline (e.g., 40-period SMA, roughly 200 days on weekly)
    data['Trend_Baseline'] = data['Close'].rolling(window=40).mean()
    
    return data

def detect_signals(df):
    """
    Detects crossovers of the Momentum line across the Zero line.
    """
    data = df.copy()
    data['Signal'] = 0
    
    # Bullish: ROC crosses above 0
    data.loc[(data['ROC'] > 0) & (data['ROC'].shift(1) <= 0), 'Signal'] = 1
    
    # Bearish: ROC crosses below 0
    data.loc[(data['ROC'] < 0) & (data['ROC'].shift(1) >= 0), 'Signal'] = -1
    
    return data

# -----------------------------------------------------------------------------
# Main Application Logic
# -----------------------------------------------------------------------------
def main():
    # --- Sidebar Controls ---
    st.sidebar.title("Configuration")
    
    ticker = st.sidebar.text_input("Ticker Symbol", value="SPY", help="Enter stock or commodity ticker (e.g., SPY, GC=F, BTC-USD)").upper()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Timeframe Settings")
    st.sidebar.markdown("*We focus on long-term trends, avoiding daily noise.*")
    timeframe = st.sidebar.selectbox("Aggregation", ["Weekly", "Monthly"], index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Momentum Parameters")
    roc_period = st.sidebar.slider("Momentum Lookback (Periods)", min_value=5, max_value=52, value=12, help="Standard is 12 for Monthly (1-year momentum) or 12-26 for Weekly.")
    smooth_period = st.sidebar.slider("Momentum Smoothing", min_value=1, max_value=10, value=3, help="Smooths the momentum line to reduce false signals.")

    # --- Main Content ---
    st.markdown('<div class="main-header">Momentum Structure Analyzer</div>', unsafe_allow_html=True)
    st.write(f"""
    Analyzing **{ticker}** on a **{timeframe}** timeframe. 
    This tool visualizes momentum building up or breaking down relative to price, spotting trend shifts before they become obvious.
    """)
    
    # 1. Fetch Data
    raw_data = fetch_data(ticker)
    
    if raw_data is None:
        st.error(f"Could not load data for {ticker}. Please check the symbol.")
        return

    # 2. Process Data
    processed_data = resample_data(raw_data, timeframe)
    processed_data = calculate_indicators(processed_data, roc_period, smooth_period)
    processed_data = detect_signals(processed_data)
    
    # Filter for display (remove NaN from start)
    plot_data = processed_data.dropna()

    # 3. Visualization
    
    # Create Subplots: Row 1 = Price, Row 2 = Momentum
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker} Price Action & Trend Baseline", "Momentum Structure (Rate of Change)")
    )

    # -- Top Chart: Price (Candlestick) --
    fig.add_trace(go.Candlestick(
        x=plot_data.index,
        open=plot_data['Open'],
        high=plot_data['High'],
        low=plot_data['Low'],
        close=plot_data['Close'],
        name="Price"
    ), row=1, col=1)

    # -- Top Chart: Trend Baseline --
    fig.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data['Trend_Baseline'],
        line=dict(color='orange', width=2),
        name="Long-Term Baseline (SMA 40)"
    ), row=1, col=1)

    # -- Markers for Signals --
    # Buy Signals
    buy_signals = plot_data[plot_data['Signal'] == 1]
    fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals['Low'] * 0.98,
        mode='markers',
        marker=dict(symbol='triangle-up', size=10, color='green'),
        name="Bull Start"
    ), row=1, col=1)

    # Sell Signals
    sell_signals = plot_data[plot_data['Signal'] == -1]
    fig.add_trace(go.Scatter(
        x=sell_signals.index,
        y=sell_signals['High'] * 1.02,
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='red'),
        name="Bear Start"
    ), row=1, col=1)

    # -- Bottom Chart: Momentum (ROC) --
    # Zero Line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # ROC Line
    fig.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data['ROC'],
        line=dict(color='#2962FF', width=2),
        name="Momentum (ROC)",
        fill='tozeroy' # Fills to zero line to highlight strength
    ), row=2, col=1)
    
    # Smoothed Signal Line (optional, usually helps spot turns)
    if smooth_period > 1:
        fig.add_trace(go.Scatter(
            x=plot_data.index,
            y=plot_data['ROC_Signal'],
            line=dict(color='rgba(0,0,0,0.5)', width=1, dash='dot'),
            name="Smoothed Momentum"
        ), row=2, col=1)

    # Layout Updates
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Color logic for Momentum Fill
    # Note: Plotly fill color is static in simple add_trace, but visually blue implies momentum.
    
    st.plotly_chart(fig, use_container_width=True)

    # 4. Analysis / Commentary
    st.markdown("### Market Status Analysis")
    
    last_row = plot_data.iloc[-1]
    prev_row = plot_data.iloc[-2]
    
    # Determine current state
    mom_val = last_row['ROC']
    mom_prev = prev_row['ROC']
    price_trend = "Above Baseline" if last_row['Close'] > last_row['Trend_Baseline'] else "Below Baseline"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Momentum (ROC)", f"{mom_val:.2f}%", delta=f"{mom_val - mom_prev:.2f}%")
    
    with col2:
        st.metric("Long-Term Baseline", f"{last_row['Trend_Baseline']:.2f}", delta=price_trend, delta_color="off")
    
    with col3:
        status = "NEUTRAL"
        color = "gray"
        if mom_val > 0 and mom_val > mom_prev:
            status = "BULLISH ACCELERATING"
            color = "green"
        elif mom_val > 0 and mom_val < mom_prev:
            status = "BULLISH WEAKENING"
            color = "orange"
        elif mom_val < 0 and mom_val < mom_prev:
            status = "BEARISH ACCELERATING"
            color = "red"
        elif mom_val < 0 and mom_val > mom_prev:
            status = "BEARISH WEAKENING (Possible Bottom?)"
            color = "blue"
            
        st.markdown(f"**Structure Status:** <span style='color:{color}; font-weight:bold'>{status}</span>", unsafe_allow_html=True)

    st.info("""
    **How to read this chart:**
    1. **Top Panel:** Shows the price. Green triangles indicate where positive momentum was confirmed. Red triangles indicate where momentum broke down.
    2. **Bottom Panel:** This is the 'Momentum Structure'. 
       - Look for **Divergence**: If Price makes a new high, but Momentum makes a lower high, a top is near.
       - **Zero Cross**: Crossing the zero line validates the trend shift.
    """)

    # 5. Data Table
    with st.expander("View Historical Data"):
        st.dataframe(plot_data.sort_index(ascending=False).style.format("{:.2f}"))

if __name__ == "__main__":
    main()
