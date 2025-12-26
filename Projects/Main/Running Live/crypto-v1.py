import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import requests

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto Tracker",
    page_icon="📈",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Robust Data Fetching Function (Fixes YFinance Error)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)  # Cache data for 10 minutes to prevent rate limiting
def fetch_data(symbol, period="1mo", interval="1d"):
    """
    Fetches data using a custom session to avoid Yahoo Finance blocking bot requests.
    """
    try:
        # Create a session with a fake browser header
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
        })

        # Initialize Ticker with the session
        ticker = yf.Ticker(symbol, session=session)
        
        # Download history
        df = ticker.history(period=period, interval=interval)
        
        # Validation: Check if dataframe is empty
        if df.empty:
            return None, f"No data returned for {symbol}. It may be delisted or the period is too short."
        
        # Clean up data: Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Ensure Date column is datetime
        # Yahoo sometimes returns 'Date' or 'Datetime' depending on interval
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.date
        elif 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)

        return df, None

    except Exception as e:
        return None, str(e)

# -----------------------------------------------------------------------------
# 3. Main Streamlit App Logic
# -----------------------------------------------------------------------------
def main():
    st.title("📈 Live Crypto Dashboard")
    st.markdown("Returns live data from Yahoo Finance with anti-blocking headers.")

    # --- Sidebar Controls ---
    st.sidebar.header("Configuration")
    
    ticker_symbol = st.sidebar.text_input("Ticker Symbol", value="BTC-USD").upper()
    
    time_period = st.sidebar.selectbox(
        "Time Period", 
        options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=2
    )

    interval_mapping = {
        "1d": "1m", "5d": "15m", "1mo": "1d", "3mo": "1d", 
        "6mo": "1wk", "1y": "1wk", "2y": "1mo", "5y": "1mo", "max": "1mo"
    }
    # Allow user to override interval if needed, otherwise auto-select
    selected_interval = interval_mapping.get(time_period, "1d")

    st.sidebar.markdown(f"**Interval selected:** {selected_interval}")
    
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()

    # --- Data Loading ---
    with st.spinner(f'Fetching data for {ticker_symbol}...'):
        df, error_message = fetch_data(ticker_symbol, period=time_period, interval=selected_interval)

    # --- Display Logic ---
    if error_message:
        st.error(f"⚠️ Error: {error_message}")
        st.info("Tip: Try a suffix like .L (London) or .NS (India) if looking for stocks, or check your internet connection.")
    
    elif df is not None:
        # Get latest price and change
        latest_close = df['Close'].iloc[-1]
        previous_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
        delta = latest_close - previous_close
        delta_percent = (delta / previous_close) * 100

        # KPI Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric(label=f"{ticker_symbol} Price", value=f"${latest_close:,.2f}", delta=f"{delta_percent:.2f}%")
        col2.metric(label="High (Period)", value=f"${df['High'].max():,.2f}")
        col3.metric(label="Low (Period)", value=f"${df['Low'].min():,.2f}")

        st.markdown("---")

        # --- Plotting with Plotly ---
        # Switch to Candle or Line based on preference
        chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)

        fig = go.Figure()

        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name=ticker_symbol
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['Close'], 
                mode='lines', name='Close Price',
                line=dict(color='#00CC96', width=2)
            ))

        fig.update_layout(
            title=f'{ticker_symbol} - {time_period} History',
            yaxis_title='Price (USD)',
            xaxis_title='Date',
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- Raw Data Expander ---
        with st.expander("View Raw Data"):
            st.dataframe(df.sort_values(by='Date', ascending=False))
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f'{ticker_symbol}_data.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
