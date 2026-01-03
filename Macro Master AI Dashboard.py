import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import openai
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Macro Master AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- TICKER MAPPING (TV -> Yahoo Finance) ---
# This maps the TradingView specific syntax from your list to Yahoo Finance tickers
TICKER_MAP = {
    # MACRO
    "TVC:US10Y": "^TNX", "TVC:US02Y": "^IRX", "TVC:VIX": "^VIX", "TVC:DXY": "DX-Y.NYB",
    "TVC:UKX": "^FTSE", "TVC:DAX": "^GDAXI", "TVC:NI225": "^N225",
    
    # COMMODITIES
    "COMEX:GC1!": "GC=F", "COMEX:SI1!": "SI=F", "COMEX:HG1!": "HG=F",
    "NYMEX:CL1!": "CL=F", "NYMEX:NG1!": "NG=F", "ICE:BRN1!": "BZ=F",
    
    # INDICES & EQUITIES
    "SP:SPX": "^GSPC", "NASDAQ:NDX": "^NDX", "DJ:DJI": "^DJI", "RUSSELL:RUT": "^RUT",
    "NASDAQ:NVDA": "NVDA", "NASDAQ:AAPL": "AAPL", "NASDAQ:MSFT": "MSFT", 
    "NASDAQ:AMZN": "AMZN", "NASDAQ:TSLA": "TSLA", "NASDAQ:META": "META",
    "NASDAQ:GOOGL": "GOOGL", "NASDAQ:AMD": "AMD", "NASDAQ:COIN": "COIN",
    
    # CRYPTO
    "MEXC:BTCUSDT": "BTC-USD", "MEXC:ETHUSDT": "ETH-USD", "MEXC:SOLUSDT": "SOL-USD",
    "MEXC:XRPUSDT": "XRP-USD", "MEXC:DOGEUSDT": "DOGE-USD",
}

# Categories for the Sidebar
CATEGORIES = {
    "Macro & Bonds": ["TVC:US10Y", "TVC:US02Y", "TVC:VIX", "TVC:DXY"],
    "Indices": ["SP:SPX", "NASDAQ:NDX", "DJ:DJI", "RUSSELL:RUT", "TVC:UKX", "TVC:NI225"],
    "Commodities": ["COMEX:GC1!", "COMEX:SI1!", "NYMEX:CL1!", "NYMEX:NG1!"],
    "Tech Giants": ["NASDAQ:NVDA", "NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:TSLA", "NASDAQ:GOOGL"],
    "Crypto": ["MEXC:BTCUSDT", "MEXC:ETHUSDT", "MEXC:SOLUSDT", "MEXC:XRPUSDT"]
}

# --- HELPER FUNCTIONS ---

@st.cache_data(ttl=300)
def get_data(ticker_symbol, period="1y", interval="1d"):
    """Fetches historical data and calculates basic TA."""
    try:
        data = yf.download(ticker_symbol, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        
        # Flatten MultiIndex columns if present (common in new yfinance)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Basic Technical Indicators using Pandas TA
        data.ta.rsi(length=14, append=True)
        data.ta.macd(append=True)
        data.ta.sma(length=50, append=True)
        data.ta.sma(length=200, append=True)
        data.ta.bbands(length=20, append=True)
        data.ta.atr(length=14, append=True)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def get_fundamental_info(ticker_symbol):
    """Fetches fundamental info (PE, Market Cap, etc.)."""
    try:
        stock = yf.Ticker(ticker_symbol)
        return stock.info
    except:
        return {}

def generate_ai_analysis(api_key, ticker, price, change, rsi, macd_status, trend, fundamentals):
    """Generates a report using OpenAI."""
    if not api_key:
        return "⚠️ Please enter an OpenAI API Key in the sidebar to generate the report."
    
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""
    You are a professional Wall Street Global Macro Strategist. Write a concise but deep technical and fundamental analysis report for {ticker}.
    
    **Current Market Data:**
    - Price: {price}
    - Change: {change}
    - Trend (50/200 SMA): {trend}
    - RSI (14): {rsi} (Overbought > 70, Oversold < 30)
    - MACD Status: {macd_status}
    
    **Fundamental Data:**
    {fundamentals}
    
    **Instructions:**
    1. Analyze the technical structure (Trend, Momentum).
    2. Analyze the fundamental context (Valuation, Sector health).
    3. Provide a clear "Bullish", "Bearish", or "Neutral" bias with a confidence level.
    4. Suggest Key Support and Resistance levels based on the data.
    5. Format with Markdown, use bolding for key figures, and keep it professional.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Or gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "You are a helpful financial analyst assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating AI report: {str(e)}"

# --- MAIN LAYOUT ---

# Sidebar
st.sidebar.title("⚙️ Control Panel")
api_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password")
selected_category = st.sidebar.selectbox("Select Category", list(CATEGORIES.keys()))
selected_tv_ticker = st.sidebar.selectbox("Select Asset", CATEGORIES[selected_category])

# Map to YF Ticker
yf_ticker = TICKER_MAP.get(selected_tv_ticker, selected_tv_ticker) # Fallback to original if not mapped

st.sidebar.markdown("---")
st.sidebar.info(f"**Mapping:**\n{selected_tv_ticker} ➡️ {yf_ticker}")

# Main Content
st.title(f"📊 {selected_tv_ticker} Analysis Dashboard")

# Fetch Data
with st.spinner('Fetching market data...'):
    df = get_data(yf_ticker)
    info = get_fundamental_info(yf_ticker)

if df is not None:
    # Latest Values
    last_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    change_pct = ((last_close - prev_close) / prev_close) * 100
    rsi_val = df['RSI_14'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    sma_200 = df['SMA_200'].iloc[-1]
    
    # Determine Trend
    trend = "Bullish (Price > SMA50)" if last_close > sma_50 else "Bearish (Price < SMA50)"
    if sma_50 > sma_200:
        trend += " & Golden Cross Active"
    elif sma_50 < sma_200:
        trend += " & Death Cross Active"

    # Determine MACD Status
    macd = df['MACD_12_26_9'].iloc[-1]
    macdsignal = df['MACDs_12_26_9'].iloc[-1]
    macd_status = "Bullish Crossover" if macd > macdsignal else "Bearish Divergence"

    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"{last_close:.2f}", f"{change_pct:.2f}%")
    col2.metric("RSI (14)", f"{rsi_val:.1f}", delta=None)
    col3.metric("Trend", "Bullish" if last_close > sma_200 else "Bearish")
    col4.metric("Volatility (ATR)", f"{df['ATRr_14'].iloc[-1]:.2f}")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📈 Technical Charts", "🤖 AI Analyst Report", "📋 Fundamental Data"])

    with tab1:
        # Candlestick Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1), name='SMA 200'))
        
        fig.update_layout(title=f'{selected_tv_ticker} Price Action', height=600, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Indicators
        st.subheader("Key Levels")
        st.dataframe(df.tail(5)[['Close', 'RSI_14', 'SMA_50', 'SMA_200', 'BBU_20_2.0', 'BBL_20_2.0']], use_container_width=True)

    with tab2:
        st.subheader("🧠 Generative AI Market Insight")
        st.markdown("_Click the button below to generate a real-time analysis report based on the live data calculated above._")
        
        if st.button("Generate AI Analysis Report"):
            with st.spinner("AI is analyzing charts and fundamentals..."):
                # Prepare fundamental summary string
                fund_summary = {
                    "Market Cap": info.get('marketCap', 'N/A'),
                    "PE Ratio": info.get('trailingPE', 'N/A'),
                    "Sector": info.get('sector', 'Macro/Crypto'),
                    "Beta": info.get('beta', 'N/A'),
                    "52W High": info.get('fiftyTwoWeekHigh', 'N/A')
                }
                
                report = generate_ai_analysis(
                    api_key, 
                    selected_tv_ticker, 
                    f"{last_close:.2f}", 
                    f"{change_pct:.2f}%", 
                    f"{rsi_val:.1f}", 
                    macd_status, 
                    trend, 
                    fund_summary
                )
                
                st.markdown("### 📝 AI Analyst Report")
                st.markdown("---")
                st.markdown(report)

    with tab3:
        st.subheader("Fundamental Snapshot")
        if info:
            c1, c2 = st.columns(2)
            with c1:
                st.json({k: v for k, v in info.items() if k in ['sector', 'industry', 'marketCap', 'trailingPE', 'forwardPE', 'dividendYield']})
            with c2:
                st.write(info.get('longBusinessSummary', 'No summary available for this asset class.'))
        else:
            st.warning("Fundamental data not available for this asset class (likely Crypto or Index Future).")

else:
    st.error(f"Could not load data for {yf_ticker}. Please check the ticker mapping.")
