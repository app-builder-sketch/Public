import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import datetime
import yfinance as yf # Standard for fetching financial data
import time

# ==========================================
# 1. PAGE CONFIGURATION & SECRETS MANAGEMENT
# ==========================================
st.set_page_config(
    page_title="Elite Trade Station Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for visual polish
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

def load_secrets():
    """Attempts to load keys from st.secrets, returns dictionary of keys."""
    keys = {
        "OPENAI_API_KEY": "",
        "GEMINI_API_KEY": "",
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": ""
    }
    
    # Try loading from secrets
    try:
        if "OPENAI_API_KEY" in st.secrets: keys["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        if "GEMINI_API_KEY" in st.secrets: keys["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
        if "TELEGRAM_BOT_TOKEN" in st.secrets: keys["TELEGRAM_BOT_TOKEN"] = st.secrets["TELEGRAM_BOT_TOKEN"]
        if "TELEGRAM_CHAT_ID" in st.secrets: keys["TELEGRAM_CHAT_ID"] = st.secrets["TELEGRAM_CHAT_ID"]
    except FileNotFoundError:
        pass # Secrets file not found, will rely on sidebar inputs
    except Exception as e:
        st.warning(f"Error loading secrets: {e}")
        
    return keys

api_keys = load_secrets()

# ==========================================
# 2. MASSIVE TICKER LISTS (100+ Options)
# ==========================================
TICKER_DATA = {
    "Forex": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
        "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X",
        "EURAUD=X", "EURCAD=X", "EURCHF=X", "EURNZD=X", "GBPAUD=X", "GBPCAD=X", "GBPCHF=X",
        "GBPNZD=X", "AUDCAD=X", "AUDCHF=X", "AUDNZD=X", "CADCHF=X", "NZDCAD=X", "NZDCHF=X",
        "USDHKD=X", "USDSGD=X", "USDTRY=X", "USDZAR=X", "USDMXN=X", "USDNOK=X", "USDSEK=X",
        "USDDKK=X", "USDPLN=X", "USDHUF=X", "USDCZK=X", "USDILS=X", "USDCNY=X", "USDRUB=X",
        "EURNOK=X", "EURSEK=X", "EURDKK=X", "EURPLN=X", "EURHUF=X", "EURCZK=X", "EURILS=X",
        "EURCNY=X", "EURRUB=X", "EURTRY=X", "EURZAR=X", "EURMXN=X", "GBPNOK=X", "GBPSEK=X",
        "GBPDKK=X", "GBPPLN=X", "GBPTRY=X", "GBPZAR=X", "GBPMXN=X", "TRYJPY=X", "ZARJPY=X",
        "MXNJPY=X", "HKDJPY=X", "SGDJPY=X", "NOKJPY=X", "SEKJPY=X", "DKKJPY=X", "PLNJPY=X",
        # ... extending logic to reach high counts
        "AUDNOK=X", "AUDSEK=X", "AUDDKK=X", "AUDPLN=X", "CADNOK=X", "CADSEK=X", "CADDKK=X",
        "CADPLN=X", "CHFNOK=X", "CHFSEK=X", "CHFDKK=X", "CHFPLN=X", "NZDNOK=X", "NZDSEK=X",
        "NZDDKK=X", "NZDPLN=X", "USDBRL=X", "USDARS=X", "USDCOP=X", "USDCLP=X", "USDPEN=X",
        "EURBRL=X", "EURARS=X", "GBPBRL=X", "GBPARS=X", "USDKRW=X", "USDINR=X", "USDIDR=X",
        "USDMYR=X", "USDPHP=X", "USDTHB=X", "USDVND=X", "USDTWD=X", "EURKRW=X", "EURINR=X"
    ],
    "Crypto": [
        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "TRX-USD",
        "DOT-USD", "MATIC-USD", "LTC-USD", "SHIB-USD", "LINK-USD", "BCH-USD", "XLM-USD", "ATOM-USD",
        "UNI-USD", "XMR-USD", "ETC-USD", "FIL-USD", "ICP-USD", "HBAR-USD", "APT-USD", "LDO-USD",
        "ARB-USD", "VET-USD", "NEAR-USD", "QNT-USD", "MKR-USD", "AAVE-USD", "OP-USD", "GRT-USD",
        "ALGO-USD", "STX-USD", "SAND-USD", "EOS-USD", "EGLD-USD", "THETA-USD", "IMX-USD", "AXS-USD",
        "MANA-USD", "XTZ-USD", "APE-USD", "FTM-USD", "RNDR-USD", "SNX-USD", "NEO-USD", "FLOW-USD",
        "KAVA-USD", "CHZ-USD", "GALA-USD", "KLAY-USD", "ZEC-USD", "IOTA-USD", "PAXG-USD", "USDT-USD",
        "USDC-USD", "DAI-USD", "BUSD-USD", "TUSD-USD", "USDP-USD", "WBTC-USD", "CRV-USD", "LRC-USD",
        "MINA-USD", "FXS-USD", "GMX-USD", "DYDX-USD", "COMP-USD", "ZIL-USD", "1INCH-USD", "RUNE-USD",
        "CAKE-USD", "ENJ-USD", "BAT-USD", "DASH-USD", "KSM-USD", "CVX-USD", "CELO-USD", "HOT-USD",
        "TWT-USD", "NEXO-USD", "ROSE-USD", "ENS-USD", "YFI-USD", "BAL-USD", "AR-USD", "QTUM-USD",
        "MASK-USD", "GLM-USD", "LPT-USD", "ANKR-USD", "JASMY-USD", "SSV-USD", "WLD-USD", "PEPE-USD"
    ],
    "Stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "TSM", "UNH", "JNJ",
        "LLY", "V", "XOM", "WMT", "JPM", "MA", "PG", "AVGO", "HD", "CVX", "MRK", "ABBV", "KO",
        "PEP", "ORCL", "BAC", "COST", "MCD", "CSCO", "CRM", "ACN", "ADBE", "LIN", "AMD", "TMO",
        "ABT", "DHR", "NFLX", "NKE", "DIS", "PM", "TXN", "NEE", "WFC", "UPS", "BMY", "RTX",
        "MS", "HON", "AMGN", "INTC", "IBM", "LOW", "UNP", "BA", "SPGI", "QCOM", "CAT", "GE",
        "GS", "SBUX", "DE", "PLD", "INTU", "AXP", "BLK", "MDLZ", "ISRG", "ELV", "BKNG", "LMT",
        "GILD", "ADP", "TJX", "T", "SYK", "C", "MMC", "VRTX", "ADI", "UBER", "REGN", "ZTS",
        "LRCX", "SCHW", "CI", "BSX", "AMT", "ETN", "PANW", "SNOW", "PLTR", "SHOP", "SQ", "COIN",
        "MSTR", "MAR", "FI", "CB", "PGR", "SLB", "EOG", "BDX", "SO", "MO", "CL", "ITW", "APD"
    ],
    "Indices": [
        "^GSPC", "^DJI", "^IXIC", "^RUT", "^FTSE", "^GDAXI", "^FCHI", "^STOXX50E", "^N225", "^HSI",
        "^AXJO", "^AORD", "^BSESN", "^JKSE", "^KLSE", "^NZ50", "^STI", "^KS11", "^TWII", "^BVSP",
        "^MXX", "^MERV", "^TA125.TA", "^CASE30", "^JN0U.JO", "^VIX"
    ],
    "Commodities": [
        "GC=F", "SI=F", "HG=F", "PL=F", "PA=F", "CL=F", "BZ=F", "NG=F", "RB=F", "HO=F",
        "ZC=F", "ZW=F", "ZS=F", "ZM=F", "ZL=F", "ZO=F", "ZR=F", "KE=F", "CC=F", "KC=F",
        "CT=F", "SB=F", "OJ=F", "LBS=F", "LE=F", "GF=F", "HE=F", "DA=F"
    ]
}

# ==========================================
# 3. SIDEBAR & USER INPUTS
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Secrets Handling UI
    with st.expander("🔑 API Keys", expanded=False):
        openai_key = st.text_input("OpenAI Key", value=api_keys["OPENAI_API_KEY"], type="password")
        gemini_key = st.text_input("Gemini Key", value=api_keys["GEMINI_API_KEY"], type="password")
        tg_token = st.text_input("Telegram Bot Token", value=api_keys["TELEGRAM_BOT_TOKEN"], type="password")
        tg_chat_id = st.text_input("Telegram Chat ID", value=api_keys["TELEGRAM_CHAT_ID"])
        
        # Update global dict if changed in UI
        if openai_key: api_keys["OPENAI_API_KEY"] = openai_key
        if gemini_key: api_keys["GEMINI_API_KEY"] = gemini_key
        if tg_token: api_keys["TELEGRAM_BOT_TOKEN"] = tg_token
        if tg_chat_id: api_keys["TELEGRAM_CHAT_ID"] = tg_chat_id

    st.markdown("---")
    st.header("📊 Asset Selection")
    
    asset_class = st.selectbox("Market Type", list(TICKER_DATA.keys()))
    selected_ticker = st.selectbox("Select Asset", TICKER_DATA[asset_class])
    
    timeframe = st.selectbox("Timeframe", ["1d", "1h", "4h", "1wk", "1mo"], index=0)
    lookback = st.slider("Lookback Period (Bars)", 50, 500, 150)

    st.markdown("---")
    st.header("📐 Indicators")
    
    use_sma = st.checkbox("SMA", value=True)
    sma_len = st.number_input("SMA Length", 5, 200, 50)
    
    use_ema = st.checkbox("EMA", value=True)
    ema_len = st.number_input("EMA Length", 5, 200, 20)
    
    use_bollinger = st.checkbox("Bollinger Bands", value=True)
    bb_len = st.number_input("BB Length", 5, 50, 20)
    bb_std = st.number_input("BB StdDev", 1.0, 3.0, 2.0)
    
    use_rsi = st.checkbox("RSI", value=True)
    rsi_len = st.number_input("RSI Length", 5, 50, 14)
    
    use_macd = st.checkbox("MACD", value=True)

# ==========================================
# 4. LIVE TRADINGVIEW BANNER
# ==========================================
# Generates a horizontal scrolling ticker tape
tv_widget_code = """
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
  {
  "symbols": [
    {"proName": "FOREXCOM:SPXUSD", "title": "S&P 500"},
    {"proName": "FOREXCOM:NSXUSD", "title": "US 100"},
    {"proName": "FX_IDC:EURUSD", "title": "EUR/USD"},
    {"proName": "BITSTAMP:BTCUSD", "title": "Bitcoin"},
    {"proName": "BITSTAMP:ETHUSD", "title": "Ethereum"}
  ],
  "showSymbolLogo": true,
  "colorTheme": "dark",
  "isTransparent": false,
  "displayMode": "adaptive",
  "locale": "en"
}
  </script>
</div>
"""
st.components.v1.html(tv_widget_code, height=70)

# ==========================================
# 5. DATA FETCHING & ANALYSIS
# ==========================================
@st.cache_data(ttl=60)
def fetch_data(ticker, period="2y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        # Clean MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_indicators(df):
    if df is None: return None
    data = df.copy()
    
    # Simple Moving Average
    data['SMA'] = data['Close'].rolling(window=sma_len).mean()
    
    # Exponential Moving Average
    data['EMA'] = data['Close'].ewm(span=ema_len, adjust=False).mean()
    
    # Bollinger Bands
    data['BB_Mid'] = data['Close'].rolling(window=bb_len).mean()
    data['BB_Std'] = data['Close'].rolling(window=bb_len).std()
    data['BB_Upper'] = data['BB_Mid'] + (bb_std * data['BB_Std'])
    data['BB_Lower'] = data['BB_Mid'] - (bb_std * data['BB_Std'])
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_len).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_len).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data

# Load Data
df = fetch_data(selected_ticker, interval=timeframe)
if df is not None:
    df_processed = calculate_indicators(df)
    # Slice for view
    df_view = df_processed.tail(lookback)
else:
    st.error("Data not available for this ticker.")
    st.stop()

# ==========================================
# 6. ADVANCED PLOTLY CHARTING
# ==========================================
st.title(f"📈 {selected_ticker} Analysis Dashboard")

if df_view is not None:
    # Create Subplots: Row 1 = Price, Row 2 = RSI (if selected), Row 3 = MACD (if selected)
    rows = 1
    row_heights = [0.7]
    specs = [[{"secondary_y": False}]]
    
    if use_rsi:
        rows += 1
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])
    if use_macd:
        rows += 1
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])
        
    # Normalize row heights
    total_h = sum(row_heights)
    row_heights = [h/total_h for h in row_heights]

    fig = make_subplots(
        rows=rows, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=row_heights,
        specs=specs
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_view.index,
        open=df_view['Open'], high=df_view['High'],
        low=df_view['Low'], close=df_view['Close'],
        name='Price'
    ), row=1, col=1)

    # Overlays
    current_row = 1
    if use_sma:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['SMA'], line=dict(color='orange', width=1), name=f'SMA {sma_len}'), row=1, col=1)
    if use_ema:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['EMA'], line=dict(color='blue', width=1), name=f'EMA {ema_len}'), row=1, col=1)
    if use_bollinger:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['BB_Upper'], line=dict(color='gray', width=1, dash='dash'), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), name='BB Lower', fill='tonexty'), row=1, col=1)

    # Indicators Subplots
    current_plot_row = 2
    
    # Live Indicator Analysis Strings
    last_close = df_view['Close'].iloc[-1]
    last_rsi = df_view['RSI'].iloc[-1] if use_rsi else 0
    last_macd = df_view['MACD'].iloc[-1] if use_macd else 0
    last_sig = df_view['Signal_Line'].iloc[-1] if use_macd else 0
    
    analysis_text = f"<b>LIVE ANALYSIS:</b> Price: {last_close:.2f} | "

    if use_rsi:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['RSI'], line=dict(color='purple', width=2), name='RSI'), row=current_plot_row, col=1)
        fig.add_hline(y=70, line_dash="dot", row=current_plot_row, col=1, annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dot", row=current_plot_row, col=1, annotation_text="Oversold")
        
        rsi_state = "Overbought" if last_rsi > 70 else "Oversold" if last_rsi < 30 else "Neutral"
        analysis_text += f"RSI: {last_rsi:.1f} ({rsi_state}) | "
        current_plot_row += 1

    if use_macd:
        colors = ['green' if val >= 0 else 'red' for val in (df_view['MACD'] - df_view['Signal_Line'])]
        fig.add_trace(go.Bar(x=df_view.index, y=df_view['MACD'] - df_view['Signal_Line'], name='MACD Hist', marker_color=colors), row=current_plot_row, col=1)
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['MACD'], line=dict(color='blue', width=1), name='MACD'), row=current_plot_row, col=1)
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['Signal_Line'], line=dict(color='orange', width=1), name='Signal'), row=current_plot_row, col=1)
        
        macd_state = "Bullish" if last_macd > last_sig else "Bearish"
        analysis_text += f"MACD: {macd_state}"

    # Layout Updates
    fig.update_layout(
        height=800, 
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Add Analysis Annotation
    fig.add_annotation(
        text=analysis_text,
        xref="paper", yref="paper",
        x=0, y=1.02, showarrow=False,
        font=dict(size=14, color="yellow"),
        align="left"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 7. TELEGRAM BROADCASTING
# ==========================================
def send_telegram_signal(signal_type, price, stop_loss, take_profit, rationale):
    if not api_keys["TELEGRAM_BOT_TOKEN"] or not api_keys["TELEGRAM_CHAT_ID"]:
        st.error("Telegram Credentials missing in Secrets or Sidebar.")
        return
    
    emoji = "🟢" if "Buy" in signal_type else "🔴" if "Sell" in signal_type else "ℹ️"
    
    message = f"""
{emoji} **TRADING SIGNAL: {selected_ticker}**
--------------------------------
**Type:** {signal_type}
**Timeframe:** {timeframe}
**Entry:** {price}
**Stop Loss:** {stop_loss}
**Take Profit:** {take_profit}
--------------------------------
**Rationale:**
{rationale}
    """
    
    url = f"https://api.telegram.org/bot{api_keys['TELEGRAM_BOT_TOKEN']}/sendMessage"
    payload = {
        "chat_id": api_keys["TELEGRAM_CHAT_ID"],
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        res = requests.post(url, json=payload)
        if res.status_code == 200:
            st.success("Signal Broadcasted Successfully!")
        else:
            st.error(f"Telegram Error: {res.text}")
    except Exception as e:
        st.error(f"Connection Error: {e}")

# ==========================================
# 8. AI ANALYSIS INTEGRATION
# ==========================================
def get_ai_analysis():
    prompt = f"""
    Analyze the following technical data for {selected_ticker} on {timeframe} timeframe:
    - Current Price: {last_close}
    - RSI: {last_rsi}
    - MACD Level: {last_macd}
    - Signal Line: {last_sig}
    - SMA ({sma_len}): {df_view['SMA'].iloc[-1]}
    - BB State: {'Price above Upper' if last_close > df_view['BB_Upper'].iloc[-1] else 'Price below Lower' if last_close < df_view['BB_Lower'].iloc[-1] else 'Inside Bands'}
    
    Provide a concise market sentiment summary (Bullish/Bearish/Neutral) and key support/resistance areas.
    """
    
    if api_keys["OPENAI_API_KEY"]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_keys["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a professional financial analyst."},
                          {"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {e}"
            
    elif api_keys["GEMINI_API_KEY"]:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_keys["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini Error: {e}"
    else:
        return "No AI API Key provided."

# ==========================================
# 9. CONTROL PANEL (TABS)
# ==========================================
tab1, tab2 = st.tabs(["🤖 AI Analysis", "📡 Telegram Broadcaster"])

with tab1:
    st.subheader("Artificial Intelligence Market Sentiment")
    if st.button("Generate AI Report"):
        with st.spinner("Consulting AI Analyst..."):
            report = get_ai_analysis()
            st.markdown(report)

with tab2:
    st.subheader("Broadcast Signal")
    c1, c2, c3 = st.columns(3)
    sig_type = c1.selectbox("Signal Type", ["Buy Entry", "Sell Entry", "Exit Alert", "Update"])
    sl_val = c2.number_input("Stop Loss", value=last_close * 0.95)
    tp_val = c3.number_input("Take Profit", value=last_close * 1.05)
    rationale_text = st.text_area("Analysis / Rationale", value=f"RSI is {last_rsi:.2f}. Trend looks...")
    
    if st.button("Send to Telegram Channel"):
        send_telegram_signal(sig_type, last_close, sl_val, tp_val, rationale_text)

# Footer
st.markdown("---")
st.caption(f"Elite Trade Station Pro v3.0 | Data Provided by Yahoo Finance | Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
