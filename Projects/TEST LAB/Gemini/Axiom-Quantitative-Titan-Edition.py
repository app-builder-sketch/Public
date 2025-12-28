import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from openai import OpenAI
import streamlit.components.v1 as components
from datetime import datetime
from scipy.stats import linregress

# ==========================================
# 1. CORE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Axiom Quantitative | Titan Edition",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. UI/UX ENGINE
# ==========================================
def inject_axiom_css(is_mobile):
    """
    Injects the 'DarkPool/Neon' aesthetic CSS.
    Adapts font sizes and margins based on the Mobile Toggle.
    """
    base_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=SF+Pro+Display:wght@300;500;700&display=swap');
        
        .stApp { background-color: #050505; color: #e0e0e0; font-family: 'SF Pro Display', sans-serif; }
        
        /* TICKER MARQUEE */
        .ticker-wrap {
            width: 100%;
            overflow: hidden;
            background-color: #0a0a0a;
            border-bottom: 1px solid #333;
            white-space: nowrap;
            box-sizing: border-box;
            height: 40px;
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .ticker {
            display: inline-block;
            animation: marquee 45s linear infinite;
        }
        @keyframes marquee {
            0%   { transform: translate(100%, 0); }
            100% { transform: translate(-100%, 0); }
        }
        .ticker-item {
            display: inline-block;
            padding: 0 2rem;
            font-family: 'Roboto Mono', monospace;
            font-size: 0.85rem;
            color: #00F0FF;
            text-shadow: 0 0 5px rgba(0, 240, 255, 0.5);
        }
        
        /* METRICS: Glassmorphism */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.02);
            border-left: 2px solid #333;
            backdrop-filter: blur(10px);
            border-radius: 0px;
            padding: 15px;
            transition: all 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-left: 2px solid #00F0FF;
            background: rgba(255, 255, 255, 0.05);
        }
        div[data-testid="stMetricLabel"] { color: #888; letter-spacing: 1px; text-transform: uppercase; }
        div[data-testid="stMetricValue"] { color: #fff; font-weight: 300; }
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] { gap: 5px; background-color: transparent; border-bottom: 1px solid #222; }
        .stTabs [data-baseweb="tab"] { background-color: transparent; border: none; color: #666; font-size: 0.9rem; }
        .stTabs [aria-selected="true"] { color: #fff; border-bottom: 2px solid #00F0FF; }
        
        /* INPUT FIELDS */
        .stTextInput > div > div > input { color: #00F0FF; background-color: #111; border: 1px solid #333; }
        .stTextArea > div > div > textarea { background-color: #0a0a0a; color: #00F0FF; border: 1px solid #333; font-family: 'Roboto Mono'; }
        
        /* BUTTONS */
        button { border-radius: 4px !important; }

        /* MOBILE REPORT CARDS */
        .report-card {
            background-color: #111;
            border-left: 4px solid #00F0FF;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 10px;
            font-family: 'SF Pro Display', sans-serif;
        }
        .report-header { font-size: 1.1rem; font-weight: bold; color: #fff; margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 5px; }
        .report-item { margin-bottom: 5px; font-size: 0.9rem; color: #aaa; }
        .highlight { color: #00F0FF; font-weight: bold; }
    </style>
    """
    
    desktop_css = """
    <style>
        div[data-testid="stMetricLabel"] { font-size: 0.75rem; }
        .js-plotly-plot .plotly .modebar { orientation: v; left: 0; }
    </style>
    """
    
    mobile_css = """
    <style>
        div[data-testid="stMetric"] {
            margin-bottom: 10px;
            border-left: 4px solid #00F0FF; 
            padding: 20px; 
        }
        div[data-testid="stMetricLabel"] { font-size: 0.9rem; font-weight: 700; }
        div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
        .js-plotly-plot { height: 100% !important; } 
        button { min-height: 55px !important; margin-top: 15px !important; font-size: 1.1rem !important; }
    </style>
    """
    
    st.markdown(base_css, unsafe_allow_html=True)
    if is_mobile:
        st.markdown(mobile_css, unsafe_allow_html=True)
    else:
        st.markdown(desktop_css, unsafe_allow_html=True)

def render_ticker_banner():
    """Renders scrolling marquee of live prices."""
    try:
        tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "SPY", "QQQ", "IWM", "NVDA", "GLD", "USO", "VIX"]
        data = yf.download(tickers, period="1d", interval="1d", progress=False)['Close'].iloc[-1]
        items = []
        for t in tickers:
            if t in data:
                items.append(f"{t}: ${data[t]:,.2f}")
        
        html = f"""
        <div class="ticker-wrap">
            <div class="ticker">
                {" | ".join([f"<span class='ticker-item'>{i}</span>" for i in items])}
                 | {" | ".join([f"<span class='ticker-item'>{i}</span>" for i in items])}
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    except:
        st.markdown("""<div class="ticker-wrap"><div class="ticker"><span class='ticker-item'>💠 AXIOM QUANTITATIVE SYSTEM ONLINE</span></div></div>""", unsafe_allow_html=True)

def render_live_clock():
    """Renders World Clock (NY/Lon/Tok)."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@500&display=swap');
        body { margin: 0; background: transparent; font-family: 'Roboto Mono', monospace; color: #fff; overflow: hidden; }
        .clock-container {
            display: flex; justify-content: space-between; align-items: center;
            background: rgba(0, 0, 0, 0.3); border: 1px solid #222;
            padding: 8px 15px; border-radius: 4px; backdrop-filter: blur(5px);
        }
        .clock-box { text-align: center; width: 32%; }
        .city { font-size: 0.65rem; color: #666; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 2px; }
        .time { font-size: 1.1rem; font-weight: bold; color: #e0e0e0; text-shadow: 0 0 8px rgba(255, 255, 255, 0.1); }
        .accent-ny { border-bottom: 2px solid #00F0FF; }
        .accent-lon { border-bottom: 2px solid #FF0055; }
        .accent-tok { border-bottom: 2px solid #D500F9; }
    </style>
    </head>
    <body>
        <div class="clock-container">
            <div class="clock-box accent-ny"><div class="city">NEW YORK</div><div class="time" id="ny">--:--:--</div></div>
            <div class="clock-box accent-lon"><div class="city">LONDON</div><div class="time" id="lon">--:--:--</div></div>
            <div class="clock-box accent-tok"><div class="city">TOKYO</div><div class="time" id="tok">--:--:--</div></div>
        </div>
        <script>
            function update() {
                const now = new Date();
                const fmt = (tz) => new Intl.DateTimeFormat('en-US', {timeZone: tz, hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false}).format(now);
                document.getElementById('ny').innerText = fmt('America/New_York');
                document.getElementById('lon').innerText = fmt('Europe/London');
                document.getElementById('tok').innerText = fmt('Asia/Tokyo');
            }
            setInterval(update, 1000); update();
        </script>
    </body>
    </html>
    """
    components.html(html, height=80)

# ==========================================
# 3. DATA SERVICE
# ==========================================
class DataService:
    @staticmethod
    def get_asset_classes():
        return [
            "Crypto (Major)", "Crypto (Alt/Meme)", 
            "Indices & ETFs", "US Tech (Mag 7+)", 
            "US Large Cap", "High Volatility", 
            "Commodities & Forex"
        ]

    @staticmethod
    def get_tickers_by_class(asset_class):
        universe = {
            "Crypto (Major)": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "TRX-USD", "LINK-USD", "MATIC-USD", "LTC-USD", "BCH-USD", "UNI-USD", "XLM-USD", "ATOM-USD", "XMR-USD", "ETC-USD", "FIL-USD", "HBAR-USD", "ICP-USD", "APT-USD", "NEAR-USD", "QNT-USD", "ALGO-USD"],
            "Crypto (Alt/Meme)": ["SHIB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FLOKI-USD", "ARB-USD", "OP-USD", "RNDR-USD", "INJ-USD", "STX-USD", "IMX-USD", "VET-USD", "MKR-USD", "AAVE-USD", "GRT-USD", "SAND-USD", "MANA-USD", "AXS-USD", "THETA-USD", "EOS-USD", "XTZ-USD", "KAVA-USD", "NEO-USD", "FLOW-USD", "GALA-USD", "CHZ-USD", "CRV-USD", "LDO-USD"],
            "Indices & ETFs": ["SPY", "QQQ", "IWM", "DIA", "TLT", "VXX", "UVXY", "SQQQ", "TQQQ", "SOXL", "SOXS", "XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "ARKK", "SMH", "EEM", "FXI", "EWZ", "GDX", "GDXJ", "JETS", "TAN", "URA", "HYG", "LQD", "TIP", "VNQ", "XLB"],
            "US Tech (Mag 7+)": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX", "ADI", "KLAC", "CRM", "ADBE", "ORCL", "CSCO", "NFLX", "IBM", "NOW", "UBER", "ABNB", "PANW", "SNOW", "PLTR", "CRWD", "DELL", "SMCI"],
            "US Large Cap": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "V", "MA", "AXP", "JNJ", "UNH", "LLY", "PFE", "MRK", "ABBV", "TMO", "DHR", "BMY", "CVS", "PG", "KO", "PEP", "COST", "WMT", "TGT", "HD", "MCD", "NKE", "SBUX", "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "PSX", "VLO", "KMI", "BA", "CAT", "DE", "GE", "HON", "LMT", "RTX", "MMM", "UPS", "UNP"],
            "High Volatility": ["MSTR", "COIN", "MARA", "RIOT", "CLSK", "HUT", "GME", "AMC", "HOOD", "DKNG", "ROKU", "TDOC", "ZM", "PTON", "CVNA", "UPST", "AFRM", "SOFI", "LCID", "RIVN", "NIO", "XPEV", "LI", "BABA", "PDD", "JD", "BIDU", "TCEHY", "BILI", "FUTU", "SAVA"],
            "Commodities & Forex": ["GLD", "SLV", "IAU", "PPLT", "PALL", "USO", "UNG", "DBC", "DBA", "CORN", "WEAT", "EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "BTC=F", "ETH=F"]
        }
        return universe.get(asset_class, [])

    @staticmethod
    def fetch_data(ticker, timeframe, limit=500):
        tf_map = {"15m": "1mo", "1h": "6mo", "4h": "1y", "1d": "2y", "1wk": "5y"}
        period = tf_map.get(timeframe, "1y")
        try:
            # Enhanced safe download based on Terminal logic
            df = yf.download(ticker, period=period, interval=timeframe, progress=False)
            
            # Handle MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                try: 
                    df = df.xs(ticker, axis=1, level=0)
                except: 
                    df.columns = df.columns.get_level_values(0)
            
            # Standardization
            cols = {c: c.capitalize() for c in df.columns}
            df = df.rename(columns=cols)
            if 'Adj close' in df.columns: 
                df['Close'] = df['Adj close']
            
            # Resample 4H if needed (Yahoo doesn't natively support 4h)
            if timeframe == "4h":
                df = df.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
                
            return df.dropna().tail(limit)
        except: return pd.DataFrame()

    @staticmethod
    def get_fundamentals(ticker):
        if "-" in ticker or "=" in ticker: return None
        try:
            stock = yf.Ticker(ticker)
            return {
                "Market Cap": stock.info.get("marketCap", "N/A"),
                "P/E Ratio": stock.info.get("trailingPE", "N/A"),
                "Rev Growth": stock.info.get("revenueGrowth", "N/A"),
                "Debt/Equity": stock.info.get("debtToEquity", "N/A"),
                "Summary": stock.info.get("longBusinessSummary", "No Data")
            }
        except: return None

    @staticmethod
    def get_macro_data():
        assets = {"S&P 500": "SPY", "Nasdaq": "QQQ", "10Y Yield": "^TNX", "VIX": "^VIX", "DXY": "DX-Y.NYB"}
        try:
            data = yf.download(list(assets.values()), period="5d", interval="1d", progress=False)['Close']
            prices = {k: data[v].iloc[-1] for k,v in assets.items() if v in data}
            changes = {k: ((data[v].iloc[-1]-data[v].iloc[-2])/data[v].iloc[-2])*100 for k,v in assets.items() if v in data}
            return prices, changes
        except: return {}, {}

# ==========================================
# 4. QUANTITATIVE ENGINE (PHYSICS & TITAN & GOD MODE)
# ==========================================
class QuantEngine:
    # --- HELPER FUNCTIONS ---
    @staticmethod
    def calculate_wma(series, length):
        return series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length + 1)) / (length * (length + 1) / 2), raw=True)

    @staticmethod
    def calculate_hma(series, length):
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wma_half = QuantEngine.calculate_wma(series, half_length)
        wma_full = QuantEngine.calculate_wma(series, length)
        diff = 2 * wma_half - wma_full
        return QuantEngine.calculate_wma(diff, sqrt_length)
    
    @staticmethod
    def calculate_atr(df, length=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(length).mean()

    @staticmethod
    def calculate_supertrend(df, period=10, multiplier=3):
        atr = QuantEngine.calculate_atr(df, period)
        hl2 = (df['High'] + df['Low']) / 2
        final_upperband = hl2 + (multiplier * atr)
        final_lowerband = hl2 - (multiplier * atr)
        
        # Numba-friendly logic replaced with vectorized approximation for speed/compatibility
        close = df['Close'].values
        upper = final_upperband.values
        lower = final_lowerband.values
        st = np.zeros(len(df))
        trend = np.zeros(len(df)) 
        
        st[0] = lower[0]
        trend[0] = 1
        
        for i in range(1, len(df)):
            if close[i-1] > st[i-1]: 
                st[i] = max(lower[i], st[i-1]) if close[i] > st[i-1] else upper[i]
                trend[i] = 1 if close[i] > st[i-1] else -1
                if close[i] < lower[i] and trend[i-1] == 1: 
                     st[i] = upper[i]
                     trend[i] = -1
            else: 
                st[i] = min(upper[i], st[i-1]) if close[i] < st[i-1] else lower[i]
                trend[i] = -1 if close[i] < st[i-1] else 1
                if close[i] > upper[i] and trend[i-1] == -1: 
                    st[i] = lower[i]
                    trend[i] = 1
        return pd.Series(trend, index=df.index)

    @staticmethod
    def tanh(x): return np.tanh(np.clip(x, -20, 20))

    # --- AXIOM ORIGINAL ENGINES ---
    @staticmethod
    def calc_chedo(df, length=50):
        c = df['Close'].values
        log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
        mu = pd.Series(log_ret).rolling(length).mean().values
        sigma = pd.Series(log_ret).rolling(length).std().values
        v = sigma / (np.abs(mu) + 1e-9)
        abs_ret_v = np.abs(log_ret) * v
        hyper_dist = np.log(abs_ret_v + np.sqrt(abs_ret_v**2 + 1))
        kappa_h = QuantEngine.tanh(pd.Series(hyper_dist).rolling(length).mean().values)
        diff_ret = np.diff(log_ret, prepend=0)
        lyap = np.log(np.abs(diff_ret) + 1e-9)
        lambda_n = QuantEngine.tanh((pd.Series(lyap).rolling(length).mean().values + 5) / 7)
        ent = pd.Series(log_ret**2).rolling(length).sum().values
        ent_n = QuantEngine.tanh(ent * 10)
        raw = (0.4 * kappa_h) + (0.3 * lambda_n) + (0.3 * ent_n)
        df['CHEDO'] = 2 / (1 + np.exp(-raw * 4)) - 1
        return df

    @staticmethod
    def calc_rqzo(df, harmonics=25):
        src = df['Close']
        mn, mx = src.rolling(100).min(), src.rolling(100).max()
        norm = (src - mn) / (mx - mn + 1e-9)
        v = np.abs(norm.diff())
        c_limit = 0.05
        gamma = 1 / np.sqrt(1 - (np.minimum(v, c_limit*0.99)/c_limit)**2)
        idx = np.arange(len(df))
        tau = (idx % 100) / gamma.fillna(1.0)
        zeta = np.zeros(len(df))
        for n in range(1, harmonics + 1):
            amp = n ** -0.5
            theta = tau * np.log(n)
            zeta += amp * np.sin(theta)
        df['RQZO'] = pd.Series(zeta).fillna(0)
        return df

    @staticmethod
    def calc_apex_flux(df, length=14):
        rg = df['High'] - df['Low']
        body = np.abs(df['Close'] - df['Open'])
        eff_raw = np.where(rg == 0, 0, body / rg)
        eff_series = pd.Series(eff_raw, index=df.index) 
        eff_sm = eff_series.ewm(span=length).mean()
        vol_avg = df['Volume'].rolling(55).mean()
        v_rat_raw = np.where(vol_avg == 0, 1, df['Volume'] / vol_avg)
        v_rat_series = pd.Series(v_rat_raw, index=df.index)
        direction = np.sign(df['Close'] - df['Open'])
        raw = direction * eff_sm * v_rat_series
        df['Apex_Flux'] = raw.ewm(span=5).mean()
        df['Apex_State'] = np.select([df['Apex_Flux'] > 0.6, df['Apex_Flux'] < -0.6], ["Super Bull", "Super Bear"], default="Neutral")
        return df

    @staticmethod
    def calc_smc(df, length=55):
        # Original Basic SMC (Retained for backwards compatibility)
        def wma(s, l):
            w = np.arange(1, l+1)
            return s.rolling(l).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
        close = df['Close']
        half = int(length/2); sqrt = int(np.sqrt(length))
        wma_f = wma(close, length); wma_h = wma(close, half)
        df['HMA_Trend'] = wma(2*wma_h - wma_f, sqrt)
        df['Trend_Dir'] = np.where(close > df['HMA_Trend'], 1, -1)
        p = 5
        df['Pivot_H'] = df['High'].rolling(p*2+1, center=True).max() == df['High']
        df['Pivot_L'] = df['Low'].rolling(p*2+1, center=True).min() == df['Low']
        return df

    # --- TITAN & GOD MODE UPGRADES ---
    @staticmethod
    def calc_god_mode_technicals(df):
        """Calculates 10 God Mode Indicators from Terminal."""
        # 1. Apex Trend & Liquidity Master (HMA + ATR Bands)
        apex_mult = 1.5
        df['GM_Apex_Base'] = QuantEngine.calculate_hma(df['Close'], 55)
        df['GM_Apex_ATR'] = QuantEngine.calculate_atr(df, 55)
        df['GM_Apex_Upper'] = df['GM_Apex_Base'] + (df['GM_Apex_ATR'] * apex_mult)
        df['GM_Apex_Lower'] = df['GM_Apex_Base'] - (df['GM_Apex_ATR'] * apex_mult)
        df['GM_Apex_Trend'] = np.where(df['Close'] > df['GM_Apex_Upper'], 1, np.where(df['Close'] < df['GM_Apex_Lower'], -1, 0))
        df['GM_Apex_Trend'] = df['GM_Apex_Trend'].replace(to_replace=0, method='ffill')

        # 2. Squeeze Momentum (LazyBear)
        df['Sqz_Basis'] = df['Close'].rolling(20).mean()
        df['Sqz_Dev'] = df['Close'].rolling(20).std() * 2.0
        df['Sqz_Upper_BB'] = df['Sqz_Basis'] + df['Sqz_Dev']
        df['Sqz_Lower_BB'] = df['Sqz_Basis'] - df['Sqz_Dev']
        df['Sqz_Ma_KC'] = df['Close'].rolling(20).mean()
        df['Sqz_Range_MA'] = QuantEngine.calculate_atr(df, 20)
        df['Sqz_Upper_KC'] = df['Sqz_Ma_KC'] + (df['Sqz_Range_MA'] * 1.5)
        df['Sqz_Lower_KC'] = df['Sqz_Ma_KC'] - (df['Sqz_Range_MA'] * 1.5)
        df['Squeeze_On'] = (df['Sqz_Lower_BB'] > df['Sqz_Lower_KC']) & (df['Sqz_Upper_BB'] < df['Sqz_Upper_KC'])
        
        # Linear Regression Momentum for Squeeze
        highest = df['High'].rolling(20).max()
        lowest = df['Low'].rolling(20).min()
        avg_val = (highest + lowest + df['Sqz_Ma_KC']) / 3
        df['Sqz_Mom'] = (df['Close'] - avg_val).rolling(20).mean() * 100 

        # 3. Money Flow Matrix
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_src = (rsi - 50) * 2 # Normalize
        mf_vol = df['Volume'] / df['Volume'].rolling(14).mean()
        df['MF_Matrix'] = (rsi_src * mf_vol).ewm(span=3).mean()

        # 4. Dark Vector (SuperTrend)
        df['DarkVector_Trend'] = QuantEngine.calculate_supertrend(df, 10, 4.0)

        # 5. Gann High Low
        gann_len = 3
        df['Gann_High'] = df['High'].rolling(gann_len).mean()
        df['Gann_Low'] = df['Low'].rolling(gann_len).mean()
        df['Gann_Trend'] = np.where(df['Close'] > df['Gann_High'].shift(1), 1, np.where(df['Close'] < df['Gann_Low'].shift(1), -1, 0))
        df['Gann_Trend'] = df['Gann_Trend'].replace(to_replace=0, method='ffill')

        # 6. EVWM (Elastic Volume Weighted Momentum)
        ev_len = 21
        ev_base = QuantEngine.calculate_hma(df['Close'], ev_len)
        ev_atr = QuantEngine.calculate_atr(df, ev_len)
        ev_elast = (df['Close'] - ev_base) / ev_atr
        ev_force = np.sqrt((df['Volume'] / df['Volume'].rolling(20).mean()).ewm(span=5).mean())
        df['EVWM'] = ev_elast * ev_force

        # GOD MODE SCORE
        df['GM_Score'] = (
            df['GM_Apex_Trend'] + 
            df['Gann_Trend'] + 
            df['DarkVector_Trend'] + 
            np.sign(df['Sqz_Mom'])
        )
        return df

    @staticmethod
    def calc_fib_targets(df):
        """Titan Mobile Fibonacci Calculation."""
        lookback = 50
        recent = df.iloc[-lookback:]
        h, l = recent['High'].max(), recent['Low'].min()
        d = h - l
        last = df.iloc[-1]
        
        # Direction based on God Mode or Basic Trend
        is_bull = last.get('GM_Apex_Trend', last.get('Trend_Dir', 1)) == 1
        
        # Smart Stop & Targets
        stop = last['HMA_Trend'] if 'HMA_Trend' in df else (l if is_bull else h)
        risk = abs(last['Close'] - stop)
        if risk == 0: risk = last['Close'] * 0.01

        return {
            'fib_382': h - (d * 0.382),
            'fib_618': h - (d * 0.618),
            'tp1': last['Close'] + (risk * 1.5) if is_bull else last['Close'] - (risk * 1.5),
            'tp2': last['Close'] + (risk * 3.0) if is_bull else last['Close'] - (risk * 3.0),
            'tp3': last['Close'] + (risk * 5.0) if is_bull else last['Close'] - (risk * 5.0),
            'smart_stop': stop
        }

    @staticmethod
    def calc_fear_greed_v4(df):
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain/loss))
        macd = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['FG_Index'] = (rsi + (macd * 10)).clip(0, 100).rolling(5).mean()
        return df

    @staticmethod
    def calc_volume_profile(df, bins=50):
        price_min = df['Low'].min(); price_max = df['High'].max()
        price_bins = np.linspace(price_min, price_max, bins)
        df['Mid'] = (df['Close'] + df['Open']) / 2
        df['Bin'] = pd.cut(df['Mid'], bins=price_bins, labels=price_bins[:-1], include_lowest=True)
        vp = df.groupby('Bin', observed=False)['Volume'].sum().reset_index()
        vp['Price'] = vp['Bin'].astype(float)
        poc_idx = vp['Volume'].idxmax()
        poc_price = vp.loc[poc_idx, 'Price']
        return vp, poc_price

    @staticmethod
    def calc_smc_advanced(df, internal_len=5, swing_len=20):
        """
        Upgraded SMC Engine with LuxAlgo Features:
        - FVG (Fair Value Gaps)
        - Order Blocks
        - Equal Highs/Lows (EQH/EQL)
        - Premium/Discount Zones
        """
        smc = {'structures': [], 'order_blocks': [], 'fvgs': [], 'eqh_eql': [], 'zones': {}}
        
        # 1. Structures (BOS/CHoCH) - Internal
        df['PH'] = df['High'].rolling(internal_len*2+1, center=True).max() == df['High']
        df['PL'] = df['Low'].rolling(internal_len*2+1, center=True).min() == df['Low']
        
        # Swing Structure (for Premium/Discount)
        df['Swing_PH'] = df['High'].rolling(swing_len*2+1, center=True).max() == df['High']
        df['Swing_PL'] = df['Low'].rolling(swing_len*2+1, center=True).min() == df['Low']
        
        # Identify last Swing High/Low for Premium/Discount
        last_swing_h = None
        last_swing_l = None
        
        # Iteration for Structures & Order Blocks
        last_h = None; last_l = None
        
        # EQH/EQL Threshold (e.g., 0.1 ATR)
        atr_val = df['GM_Apex_ATR'].iloc[-1] if 'GM_Apex_ATR' in df else (df['High']-df['Low']).mean()
        eq_thresh = atr_val * 0.1

        for i in range(swing_len, len(df)):
            # Internal Structure
            if df['PH'].iloc[i]:
                if last_h: 
                    smc['structures'].append({'x0':last_h[0], 'y':last_h[1], 'x1':df.index[i], 'color':'red', 'type': 'Structure'})
                    # Check EQH
                    if abs(last_h[1] - df['High'].iloc[i]) < eq_thresh:
                        smc['eqh_eql'].append({'x': df.index[i], 'y': df['High'].iloc[i], 'type': 'EQH'})
                last_h = (df.index[i], df['High'].iloc[i])
            
            if df['PL'].iloc[i]:
                if last_l: 
                    smc['structures'].append({'x0':last_l[0], 'y':last_l[1], 'x1':df.index[i], 'color':'green', 'type': 'Structure'})
                    # Check EQL
                    if abs(last_l[1] - df['Low'].iloc[i]) < eq_thresh:
                        smc['eqh_eql'].append({'x': df.index[i], 'y': df['Low'].iloc[i], 'type': 'EQL'})
                last_l = (df.index[i], df['Low'].iloc[i])

            # Track Swings for Premium/Discount
            if df['Swing_PH'].iloc[i]: last_swing_h = df['High'].iloc[i]
            if df['Swing_PL'].iloc[i]: last_swing_l = df['Low'].iloc[i]

        # 2. Fair Value Gaps (Vectorizedish)
        # Bullish FVG: Low[i] > High[i-2]
        # Bearish FVG: High[i] < Low[i-2]
        for i in range(2, len(df)):
            if df['Low'].iloc[i] > df['High'].iloc[i-2]: # Bullish FVG
                smc['fvgs'].append({
                    'x0': df.index[i-2], 'x1': df.index[i], 
                    'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i], 
                    'color': 'rgba(0, 255, 104, 0.3)'
                })
            if df['High'].iloc[i] < df['Low'].iloc[i-2]: # Bearish FVG
                smc['fvgs'].append({
                    'x0': df.index[i-2], 'x1': df.index[i], 
                    'y0': df['Low'].iloc[i-2], 'y1': df['High'].iloc[i], 
                    'color': 'rgba(255, 0, 8, 0.3)'
                })

        # 3. Order Blocks (Simplified)
        if last_l:
            mask = df['PL']
            if mask.any():
                idx = df[mask].index[-1]
                row = df.loc[idx]
                smc['order_blocks'].append({
                    'x0': idx, 'x1': df.index[-1], 
                    'y0': row['Low'], 'y1': row['High'], 
                    'color': 'rgba(33, 87, 243, 0.4)'
                })
        if last_h:
            mask = df['PH']
            if mask.any():
                idx = df[mask].index[-1]
                row = df.loc[idx]
                smc['order_blocks'].append({
                    'x0': idx, 'x1': df.index[-1], 
                    'y0': row['Low'], 'y1': row['High'], 
                    'color': 'rgba(255, 0, 0, 0.4)'
                })
        
        # 4. Premium / Discount Zones
        if last_swing_h and last_swing_l:
            mid = (last_swing_h + last_swing_l) / 2
            smc['zones'] = {
                'high': last_swing_h,
                'low': last_swing_l,
                'mid': mid
            }

        return smc

    @staticmethod
    def run_monte_carlo(df, days=30, sims=100):
        last_price = df['Close'].iloc[-1]
        returns = df['Close'].pct_change().dropna()
        mu = returns.mean(); sigma = returns.std()
        sim_rets = np.random.normal(mu, sigma, (days, sims))
        paths = np.zeros((days, sims)); paths[0] = last_price
        for t in range(1, days): paths[t] = paths[t-1] * (1 + sim_rets[t])
        return paths

    @staticmethod
    def calc_day_of_week_dna(ticker):
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            df['Day'] = df.index.day_name()
            df['Ret'] = df['Close'].pct_change() * 100
            stats = df.groupby('Day')['Ret'].mean()
            return stats
        except: return None

    @staticmethod
    def calc_mtf_trend(ticker):
        try:
            trends = {}
            for tf in ['1d', '1wk']:
                df = yf.download(ticker, period="1y", interval=tf, progress=False)
                sma50 = df['Close'].rolling(50).mean().iloc[-1]
                trends[tf] = "BULL" if df['Close'].iloc[-1] > sma50 else "BEAR"
            return trends
        except: return {}

# ==========================================
# 5. INTELLIGENCE (AI & BROADCAST)
# ==========================================
class Intelligence:
    @staticmethod
    def get_secret(key):
        try: return st.secrets.get(key, "")
        except: return ""

    @staticmethod
    def generate_quick_signal(df, ticker, fibs):
        """Generates the Normal/Quick trade signal (Short format)."""
        last = df.iloc[-1]
        trend = "🐂 BULLISH" if last.get('GM_Apex_Trend', 0) == 1 else "🐻 BEARISH"
        
        flux = last['Apex_Flux']
        if flux > 0.6: flux_txt = "🟢 SUPER BULL"
        elif flux < -0.6: flux_txt = "🔴 SUPER BEAR"
        else: flux_txt = "⚪ NEUTRAL"
        
        gm_score = last.get('GM_Score', 0)
        
        return f"""⚡ *TITAN GOD MODE SIGNAL*
💎 Asset: *{ticker}*
💰 Price: ${last['Close']:,.2f}
🔮 God Mode Score: {gm_score:.0f}/4
🌊 Trend: {trend}
⚛️ Flux: {flux_txt} ({flux:.2f})

🎯 *EXECUTION*
🛑 Stop: {fibs['smart_stop']:.2f}
1️⃣ TP1: {fibs['tp1']:.2f}
2️⃣ TP2: {fibs['tp2']:.2f}
3️⃣ TP3: {fibs['tp3']:.2f}
"""

    @staticmethod
    def generate_mobile_card(last, ticker, fibs, fg_index):
        is_bull = last.get('GM_Apex_Trend', 0) == 1
        direction = "LONG 🐂" if is_bull else "SHORT 🐻"
        
        # God Mode Logic for Confidence
        gm_score = last.get('GM_Score', 0)
        conf = "MAX 🔥" if abs(gm_score) >= 3 else "HIGH" if abs(gm_score) >= 2 else "LOW"
        
        return f"""
        <div class="report-card">
            <div class="report-header">💠 SIGNAL: {direction}</div>
            <div class="report-item">Asset: <span class="highlight">{ticker}</span></div>
            <div class="report-item">Titan Score: <span class="highlight">{gm_score:.0f}/4 ({conf})</span></div>
            <div class="report-item">Sentiment: <span class="highlight">{fg_index:.0f}/100</span></div>
        </div>
        <div class="report-card">
            <div class="report-header">🎯 TITAN EXECUTION</div>
            <div class="report-item">Entry: <span class="highlight">{last['Close']:.2f}</span></div>
            <div class="report-item">🛑 STOP: <span class="highlight">{fibs['smart_stop']:.2f}</span></div>
            <div class="report-item">1️⃣ TP1: <span class="highlight">{fibs['tp1']:.2f}</span></div>
            <div class="report-item">2️⃣ TP2: <span class="highlight">{fibs['tp2']:.2f}</span></div>
            <div class="report-item">3️⃣ TP3: <span class="highlight">{fibs['tp3']:.2f}</span></div>
        </div>
        """

    @staticmethod
    def generate_signal_report(df, ticker):
        last = df.iloc[-1]
        trend_emoji = "🐂 BULLISH" if last.get('GM_Apex_Trend', 0) == 1 else "🐻 BEARISH"
        
        flux_val = last['Apex_Flux']
        flux_emoji = "🟢 Superconductor" if abs(flux_val) > 0.6 else "⚪ Neutral"
        
        chedo_val = last['CHEDO']
        entropy_status = "⚠️ CRITICAL CHAOS" if abs(chedo_val) > 0.8 else "✅ Stable State"
        
        rqzo_val = last['RQZO']
        
        # God Mode Metrics
        squeeze_txt = "💥 SQUEEZE FIRING" if last['Sqz_Mom'] > 0 and last['Squeeze_On'] else "💤 Squeeze Off"
        money_flow = f"{last['MF_Matrix']:.2f}"
        
        return f"""🚨 *AXIOM TITAN REPORT* 🚨
📅 {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
💎 Asset: *{ticker}*
💰 Price: *${last['Close']:,.2f}*

🌊 *GOD MODE TECHNICALS*
• Titan Score: {last['GM_Score']:.0f}/4
• Trend: {trend_emoji}
• Squeeze: {squeeze_txt}
• Money Flow: {money_flow}

⚛️ *QUANTUM PHYSICS*
• Flux Vector: {flux_emoji} ({flux_val:.2f})
• Entropy (CHEDO): {entropy_status} ({chedo_val:.2f})
• Relativity (RQZO): {rqzo_val:.2f}

🛡️ *STRATEGIC OUTLOOK*
Regime: {"High-Energy" if abs(rqzo_val) > 1 else "Low-Energy"} {trend_emoji.split()[1]}
"""

    @staticmethod
    def analyze(df, ticker, api_key):
        if not api_key: return "❌ Missing OpenAI API Key."
        last = df.iloc[-1]
        
        # God Mode Data
        gm_score = last.get('GM_Score', 0)
        apex_trend = "BULL" if last.get('GM_Apex_Trend') == 1 else "BEAR"
        
        system_prompt = """
        You are 'Axiom', a Tier-1 Quantitative Physicist and Market Maker.
        Analyze using FIRST PRINCIPLES.
        
        New Inputs:
        - God Mode Score (Max 4): Aggregate of Trend, Gann, Vector, and Momentum.
        - Apex Trend: The primary directional bias.
        
        OUTPUT:
        ### ⚛️ Quantum State & Explanations
        ### 🛡️ Risk Physics
        ### 🚀 Titan Strategy Options
        """
        user_prompt = f"Asset: {ticker} | Price: {last['Close']} | GM Score: {gm_score} | Apex Trend: {apex_trend} | CHEDO: {last['CHEDO']:.3f} | RQZO: {last['RQZO']:.3f} | Flux: {last['Apex_Flux']:.3f}"
        
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e: return f"❌ AI Error: {str(e)}"

    @staticmethod
    def broadcast_telegram(message, bot_token, chat_id):
        if not bot_token or not chat_id: return False, "❌ Missing Telegram Credentials"
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        try:
            max_len = 4000
            if len(message) <= max_len:
                requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
            else:
                for i in range(0, len(message), max_len):
                    chunk = message[i:i+max_len]
                    requests.post(url, json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"})
            return True, "✅ Broadcast Sent"
        except Exception as e: return False, f"❌ Broadcast Failed: {str(e)}"

# ==========================================
# 6. GRAPHICS
# ==========================================
class Graphics:
    @staticmethod
    def render_dashboard(df, ticker, is_mobile):
        # Updated Row Heights for God Mode
        row_heights = [0.4, 0.1, 0.1, 0.1, 0.15, 0.15] 
        total_height = 1000 if not is_mobile else 1400
        
        fig = make_subplots(
            rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
            row_heights=row_heights, 
            subplot_titles=("Price & SMC", "Entropy (CHEDO)", "Relativity (RQZO)", "Money Flow", "Squeeze Mom", "Flux")
        )

        # P1: Price + SMC + Apex Cloud (God Mode)
        smc = QuantEngine.calc_smc_advanced(df)
        
        # Premium / Discount Zones
        if 'zones' in smc and smc['zones']:
            # Discount Zone (Green tint at bottom)
            fig.add_shape(type="rect", x0=df.index[0], x1=df.index[-1], 
                          y0=smc['zones']['low'], y1=smc['zones']['mid'], 
                          fillcolor="green", opacity=0.05, layer="below", line_width=0, row=1, col=1)
            # Premium Zone (Red tint at top)
            fig.add_shape(type="rect", x0=df.index[0], x1=df.index[-1], 
                          y0=smc['zones']['mid'], y1=smc['zones']['high'], 
                          fillcolor="red", opacity=0.05, layer="below", line_width=0, row=1, col=1)

        # Apex Cloud (God Mode)
        fig.add_trace(go.Scatter(x=df.index, y=df['GM_Apex_Upper'], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['GM_Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.05)', line=dict(width=0), name="Apex Cloud"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['GM_Apex_Base'], line=dict(color='yellow', width=1), name="Apex Base"), row=1, col=1)

        # Candlesticks
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)

        # SMC Overlays (FVG/OB/Structure)
        for s in smc['structures']:
            fig.add_shape(type="line", x0=s['x0'], x1=s['x1'], y0=s['y'], y1=s['y'], line=dict(color=s['color'], width=1, dash="dot"), row=1, col=1)
        for ob in smc['order_blocks']:
            fig.add_shape(type="rect", x0=ob['x0'], x1=ob['x1'], y0=ob['y0'], y1=ob['y1'], fillcolor=ob['color'], opacity=0.3, line_width=0, row=1, col=1)
        for fvg in smc['fvgs']:
            fig.add_shape(type="rect", x0=fvg['x0'], x1=fvg['x1'], y0=fvg['y0'], y1=fvg['y1'], fillcolor=fvg['color'], opacity=0.3, line_width=0, row=1, col=1)
            
        # EQH/EQL Markers
        for eq in smc['eqh_eql']:
            fig.add_annotation(x=eq['x'], y=eq['y'], text=eq['type'], showarrow=True, arrowhead=1, row=1, col=1, font=dict(color='white', size=8))

        # P2: CHEDO
        fig.add_hrect(y0=0.8, y1=1.2, row=2, col=1, fillcolor="#FF0055", opacity=0.15, line_width=0)
        fig.add_hrect(y0=-1.2, y1=-0.8, row=2, col=1, fillcolor="#00F0FF", opacity=0.15, line_width=0)
        fig.add_trace(go.Scatter(x=df.index, y=df['CHEDO'], line=dict(color='#00F0FF', width=2), fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.05)', name="Entropy"), row=2, col=1)

        # P3: RQZO
        fig.add_trace(go.Scatter(x=df.index, y=df['RQZO'], line=dict(color='#D500F9', width=1.5), fill='tozeroy', fillcolor='rgba(213, 0, 249, 0.05)', name="Relativity"), row=3, col=1)
        
        # P4: Money Flow Matrix
        fig.add_trace(go.Scatter(x=df.index, y=df['MF_Matrix'], fill='tozeroy', line=dict(color='cyan', width=1), name="Money Flow"), row=4, col=1)

        # P5: Squeeze Momentum
        colors_sqz = ['#00E676' if v > 0 else '#FF5252' for v in df['Sqz_Mom']]
        fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=colors_sqz, name="Squeeze"), row=5, col=1)

        # P6: Flux
        colors_flux = np.where(df['Apex_Flux'] > 0.6, '#00E676', np.where(df['Apex_Flux'] < -0.6, '#FF1744', np.where(df['Apex_Flux'] > 0, '#2979FF', '#B0BEC5')))
        fig.add_trace(go.Bar(x=df.index, y=df['Apex_Flux'], marker_color=colors_flux, marker_line_width=0, name="Flux"), row=6, col=1)

        fig.update_layout(height=total_height, template="plotly_dark", paper_bgcolor="#050505", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=40 if not is_mobile else 0, t=20, b=0), hovermode="x unified", showlegend=False, font=dict(family="SF Pro Display, Roboto Mono, sans-serif", size=12))
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)', zeroline=False)
        fig.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 7. MAIN APP
# ==========================================
def main():
    st.sidebar.markdown("## 💠 AXIOM TITAN")
    is_mobile = st.sidebar.toggle("📱 Mobile Optimized", value=True)
    inject_axiom_css(is_mobile)
    
    render_ticker_banner()
    render_live_clock()
    
    with st.sidebar.expander("🔐 API Credentials", expanded=False):
        val_ai = Intelligence.get_secret("OPENAI_API_KEY")
        val_tk = Intelligence.get_secret("TELEGRAM_TOKEN")
        val_ch = Intelligence.get_secret("TELEGRAM_CHAT_ID")
        openai_key = st.text_input("OpenAI Key", value=val_ai, type="password")
        tg_token = st.text_input("Telegram Token", value=val_tk, type="password")
        tg_chat = st.text_input("Chat ID", value=val_ch)

    st.sidebar.markdown("### Market Data")
    asset_class = st.sidebar.selectbox("Sector", DataService.get_asset_classes())
    ticker = st.sidebar.selectbox("Ticker", DataService.get_tickers_by_class(asset_class))
    tf = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=2)
    
    with st.spinner("Processing Physics & God Mode Engines..."):
        df = DataService.fetch_data(ticker, tf)
        if df.empty:
            st.error("Market Data Unavailable.")
            return
        try:
            # Physics Engines
            df = QuantEngine.calc_chedo(df)
            df = QuantEngine.calc_rqzo(df)
            df = QuantEngine.calc_apex_flux(df)
            df = QuantEngine.calc_smc(df) # Basic SMC for legacy support
            df = QuantEngine.calc_fear_greed_v4(df)
            
            # God Mode Engines (Upgrades)
            df = QuantEngine.calc_god_mode_technicals(df)
            
            # Titan Logic
            fibs = QuantEngine.calc_fib_targets(df)
            
            last = df.iloc[-1]
            fund = DataService.get_fundamentals(ticker)
            macro_p, macro_c = DataService.get_macro_data()
        except Exception as e:
            st.error(f"Computation Error: {e}"); return

    # Titan Mobile Report Card
    if is_mobile:
        st.markdown(Intelligence.generate_mobile_card(last, ticker, fibs, last.get('FG_Index', 50)), unsafe_allow_html=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Titan Score", f"{last['GM_Score']:.0f}/4")
        c2.metric("Entropy", f"{last['CHEDO']:.2f}", delta="Risk" if abs(last['CHEDO'])>0.7 else "Stable")
        c3.metric("Relativity", f"{last['RQZO']:.2f}")
        c4.metric("Flux", f"{last['Apex_Flux']:.2f}", delta=last['Apex_State'])

    # EXTENDED TABS
    tabs = st.tabs(["📉 Titan Tech", "🌍 Macro", "📅 Seasonality", "🧠 AI", "📊 Volume", "🔮 Quant", "📡 Broadcast"])
    
    with tabs[0]: Graphics.render_dashboard(df, ticker, is_mobile)
    with tabs[1]:
        st.subheader("Global Macro Context")
        c1, c2 = st.columns(2)
        c1.metric("S&P 500", f"${macro_p.get('S&P 500',0):.2f}", f"{macro_c.get('S&P 500',0):.2f}%")
        c2.metric("VIX", f"{macro_p.get('VIX',0):.2f}", f"{macro_c.get('VIX',0):.2f}%")
        if fund:
            st.write(f"**{ticker} Fundamentals**: Cap {fund['Market Cap']} | PE {fund['P/E Ratio']}")
    with tabs[2]:
        st.subheader("Day of Week DNA")
        dna = QuantEngine.calc_day_of_week_dna(ticker)
        if dna is not None: st.bar_chart(dna)
    with tabs[3]:
        if st.button("Run Intelligence", use_container_width=is_mobile):
            res = Intelligence.analyze(df, ticker, openai_key)
            st.markdown(res)
    with tabs[4]:
        vp, poc = QuantEngine.calc_volume_profile(df)
        st.bar_chart(vp.set_index('Price')['Volume'])
        st.caption(f"POC Level: {poc:.2f}")
    with tabs[5]:
        mc = QuantEngine.run_monte_carlo(df)
        st.line_chart(mc[:, :20]) # Show first 20 paths
    with tabs[6]:
        h = 500 if is_mobile else 650
        tv_sym = ticker.replace("-", "").replace("=X", "").replace("=F", "")
        html = f"""<div class="tradingview-widget-container"><div id="tradingview_widget"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{"width": "100%", "height": {h}, "symbol": "{tv_sym}", "interval": "D", "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "en", "toolbar_bg": "#f1f3f6", "enable_publishing": false, "hide_side_toolbar": false, "allow_symbol_change": true, "details": true, "container_id": "tradingview_widget"}});</script></div>"""
        st.components.v1.html(html, height=h)
        
        st.subheader("📡 Signal Broadcaster")
        st.markdown("#### ⚡ Quick Signal")
        default_quick = Intelligence.generate_quick_signal(df, ticker, fibs)
        msg_quick = st.text_area("Payload", value=default_quick, height=150, key="quick")
        if st.button("🚀 Send Trade Signal", use_container_width=is_mobile):
            success, info = Intelligence.broadcast_telegram(msg_quick, tg_token, tg_chat)
            if success: st.success(info)
            else: st.error(info)
            
        st.divider()
        st.markdown("#### 📄 Detailed Report & Education")
        default_report = Intelligence.generate_signal_report(df, ticker)
        msg_report = st.text_area("Report Payload", value=default_report, height=350, key="report")
        if st.button("📨 Send Detailed Report", use_container_width=is_mobile):
            success, info = Intelligence.broadcast_telegram(msg_report, tg_token, tg_chat)
            if success: st.success(info)
            else: st.error(info)

if __name__ == "__main__":
    main()
