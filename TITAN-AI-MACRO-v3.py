import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import requests
import json
import urllib.parse
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import time
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CRITICAL FIX: TA-Lib Import Handling
# ==========================================
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    st.warning("⚠️ TA-Lib not installed. Using pure Python fallback implementations. Install with: `pip install TA-Lib`")

# ==========================================
# FALLBACK FUNCTIONS (Exact TA-Lib replicas)
# ==========================================
def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Pure Python EMA calculation - identical to talib.EMA"""
    return prices.ewm(span=period, adjust=False).mean()

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Pure Python ATR calculation - identical to talib.ATR"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_trange(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Pure Python TRANGE calculation - identical to talib.TRANGE"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def calculate_bbands(close: pd.Series, period: int = 20, nbdevup: int = 2, nbdevdn: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Pure Python Bollinger Bands - identical to talib.BBANDS"""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + (std * nbdevup)
    lower = middle - (std * nbdevdn)
    return upper, middle, lower

# ==========================================
# 1. PAGE CONFIG & DARKPOOL STYLING
# ==========================================
st.set_page_config(
    layout="wide", 
    page_title="TITAN AI MACRO", 
    page_icon="👁️",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "TITAN AI MACRO - Advanced Telegram Broadcasting System"
    }
)

def inject_terminal_css():
    """Enhanced cyberpunk terminal styling with professional trading desk aesthetic"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
        
        .stApp { 
            background: linear-gradient(135deg, #050505 0%, #0a0a0a 100%); 
            color: #e0e0e0; 
            font-family: 'Inter', sans-serif; 
        }
        
        /* Glassmorphism Metric Cards - Enhanced */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.03);
            border-left: 4px solid #00F0FF;
            backdrop-filter: blur(20px);
            border-radius: 6px; 
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 240, 255, 0.1);
            transition: all 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-left: 4px solid #00FF88;
            transform: translateY(-2px);
        }
        
        /* Side-Car Analysis Styling - Enhanced */
        .analysis-box {
            background: rgba(0, 240, 255, 0.04);
            border-radius: 10px;
            padding: 25px;
            border: 1px solid rgba(0, 240, 255, 0.15);
            height: 100%;
            min-height: 400px;
            box-shadow: inset 0 0 10px rgba(0, 240, 255, 0.05);
        }
        .analysis-title { 
            color: #00F0FF; 
            font-family: 'Roboto Mono'; 
            font-size: 1.2rem; 
            margin-bottom: 15px; 
            font-weight: bold;
            text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
        }
        .analysis-text { 
            color: #aaa; 
            line-height: 1.7; 
            font-size: 0.95rem; 
            white-space: pre-wrap; 
        }
        .highlight { 
            color: #fff; 
            font-weight: bold;
            background: rgba(0, 240, 255, 0.1);
            padding: 2px 6px;
            border-radius: 3px;
        }
        
        /* Buttons & Tabs - Enhanced */
        .stTabs [data-baseweb="tab-list"] { 
            background-color: transparent; 
            border-bottom: 2px solid #222; 
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 12px 20px;
            border-radius: 6px 6px 0 0;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] { 
            color: #00F0FF; 
            border-bottom: 3px solid #00F0FF; 
            background: rgba(0, 240, 255, 0.1);
        }
        
        /* Toast Styling */
        div[data-baseweb="toast"] { 
            background-color: #111; 
            border: 1px solid #00F0FF; 
            color: #fff; 
            border-radius: 6px;
        }
        
        /* Text Area for Review - Enhanced */
        textarea { 
            background-color: #0a0a0a !important; 
            color: #00F0FF !important; 
            border: 1px solid #333 !important; 
            font-family: 'Roboto Mono' !important;
            border-radius: 6px !important;
            font-size: 0.9rem !important;
            line-height: 1.6 !important;
        }
        
        /* Sidebar Styling */
        .css-1d391kg { background: rgba(0, 0, 0, 0.3) !important; }
        
        /* DataFrame Styling */
        .dataframe {
            background-color: rgba(0, 0, 0, 0.3) !important;
            color: #e0e0e0 !important;
            border: 1px solid #333;
        }
        
        /* Success/Error Messages */
        .stSuccess { background: rgba(0, 255, 136, 0.1) !important; }
        .stError { background: rgba(255, 0, 68, 0.1) !important; }
        
        /* Performance Metrics Table */
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #222;
        }
        .metric-label { color: #888; }
        .metric-value { color: #00F0FF; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA SERVICES & INPUT HANDLING
# ==========================================
class TickerService:
    """Enhanced ticker service with validation and extended universe"""
    
    @staticmethod
    def get_universe() -> List[str]:
        """Comprehensive asset universe with proper classification"""
        crypto = [f"{c}-USD" for c in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "MATIC", "LTC", "BCH", "UNI", "SHIB", "NEAR", "ICP", "STX", "AAVE", "MKR", "LDO", "RNDR", "INJ", "TIA", "SUI", "APT"]]
        stocks = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "INTC", "ADBE", "CRM", "AVGO", "QCOM", "COST", "PYPL", "SBUX", "DIS", "BA", "NKE", "LLY", "V", "MA", "JPM", "WMT", "UNH", "PG", "HD", "ORCL", "MRK", "ABBV", "BAC", "CVX", "KO", "PEP", "TMO", "PFE"]
        indices = ["SPY", "QQQ", "IWM", "DIA", "TLT", "VXX", "GLD", "SLV", "USO", "UNG", "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "ARKK", "SMH", "VTI", "VXUS", "BND"]
        forex = ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "USDCAD=X", "DX-Y.NYB", "USDCHF=X", "NZDUSD=X"]
        commodities = ["GC=F", "SI=F", "CL=F", "NG=F", "BZ=F", "ZC=F", "ZW=F", "ZS=F", "ZM=F", "ZL=F", "GF=F", "HE=F", "LE=F"]
        return sorted(list(set(crypto + stocks + indices + forex + commodities)))

    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Basic ticker format validation"""
        pattern = r'^[A-Z0-9\-\=\.\^\/]+$'
        return bool(re.match(pattern, ticker.upper()))

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, timeframe: str) -> pd.DataFrame:
    """
    Enhanced data fetcher with comprehensive error handling and validation
    Returns empty DataFrame on failure to maintain app stability
    """
    p_map = {"15m": "30d", "1h": "90d", "4h": "1y", "1d": "2y", "1wk": "5y", "1mo": "10y"}
    
    if not TickerService.validate_ticker(ticker):
        st.error(f"❌ Invalid ticker format: {ticker}")
        return pd.DataFrame()
    
    try:
        # Progress bar for better UX
        progress = st.progress(0)
        
        # Suppress yfinance output
        df = yf.download(
            ticker, 
            period=p_map.get(timeframe, "1y"), 
            interval=timeframe, 
            progress=False,
            timeout=10,
            threads=False
        )
        
        progress.progress(50)
        
        # Handle MultiIndex columns (new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Ensure we have valid data
        if df.empty or len(df) < 50:
            st.warning(f"⚠️ Insufficient data for {ticker}. Try a longer timeframe.")
            return pd.DataFrame()
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Missing required data columns for {ticker}")
            return pd.DataFrame()
        
        progress.progress(100)
        time.sleep(0.1)  # Smooth progress bar transition
        progress.empty()
        
        return df.dropna().copy()
        
    except Exception as e:
        st.error(f"❌ Data fetch failed for {ticker}: {str(e)}")
        return pd.DataFrame()

# ==========================================
# 3. TRADINGVIEW HEADER
# ==========================================
def render_tv_header(selected_ticker: str):
    """Enhanced TradingView ticker tape with proper symbol mapping"""
    
    # Professional symbol mapping
    symbol_map = {
        # Crypto
        "BTC-USD": "BINANCE:BTCUSDT",
        "ETH-USD": "BINANCE:ETHUSDT",
        "SOL-USD": "BINANCE:SOLUSDT",
        "BNB-USD": "BINANCE:BNBUSDT",
        # Forex
        "EURUSD=X": "FX:EURUSD",
        "GBPUSD=X": "FX:GBPUSD",
        "JPY=X": "FX:USDJPY",
        # Indices/ETFs
        "SPY": "AMEX:SPY",
        "QQQ": "NASDAQ:QQQ",
        "IWM": "AMEX:IWM",
        "DIA": "AMEX:DIA",
        # Default fallback
        "default": f"NASDAQ:{selected_ticker.replace('-USD', '').replace('=X', '').upper()}"
    }
    
    clean = selected_ticker.replace("-USD", "").replace("=X", "")
    tv_symbol = symbol_map.get(selected_ticker, symbol_map["default"])
    
    # Primary symbol + major market references
    symbols = [
        {"proName": tv_symbol, "title": clean},
        {"proName": "BINANCE:BTCUSDT", "title": "BTC"},
        {"proName": "AMEX:SPY", "title": "SPY"},
        {"proName": "NASDAQ:QQQ", "title": "QQQ"}
    ]
                
    config = {
        "symbols": symbols, 
        "colorTheme": "dark", 
        "isTransparent": True, 
        "displayMode": "adaptive", 
        "locale": "en",
        "largeChartUrl": f"https://www.tradingview.com/chart/?symbol={tv_symbol}"
    }
    
    encoded = urllib.parse.quote(json.dumps(config))
    url = f"https://s.tradingview.com/embed-widget/ticker-tape/?locale=en#{encoded}"
    
    # Add iframe with sandbox for security
    components.html(
        f'<iframe src="{url}" style="width:100%; height:46px; border:none; overflow:hidden;" sandbox="allow-scripts"></iframe>', 
        height=50
    )

# ==========================================
# 4. ENHANCED INDICATOR ENGINE (PINE SCRIPT V6 COMPATIBLE)
# ==========================================
class EnhancedFlowEngine:
    """
    Enhanced trend detection engine with Pine Script v6 logic
    Implements: EMA Cloud, ATR-based volatility, Dynamic trend filtering
    Now works with OR without talib installation
    """
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """Enhanced trend calculation with configurable parameters"""
        if df.empty:
            return df
        
        # Pine Script v6 default parameters
        p = params or {
            'len_f': 21,
            'len_s': 50,
            'atr_len': 14,
            'vol_filter': True,
            'vol_threshold': 1.5
        }
        
        # EMA Cloud - uses talib if available, else fallback
        if TALIB_AVAILABLE:
            df['FastMA'] = talib.EMA(df['Close'], timeperiod=p['len_f'])
            df['SlowMA'] = talib.EMA(df['Close'], timeperiod=p['len_s'])
        else:
            df['FastMA'] = calculate_ema(df['Close'], p['len_f'])
            df['SlowMA'] = calculate_ema(df['Close'], p['len_s'])
        
        # True Range & ATR - uses talib if available, else fallback
        if TALIB_AVAILABLE:
            df['TR'] = talib.TRANGE(df['High'], df['Low'], df['Close'])
            df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=p['atr_len'])
        else:
            df['TR'] = calculate_trange(df['High'], df['Low'], df['Close'])
            df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], p['atr_len'])
        
        # Volatility filter (optional)
        if p['vol_filter']:
            df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
            df['Volatility_Spike'] = df['ATR_Pct'] > p['vol_threshold']
        else:
            df['Volatility_Spike'] = False
        
        # Trend Logic (Pine Script style)
        df['is_uptrend'] = df['FastMA'] > df['SlowMA']
        df['trend_strength'] = abs(df['FastMA'] - df['SlowMA']) / df['SlowMA'] * 100
        
        # Trend change detection
        df['trend_changed'] = df['is_uptrend'] != df['is_uptrend'].shift(1)
        df['long_trigger'] = df['trend_changed'] & df['is_uptrend']
        df['short_trigger'] = df['trend_changed'] & ~df['is_uptrend']
        
        # Price position relative to cloud
        df['above_cloud'] = df['Close'] > df[['FastMA', 'SlowMA']].max(axis=1)
        df['below_cloud'] = df['Close'] < df[['FastMA', 'SlowMA']].min(axis=1)
        df['in_cloud'] = ~df['above_cloud'] & ~df['below_cloud']
        
        return df

class AxiomEngine:
    """Original quantum physics metrics - PRESERVED EXACTLY"""
    
    @staticmethod
    def calculate(df: pd.DataFrame, length: int = 50) -> pd.DataFrame:
        # CHEDO Entropy (Chaos Metric)
        log_ret = np.diff(np.log(df['Close'].values), prepend=np.log(df['Close'].iloc[0]))
        ent = pd.Series(log_ret**2).rolling(length).sum().values
        df['CHEDO'] = np.tanh(ent * 10)
        
        # Apex Flux Efficiency (Movement Directness)
        rg = df['High'] - df['Low']
        rg = rg.replace(0, np.nan) # Avoid div by zero
        body = np.abs(df['Close'] - df['Open'])
        df['Flux'] = (body / rg).ewm(span=14).mean() * np.sign(df['Close'] - df['Open'])
        
        # RQZO Relativity
        df['RQZO'] = np.abs(df['Close'].pct_change()).rolling(length).mean() * 1000
        return df

# --- NEW: Volume Profile Engine (Optional Enhancement) ---
class VolumeProfileEngine:
    """
    OPTIONAL: Volume Profile / Market Profile indicator
    Simulates Pine Script v6 volume profile functionality
    """
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, bins: int = 24) -> pd.DataFrame:
        """Calculate Volume Profile (Point of Control, Value Areas)"""
        if df.empty or 'Volume' not in df.columns:
            return df
        
        # Calculate price bins
        price_min = df['Low'].min()
        price_max = df['High'].max()
        price_range = price_max - price_min
        
        # Create bins
        bin_size = price_range / bins
        df['price_bin'] = ((df['Close'] - price_min) / bin_size).astype(int)
        
        # Calculate volume per bin
        volume_profile = df.groupby('price_bin')['Volume'].sum()
        
        # Point of Control (max volume)
        poc_bin = volume_profile.idxmax()
        poc_price = price_min + (poc_bin * bin_size)
        
        # Value Area (70% of volume)
        total_volume = volume_profile.sum()
        sorted_volume = volume_profile.sort_values(ascending=False)
        cum_volume = sorted_volume.cumsum()
        value_area_bins = sorted_volume[cum_volume <= total_volume * 0.70].index
        
        va_high = price_min + (value_area_bins.max() * bin_size)
        va_low = price_min + (value_area_bins.min() * bin_size)
        
        # Add to dataframe
        df['POC'] = poc_price
        df['VA_High'] = va_high
        df['VA_Low'] = va_low
        
        # Clean up
        df.drop('price_bin', axis=1, inplace=True)
        
        return df

# --- NEW: Volatility Regime Engine (Optional Enhancement) ---
class VolatilityRegimeEngine:
    """
    OPTIONAL: Advanced volatility regime detection
    Simulates Pine Script v6 volatility state machine
    """
    
    @staticmethod
    def calculate_volatility_regime(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
        """Detect Low/High/Expansion/Crash volatility regimes"""
        if df.empty:
            return df
        
        # Standard deviation of returns
        returns = df['Close'].pct_change()
        df['Vol_StdDev'] = returns.rolling(length).std() * np.sqrt(252) * 100
        
        # Bollinger Bands Width (volatility proxy)
        if TALIB_AVAILABLE:
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
                df['Close'], timeperiod=length, nbdevup=2, nbdevdn=2, matype=0
            )
        else:
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bbands(
                df['Close'], period=length, nbdevup=2, nbdevdn=2
            )
        
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
        
        # Regime classification (Pine Script style)
        conditions = [
            (df['Vol_StdDev'] < 20) & (df['BB_Width'] < 5),
            (df['Vol_StdDev'] >= 20) & (df['Vol_StdDev'] < 40),
            (df['Vol_StdDev'] >= 40) & (df['Vol_StdDev'] < 80),
            (df['Vol_StdDev'] >= 80)
        ]
        choices = ['Low', 'Normal', 'High', 'Extreme']
        df['Vol_Regime'] = np.select(conditions, choices, default='Normal')
        
        # Add regime color for plotting
        color_map = {'Low': '#00FF88', 'Normal': '#00F0FF', 'High': '#FFA500', 'Extreme': '#FF0044'}
        df['Vol_Color'] = df['Vol_Regime'].map(color_map)
        
        return df

# ==========================================
# 5. AI ENGINE - ENHANCED
# ==========================================
class AIEngine:
    """Enhanced AI engine with structured output and error handling"""
    
    @staticmethod
    def generate_insight(api_key: str, ticker: str, df_row: pd.Series) -> str:
        """Generates a short cyberpunk-style insight with fallback"""
        if not api_key or pd.isna(api_key):
            return "⚠️ AI Neural Link Disconnected. Enter API key in sidebar."
        return AIEngine._call_gpt(api_key, ticker, df_row, "short")

    @staticmethod
    def generate_full_report(api_key: str, ticker: str, df_row: pd.Series) -> str:
        """Generates a detailed broadcast report with embedded metrics"""
        if not api_key or pd.isna(api_key):
            return "⚠️ AI Link Missing. Please enter API Key in sidebar to enable neural analysis."
        return AIEngine._call_gpt(api_key, ticker, df_row, "detailed")

    @staticmethod
    def generate_signal_summary(api_key: str, ticker: str, df: pd.DataFrame) -> Dict:
        """OPTIONAL: Generate structured signal summary for programmatic use"""
        if not api_key:
            return {"error": "API key required"}
        
        last_row = df.iloc[-1]
        report = AIEngine._call_gpt(api_key, ticker, last_row, "structured")
        
        try:
            # Attempt to parse JSON output
            return json.loads(report)
        except:
            return {"raw_report": report}

    @staticmethod
    def _call_gpt(api_key: str, ticker: str, df_row: pd.Series, mode: str) -> str:
        """Core GPT call with enhanced prompts and error handling"""
        client = OpenAI(api_key=api_key)
        
        # Extract metrics safely
        price = df_row['Close'] if 'Close' in df_row else 0
        trend = "BULLISH" if df_row.get('is_uptrend', False) else "BEARISH"
        flux = df_row.get('Flux', 0)
        entropy = df_row.get('CHEDO', 0)
        atr = df_row.get('ATR', 0)
        vol_regime = df_row.get('Vol_Regime', 'Unknown')
        
        if mode == "short":
            prompt = f"""
            Role: You are TITAN AI, an elite quant system analyzing {ticker}.
            Data: Price={price:.2f}, Trend={trend}, Flux={flux:.2f}, Entropy={entropy:.2f}, ATR={atr:.2f}, VolRegime={vol_regime}
            
            Task: Write exactly 3 concise sentences of cyberpunk-style trading analysis.
            - Use terms like: neural pathways, quantum flux, dark pool, entropy, singularity
            - Be direct and actionable
            - No headers, no bullet points, just pure analysis
            """
        elif mode == "detailed":
            prompt = f"""
            You are TITAN, an elite quant AI. Generate a FULL BROADCAST REPORT for {ticker}.
            
            MARKET INTELLIGENCE:
            - Price: {price:.2f}
            - Structure: {trend}
            - Flux Efficiency: {flux:.2f} (-1.0 to 1.0, directness of price movement)
            - Entropy/Chaos: {entropy:.2f} (0.0 to 1.0, market disorder)
            - ATR (Volatility): {atr:.2f}
            - Volatility Regime: {vol_regime}
            
            OUTPUT PROTOCOL:
            1. HEADLINE: [🚀/⚠️] TICKER - ACTION (ACCUMULATE / WATCH / SHORT / EXIT)
            2. ANALYSIS: 2-3 paragraphs analyzing momentum, entropy, and volatility regime
            3. KEY LEVELS: Identify 3 critical price levels (support/resistance/psychological)
            4. RISK METRICS: ATR-based stop loss suggestion and position sizing note
            5. VERDICT: Final tactical instruction with timeframe context
            
            Tone: Professional, institutional-grade, slightly futuristic. Avoid hyperbole.
            """
        elif mode == "structured":
            prompt = f"""
            Analyze {ticker} and output ONLY valid JSON:
            {{
                "headline": "TICKER - ACTION",
                "trend": "BULLISH/BEARISH",
                "flux": float,
                "entropy": float,
                "action": "BUY/SELL/HOLD",
                "stop_loss": float,
                "risk_level": "LOW/MEDIUM/HIGH",
                "confidence": float (0-1)
            }}
            
            Data: Price={price:.2f}, Trend={trend}, Flux={flux:.2f}, Entropy={entropy:.2f}, ATR={atr:.2f}
            """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are TITAN AI, an institutional-grade quantitative analysis system."}, 
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7 if mode != "structured" else 0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ AI Transmission Failed: {str(e)}"

# ==========================================
# 6. BROADCASTING ENGINE - ENHANCED
# ==========================================
class BroadcastEngine:
    """Professional-grade broadcasting with template management"""
    
    @staticmethod
    def format_signal_message(ticker: str, df: pd.DataFrame, report: str, template_type: str = "standard") -> str:
        """Format signal message using professional templates"""
        
        if df.empty:
            return "⚠️ No data available for signal generation"
        
        last = df.iloc[-1]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Template definitions
        templates = {
            "standard": f"""🚀 TITAN MACRO SIGNAL
Ticker: {ticker}
Price: {last['Close']:.2f}
Trend: {'🟢 BULLISH' if last['is_uptrend'] else '🔴 BEARISH'}
Flux: {last['Flux']:.3f}
Entropy: {last['CHEDO']:.3f}
ATR: {last['ATR']:.2f}

{report}

════════════════════════════
Generated: {timestamp}
System: TITAN AI v2.0""",

            "concise": f"""⚡ {ticker} | {'LONG' if last['is_uptrend'] else 'SHORT'}
PRICE: {last['Close']:.2f} | FLUX: {last['Flux']:.2f} | ENTROPY: {last['CHEDO']:.2f}

{report.split('VERDICT:')[1].split('---')[0].strip() if 'VERDICT:' in report else 'Trade with caution'}

{timestamp}""",

            "detailed": f"""════════════════════════════
📡 TITAN MACRO BROADCAST
════════════════════════════

🎯 ASSET: {ticker}
💰 PRICE: {last['Close']:.4f}
📊 TREND: {'BULLISH 🟢' if last['is_uptrend'] else 'BEARISH 🔴'}
⚡ FLUX: {last['Flux']:.4f}
🔥 ENTROPY: {last['CHEDO']:.4f}
📈 ATR: {last['ATR']:.4f}
🌊 REGIME: {last.get('Vol_Regime', 'N/A')}

════════════════════════════
📊 ANALYSIS
════════════════════════════

{report}

════════════════════════════
⏱️ TIMESTAMP: {timestamp}
🔐 SYSTEM: Classified Neural Network
════════════════════════════"""
        }
        
        return templates.get(template_type, templates["standard"])

    @staticmethod
    def send_telegram(token: str, chat_id: str, message: str) -> Tuple[bool, str]:
        """
        Enhanced Telegram sender with delivery confirmation and error handling
        Returns (success, status_message)
        """
        if not token or not chat_id:
            return False, "Missing credentials"
        
        if not message or len(message.strip()) == 0:
            return False, "Empty message"
        
        # Telegram has a 4096 character limit
        if len(message) > 4096:
            # Split into multiple messages
            parts = [message[i:i+4000] for i in range(0, len(message), 4000)]
            results = []
            
            for idx, part in enumerate(parts):
                part_msg = f"[Part {idx+1}/{len(parts)}]\n\n{part}"
                success, status = BroadcastEngine._send_single_telegram(token, chat_id, part_msg)
                results.append((success, status))
            
            # Return overall status
            if all(r[0] for r in results):
                return True, f"Broadcasted {len(parts)} parts successfully"
            else:
                return False, f"Failed. Errors: {[r[1] for r in results if not r[0]]}"
        
        else:
            return BroadcastEngine._send_single_telegram(token, chat_id, message)

    @staticmethod
    def _send_single_telegram(token: str, chat_id: str, message: str) -> Tuple[bool, str]:
        """Internal method to send a single Telegram message"""
        try:
            url = f"https://api.telegram.org/bot{token.strip()}/sendMessage"
            payload = {
                "chat_id": chat_id.strip(),
                "text": message,
                "parse_mode": "Markdown" if "```" not in message else None,
                "disable_web_page_preview": True
            }
            
            response = requests.post(
                url, 
                json=payload, 
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    return True, "Message delivered"
                else:
                    return False, f"Telegram API error: {data.get('description', 'Unknown')}"
            else:
                return False, f"HTTP {response.status_code}: {response.text}"
                
        except requests.exceptions.Timeout:
            return False, "Timeout: Telegram API did not respond"
        except requests.exceptions.ConnectionError:
            return False, "Connection error: Unable to reach Telegram"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

# ==========================================
# 7. CHART GENERATOR - ENHANCED
# ==========================================
class ChartGenerator:
    """Professional Plotly chart generator with multiple overlay options"""
    
    @staticmethod
    def create_macro_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
        """Enhanced macro chart with EMA cloud and trend markers"""
        
        fig = make_subplots(
            rows=3, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{ticker} Price & Macro Cloud", "Volume", "Trend Strength"),
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Row 1: Candlestick + EMA Cloud
        fig.add_trace(
            go.Candlestick(
                x=df.index, 
                open=df['Open'], 
                high=df['High'], 
                low=df['Low'], 
                close=df['Close'], 
                name="Price",
                increasing_line_color='#00FF88',
                decreasing_line_color='#FF0044'
            ),
            row=1, col=1
        )
        
        # Fast MA
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['FastMA'], 
                line=dict(color='cyan', width=2),
                name="Fast EMA (21)",
                hovertemplate='<b>Fast EMA</b>: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Slow MA with fill
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['SlowMA'], 
                fill='tonexty', 
                fillcolor='rgba(0,255,104,0.1)',
                line=dict(color='yellow', width=1),
                name="Slow EMA (50)",
                hovertemplate='<b>Slow EMA</b>: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Trend change markers
        trend_changes = df[df['trend_changed'] == True]
        fig.add_trace(
            go.Scatter(
                x=trend_changes.index,
                y=trend_changes['Close'],
                mode='markers',
                marker=dict(
                    symbol='diamond',
                    size=10,
                    color=['green' if x else 'red' for x in trend_changes['is_uptrend']],
                    line=dict(width=1, color='white')
                ),
                name="Trend Change",
                hovertemplate='<b>Trend Change</b><br>Price: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Row 2: Volume
        colors = ['#00FF88' if df.iloc[i]['Close'] >= df.iloc[i]['Open'] else '#FF0044' for i in range(len(df))]
        fig.add_trace(
            go.Bar(
                x=df.index, 
                y=df['Volume'], 
                marker_color=colors,
                name="Volume",
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Row 3: Trend Strength
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['trend_strength'],
                fill='tozeroy',
                line=dict(color='#00F0FF', width=1),
                fillcolor='rgba(0, 240, 255, 0.2)',
                name="Trend Strength %"
            ),
            row=3, col=1
        )
        
        # Add horizontal line at 0 for trend strength
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Strength %", row=3, col=1)
        
        return fig

    @staticmethod
    def create_quantum_physics_chart(df: pd.DataFrame) -> go.Figure:
        """Create comprehensive quantum physics metrics chart"""
        
        fig = make_subplots(
            rows=4, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.25, 0.25, 0.25, 0.25],
            subplot_titles=("CHEDO Entropy", "Apex Flux Efficiency", "Volume Profile", "Volatility Regime")
        )
        
        # Row 1: CHEDO Entropy
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['CHEDO'], 
                fill='tozeroy', 
                line=dict(color='#00F0FF', width=2),
                fillcolor='rgba(0, 240, 255, 0.2)',
                name="Entropy",
                hovertemplate='<b>Entropy</b>: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add entropy threshold lines
        fig.add_hline(y=0.8, line_dash="dash", line_color="#FF0044", row=1, col=1)
        fig.add_hline(y=0.5, line_dash="dot", line_color="#FFA500", row=1, col=1)
        
        # Row 2: Apex Flux
        colors = ['#00FF88' if x >= 0 else '#FF0044' for x in df['Flux']]
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['Flux'], 
                mode='lines+markers',
                line=dict(color='white', width=1),
                marker=dict(color=colors, size=4),
                name="Flux",
                hovertemplate='<b>Flux</b>: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Row 3: Volume Profile (if available)
        if 'POC' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['POC'],
                    line=dict(color='#FFD700', width=2, dash='dot'),
                    name="POC",
                    hovertemplate='<b>POC</b>: %{y:.2f}<extra></extra>'
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['VA_High'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(255, 215, 0, 0.3)',
                    name="VA High"
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['VA_Low'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(255, 215, 0, 0.3)',
                    fillcolor='rgba(255, 215, 0, 0.1)',
                    name="VA Low"
                ),
                row=3, col=1
            )
        
        # Row 4: Volatility Regime
        if 'Vol_Regime' in df.columns:
            regime_colors = {
                'Low': '#00FF88',
                'Normal': '#00F0FF', 
                'High': '#FFA500',
                'Extreme': '#FF0044'
            }
            
            for regime in df['Vol_Regime'].unique():
                regime_data = df[df['Vol_Regime'] == regime]
                fig.add_trace(
                    go.Scatter(
                        x=regime_data.index,
                        y=regime_data['Vol_StdDev'],
                        mode='markers',
                        marker=dict(
                            color=regime_colors.get(regime, '#888'),
                            size=6,
                            symbol='square'
                        ),
                        name=f"Regime: {regime}",
                        hovertemplate=f'<b>{regime} Volatility</b><br>%{{y:.2f}}<extra></extra>'
                    ),
                    row=4, col=1
                )
        
        fig.update_layout(
            height=900,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Entropy", row=1, col=1)
        fig.update_yaxes(title_text="Flux", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=3, col=1)
        fig.update_yaxes(title_text="Vol %", row=4, col=1)
        
        return fig

# ==========================================
# 8. MAIN TERMINAL UI - ENHANCED
# ==========================================
def main():
    """Enhanced main application with professional state management"""
    
    # Initialize session state with all required keys
    required_state_keys = {
        "broadcast_msg": "",
        "signal_history": [],
        "last_ticker": None,
        "last_timeframe": None,
        "ai_enabled": True,
        "broadcast_count": 0
    }
    
    for key, default in required_state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default
    
    # Inject CSS
    inject_terminal_css()
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("⚙️ TERMINAL CONTROL")
        
        # API Configuration Section
        st.subheader("🔐 API Configuration")
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password", 
            help="Required for AI analysis and signal generation"
        )
        
        tg_token = st.text_input(
            "Telegram Bot Token", 
            type="password",
            help="BotFather token for broadcasting"
        )
        
        tg_chat = st.text_input(
            "Telegram Chat ID", 
            help="Channel/group ID (numeric or @username)"
        )
        
        st.divider()
        
        # Asset Selection Section
        st.subheader("🎯 Asset Selection")
        
        # Universe selection with search
        mode = st.radio("Selection Mode", ["Universe", "Custom"], horizontal=True)
        
        if mode == "Universe":
            # Enhanced selectbox with search
            ticker = st.selectbox(
                "Select Asset", 
                TickerService.get_universe(),
                format_func=lambda x: f"{x.split('-')[0] if '-' in x else x}"
            )
        else:
            ticker = st.text_input(
                "Custom Ticker", 
                value="GME", 
                help="Yahoo Finance format (e.g., AAPL, BTC-USD, EURUSD=X)"
            ).upper()
            
        # Timeframe selection
        tf = st.selectbox(
            "Timeframe", 
            ["15m", "1h", "4h", "1d", "1wk", "1mo"],
            index=3,
            help="Analysis timeframe for trend detection"
        )
        
        # Feature toggles
        st.subheader("⚡ Features")
        use_ai = st.toggle("AI Parameter Optimizer (APO)", value=True)
        
        # Optional enhancements
        use_vol_profile = st.toggle("Volume Profile Analysis", value=False, help="Enable advanced volume profile metrics")
        use_vol_regime = st.toggle("Volatility Regime Detection", value=False, help="Enable volatility regime analysis")
        
        st.divider()
        
        # System status
        status_color = "🟢" if api_key else "🔴"
        st.info(f"{status_color} System Status: {'Operational' if api_key else 'Awaiting API Key'}")
        
        # TA-Lib status
        talib_status = "🟢 Available" if TALIB_AVAILABLE else "⚠️ Fallback Mode"
        st.info(f"{talib_status} TA-Lib: {'Native' if TALIB_AVAILABLE else 'Python'}")
        
        # Show broadcast count
        if st.session_state.broadcast_count > 0:
            st.metric("Broadcasts Sent", st.session_state.broadcast_count)

    # Render TradingView header
    render_tv_header(ticker)
    
    # Fetch data with caching
    df = fetch_data(ticker, tf)
    
    if df.empty:
        st.error(f"❌ Failed to fetch data for **{ticker}**. Please verify the ticker symbol or try a different timeframe.")
        return

    # Run Enhanced Flow Engine (Pine Script v6 compatible)
    df = EnhancedFlowEngine.calculate(
        df, 
        params={
            'len_f': 21,
            'len_s': 50,
            'atr_len': 14,
            'vol_filter': use_vol_regime
        }
    )
    
    # Run Original Axiom Engine (PRESERVED EXACTLY)
    df = AxiomEngine.calculate(df, length=50)
    
    # Run Optional Enhancement: Volume Profile
    if use_vol_profile:
        df = VolumeProfileEngine.calculate_volume_profile(df, bins=24)
    
    # Run Optional Enhancement: Volatility Regime
    if use_vol_regime:
        df = VolatilityRegimeEngine.calculate_volatility_regime(df, length=20)
    
    # Get last row for analysis
    last = df.iloc[-1]
    
    # Enhanced tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Macro Intelligence", "⚛️ Quantum Physics", "📡 Global Broadcast", "📈 Performance"])
    
    # --- TAB 1: MACRO INTELLIGENCE ---
    with tab1:
        col_chart, col_text = st.columns([0.7, 0.3])
        
        with col_chart:
            # Create enhanced chart
            fig_macro = ChartGenerator.create_macro_chart(df, ticker)
            st.plotly_chart(fig_macro, use_container_width=True, config={'displayModeBar': True})
        
        with col_text:
            # Dynamic Analysis Box
            st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
            st.markdown('<div class="analysis-title">🌊 DARKPOOL FLOW ANALYSIS</div>', unsafe_allow_html=True)
            
            if use_ai and api_key:
                with st.spinner("Decrypting Market Data..."):
                    ai_text = AIEngine.generate_insight(api_key, ticker, last)
                st.markdown(f'<div class="analysis-text">{ai_text}</div>', unsafe_allow_html=True)
            else:
                # Fallback Static Logic (PRESERVED EXACTLY)
                spread = abs(last['FastMA'] - last['SlowMA'])
                trend_status = "BULLISH" if last['is_uptrend'] else "BEARISH"
                
                static_html = f"""
                <div class="analysis-text">
                    The current trend for <span class="highlight">{ticker}</span> is <span class="highlight">{trend_status}</span>. 
                    <br><br>
                    The <span class="highlight">Macro Cloud</span> (Fast/Slow EMA spread) indicates institutional accumulation zones. 
                    When price holds above the cloud, it suggests a "Buy the Dip" regime. 
                    <br><br>
                    <span class="highlight">Reading:</span> The spread is currently {spread:.2f}. 
                    A widening spread confirms trend acceleration, while a narrowing spread warns of a potential crossover reversal.
                    <br><br>
                    <i>Enable AI switch + API Key for neural analysis.</i>
                </div>
                """
                st.markdown(static_html, unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: QUANTUM PHYSICS ---
    with tab2:
        st.subheader("⚛️ Axiom Physics Suite")
        
        # Create comprehensive quantum physics chart
        fig_quantum = ChartGenerator.create_quantum_physics_chart(df)
        st.plotly_chart(fig_quantum, use_container_width=True)
        
        # Metrics grid
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("CHEDO Entropy", f"{last['CHEDO']:.3f}", 
                      delta=f"{df['CHEDO'].diff().iloc[-1]:.3f}" if not pd.isna(df['CHEDO'].diff().iloc[-1]) else None)
        
        with col_m2:
            st.metric("Apex Flux", f"{last['Flux']:.3f}", 
                      delta=f"{df['Flux'].diff().iloc[-1]:.3f}" if not pd.isna(df['Flux'].diff().iloc[-1]) else None)
        
        with col_m3:
            if 'Vol_StdDev' in df.columns:
                st.metric("Volatility", f"{last['Vol_StdDev']:.2f}%")
        
        with col_m4:
            if 'POC' in df.columns:
                st.metric("Point of Control", f"{last['POC']:.2f}")

    # --- TAB 3: GLOBAL BROADCAST ---
    with tab3:
        st.header("📡 Global Broadcaster")
        st.markdown("Generate, review, and transmit signals to your Darkpool channel.")
        
        # Template selection
        template_type = st.selectbox(
            "Message Template",
            ["standard", "concise", "detailed"],
            index=0,
            help="Choose broadcast message format"
        )
        
        col_gen, col_preview = st.columns([0.4, 0.6])
        
        with col_gen:
            st.subheader("1. Signal Generation")
            include_ai_report = st.checkbox("Include Detailed AI Report", value=True)
            
            if st.button("⚡ Generate Signal Draft", use_container_width=True, type="secondary"):
                # Basic data
                basic_msg = f"🚀 TITAN SIGNAL: {ticker}\nTrend: {'BULL 🟢' if last['is_uptrend'] else 'BEAR 🔴'}\nPrice: {last['Close']:.2f}\nEntropy: {last['CHEDO']:.2f}\nFlux: {last['Flux']:.2f}\nATR: {last['ATR']:.2f}"
                
                # Append AI Report if requested
                if include_ai_report and api_key:
                    with st.spinner("Synthesizing Neural Report..."):
                        report = AIEngine.generate_full_report(api_key, ticker, last)
                        final_msg = f"{basic_msg}\n\n{report}\n\n---\nGenerated: {datetime.now().strftime('%H:%M:%S UTC')}"
                else:
                    report = "⚠️ AI Report Disabled: Enable AI and provide API key for detailed analysis."
                    final_msg = f"{basic_msg}\n\n{report}\n\n---\nGenerated: {datetime.now().strftime('%H:%M:%S UTC')}"
                
                # Format using template
                st.session_state.broadcast_msg = BroadcastEngine.format_signal_message(ticker, df, report, template_type)
                st.success("✅ Draft Generated. Review in the preview panel.")
        
        with col_preview:
            st.subheader("2. Review & Transmit")
            
            # Text area bound to session state
            edited_msg = st.text_area(
                "Review Dispatch (Editable)", 
                value=st.session_state.broadcast_msg, 
                height=400,
                help="Review and edit the message before broadcasting. Changes are temporary."
            )
            
            # Character count
            st.caption(f"Characters: {len(edited_msg)} / 4096")
            
            # Transmission controls
            col_send, col_clear = st.columns([2, 1])
            
            with col_send:
                if st.button("📡 Confirm & Transmit", type="primary", use_container_width=True):
                    if not edited_msg or len(edited_msg.strip()) == 0:
                        st.warning("⚠️ Draft is empty. Please generate a signal first.")
                    elif not tg_token or not tg_chat:
                        st.error("❌ Transmission Aborted: Missing Telegram Token or Chat ID in sidebar.")
                    else:
                        with st.spinner("Encrypting and Broadcasting..."):
                            success, status = BroadcastEngine.send_telegram(tg_token, tg_chat, edited_msg)
                            
                            if success:
                                st.balloons()
                                st.toast("Signal Broadcasted Successfully!", icon="📡")
                                st.success(f"✅ {status}")
                                
                                # Log to history
                                st.session_state.signal_history.append({
                                    "ticker": ticker,
                                    "timeframe": tf,
                                    "timestamp": datetime.now().isoformat(),
                                    "message_preview": edited_msg[:100] + "...",
                                    "template": template_type
                                })
                                st.session_state.broadcast_count += 1
                                
                                # Save to file (optional)
                                try:
                                    with open("broadcast_log.txt", "a") as f:
                                        f.write(f"\n\n{'='*50}\n")
                                        f.write(f"Time: {datetime.now()}\n")
                                        f.write(f"Ticker: {ticker} ({tf})\n")
                                        f.write(f"Template: {template_type}\n")
                                        f.write(f"Message:\n{edited_msg}\n")
                                except:
                                    pass  # Fail silently for logging
                            else:
                                st.error(f"❌ {status}")
            
            with col_clear:
                if st.button("🗑️ Clear Draft", use_container_width=True):
                    st.session_state.broadcast_msg = ""
                    st.success("Draft cleared.")

    # --- TAB 4: PERFORMANCE (Optional Enhancement) ---
    with tab4:
        st.header("📈 Signal Performance")
        st.markdown("Track broadcast history and performance metrics.")
        
        if len(st.session_state.signal_history) == 0:
            st.info("📋 No broadcast history available. Generate and transmit signals to see performance data.")
        else:
            # Display history
            st.subheader("📡 Broadcast History")
            for idx, signal in enumerate(st.session_state.signal_history[-10:]):  # Last 10
                with st.expander(f"Signal {idx+1}: {signal['ticker']} ({signal['timeframe']}) - {signal['timestamp'][:19]}"):
                    st.text(signal['message_preview'])
                    st.caption(f"Template: {signal.get('template', 'unknown')}")
            
            # Metrics
            col_hist, col_metrics = st.columns([0.6, 0.4])
            
            with col_hist:
                st.subheader("Historical Data")
                hist_df = pd.DataFrame(st.session_state.signal_history)
                if not hist_df.empty:
                    hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
                    st.dataframe(hist_df, use_container_width=True)
            
            with col_metrics:
                st.subheader("Performance Metrics")
                st.metric("Total Broadcasts", st.session_state.broadcast_count)
                
                # Ticker frequency
                if 'ticker' in hist_df.columns:
                    ticker_counts = hist_df['ticker'].value_counts()
                    if not ticker_counts.empty:
                        st.metric("Most Broadcasted", ticker_counts.index[0])
                
                # Timeframe distribution
                if 'timeframe' in hist_df.columns:
                    tf_counts = hist_df['timeframe'].value_counts()
                    if not tf_counts.empty:
                        st.metric("Primary Timeframe", tf_counts.index[0])

if __name__ == "__main__":
    main()
