# app.py - Multi-Asset TA Pro with Triple-Model Sentiment Intelligence
# Required installations:
# pip install streamlit yfinance plotly pandas numpy transformers vaderSentiment textblob wordcloud matplotlib nltk ta scipy

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Sentiment analysis imports
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib

import nltk
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

# Technical analysis imports
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator, IchimokuIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from ta.others import DailyReturnIndicator

# Set page config
st.set_page_config(
    page_title="TA Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.3rem solid #1E88E5;
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #F44336;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #9E9E9E;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">Multi-Asset TA Pro with Triple-Model Sentiment Intelligence</h1>', unsafe_allow_html=True)

# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'sentiment_cache' not in st.session_state:
    st.session_state.sentiment_cache = {}
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = {}

# Sidebar configuration
with st.sidebar:
    st.header("📊 Asset Configuration")
    
    # Asset selection
    asset_types = {
        "Cryptocurrency": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD", "DOT-USD", "DOGE-USD", "AVAX-USD"],
        "Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM"],
        "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"],
        "Commodities": ["GC=F", "CL=F", "SI=F", "NG=F", "ZC=F", "ZS=F"],
        "Indices": ["^GSPC", "^IXIC", "^DJI", "^FTSE", "^N225", "^HSI"]
    }
    
    selected_type = st.selectbox("Asset Type", list(asset_types.keys()))
    selected_assets = st.multiselect(
        "Select Assets",
        asset_types[selected_type],
        default=[asset_types[selected_type][0]]
    )
    
    # Timeframe selection
    timeframes = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "1d": "1d", "1wk": "1wk", "1mo": "1mo"
    }
    timeframe = st.selectbox("Timeframe", list(timeframes.keys()))
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Technical indicators
    st.header("📈 Technical Indicators")
    
    with st.expander("Trend Indicators", expanded=True):
        ma_period = st.slider("Moving Average Period", 5, 200, (20, 50))
        show_ema = st.checkbox("Show EMA", True)
        show_ichimoku = st.checkbox("Show Ichimoku Cloud", False)
        show_adx = st.checkbox("Show ADX", False)
    
    with st.expander("Momentum Indicators"):
        show_rsi = st.checkbox("Show RSI", True)
        show_macd = st.checkbox("Show MACD", True)
        show_stochastic = st.checkbox("Show Stochastic", False)
        show_cci = st.checkbox("Show CCI", False)
    
    with st.expander("Volatility Indicators"):
        show_bollinger = st.checkbox("Show Bollinger Bands", True)
        bb_period = st.slider("BB Period", 5, 50, 20)
        bb_std = st.slider("BB Std Dev", 1, 3, 2)
        show_atr = st.checkbox("Show ATR", False)
    
    with st.expander("Volume Indicators"):
        show_volume = st.checkbox("Show Volume", True)
        show_obv = st.checkbox("Show OBV", False)
        show_vwap = st.checkbox("Show VWAP", False)
    
    # Sentiment Analysis Configuration
    st.header("🧠 Sentiment Intelligence")
    
    # Model selection
    sentiment_mode = st.selectbox(
        "Sentiment Analysis Mode",
        ["Hybrid (All Three)", "Transformer Only", "VADER Only", "TextBlob Only", "Custom Weights"],
        index=0
    )
    
    # Custom weights expander
    if sentiment_mode == "Custom Weights":
        with st.expander("Model Weights Configuration", expanded=True):
            st.markdown("**Adjust model weights (must sum to 100%)**")
            col1, col2, col3 = st.columns(3)
            with col1:
                transformer_weight = st.slider("Transformer %", 0, 100, 40)
            with col2:
                vader_weight = st.slider("VADER %", 0, 100, 35)
            with col3:
                textblob_weight = st.slider("TextBlob %", 0, 100, 25)
            
            total_weight = transformer_weight + vader_weight + textblob_weight
            if total_weight != 100:
                st.error(f"Weights sum to {total_weight}%. Must equal 100%!")
                st.stop()
    else:
        # Default weights for hybrid mode
        transformer_weight = 40
        vader_weight = 35
        textblob_weight = 25
    
    # Sentiment data source
    sentiment_source = st.selectbox(
        "Data Source",
        ["Twitter (Simulated)", "News Headlines", "Both"],
        index=0
    )
    
    # Number of posts/articles
    num_posts = st.slider("Number of Posts/Articles", 10, 500, 100)
    
    # Refresh button
    refresh_sentiment = st.button("🔄 Refresh Sentiment Analysis", type="primary")

# Sentiment Analysis Engine Class
class TripleModelSentimentEngine:
    """Hybrid sentiment analysis engine using three complementary models"""
    
    def __init__(self):
        self.models_loaded = False
        self.transformer_model = None
        self.tokenizer = None
        self.vader_analyzer = None
        
    @st.cache_resource(show_spinner=False)
    def load_models(_self):
        """Load all three sentiment analysis models with fallbacks"""
        models_status = {
            "transformer": False,
            "vader": False,
            "textblob": True  # TextBlob doesn't require model loading
        }
        
        try:
            # Try loading the primary transformer model
            _self.transformer_model = pipeline(
                "sentiment-analysis",
                model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                tokenizer="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                device=-1  # Use CPU (-1), change to 0 for GPU if available
            )
            models_status["transformer"] = True
            st.sidebar.success("✅ Transformer model loaded")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Transformer model failed: {str(e)[:50]}... Using fallback models.")
            try:
                # Try fallback transformer
                _self.transformer_model = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device=-1
                )
                models_status["transformer"] = True
                st.sidebar.success("✅ Fallback transformer loaded")
            except:
                st.sidebar.error("❌ All transformer models failed")
        
        # Load VADER
        try:
            _self.vader_analyzer = SentimentIntensityAnalyzer()
            models_status["vader"] = True
            st.sidebar.success("✅ VADER model loaded")
        except Exception as e:
            st.sidebar.error(f"❌ VADER failed: {str(e)[:50]}")
        
        _self.models_loaded = any(models_status.values())
        return models_status
    
    def analyze_transformer(self, text):
        """Analyze sentiment using transformer model"""
        if self.transformer_model is None:
            return {"label": "NEUTRAL", "score": 0.0, "normalized_score": 0.0}
        
        try:
            result = self.transformer_model(text[:512])[0]  # Limit text length
            
            # Normalize to -1 to 1 scale
            label = result['label'].lower()
            score = result['score']
            
            if 'positive' in label:
                normalized = score
            elif 'negative' in label:
                normalized = -score
            else:
                normalized = 0.0
            
            return {
                "label": result['label'],
                "score": score,
                "normalized_score": normalized
            }
        except:
            return {"label": "NEUTRAL", "score": 0.0, "normalized_score": 0.0}
    
    def analyze_vader(self, text):
        """Analyze sentiment using VADER"""
        if self.vader_analyzer is None:
            return {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0}
        
        scores = self.vader_analyzer.polarity_scores(text)
        return scores
    
    def analyze_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            return {
                "polarity": polarity,
                "subjectivity": subjectivity
            }
        except:
            return {"polarity": 0.0, "subjectivity": 0.5}
    
    def hybrid_analysis(self, text, weights=(0.4, 0.35, 0.25)):
        """Combine all three models with weighted average"""
        transformer_w, vader_w, textblob_w = weights
        
        # Get individual model results
        transformer_result = self.analyze_transformer(text)
        vader_result = self.analyze_vader(text)
        textblob_result = self.analyze_textblob(text)
        
        # Normalize all scores to -1 to 1 range
        transformer_score = transformer_result.get("normalized_score", 0.0)
        vader_score = vader_result.get("compound", 0.0)  # Already -1 to 1
        textblob_score = textblob_result.get("polarity", 0.0)  # Already -1 to 1
        
        # Calculate weighted average
        total_weight = transformer_w + vader_w + textblob_w
        if total_weight > 0:
            hybrid_score = (
                transformer_score * transformer_w +
                vader_score * vader_w +
                textblob_score * textblob_w
            ) / total_weight
        else:
            hybrid_score = 0.0
        
        # Determine sentiment category
        if hybrid_score > 0.5:
            sentiment_category = "STRONG POSITIVE"
        elif hybrid_score > 0.05:
            sentiment_category = "POSITIVE"
        elif hybrid_score < -0.5:
            sentiment_category = "STRONG NEGATIVE"
        elif hybrid_score < -0.05:
            sentiment_category = "NEGATIVE"
        else:
            sentiment_category = "NEUTRAL"
        
        # Model agreement
        model_signs = [
            1 if transformer_score > 0.05 else -1 if transformer_score < -0.05 else 0,
            1 if vader_score > 0.05 else -1 if vader_score < -0.05 else 0,
            1 if textblob_score > 0.05 else -1 if textblob_score < -0.05 else 0
        ]
        
        consensus = "HIGH" if len(set(model_signs)) <= 2 else "LOW"
        
        return {
            "hybrid_score": hybrid_score,
            "sentiment_category": sentiment_category,
            "consensus": consensus,
            "transformer": transformer_result,
            "vader": vader_result,
            "textblob": textblob_result,
            "subjectivity": textblob_result.get("subjectivity", 0.5),
            "model_scores": {
                "transformer": transformer_score,
                "vader": vader_score,
                "textblob": textblob_score
            }
        }

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(tickers, start_date, end_date, timeframe="1d"):
    """Fetch price data for multiple tickers"""
    data_dict = {}
    
    for ticker in tickers:
        cache_key = f"{ticker}_{start_date}_{end_date}_{timeframe}"
        
        if cache_key in st.session_state.data_cache:
            data_dict[ticker] = st.session_state.data_cache[cache_key]
        else:
            try:
                # Convert timeframe to yfinance format
                yf_timeframe = {
                    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                    "1h": "60m", "1d": "1d", "1wk": "1wk", "1mo": "1mo"
                }.get(timeframe, "1d")
                
                # Fetch data
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=yf_timeframe,
                    progress=False
                )
                
                if not data.empty:
                    # Clean data
                    data = data.dropna()
                    
                    # Calculate additional metrics
                    data['Returns'] = data['Close'].pct_change()
                    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
                    data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
                    
                    st.session_state.data_cache[cache_key] = data
                    data_dict[ticker] = data
                    
            except Exception as e:
                st.error(f"Error fetching data for {ticker}: {str(e)}")
                continue
    
    return data_dict

def calculate_technical_indicators(data, ma_periods=(20, 50), bb_period=20, bb_std=2):
    """Calculate technical indicators"""
    df = data.copy()
    
    # Moving Averages
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=ma_periods[0]).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=ma_periods[1]).sma_indicator()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=ma_periods[0]).ema_indicator()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=bb_period, window_dev=bb_std)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_width'] = bb.bollinger_wband()
    df['BB_pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # RSI
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # Stochastic
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['Stoch_k'] = stoch.stoch()
    df['Stoch_d'] = stoch.stoch_signal()
    
    # ATR
    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
    
    # OBV
    df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    
    # VWAP (simplified - for daily data)
    if 'vwap' not in df.columns:
        vwap = VolumeWeightedAveragePrice(
            high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']
        )
        df['VWAP'] = vwap.volume_weighted_average_price()
        
    # ADX / DMI
    try:
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ADX'] = adx.adx()
        df['DI_plus'] = adx.adx_pos()
        df['DI_minus'] = adx.adx_neg()
    except Exception:
        df['ADX'] = np.nan
        df['DI_plus'] = np.nan
        df['DI_minus'] = np.nan

    # CCI
    try:
        df['CCI'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci()
    except Exception:
        df['CCI'] = np.nan

    # Williams %R
    try:
        df['Williams_R'] = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14).williams_r()
    except Exception:
        df['Williams_R'] = np.nan

    # Ichimoku Cloud (basic components)
    try:
        ichi = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
        df['Ichi_conv'] = ichi.ichimoku_conversion_line()
        df['Ichi_base'] = ichi.ichimoku_base_line()
        df['Ichi_a'] = ichi.ichimoku_a()
        df['Ichi_b'] = ichi.ichimoku_b()
    except Exception:
        df['Ichi_conv'] = np.nan
        df['Ichi_base'] = np.nan
        df['Ichi_a'] = np.nan
        df['Ichi_b'] = np.nan
    
    # Generate signals
    df['MA_Signal'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
    df['RSI_Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    df['MACD_Signal'] = np.where(df['MACD'] > df['MACD_signal'], 1, -1)
    df['BB_Signal'] = np.where(df['Close'] < df['BB_lower'], 1, 
                               np.where(df['Close'] > df['BB_upper'], -1, 0))
    
    # Combined signal
    signals = ['MA_Signal', 'RSI_Signal', 'MACD_Signal', 'BB_Signal']
    df['Combined_Signal'] = df[signals].mean(axis=1)
    
    return df

@st.cache_data(ttl=900, show_spinner=True)
def fetch_sentiment_data(ticker, num_posts=100, source="twitter"):
    """Fetch sentiment data for a given ticker"""
    # Generate simulated sentiment data (replace with real API calls)
    posts = []
    
    # Simulate Twitter posts
    if "twitter" in source.lower() or "simulated" in source.lower():
        ticker_symbol = ticker.split('-')[0] if '-' in ticker else ticker
        
        # Simulated posts with varying sentiment
        positive_keywords = ["bullish", "moon", "🚀", "📈", "buy", "strong", "growth", "opportunity"]
        negative_keywords = ["bearish", "crash", "dump", "sell", "warning", "risk", "fud"]
        
        for i in range(num_posts):
            # Vary sentiment over time
            base_sentiment = np.sin(i / 10) * 0.5 + np.random.normal(0, 0.2)
            
            if base_sentiment > 0.2:
                sentiment = "positive"
                keyword = np.random.choice(positive_keywords)
            elif base_sentiment < -0.2:
                sentiment = "negative"
                keyword = np.random.choice(negative_keywords)
            else:
                sentiment = "neutral"
                keyword = ticker_symbol
            
            post = f"{keyword} {ticker_symbol} is looking {'good' if sentiment == 'positive' else 'bad' if sentiment == 'negative' else 'interesting'} today. " + \
                   f"#{ticker_symbol} trading at {'high' if sentiment == 'positive' else 'low'} levels. " + \
                   f"Technical analysis suggests {'bullish' if sentiment == 'positive' else 'bearish'} momentum."
            
            posts.append({
                "text": post,
                "timestamp": datetime.now() - timedelta(minutes=i*10),
                "likes": np.random.randint(0, 1000),
                "retweets": np.random.randint(0, 500),
                "true_sentiment": sentiment
            })
    
    # Simulate news headlines
    if "news" in source.lower() or "both" in source.lower():
        news_headlines = [
            f"{ticker} announces strong earnings, beats estimates",
            f"Analysts upgrade {ticker} to buy rating",
            f"Market volatility impacts {ticker} performance",
            f"{ticker} faces regulatory challenges",
            f"Institutional investors increasing {ticker} holdings",
            f"Technical breakout detected for {ticker}",
            f"{ticker} forms bullish chart pattern",
            f"Market correction affects {ticker} price"
        ]
        
        for i in range(min(20, num_posts // 2)):
            posts.append({
                "text": np.random.choice(news_headlines),
                "timestamp": datetime.now() - timedelta(hours=i*3),
                "source": "News",
                "true_sentiment": "neutral"
            })
    
    return posts

def create_sentiment_visualizations(sentiment_results, ticker):
    """Create comprehensive sentiment visualizations"""
    # Extract data
    scores = [r['hybrid_score'] for r in sentiment_results]
    categories = [r['sentiment_category'] for r in sentiment_results]
    subjectivities = [r['subjectivity'] for r in sentiment_results]
    timestamps = [r.get('timestamp', datetime.now() - timedelta(minutes=i)) 
                  for i, r in enumerate(sentiment_results)]
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Sentiment Overview", 
        "📈 Score Distribution", 
        "🧠 Model Comparison",
        "☁️ Word Cloud",
        "📰 Raw Posts"
    ])
    
    with tab1:
        # Time series plot
        fig_sentiment_ts = go.Figure()
        
        fig_sentiment_ts.add_trace(go.Scatter(
            x=timestamps[-100:],
            y=scores[-100:],
            mode='lines+markers',
            name='Hybrid Score',
            line=dict(color='blue', width=2)
        ))
        
        # Add EMA of sentiment
        if len(scores) > 6:
            sentiment_series = pd.Series(scores, index=timestamps)
            ema_sentiment = sentiment_series.ewm(span=6).mean()
            fig_sentiment_ts.add_trace(go.Scatter(
                x=ema_sentiment.index[-100:],
                y=ema_sentiment.values[-100:],
                mode='lines',
                name='EMA-6 Sentiment',
                line=dict(color='orange', width=3, dash='dash')
            ))
        
        # Add sentiment zones
        fig_sentiment_ts.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.5)
        fig_sentiment_ts.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.5)
        fig_sentiment_ts.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)
        
        fig_sentiment_ts.update_layout(
            title=f"{ticker} Sentiment Timeline",
            xaxis_title="Time",
            yaxis_title="Sentiment Score (-1 to 1)",
            height=450,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_sentiment_ts, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        avg_sentiment = np.mean(scores)
        std_sentiment = np.std(scores)
        sentiment_trend = np.polyfit(range(len(scores)), scores, 1)[0] if len(scores) > 1 else 0
        
        col1.metric("Average Sentiment", f"{avg_sentiment:.3f}")
        col2.metric("Volatility", f"{std_sentiment:.3f}")
        col3.metric("Trend", f"{sentiment_trend:.4f}")
        col4.metric("Latest Score", f"{scores[-1]:.3f}")
        
        # Model agreement statistics
        agreements = [r['consensus'] for r in sentiment_results]
        agreement_counts = pd.Series(agreements).value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Model Consensus")
            for consensus, count in agreement_counts.items():
                percentage = (count / len(agreements)) * 100
                st.progress(int(percentage), text=f"{consensus}: {percentage:.1f}%")
        
        with col2:
            st.markdown("##### Latest Model Scores")
            if sentiment_results:
                latest_scores = sentiment_results[-1]['model_scores']
                for model, score in latest_scores.items():
                    color = "green" if score > 0 else "red" if score < 0 else "gray"
                    st.markdown(
                        f"**{model.title()}**: <span style='color:{color}'>{score:.3f}</span>",
                        unsafe_allow_html=True
                    )
    
    with tab2:
        # Distribution histogram
        fig_hist = px.histogram(
            x=scores,
            nbins=30,
            title="Sentiment Score Distribution",
            labels={"x": "Sentiment Score", "y": "Count"}
        )
        fig_hist.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Box plot by category
        df_dist = pd.DataFrame({
            "Score": scores,
            "Category": categories
        })
        
        fig_box = px.box(
            df_dist,
            x="Category",
            y="Score",
            title="Sentiment by Category"
        )
        fig_box.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)
    
    with tab3:
        # Radar chart comparing model scores for latest analysis
        if sentiment_results:
            latest = sentiment_results[-1]
            model_scores = latest['model_scores']
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=[abs(model_scores['transformer']), 
                   abs(model_scores['vader']), 
                   abs(model_scores['textblob'])],
                theta=['Transformer', 'VADER', 'TextBlob'],
                fill='toself',
                name='Model Strength'
            ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Latest Model Score Magnitudes",
                height=400,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Model correlation scatter
            df_models = pd.DataFrame([r['model_scores'] for r in sentiment_results])
            
            fig_scatter = px.scatter_matrix(
                df_models,
                dimensions=['transformer', 'vader', 'textblob'],
                title="Model Score Correlations"
            )
            fig_scatter.update_layout(height=500, template="plotly_dark")
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab4:
        # Generate word cloud from posts
        if sentiment_results and 'posts' in st.session_state:
            all_text = " ".join([post['text'] for post in st.session_state.posts[:50]])
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='black',
                colormap='viridis',
                max_words=100,
                stopwords=set(stopwords.words('english'))
            ).generate(all_text)
            
            # Display word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    with tab5:
        # Display raw posts with sentiment scores
        if 'posts' in st.session_state:
            posts_data = []
            for i, post in enumerate(st.session_state.posts[:50]):
                if i < len(sentiment_results):
                    result = sentiment_results[i]
                    posts_data.append({
                        'Post': post['text'][:100] + "...",
                        'Sentiment': result['sentiment_category'],
                        'Score': result['hybrid_score'],
                        'Consensus': result['consensus'],
                        'Likes': post.get('likes', 0),
                        'Retweets': post.get('retweets', 0)
                    })
            
            df_posts = pd.DataFrame(posts_data)
            st.dataframe(df_posts, use_container_width=True)
            
            # Show detailed view for selected post
            if not df_posts.empty:
                selected_post = st.selectbox("Select post for details", range(len(df_posts)))
                if selected_post is not None:
                    st.markdown("##### Detailed Post Analysis")
                    row = df_posts.iloc[selected_post]
                    st.markdown(f"**Sentiment:** {row['Sentiment']} (Score: {row['Score']:.3f})")
                    st.markdown(f"**Model Consensus:** {row['Consensus']}")
                    st.markdown(f"**Engagement:** {row['Likes']} likes, {row['Retweets']} retweets")
                    st.markdown("**Full Text:**")
                    st.markdown(f"> {st.session_state.posts[selected_post]['text']}")
                    st.divider()

def create_price_chart_with_sentiment(data, sentiment_data=None, ticker="", indicators=None):
    """Create interactive price chart with sentiment overlay"""
    if data.empty:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{ticker} Price & Indicators", "Volume", "Sentiment")
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Add moving averages if available
    if 'SMA_20' in data.columns and indicators.get('show_ema', True):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
    
    if 'SMA_50' in data.columns and indicators.get('show_ema', True):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    
    # Add Bollinger Bands if selected
    if indicators is None:
        indicators = {}

    # Add EMA if selected
    if 'EMA_20' in data.columns and indicators.get('show_ema', True):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['EMA_20'],
                mode='lines',
                name='EMA 20',
                line=dict(color='purple', width=2, dash='dot')
            ),
            row=1, col=1
        )

    # Add Bollinger Bands if selected
    if 'BB_upper' in data.columns and indicators.get('show_bollinger', True):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(200,200,200,0.7)', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(200,200,200,0.7)', width=1),
                fill='tonexty',
                fillcolor='rgba(200,200,200,0.08)',
                showlegend=False
            ),
            row=1, col=1
        )

    # Ichimoku Cloud (optional)
    if indicators.get('show_ichimoku', False) and {'Ichi_a', 'Ichi_b'}.issubset(set(data.columns)):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Ichi_a'],
                mode='lines',
                name='Ichimoku A',
                line=dict(width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Ichi_b'],
                mode='lines',
                name='Ichimoku B',
                line=dict(width=1),
                fill='tonexty',
                fillcolor='rgba(0, 200, 0, 0.05)',
                showlegend=False
            ),
            row=1, col=1
        )

    # VWAP (optional)
    if indicators.get('show_vwap', False) and 'VWAP' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['VWAP'],
                mode='lines',
                name='VWAP',
                line=dict(width=2, dash='dash')
            ),
            row=1, col=1
        )

    # Volume bars
    if indicators.get('show_volume', True) and 'Volume' in data.columns:
        vol_colors = np.where(data['Close'] >= data['Open'], 'rgba(0,200,0,0.5)', 'rgba(200,0,0,0.5)')
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=vol_colors,
                showlegend=False
            ),
            row=2, col=1
        )

    # Sentiment subplot
    if sentiment_data:
        try:
            s_df = pd.DataFrame(sentiment_data)
            if 'timestamp' in s_df.columns:
                s_df = s_df.sort_values('timestamp')
                fig.add_trace(
                    go.Scatter(
                        x=s_df['timestamp'],
                        y=s_df['hybrid_score'],
                        mode='lines',
                        name='Sentiment Score',
                        line=dict(width=2)
                    ),
                    row=3, col=1
                )
                fig.add_hline(y=0, line_width=1, line_dash="dot", row=3, col=1)
        except Exception:
            pass

    fig.update_layout(
        height=900,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10)
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment", row=3, col=1)

    return fig


# ==========================================================
# MAIN DASHBOARD LOGIC (Added to make script fully runnable)
# ==========================================================

st.markdown("---")

# Build indicator config dict from sidebar checkboxes
indicator_cfg = {
    "show_ema": show_ema,
    "show_bollinger": show_bollinger,
    "show_ichimoku": show_ichimoku,
    "show_adx": show_adx,
    "show_rsi": show_rsi,
    "show_macd": show_macd,
    "show_stochastic": show_stochastic,
    "show_cci": show_cci,
    "show_atr": show_atr,
    "show_volume": show_volume,
    "show_obv": show_obv,
    "show_vwap": show_vwap,
}

# Resolve hybrid weights based on mode
if sentiment_mode == "Transformer Only":
    weights = (1.0, 0.0, 0.0)
elif sentiment_mode == "VADER Only":
    weights = (0.0, 1.0, 0.0)
elif sentiment_mode == "TextBlob Only":
    weights = (0.0, 0.0, 1.0)
elif sentiment_mode == "Custom Weights":
    weights = (transformer_weight/100.0, vader_weight/100.0, textblob_weight/100.0)
else:
    weights = (0.4, 0.35, 0.25)

# Load data
if not selected_assets:
    st.warning("Please select at least one asset from the sidebar.")
    st.stop()

st.subheader("📌 Multi-Asset Dashboard")

with st.spinner("Fetching market data..."):
    price_data = fetch_price_data(
        selected_assets,
        start_date=str(start_date),
        end_date=str(end_date),
        timeframe=timeframe
    )

if not price_data:
    st.error("No price data returned. Please adjust tickers, timeframe, or date range.")
    st.stop()

# Sentiment engine
engine = TripleModelSentimentEngine()
models_status = engine.load_models()

# Prepare sentiment for primary asset only (first selected), to keep UI responsive
primary_ticker = selected_assets[0]
sent_cache_key = f"{primary_ticker}_{sentiment_source}_{num_posts}_{sentiment_mode}"

do_refresh = refresh_sentiment or (sent_cache_key not in st.session_state.sentiment_cache)
if do_refresh:
    with st.spinner("Running sentiment analysis..."):
        posts = fetch_sentiment_data(primary_ticker, num_posts=num_posts, source=sentiment_source)
        st.session_state.posts = posts  # used by WordCloud tab

        sentiment_results = []
        for p in posts:
            res = engine.hybrid_analysis(p.get("text", ""), weights=weights)
            res.update({
                "timestamp": p.get("timestamp", datetime.now()),
                "text": p.get("text", ""),
                "likes": p.get("likes", 0),
                "retweets": p.get("retweets", 0),
                "source": p.get("source", "Simulated"),
            })
            sentiment_results.append(res)

        st.session_state.sentiment_cache[sent_cache_key] = sentiment_results
else:
    sentiment_results = st.session_state.sentiment_cache.get(sent_cache_key, [])

# Top KPIs
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

latest_df = price_data[primary_ticker]
if latest_df is not None and not latest_df.empty:
    last_close = float(latest_df['Close'].iloc[-1])
    last_ret = float(latest_df['Returns'].iloc[-1]) if 'Returns' in latest_df.columns and not pd.isna(latest_df['Returns'].iloc[-1]) else 0.0
    vol = float(latest_df['Volatility'].iloc[-1]) if 'Volatility' in latest_df.columns and not pd.isna(latest_df['Volatility'].iloc[-1]) else 0.0
else:
    last_close, last_ret, vol = 0.0, 0.0, 0.0

if sentiment_results:
    sent_last = float(sentiment_results[-1].get("hybrid_score", 0.0))
    sent_mean = float(np.mean([r.get("hybrid_score", 0.0) for r in sentiment_results]))
else:
    sent_last, sent_mean = 0.0, 0.0

kpi_col1.metric(f"{primary_ticker} Close", f"{last_close:,.4f}")
kpi_col2.metric("Daily Return", f"{last_ret*100:.2f}%")
kpi_col3.metric("20D Volatility", f"{vol*100:.2f}%")
kpi_col4.metric("Latest Sentiment", f"{sent_last:+.3f}", delta=f"{(sent_last - sent_mean):+.3f} vs mean")

# Main tabs
main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
    "📈 Price & Signals",
    "🧠 Sentiment Intelligence",
    "📊 Multi-Asset Compare",
    "🧾 Data"
])

with main_tab1:
    st.markdown("### Price Charts")
    asset_tabs = st.tabs(selected_assets)

    for idx, ticker in enumerate(selected_assets):
        with asset_tabs[idx]:
            df = price_data.get(ticker)
            if df is None or df.empty:
                st.warning(f"No data for {ticker}")
                continue

            # Calculate indicators
            try:
                df_ta = calculate_technical_indicators(
                    df,
                    ma_periods=ma_period,
                    bb_period=bb_period,
                    bb_std=bb_std
                )
            except Exception as e:
                st.error(f"Indicator calc failed for {ticker}: {e}")
                df_ta = df

            # Overlay sentiment only for primary ticker
            s_data = sentiment_results if ticker == primary_ticker else None

            fig = create_price_chart_with_sentiment(df_ta, sentiment_data=s_data, ticker=ticker, indicators=indicator_cfg)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            # Quick signal panel
            if 'Combined_Signal' in df_ta.columns:
                cs = float(df_ta['Combined_Signal'].iloc[-1])
                bias = "Bullish" if cs > 0.25 else "Bearish" if cs < -0.25 else "Neutral"
                st.info(f"**Combined Signal:** {cs:+.2f} → **{bias}**")

            with st.expander("More Indicators & Diagnostics", expanded=False):
                cols = st.columns(3)

                # RSI
                if indicator_cfg.get("show_rsi", True) and 'RSI' in df_ta.columns:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df_ta.index, y=df_ta['RSI'], mode='lines', name='RSI'))
                    fig_rsi.add_hline(y=70, line_dash="dash")
                    fig_rsi.add_hline(y=30, line_dash="dash")
                    fig_rsi.update_layout(height=250, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
                    cols[0].plotly_chart(fig_rsi, use_container_width=True)

                # MACD
                if indicator_cfg.get("show_macd", True) and {'MACD','MACD_signal','MACD_diff'}.issubset(df_ta.columns):
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD'], mode='lines', name='MACD'))
                    fig_macd.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD_signal'], mode='lines', name='Signal'))
                    fig_macd.add_trace(go.Bar(x=df_ta.index, y=df_ta['MACD_diff'], name='Hist', opacity=0.5))
                    fig_macd.update_layout(height=250, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
                    cols[1].plotly_chart(fig_macd, use_container_width=True)

                # ADX
                if indicator_cfg.get("show_adx", False) and {'ADX','DI_plus','DI_minus'}.issubset(df_ta.columns):
                    fig_adx = go.Figure()
                    fig_adx.add_trace(go.Scatter(x=df_ta.index, y=df_ta['ADX'], mode='lines', name='ADX'))
                    fig_adx.add_trace(go.Scatter(x=df_ta.index, y=df_ta['DI_plus'], mode='lines', name='+DI'))
                    fig_adx.add_trace(go.Scatter(x=df_ta.index, y=df_ta['DI_minus'], mode='lines', name='-DI'))
                    fig_adx.update_layout(height=250, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
                    cols[2].plotly_chart(fig_adx, use_container_width=True)

with main_tab2:
    st.markdown("### Sentiment Intelligence")
    if sentiment_results:
        # Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sent_last,
            delta={"reference": sent_mean},
            gauge={"axis": {"range": [-1, 1]}},
            title={"text": f"{primary_ticker} Hybrid Sentiment"}
        ))
        gauge.update_layout(height=300, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(gauge, use_container_width=True)

        create_sentiment_visualizations(sentiment_results, primary_ticker)
    else:
        st.info("No sentiment results yet. Click **Refresh Sentiment Analysis** in the sidebar.")

with main_tab3:
    st.markdown("### Multi-Asset Comparison (Normalized)")
    compare_df = pd.DataFrame()
    for ticker in selected_assets:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue
        series = df['Close'].copy()
        series = (series / series.iloc[0]) * 100
        compare_df[ticker] = series

    if not compare_df.empty:
        fig_cmp = go.Figure()
        for ticker in compare_df.columns:
            fig_cmp.add_trace(go.Scatter(x=compare_df.index, y=compare_df[ticker], mode='lines', name=ticker))
        fig_cmp.update_layout(height=500, template="plotly_dark", yaxis_title="Normalized (Base=100)", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.warning("Not enough data to compare.")

with main_tab4:
    st.markdown("### Raw Data")
    for ticker in selected_assets:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue
        st.markdown(f"#### {ticker}")
        st.dataframe(df.tail(200), use_container_width=True)
