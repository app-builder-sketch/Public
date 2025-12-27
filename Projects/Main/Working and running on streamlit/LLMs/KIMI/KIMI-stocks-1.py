import streamlit as st
import pandas as pd
import yfinance as yf
import openai
from datetime import datetime, date, timedelta
import io
import xlsxwriter
import requests
import numpy as np

# ------------------------------------------------------------------
# CONFIGURATION & SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="AI Stock Screener & Analyst", layout="wide")

st.title("🤖 AI Stock Screener & Analyst")
st.markdown("""
**Multi-Sector Analysis:** This agent screens stocks across multiple markets using:
1. **Fundamental Filters:** Value, Quality, Growth, and Momentum criteria
2. **Technical Analysis:** RSI, Moving Averages, Volatility scanning
3. **AI Analyst:** Generates sector-specific insights and risk assessment
""")

# ------------------------------------------------------------------
# SECRETS & SIDEBAR
# ------------------------------------------------------------------
st.sidebar.header("Configuration")

# API Key
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.sidebar.success("OpenAI Key: Loaded")
else:
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# Market Selection
analysis_mode = st.sidebar.selectbox(
    "Analysis Mode",
    ["General Market", "Junior Mining", "Custom Upload"],
    help="Select universe of stocks to analyze"
)

# Index Selection (for General Market)
selected_indices = []
if analysis_mode == "General Market":
    st.sidebar.subheader("Market Indices")
    selected_indices = st.sidebar.multiselect(
        "Select Indices",
        ["S&P 500", "NASDAQ 100", "Russell 2000", "TSX 60", "ASX 200"],
        default=["S&P 500"]
    )

# Custom Tickers
custom_tickers = []
if analysis_mode == "Custom Upload":
    st.sidebar.subheader("Custom Tickers")
    ticker_input = st.sidebar.text_area(
        "Enter tickers (one per line)",
        placeholder="AAPL\nMSFT\nGOOGL"
    )
    custom_tickers = [t.strip() for t in ticker_input.split('\n') if t.strip()]

# Telegram Integration
use_telegram = False
if "TELEGRAM_TOKEN" in st.secrets and "TELEGRAM_CHAT_ID" in st.secrets:
    tele_token = st.secrets["TELEGRAM_TOKEN"]
    tele_chat_id = st.secrets["TELEGRAM_CHAT_ID"]
    use_telegram = True
    st.sidebar.success("Telegram: Connected")
else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Telegram Setup (Optional)")
    tele_token = st.sidebar.text_input("Bot Token", type="password")
    tele_chat_id = st.sidebar.text_input("Chat ID")
    use_telegram = bool(tele_token and tele_chat_id)

# ------------------------------------------------------------------
# TICKER UNIVERSES
# ------------------------------------------------------------------
# Predefined popular tickers for major indices (simplified universe)
INDEX_UNIVERSES = {
    "S&P 500": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JNJ", "V", "JPM", 
                "UNH", "HD", "PG", "MA", "DIS", "PYPL", "NFLX", "INTC", "VZ", "ADBE"],
    "NASDAQ 100": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ADBE", "PEP",
                   "CSCO", "AVGO", "TXN", "QCOM", "CMCSA", "AMD", "INTC", "AMGN", "ISRG", "BKNG"],
    "Russell 2000": ["RBLX", "PTON", "HOOD", "PLTR", "SOFI", "UPST", "RKT", "COIN", "EXPI", "COWN",
                     "APPS", "SWBI", "VNTR", "ASO", "CROX", "CRSR", "DKS", "GPRO", "SFIX", "VSTO"],
    "TSX 60": ["SHOP.TO", "RY.TO", "TD.TO", "ENB.TO", "CP.TO", "CNQ.TO", "BAM.A.TO", "TRI.TO", "BCE.TO", "BNS.TO"],
    "ASX 200": ["CBA.AX", "BHP.AX", "CSL.AX", "NAB.AX", "ANZ.AX", "WBC.AX", "MQG.AX", "APT.AX", "TLS.AX", "WOW.AX"]
}

# Mining universe (original)
MINING_UNIVERSE = [
    "NXE", "UEC", "UUUU", "DNN", "PDN.AX", "BOE.AX", "GLO.TO",
    "LAC", "SGML", "PLS.AX", "CXO.AX", "SYA.AX", "LTR.AX", "PMET.TO", "CRE.TO",
    "KGC", "EQX", "NGD", "SILV", "MAG", "SVM", "GREG.L", "CMM.AX", "PRU.AX", "WAF.AX",
    "ERO", "IVN.TO", "HBM", "CAM.TO", "FM.TO", "ALS.TO", "SFR.AX", "29M.AX",
    "MP", "LYC.AX", "ARU.AX", "ASM.AX"
]

@st.cache_data(ttl=86400)
def get_universe_tickers(mode, indices=None, custom_list=None):
    """Returns ticker list based on selected mode"""
    if mode == "Junior Mining":
        return MINING_UNIVERSE
    elif mode == "Custom Upload":
        return custom_list or []
    else:
        # General Market - combine selected indices
        tickers = []
        for idx in (indices or []):
            tickers.extend(INDEX_UNIVERSES.get(idx, []))
        return list(set(tickers))  # Remove duplicates

# ------------------------------------------------------------------
# TECHNICAL ANALYSIS ENGINE (UNCHANGED)
# ------------------------------------------------------------------
def calculate_technicals(hist):
    if hist is None or len(hist) < 200:
        return {
            "RSI_14": None, "SMA_50": None, "SMA_200": None, 
            "Trend": "Insufficient Data", "Volatility": None
        }
    
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
    sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
    current_price = hist['Close'].iloc[-1]
    
    if current_price > sma_50 > sma_200:
        trend = "Strong Bullish"
    elif current_price < sma_50 < sma_200:
        trend = "Strong Bearish"
    elif sma_50 > sma_200:
        trend = "Bullish (Golden Cross)"
    else:
        trend = "Bearish (Death Cross)"
        
    daily_returns = hist['Close'].pct_change()
    volatility = daily_returns.std() * np.sqrt(252)

    return {
        "RSI_14": round(current_rsi, 2),
        "SMA_50": round(sma_50, 2),
        "SMA_200": round(sma_200, 2),
        "Trend": trend,
        "Volatility": f"{round(volatility*100, 1)}%"
    }

# ------------------------------------------------------------------
# DATA FETCHING (GENERALIZED)
# ------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_stock_financials(ticker_symbol, mode):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # Market cap filter (adjust for mode)
        market_cap = info.get('marketCap', 0)
        if mode == "Junior Mining":
            if market_cap < 50_000_000 or market_cap > 15_000_000_000:
                return None
        else:
            if market_cap < 200_000_000:  # Filter microcaps for general market
                return None

        # Extract sector and industry
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        # General financial data
        data = {
            "ticker": ticker_symbol,
            "name": info.get('longName', ticker_symbol),
            "country": info.get('country', 'Unknown'),
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
            "current_price": info.get('currentPrice', 0),
            
            # Valuation metrics
            "price_to_earnings": info.get('trailingPE', 999),
            "price_to_book": info.get('priceToBook', 999),
            "price_to_sales": info.get('priceToSales', 999),
            
            # Financial health
            "debt_to_equity": info.get('debtToEquity', 999),
            "current_ratio": info.get('currentRatio', 0),
            "total_cash": info.get('totalCash', 0),
            "total_debt": info.get('totalDebt', 0),
            
            # Profitability
            "roe": info.get('returnOnEquity', 0),
            "roa": info.get('returnOnAssets', 0),
            "gross_margins": info.get('grossMargins', 0),
            "operating_margins": info.get('operatingMargins', 0),
            
            # Growth
            "revenue_growth": info.get('revenueGrowth', 0),
            "earnings_growth": info.get('earningsGrowth', 0),
            
            # Trading
            "avg_volume": info.get('averageVolume', 0),
            "beta": info.get('beta', 0)
        }
        
        # Calculate net cash position
        try:
            net_cash = data["total_cash"] - data["total_debt"]
            data["net_cash_pos"] = net_cash > 0
        except:
            data["net_cash_pos"] = False

        return data
    except Exception as e:
        return None

def get_price_history(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        return stock.history(period="1y")
    except:
        return None

def get_price_at_date(hist, target_date_str):
    target_date = pd.to_datetime(target_date_str).tz_localize(None)
    if hist is None or hist.empty: return None
    hist.index = hist.index.tz_localize(None)
    hist = hist.sort_index()
    try:
        idx = hist.index.get_indexer([target_date], method='nearest')[0]
        return hist.iloc[idx]['Close']
    except:
        return hist['Close'].iloc[-1]

# ------------------------------------------------------------------
# GENERAL SCREENING LOGIC
# ------------------------------------------------------------------
def run_general_screen(universe, mode):
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    total = len(universe)
    for i, ticker in enumerate(universe):
        status_text.text(f"Screening: {ticker} ({i+1}/{total})")
        progress_bar.progress((i + 1) / total)
        
        # Fetch data
        data = get_stock_financials(ticker, mode)
        if not data: continue
            
        # Technicals
        hist = get_price_history(data['ticker']) 
        technicals = calculate_technicals(hist)
        data.update(technicals)

        # Sector-specific screening
        sector = data['sector']
        mkt_cap = data['market_cap']
        
        # Initialize criteria
        criteria = []
        
        # 1. Value Screen
        pe = data.get('price_to_earnings', 999)
        pb = data.get('price_to_book', 999)
        ps = data.get('price_to_sales', 999)
        
        if (0 < pe < 20) or (0 < pb < 2.0) or (0 < ps < 2.0):
            criteria.append("Value")
        
        # 2. Quality Screen
        roe = data.get('roe', 0)
        roa = data.get('roa', 0)
        debt_eq = data.get('debt_to_equity', 999)
        op_margin = data.get('operating_margins', 0)
        
        if (roe > 0.15) or (roa > 0.10) or (debt_eq < 50) or (op_margin > 0.15):
            criteria.append("Quality")
        
        # 3. Momentum Screen
        if (data['Trend'] == "Strong Bullish") and (data['RSI_14'] < 75):
            criteria.append("Momentum")
        
        # 4. Growth Screen
        rev_growth = data.get('revenue_growth', 0)
        earn_growth = data.get('earnings_growth', 0)
        
        if (rev_growth > 0.20) or (earn_growth > 0.15):
            criteria.append("Growth")
        
        # Mining-specific overlays
        if mode == "Junior Mining":
            # Prioritize cash and book value for miners
            if data['net_cash_pos'] and (pb < 1.5):
                criteria.append("Mining-DeepValue")
            if data['quick_ratio'] > 1.0:
                criteria.append("Mining-Solvent")

        # Save if matches any criteria
        if criteria:
            p_jan = get_price_at_date(hist, "2025-01-01")
            perf_jan = ((data['current_price'] - p_jan) / p_jan) if p_jan else 0
            
            row = data.copy()
            row.update({
                "price_jan25": p_jan,
                "perf_since_jan": perf_jan,
                "matched_criteria": criteria
            })
            results.append(row)
            
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

# ------------------------------------------------------------------
# AI ANALYST (SECTOR-ADAPTIVE)
# ------------------------------------------------------------------
def analyze_stock_with_ai(row, api_key, mode):
    client = openai.OpenAI(api_key=api_key)
    
    sector = row['sector']
    industry = row['industry']
    
    # Tailored prompts by sector
    if mode == "Junior Mining":
        asset_prompt = f"Comment on the commodity ({industry}) and Jurisdictional Risk."
        catalyst_prompt = "Comment on drill results, DFS/PFS studies, or M&A potential."
        risk_prompt = "One key risk (Dilution, Permit, Geopolitical)."
    elif sector == "Technology":
        asset_prompt = f"Comment on moat and competitive position in {industry}."
        catalyst_prompt = "Comment on product launches, market expansion, or acquisition potential."
        risk_prompt = "One key risk (Regulation, Competition, Valuation)."
    elif sector == "Healthcare":
        asset_prompt = f"Comment on pipeline strength and IP for {industry}."
        catalyst_prompt = "Comment on clinical trials, FDA approvals, or partnerships."
        risk_prompt = "One key risk (Trial Failure, Patent Cliff, Reimbursement)."
    else:
        asset_prompt = f"Comment on business quality and market position in {industry}."
        catalyst_prompt = "Comment on upcoming catalysts (earnings, expansion, M&A)."
        risk_prompt = "One key risk (Macro, Competition, Debt)."
    
    prompt = f"""
    Act as a specialized {sector} Investment Analyst. Analyze this company.
    
    [PROFILE]
    Name: {row['name']} ({row['ticker']})
    Market Cap: ${row['market_cap'] / 1e6:.0f}M
    Sector: {sector}
    
    [FINANCIAL METRICS]
    P/E: {row['price_to_earnings']}
    P/B: {row['price_to_book']}
    ROE: {row['roe']:.1%}
    Debt/Equity: {row['debt_to_equity']}
    
    [TECHNICALS]
    Trend: {row['Trend']} | RSI: {row['RSI_14']}
    
    OUTPUT REQUIREMENTS (Separated by "|"):
    1. VERDICT: Buy (Value), Buy (Growth), Buy (Speculative), Hold, or Sell.
    2. BUSINESS QUALITY: {asset_prompt}
    3. CATALYSTS: {catalyst_prompt}
    4. RISK: {risk_prompt}
    
    Example:
    Buy (Value) | Market leader with strong brand moat. | Q3 earnings beat expected. | Risk of margin compression.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are a senior {sector} analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )
        content = response.choices[0].message.content
        parts = content.split('|')
        if len(parts) < 4: return ["Hold", "AI Error", "AI Error", "AI Error"]
        return [p.strip() for p in parts[:4]]
    except Exception as e:
        return ["Error", f"API Error: {str(e)}", "", ""]

# ------------------------------------------------------------------
# TELEGRAM SENDER (UNCHANGED)
# ------------------------------------------------------------------
def send_telegram_package(token, chat_id, text, excel_buffer, filename):
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        )
        excel_buffer.seek(0)
        requests.post(
            f"https://api.telegram.org/bot{token}/sendDocument",
            data={"chat_id": chat_id, "caption": "📊 Stock Analysis Report"},
            files={"document": (filename, excel_buffer, "application/vnd.ms-excel")}
        )
        return True
    except Exception as e:
        st.error(f"Telegram Error: {e}")
        return False

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None
if "analysis_excel" not in st.session_state:
    st.session_state.analysis_excel = None

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
if st.button("🚀 Run Analysis"):
    if not api_key:
        st.error("Please provide OpenAI API Key.")
    elif not any([analysis_mode == "Junior Mining", selected_indices, custom_tickers]):
        st.error("Please select at least one market index or provide custom tickers.")
    else:
        st.subheader("1. Building Universe & Screening...")
        
        # Get tickers
        universe = get_universe_tickers(analysis_mode, selected_indices, custom_tickers)
        
        if not universe:
            st.error("No tickers in universe. Check selections.")
        else:
            st.info(f"Analyzing {len(universe)} tickers...")
            
            # Run screen
            df = run_general_screen(universe, analysis_mode)
            
            if df.empty:
                st.warning("No stocks matched the criteria.")
            else:
                # Rank by performance and select top 20
                df = df.sort_values(by='perf_since_jan', ascending=False)
                final_df = df.head(20).reset_index(drop=True)
                
                st.success(f"Identified {len(final_df)} high-potential stocks.")
                
                # AI Analysis
                st.subheader("2. AI Analyst Review...")
                prog = st.progress(0)
                for i, idx in enumerate(final_df.index):
                    row = final_df.loc[idx]
                    insights = analyze_stock_with_ai(row, api_key, analysis_mode)
                    final_df.loc[idx, 'AI_Verdict'] = insights[0]
                    final_df.loc[idx, 'Business_Quality'] = insights[1]
                    final_df.loc[idx, 'Catalysts'] = insights[2]
                    final_df.loc[idx, 'Key_Risk'] = insights[3]
                    prog.progress((i+1)/len(final_df))
                prog.empty()
                
                st.session_state.analysis_df = final_df
                st.session_state.analysis_done = True
                
                # Create Excel
                output_cols = [
                    'ticker', 'name', 'sector', 'market_cap', 'AI_Verdict',
                    'price_to_earnings', 'price_to_book', 'roe', 'revenue_growth',
                    'Business_Quality', 'Catalysts', 'Key_Risk', 'matched_criteria'
                ]
                output_df = final_df[output_cols].copy()
                output_df['matched_criteria'] = output_df['matched_criteria'].apply(lambda x: ", ".join(x))
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    output_df.to_excel(writer, index=False, sheet_name="Stock Picks")
                buffer.seek(0)
                st.session_state.analysis_excel = buffer.getvalue()

# ------------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------------
if st.session_state.analysis_done and st.session_state.analysis_df is not None:
    final_df = st.session_state.analysis_df
    
    st.write("### 🎯 Top Stock Picks")
    
    # Show summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", len(final_df))
    with col2:
        buy_count = len(final_df[final_df['AI_Verdict'].str.contains('Buy')])
        st.metric("Buy Ratings", buy_count)
    with col3:
        sectors = final_df['sector'].nunique()
        st.metric("Sectors Covered", sectors)
    with col4:
        top_performer = final_df.iloc[0]
        st.metric("Top Performer", top_performer['ticker'])
    
    # Display dataframe
    st.dataframe(final_df[[
        'ticker', 'name', 'sector', 'market_cap', 'AI_Verdict', 
        'price_to_earnings', 'revenue_growth', 'matched_criteria'
    ]])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fname = f"StockAnalysis_{date.today()}.xlsx"
        st.download_button(
            "📥 Download Analysis Report", 
            data=st.session_state.analysis_excel, 
            file_name=fname,
            mime="application/vnd.ms-excel"
        )

    with col2:
        if use_telegram:
            if st.button("📡 Broadcast Top Pick to Telegram"):
                st.info("Transmitting...")
                top_stock = final_df.iloc[0]
                
                signal_msg = f"""
📊 **STOCK ALERT** 📊

**Top Target:** {top_stock['name']} ({top_stock['ticker']})
**Verdict:** {top_stock['AI_Verdict']}

**Fundamentals:**
• Sector: {top_stock['sector']}
• Market Cap: ${top_stock['market_cap']/1e6:.1f}M
• P/E: {top_stock['price_to_earnings']}
• ROE: {top_stock['roe']:.1%}

**Catalyst:** {top_stock['Catalysts']}
"""
                send_buffer = io.BytesIO(st.session_state.analysis_excel)
                if send_telegram_package(tele_token, tele_chat_id, signal_msg, send_buffer, "Analysis_Report.xlsx"):
                    st.success("✅ Alert Sent!")
                else:
                    st.error("❌ Transmission Failed.")

st.markdown("---")
st.caption("Disclaimer: AI analysis is for informational purposes only. Always verify data and consult financial advisors.")
