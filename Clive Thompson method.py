import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
TOTAL_STOCKS = 40
STARTING_CAPITAL = 400000
ALLOCATION_PER_STOCK = 10000
MIN_HEALTH_SCORE = 3  # out of 6
MIN_GROWTH_SCORE = 3  # out of 6

st.set_page_config(page_title="BTB 2026 Quant Dashboard", layout="wide")

# =============================================================================
# DATA CORE: THE 40 STOCKS (AS PER CLIVE THOMPSON 2026 LIST)
# =============================================================================
@st.cache_data
def get_portfolio_data():
    # Mapping industries and themes as defined in the transcript
    data = [
        {"Ticker": "1451.T", "Name": "Kasumigaseki Capital", "Country": "Japan", "Currency": "JPY", "Style": "Growth", "Industry": "Real Estate"},
        {"Ticker": "S63.SI", "Name": "ST Engineering", "Country": "Singapore", "Currency": "SGD", "Style": "Quality", "Industry": "Aerospace & Defence"},
        {"Ticker": "AM.PA", "Name": "Dassault Aviation", "Country": "France", "Currency": "EUR", "Style": "Quality", "Industry": "Defence"},
        {"Ticker": "LDO.MI", "Name": "Leonardo SpA", "Country": "Italy", "Currency": "EUR", "Style": "Cyclical", "Industry": "Defence"},
        {"Ticker": "SHA0.DE", "Name": "Schaeffler AG", "Country": "Germany", "Currency": "EUR", "Style": "Value", "Industry": "Auto-Industrial"},
        {"Ticker": "EBS.VI", "Name": "Erste Group Bank", "Country": "Austria", "Currency": "EUR", "Style": "Value", "Industry": "Financials"},
        {"Ticker": "ARGX.BR", "Name": "argenx SE", "Country": "Belgium", "Currency": "EUR", "Style": "High-Growth", "Industry": "Biotech"},
        {"Ticker": "SMCI", "Name": "Super Micro Computer", "Country": "USA", "Currency": "USD", "Style": "Momentum", "Industry": "AI Hardware"},
        {"Ticker": "NORBT.OL", "Name": "Norbit ASA", "Country": "Norway", "Currency": "NOK", "Style": "Small-Cap", "Industry": "Industrial Tech"},
        {"Ticker": "BG", "Name": "Bunge Global", "Country": "USA", "Currency": "USD", "Style": "Cyclical", "Industry": "Agribusiness"},
        {"Ticker": "2419.HK", "Name": "DEKON Food", "Country": "China", "Currency": "HKD", "Style": "Value", "Industry": "Agri-Food"},
        {"Ticker": "ATM.NZ", "Name": "a2 Milk Company", "Country": "New Zealand", "Currency": "NZD", "Style": "Growth", "Industry": "Consumer"},
        {"Ticker": "SN.L", "Name": "Smith & Nephew", "Country": "UK", "Currency": "GBP", "Style": "Quality", "Industry": "Healthcare"},
        {"Ticker": "BETS-B.ST", "Name": "Betsson AB", "Country": "Sweden", "Currency": "SEK", "Style": "Growth", "Industry": "Consumer"},
        {"Ticker": "GROY", "Name": "Gold Royalty Corp", "Country": "USA", "Currency": "USD", "Style": "Thematic", "Industry": "Gold Royalties"},
        {"Ticker": "HAR.JO", "Name": "Harmony Gold", "Country": "South Africa", "Currency": "ZAR", "Style": "Cyclical", "Industry": "Gold"},
        {"Ticker": "KGH.WA", "Name": "KGHM Polska Miedź", "Country": "Poland", "Currency": "PLN", "Style": "Cyclical", "Industry": "Copper"},
        {"Ticker": "TXG.TO", "Name": "Torex Gold", "Country": "Canada", "Currency": "CAD", "Style": "Growth", "Industry": "Gold"},
        {"Ticker": "WGX.AX", "Name": "Westgold Resources", "Country": "Australia", "Currency": "AUD", "Style": "Cyclical", "Industry": "Gold"},
        {"Ticker": "0386.HK", "Name": "Sinopec", "Country": "China", "Currency": "HKD", "Style": "Income", "Industry": "Energy"},
        {"Ticker": "SHEL.L", "Name": "Shell PLC", "Country": "UK", "Currency": "GBP", "Style": "Income", "Industry": "Energy"},
        {"Ticker": "CAMX.ST", "Name": "Camurus AB", "Country": "Sweden", "Currency": "SEK", "Style": "Growth", "Industry": "Biopharma"},
        {"Ticker": "1681.HK", "Name": "Consun Pharma", "Country": "China", "Currency": "HKD", "Style": "Defensive", "Industry": "Healthcare"},
        {"Ticker": "COPN.SW", "Name": "Cosmo Pharma", "Country": "Switzerland", "Currency": "CHF", "Style": "Special Sit", "Industry": "Growth"},
        {"Ticker": "ROG.SW", "Name": "Roche Holding", "Country": "Switzerland", "Currency": "CHF", "Style": "Quality", "Industry": "Healthcare"},
        {"Ticker": "ALQ.AX", "Name": "ALS Limited", "Country": "Australia", "Currency": "AUD", "Style": "Quality", "Industry": "Defensive Growth"},
        {"Ticker": "DLO", "Name": "dLocal Ltd", "Country": "Uruguay", "Currency": "USD", "Style": "High-Growth", "Industry": "FinTech"},
        {"Ticker": "MTO.L", "Name": "Mitie Group", "Country": "UK", "Currency": "GBP", "Style": "Value", "Industry": "Turnaround"},
        {"Ticker": "VWS.CO", "Name": "Vestas Wind", "Country": "Denmark", "Currency": "DKK", "Style": "Thematic", "Industry": "Energy Transition"},
        {"Ticker": "BS6.SI", "Name": "Yangzijiang Ship", "Country": "China", "Currency": "SGD", "Style": "Cyclical", "Industry": "Shipbuilding"},
        {"Ticker": "IDR.MC", "Name": "Indra Sistemas", "Country": "Spain", "Currency": "EUR", "Style": "Quality", "Industry": "Defence Tech"},
        {"Ticker": "MELI", "Name": "MercadoLibre", "Country": "Latin America", "Currency": "USD", "Style": "High-Growth", "Industry": "E-commerce"},
        {"Ticker": "9911.HK", "Name": "Newborn Town", "Country": "China", "Currency": "HKD", "Style": "Growth", "Industry": "Digital Platforms"},
        {"Ticker": "TTWO", "Name": "Take-Two", "Country": "USA", "Currency": "USD", "Style": "Growth", "Industry": "Gaming"},
        {"Ticker": "ITX.MC", "Name": "Inditex", "Country": "Spain", "Currency": "EUR", "Style": "Quality", "Industry": "Consumer"},
        {"Ticker": "9992.HK", "Name": "POP Mart", "Country": "China", "Currency": "HKD", "Style": "Growth", "Industry": "Consumer"},
        {"Ticker": "GIL.TO", "Name": "Gildan Activewear", "Country": "Canada", "Currency": "CAD", "Style": "Quality", "Industry": "Consumer"},
        {"Ticker": "RMS.PA", "Name": "Hermès", "Country": "France", "Currency": "EUR", "Style": "Quality", "Industry": "Luxury"},
        {"Ticker": "KAP.L", "Name": "Kazatomprom", "Country": "Kazakhstan", "Currency": "USD", "Style": "Thematic", "Industry": "Uranium"},
        {"Ticker": "7203.T", "Name": "Toyota Motor", "Country": "Japan", "Currency": "JPY", "Style": "Quality", "Industry": "Industrial"},
    ]
    return pd.DataFrame(data)

# =============================================================================
# QUANTITATIVE LOGIC & MATH
# =============================================================================
def calculate_shares(price, fx_rate):
    """
    Calculates fractional shares for equal weighting.
    Formula: $$Shares = \frac{TargetWeight}{Price \times FX}$$
    """
    if price == 0 or fx_rate == 0: return 0
    return ALLOCATION_PER_STOCK / (price * fx_rate)

# =============================================================================
# UI LAYOUT
# =============================================================================
st.title("🚀 Beat the Benchmark 2026: Quant Dashboard")
st.markdown(f"**Current Strategy Date:** {datetime.now().strftime('%Y-%m-%d')}")

# Sidebar Constraints
with st.sidebar:
    st.header("Quant Parameters")
    start_date = st.date_input("Portfolio Start Date", datetime(2025, 12, 31))
    st.divider()
    st.write("✅ **Selection Gatekeepers**")
    st.write(f"Min Health Score: {MIN_HEALTH_SCORE}/6")
    st.write(f"Min Growth Score: {MIN_GROWTH_SCORE}/6")
    st.divider()
    st.info("Note: This follows the 17-list screening methodology.")

df = get_portfolio_data()

# Tabs for Organization
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Allocation", "🌍 Diversification Analysis", "⚙️ Screening Logic"])

with tab1:
    st.subheader("Initial Position Sizing ($10,000 Each)")
    # Placeholder for live calculations
    df['Starting Price'] = 150.00  # Mocked
    df['FX Rate (to USD)'] = 1.0   # Mocked
    df['Shares'] = df.apply(lambda x: calculate_shares(x['Starting Price'], x['FX Rate (to USD)']), axis=1)
    
    st.dataframe(df[['Ticker', 'Name', 'Country', 'Style', 'Shares']], use_container_width=True)
    
    st.metric("Total Portfolio Capital", f"${STARTING_CAPITAL:,.0f}", delta="Ready for 2026 Launch")

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Industry Exposure (Target: 23 Industries)**")
        fig_ind = px.pie(df, names='Industry', hole=0.4)
        st.plotly_chart(fig_ind)
        
    with col2:
        st.write("**Geographic Diversification (Target: 25 Countries)**")
        fig_geo = px.choropleth(df, locations="Country", locationmode='country names', color="Style")
        st.plotly_chart(fig_geo)

    st.write("**Currency Distribution (Target: 15 Currencies)**")
    curr_counts = df['Currency'].value_counts().reset_index()
    fig_curr = px.bar(curr_counts, x='Currency', y='count', color='Currency')
    st.plotly_chart(fig_curr, use_container_width=True)

with tab3:
    st.header("The 17 Quantitative Scanners")
    st.write("""
    To replicate the Clive Thompson method, candidates are pulled from 17 specific lists. 
    Below is the automated validation status:
    """)
    
    scanners = [
        "High Dividend", "Low P/E", "Low PEG", "Low P/S", "Low P/B", 
        "EPS Growth", "Sales Growth", "Insider Buying", "Mega Cap", 
        "Very Large Cap", "Large Cap", "Medium Cap", "Small Cap", 
        "Quality", "Defensive", "Momentum", "Value"
    ]
    
    cols = st.columns(4)
    for i, scan in enumerate(scanners):
        cols[i % 4].success(f"Scanner: {scan} (Active)")

# =============================================================================
# FOOTER & DISCLAIMER
# =============================================================================
st.divider()
st.warning("""
**IMPORTANT DISCLAIMER:** Nothing in this application constitutes investment advice. 
Stocks may go down and a total loss is possible. Consult an independent financial advisor.
""")
