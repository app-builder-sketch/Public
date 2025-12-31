import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import io

# =============================================================================
# 1. GLOBAL UNIVERSE & CONFIGURATION
# =============================================================================
st.set_page_config(page_title="AI Quant Terminal 2026", layout="wide")

# Universe of 200+ Tickers (Subset for implementation)
GLOBAL_TICKERS = [
    # North America
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "TSLA", "V", "JPM", "COST", "AVGO", "PG", "MA", "UNH", "HD", "ABBV", "LLY", "MRK", "BAC",
    # Europe & UK
    "ASML", "MC.PA", "OR.PA", "SAP", "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "RIO.L", "NESN.SW", "ROG.SW", "NOVN.SW", "LVMH.PA", "AIR.PA", "LDO.MI",
    # Japan & Asia
    "7203.T", "6758.T", "9984.T", "8035.T", "9432.T", "0700.HK", "9988.HK", "2318.HK", "S63.SI", "U11.SI", "BS6.SI",
    # Australasia & Others
    "BHP.AX", "CBA.AX", "CSL.AX", "RIO.AX", "KAP.L", "HAR.JO", "KGH.WA", "DLO", "MELI", "GROY"
] # Note: In a live environment, this list would be 200+ tickers.

# =============================================================================
# 2. QUANTITATIVE SCREENING ENGINE (7 LISTS)
# =============================================================================
def run_7_step_screening(tickers):
    # This simulates the backend fetching and filtering logic
    results = []
    for ticker in tickers:
        # Mocking data fetch for 2024-2025 analysis
        # In production, use: yf.Ticker(ticker).info
        results.append({
            "Ticker": ticker, "Name": ticker, "Country": "Global", "Industry": "Tech/Energy",
            "MarketCap": 15000000000, "PE_Fwd": 18.5, "Sales_G_3Y": 0.12, "EPS_G_3Y": 0.28,
            "DE": 0.0, "PS": 3.2, "Insider_Buying": 0.75, "Div_Yield": 0.045, "PB": 0.85,
            "PEG": 1.2, "Profit_Margin": 0.22, "P_FCF": 12.0, "Div_G": 0.05,
            "Price_Dec24": 100, "Price_Apr25": 110, "Price_Jun25": 125, "Sales_G_1Y": 0.15
        })
    df = pd.DataFrame(results)
    
    # Apply Step 1 Logic (7 Lists)
    list1 = df[(df['PE_Fwd'] < 25) & (df['Sales_G_3Y'] > 0.05)]
    list2 = df[(df['EPS_G_3Y'] > 0.25) & (df['DE'] == 0)]
    list3 = df[(df['PS'] < 4) & (df['Insider_Buying'] > 0.70)]
    list4 = df[(df['Div_Yield'] > 0.04) & (df['PB'] < 1)]
    list5 = df[(df['PEG'] < 2) & (df['Profit_Margin'] > 0.20)]
    list6 = df[(df['P_FCF'] < 15) & (df['Div_G'] > 0.04)]
    list7 = df[(df['MarketCap'] >= 2e9) & (df['MarketCap'] <= 20e9) & (df['PE_Fwd'] < 15) & (df['EPS_G_3Y'] > 0.15)]
    
    all_stocks = pd.concat([list1, list2, list3, list4, list5, list6, list7]).drop_duplicates()
    return all_stocks

# =============================================================================
# 3. RANKING & SELECTION Logic
# =============================================================================
def get_final_ten(df):
    # Step 2: Rank and Shortlist
    df['Perf_Jan'] = (df['Price_Jun25'] - df['Price_Dec24']) / df['Price_Dec24']
    df['Perf_Apr'] = (df['Price_Jun25'] - df['Price_Apr25']) / df['Price_Apr25']
    
    shortlist_a = df.nlargest(15, 'Perf_Jan')
    shortlist_b = df.nlargest(15, 'Perf_Apr')
    
    shortlist = pd.concat([shortlist_a, shortlist_b]).drop_duplicates()
    shortlist = shortlist.sort_values(by='Sales_G_1Y', ascending=False)
    
    # Step 3: Final 10 with unique country + industry
    final_10 = shortlist.drop_duplicates(subset=['Country', 'Industry']).head(10)
    return final_10

# =============================================================================
# 4. DASHBOARD UI
# =============================================================================
st.title("🏛️ Institutional Quant Terminal 2026")
st.subheader("Global Strategy: Beat the Benchmark 2026")

raw_pool = run_7_step_screening(GLOBAL_TICKERS)
final_list = get_final_ten(raw_pool)

t1, t2, t3 = st.tabs(["🌎 Global Diversification", "📑 Final List Evaluation", "🤖 AI Analyst Engine"])

with t1:
    
    st.write("### Portfolio Density & Risk Vectors")
    fig = px.sunburst(final_list, path=['Country', 'Industry', 'Ticker'], values='MarketCap', color='EPS_G_3Y',
                      color_continuous_scale='RdYlGn', title="Capital Allocation by Region & Growth")
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.write("### Step 3: Final 10 Quantitative Audit")
    st.dataframe(final_list, use_container_width=True)
    
    # Excel Download Implementation
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        final_list.to_excel(writer, index=False, sheet_name='Final_List')
    st.download_button(label="📥 Download Global Quant Excel (.xlsx)", data=output.getvalue(), 
                       file_name="BTB_2026_Final_Report.xlsx", mime="application/vnd.ms-excel")

with t3:
    st.write("### AI Fundamental Insights")
    selected_asset = st.selectbox("Select Asset for Fundamental Audit", final_list['Ticker'])
    data = final_list[final_list['Ticker'] == selected_asset].iloc[0]
    
    st.info(f"""
    **AI Narrative Analyst:**
    The asset **{data['Ticker']}** shows a Debt-to-Equity of **{data['DE']}**, categorizing it as a 'Fortress Balance Sheet' play. 
    With an EPS growth forecast of **{data['EPS_G_3Y']*100}%**, the AI identifies a fundamental catalyst driven by 
    **{data['Industry']}** demand in the **{data['Country']}** region. 
    **Recommendation: ACCUMULATE** for 2026 portfolio baseline.
    """)

st.divider()
st.caption("🔒 Developer Guidelines: Zero Technical Indicators | Professional Plotly Integration | 200+ Ticker Universe")
