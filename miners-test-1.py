import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. CONFIGURATION & STYLING (The "Axiom" Look)
# ==========================================
st.set_page_config(page_title="Titan Miner Scout", page_icon="⛏️", layout="wide")

# Custom CSS inspired by Axiom/Titan
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 5px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.2rem !important; color: #00ff00; }
    h1, h2, h3 { color: #ffffff; text-shadow: 0 0 10px rgba(0,255,0,0.3); }
    .stDataFrame { border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (Cached & Robust)
# ==========================================

@st.cache_data(ttl=3600)
def get_macro_context():
    """Fetches spot prices for Gold, Silver, and DXY (Source: Titan Terminal)."""
    tickers = {"Gold": "GC=F", "Silver": "SI=F", "DXY": "DX-Y.NYB", "Miners ETF": "GDX"}
    try:
        data = yf.download(list(tickers.values()), period="5d", interval="1d", progress=False)['Close']
        res = {}
        for name, sym in tickers.items():
            if sym in data.columns:
                curr = data[sym].iloc[-1]
                prev = data[sym].iloc[-2]
                chg = ((curr - prev) / prev) * 100
                res[name] = (curr, chg)
        return res
    except:
        return {}

@st.cache_data(ttl=3600)
def analyze_miner(ticker):
    """
    Analyzes a single miner with caching to prevent UI freeze.
    Strictly follows the '3-Pillar' transcript logic.
    """
    try:
        stock = yf.Ticker(ticker)
        # Fetch minimal required data
        info = stock.info
        financials = stock.financials
        cashflow = stock.cashflow

        if financials.empty or cashflow.empty:
            return {"Ticker": ticker, "Status": "FAIL", "Reason": "No Data"}

        # 1. Profitability (Net Income > 0)
        try:
            # Try/Except block handles yfinance labeling inconsistencies
            ni_row = financials.loc['Net Income'] if 'Net Income' in financials.index else financials.loc['NetIncome']
            net_income = ni_row.iloc[0]
        except: net_income = 0

        # 2. Cash Flow (OCF > 0)
        try:
            if 'Total Cash From Operating Activities' in cashflow.index:
                ocf_row = cashflow.loc['Total Cash From Operating Activities']
            elif 'Operating Cash Flow' in cashflow.index:
                ocf_row = cashflow.loc['Operating Cash Flow']
            else: ocf_row = None
            ocf = ocf_row.iloc[0] if ocf_row is not None else 0
        except: ocf = 0

        # 3. Growth (Rev or Earnings Growth > 0)
        eg = info.get('earningsGrowth', 0)
        rg = info.get('revenueGrowth', 0)
        # Handle NoneTypes from API
        eg = 0 if eg is None else eg
        rg = 0 if rg is None else rg
        
        passes_profit = net_income > 0
        passes_cf = ocf > 0
        passes_growth = (eg > 0) or (rg > 0)

        # Status Logic
        status = "PASS" if (passes_profit and passes_cf and passes_growth) else "FAIL"

        return {
            "Ticker": ticker,
            "Status": status,
            "Net Income": net_income,
            "Operating Cash Flow": ocf,
            "Earnings Growth": eg * 100,
            "Revenue Growth": rg * 100,
            "Market Cap": info.get('marketCap', 0),
            "Close": info.get('previousClose', 0),
            "Pass_Profit": passes_profit,
            "Pass_CF": passes_cf,
            "Pass_Growth": passes_growth
        }
    except Exception as e:
        return {"Ticker": ticker, "Status": "FAIL", "Reason": f"Error: {str(e)}"}

# ==========================================
# 3. UI DASHBOARD
# ==========================================
def main():
    st.title("⛏️ Titan Miner Scout")
    st.markdown("### *Institutional-Grade Screener based on Fundamental Pillars*")
    
    # --- MACRO CONTEXT ROW (Upgrade from Titan Terminal) ---
    macro = get_macro_context()
    if macro:
        cols = st.columns(4)
        for i, (name, (price, chg)) in enumerate(macro.items()):
            cols[i].metric(name, f"${price:,.2f}", f"{chg:+.2f}%")
    st.divider()

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("⚙️ Configuration")
    default_tickers = "SVM, TXG.TO, GROY, GOLD, NEM, AEM, WPM, PAAS, KGC, AU"
    user_input = st.sidebar.text_area("Ticker List", value=default_tickers, height=150)
    
    if st.sidebar.button("🚀 Run Titan Analysis", use_container_width=True):
        tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]
        
        results = []
        # Progress bar for better UX
        bar = st.progress(0)
        status = st.empty()

        for i, t in enumerate(tickers):
            status.caption(f"Extracting financials for {t}...")
            data = analyze_miner(t)
            if "Reason" not in data:
                results.append(data)
            bar.progress((i + 1) / len(tickers))
        
        status.empty()
        bar.empty()

        if results:
            df = pd.DataFrame(results)
            pass_df = df[df['Status'] == "PASS"]
            fail_df = df[df['Status'] == "FAIL"]

            # --- TABBED INTERFACE (Upgrade from Axiom) ---
            tab1, tab2, tab3 = st.tabs(["✅ Qualified Miners", "❌ Disqualified", "📊 Visualization"])

            with tab1:
                st.subheader(f"Qualified Candidates ({len(pass_df)})")
                if not pass_df.empty:
                    # Formatting for cleaner display
                    disp_df = pass_df[["Ticker", "Close", "Net Income", "Operating Cash Flow", "Earnings Growth", "Revenue Growth"]]
                    st.dataframe(
                        disp_df.style.format({
                            "Close": "${:.2f}",
                            "Net Income": "${:,.0f}",
                            "Operating Cash Flow": "${:,.0f}",
                            "Earnings Growth": "{:.2f}%",
                            "Revenue Growth": "{:.2f}%"
                        }).background_gradient(subset=["Operating Cash Flow"], cmap="Greens"),
                        use_container_width=True
                    )
                else:
                    st.warning("No miners met the strict '3-Pillar' criteria.")

            with tab2:
                st.subheader(f"Disqualified ({len(fail_df)})")
                if not fail_df.empty:
                    def failure_reason(row):
                        reasons = []
                        if not row['Pass_Profit']: reasons.append("Unprofitable")
                        if not row['Pass_CF']: reasons.append("Neg Cash Flow")
                        if not row['Pass_Growth']: reasons.append("No Growth")
                        return ", ".join(reasons)
                    
                    fail_df['Reason'] = fail_df.apply(failure_reason, axis=1)
                    st.dataframe(fail_df[["Ticker", "Reason", "Net Income", "Operating Cash Flow"]], use_container_width=True)

            with tab3:
                # --- PLOTLY VISUALIZATION (Upgrade from TITAN MOBILE) ---
                st.subheader("Financial Health Radar")
                if not pass_df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=pass_df['Ticker'], 
                        y=pass_df['Net Income'], 
                        name='Net Income',
                        marker_color='#00ff00'
                    ))
                    fig.add_trace(go.Bar(
                        x=pass_df['Ticker'], 
                        y=pass_df['Operating Cash Flow'], 
                        name='Op Cash Flow',
                        marker_color='#00ccff'
                    ))
                    fig.update_layout(
                        barmode='group', 
                        template='plotly_dark', 
                        height=500,
                        title="Profit vs. Cash Flow",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No passing data to visualize.")

if __name__ == "__main__":
    main()
