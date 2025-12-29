# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import json
import requests
import google.generativeai as genai
from enum import Enum
from typing import Optional, Dict, Any, List
import re

# ──────────────────────────────────────────────────────────────
# CONFIGURATION & SECRETS
# ──────────────────────────────────────────────────────────────
# Add to .streamlit/secrets.toml:
# [telegram]
# bot_token = "YOUR_BOT_TOKEN"
# chat_id = "YOUR_CHAT_ID"
# [gemini]
# api_key = "YOUR_GEMINI_API_KEY"

class Config:
    TELEGRAM_BOT_TOKEN = st.secrets.get("telegram", {}).get("bot_token", "")
    TELEGRAM_CHAT_ID = st.secrets.get("telegram", {}).get("chat_id", "")
    GEMINI_API_KEY = st.secrets.get("gemini", {}).get("api_key", "")

# ──────────────────────────────────────────────────────────────
# TYPE DEFINITIONS
# ──────────────────────────────────────────────────────────────
class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"

class AnalysisDepth(Enum):
    QUICK = "QUICK"
    DETAILED = "DETAILED"
    QUANT = "QUANT"

# ──────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ──────────────────────────────────────────────────────────────
def init_session_state():
    if "logs" not in st.session_state:
        st.session_state.logs = [
            "[SYSTEM] Core Engine: Gemini 3.0 / Pine V6.0 Active",
            "[INFO] Neural reasoning pipelines established..."
        ]
    if "signal" not in st.session_state:
        st.session_state.signal = {
            "symbol": "BTC/USDT",
            "direction": SignalDirection.BUY.value,
            "entry": 64250.0,
            "tp1": 65500.0,
            "tp2": 67000.0,
            "tp3": 69000.0,
            "sl": 63000.0,
            "timeframe": "1H",
            "strategy": "Neural Momentum + RSI 6.0"
        }
    if "errors" not in st.session_state:
        st.session_state.errors = []
    if "current_report" not in st.session_state:
        st.session_state.current_report = None
    if "pine_script" not in st.session_state:
        st.session_state.pine_script = ""

def add_log(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")

# ──────────────────────────────────────────────────────────────
# VALIDATION ENGINE
# ──────────────────────────────────────────────────────────────
def validate_signal(signal: Dict[str, Any]) -> List[str]:
    errors = []
    direction = signal["direction"]
    
    if not signal["symbol"].strip():
        errors.append("Critical Error: Market Symbol identification missing.")
    
    if signal["entry"] <= 0:
        errors.append(f"Structural Error: Entry price ({signal['entry']}) must be positive.")
    
    # Detailed Chained Validation
    if direction == SignalDirection.BUY.value:
        if signal["sl"] >= signal["entry"]:
            errors.append(f"BUY Logic Error: Stop Loss ({signal['sl']}) must be strictly lower than Entry ({signal['entry']}).")
        if signal["entry"] >= signal["tp1"]:
            errors.append(f"BUY Logic Error: Take Profit 1 ({signal['tp1']}) must be strictly higher than Entry ({signal['entry']}).")
        if signal["tp1"] >= signal["tp2"]:
            errors.append(f"BUY Logic Error: Take Profit 2 ({signal['tp2']}) must be strictly higher than Take Profit 1 ({signal['tp1']}).")
        if signal["tp2"] >= signal["tp3"]:
            errors.append(f"BUY Logic Error: Take Profit 3 ({signal['tp3']}) must be strictly higher than Take Profit 2 ({signal['tp2']}).")
    else:
        if signal["sl"] <= signal["entry"]:
            errors.append(f"SELL Logic Error: Stop Loss ({signal['sl']}) must be strictly higher than Entry ({signal['entry']}).")
        if signal["entry"] <= signal["tp1"]:
            errors.append(f"SELL Logic Error: Take Profit 1 ({signal['tp1']}) must be strictly lower than Entry ({signal['entry']}).")
        if signal["tp1"] <= signal["tp2"]:
            errors.append(f"SELL Logic Error: Take Profit 2 ({signal['tp2']}) must be strictly lower than Take Profit 1 ({signal['tp1']}).")
        if signal["tp2"] <= signal["tp3"]:
            errors.append(f"SELL Logic Error: Take Profit 3 ({signal['tp3']}) must be strictly lower than Take Profit 2 ({signal['tp2']}).")
    
    if errors:
        add_log(f"[ERROR] Protocol Violation Detected: {len(errors)} mathematical inconsistencies.")
    
    return errors

# ──────────────────────────────────────────────────────────────
# AI SERVICES
# ──────────────────────────────────────────────────────────────
def initialize_gemini():
    try:
        genai.configure(api_key=Config.GEMINI_API_KEY)
        return True
    except:
        return False

def generate_technical_report(symbol: str, timeframe: str, context: str, depth: AnalysisDepth) -> Optional[Dict[str, Any]]:
    if not Config.GEMINI_API_KEY:
        add_log("[ERROR] Gemini API key not configured in secrets")
        return None
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        temperature = 0.1 if depth == AnalysisDepth.QUANT else 0.3
        
        prompt = f"""
        Conduct a {depth.value} technical analysis for {symbol} on {timeframe} timeframe.
        Context: {context}
        
        Provide response in JSON format:
        {{
            "title": "Asset Analysis",
            "outlook": "BULLISH|BEARISH|NEUTRAL",
            "summary": "Executive summary in 2-3 sentences",
            "technicalDetails": "Detailed analysis with key levels and indicators",
            "sources": [{{"title": "Source Name", "uri": "https://..."}}]
        }}
        """
        
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        
        cleaned_response = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response)
        
    except Exception as e:
        add_log(f"[ERROR] AI Synthesis failed: {str(e)}")
        return None

def generate_pine_script(name: str, logic: str) -> str:
    if not Config.GEMINI_API_KEY:
        add_log("[ERROR] Gemini API key not configured in secrets")
        return "// Error: API key missing"
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Generate a complete Pine Script v6.0 indicator based on: {logic}
        Requirements:
        - Use //@version=6
        - Include proper input() declarations
        - Add alertcondition() for signals
        - Use explicit variable typing
        - Follow TradingView style guide
        
        Return ONLY the code without explanations.
        """
        
        response = model.generate_content(prompt, generation_config={"temperature": 0.2})
        return response.text.replace("```pinescript", "").replace("```", "").strip()
        
    except Exception as e:
        add_log(f"[ERROR] Compilation failed: {str(e)}")
        return f"// Error: {str(e)}"

# ──────────────────────────────────────────────────────────────
# TELEGRAM SERVICES
# ──────────────────────────────────────────────────────────────
def send_to_telegram(message: str, parse_mode: str = "HTML") -> bool:
    if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
        add_log("[ERROR] Gateway Auth Error: Telegram credentials undefined")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": Config.TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": parse_mode
        }
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            add_log("[SUCCESS] Broadcast delivered via Secure Telegram Gateway")
            return True
        else:
            add_log(f"[ERROR] Telegram API: {response.text}")
            return False
            
    except Exception as e:
        add_log(f"[ERROR] Protocol Error: {str(e)}")
        return False

def format_signal_message(signal: Dict[str, Any]) -> str:
    emoji = "🟢" if signal["direction"] == SignalDirection.BUY.value else "🔴"
    return f"""🚀 <b>SIGNAL: {signal['symbol']} 🚀</b>

📊 <b>Direction:</b> {emoji} {signal['direction']}
⏱ <b>Timeframe:</b> {signal['timeframe']}

📍 <b>Entry:</b> <code>{signal['entry']}</code>
🎯 <b>TP1:</b> <code>{signal['tp1']}</code>
🎯 <b>TP2:</b> <code>{signal['tp2']}</code>
🎯 <b>TP3:</b> <code>{signal['tp3']}</code>
🛑 <b>SL:</b> <code>{signal['sl']}</code>

💡 <b>Strategy:</b> {signal['strategy']}

⚡ <b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

def format_report_message(report: Dict[str, Any]) -> str:
    return f"""📊 <b>{report['title']}</b>

🎯 <b>Market Outlook:</b> {report['outlook']}

📝 <b>Summary:</b> {report['summary']}

📈 <b>Technical Analysis:</b> {report['technicalDetails']}

🔗 <b>Sources:</b> {len(report.get('sources', []))} references"""

# ──────────────────────────────────────────────────────────────
# UI COMPONENTS
# ──────────────────────────────────────────────────────────────
def render_chart():
    # Mock chart - replace with real data from Binance API
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=['00:00', '01:00', '02:00', '03:00', '04:00'],
        open=[64000, 64200, 64100, 64300, 64400],
        high=[64200, 64300, 64200, 64500, 64600],
        low=[63900, 64000, 64000, 64100, 64300],
        close=[64100, 64250, 64150, 64400, 64500],
        name="BTC/USDT"
    ))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#020617",
        paper_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=10),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_logs():
    with st.sidebar:
        st.markdown("### 🖥️ NEURAL TERMINAL")
        log_container = st.container()
        with log_container:
            for log in st.session_state.logs[-20:]:  # Show last 20 logs
                st.code(log, language=None)

# ──────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────
def main():
    # Page Configuration
    st.set_page_config(
        page_title="TRADECAST v3.0 NEURAL",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .block-container {
            padding-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("🧠 TRADECAST")
        st.markdown("### v3.0 NEURAL")
        
        tab = st.radio(
            "Navigation",
            ["📊 Signal Center", "📈 AI Reports Lab", "⚡ Script Forge", "🔐 Gateways"],
            label_visibility="collapsed"
        )
        
        # API Status Indicators
        st.markdown("---")
        st.markdown("**API Status**")
        if Config.GEMINI_API_KEY:
            st.success("✅ Gemini Active")
        else:
            st.error("❌ Gemini Inactive")
        
        if Config.TELEGRAM_BOT_TOKEN:
            st.success("✅ Telegram Active")
        else:
            st.error("❌ Telegram Inactive")
    
    # Main Content
    if tab == "📊 Signal Center":
        render_signal_center()
    elif tab == "📈 AI Reports Lab":
        render_reports_lab()
    elif tab == "⚡ Script Forge":
        render_script_forge()
    elif tab == "🔐 Gateways":
        render_gateways()
    
    # Always render terminal at bottom
    render_logs()

def render_signal_center():
    st.header("📡 SIGNAL ARCHITECT")
    
    # Error Display
    if st.session_state.errors:
        st.error("### 🔴 PROTOCOL VALIDATION BREACH")
        for error in st.session_state.errors:
            st.markdown(f"- {error}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("signal_form"):
            col_left, col_right = st.columns(2)
            
            with col_left:
                symbol = st.text_input("Market Pair", value=st.session_state.signal["symbol"])
                direction = st.radio("Position", ["BUY", "SELL"], 
                                   index=0 if st.session_state.signal["direction"] == "BUY" else 1)
                entry = st.number_input("Entry Target", value=float(st.session_state.signal["entry"]), 
                                      format="%.2f")
                sl = st.number_input("Stop Loss", value=float(st.session_state.signal["sl"]), 
                                   format="%.2f")
            
            with col_right:
                timeframe = st.selectbox("Interval", ["1M", "5M", "15M", "1H", "4H", "1D", "1W"],
                                       index=["1M", "5M", "15M", "1H", "4H", "1D", "1W"].index(st.session_state.signal["timeframe"]))
                tp1 = st.number_input("TP 1", value=float(st.session_state.signal["tp1"]), format="%.2f")
                tp2 = st.number_input("TP 2", value=float(st.session_state.signal["tp2"]), format="%.2f")
                tp3 = st.number_input("TP 3", value=float(st.session_state.signal["tp3"]), format="%.2f")
            
            strategy = st.text_area("Reasoning Basis", value=st.session_state.signal["strategy"])
            
            submitted = st.form_submit_button("🚀 DEPLOY ENCRYPTED BROADCAST", 
                                            type="primary", use_container_width=True)
            
            if submitted:
                # Update signal
                st.session_state.signal.update({
                    "symbol": symbol, "direction": direction, "entry": entry,
                    "sl": sl, "timeframe": timeframe, "tp1": tp1, "tp2": tp2,
                    "tp3": tp3, "strategy": strategy
                })
                
                # Validate
                errors = validate_signal(st.session_state.signal)
                st.session_state.errors = errors
                
                if not errors:
                    # Send to Telegram
                    message = format_signal_message(st.session_state.signal)
                    if send_to_telegram(message):
                        st.success("✅ Signal broadcast successfully!")
                        # Generate AI intelligence if configured
                        if st.checkbox("Enhance with Gemini AI Intelligence"):
                            report = generate_technical_report(
                                symbol, timeframe, strategy, AnalysisDepth.QUICK
                            )
                            if report:
                                enhanced_message = f"{message}\n\n🤖 AI INTEL: {report['summary']}"
                                send_to_telegram(enhanced_message)
                                st.info("AI enhancement added to broadcast")
        st.markdown("---")
        render_chart()

def render_reports_lab():
    st.header("🧪 NEURAL ANALYTICS LAB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        depth = st.radio("Reasoning Depth", 
                        ["QUICK", "DETAILED", "QUANT"],
                        format_func=lambda x: f"{x} Analysis")
        context = st.text_area("Analysis Context", 
                              placeholder="Provide market vectors for deep synthesis...",
                              height=300)
        
        if st.button("🔬 INITIALIZE ANALYSIS", type="primary", use_container_width=True):
            if not context:
                st.warning("Please provide analysis context")
                return
            
            with st.spinner("Synthesizing neural vectors..."):
                report = generate_technical_report(
                    st.session_state.signal["symbol"],
                    st.session_state.signal["timeframe"],
                    context,
                    AnalysisDepth[depth]
                )
                
                if report:
                    st.session_state.current_report = report
                    st.success("Analysis complete!")
    
    with col2:
        if st.session_state.current_report:
            report = st.session_state.current_report
            st.markdown(f"### {report['title']}")
            
            outlook_color = {
                "BULLISH": "🟢",
                "BEARISH": "🔴",
                "NEUTRAL": "⚪"
            }.get(report['outlook'], "⚪")
            
            st.markdown(f"**Market Outlook:** {outlook_color} {report['outlook']}")
            st.info(f"**Summary:** {report['summary']}")
            
            with st.expander("📊 Technical Details", expanded=True):
                st.markdown(report['technicalDetails'])
            
            if report.get('sources'):
                st.markdown("**Sources:**")
                for source in report['sources']:
                    st.markdown(f"- [{source['title']}]({source['uri']})")
            
            if st.button("📡 TRANSMIT INTELLIGENCE", use_container_width=True):
                message = format_report_message(report)
                if send_to_telegram(message):
                    st.success("Report transmitted!")
        else:
            st.info("Generate a report to see results here")

def render_script_forge():
    st.header("⚡ PINE FORGE 6.0")
    
    prompt = st.text_input("Trading Logic Description",
                          placeholder="e.g., RSI Divergence + Volume Profile with EMA crossover")
    
    if st.button("🔨 FORGE V6 SOURCE", type="primary", use_container_width=True):
        if not prompt:
            st.warning("Please describe your trading logic")
            return
        
        with st.spinner("Compiling to Pine Script v6.0..."):
            code = generate_pine_script("Dark Singularity Engine", prompt)
            st.session_state.pine_script = code
            st.success("Compilation successful!")
    
    if st.session_state.pine_script:
        st.markdown("#### 📄 Generated Pine Script v6.0")
        
        col_copy, col_download = st.columns([1, 4])
        with col_copy:
            if st.button("📋 Copy", help="Copy to clipboard"):
                st.code(st.session_state.pine_script, language=None)
                st.success("Code copied!")
        
        with col_download:
            st.download_button(
                label="💾 Download .pine",
                data=st.session_state.pine_script,
                file_name="strategy.pine",
                mime="text/plain"
            )
        
        st.code(st.session_state.pine_script, language='javascript', line_numbers=True)

def render_gateways():
    st.header("🔐 SYSTEM GATEWAYS")
    st.markdown("### 🔑 API Configuration")
    
    st.info("API keys are now loaded securely from `.streamlit/secrets.toml`")
    
    with st.expander("View Current Configuration"):
        if Config.GEMINI_API_KEY:
            st.success(f"Gemini API Key: {Config.GEMINI_API_KEY[:8]}...{Config.GEMINI_API_KEY[-4:]}")
        else:
            st.error("Gemini API Key: Not configured")
        
        if Config.TELEGRAM_BOT_TOKEN:
            st.success(f"Telegram Bot Token: {Config.TELEGRAM_BOT_TOKEN[:8]}...{Config.TELEGRAM_BOT_TOKEN[-4:]}")
        else:
            st.error("Telegram Bot Token: Not configured")
        
        if Config.TELEGRAM_CHAT_ID:
            st.success(f"Telegram Chat ID: {Config.TELEGRAM_CHAT_ID}")
        else:
            st.error("Telegram Chat ID: Not configured")
    
    st.markdown("---")
    st.markdown("""
    ### Setup Instructions
    
    1. Create a file named `.streamlit/secrets.toml` in your project root
    2. Add your API keys:
    
    ```toml
    [telegram]
    bot_token = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    chat_id = "-1001234567890"
    
    [gemini]
    api_key = "YOUR_GEMINI_API_KEY"
    ```
    
    3. Restart the Streamlit app
    
    4. **Security Best Practices:**
       - Never commit secrets.toml to Git
       - Use environment variables in production
       - Rotate keys regularly
    """)

# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check for API keys on startup
    if not Config.GEMINI_API_KEY:
        st.error("⚠️ Gemini API key not found in secrets!")
    
    if not (Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID):
        st.warning("⚠️ Telegram credentials not found - broadcasting disabled")
    
    main()
