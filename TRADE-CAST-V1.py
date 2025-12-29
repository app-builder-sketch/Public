import streamlit as st
import re
import time
import random
from datetime import datetime

# ==========================================
# 1. CONFIG & STYLING (Dark Mode / Cyberpunk)
# ==========================================
st.set_page_config(
    page_title="TRADECAST v3.0",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# Custom CSS to match the React styling (Tailwind approximation)
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #020617;
        color: #e2e8f0;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Inputs */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {
        background-color: #020617;
        color: white;
        border: 1px solid #334155;
        border-radius: 12px;
    }
    
    /* Buttons (Primary - Indigo) */
    .stButton > button {
        background: linear-gradient(to right, #4f46e5, #7e22ce);
        color: white;
        border: none;
        border-radius: 20px;
        font-weight: bold;
        letter-spacing: 0.1em;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 10px;
        color: #94a3b8;
        font-size: 14px;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #312e81;
        color: #818cf8;
    }

    /* Logs Container */
    .log-container {
        font-family: monospace;
        font-size: 12px;
        background-color: #020617;
        border-top: 1px solid #1e293b;
        padding: 10px;
        height: 150px;
        overflow-y: auto;
    }
    
    /* Syntax Highlighting overrides */
    .highlight-keyword { color: #3b82f6; }
    .highlight-function { color: #fbbf24; }
    .highlight-number { color: #10b981; }
    .highlight-string { color: #94a3b8; }
    .highlight-comment { color: #64748b; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if 'logs' not in st.session_state:
    st.session_state.logs = [
        "[SYSTEM] Core Engine: Gemini 3.0 / Pine V6.0 Active",
        "[INFO] Neural reasoning pipelines established..."
    ]

if 'signal' not in st.session_state:
    st.session_state.signal = {
        'symbol': 'BTC/USDT',
        'direction': 'BUY',
        'entry': 64250.0,
        'sl': 63000.0,
        'tp1': 65500.0,
        'tp2': 67000.0,
        'tp3': 69000.0,
        'timeframe': '1H',
        'strategy': 'Neural Momentum + RSI 6.0'
    }

if 'config' not in st.session_state:
    st.session_state.config = {
        'bot_token': '',
        'chat_id': ''
    }

if 'report_data' not in st.session_state:
    st.session_state.report_data = None

if 'pine_code' not in st.session_state:
    st.session_state.pine_code = ""

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def log(msg, type="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{type}]"
    entry = f"[{timestamp}] {prefix} {msg}"
    st.session_state.logs.append(entry)

def highlight_pine_script(code):
    if not code: return ""
    lines = code.split('\n')
    formatted_html = ""
    for line in lines:
        # Simple regex replacements (order matters)
        # Escape HTML
        line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Comments
        line = re.sub(r'(//.*)', r'<span class="highlight-comment">\1</span>', line)
        # Strings
        line = re.sub(r'(".*?"|\'.*?\')', r'<span class="highlight-string">\1</span>', line)
        # Keywords
        line = re.sub(r'\b(indicator|strategy|input|plot|if|else|for|true|false|alert|alertcondition|runtime|request|library)\b', r'<span class="highlight-keyword">\1</span>', line)
        # Built-ins
        line = re.sub(r'\b(ta\.[a-z_]+|ema|sma|rsi|macd|color\.[a-z_]+|math\.[a-z_]+|chart\.[a-z_]+)\b', r'<span class="highlight-function">\1</span>', line)
        # Numbers (negative lookbehind to avoid hex colors)
        line = re.sub(r'(?<!#)\b(\d+(\.\d+)?)\b', r'<span class="highlight-number">\1</span>', line)
        
        formatted_html += f"<div style='min-height: 1.25rem; font-family: monospace;'>{line}</div>"
    return formatted_html

def validate_signal(s):
    errs = []
    if not s['symbol'].strip(): errs.append("Critical Error: Market Symbol identification missing.")
    if s['entry'] <= 0: errs.append(f"Structural Error: Entry price ({s['entry']}) must be positive.")
    
    if s['direction'] == 'BUY':
        if s['sl'] >= s['entry']: errs.append(f"BUY Logic Error: SL ({s['sl']}) must be lower than Entry.")
        if s['entry'] >= s['tp1']: errs.append(f"BUY Logic Error: TP1 ({s['tp1']}) must be higher than Entry.")
        if s['tp1'] >= s['tp2']: errs.append(f"BUY Logic Error: TP2 must be higher than TP1.")
        if s['tp2'] >= s['tp3']: errs.append(f"BUY Logic Error: TP3 must be higher than TP2.")
    else: # SELL
        if s['sl'] <= s['entry']: errs.append(f"SELL Logic Error: SL ({s['sl']}) must be higher than Entry.")
        if s['entry'] <= s['tp1']: errs.append(f"SELL Logic Error: TP1 ({s['tp1']}) must be lower than Entry.")
        if s['tp1'] <= s['tp2']: errs.append(f"SELL Logic Error: TP2 must be lower than TP1.")
        if s['tp2'] <= s['tp3']: errs.append(f"SELL Logic Error: TP3 must be lower than TP2.")
        
    return errs

# Mock Service Functions
def generate_report_mock(symbol, depth, context):
    time.sleep(1.5) # Simulate API latency
    return {
        "title": f"HYPER-VECTOR ANALYSIS: {symbol}",
        "outlook": "BULLISH" if random.choice([True, False]) else "BEARISH",
        "summary": f"Neural consensus indicates strong accumulation zones. {depth} scan complete. Context: {context[:50]}...",
        "technicalDetails": "RSI(14): 45.2 (Neutral)\nMACD: Bullish Crossover imminent\nVolume Delta: +450 BTC\n\nDeep learning models suggest a deviation from standard accumulation distribution phases.",
        "sources": [{"title": "Binance Order Book Data", "uri": "#"}, {"title": "Glassnode On-Chain Metrics", "uri": "#"}]
    }

def generate_pine_mock(prompt):
    time.sleep(1.5)
    return f"""//@version=6
indicator("{prompt[:20]} Strategy", overlay=true)

// Input Parameters
len = input.int(14, "Length")
src = input.source(close, "Source")

// Logic generated by Gemini 3.0
rsiValue = ta.rsi(src, len)
maValue = ta.sma(src, 50)

// Plotting
plot(maValue, color=color.blue, title="SMA 50")
plotshape(ta.crossover(rsiValue, 30), style=shape.triangleup, location=location.belowbar, color=color.green, size=size.small, title="Buy Signal")
plotshape(ta.crossunder(rsiValue, 70), style=shape.triangledown, location=location.abovebar, color=color.red, size=size.small, title="Sell Signal")
"""

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("### 🧠 TRADECAST")
    st.markdown("<p style='font-size: 10px; color: #818cf8; letter-spacing: 0.2em; font-weight: bold;'>v3.0 NEURAL PROTOCOL</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    nav_selection = st.radio(
        "Navigation",
        ["Signal Center", "AI Reports Lab", "Script Forge", "Gateways"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Latency Indicator
    st.markdown("""
    <div style='background: #1e293b; padding: 10px; border-radius: 10px; border: 1px solid #334155;'>
        <p style='font-size: 9px; color: #64748b; font-weight: bold; margin: 0;'>ENGINE LATENCY</p>
        <div style='display: flex; align-items: center; gap: 10px; margin-top: 5px;'>
            <div style='flex-grow: 1; height: 4px; background: #334155; border-radius: 2px;'>
                <div style='width: 85%; height: 100%; background: #10b981; border-radius: 2px;'></div>
            </div>
            <span style='font-size: 9px; color: #10b981; font-weight: bold;'>OPTIMAL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. MAIN CONTENT
# ==========================================
st.markdown(f"""
<div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 20px; margin-bottom: 20px;'>
    <div>
        <h2 style='margin:0; font-weight: 900; letter-spacing: -1px;'>{nav_selection.upper()}</h2>
        <p style='margin:0; font-size: 10px; letter-spacing: 0.3em; color: #64748b; font-weight: bold; text-decoration: underline;'>GEMINI 3.0 / GPT 5.2 REASONING PROTOCOL</p>
    </div>
    <div style='background: #0f172a; padding: 5px 15px; border-radius: 20px; border: 1px solid #1e293b; display: flex; align-items: center; gap: 10px;'>
        <div style='width: 8px; height: 8px; background: #10b981; border-radius: 50%;'></div>
        <span style='font-size: 10px; font-weight: bold; color: #cbd5e1;'>SEARCH GROUNDING: ON</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------
# TAB 1: SIGNAL CENTER (DASHBOARD)
# ------------------------------------------
if nav_selection == "Signal Center":
    col_main, col_preview = st.columns([2, 1])
    
    with col_main:
        # Dummy Chart
        chart_data = [random.uniform(60000, 65000) for _ in range(50)]
        st.line_chart(chart_data, height=250)
        
        st.markdown("### 📈 SIGNAL ARCHITECT")
        
        # Inputs
        c1, c2 = st.columns([2, 1])
        with c1:
            st.session_state.signal['symbol'] = st.text_input("MARKET PAIR", value=st.session_state.signal['symbol'])
        with c2:
            st.session_state.signal['timeframe'] = st.selectbox("INTERVAL", ["1M", "5M", "15M", "1H", "4H", "1D", "1W"], index=3)
            
        c3, c4, c5 = st.columns(3)
        with c3:
            # Custom toggle for Buy/Sell
            direction_label = "BUY 🟢" if st.session_state.signal['direction'] == "BUY" else "SELL 🔴"
            if st.button(f"Current: {direction_label}", use_container_width=True):
                st.session_state.signal['direction'] = "SELL" if st.session_state.signal['direction'] == "BUY" else "BUY"
                st.rerun()
                
        c6, c7 = st.columns(2)
        with c6:
            st.session_state.signal['entry'] = st.number_input("ENTRY TARGET", value=st.session_state.signal['entry'])
        with c7:
            st.session_state.signal['sl'] = st.number_input("STOP LOSS", value=st.session_state.signal['sl'])
            
        c8, c9, c10 = st.columns(3)
        with c8: st.session_state.signal['tp1'] = st.number_input("TP 1", value=st.session_state.signal['tp1'])
        with c9: st.session_state.signal['tp2'] = st.number_input("TP 2", value=st.session_state.signal['tp2'])
        with c10: st.session_state.signal['tp3'] = st.number_input("TP 3", value=st.session_state.signal['tp3'])
        
        st.session_state.signal['strategy'] = st.text_area("REASONING BASIS", value=st.session_state.signal['strategy'], height=100)
        
        if st.button("DEPLOY ENCRYPTED BROADCAST", use_container_width=True):
            errors = validate_signal(st.session_state.signal)
            if errors:
                for e in errors: log(e, "ERROR")
                st.error(f"Protocol Validation Breach: {len(errors)} errors detected.")
            elif not st.session_state.config['bot_token']:
                log("Gateway Auth Error: Credentials undefined.", "ERROR")
                st.error("Missing API Credentials. Check Gateways tab.")
            else:
                with st.spinner("TRANSMITTING NEURAL PACKET..."):
                    time.sleep(1.5) # Simulate network
                    log(f"Deploying neural broadcast: {st.session_state.signal['symbol']}", "ACTION")
                    log("Broadcast delivered via Secure Telegram Gateway.", "SUCCESS")
                    st.success("SIGNAL BROADCASTED SUCCESSFULLY")

    with col_preview:
        # Gateway Preview Card
        sig = st.session_state.signal
        is_buy = sig['direction'] == 'BUY'
        color_class = "#10b981" if is_buy else "#f43f5e" # Emerald vs Rose
        
        st.markdown(f"""
        <div style='background-color: #0f172a; border-radius: 20px; padding: 20px; border: 1px solid #1e293b; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);'>
            <div style='display: flex; align-items: center; gap: 10px; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-bottom: 15px;'>
                <span style='font-size: 20px;'>📱</span>
                <span style='font-size: 10px; font-weight: bold; letter-spacing: 0.2em; color: white;'>GATEWAY PREVIEW</span>
            </div>
            <div style='background-color: #020617; padding: 20px; border-radius: 15px; border-left: 4px solid {color_class}; position: relative;'>
                <p style='font-weight: 900; color: white; margin-bottom: 15px; text-transform: uppercase;'>🚀 SIGNAL: {sig['symbol']} 🚀</p>
                <div style='font-family: monospace; font-size: 11px; font-weight: bold; color: #94a3b8;'>
                    <p>DIRECTION: <span style='color: {color_class};'>{'🟢 BUY' if is_buy else '🔴 SELL'}</span></p>
                    <p>TF: <span style='color: white;'>{sig['timeframe']}</span></p>
                    <hr style='border-color: #334155; opacity: 0.5;'>
                    <p>📍 ENTRY: <span style='color: white; text-decoration: underline;'>{sig['entry']}</span></p>
                    <p style='color: #10b981;'>🎯 TP1: {sig['tp1']}</p>
                    <p style='color: #34d399;'>🎯 TP2: {sig['tp2']}</p>
                    <p style='color: #6ee7b7;'>🎯 TP3: {sig['tp3']}</p>
                    <p style='color: #f43f5e;'>🛑 SL: {sig['sl']}</p>
                    <hr style='border-color: #334155; opacity: 0.5;'>
                    <p style='color: #818cf8; font-style: italic; line-height: 1.5;'>{sig['strategy']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        errors = validate_signal(sig)
        status_color = "#f43f5e" if errors else "#10b981"
        status_text = "INVALID" if errors else "VERIFIED"
        
        st.markdown(f"""
        <div style='display: flex; justify-content: space-between; margin-top: 20px; padding: 0 10px;'>
            <span style='font-size: 9px; font-weight: bold; color: #64748b;'>LOGIC STATUS:</span>
            <span style='font-size: 9px; font-weight: bold; color: {status_color}; border: 1px solid {status_color}; padding: 2px 8px; border-radius: 10px; background: {status_color}22;'>{status_text}</span>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 2: AI REPORTS LAB
# ------------------------------------------
elif nav_selection == "AI Reports Lab":
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 📊 NEURAL ANALYTICS")
        depth = st.radio("REASONING DEPTH", ["QUICK", "DETAILED", "QUANT"], horizontal=True)
        context = st.text_area("MARKET CONTEXT", height=200, placeholder="Provide market vectors for deep synthesis...")
        
        if st.button("INITIALIZE ANALYSIS", use_container_width=True):
            if not context:
                st.error("Context required.")
            else:
                with st.spinner(f"SYNTHESIZING {depth} VECTORS..."):
                    log(f"Initializing {depth} Neural Thinking...", "AI")
                    st.session_state.report_data = generate_report_mock(st.session_state.signal['symbol'], depth, context)
                    log("Synthesis complete. Reasoning fidelity: High.", "AI")
                    st.rerun()

    with c2:
        report = st.session_state.report_data
        if report:
            outlook_color = "#10b981" if report['outlook'] == "BULLISH" else "#f43f5e"
            st.markdown(f"""
            <div style='background-color: #0f172a; padding: 30px; border-radius: 30px; border: 1px solid #1e293b;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <h3 style='margin:0; text-decoration: underline; text-decoration-color: #4f46e5; text-decoration-thickness: 4px;'>{report['title']}</h3>
                    <span style='color: {outlook_color}; background: {outlook_color}22; border: 1px solid {outlook_color}44; padding: 5px 10px; border-radius: 15px; font-size: 10px; font-weight: bold;'>{report['outlook']}</span>
                </div>
                <div style='margin-top: 20px; background: #020617; padding: 20px; border-radius: 15px; border: 1px solid #1e293b;'>
                    <p style='color: #818cf8; font-size: 10px; font-weight: bold; margin: 0;'>AI EXECUTIVE SUMMARY</p>
                    <p style='font-style: italic; color: #e2e8f0; margin-top: 5px;'>"{report['summary']}"</p>
                </div>
                <div style='margin-top: 20px; white-space: pre-line; font-size: 12px; color: #94a3b8;'>
                    {report['technicalDetails']}
                </div>
                <div style='margin-top: 20px;'>
                    <p style='font-size: 9px; font-weight: bold; color: #64748b;'>GROUNDING SOURCES</p>
                    {''.join([f"<div style='background: #4f46e511; padding: 10px; margin-bottom: 5px; border-radius: 10px; font-size: 11px; color: #cbd5e1;'>🔗 {src['title']}</div>" for src in report['sources']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("TRANSMIT INTELLIGENCE", use_container_width=True):
                if st.session_state.config['bot_token']:
                    log(f"Deploying neural intelligence broadcast: {report['title']}", "ACTION")
                    st.success("Intelligence Delivered.")
                else:
                    st.error("Gateways Offline.")
        else:
            st.info("Engine Standby. Initialize Analysis to begin.")

# ------------------------------------------
# TAB 3: SCRIPT FORGE
# ------------------------------------------
elif nav_selection == "Script Forge":
    st.markdown("### 🛠 PINE FORGE 6.0")
    
    prompt = st.text_input("LOGIC PROMPT", placeholder="Describe trading logic (e.g. RSI Divergence + Volume Profile)...")
    
    if st.button("FORGE V6 SOURCE"):
        if prompt:
            with st.spinner("COMPILING LOGIC..."):
                log("Compiling logic to Pine Script v6.0 Standard...", "AI")
                st.session_state.pine_code = generate_pine_mock(prompt)
                log("Code validated against V6.0 specification.", "SUCCESS")
                st.rerun()
    
    if st.session_state.pine_code:
        st.markdown("---")
        c1, c2 = st.columns([4, 1])
        with c1: st.caption("V6.0 ALPHA CORE PREVIEW")
        with c2: 
            if st.button("COPY SOURCE"):
                st.toast("Code copied to clipboard (simulated)")
                log("Pine V6 Source cloned.", "SYSTEM")
        
        # Custom HTML Renderer for Syntax Highlighting
        formatted_code = highlight_pine_script(st.session_state.pine_code)
        st.markdown(f"""
        <div style='background-color: #020617; padding: 20px; border-radius: 15px; border: 1px solid #334155; overflow-x: auto;'>
            {formatted_code}
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 4: GATEWAYS (SETTINGS)
# ------------------------------------------
elif nav_selection == "Gateways":
    st.markdown("### ⚙️ SYSTEM GATEWAYS")
    
    with st.container():
        st.markdown("""
        <div style='background: #0f172a; padding: 30px; border-radius: 30px; border: 1px solid #1e293b;'>
            <h4 style='color: white; margin-bottom: 20px;'>TELEGRAM CONFIGURATION</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.config['bot_token'] = st.text_input("BOT API KEY", value=st.session_state.config['bot_token'], type="password", placeholder="0000000:AAxxxx...")
        st.session_state.config['chat_id'] = st.text_input("DESTINATION CHAT ID", value=st.session_state.config['chat_id'], placeholder="-100xxxx...")
        
        if st.button("TEST CONNECTION"):
            if st.session_state.config['bot_token'] and st.session_state.config['chat_id']:
                with st.spinner("Pinging Gateway..."):
                    time.sleep(1)
                    log("Connection Established with Telegram API.", "SUCCESS")
                    st.success("GATEWAY ONLINE")
            else:
                st.error("Invalid Credentials.")

# ==========================================
# 6. FOOTER (LOGS)
# ==========================================
st.markdown("---")
st.markdown("""
<div style='display: flex; justify-content: space-between; font-size: 10px; font-weight: bold; color: #475569; margin-bottom: 5px;'>
    <span>NEURAL OVERRIDE TERMINAL</span>
    <span>STATUS: <span style='color: #10b981;'>ONLINE</span></span>
</div>
""", unsafe_allow_html=True)

# Render Logs
log_html = ""
for i, l in enumerate(st.session_state.logs):
    color = "#64748b"
    if "[ERROR]" in l: color = "#f43f5e"
    elif "[SUCCESS]" in l: color = "#10b981"
    elif "[AI]" in l: color = "#818cf8"
    
    log_html += f"<div style='color: {color}; margin-bottom: 4px;'><span style='opacity: 0.3; margin-right: 10px;'>{str(i+1).zfill(3)}</span>{l}</div>"

st.markdown(f"<div class='log-container'>{log_html}</div>", unsafe_allow_html=True)
