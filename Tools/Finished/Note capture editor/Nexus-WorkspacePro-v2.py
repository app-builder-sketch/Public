import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import threading
import os
import hashlib
import re
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Dict, List, Optional

# ==========================================
# 1. CONFIGURATION & STATE
# ==========================================
st.set_page_config(
    page_title="Nexus Workspace Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "authenticated" not in st.session_state:
    st.session_state.update({
        "authenticated": True, # Mock auth for immediate access
        "user_name": "Admin",
        "chat_history": [],
        "ai_usage": []
    })

# Professional Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #1E40AF 0%, #0D9488 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0D9488;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b;
        color: #38bdf8;
        border-bottom: 2px solid #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATABASE MODULE (Restored Full Schema)
# ==========================================
class NexusDatabase:
    def __init__(self):
        self.db_name = "nexus_production.db"
        self.lock = threading.Lock()
        self.init_database()

    @contextmanager
    def get_connection(self):
        with self.lock:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def init_database(self):
        with self.get_connection() as conn:
            c = conn.cursor()
            # 1. Notes (With Versioning columns from your original code)
            c.execute("""CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                tags TEXT DEFAULT '[]',
                is_pinned BOOLEAN DEFAULT 0,
                version INTEGER DEFAULT 1,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            
            # 2. Snippets (Restored)
            c.execute("""CREATE TABLE IF NOT EXISTS snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                language TEXT DEFAULT 'python',
                content TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            
            # 3. AI Usage (Restored full logging)
            c.execute("""CREATE TABLE IF NOT EXISTS ai_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                estimated_cost REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

    # --- Note Operations ---
    def upsert_note(self, note_id, title, content, tags):
        with self.get_connection() as conn:
            if note_id:
                conn.cursor().execute("UPDATE notes SET title=?, content=?, tags=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (title, content, tags, note_id))
            else:
                conn.cursor().execute("INSERT INTO notes (title, content, tags) VALUES (?, ?, ?)", (title, content, tags))

    def get_notes(self):
        with self.get_connection() as conn:
            return [dict(row) for row in conn.cursor().execute("SELECT * FROM notes ORDER BY is_pinned DESC, updated_at DESC").fetchall()]

    def delete_note(self, note_id):
        with self.get_connection() as conn:
            conn.cursor().execute("DELETE FROM notes WHERE id=?", (note_id,))

    # --- Snippet Operations ---
    def add_snippet(self, name, language, content, description=""):
        with self.get_connection() as conn:
            conn.cursor().execute("INSERT INTO snippets (name, language, content, description) VALUES (?, ?, ?, ?)", 
                                (name, language, content, description))

    def get_snippets(self):
        with self.get_connection() as conn:
            return [dict(row) for row in conn.cursor().execute("SELECT * FROM snippets ORDER BY created_at DESC").fetchall()]

    # --- Analytics & Logging ---
    def log_ai_usage(self, provider, model, p_tok, c_tok, cost):
        with self.get_connection() as conn:
            conn.cursor().execute("INSERT INTO ai_usage (provider, model, prompt_tokens, completion_tokens, estimated_cost) VALUES (?, ?, ?, ?, ?)",
                                (provider, model, p_tok, c_tok, cost))

    def get_stats(self):
        with self.get_connection() as conn:
            c = conn.cursor()
            notes = c.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            snippets = c.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
            cost = c.execute("SELECT SUM(estimated_cost) FROM ai_usage").fetchone()[0] or 0.0
            reqs = c.execute("SELECT COUNT(*) FROM ai_usage").fetchone()[0]
            return {"notes": notes, "snippets": snippets, "cost": cost, "requests": reqs}

db = NexusDatabase()

# ==========================================
# 3. ADVANCED AI ASSISTANT (Restored Logic)
# ==========================================
class AIAssistant:
    def __init__(self):
        self.providers = self._init_providers()
        self.current_provider = "openai"

    def _init_providers(self):
        # Auto-load keys
        keys = {}
        if "openai" in st.secrets: keys["openai"] = st.secrets["openai"]["api_key"]
        elif os.getenv("OPENAI_API_KEY"): keys["openai"] = os.getenv("OPENAI_API_KEY")
        
        if "anthropic" in st.secrets: keys["anthropic"] = st.secrets["anthropic"]["api_key"]
        
        if "google" in st.secrets: keys["google"] = st.secrets["google"]["api_key"]
        return keys

    def chat_completion(self, messages, temperature=0.7):
        """Unified completion handler"""
        provider = self.current_provider
        if provider not in self.providers:
            return {"error": f"Provider {provider} not configured", "content": "⚠️ **Error**: API Key missing."}

        try:
            if provider == "openai":
                import openai
                client = openai.OpenAI(api_key=self.providers["openai"])
                res = client.chat.completions.create(
                    model="gpt-4-turbo", messages=messages, temperature=temperature
                )
                content = res.choices[0].message.content
                # Cost calc (Approx)
                cost = (res.usage.total_tokens / 1000) * 0.03
                db.log_ai_usage("openai", "gpt-4", res.usage.prompt_tokens, res.usage.completion_tokens, cost)
                return {"content": content, "cost": cost}

            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=self.providers["anthropic"])
                # Convert format
                anth_msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
                res = client.messages.create(
                    model="claude-3-opus-20240229", messages=anth_msgs, max_tokens=2000
                )
                content = res.content[0].text
                db.log_ai_usage("anthropic", "claude-3", 0, 0, 0.01)
                return {"content": content, "cost": 0.01}
                
            elif provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=self.providers["google"])
                model = genai.GenerativeModel('gemini-pro')
                # Simple prompt extraction for Gemini
                last_msg = messages[-1]["content"]
                res = model.generate_content(last_msg)
                db.log_ai_usage("google", "gemini-pro", 0, 0, 0.0)
                return {"content": res.text, "cost": 0.0}

        except Exception as e:
            return {"error": str(e), "content": f"❌ Error: {str(e)}"}

    # --- Restored Advanced Methods ---
    def code_analysis(self, code, language):
        messages = [
            {"role": "system", "content": f"You are an expert {language} developer. Analyze for bugs and optimization."},
            {"role": "user", "content": f"Analyze this code:\n\n{code}"}
        ]
        return self.chat_completion(messages)

    def summarize_note(self, content, length="medium"):
        prompts = {
            "short": "1-2 sentences", 
            "medium": "one paragraph", 
            "detailed": "comprehensive bullet points"
        }
        messages = [
            {"role": "system", "content": "You are a professional summarizer."},
            {"role": "user", "content": f"Provide a {prompts[length]} summary of:\n\n{content}"}
        ]
        return self.chat_completion(messages)

    def convert_format(self, input_text, input_fmt, output_fmt):
        messages = [
            {"role": "system", "content": "You are a data format expert."},
            {"role": "user", "content": f"Convert this {input_fmt} to {output_fmt}:\n\n{input_text}"}
        ]
        return self.chat_completion(messages)

ai = AIAssistant()

# ==========================================
# 4. DASHBOARD PAGE
# ==========================================
def render_dashboard():
    st.markdown('<div class="main-header">📊 Nexus Dashboard</div>', unsafe_allow_html=True)
    stats = db.get_stats()
    
    # 1. Stats Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("📝 Total Notes", stats["notes"], "+1")
    with c2: st.metric("💻 Snippets", stats["snippets"], "+2")
    with c3: st.metric("🤖 AI Requests", stats["requests"], "+5")
    with c4: st.metric("💰 Total Cost", f"${stats['cost']:.4f}")

    # 2. Analytics Charts (Restored from your Analytics Page)
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Activity Trends")
        # Mock data to simulate the structure you requested
        df_act = pd.DataFrame({
            "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "Actions": [12, 19, 15, 25, 22, 30, 18]
        })
        fig = px.line(df_act, x="Day", y="Actions", title="Weekly Output", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        st.subheader("🤖 AI Usage Distribution")
        df_ai = pd.DataFrame({
            "Type": ["Code Gen", "Chat", "Summaries", "Analysis"],
            "Count": [45, 30, 15, 10]
        })
        fig = px.pie(df_ai, values="Count", names="Type", title="Request Types", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. SMART NOTES PAGE
# ==========================================
def render_notes():
    st.markdown('<div class="main-header">📝 Smart Notes</div>', unsafe_allow_html=True)
    
    if "active_note" not in st.session_state: st.session_state.active_note = None
    
    col_list, col_editor = st.columns([1, 2])
    
    with col_list:
        if st.button("➕ New Note", use_container_width=True): st.session_state.active_note = None
        
        notes = db.get_notes()
        for note in notes:
            with st.container(border=True):
                c1, c2 = st.columns([5,1])
                pinned = "📌 " if note['is_pinned'] else ""
                if c1.button(f"{pinned}{note['title']}", key=f"n_{note['id']}", use_container_width=True):
                    st.session_state.active_note = note
                # Delete logic
                if c2.button("🗑️", key=f"del_{note['id']}"):
                    db.delete_note(note['id'])
                    st.rerun()

    with col_editor:
        note = st.session_state.active_note
        with st.form("note_form"):
            title = st.text_input("Title", value=note['title'] if note else "")
            tags = st.text_input("Tags", value=note['tags'] if note else "")
            content = st.text_area("Body", value=note['content'] if note else "", height=500)
            
            # --- Restored Note Enhancer ---
            with st.expander("✨ AI Enhancer"):
                c_act, c_tone = st.columns(2)
                action = c_act.selectbox("Action", ["Summarize", "Fix Grammar", "Expand", "Action Items"])
                tone = c_tone.selectbox("Tone", ["Professional", "Casual", "Concise"])
                
                if st.form_submit_button("Run Enhancement"):
                    if not content:
                        st.warning("Enter content first")
                    else:
                        prompt = f"{action} this text in a {tone} tone:\n\n{content}"
                        with st.spinner("Processing..."):
                            res = ai.chat_completion([{"role": "user", "content": prompt}])
                            st.info(res["content"])
                            if "error" in res: st.error(res["error"])

            if st.form_submit_button("💾 Save Note"):
                db.upsert_note(note['id'] if note else None, title, content, tags)
                st.success("Saved!")
                st.rerun()

# ==========================================
# 6. CODE & SNIPPETS PAGE (Restored)
# ==========================================
def render_snippets():
    st.markdown('<div class="main-header">💻 Code Studio</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["Library", "AI Generator", "Analysis"])
    
    # Tab 1: Snippet Library
    with tabs[0]:
        snippets = db.get_snippets()
        for snip in snippets:
            with st.expander(f"{snip['language'].upper()}: {snip['name']}"):
                st.code(snip['content'], language=snip['language'])
                st.caption(snip['description'])
    
    # Tab 2: Generator (Save to library)
    with tabs[1]:
        st.subheader("Generate New Code")
        req_lang = st.selectbox("Language", ["Python", "JavaScript", "SQL", "HTML/CSS"])
        req_desc = st.text_area("Describe functionality...")
        
        if st.button("🚀 Generate Code"):
            msg = [{"role": "user", "content": f"Write {req_lang} code to: {req_desc}"}]
            res = ai.chat_completion(msg)
            st.session_state.generated_code = res["content"]
            
        if "generated_code" in st.session_state:
            st.code(st.session_state.generated_code)
            if st.button("💾 Save to Library"):
                db.add_snippet(f"AI Gen: {req_desc[:20]}...", req_lang.lower(), st.session_state.generated_code, req_desc)
                st.success("Snippet Saved!")

    # Tab 3: Analysis (Restored Code Analysis)
    with tabs[2]:
        st.subheader("AI Code Review")
        input_code = st.text_area("Paste Code for Review", height=300)
        input_lang = st.selectbox("Code Language", ["python", "javascript", "java", "cpp"])
        
        if st.button("🔍 Analyze Code"):
            with st.spinner("Analyzing logic, security, and performance..."):
                res = ai.code_analysis(input_code, input_lang)
                st.markdown(res["content"])

# ==========================================
# 7. AI ASSISTANT & TOOLS PAGE (Restored)
# ==========================================
def render_ai_tools():
    st.markdown('<div class="main-header">🤖 AI Laboratory</div>', unsafe_allow_html=True)
    
    # Provider Selection
    c1, c2, c3 = st.columns(3)
    if c1.button("🟢 OpenAI", use_container_width=True): ai.current_provider = "openai"
    if c2.button("🟣 Anthropic", use_container_width=True): ai.current_provider = "anthropic"
    if c3.button("🔵 Google", use_container_width=True): ai.current_provider = "google"
    st.caption(f"Current Provider: **{ai.current_provider.upper()}**")

    t1, t2 = st.tabs(["💬 Chat", "🔄 Converter"])
    
    with t1:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.write(msg["content"])
            
        if prompt := st.chat_input("Ask Nexus AI..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    res = ai.chat_completion(st.session_state.chat_history)
                    st.write(res["content"])
                    st.session_state.chat_history.append({"role": "assistant", "content": res["content"]})
                    if "cost" in res: st.caption(f"Cost: ${res['cost']:.5f}")

    with t2:
        st.subheader("Universal Format Converter")
        c_in, c_out = st.columns(2)
        fmt_in = c_in.selectbox("Input", ["JSON", "CSV", "XML", "Text"])
        fmt_out = c_out.selectbox("Output", ["Markdown Table", "HTML", "YAML", "SQL"])
        raw_text = st.text_area("Input Data")
        
        if st.button("🔄 Convert Format"):
            res = ai.convert_format(raw_text, fmt_in, fmt_out)
            st.code(res["content"], language=fmt_out.lower() if fmt_out != "Markdown Table" else "markdown")

# ==========================================
# 8. EXPORT HUB (Restored Logic)
# ==========================================
def render_export():
    st.markdown('<div class="main-header">📤 Export Hub</div>', unsafe_allow_html=True)
    
    st.subheader("Select Content")
    c1, c2 = st.columns(2)
    inc_notes = c1.checkbox("Include Notes", True)
    inc_snips = c2.checkbox("Include Snippets", True)
    
    st.subheader("Export Format")
    fmt = st.radio("Format", ["JSON (Full Backup)", "CSV (Table Data)", "Markdown (Readable)"])
    
    if st.button("🚀 Generate Export Package"):
        data_pack = {}
        if inc_notes: data_pack["notes"] = db.get_notes()
        if inc_snips: data_pack["snippets"] = db.get_snippets()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        if "JSON" in fmt:
            out_data = json.dumps(data_pack, indent=2)
            mime = "application/json"
            ext = "json"
        elif "CSV" in fmt:
            # Flatten for CSV
            if inc_notes:
                out_data = pd.DataFrame(data_pack["notes"]).to_csv(index=False)
            else:
                out_data = pd.DataFrame(data_pack["snippets"]).to_csv(index=False)
            mime = "text/csv"
            ext = "csv"
        else:
            out_data = "# Nexus Export\n\n"
            if inc_notes:
                for n in data_pack.get("notes", []):
                    out_data += f"## {n['title']}\n{n['content']}\n\n---\n\n"
            mime = "text/markdown"
            ext = "md"
            
        st.download_button(
            label=f"📥 Download {ext.upper()}",
            data=out_data,
            file_name=f"nexus_export_{timestamp}.{ext}",
            mime=mime
        )

# ==========================================
# 9. MAIN ROUTER
# ==========================================
def main():
    with st.sidebar:
        st.title("🧠 Nexus Pro")
        menu = st.radio(
            "Navigate", 
            ["Dashboard", "Smart Notes", "Code Studio", "AI Lab", "Export Hub"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()

    if menu == "Dashboard": render_dashboard()
    elif menu == "Smart Notes": render_notes()
    elif menu == "Code Studio": render_snippets()
    elif menu == "AI Lab": render_ai_tools()
    elif menu == "Export Hub": render_export()

if __name__ == "__main__":
    main()
