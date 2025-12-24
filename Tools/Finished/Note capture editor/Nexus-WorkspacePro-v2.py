import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import time
import threading
import os
from datetime import datetime, timedelta
from contextlib import contextmanager

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Nexus Workspace Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .stApp { background-color: #0E1117; }
    div[data-testid="stExpander"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-active { background-color: #064E3B; color: #34D399; }
    .status-inactive { background-color: #450a0a; color: #FCA5A5; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BACKEND: DATABASE MODULE
# ==========================================
class NexusDatabase:
    def __init__(self):
        self.db_name = "nexus_workspace.db"
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
            c.execute("""CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT, content TEXT, tags TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, language TEXT, code TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS ai_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT, tokens INTEGER, cost REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

    def add_note(self, title, content, tags=""):
        with self.get_connection() as conn:
            conn.cursor().execute("INSERT INTO notes (title, content, tags) VALUES (?, ?, ?)", (title, content, tags))
    
    def get_notes(self):
        with self.get_connection() as conn:
            return [dict(row) for row in conn.cursor().execute("SELECT * FROM notes ORDER BY updated_at DESC").fetchall()]

    def update_note(self, note_id, title, content):
        with self.get_connection() as conn:
            conn.cursor().execute("UPDATE notes SET title=?, content=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (title, content, note_id))

    def delete_note(self, note_id):
        with self.get_connection() as conn:
            conn.cursor().execute("DELETE FROM notes WHERE id=?", (note_id,))

    def log_ai_usage(self, provider, tokens, cost):
        with self.get_connection() as conn:
            conn.cursor().execute("INSERT INTO ai_logs (provider, tokens, cost) VALUES (?, ?, ?)", (provider, tokens, cost))

    def get_analytics(self):
        with self.get_connection() as conn:
            notes = conn.cursor().execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            snippets = conn.cursor().execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
            ai = conn.cursor().execute("SELECT SUM(cost) FROM ai_logs").fetchone()
            return {"notes": notes, "snippets": snippets, "ai_cost": ai[0] if ai[0] else 0.0}

db = NexusDatabase()

# ==========================================
# 3. BACKEND: AI ASSISTANT (AUTO-LOAD)
# ==========================================
class AIAssistant:
    def __init__(self):
        self.api_keys = {}
        self.providers_status = {}
        self._auto_load_keys()

    def _auto_load_keys(self):
        """Automatically load keys from secrets.toml or environment variables"""
        
        # 1. Load OpenAI
        self.api_keys["openai"] = self._get_key("openai")
        self.providers_status["openai"] = bool(self.api_keys["openai"])
        
        # 2. Load Anthropic
        self.api_keys["anthropic"] = self._get_key("anthropic")
        self.providers_status["anthropic"] = bool(self.api_keys["anthropic"])

        # 3. Load Google
        self.api_keys["google"] = self._get_key("google")
        self.providers_status["google"] = bool(self.api_keys["google"])

    def _get_key(self, provider):
        """Helper to find key in secrets.toml (nested or flat) or OS Env"""
        # Try secrets.toml: [provider] api_key = "..."
        try:
            if provider in st.secrets and "api_key" in st.secrets[provider]:
                return st.secrets[provider]["api_key"]
        except:
            pass
            
        # Try OS Environment Variable
        env_var = f"{provider.upper()}_API_KEY"
        if os.getenv(env_var):
            return os.getenv(env_var)
            
        return None

    def generate_response(self, prompt, provider="openai", system_role="You are a helpful assistant."):
        if not self.api_keys.get(provider):
            time.sleep(1)
            return {
                "content": f"⚠️ **API Key Missing**: Could not find API key for **{provider}** in `secrets.toml`.\n\nSimulating response...",
                "cost": 0.0, "tokens": 0
            }

        try:
            if provider == "openai":
                import openai
                client = openai.OpenAI(api_key=self.api_keys["openai"])
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "system", "content": system_role}, {"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content
                tokens = response.usage.total_tokens
                cost = (tokens / 1000) * 0.03
                
            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=self.api_keys["anthropic"])
                response = client.messages.create(
                    model="claude-3-opus-20240229", max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                tokens = 0; cost = 0.01

            elif provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=self.api_keys["google"])
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                content = response.text
                tokens = 0; cost = 0.0

            db.log_ai_usage(provider, tokens, cost)
            return {"content": content, "cost": cost, "tokens": tokens}

        except Exception as e:
            return {"content": f"❌ Error: {str(e)}", "cost": 0, "tokens": 0}

ai = AIAssistant()

# ==========================================
# 4. DASHBOARD PAGE
# ==========================================
def render_dashboard():
    st.markdown('<div class="main-header">📊 Nexus Dashboard</div>', unsafe_allow_html=True)
    analytics = db.get_analytics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📝 Total Notes", analytics["notes"])
    with col2: st.metric("💻 Snippets", analytics["snippets"])
    with col3: st.metric("💰 AI Cost", f"${analytics['ai_cost']:.4f}")
    with col4: st.metric("🤖 Active AI", sum(ai.providers_status.values()))

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Weekly Activity")
        dates = [datetime.now() - timedelta(days=x) for x in range(7)][::-1]
        fig = px.line(x=dates, y=[5, 2, 8, 4, 10, 6, 8], labels={'x': 'Date', 'y': 'Actions'}, title="Activity Trend", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. NOTES PAGE
# ==========================================
def render_notes():
    st.markdown('<div class="main-header">📝 Smart Notes</div>', unsafe_allow_html=True)
    if "editing_note" not in st.session_state: st.session_state.editing_note = None

    col_list, col_editor = st.columns([1, 2])
    with col_list:
        if st.button("➕ New Note", use_container_width=True): st.session_state.editing_note = None
        for note in db.get_notes():
            with st.container(border=True):
                st.markdown(f"**{note['title']}**")
                c1, c2 = st.columns(2)
                if c1.button("Edit", key=f"e_{note['id']}"): st.session_state.editing_note = note
                if c2.button("Del", key=f"d_{note['id']}"): db.delete_note(note['id']); st.rerun()

    with col_editor:
        note = st.session_state.editing_note
        with st.form("editor"):
            title = st.text_input("Title", value=note['title'] if note else "")
            content = st.text_area("Body", value=note['content'] if note else "", height=400)
            
            with st.expander("✨ AI Assistant"):
                action = st.selectbox("Action", ["Summarize", "Fix Grammar", "Expand"])
                if st.form_submit_button("Run AI"):
                    res = ai.generate_response(f"{action}: {content}")
                    st.info(res["content"])

            if st.form_submit_button("💾 Save"):
                if note: db.update_note(note['id'], title, content)
                else: db.add_note(title, content)
                st.session_state.editing_note = None; st.rerun()

# ==========================================
# 6. AI PAGE
# ==========================================
def render_ai_assistant():
    st.markdown('<div class="main-header">🤖 AI Assistant Hub</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🔑 Provider Status")
        for prov, status in ai.providers_status.items():
            color = "status-active" if status else "status-inactive"
            text = "Active" if status else "Missing Key"
            st.markdown(f"**{prov.title()}**: <span class='status-badge {color}'>{text}</span>", unsafe_allow_html=True)
        
        active_providers = [p for p, s in ai.providers_status.items() if s]
        if active_providers:
            provider = st.selectbox("Select Provider", active_providers)
        else:
            st.warning("No API Keys found in secrets.toml")
            provider = "openai" # default to mock

    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    if prompt := st.chat_input("Ask Nexus..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner(f"Asking {provider}..."):
                res = ai.generate_response(prompt, provider=provider)
                st.write(res["content"])
                st.session_state.chat_history.append({"role": "assistant", "content": res["content"]})

# ==========================================
# 7. MAIN APP
# ==========================================
def main():
    with st.sidebar:
        st.title("🧠 Nexus Pro")
        menu = st.radio("Navigate", ["Dashboard", "Smart Notes", "AI Assistant"], label_visibility="collapsed")
        st.markdown("---")
        st.caption("v3.1 | Auto-Load Enabled")

    if menu == "Dashboard": render_dashboard()
    elif menu == "Smart Notes": render_notes()
    elif menu == "AI Assistant": render_ai_assistant()

if __name__ == "__main__":
    main()
