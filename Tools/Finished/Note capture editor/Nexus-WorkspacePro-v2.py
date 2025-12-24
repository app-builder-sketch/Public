import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import hashlib
from datetime import datetime, timedelta
import threading
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

# Professional Theme CSS
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
    .stApp {
        background-color: #0E1117;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #0D9488;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stExpander"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
    }
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
            # Notes Table
            c.execute("""CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT, content TEXT, tags TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            # Snippets Table
            c.execute("""CREATE TABLE IF NOT EXISTS snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, language TEXT, code TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            # AI Logs
            c.execute("""CREATE TABLE IF NOT EXISTS ai_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT, tokens INTEGER, cost REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

    # --- CRUD Operations ---
    def add_note(self, title, content, tags=""):
        with self.get_connection() as conn:
            conn.cursor().execute("INSERT INTO notes (title, content, tags) VALUES (?, ?, ?)", 
                                (title, content, tags))
    
    def get_notes(self):
        with self.get_connection() as conn:
            return [dict(row) for row in conn.cursor().execute("SELECT * FROM notes ORDER BY updated_at DESC").fetchall()]

    def update_note(self, note_id, title, content):
        with self.get_connection() as conn:
            conn.cursor().execute("UPDATE notes SET title=?, content=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", 
                                (title, content, note_id))

    def delete_note(self, note_id):
        with self.get_connection() as conn:
            conn.cursor().execute("DELETE FROM notes WHERE id=?", (note_id,))

    def log_ai_usage(self, provider, tokens, cost):
        with self.get_connection() as conn:
            conn.cursor().execute("INSERT INTO ai_logs (provider, tokens, cost) VALUES (?, ?, ?)", 
                                (provider, tokens, cost))

    def get_analytics(self):
        with self.get_connection() as conn:
            notes_count = conn.cursor().execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            snippets_count = conn.cursor().execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
            ai_data = conn.cursor().execute("SELECT SUM(cost), SUM(tokens) FROM ai_logs").fetchone()
            total_cost = ai_data[0] if ai_data[0] else 0.0
            return {"notes": notes_count, "snippets": snippets_count, "ai_cost": total_cost}

# Initialize Singleton
db = NexusDatabase()

# ==========================================
# 3. BACKEND: AI ASSISTANT MODULE
# ==========================================
class AIAssistant:
    def __init__(self):
        # Graceful handling if secrets are missing
        self.api_keys = {
            "openai": st.secrets.get("openai", {}).get("api_key"),
            "anthropic": st.secrets.get("anthropic", {}).get("api_key"),
            "google": st.secrets.get("google", {}).get("api_key")
        }
        
    def generate_response(self, prompt, provider="openai", system_role="You are a helpful assistant."):
        # Mock response if no keys are present (for demo purposes)
        if not self.api_keys.get(provider):
            time.sleep(1) # Simulate network
            return {
                "content": f"**[DEMO MODE]** API Key for {provider} not found.\n\nSimulated response to: '{prompt}'",
                "cost": 0.0, "tokens": 0
            }

        try:
            # --- OpenAI Implementation ---
            if provider == "openai":
                import openai
                client = openai.OpenAI(api_key=self.api_keys["openai"])
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "system", "content": system_role}, {"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content
                tokens = response.usage.total_tokens
                cost = (tokens / 1000) * 0.03 # Approx cost
                
            # --- Anthropic Implementation ---
            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=self.api_keys["anthropic"])
                response = client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                tokens = 0 # Anthropic usage tracking is complex, simplifying
                cost = 0.01 

            # --- Google Gemini Implementation ---
            elif provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=self.api_keys["google"])
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                content = response.text
                tokens = 0
                cost = 0.0

            # Log Usage
            db.log_ai_usage(provider, tokens, cost)
            return {"content": content, "cost": cost, "tokens": tokens}

        except Exception as e:
            return {"content": f"⚠️ Error: {str(e)}", "cost": 0, "tokens": 0}

ai = AIAssistant()

# ==========================================
# 4. PAGE: DASHBOARD
# ==========================================
def render_dashboard():
    st.markdown('<div class="main-header">📊 Nexus Dashboard</div>', unsafe_allow_html=True)
    
    analytics = db.get_analytics()
    
    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Total Notes", analytics["notes"], "+2")
    with col2:
        st.metric("💻 Snippets", analytics["snippets"], "+5")
    with col3:
        st.metric("💰 AI Cost", f"${analytics['ai_cost']:.4f}")
    with col4:
        st.metric("⚡ System Status", "Online", delta_color="normal")

    # Activity Charts (Mock Data for visualization)
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Weekly Activity")
        dates = [datetime.now() - timedelta(days=x) for x in range(7)]
        dates.reverse()
        chart_data = pd.DataFrame({
            "Date": dates,
            "Notes Created": [2, 5, 1, 6, 8, 3, 5],
            "AI Requests": [10, 15, 8, 22, 18, 12, 20]
        })
        fig = px.line(chart_data, x="Date", y=["Notes Created", "AI Requests"], 
                      template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        st.subheader("Storage Distribution")
        fig = px.pie(values=[analytics["notes"], analytics["snippets"], 15], 
                     names=["Notes", "Snippets", "Assets"], 
                     template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. PAGE: SMART NOTES
# ==========================================
def render_notes():
    st.markdown('<div class="main-header">📝 Smart Notes</div>', unsafe_allow_html=True)
    
    if "editing_note" not in st.session_state:
        st.session_state.editing_note = None

    col_list, col_editor = st.columns([1, 2])

    with col_list:
        st.subheader("Library")
        if st.button("➕ New Note", use_container_width=True):
            st.session_state.editing_note = None
        
        notes = db.get_notes()
        for note in notes:
            with st.container(border=True):
                st.markdown(f"**{note['title']}**")
                st.caption(note['updated_at'])
                c1, c2 = st.columns(2)
                if c1.button("Edit", key=f"edit_{note['id']}"):
                    st.session_state.editing_note = note
                if c2.button("Delete", key=f"del_{note['id']}"):
                    db.delete_note(note['id'])
                    st.rerun()

    with col_editor:
        st.subheader("Editor")
        note = st.session_state.editing_note
        
        with st.form("note_editor"):
            title = st.text_input("Title", value=note['title'] if note else "")
            tags = st.text_input("Tags (comma separated)", value=note['tags'] if note else "")
            content = st.text_area("Content", value=note['content'] if note else "", height=400)
            
            # AI Helper inside Editor
            with st.expander("✨ AI Tools"):
                ai_action = st.selectbox("Action", ["Summarize", "Fix Grammar", "Expand"])
                if st.form_submit_button("Run AI Helper"):
                    if not content:
                        st.warning("Write some content first.")
                    else:
                        prompt = f"{ai_action} this text:\n\n{content}"
                        with st.spinner("AI Working..."):
                            res = ai.generate_response(prompt)
                            st.info(res["content"])

            # Save Button
            if st.form_submit_button("💾 Save Note"):
                if note:
                    db.update_note(note['id'], title, content)
                    st.success("Note updated!")
                else:
                    db.add_note(title, content, tags)
                    st.success("Note created!")
                st.session_state.editing_note = None
                st.rerun()

# ==========================================
# 6. PAGE: AI ASSISTANT
# ==========================================
def render_ai_assistant():
    st.markdown('<div class="main-header">🤖 AI Assistant Hub</div>', unsafe_allow_html=True)

    # Sidebar Settings
    with st.sidebar:
        st.header("AI Configuration")
        provider = st.selectbox("Provider", ["openai", "anthropic", "google"])
        model_temp = st.slider("Temperature", 0.0, 1.0, 0.7)

    # Chat Interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask Nexus AI..."):
        # User Message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # AI Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ai.generate_response(prompt, provider=provider)
                st.write(response["content"])
                st.caption(f"Cost: ${response['cost']:.5f}")
                
        st.session_state.chat_history.append({"role": "assistant", "content": response["content"]})

# ==========================================
# 7. PAGE: SETTINGS & EXPORT
# ==========================================
def render_settings():
    st.markdown('<div class="main-header">⚙️ Settings</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["👤 Profile", "📤 Export"])
    
    with tab1:
        st.text_input("Username", "Admin User")
        st.text_input("Email", "admin@nexus.com")
        if st.button("Update Profile"):
            st.success("Profile saved locally.")
            
    with tab2:
        st.subheader("Backup Workspace")
        export_format = st.radio("Format", ["JSON", "CSV", "Markdown"])
        if st.button("Download Data"):
            notes = db.get_notes()
            df = pd.DataFrame(notes)
            
            if export_format == "CSV":
                data = df.to_csv(index=False).encode('utf-8')
                mime = "text/csv"
            else:
                data = df.to_json(orient="records").encode('utf-8')
                mime = "application/json"
                
            st.download_button(
                label="📥 Download Export",
                data=data,
                file_name=f"nexus_export.{export_format.lower()}",
                mime=mime
            )

# ==========================================
# 8. MAIN APP ROUTER
# ==========================================
def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown('<h2>🧠 Nexus Pro</h2>', unsafe_allow_html=True)
        
        menu = st.radio(
            "Navigate", 
            ["Dashboard", "Smart Notes", "AI Assistant", "Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.caption(f"v3.0.0 | Connected to {db.db_name}")

    # Routing
    if menu == "Dashboard":
        render_dashboard()
    elif menu == "Smart Notes":
        render_notes()
    elif menu == "AI Assistant":
        render_ai_assistant()
    elif menu == "Settings":
        render_settings()

if __name__ == "__main__":
    main()
