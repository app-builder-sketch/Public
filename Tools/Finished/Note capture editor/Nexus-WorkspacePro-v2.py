import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import time
import threading
import os
import json
from datetime import datetime, timedelta
from contextlib import contextmanager

# ==========================================
# 1. PREMIUM CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Nexus Workspace Ultimate",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Nexus Workspace Pro v4.0\nHigh-performance knowledge management."
    }
)

# Advanced CSS for "Excellent Usability"
st.markdown("""
<style>
    /* smooth transitions */
    .stApp { transition: all 0.2s ease; }
    
    /* Header Gradient */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #3B82F6 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 2rem;
    }
    
    /* Card Styling */
    div[data-testid="stExpander"] {
        border: 1px solid #334155;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        background-color: #1E293B;
    }
    
    /* Input Fields Polish */
    .stTextInput input, .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #475569;
    }
    
    /* Status Badges */
    .badge {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-success { background: #065F46; color: #34D399; }
    .badge-error { background: #7F1D1D; color: #FCA5A5; }
    
    /* Hover Effects for Buttons */
    button[kind="secondary"]:hover {
        border-color: #3B82F6 !important;
        color: #3B82F6 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ROBUST BACKEND (Thread-Safe)
# ==========================================
class NexusDatabase:
    def __init__(self):
        self.db_name = "nexus_ultimate.db"
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
                is_pinned BOOLEAN DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS ai_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT, tokens INTEGER, cost REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

    # --- Enhanced CRUD ---
    def upsert_note(self, note_id, title, content, tags=""):
        with self.get_connection() as conn:
            if note_id:
                conn.cursor().execute(
                    "UPDATE notes SET title=?, content=?, tags=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", 
                    (title, content, tags, note_id)
                )
            else:
                conn.cursor().execute(
                    "INSERT INTO notes (title, content, tags) VALUES (?, ?, ?)", 
                    (title, content, tags)
                )

    def get_notes(self, search_query=""):
        with self.get_connection() as conn:
            sql = "SELECT * FROM notes"
            params = []
            if search_query:
                sql += " WHERE title LIKE ? OR content LIKE ?"
                params = [f"%{search_query}%", f"%{search_query}%"]
            sql += " ORDER BY is_pinned DESC, updated_at DESC"
            return [dict(row) for row in conn.cursor().execute(sql, params).fetchall()]

    def delete_note(self, note_id):
        with self.get_connection() as conn:
            conn.cursor().execute("DELETE FROM notes WHERE id=?", (note_id,))

    def toggle_pin(self, note_id, current_status):
        with self.get_connection() as conn:
            conn.cursor().execute("UPDATE notes SET is_pinned=? WHERE id=?", (not current_status, note_id))

    def get_stats(self):
        with self.get_connection() as conn:
            count = conn.cursor().execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            last_active = conn.cursor().execute("SELECT updated_at FROM notes ORDER BY updated_at DESC LIMIT 1").fetchone()
            ai_cost = conn.cursor().execute("SELECT SUM(cost) FROM ai_logs").fetchone()[0] or 0.0
            return {"count": count, "last_active": last_active[0] if last_active else "Never", "cost": ai_cost}

    def log_ai(self, provider, tokens, cost):
        with self.get_connection() as conn:
            conn.cursor().execute("INSERT INTO ai_logs (provider, tokens, cost) VALUES (?, ?, ?)", (provider, tokens, cost))

db = NexusDatabase()

# ==========================================
# 3. SMART AI ENGINE
# ==========================================
class SmartAI:
    def __init__(self):
        self._load_keys()
    
    def _load_keys(self):
        # Graceful loading from st.secrets or Environment
        self.keys = {
            "openai": st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY"),
            "anthropic": st.secrets.get("anthropic", {}).get("api_key") or os.getenv("ANTHROPIC_API_KEY"),
            "google": st.secrets.get("google", {}).get("api_key") or os.getenv("GOOGLE_API_KEY")
        }
    
    def get_active_providers(self):
        return [k for k, v in self.keys.items() if v]

    def stream_response(self, prompt, provider):
        """Generates a response. Includes a 'streaming' UI effect generator."""
        if not self.keys.get(provider):
            yield f"⚠️ **Error**: API Key for {provider} not found."
            return

        full_response = ""
        try:
            # Actual API Calls (Simplified for sync/streaming simulation)
            if provider == "openai":
                import openai
                client = openai.OpenAI(api_key=self.keys["openai"])
                # Simulating stream for UX consistency across providers
                res = client.chat.completions.create(model="gpt-4-turbo", messages=[{"role": "user", "content": prompt}])
                text = res.choices[0].message.content
                db.log_ai("openai", res.usage.total_tokens, 0.01)

            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=self.keys["anthropic"])
                res = client.messages.create(model="claude-3-opus-20240229", max_tokens=1000, messages=[{"role": "user", "content": prompt}])
                text = res.content[0].text
                db.log_ai("anthropic", 0, 0.01)

            elif provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=self.keys["google"])
                model = genai.GenerativeModel('gemini-pro')
                res = model.generate_content(prompt)
                text = res.text
                db.log_ai("google", 0, 0.0)

            # Simulated Streaming Effect for UX
            for word in text.split():
                full_response += word + " "
                yield full_response
                time.sleep(0.02) # Typing effect

        except Exception as e:
            yield f"❌ **System Error**: {str(e)}"

ai = SmartAI()

# ==========================================
# 4. UX COMPONENTS
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🧠 Nexus Ultimate")
        
        # Navigation with Icons
        page = st.radio(
            "Menu", 
            ["📊 Dashboard", "📝 Knowledge Base", "🤖 AI Lab", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # AI Status
        st.caption("SYSTEM STATUS")
        providers = ai.get_active_providers()
        if providers:
            for p in providers:
                st.markdown(f"🟢 **{p.title()}** Ready")
        else:
            st.warning("No AI Keys Detected")
            
        return page

# ==========================================
# 5. PAGES
# ==========================================
def page_dashboard():
    st.markdown('<div class="main-header">📊 Workspace Overview</div>', unsafe_allow_html=True)
    stats = db.get_stats()
    
    # Hero Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("📚 Knowledge Base", f"{stats['count']} Notes", delta="Active")
    c2.metric("🕒 Last Update", stats['last_active'].split()[0] if stats['last_active'] != "Never" else "N/A")
    c3.metric("💳 AI Usage Estimate", f"${stats['cost']:.3f}")

    st.markdown("### 📈 Analytics")
    # Improved Charting
    dates = [datetime.now() - timedelta(days=x) for x in range(7)][::-1]
    data = pd.DataFrame({
        "Date": dates,
        "Productivity": [12, 18, 10, 25, 20, 32, 28] # Mock data for visuals
    })
    
    fig = px.area(data, x="Date", y="Productivity", template="plotly_dark", line_shape="spline")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

def page_notes():
    st.markdown('<div class="main-header">📝 Knowledge Base</div>', unsafe_allow_html=True)
    
    # 1. Search Bar (Excellent Usability Feature)
    col_search, col_new = st.columns([4, 1])
    with col_search:
        search_q = st.text_input("🔍 Search notes...", placeholder="Type to filter titles or content...", label_visibility="collapsed")
    with col_new:
        if st.button("➕ Create", use_container_width=True, type="primary"):
            st.session_state.active_note = None
            st.rerun()

    # 2. Main Interface
    col_list, col_editor = st.columns([1, 2])
    
    # Left Column: Note List
    with col_list:
        notes = db.get_notes(search_q)
        if not notes:
            st.info("No notes found.")
        
        for note in notes:
            # Card-like styling for list items
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                title_display = f"{'📌 ' if note['is_pinned'] else ''}{note['title']}"
                if c1.button(title_display, key=f"sel_{note['id']}", use_container_width=True):
                    st.session_state.active_note = note
                
                # Pin Toggle
                if c2.button("★", key=f"pin_{note['id']}", help="Toggle Pin"):
                    db.toggle_pin(note['id'], note['is_pinned'])
                    st.rerun()

    # Right Column: Robust Editor
    with col_editor:
        if "active_note" not in st.session_state: st.session_state.active_note = None
        note = st.session_state.active_note
        
        if note is not None:
            with st.form("note_editor"):
                # Header
                st.caption(f"Editing: {note.get('updated_at', 'New Note')}")
                new_title = st.text_input("Title", value=note.get('title', ''))
                new_tags = st.text_input("Tags", value=note.get('tags', ''), placeholder="work, ideas, python")
                new_content = st.text_area("Content", value=note.get('content', ''), height=450)
                
                # Footer Actions
                c_save, c_ai, c_del = st.columns([1, 2, 1])
                
                save_clicked = c_save.form_submit_button("💾 Save Changes", type="primary")
                
                # Safe Delete (Usability Fix)
                delete_trigger = c_del.form_submit_button("🗑️ Delete")

                if save_clicked:
                    if not new_title:
                        st.error("Title is required.")
                    else:
                        db.upsert_note(note.get('id'), new_title, new_content, new_tags)
                        st.toast("✅ Note saved successfully!", icon="💾")
                        time.sleep(0.5)
                        st.rerun()
                
                if delete_trigger:
                    st.session_state.confirm_delete = note.get('id')

            # Delete Confirmation Dialog (Outside Form)
            if st.session_state.get('confirm_delete') == note.get('id'):
                st.warning("⚠️ Are you sure you want to delete this note?")
                col_y, col_n = st.columns(2)
                if col_y.button("Yes, Delete Permanently", type="primary"):
                    db.delete_note(note['id'])
                    st.session_state.active_note = None
                    st.session_state.confirm_delete = None
                    st.toast("🗑️ Note deleted.")
                    st.rerun()
                if col_n.button("Cancel"):
                    st.session_state.confirm_delete = None
                    st.rerun()
        else:
            # Empty State
            with st.container(border=True):
                st.markdown("""
                ### 👋 Welcome to the Editor
                Select a note from the left to edit, or click **Create** to start fresh.
                
                **Pro Tips:**
                - Use `markdown` for formatting
                - Pin important notes for quick access
                """)

def page_ai():
    st.markdown('<div class="main-header">🤖 AI Laboratory</div>', unsafe_allow_html=True)
    
    # Sidebar control within page
    col_chat, col_params = st.columns([3, 1])
    
    with col_params:
        with st.container(border=True):
            st.markdown("### ⚙️ Model Config")
            active_providers = ai.get_active_providers()
            if not active_providers:
                st.error("No API Keys found.")
                provider = None
            else:
                provider = st.selectbox("Provider", active_providers)
                
            st.slider("Creativity", 0.0, 1.0, 0.7, key="temp")
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()

    with col_chat:
        if "messages" not in st.session_state: st.session_state.messages = []

        # Chat Container
        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.messages:
                avatar = "👤" if msg['role'] == "user" else "🤖"
                with st.chat_message(msg['role'], avatar=avatar):
                    st.write(msg['content'])

        # Input Area
        if prompt := st.chat_input("Ask Nexus AI..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.write(prompt)
                
                with st.chat_message("assistant", avatar="🤖"):
                    if provider:
                        # Streaming UI Effect
                        response_placeholder = st.empty()
                        full_response = ""
                        for chunk in ai.stream_response(prompt, provider):
                            full_response = chunk
                            response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        st.error("Please configure an API Key in .streamlit/secrets.toml")

def page_settings():
    st.markdown('<div class="main-header">⚙️ Data & Settings</div>', unsafe_allow_html=True)
    
    st.subheader("📤 Export Workspace")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Download your entire knowledge base for backup or migration.")
        format_choice = st.selectbox("Format", ["JSON (Complete)", "Markdown (Readable)", "CSV (Table)"])
    
    with col2:
        st.write("") # Spacer
        st.write("") # Spacer
        if st.button("📦 Generate Backup", type="primary", use_container_width=True):
            with st.spinner("Packing data..."):
                notes = db.get_notes()
                timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
                
                if "JSON" in format_choice:
                    data = pd.DataFrame(notes).to_json(orient="records", indent=2)
                    mime = "application/json"
                    ext = "json"
                elif "CSV" in format_choice:
                    data = pd.DataFrame(notes).to_csv(index=False)
                    mime = "text/csv"
                    ext = "csv"
                else:
                    data = "\n\n".join([f"# {n['title']}\n{n['content']}\n---" for n in notes])
                    mime = "text/markdown"
                    ext = "md"
                
                st.download_button(
                    label=f"⬇️ Download nexus_backup_{timestamp}.{ext}",
                    data=data,
                    file_name=f"nexus_backup_{timestamp}.{ext}",
                    mime=mime,
                    use_container_width=True
                )
                st.toast("Backup ready for download!", icon="✅")

# ==========================================
# 6. MAIN ROUTER
# ==========================================
def main():
    page_selection = render_sidebar()
    
    if "Dashboard" in page_selection:
        page_dashboard()
    elif "Knowledge" in page_selection:
        page_notes()
    elif "AI" in page_selection:
        page_ai()
    elif "Settings" in page_selection:
        page_settings()

if __name__ == "__main__":
    main()
