import streamlit as st
import io
import json
import xml.etree.ElementTree as ET
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

# =============================================================================
# Text to .txt Mobile: Simplified Mobile-First Notes with Persistence
# =============================================================================

# Mobile-optimized layout
st.set_page_config(
    page_title="Mobile Notes",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"  # Better for mobile
)

# ----------------------------
# Persistence Setup
# ----------------------------
DATA_FILE = "notes_data.json"

def load_notes() -> List[Dict[str, Any]]:
    """Load notes from JSON file"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_notes(notes: List[Dict[str, Any]]) -> None:
    """Save notes to JSON file"""
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
    except Exception:
        st.error("Could not save notes")

# ----------------------------
# Helpers
# ----------------------------
def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_filename(name: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._ "
    cleaned = "".join(c for c in (name or "") if c in keep).strip()
    return cleaned or "untitled"

def ensure_ext(name: str, ext: str) -> str:
    if not ext.startswith("."):
        ext = "." + ext
    return name if name.lower().endswith(ext.lower()) else f"{name}{ext}"

def format_time_ago(timestamp: str) -> str:
    """Format timestamp for mobile display"""
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 365:
            return f"{diff.days // 365}y ago"
        elif diff.days > 30:
            return f"{diff.days // 30}mo ago"
        elif diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "Just now"
    except:
        return timestamp

# ----------------------------
# XML Logic
# ----------------------------
def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def note_to_xml(n: Dict[str, Any]) -> str:
    root = ET.Element("note", attrib={"id": str(n.get("id", ""))})
    ET.SubElement(root, "title").text = n.get("title", "")
    ET.SubElement(root, "created_at").text = n.get("created_at", "")
    ET.SubElement(root, "body").text = n.get("body", "")
    _indent_xml(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")

# ----------------------------
# Session State Management
# ----------------------------
if "notes" not in st.session_state:
    st.session_state.notes = load_notes()

if "active_id" not in st.session_state:
    st.session_state.active_id = None

if "show_delete_confirm" not in st.session_state:
    st.session_state.show_delete_confirm = None

def get_note(nid):
    return next((n for n in st.session_state.notes if n["id"] == nid), None)

def create_note():
    new_id = (max([n["id"] for n in st.session_state.notes], default=0) + 1)
    new_note = {
        "id": new_id,
        "title": f"Note {new_id}",
        "body": "",
        "created_at": now_stamp(),
        "updated_at": now_stamp()
    }
    st.session_state.notes.append(new_note)
    st.session_state.active_id = new_id
    save_notes(st.session_state.notes)
    return new_note

def update_note(nid: int, title: str, body: str):
    for note in st.session_state.notes:
        if note["id"] == nid:
            if note["title"] != title or note["body"] != body:
                note["title"] = title
                note["body"] = body
                note["updated_at"] = now_stamp()
                save_notes(st.session_state.notes)
            break

def delete_note(nid):
    st.session_state.notes = [n for n in st.session_state.notes if n["id"] != nid]
    st.session_state.active_id = None
    st.session_state.show_delete_confirm = None
    save_notes(st.session_state.notes)

# =============================================================================
# Mobile UI Layout
# =============================================================================

# Custom CSS for better mobile experience
st.markdown("""
<style>
    /* Mobile-friendly buttons and inputs */
    .stButton > button {
        border-radius: 10px;
        height: 3em;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        font-size: 16px !important;
    }
    
    /* Better touch targets */
    .note-item {
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 12px;
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        transition: all 0.2s;
    }
    
    .note-item:hover {
        background: #e9ecef;
        cursor: pointer;
    }
    
    .note-item.active {
        background: #e3f2fd;
        border-color: #2196f3;
    }
    
    /* Compact headers */
    .compact-header {
        padding: 0.5rem 0;
    }
    
    /* Hide streamlit branding for cleaner mobile view */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stTextArea > div > div > textarea {
            min-height: 200px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header with search
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.markdown("<h1 class='compact-header'>📝 Mobile Notes</h1>", unsafe_allow_html=True)
with col2:
    if st.button("➕ New", type="primary", use_container_width=True):
        create_note()
        st.rerun()

# Search functionality
search_term = st.text_input("", placeholder="🔍 Search notes...", label_visibility="collapsed")

st.markdown("---")

# --- Notes List ---
if st.session_state.notes:
    # Filter notes based on search
    filtered_notes = st.session_state.notes
    if search_term:
        filtered_notes = [
            n for n in st.session_state.notes
            if search_term.lower() in n["title"].lower() or search_term.lower() in n["body"].lower()
        ]
    
    # Sort by updated time (newest first)
    filtered_notes = sorted(filtered_notes, key=lambda x: x.get("updated_at", x["created_at"]), reverse=True)
    
    # Mobile-friendly notes list
    if not search_term or filtered_notes:
        st.markdown("### Your Notes")
        
        for note in filtered_notes:
            is_active = note["id"] == st.session_state.active_id
            active_class = "active" if is_active else ""
            
            # Extract preview text
            preview = note["body"][:80] + "..." if len(note["body"]) > 80 else note["body"]
            time_ago = format_time_ago(note.get("updated_at", note["created_at"]))
            
            col_a, col_b = st.columns([0.8, 0.2])
            with col_a:
                # Clickable note item
                if st.button(
                    f"**{note['title']}**  \n"
                    f"{preview}  \n"
                    f"<small>{time_ago}</small>",
                    key=f"note_{note['id']}",
                    use_container_width=True,
                    help="Tap to edit"
                ):
                    st.session_state.active_id = note["id"]
                    st.rerun()
            
            with col_b:
                # Quick delete button
                if st.button("🗑️", key=f"delete_{note['id']}", help="Delete note"):
                    st.session_state.show_delete_confirm = note["id"]
                    st.rerun()
    
    if search_term and not filtered_notes:
        st.info("No notes found matching your search.")
else:
    st.info("👋 Welcome! Tap '➕ New' to create your first note.")

st.markdown("---")

# --- Editor Section ---
active_note = get_note(st.session_state.active_id) if st.session_state.active_id else None

if active_note:
    # Editor Header
    col_title, col_close = st.columns([0.9, 0.1])
    with col_title:
        st.markdown(f"### ✏️ Editing: {active_note['title']}")
    with col_close:
        if st.button("✕", help="Close editor"):
            st.session_state.active_id = None
            st.rerun()
    
    # Title Input
    new_title = st.text_input(
        "Title",
        value=active_note["title"],
        label_visibility="collapsed",
        placeholder="Note title...",
        key=f"title_{active_note['id']}"
    )
    
    # Body Input (mobile-optimized)
    new_body = st.text_area(
        "Content",
        value=active_note["body"],
        height=300,
        label_visibility="collapsed",
        placeholder="Start typing your note here...",
        key=f"body_{active_note['id']}"
    )
    
    # Auto-save indicator
    if new_title != active_note["title"] or new_body != active_note["body"]:
        update_note(active_note["id"], new_title, new_body)
        st.caption("💾 Auto-saved")
    
    # Mobile Action Bar
    st.markdown("---")
    col_export, col_delete = st.columns(2)
    
    with col_export:
        with st.popover("📤 Export", use_container_width=True):
            fmt = st.selectbox("Format", ["txt", "xml", "json", "csv"], key="export_format")
            
            # Prepare export data
            if fmt == "txt":
                file_data = f"{active_note['title']}\n{'-'*20}\n{active_note['body']}"
                mime_type = "text/plain"
                ext = ".txt"
            elif fmt == "xml":
                file_data = note_to_xml(active_note)
                mime_type = "application/xml"
                ext = ".xml"
            elif fmt == "json":
                file_data = json.dumps(active_note, indent=2)
                mime_type = "application/json"
                ext = ".json"
            else:  # csv
                import csv
                import io
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["Title", "Content", "Created", "Updated"])
                writer.writerow([
                    active_note["title"],
                    active_note["body"],
                    active_note["created_at"],
                    active_note.get("updated_at", "")
                ])
                file_data = output.getvalue()
                mime_type = "text/csv"
                ext = ".csv"
            
            st.download_button(
                label=f"Download as {fmt.upper()}",
                data=file_data,
                file_name=safe_filename(ensure_ext(active_note["title"], ext)),
                mime=mime_type,
                use_container_width=True
            )
    
    with col_delete:
        if st.session_state.show_delete_confirm == active_note["id"]:
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("✅ Confirm", type="primary", use_container_width=True):
                    delete_note(active_note["id"])
                    st.rerun()
            with col_cancel:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_delete_confirm = None
                    st.rerun()
        else:
            if st.button("🗑️ Delete", type="secondary", use_container_width=True):
                st.session_state.show_delete_confirm = active_note["id"]
                st.rerun()
    
    # Note info
    st.caption(f"Created: {active_note['created_at']} • Last updated: {active_note.get('updated_at', 'Never')}")

# Delete confirmation modal
if st.session_state.show_delete_confirm and not active_note:
    st.warning("Are you sure you want to delete this note?")
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("Yes, delete it", type="primary"):
            delete_note(st.session_state.show_delete_confirm)
            st.rerun()
    with col_no:
        if st.button("Cancel"):
            st.session_state.show_delete_confirm = None
            st.rerun()

# Stats footer
if st.session_state.notes:
    total_notes = len(st.session_state.notes)
    total_chars = sum(len(n["body"]) for n in st.session_state.notes)
    st.caption(f"📊 {total_notes} notes • {total_chars:,} characters")

# Backup reminder
if st.session_state.notes:
    with st.expander("💾 Backup Options"):
        if st.button("Export All Notes as JSON", use_container_width=True):
            backup_data = json.dumps(st.session_state.notes, indent=2)
            st.download_button(
                label="Download Backup",
                data=backup_data,
                file_name=f"notes_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        st.caption("Your notes are automatically saved locally. Download a backup for safety.")
