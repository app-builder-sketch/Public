import streamlit as st
import io
import json
import xml.etree.ElementTree as ET
import os
import zipfile
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path
import re
import plotly.express as px
import collections

# =============================================================================
# Desktop Notes Pro - Full-Featured Note-Taking & File Conversion
# =============================================================================

st.set_page_config(
    page_title="Notes Pro",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Persistence Setup
# ----------------------------
DATA_FILE = "notes_data.json"
SETTINGS_FILE = "settings.json"
TEMPLATES_FILE = "templates.json"

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

def load_settings() -> Dict[str, Any]:
    """Load user settings"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {
        "theme": "light",
        "layout": "desktop",
        "auto_backup": False,
        "backup_interval": 24
    }

def save_settings(settings: Dict[str, Any]) -> None:
    """Save user settings"""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass

def load_templates() -> List[Dict[str, Any]]:
    """Load note templates"""
    try:
        if os.path.exists(TEMPLATES_FILE):
            with open(TEMPLATES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return [
        {
            "id": 1,
            "name": "Meeting Notes",
            "content": "# Meeting Notes\n\n**Date:** {date}\n**Attendees:** \n\n## Agenda\n- \n\n## Discussion\n\n\n## Action Items\n- [ ] \n\n## Next Steps\n"
        },
        {
            "id": 2,
            "name": "To-Do List",
            "content": "# To-Do List - {date}\n\n## Priority\n- [ ] \n\n## Today\n- [ ] \n\n## This Week\n- [ ] \n\n## Someday\n- [ ] "
        },
        {
            "id": 3,
            "name": "Daily Journal",
            "content": "# Journal Entry - {date}\n\n## Morning Thoughts\n\n\n## Today's Highlights\n\n\n## Grateful For\n- \n\n## Tomorrow's Goals\n- "
        },
        {
            "id": 4,
            "name": "Project Plan",
            "content": "# Project: [Name]\n\n**Start Date:** {date}\n**Status:** Planning\n\n## Objectives\n\n\n## Milestones\n1. \n\n## Resources Needed\n- \n\n## Risks & Mitigation\n"
        }
    ]

def save_templates(templates: List[Dict[str, Any]]) -> None:
    """Save templates"""
    try:
        with open(TEMPLATES_FILE, 'w', encoding='utf-8') as f:
            json.dump(templates, f, indent=2)
    except Exception:
        pass

# ----------------------------
# File Conversion Helpers
# ----------------------------
def extract_zip(uploaded_file):
    """Extract ZIP and return list of files with contents"""
    extracted_files = []
    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
            for filename in z.namelist():
                if not filename.endswith('/'):
                    try:
                        content = z.read(filename)
                        try:
                            text_content = content.decode('utf-8')
                            extracted_files.append({
                                'name': filename,
                                'content': text_content,
                                'type': 'text',
                                'size': len(content)
                            })
                        except UnicodeDecodeError:
                            extracted_files.append({
                                'name': filename,
                                'content': content,
                                'type': 'binary',
                                'size': len(content)
                            })
                    except Exception as e:
                        st.warning(f"Could not extract {filename}: {str(e)}")
    except Exception as e:
        st.error(f"Error extracting ZIP: {str(e)}")
    return extracted_files

def convert_to_txt(content: str, source_format: str) -> str:
    """Convert various formats to plain text"""
    if source_format == 'json':
        try:
            data = json.loads(content)
            return json.dumps(data, indent=2)
        except:
            return content
    elif source_format == 'xml':
        try:
            root = ET.fromstring(content)
            return ET.tostring(root, encoding='unicode', method='text')
        except:
            return content
    elif source_format == 'csv':
        try:
            df = pd.read_csv(io.StringIO(content))
            return df.to_string()
        except:
            return content
    return content

def convert_to_xml(title: str, content: str) -> str:
    """Convert text to XML format"""
    root = ET.Element("document")
    ET.SubElement(root, "title").text = title
    ET.SubElement(root, "content").text = content
    ET.SubElement(root, "created").text = datetime.now().isoformat()
    _indent_xml(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)

def convert_to_json(title: str, content: str) -> str:
    """Convert text to JSON format"""
    data = {
        "title": title,
        "content": content,
        "created": datetime.now().isoformat()
    }
    return json.dumps(data, indent=2)

def convert_to_csv(title: str, content: str) -> str:
    """Convert text to CSV format"""
    df = pd.DataFrame([{
        "Title": title,
        "Content": content,
        "Created": datetime.now().isoformat()
    }])
    return df.to_csv(index=False)

def convert_to_py(content: str, as_docstring: bool = True) -> str:
    """Convert text to Python file format"""
    if as_docstring:
        return f'"""\n{content}\n"""\n\n# Generated from note\n# Created: {datetime.now().isoformat()}\n'
    else:
        lines = content.split('\n')
        return '\n'.join(f'# {line}' for line in lines)

# ----------------------------
# Security Helpers
# ----------------------------
def hash_password(password: str) -> str:
    """Hash password for storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password: str, hashed: str) -> bool:
    """Check if password matches hash"""
    return hash_password(password) == hashed

# ----------------------------
# Helper Functions
# ----------------------------
def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_filename(name: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._"
    cleaned = "".join(c for c in (name or "") if c in keep).strip()
    return cleaned or "untitled"

def ensure_ext(name: str, ext: str) -> str:
    if not ext.startswith("."):
        ext = "." + ext
    return name if name.lower().endswith(ext.lower()) else f"{name}{ext}"

def format_time_ago(timestamp: str) -> str:
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
    if n.get("tags"):
        tags_elem = ET.SubElement(root, "tags")
        for tag in n["tags"]:
            ET.SubElement(tags_elem, "tag").text = tag
    _indent_xml(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)

def extract_tags(text: str) -> List[str]:
    """Extract #tags from text"""
    return list(set(re.findall(r'#(\w+)', text)))

def render_markdown_preview(text: str) -> str:
    """Simple markdown rendering"""
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Headers
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    # Checkboxes
    text = re.sub(r'- \[ \]', '☐', text)
    text = re.sub(r'- \[x\]', '☑', text)
    return text

# ----------------------------
# Session State Management
# ----------------------------
if "notes" not in st.session_state:
    st.session_state.notes = load_notes()

if "settings" not in st.session_state:
    st.session_state.settings = load_settings()

if "templates" not in st.session_state:
    st.session_state.templates = load_templates()

if "active_id" not in st.session_state:
    st.session_state.active_id = None

if "show_delete_confirm" not in st.session_state:
    st.session_state.show_delete_confirm = None

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "notes"

if "extracted_files" not in st.session_state:
    st.session_state.extracted_files = []

if "search_term" not in st.session_state:
    st.session_state.search_term = ""

if "selected_tags" not in st.session_state:
    st.session_state.selected_tags = []

if "show_archived" not in st.session_state:
    st.session_state.show_archived = False

if "markdown_preview" not in st.session_state:
    st.session_state.markdown_preview = False

if "locked_notes" not in st.session_state:
    st.session_state.locked_notes = set()

if "search_history" not in st.session_state:
    st.session_state.search_history = []

def get_note(nid):
    return next((n for n in st.session_state.notes if n["id"] == nid), None)

def create_note(template_id=None):
    new_id = (max([n["id"] for n in st.session_state.notes], default=0) + 1)
    
    # Apply template if specified
    body = ""
    title = f"Note {new_id}"
    if template_id:
        template = next((t for t in st.session_state.templates if t["id"] == template_id), None)
        if template:
            body = template["content"].replace("{date}", datetime.now().strftime("%Y-%m-%d"))
            title = template["name"]
    
    new_note = {
        "id": new_id,
        "title": title,
        "body": body,
        "created_at": now_stamp(),
        "updated_at": now_stamp(),
        "tags": [],
        "pinned": False,
        "favorite": False,
        "archived": False,
        "locked": False,
        "password_hash": None,
        "attachments": [],
        "linked_notes": [],
        "version_history": [],
        "category": "General"
    }
    st.session_state.notes.append(new_note)
    st.session_state.active_id = new_id
    save_notes(st.session_state.notes)
    return new_note

def update_note(nid: int, title: str, body: str):
    for note in st.session_state.notes:
        if note["id"] == nid:
            if note["title"] != title or note["body"] != body:
                # Save to version history
                if "version_history" not in note:
                    note["version_history"] = []
                note["version_history"].append({
                    "title": note["title"],
                    "body": note["body"],
                    "timestamp": note["updated_at"]
                })
                # Keep only last 10 versions
                note["version_history"] = note["version_history"][-10:]
                
                note["title"] = title
                note["body"] = body
                note["updated_at"] = now_stamp()
                
                # Auto-extract tags
                note["tags"] = extract_tags(body)
                
                save_notes(st.session_state.notes)
            break

def delete_note(nid):
    st.session_state.notes = [n for n in st.session_state.notes if n["id"] != nid]
    st.session_state.active_id = None
    st.session_state.show_delete_confirm = None
    save_notes(st.session_state.notes)

def toggle_pin(nid):
    note = get_note(nid)
    if note:
        note["pinned"] = not note.get("pinned", False)
        save_notes(st.session_state.notes)

def toggle_favorite(nid):
    note = get_note(nid)
    if note:
        note["favorite"] = not note.get("favorite", False)
        save_notes(st.session_state.notes)

def toggle_archive(nid):
    note = get_note(nid)
    if note:
        note["archived"] = not note.get("archived", False)
        save_notes(st.session_state.notes)

def duplicate_note(nid):
    note = get_note(nid)
    if note:
        new_id = (max([n["id"] for n in st.session_state.notes], default=0) + 1)
        new_note = note.copy()
        new_note["id"] = new_id
        new_note["title"] = f"{note['title']} (Copy)"
        new_note["created_at"] = now_stamp()
        new_note["updated_at"] = now_stamp()
        new_note["version_history"] = []
        st.session_state.notes.append(new_note)
        save_notes(st.session_state.notes)
        return new_id

# =============================================================================
# Dynamic Styling Based on Theme
# =============================================================================
theme = st.session_state.settings.get("theme", "light")
layout = st.session_state.settings.get("layout", "desktop")

if theme == "dark":
    bg_color = "#1e1e1e"
    text_color = "#ffffff"
    card_bg = "#2d2d2d"
    border_color = "#404040"
    hover_bg = "#3d3d3d"
    chart_template = "plotly_dark"
else:
    bg_color = "#ffffff"
    text_color = "#000000"
    card_bg = "#f8f9fa"
    border_color = "#e9ecef"
    hover_bg = "#e9ecef"
    chart_template = "plotly_white"

# Mobile vs Desktop CSS
if layout == "mobile":
    button_height = "3em"
    font_size = "16px"
    padding = "12px 16px"
    border_radius = "12px"
else:
    button_height = "2.5em"
    font_size = "14px"
    padding = "8px 12px"
    border_radius = "8px"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    
    .stButton > button {{
        border-radius: {border_radius};
        height: {button_height};
        font-size: {font_size};
    }}
    
    .note-card {{
        padding: {padding};
        margin: 8px 0;
        border-radius: {border_radius};
        background: {card_bg};
        border: 1px solid {border_color};
        transition: all 0.2s;
    }}
    
    .note-card:hover {{
        background: {hover_bg};
        cursor: pointer;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    
    .note-card.pinned {{
        border-left: 4px solid #ff9800;
    }}
    
    .note-card.favorite {{
        border-left: 4px solid #f44336;
    }}
    
    .tag {{
        display: inline-block;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 12px;
        background: #2196f3;
        color: white;
        font-size: 12px;
    }}
    
    .stat-box {{
        padding: 15px;
        border-radius: 8px;
        background: {card_bg};
        border: 1px solid {border_color};
        text-align: center;
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - Navigation & Settings
# =============================================================================
with st.sidebar:
    st.title("📝 Notes Pro")
    
    # --- LIVE CLOCK INSERTION ---
    st.markdown(
        f"""
        <div style="
            padding: 10px;
            border-radius: 8px;
            background: {card_bg};
            border: 1px solid {border_color};
            text-align: center;
            margin-bottom: 20px;
            color: {text_color};
            font-family: monospace;
            font-size: 1.1em;
        ">
            <div id="clock_time" style="font-weight: bold;"></div>
            <div id="clock_date" style="font-size: 0.8em; opacity: 0.8;"></div>
        </div>
        <script>
            function updateClock() {{
                const now = new Date();
                const timeStr = now.toLocaleTimeString([], {{hour: '2-digit', minute: '2-digit', second: '2-digit'}});
                const dateStr = now.toLocaleDateString([], {{weekday: 'short', year: 'numeric', month: 'short', day: 'numeric'}});
                
                const tDiv = document.getElementById('clock_time');
                const dDiv = document.getElementById('clock_date');
                
                if (tDiv) tDiv.innerHTML = timeStr;
                if (dDiv) dDiv.innerHTML = dateStr;
            }}
            setInterval(updateClock, 1000);
            updateClock();
        </script>
        """,
        unsafe_allow_html=True
    )
    # --- END CLOCK INSERTION ---
    
    # View Mode Selection
    st.markdown("### 📂 Views")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📝", use_container_width=True, help="Notes"):
            st.session_state.view_mode = "notes"
            st.rerun()
    with col2:
        if st.button("🔄", use_container_width=True, help="Converter"):
            st.session_state.view_mode = "converter"
            st.rerun()
    with col3:
        if st.button("⚙️", use_container_width=True, help="Settings"):
            st.session_state.view_mode = "settings"
            st.rerun()
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    # New Note with Template
    with st.popover("➕ New Note", use_container_width=True):
        st.markdown("**Choose Template:**")
        if st.button("📄 Blank Note", use_container_width=True):
            create_note()
            st.rerun()
        
        for template in st.session_state.templates:
            if st.button(f"📋 {template['name']}", key=f"template_{template['id']}", use_container_width=True):
                create_note(template["id"])
                st.rerun()
    
    # Search
    st.markdown("### 🔍 Search & Filter")
    search_input = st.text_input("Search notes", value=st.session_state.search_term, placeholder="Search titles and content...")
    if search_input != st.session_state.search_term:
        st.session_state.search_term = search_input
        if search_input and search_input not in st.session_state.search_history:
            st.session_state.search_history.insert(0, search_input)
            st.session_state.search_history = st.session_state.search_history[:10]
    
    # Recent searches
    if st.session_state.search_history:
        st.markdown("**Recent Searches:**")
        for i, search in enumerate(st.session_state.search_history[:5]):
            if st.button(f"🔍 {search}", key=f"recent_search_{i}", use_container_width=True):
                st.session_state.search_term = search
                st.rerun()
    
    # Tag Filter
    all_tags = set()
    for note in st.session_state.notes:
        all_tags.update(note.get("tags", []))
    
    if all_tags:
        st.markdown("**Filter by Tags:**")
        selected_tags = st.multiselect("", sorted(all_tags), default=st.session_state.selected_tags, label_visibility="collapsed")
        st.session_state.selected_tags = selected_tags
    
    # Category Filter
    categories = set(note.get("category", "General") for note in st.session_state.notes)
    if len(categories) > 1:
        selected_category = st.selectbox("Category", ["All"] + sorted(categories))
    else:
        selected_category = "All"
    
    # Date Range Filter
    with st.expander("📅 Date Range"):
        date_filter = st.radio("Show notes from:", ["All Time", "Today", "This Week", "This Month", "Custom Range"])
        if date_filter == "Custom Range":
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
    
    # View Options
    st.markdown("---")
    st.markdown("### 👁️ View Options")
    show_archived = st.checkbox("Show Archived", value=st.session_state.show_archived)
    st.session_state.show_archived = show_archived
    
    show_favorites_only = st.checkbox("Favorites Only")
    show_pinned_only = st.checkbox("Pinned Only")
    
    # Sort Options
    sort_by = st.selectbox("Sort by", ["Updated (Newest)", "Updated (Oldest)", "Created (Newest)", "Created (Oldest)", "Title (A-Z)", "Title (Z-A)"])
    
    st.markdown("---")
    
    # Stats
    st.markdown("### 📊 Statistics")
    total_notes = len([n for n in st.session_state.notes if not n.get("archived", False)])
    archived_notes = len([n for n in st.session_state.notes if n.get("archived", False)])
    favorite_notes = len([n for n in st.session_state.notes if n.get("favorite", False)])
    total_chars = sum(len(n["body"]) for n in st.session_state.notes)
    
    st.metric("Total Notes", total_notes)
    st.metric("Favorites", favorite_notes)
    st.metric("Archived", archived_notes)
    st.metric("Total Characters", f"{total_chars:,}")

# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

# =============================================================================
# NOTES MODE
# =============================================================================
if st.session_state.view_mode == "notes":
    col_list, col_editor = st.columns([1, 2])
    
    # --- LEFT: Notes List ---
    with col_list:
        st.markdown("### 📋 Notes")
        
        # Filter notes
        filtered_notes = st.session_state.notes
        
        # Apply filters
        if not show_archived:
            filtered_notes = [n for n in filtered_notes if not n.get("archived", False)]
        
        if show_favorites_only:
            filtered_notes = [n for n in filtered_notes if n.get("favorite", False)]
        
        if show_pinned_only:
            filtered_notes = [n for n in filtered_notes if n.get("pinned", False)]
        
        if st.session_state.search_term:
            search_lower = st.session_state.search_term.lower()
            filtered_notes = [
                n for n in filtered_notes
                if search_lower in n["title"].lower() or search_lower in n["body"].lower()
            ]
        
        if st.session_state.selected_tags:
            filtered_notes = [
                n for n in filtered_notes
                if any(tag in n.get("tags", []) for tag in st.session_state.selected_tags)
            ]
        
        if selected_category != "All":
            filtered_notes = [n for n in filtered_notes if n.get("category", "General") == selected_category]
        
        # Date filter
        if date_filter != "All Time":
            now = datetime.now()
            if date_filter == "Today":
                start = now.replace(hour=0, minute=0, second=0)
                filtered_notes = [n for n in filtered_notes if datetime.strptime(n["updated_at"], "%Y-%m-%d %H:%M:%S") >= start]
            elif date_filter == "This Week":
                start = now - timedelta(days=now.weekday())
                filtered_notes = [n for n in filtered_notes if datetime.strptime(n["updated_at"], "%Y-%m-%d %H:%M:%S") >= start]
            elif date_filter == "This Month":
                start = now.replace(day=1, hour=0, minute=0, second=0)
                filtered_notes = [n for n in filtered_notes if datetime.strptime(n["updated_at"], "%Y-%m-%d %H:%M:%S") >= start]
        
        # Sort notes
        if sort_by == "Updated (Newest)":
            filtered_notes = sorted(filtered_notes, key=lambda x: x.get("updated_at", x["created_at"]), reverse=True)
        elif sort_by == "Updated (Oldest)":
            filtered_notes = sorted(filtered_notes, key=lambda x: x.get("updated_at", x["created_at"]))
        elif sort_by == "Created (Newest)":
            filtered_notes = sorted(filtered_notes, key=lambda x: x["created_at"], reverse=True)
        elif sort_by == "Created (Oldest)":
            filtered_notes = sorted(filtered_notes, key=lambda x: x["created_at"])
        elif sort_by == "Title (A-Z)":
            filtered_notes = sorted(filtered_notes, key=lambda x: x["title"].lower())
        else:  # Title (Z-A)
            filtered_notes = sorted(filtered_notes, key=lambda x: x["title"].lower(), reverse=True)
        
        # Separate pinned notes
        pinned_notes = [n for n in filtered_notes if n.get("pinned", False)]
        unpinned_notes = [n for n in filtered_notes if not n.get("pinned", False)]
        
        # Display pinned notes first
        if pinned_notes:
            st.markdown("**📌 Pinned**")
            for note in pinned_notes:
                display_note_card(note)
            st.markdown("---")
        
        # Display other notes
        if unpinned_notes:
            for note in unpinned_notes:
                display_note_card(note)
        
        if not filtered_notes:
            st.info("No notes found. Create one to get started!")
    
    # --- RIGHT: Editor ---
    with col_editor:
        active_note = get_note(st.session_state.active_id) if st.session_state.active_id else None
        
        if active_note:
            # Check if note is locked
            if active_note.get("locked", False) and active_note["id"] in st.session_state.locked_notes:
                pass  # Note is unlocked in this session
            elif active_note.get("locked", False):
                st.markdown(f"### 🔒 Locked Note: {active_note['title']}")
                password = st.text_input("Enter password to unlock", type="password", key="unlock_password")
                col_unlock, col_back = st.columns([1, 1])
                with col_unlock:
                    if st.button("🔓 Unlock", type="primary", use_container_width=True):
                        if active_note.get("password_hash") and check_password(password, active_note["password_hash"]):
                            st.session_state.locked_notes.add(active_note["id"])
                            st.success("Note unlocked!")
                            st.rerun()
                        else:
                            st.error("Incorrect password")
                with col_back:
                    if st.button("← Back", use_container_width=True):
                        st.session_state.active_id = None
                        st.rerun()
                return
            
            # Editor Header
            col_title, col_actions = st.columns([3, 1])
            with col_title:
                st.markdown(f"### ✏️ {active_note['title']}")
            with col_actions:
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    pin_icon = "📌" if active_note.get("pinned") else "📍"
                    if st.button(pin_icon, help="Pin/Unpin", key="pin_note"):
                        toggle_pin(active_note["id"])
                        st.rerun()
                with col_b:
                    fav_icon = "⭐" if active_note.get("favorite") else "☆"
                    if st.button(fav_icon, help="Favorite", key="fav_note"):
                        toggle_favorite(active_note["id"])
                        st.rerun()
                with col_c:
                    if st.button("📋", help="Duplicate", key="dup_note"):
                        new_id = duplicate_note(active_note["id"])
                        st.session_state.active_id = new_id
                        st.success("Note duplicated!")
                        st.rerun()
                with col_d:
                    if st.button("✕", help="Close", key="close_editor"):
                        st.session_state.active_id = None
                        st.rerun()
            
            # Title and Category
            col_title_input, col_category = st.columns([3, 1])
            with col_title_input:
                new_title = st.text_input(
                    "Title",
                    value=active_note["title"],
                    label_visibility="collapsed",
                    placeholder="Note title...",
                    key=f"title_{active_note['id']}"
                )
            with col_category:
                categories = sorted(set(n.get("category", "General") for n in st.session_state.notes))
                if "General" not in categories:
                    categories.insert(0, "General")
                new_category = st.selectbox("Category", categories, index=categories.index(active_note.get("category", "General")), key=f"cat_{active_note['id']}")
            
            # Editor Toolbar
            col_preview, col_lock, col_archive = st.columns([1, 1, 1])
            with col_preview:
                preview_mode = st.checkbox("📖 Markdown Preview", value=st.session_state.markdown_preview, key="preview_toggle")
                st.session_state.markdown_preview = preview_mode
            with col_lock:
                lock_status = active_note.get("locked", False)
                if st.button(f"{'🔓 Unlock' if lock_status else '🔒 Lock'}", key="lock_toggle"):
                    if not lock_status:
                        # Lock the note
                        with st.form("lock_form"):
                            password = st.text_input("Set password", type="password")
                            password_confirm = st.text_input("Confirm password", type="password")
                            if st.form_submit_button("Lock Note"):
                                if password and password == password_confirm:
                                    active_note["locked"] = True
                                    active_note["password_hash"] = hash_password(password)
                                    save_notes(st.session_state.notes)
                                    st.success("Note locked!")
                                    st.rerun()
                                else:
                                    st.error("Passwords don't match")
                    else:
                        # Unlock the note
                        active_note["locked"] = False
                        active_note["password_hash"] = None
                        if active_note["id"] in st.session_state.locked_notes:
                            st.session_state.locked_notes.remove(active_note["id"])
                        save_notes(st.session_state.notes)
                        st.success("Note unlocked!")
                        st.rerun()
            with col_archive:
                archive_status = active_note.get("archived", False)
                if st.button(f"{'📤 Unarchive' if archive_status else '📥 Archive'}", key="archive_toggle"):
                    toggle_archive(active_note["id"])
                    st.rerun()
            
            # Body Editor
            if preview_mode:
                col_edit, col_prev = st.columns(2)
                with col_edit:
                    st.markdown("**Edit:**")
                    new_body = st.text_area(
                        "Content",
                        value=active_note["body"],
                        height=400,
                        label_visibility="collapsed",
                        placeholder="Start typing...",
                        key=f"body_{active_note['id']}"
                    )
                with col_prev:
                    st.markdown("**Preview:**")
                    preview_html = render_markdown_preview(new_body)
                    st.markdown(f'<div style="border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; min-height: 400px;">{preview_html}</div>', unsafe_allow_html=True)
            else:
                new_body = st.text_area(
                    "Content",
                    value=active_note["body"],
                    height=400,
                    label_visibility="collapsed",
                    placeholder="Start typing...",
                    key=f"body_{active_note['id']}"
                )
            
            # Auto-save indicator
            if new_title != active_note["title"] or new_body != active_note["body"] or new_category != active_note.get("category", "General"):
                active_note["category"] = new_category
                update_note(active_note["id"], new_title, new_body)
                st.caption("💾 Auto-saved")
            
            # Tags display
            if active_note.get("tags"):
                st.markdown("**Tags:** " + " ".join([f'<span class="tag">#{tag}</span>' for tag in active_note["tags"]]), unsafe_allow_html=True)
            
            # Linked Notes
            if active_note.get("linked_notes"):
                st.markdown("**🔗 Linked Notes:**")
                for linked_id in active_note["linked_notes"]:
                    linked = get_note(linked_id)
                    if linked:
                        if st.button(f"→ {linked['title']}", key=f"link_{linked_id}"):
                            st.session_state.active_id = linked_id
                            st.rerun()
            
            # Attachments
            if active_note.get("attachments"):
                st.markdown("**📎 Attachments:**")
                for i, attachment in enumerate(active_note["attachments"]):
                    col_name, col_download = st.columns([3, 1])
                    with col_name:
                        st.text(attachment["name"])
                    with col_download:
                        if attachment["type"] == "text":
                            st.download_button(
                                "⬇️",
                                data=attachment["content"],
                                file_name=attachment["name"],
                                key=f"attach_dl_{i}"
                            )
            
            st.markdown("---")
            
            # Action Bar
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Export
                with st.popover("📤 Export", use_container_width=True):
                    fmt = st.selectbox("Format", ["txt", "xml", "json", "csv", "py", "md"], key="export_format")
                    
                    if fmt == "txt":
                        file_data = f"{active_note['title']}\n{'-'*40}\n{active_note['body']}"
                        mime_type = "text/plain"
                    elif fmt == "xml":
                        file_data = note_to_xml(active_note)
                        mime_type = "application/xml"
                    elif fmt == "json":
                        file_data = json.dumps(active_note, indent=2)
                        mime_type = "application/json"
                    elif fmt == "py":
                        file_data = convert_to_py(active_note['body'])
                        mime_type = "text/x-python"
                    elif fmt == "md":
                        file_data = f"# {active_note['title']}\n\n{active_note['body']}\n\n---\n*Tags: {', '.join(active_note.get('tags', []))}\nCreated: {active_note['created_at']}*"
                        mime_type = "text/markdown"
                    else:  # csv
                        file_data = convert_to_csv(active_note["title"], active_note["body"])
                        mime_type = "text/csv"
                    
                    st.download_button(
                        label=f"⬇️ Download as {fmt.upper()}",
                        data=file_data,
                        file_name=safe_filename(ensure_ext(active_note["title"], f".{fmt}")),
                        mime=mime_type,
                        use_container_width=True
                    )
            
            with col2:
                # Link to another note
                with st.popover("🔗 Link Note", use_container_width=True):
                    other_notes = [n for n in st.session_state.notes if n["id"] != active_note["id"]]
                    if other_notes:
                        link_to = st.selectbox("Link to:", [n["title"] for n in other_notes], key="link_select")
                        if st.button("Add Link", use_container_width=True):
                            linked_note = next(n for n in other_notes if n["title"] == link_to)
                            if "linked_notes" not in active_note:
                                active_note["linked_notes"] = []
                            if linked_note["id"] not in active_note["linked_notes"]:
                                active_note["linked_notes"].append(linked_note["id"])
                                save_notes(st.session_state.notes)
                                st.success(f"Linked to {link_to}")
                                st.rerun()
                    else:
                        st.info("No other notes to link")
            
            with col3:
                # Version History
                with st.popover("📜 History", use_container_width=True):
                    if active_note.get("version_history"):
                        st.markdown("**Previous Versions:**")
                        for i, version in enumerate(reversed(active_note["version_history"])):
                            with st.expander(f"v{len(active_note['version_history'])-i} - {version['timestamp']}"):
                                st.text_area("Title", value=version["title"], disabled=True, key=f"hist_title_{i}")
                                st.text_area("Content", value=version["body"], height=150, disabled=True, key=f"hist_body_{i}")
                                if st.button(f"Restore This Version", key=f"restore_{i}"):
                                    active_note["title"] = version["title"]
                                    active_note["body"] = version["body"]
                                    active_note["updated_at"] = now_stamp()
                                    save_notes(st.session_state.notes)
                                    st.success("Version restored!")
                                    st.rerun()
                    else:
                        st.info("No version history yet")
            
            with col4:
                # Delete
                if st.button("🗑️ Delete", type="secondary", use_container_width=True):
                    st.session_state.show_delete_confirm = active_note["id"]
                    st.rerun()
            
            # Delete confirmation
            if st.session_state.show_delete_confirm == active_note["id"]:
                st.warning("⚠️ Are you sure you want to delete this note?")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅ Yes, Delete", type="primary", use_container_width=True):
                        delete_note(active_note["id"])
                        st.rerun()
                with col_no:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state.show_delete_confirm = None
                        st.rerun()
            
            # Note metadata
            st.caption(f"**Created:** {active_note['created_at']} | **Updated:** {active_note.get('updated_at', 'Never')} | **Category:** {active_note.get('category', 'General')}")
        
        else:
            st.info("👈 Select a note from the list to start editing")
            
            # Quick stats dashboard
            st.markdown("### 📊 Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="stat-box"><h2>📝</h2><p>Total Notes</p><h3>' + str(len([n for n in st.session_state.notes if not n.get("archived")])) + '</h3></div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="stat-box"><h2>⭐</h2><p>Favorites</p><h3>' + str(len([n for n in st.session_state.notes if n.get("favorite")])) + '</h3></div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="stat-box"><h2>🏷️</h2><p>Tags</p><h3>' + str(len(all_tags)) + '</h3></div>', unsafe_allow_html=True)
            with col4:
                total_words = sum(len(n["body"].split()) for n in st.session_state.notes)
                st.markdown('<div class="stat-box"><h2>📄</h2><p>Words</p><h3>' + f"{total_words:,}" + '</h3></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- PLOTLY CHARTS (New Graphics) ---
            if st.session_state.notes:
                # Prepare data for charts
                notes_data = st.session_state.notes
                
                # 1. Notes by Category
                categories = [n.get("category", "General") for n in notes_data]
                cat_df = pd.DataFrame(categories, columns=["Category"])
                cat_counts = cat_df["Category"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                
                # 2. Activity Timeline (Created dates)
                dates = [n["created_at"].split(" ")[0] for n in notes_data]
                date_df = pd.DataFrame(dates, columns=["Date"])
                date_counts = date_df["Date"].value_counts().reset_index()
                date_counts.columns = ["Date", "Count"]
                date_counts = date_counts.sort_values("Date")
                
                # 3. Tag Usage
                all_note_tags = []
                for n in notes_data:
                    all_note_tags.extend(n.get("tags", []))
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown("#### Notes by Category")
                    fig1 = px.pie(cat_counts, values="Count", names="Category", hole=0.4, template=chart_template)
                    fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col_chart2:
                    st.markdown("#### Creation Activity")
                    fig2 = px.line(date_counts, x="Date", y="Count", markers=True, template=chart_template)
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig2, use_container_width=True)
                
                if all_note_tags:
                    st.markdown("#### Top Tags")
                    tag_counts = collections.Counter(all_note_tags)
                    tag_df = pd.DataFrame(tag_counts.most_common(10), columns=["Tag", "Count"])
                    fig3 = px.bar(tag_df, x="Tag", y="Count", color="Count", template=chart_template)
                    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig3, use_container_width=True)

# Helper function for note cards
def display_note_card(note):
    is_active = note["id"] == st.session_state.active_id
    
    # Card styling
    card_class = "note-card"
    if note.get("pinned"):
        card_class += " pinned"
    elif note.get("favorite"):
        card_class += " favorite"
    
    # Icons
    icons = []
    if note.get("locked"):
        icons.append("🔒")
    if note.get("favorite"):
        icons.append("⭐")
    if note.get("pinned"):
        icons.append("📌")
    if note.get("archived"):
        icons.append("📦")
    
    icon_str = " ".join(icons)
    
    # Preview
    preview = note["body"][:100] + "..." if len(note["body"]) > 100 else note["body"]
    time_ago = format_time_ago(note.get("updated_at", note["created_at"]))
    
    # Tags
    tags_html = " ".join([f'<span class="tag">#{tag}</span>' for tag in note.get("tags", [])[:3]])
    
    col_main, col_actions = st.columns([4, 1])
    
    with col_main:
        if st.button(
            f"{icon_str} **{note['title']}**",
            key=f"note_{note['id']}",
            use_container_width=True,
            help="Click to edit"
        ):
            st.session_state.active_id = note["id"]
            st.rerun()
        st.caption(f"{preview}")
        if tags_html:
            st.markdown(tags_html, unsafe_allow_html=True)
        st.caption(f"🕐 {time_ago}")
    
    with col_actions:
        if st.button("⋮", key=f"menu_{note['id']}", help="Quick actions"):
            # This would open a context menu - simplified here
            pass

# =============================================================================
# CONVERTER MODE
# =============================================================================
elif st.session_state.view_mode == "converter":
    st.markdown("## 🔄 File Converter & Unzipper")
    
    uploaded_file = st.file_uploader(
        "Upload file to convert",
        type=["txt", "json", "xml", "csv", "py", "zip", "md"],
        help="Upload text files or ZIP archives"
    )
    
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'zip':
            st.success(f"📦 ZIP file detected: {uploaded_file.name}")
            
            if st.button("🔓 Extract ZIP Contents", type="primary"):
                with st.spinner("Extracting..."):
                    st.session_state.extracted_files = extract_zip(uploaded_file)
                st.rerun()
            
            if st.session_state.extracted_files:
                st.markdown(f"### 📁 Extracted Files ({len(st.session_state.extracted_files)})")
                
                for idx, file_info in enumerate(st.session_state.extracted_files):
                    with st.expander(f"📄 {file_info['name']} ({file_info['size']} bytes)"):
                        if file_info['type'] == 'text':
                            st.text_area(
                                "Content",
                                value=file_info['content'][:5000],
                                height=200,
                                key=f"extracted_{idx}",
                                label_visibility="collapsed"
                            )
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("💾 Save as Note", key=f"save_{idx}"):
                                    new_id = (max([n["id"] for n in st.session_state.notes], default=0) + 1)
                                    new_note = {
                                        "id": new_id,
                                        "title": file_info['name'],
                                        "body": file_info['content'],
                                        "created_at": now_stamp(),
                                        "updated_at": now_stamp(),
                                        "tags": extract_tags(file_info['content']),
                                        "pinned": False,
                                        "favorite": False,
                                        "archived": False,
                                        "locked": False,
                                        "category": "Imported"
                                    }
                                    st.session_state.notes.append(new_note)
                                    save_notes(st.session_state.notes)
                                    st.success("✅ Saved as note!")
                            
                            with col2:
                                st.download_button(
                                    "⬇️ Download",
                                    data=file_info['content'],
                                    file_name=file_info['name'],
                                    mime="text/plain",
                                    key=f"download_{idx}"
                                )
                        else:
                            st.info("Binary file - download to view")
                            st.download_button(
                                "⬇️ Download Binary",
                                data=file_info['content'],
                                file_name=file_info['name'],
                                key=f"download_bin_{idx}"
                            )
        
        else:
            try:
                content = uploaded_file.read().decode('utf-8')
                
                st.success(f"✅ Loaded: {uploaded_file.name}")
                st.text_area("File Content", value=content[:2000], height=200, disabled=True)
                
                if len(content) > 2000:
                    st.caption(f"Showing first 2000 characters of {len(content)} total")
                
                st.markdown("---")
                st.markdown("### 🔄 Convert To:")
                
                col1, col2, col3 = st.columns(3)
                
                conversions = [
                    ("📄 TXT", "txt", "text/plain"),
                    ("📋 JSON", "json", "application/json"),
                    ("🏷️ XML", "xml", "application/xml"),
                    ("📊 CSV", "csv", "text/csv"),
                    ("🐍 Python", "py", "text/x-python"),
                    ("📝 Markdown", "md", "text/markdown")
                ]
                
                for i, (label, fmt, mime) in enumerate(conversions):
                    col = [col1, col2, col3][i % 3]
                    with col:
                        if st.button(label, use_container_width=True, key=f"conv_{fmt}"):
                            if fmt == "txt":
                                converted = convert_to_txt(content, file_extension)
                            elif fmt == "json":
                                converted = convert_to_json(uploaded_file.name, content)
                            elif fmt == "xml":
                                converted = convert_to_xml(uploaded_file.name, content)
                            elif fmt == "csv":
                                converted = convert_to_csv(uploaded_file.name, content)
                            elif fmt == "py":
                                converted = convert_to_py(content)
                            else:  # md
                                converted = f"# {uploaded_file.name}\n\n{content}"
                            
                            st.download_button(
                                f"⬇️ Download {fmt.upper()}",
                                data=converted,
                                file_name=safe_filename(ensure_ext(uploaded_file.name.rsplit('.', 1)[0], f".{fmt}")),
                                mime=mime,
                                key=f"dl_{fmt}"
                            )
                
                st.markdown("---")
                if st.button("💾 Save as Note", type="primary", use_container_width=True):
                    new_id = (max([n["id"] for n in st.session_state.notes], default=0) + 1)
                    new_note = {
                        "id": new_id,
                        "title": uploaded_file.name,
                        "body": content,
                        "created_at": now_stamp(),
                        "updated_at": now_stamp(),
                        "tags": extract_tags(content),
                        "pinned": False,
                        "favorite": False,
                        "archived": False,
                        "locked": False,
                        "category": "Imported"
                    }
                    st.session_state.notes.append(new_note)
                    save_notes(st.session_state.notes)
                    st.success("✅ Saved as note!")
                    st.balloons()
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    else:
        st.info("👆 Upload a file to get started")
        
        st.markdown("### 📌 Supported Features:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **File Operations:**
            - 📦 Extract ZIP archives
            - 🔄 Convert formats
            - 💾 Import to notes
            - ⬇️ Export conversions
            """)
        with col2:
            st.markdown("""
            **Supported Formats:**
            - TXT, JSON, XML
            - CSV, Markdown
            - Python (.py)
            - ZIP archives
            """)

# =============================================================================
# SETTINGS MODE
# =============================================================================
elif st.session_state.view_mode == "settings":
    st.markdown("## ⚙️ Settings")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Appearance", "📋 Templates", "💾 Backup", "📊 Advanced"])
    
    with tab1:
        st.markdown("### Theme")
        theme = st.radio("Color Scheme", ["light", "dark"], index=0 if st.session_state.settings["theme"] == "light" else 1)
        if theme != st.session_state.settings["theme"]:
            st.session_state.settings["theme"] = theme
            save_settings(st.session_state.settings)
            st.success("Theme updated! Refresh to see changes.")
        
        st.markdown("### Layout")
        layout = st.radio("Interface Layout", ["desktop", "mobile"], index=0 if st.session_state.settings["layout"] == "desktop" else 1)
        if layout != st.session_state.settings["layout"]:
            st.session_state.settings["layout"] = layout
            save_settings(st.session_state.settings)
            st.success("Layout updated! Refresh to see changes.")
            st.rerun()
    
    with tab2:
        st.markdown("### Note Templates")
        
        for template in st.session_state.templates:
            with st.expander(f"📋 {template['name']}"):
                new_name = st.text_input("Template Name", value=template["name"], key=f"tpl_name_{template['id']}")
                new_content = st.text_area("Template Content", value=template["content"], height=200, key=f"tpl_content_{template['id']}")
                
                col_save, col_delete = st.columns(2)
                with col_save:
                    if st.button("💾 Save Changes", key=f"save_tpl_{template['id']}"):
                        template["name"] = new_name
                        template["content"] = new_content
                        save_templates(st.session_state.templates)
                        st.success("Template updated!")
                with col_delete:
                    if st.button("🗑️ Delete Template", key=f"del_tpl_{template['id']}"):
                        st.session_state.templates = [t for t in st.session_state.templates if t["id"] != template["id"]]
                        save_templates(st.session_state.templates)
                        st.rerun()
        
        st.markdown("---")
        st.markdown("### Create New Template")
        new_tpl_name = st.text_input("Template Name", placeholder="My Template")
        new_tpl_content = st.text_area("Template Content", placeholder="# Title\n\nContent here...", height=150)
        
        if st.button("➕ Add Template", type="primary"):
            if new_tpl_name and new_tpl_content:
                new_id = max([t["id"] for t in st.session_state.templates], default=0) + 1
                st.session_state.templates.append({
                    "id": new_id,
                    "name": new_tpl_name,
                    "content": new_tpl_content
                })
                save_templates(st.session_state.templates)
                st.success("Template created!")
                st.rerun()
            else:
                st.error("Please fill in all fields")
    
    with tab3:
        st.markdown("### Backup & Restore")
        
        # Auto-backup settings
        st.markdown("#### Automatic Backup")
        auto_backup = st.checkbox("Enable auto-backup", value=st.session_state.settings.get("auto_backup", False))
        if auto_backup:
            backup_interval = st.number_input("Backup interval (hours)", min_value=1, max_value=168, value=st.session_state.settings.get("backup_interval", 24))
            st.session_state.settings["auto_backup"] = auto_backup
            st.session_state.settings["backup_interval"] = backup_interval
            save_settings(st.session_state.settings)
        
        st.markdown("---")
        
        # Manual backup
        st.markdown("#### Manual Backup/Restore")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Export All Data**")
            if st.button("📦 Create Backup", use_container_width=True):
                backup_data = json.dumps({
                    "notes": st.session_state.notes,
                    "templates": st.session_state.templates,
                    "settings": st.session_state.settings
                }, indent=2)
                st.download_button(
                    label="⬇️ Download Backup",
                    data=backup_data,
                    file_name=f"notes_pro_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col2:
            st.markdown("**Restore from Backup**")
            restore_file = st.file_uploader("Upload backup file", type=['json'], key="restore_backup")
            if restore_file:
                try:
                    backup_data = json.loads(restore_file.read().decode('utf-8'))
                    if st.button("✅ Restore Backup", type="primary", use_container_width=True):
                        st.session_state.notes = backup_data.get("notes", [])
                        st.session_state.templates = backup_data.get("templates", [])
                        st.session_state.settings = backup_data.get("settings", {})
                        save_notes(st.session_state.notes)
                        save_templates(st.session_state.templates)
                        save_settings(st.session_state.settings)
                        st.success("Backup restored!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Invalid backup file: {str(e)}")
        
        st.markdown("---")
        
        # Export to folder
        st.markdown("#### Export Notes to Folder")
        col_format, col_export = st.columns([1, 1])
        
        with col_format:
            bulk_format = st.selectbox("Export format", ["txt", "json", "xml", "csv", "py", "md"], key="bulk_format")
        
        with col_export:
            st.write("")
            if st.button("💾 Export All to Downloads", type="primary", use_container_width=True):
                try:
                    downloads_path = Path.home() / "Downloads" / "NotesProExport"
                    downloads_path.mkdir(exist_ok=True)
                    
                    for note in st.session_state.notes:
                        filename = ensure_ext(safe_filename(note["title"]), f".{bulk_format}")
                        filepath = downloads_path / filename
                        
                        if bulk_format == "txt":
                            content = f"{note['title']}\n{'-'*40}\n{note['body']}"
                        elif bulk_format == "json":
                            content = json.dumps(note, indent=2)
                        elif bulk_format == "xml":
                            content = note_to_xml(note)
                        elif bulk_format == "csv":
                            content = convert_to_csv(note["title"], note["body"])
                        elif bulk_format == "py":
                            content = convert_to_py(note["body"])
                        else:  # md
                            content = f"# {note['title']}\n\n{note['body']}"
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                    
                    st.success(f"✅ Exported {len(st.session_state.notes)} notes to {downloads_path}")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error exporting: {str(e)}")
    
    with tab4:
        st.markdown("### Advanced Settings")
        
        st.markdown("#### Data Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Notes", type="secondary", use_container_width=True):
                if st.checkbox("I understand this will delete all notes"):
                    if st.button("⚠️ Confirm Delete All"):
                        st.session_state.notes = []
                        save_notes(st.session_state.notes)
                        st.success("All notes deleted")
                        st.rerun()
        
        with col2:
            if st.button("🔄 Reset Settings", use_container_width=True):
                st.session_state.settings = {
                    "theme": "light",
                    "layout": "desktop",
                    "auto_backup": False,
                    "backup_interval": 24
                }
                save_settings(st.session_state.settings)
                st.success("Settings reset!")
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("#### Storage Info")
        data_size = len(json.dumps(st.session_state.notes).encode('utf-8'))
        st.metric("Data Size", f"{data_size / 1024:.2f} KB")
        
        st.markdown("#### App Version")
        st.info("Notes Pro v2.0 - Desktop Edition")
        
        st.markdown("#### About")
        st.markdown("""
        **Notes Pro** is a full-featured note-taking application with:
        - 🏷️ Tags & Categories
        - 🔍 Advanced Search
        - 📎 File Attachments  
        - 🔗 Note Linking
        - 📋 Templates
        - 🌙 Dark Mode
        - 📊 Rich Text Editor
        - 🔄 Version History
        - 📱 Quick Actions
        - 🔐 Password Protection
        - 💾 Cloud Sync Ready
        - 🎤 Voice-to-Text Ready
        """)

# Show current view mode indicator
st.sidebar.markdown("---")
st.sidebar.caption(f"Current View: **{st.session_state.view_mode.title()}**")
st.sidebar.caption(f"Theme: **{st.session_state.settings['theme'].title()}**")
st.sidebar.caption(f"Layout: **{st.session_state.settings['layout'].title()}**")
