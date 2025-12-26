import streamlit as st
import io
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Any, List, Optional

# =============================================================================
# Text to .txt Mobile: Simplified Mobile-First Notes
# =============================================================================

# 'centered' layout reads better on phones than 'wide'
st.set_page_config(page_title="Text to .txt- Mobile", page_icon="📱", layout="centered")

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
# State Management
# ----------------------------
if "notes" not in st.session_state:
    st.session_state.notes = []
if "active_id" not in st.session_state:
    st.session_state.active_id = None

def get_note(nid):
    return next((n for n in st.session_state.notes if n["id"] == nid), None)

def create_note():
    new_id = (max([n["id"] for n in st.session_state.notes], default=0) + 1)
    st.session_state.notes.append({
        "id": new_id,
        "title": f"Note {new_id}",
        "body": "",
        "created_at": now_stamp()
    })
    st.session_state.active_id = new_id

def delete_note(nid):
    st.session_state.notes = [n for n in st.session_state.notes if n["id"] != nid]
    st.session_state.active_id = None
    st.rerun()

# =============================================================================
# Mobile UI Layout
# =============================================================================

st.title("📱 Notes")

# --- Top Control Bar ---
# We use columns here, but simple ones that stack well if needed
col_nav, col_new = st.columns([0.7, 0.3])

with col_new:
    if st.button("➕ New", use_container_width=True):
        create_note()
        st.rerun()

with col_nav:
    if not st.session_state.notes:
        st.write("Create a note to start.")
        active_note = None
    else:
        # Create a lookup for the selectbox
        # Sort newest first
        sorted_notes = sorted(st.session_state.notes, key=lambda x: x["id"], reverse=True)
        note_options = {f"{n['title']}": n["id"] for n in sorted_notes}
        
        # Determine index of current selection
        current_index = 0
        if st.session_state.active_id in note_options.values():
            # Find the key (title) that corresponds to the active ID
            curr_key = next(k for k, v in note_options.items() if v == st.session_state.active_id)
            current_index = list(note_options.keys()).index(curr_key)

        selected_label = st.selectbox(
            "Select Note",
            options=list(note_options.keys()),
            index=current_index,
            label_visibility="collapsed"
        )
        
        # Update active ID based on selection
        if selected_label:
            st.session_state.active_id = note_options[selected_label]
            active_note = get_note(st.session_state.active_id)
        else:
            active_note = None

st.markdown("---")

# --- Main Editor ---
if active_note:
    # 1. Title Input
    new_title = st.text_input("Title", value=active_note["title"], label_visibility="collapsed", placeholder="Note Title...")
    
    # 2. Body Input
    # Height is reduced slightly to fit mobile screens better without scrolling too much
    new_body = st.text_area("Body", value=active_note["body"], height=350, label_visibility="collapsed", placeholder="Start typing...")

    # Update State
    if new_title != active_note["title"] or new_body != active_note["body"]:
        active_note["title"] = new_title
        active_note["body"] = new_body
        active_note["updated_at"] = now_stamp()

    # --- Mobile Toolbox (Expandable) ---
    # Kept closed by default to save space
    with st.expander("🛠️ Toolbox (Export / Delete)", expanded=False):
        
        st.caption("Export Options")
        fmt = st.selectbox("Format", ["txt", "xml", "json"])
        
        # Prepare Data
        file_data = ""
        mime_type = "text/plain"
        ext = ".txt"

        if fmt == "txt":
            file_data = f"{active_note['title']}\n{'-'*20}\n{active_note['body']}"
        elif fmt == "xml":
            file_data = note_to_xml(active_note)
            mime_type = "application/xml"
            ext = ".xml"
        elif fmt == "json":
            file_data = json.dumps(active_note, indent=2)
            mime_type = "application/json"
            ext = ".json"

        # Big Tappable Download Button
        st.download_button(
            label="⬇️ Download File",
            data=file_data,
            file_name=safe_filename(ensure_ext(active_note["title"], ext)),
            mime=mime_type,
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Big Tappable Delete Button
        if st.button("🗑️ Delete Note", type="primary", use_container_width=True):
            delete_note(active_note["id"])

else:
    st.info("👆 Tap 'New' to create a note.")
