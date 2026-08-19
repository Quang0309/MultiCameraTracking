import streamlit as st
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

st.set_page_config(
    page_title="Multi-Camera Person Re-Identification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Multi-Camera Person Re-Identification")

st.sidebar.title("Configuration")
model_type = st.sidebar.selectbox("Model Selector", ["ResNet-50", "OSNet", "TransReID"])

# Save model selection in session state
st.session_state["model_type"] = model_type

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This application demonstrates Multi-Camera Person Tracking and Re-Identification.
- **Gallery Search**: Find matches for a query image in a pre-built FAISS gallery.
- **Live Detection**: Run object detection and ReID tracking on a video using a query image.
""")

page = st.sidebar.radio("Navigation", ["Gallery Search", "Live Detection"])

if page == "Gallery Search":
    from app.pages import gallery_search
    gallery_search.render()
elif page == "Live Detection":
    from app.pages import live_detection
    live_detection.render()
