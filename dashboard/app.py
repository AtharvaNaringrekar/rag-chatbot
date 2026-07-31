import os
# Bypasses system HTTP/HTTPS proxies for localhost/loopback calls
os.environ["NO_PROXY"] = "127.0.0.1,localhost,localhost:8000,127.0.0.1:8000"

import logging
import time
import streamlit as st
import requests

# Configure logging for Streamlit dashboard output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App Configuration
st.set_page_config(
    page_title="Spintly Technical Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_V1_URL = f"{BACKEND_URL}/api/v1"

# Custom Styling for modern ChatGPT-like UI
st.markdown("""
<style>
    .reportview-container { background: #121212; }
    .sidebar .sidebar-content { background: #1e1e1e; }
    .stButton>button { width: 100%; border-radius: 6px; }
    .status-badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 14px; display: inline-block; margin-bottom: 5px; }
    .status-ok { background-color: #2e7d32; color: white; }
    .status-error { background-color: #c62828; color: white; }

    /* Add padding to the bottom of the main scroll container so fixed composer doesn't cover content */
    .main .block-container {
        padding-bottom: 160px !important;
    }

    /* Style the horizontal block containing the chat input to be the fixed bottom composer bar */
    div[data-testid="stHorizontalBlock"]:has(textarea) {
        position: fixed !important;
        bottom: 45px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 90% !important;
        max-width: 760px !important;
        background-color: #1e2225 !important;
        border: 1px solid #3f4447 !important;
        border-radius: 16px !important;
        padding: 12px 16px !important;
        z-index: 9999 !important;
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.6) !important;
    }
    
    /* Ensure the columns inside this block are aligned vertically in the center */
    div[data-testid="stHorizontalBlock"]:has(textarea) > div {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Style the stFileUploader widget inside the composer row */
    div[data-testid="stHorizontalBlock"]:has(textarea) div.stFileUploader {
        margin: 0 !important;
        padding: 0 !important;
        width: auto !important;
    }
    
    div[data-testid="stHorizontalBlock"]:has(textarea) div.stFileUploader section {
        min-height: 0 !important;
        padding: 0 !important;
        border: none !important;
        background-color: transparent !important;
        width: 40px !important;
        height: 40px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Hide all child nodes inside the file uploader section except the hidden input element */
    div[data-testid="stHorizontalBlock"]:has(textarea) div.stFileUploader section > *:not(input) {
        display: none !important;
    }
    
    /* Hide the built-in file uploader details list cards entirely */
    div[data-testid="stHorizontalBlock"]:has(textarea) div.stFileUploader [data-testid="stFileUploaderFile"],
    div[data-testid="stHorizontalBlock"]:has(textarea) div.stFileUploader div[data-testid="stFileUploaderDropzone"] + div {
        display: none !important;
    }
    
    /* Draw a clean circular paperclip button '📎' on the file uploader dropzone */
    div[data-testid="stHorizontalBlock"]:has(textarea) div.stFileUploader section::after {
        content: "📎" !important;
        font-size: 20px !important;
        color: #e0e0e0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 40px !important;
        height: 40px !important;
        background-color: #2c3034 !important;
        border: 1px solid #495057 !important;
        border-radius: 50% !important;
        cursor: pointer !important;
        transition: background-color 0.2s, border-color 0.2s, color 0.2s;
    }
    
    div[data-testid="stHorizontalBlock"]:has(textarea) div.stFileUploader section:hover::after {
        background-color: #3d4246 !important;
        border-color: #6c757d !important;
        color: #fff !important;
    }

    /* Style st.text_area inside composer block */
    div[data-testid="stHorizontalBlock"]:has(textarea) div[data-testid="stTextArea"] {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
    }
    
    div[data-testid="stHorizontalBlock"]:has(textarea) div[data-testid="stTextArea"] textarea {
        background-color: #2b3035 !important;
        border: 1px solid #495057 !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
        height: 75px !important;
        min-height: 75px !important;
        max-height: 75px !important;
        resize: none !important;
        padding: 10px 14px !important;
        font-size: 14px !important;
    }
    
    div[data-testid="stHorizontalBlock"]:has(textarea) div[data-testid="stTextArea"] textarea:focus {
        border-color: #6c757d !important;
        box-shadow: 0 0 0 1px #6c757d !important;
    }

    /* Style send button inside composer block to match textarea height */
    div[data-testid="stHorizontalBlock"]:has(textarea) div.stButton button {
        background-color: #0b5ed7 !important;
        border: 1px solid #0a58ca !important;
        border-radius: 12px !important;
        color: #fff !important;
        height: 75px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: background-color 0.2s, border-color 0.2s;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
    }
    
    div[data-testid="stHorizontalBlock"]:has(textarea) div.stButton button:hover {
        background-color: #3182ce !important;
        border-color: #3182ce !important;
    }

    /* Pinned preview container styled to display BELOW the chat composer */
    .preview-container {
        position: fixed !important;
        bottom: 10px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: calc(100% - 40px) !important;
        max-width: 740px !important;
        background-color: #1f2326 !important;
        border: 1px solid #3f4447 !important;
        border-radius: 8px !important;
        padding: 4px 12px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        z-index: 10001 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4) !important;
    }
    
    .preview-thumbnail {
        width: 28px !important;
        height: 28px !important;
        border-radius: 4px !important;
        object-fit: cover !important;
        border: 1px solid #4f5457 !important;
    }
    
    .preview-filename {
        color: #e0e0e0 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        font-family: inherit !important;
    }
    
    .preview-remove-link {
        color: #9f9f9f !important;
        text-decoration: none !important;
        font-size: 18px !important;
        font-weight: normal !important;
        transition: color 0.2s !important;
        padding: 2px 6px !important;
        cursor: pointer !important;
    }
    
    .preview-remove-link:hover {
        color: #ff4d4d !important;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health():
    """
    Call health check endpoint on FastAPI backend.
    """
    try:
        url = f"{API_V1_URL}/health"
        response = requests.get(url, timeout=10.0, proxies={"http": None, "https": None})
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Health check failed. Code: {response.status_code}, Body: {response.text}")
    except Exception as e:
        logger.error(f"Health check failed with exception: {e}")
    return None


def clean_display_answer(text: str, dev_mode: bool = False) -> str:
    """
    Clean the final response for display in the Streamlit chat UI by stripping out
    source citations and general documentation prefaces.
    """
    # 1. Clean JSON escape sequences
    text = text.replace("\\n", "\n").replace('\\"', '"').replace('\\`', '`')

    # If Developer Mode is enabled, keep all debug outputs intact
    if dev_mode:
        return text.strip()

    # Check for hallucination/pre-trained knowledge indicators
    hallucination_indicators = [
        "developed by microsoft",
        "knowledge cutoff",
        "knowledge is limited",
        "don't have internet",
        "hypothetical response",
        "hypothetically",
        "limitations of my training",
        "as an ai",
        "my knowledge up until",
        "not present in the provided context",
        "not mentioned in the retrieved context",
        "without access to real-time"
    ]
    if any(indicator in text.lower() for indicator in hallucination_indicators):
        return "I could not find this information in the available Spintly documentation."
    
    # 2. Split and remove "Source References" & "REFERENCE SENTENCES" sections
    for marker in [
        "## Source References", 
        "### Source References", 
        "Source References:", 
        "## Source Reference", 
        "### Source Reference",
        "REFERENCE SENTENCES:",
        "REFERENCE SENTENCES"
    ]:
        if marker in text:
            text = text.split(marker)[0]
            
    # 3. Strip trailing markdown page dividers and whitespace
    text = text.strip()
    if text.endswith("---"):
        text = text[:-3].strip()
        
    # 4. Remove sentences referencing Spintly documentation or context blocks
    import re
    
    # We strip matching introductory sentences from the beginning of the text
    # using a loop, since there could be multiple such preface sentences.
    while True:
        original = text
        
        # Pattern matching: "The source document titled '...' provides info..."
        text = re.sub(
            r'(?i)^\s*the\s+(?:source\s+)?document\s+(?:titled|named)?\s+[\'"][^\'"]+[\'"]\s+provides\s+[^.:]+?(?:\.|\:)\s*',
            '',
            text
        )
        
        # Pattern matching: "According to the context/document/reference..."
        text = re.sub(
            r'(?i)^\s*according\s+to\s+the\s+(?:retrieved\s+|available\s+|provided\s+|uploaded\s+)?(?:spintly\s+)?(?:documentation|document|context|file|collection|pdf|source|reference|text)s?,?\s*[^.:]+?(?:\.|\:)\s*', 
            '', 
            text
        )
        
        # Pattern matching: "Based on the retrieved context/document/reference..."
        text = re.sub(
            r'(?i)^\s*based\s+on\s+the\s+(?:retrieved\s+|available\s+|provided\s+|uploaded\s+)?(?:spintly\s+)?(?:documentation|document|context|file|collection|pdf|source|reference|text)s?,?\s*[^.:]+?(?:\.|\:)\s*', 
            '', 
            text
        )
        
        # Pattern matching: "The document indicates/states..."
        text = re.sub(
            r'(?i)^\s*the\s+(?:retrieved\s+|available\s+|provided\s+|uploaded\s+)?(?:spintly\s+)?(?:documentation|document|context|file|collection|pdf|source|reference|text)s?\s+(?:indicates?|states?|specifies|shows|reports|notes|mentions|says)[^.:]+?(?:\.|\:)\s*', 
            '', 
            text
        )
        
        # Pattern matching: "Here are the relevant details/information..."
        text = re.sub(
            r'(?i)^\s*(?:here\s+are\s+the\s+relevant\s+(?:details|information|endpoints|apis)(?:\s+for\s+[^.:]+?)?\s*(?:\.|\:)\s*)',
            '',
            text
        )
        
        # Standard prefix check (non-sentence-ending prefaces)
        text = re.sub(
            r'(?i)^\s*(?:according\s+to\s+the\s+(?:retrieved\s+|available\s+|provided\s+|uploaded\s+)?(?:spintly\s+)?(?:documentation|document|context|file|collection|pdf|source|reference|text)s?,?\s*)', 
            '', 
            text
        )
        text = re.sub(
            r'(?i)^\s*(?:based\s+on\s+the\s+(?:retrieved\s+|available\s+|provided\s+|uploaded\s+)?(?:spintly\s+)?(?:documentation|document|context|file|collection|pdf|source|reference|text)s?,?\s*)', 
            '', 
            text
        )
        text = re.sub(
            r'(?i)^\s*(?:the\s+(?:retrieved\s+|available\s+|provided\s+|uploaded\s+)?(?:spintly\s+)?(?:documentation|document|context|file|collection|pdf|source|reference|text)s?\s+(?:indicates?|states?|specifies|shows|reports|notes|mentions|says)(?:\s+that)?\s*)', 
            '', 
            text
        )
        text = re.sub(
            r'(?i)^\s*(?:the\s+source\s+reference\s+(?:from\s+[\'"][^\'"]+[\'"]\s+)?(?:indicates?|states?|specifies|shows|reports|notes|mentions|says)(?:\s+that)?\s*)', 
            '', 
            text
        )
        
        # If no changes were made in this iteration, exit the loop
        if text == original:
            break
            
    # General cleanup of remaining in-sentence references to documents or PDFs
    text = re.sub(
        r'(?i)\b(?:retrieved|uploaded|provided|available)\s+(?:spintly\s+)?(?:documentation|document|pdf|file|context|collection)\b',
        'documentation',
        text
    )
    
    # Capitalize the first letter if we stripped the prefix sentence starter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
        
    return text.strip()


# Session State initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "screenshot_uploader_0"
if "processing" not in st.session_state:
    st.session_state.processing = False
if "user_query" not in st.session_state:
    st.session_state.user_query = None
if "attached_img_val" not in st.session_state:
    st.session_state.attached_img_val = None
if "attached_img_name" not in st.session_state:
    st.session_state.attached_img_name = None
if "attached_img_type" not in st.session_state:
    st.session_state.attached_img_type = None
if "attached_img_b64" not in st.session_state:
    st.session_state.attached_img_b64 = None
if "textarea_key" not in st.session_state:
    st.session_state.textarea_key = "textarea_widget_0"

# Listen to removal query parameter from custom html anchor link
if st.query_params.get("remove") == "1":
    if "uploader_key" in st.session_state:
        import re
        match = re.search(r'\d+$', st.session_state.uploader_key)
        if match:
            num = int(match.group()) + 1
            st.session_state.uploader_key = f"screenshot_uploader_{num}"
    st.query_params.clear()
    st.rerun()

#Header
st.title("Spintly Technical Support Assistant")
st.caption("AI assistant for Spintly API documentation, integration guidance, and error diagnostics.")

# SIDEBAR CONTROL PANEL (Minimal sidebar)
with st.sidebar:
    st.header("⚙️ Control Center")
    
    # 1. System Health Status Indicators
    st.subheader("System Status")
    health = check_backend_health()
    if health:
        db_connected = health.get("database_connected", False)
        ollama_ready = health.get("ollama_reachable", False)
        api_status = health.get("status", "ok")
        
        # API Badge
        if api_status == "ok":
            st.markdown('<span class="status-badge status-ok">API: Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-error">API: Degraded</span>', unsafe_allow_html=True)
            
        # Database Badge
        if db_connected:
            st.markdown('<span class="status-badge status-ok">DB: Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-error">DB: Offline</span>', unsafe_allow_html=True)
            
        # LLM Badge
        if ollama_ready:
            st.markdown('<span class="status-badge status-ok">LLM: Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-error">LLM: Offline</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-error">API: Offline</span>', unsafe_allow_html=True)
        st.markdown('<span class="status-badge status-error">DB: Offline</span>', unsafe_allow_html=True)
        st.markdown('<span class="status-badge status-error">LLM: Offline</span>', unsafe_allow_html=True)
        st.caption(f"Backend unreachable at {BACKEND_URL}")

    # 3. Developer Mode Toggle
    st.subheader("🛠️ Debug Settings")
    dev_mode = st.toggle(
        "Developer Mode", 
        value=False, 
        help="Enable to inspect RAG retrieval logs, cited sources, and visual model reasoning."
    )

    st.markdown("---")

    # 4. Clear Chat Button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.toast("Chat history cleared!", icon="🗑️")
        st.rerun()

    st.markdown("---")


# MAIN CONVERSATIONAL AREA
# Render conversation messages from history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        # Render attached image if present (display above user message)
        if "image_b64" in msg and msg["image_b64"]:
            import base64
            img_bytes = base64.b64decode(msg["image_b64"])
            st.image(img_bytes, width=280)
            
        if msg["role"] == "assistant":
            content_clean = clean_display_answer(msg["content"], dev_mode=dev_mode)
        else:
            content_clean = msg["content"].replace("\\n", "\n").replace('\\"', '"').replace('\\`', '`')
        st.markdown(content_clean)
        
        # Render OCR/Vision diagnostics details if present (only in Developer Mode)
        if dev_mode and "vision_description" in msg and msg["vision_description"]:
            with st.expander("🔍 Screenshot Extraction Details"):
                st.write(f"**Visual Scene:** {msg['vision_description']}")
                st.write("**Transcribed OCR Text:**")
                st.code(msg["extracted_text"], language="text")
                
        # Render cited sources if present (only in Developer Mode)
        if dev_mode and "sources" in msg and msg["sources"]:
            with st.expander("📚 Cited Sources"):
                for src in msg["sources"]:
                    desc = f"**File:** {src.get('filename')} (Similarity: {src.get('similarity_score'):.4f})"
                    parts = []
                    if src.get("page_number"):
                        parts.append(f"Page: {src['page_number']}")
                    if src.get("section"):
                        parts.append(f"Section: {src['section']}")
                    if src.get("api_path"):
                        parts.append(f"Endpoint: {src.get('http_method')} {src['api_path']}")
                    
                    details = f", {', '.join(parts)}" if parts else ""
                    st.markdown(f"- {desc}{details}")


# COMPOSER ROW WITH STREAMLIT COLUMNS
col_attach, col_input, col_send = st.columns([1, 9, 2])

with col_attach:
    attached_screenshot = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"],
        key=st.session_state.uploader_key,
        label_visibility="collapsed",
        disabled=st.session_state.processing
    )

# THUMBNAIL PREVIEW DIRECTLY BELOW COMPOSER
if attached_screenshot and not st.session_state.processing:
    import base64
    img_b64 = base64.b64encode(attached_screenshot.getvalue()).decode()
    
    st.markdown(f"""
    <div class="preview-container">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 16px; display: flex; align-items: center;">📷</span>
            <img src="data:image/{attached_screenshot.type.split('/')[-1]};base64,{img_b64}" class="preview-thumbnail"/>
            <span class="preview-filename">{attached_screenshot.name}</span>
        </div>
        <a href="/?remove=1" target="_self" class="preview-remove-link" title="Remove image">✕</a>
    </div>
    """, unsafe_allow_html=True)

with col_input:
    # Initialize session state value for text area if not exists
    if "textarea_val" not in st.session_state:
        st.session_state.textarea_val = ""
        
    user_input_text = st.text_area(
        "Chat Input",
        value=st.session_state.textarea_val,
        placeholder="Ask anything about the Spintly APIs..." if not st.session_state.processing else "Generating response...",
        label_visibility="collapsed",
        key=st.session_state.textarea_key,
        disabled=st.session_state.processing
    )

with col_send:
    send_clicked = st.button("Send", key="send_btn", disabled=st.session_state.processing)

# Process initial message submission
if not st.session_state.processing and send_clicked and st.session_state.get(st.session_state.textarea_key, "").strip():
    user_input = st.session_state[st.session_state.textarea_key].strip()
    st.session_state.user_query = user_input
    st.session_state.textarea_val = ""
    
    # Rotate the textarea key to force a clean reset of the widget
    import re
    match = re.search(r'\d+$', st.session_state.textarea_key)
    if match:
        num = int(match.group()) + 1
        st.session_state.textarea_key = f"textarea_widget_{num}"
    
    if attached_screenshot:
        import base64
        st.session_state.attached_img_val = attached_screenshot.getvalue()
        st.session_state.attached_img_name = attached_screenshot.name
        st.session_state.attached_img_type = attached_screenshot.type
        st.session_state.attached_img_b64 = base64.b64encode(attached_screenshot.getvalue()).decode()
    else:
        st.session_state.attached_img_val = None
        st.session_state.attached_img_name = None
        st.session_state.attached_img_type = None
        st.session_state.attached_img_b64 = None
        
    st.session_state.processing = True
    st.rerun()

# Execute request when processing is active
if st.session_state.processing and st.session_state.user_query:
    user_input = st.session_state.user_query
    
    # 1. Display active user query
    with st.chat_message("user"):
        if st.session_state.attached_img_val:
            st.image(st.session_state.attached_img_val, width=280)
        st.markdown(user_input)
        
    # 2. Query API Backend with loader
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        if st.session_state.attached_img_val:
            with st.spinner("Searching documentation context..."):
                files = {"image": (st.session_state.attached_img_name, st.session_state.attached_img_val, st.session_state.attached_img_type)}
                data = {"prompt": user_input, "top_k": 3}
                
                try:
                    url = f"{API_V1_URL}/chat/vision"
                    logger.info(f"[DASHBOARD] Sending image chat request to: {url}")
                    logger.info(f"[DASHBOARD] Request Payload: {data}")
                    logger.info(f"[DASHBOARD] Timeout: 240.0s")
                    
                    start_dash = time.time()
                    resp = requests.post(
                        url, 
                        files=files, 
                        data=data, 
                        timeout=240.0, 
                        proxies={"http": None, "https": None}
                    )
                    duration = time.time() - start_dash
                    logger.info(f"[DASHBOARD] Received response in {duration:.3f} seconds.")
                    logger.info(f"[DASHBOARD] Status Code: {resp.status_code}")
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        
                        # Display response text
                        answer_clean = clean_display_answer(result["answer"], dev_mode=dev_mode)
                        response_placeholder.markdown(answer_clean)
                    
                        # Display debug/retrieval details (only in Developer Mode)
                        if dev_mode:
                            with st.expander("🔍 Screenshot Extraction Details"):
                                st.write(f"**Visual Scene:** {result['vision_description']}")
                                st.write("**Transcribed OCR Text:**")
                                st.code(result["extracted_text"], language="text")
                                
                            if result.get("sources"):
                                with st.expander("📚 Cited Sources"):
                                    for src in result["sources"]:
                                        desc = f"**File:** {src.get('filename')} (Similarity: {src.get('similarity_score'):.4f})"
                                        parts = []
                                        if src.get("page_number"):
                                            parts.append(f"Page: {src['page_number']}")
                                        if src.get("section"):
                                            parts.append(f"Section: {src['section']}")
                                        if src.get("api_path"):
                                            parts.append(f"Endpoint: {src.get('http_method')} {src['api_path']}")
                                        
                                        details = f", {', '.join(parts)}" if parts else ""
                                        st.markdown(f"- {desc}{details}")
                                    
                        # Record history
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": user_input,
                            "image_b64": st.session_state.attached_img_b64
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result.get("sources", []),
                            "extracted_text": result["extracted_text"],
                            "vision_description": result["vision_description"]
                        })
                        
                        # Clear uploader and text on success by incrementing uploader key
                        if "uploader_key" in st.session_state:
                            import re
                            match = re.search(r'\d+$', st.session_state.uploader_key)
                            if match:
                                num = int(match.group()) + 1
                                st.session_state.uploader_key = f"screenshot_uploader_{num}"
                        
                        # Reset states
                        st.session_state.processing = False
                        st.session_state.user_query = None
                        st.session_state.attached_img_val = None
                        st.session_state.attached_img_name = None
                        st.session_state.attached_img_type = None
                        st.session_state.attached_img_b64 = None
                        st.session_state.textarea_val = ""
                        st.rerun()
                        
                    else:
                        detail = resp.json().get("detail", "Error communicating with LLM.")
                        response_placeholder.error(f"Error: {detail}")
                        st.session_state.processing = False
                        st.session_state.user_query = None
                except Exception as e:
                    response_placeholder.error(f"API Connection Error: {e}")
                    st.session_state.processing = False
                    st.session_state.user_query = None
            
        else:
            with st.spinner("Searching documentation context..."):
                payload = {"prompt": user_input, "top_k": 3}
                try:
                    url = f"{API_V1_URL}/chat"
                    logger.info(f"[DASHBOARD] Sending chat request to: {url}")
                    logger.info(f"[DASHBOARD] Request Payload: {payload}")
                    logger.info(f"[DASHBOARD] Timeout: 240.0s")
                    
                    start_dash = time.time()
                    resp = requests.post(
                        url, 
                        json=payload, 
                        timeout=240.0, 
                        proxies={"http": None, "https": None}
                    )
                    duration = time.time() - start_dash
                    logger.info(f"[DASHBOARD] Received response in {duration:.3f} seconds.")
                    logger.info(f"[DASHBOARD] Status Code: {resp.status_code}")
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        
                        # Display response text
                        answer_clean = clean_display_answer(result["answer"], dev_mode=dev_mode)
                        response_placeholder.markdown(answer_clean)
                        
                        # Display cited source blocks (only in Developer Mode)
                        if dev_mode and result.get("sources"):
                            with st.expander("📚 Cited Sources"):
                                for src in result["sources"]:
                                    desc = f"**File:** {src.get('filename')} (Similarity: {src.get('similarity_score'):.4f})"
                                    parts = []
                                    if src.get("page_number"):
                                        parts.append(f"Page: {src['page_number']}")
                                    if src.get("section"):
                                        parts.append(f"Section: {src['section']}")
                                    if src.get("api_path"):
                                        parts.append(f"Endpoint: {src.get('http_method')} {src['api_path']}")
                                    
                                    details = f", {', '.join(parts)}" if parts else ""
                                    st.markdown(f"- {desc}{details}")
                                    
                        # Record history
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": user_input
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result.get("sources", [])
                        })
                        
                        # Reset states
                        st.session_state.processing = False
                        st.session_state.user_query = None
                        st.session_state.textarea_val = ""
                        st.rerun()
                        
                    else:
                        detail = resp.json().get("detail", "Error communicating with LLM.")
                        response_placeholder.error(f"Error: {detail}")
                        st.session_state.processing = False
                        st.session_state.user_query = None
                except Exception as e:
                    response_placeholder.error(f"API Connection Error: {e}")
                    st.session_state.processing = False
                    st.session_state.user_query = None

# Automatically focus back on input box after response rendering completes
if not st.session_state.processing:
    st.components.v1.html(
        """
        <script>
            const textarea = window.parent.document.querySelector('textarea');
            if (textarea) {
                textarea.focus();
            }
        </script>
        """,
        height=0,
        width=0
    )
