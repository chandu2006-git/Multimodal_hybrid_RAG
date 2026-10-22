
import hashlib
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(PROJECT_ROOT / ".env")

from ingestion.document_ingestor import ingest_document
from rag.pipeline import MultimodalRAGPipeline
from retriever.image_retriever import ImageRetriever
from generation.llm_judge import LLMJudge

DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "processed" / "uploads"
IMAGE_ROOT = DATA_DIR / "images"
TABLE_ROOT = DATA_DIR / "processed" / "tables"
IMAGE_INDEX_ROOT = DATA_DIR / "processed" / "image_index"

for directory in (
    UPLOAD_DIR,
    IMAGE_ROOT,
    TABLE_ROOT,
    IMAGE_INDEX_ROOT,
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. CHATGPT-INSPIRED DARK UI
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #17191F;
        --sidebar: #111318;
        --surface: #20242D;
        --surface-hover: #282D38;
        --border: #343A47;
        --text: #F1F5F9;
        --muted: #9AA5B5;
        --accent: #19C6A0;
        --accent-hover: #12A989;
        --accent-soft: rgba(25, 198, 160, 0.12);
    }

    /* ---------- App foundation ---------- */

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                     'Segoe UI', sans-serif;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(23, 25, 31, 0.92) !important;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.4rem;
        padding-bottom: 7rem;
    }

    /* ---------- Typography ---------- */

    h1, h2, h3, h4 {
        color: var(--text) !important;
        font-weight: 700 !important;
        letter-spacing: -0.035em;
    }

    p, li, label, [data-testid="stMarkdownContainer"] {
        color: #D7DDE7;
    }

    .muted {
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.65;
    }

    .brand {
        font-size: 1.18rem;
        font-weight: 800;
        letter-spacing: -0.045em;
        color: var(--text);
        margin-bottom: 0.4rem;
    }

    .brand-mark {
        color: var(--accent);
    }

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background: var(--sidebar) !important;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #282D37;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #E5EAF2 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #292E38 !important;
        margin: 1.25rem 0;
    }

    /* ---------- Welcome screen ---------- */

    .welcome {
        text-align: center;
        padding: 11vh 1rem 2.2rem;
    }

    .welcome .symbol {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 68px;
        height: 68px;
        border-radius: 22px;
        color: var(--accent);
        background: var(--accent-soft);
        border: 1px solid rgba(25, 198, 160, 0.24);
        font-size: 2.2rem;
        margin-bottom: 1.5rem;
    }

    .welcome h1 {
        font-size: clamp(1.8rem, 4vw, 2.65rem);
        line-height: 1.2;
        margin-bottom: 0.85rem;
    }

    .welcome p {
        color: var(--muted) !important;
        font-size: 1rem;
        line-height: 1.8;
    }

    /* ---------- Chat messages ---------- */

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 1.1rem 0;
    }

    [data-testid="stChatMessageContent"] {
        color: var(--text) !important;
        line-height: 1.85;
        font-size: 0.96rem;
    }

    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li {
        color: #DCE2EB !important;
        line-height: 1.85;
    }

    [data-testid="stChatMessageContent"] strong {
        color: #FFFFFF !important;
    }

    [data-testid="stChatMessageContent"] a {
        color: var(--accent) !important;
    }

    [data-testid="stChatMessageContent"] code {
        color: #8DEBD3;
        background: #252A34;
        border-radius: 5px;
        padding: 0.15rem 0.35rem;
    }


    /* ---------- Chat composer ---------- */

    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background: #17191F !important;
    }

    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] textarea {
        background: #242832 !important;
        color: #F1F5F9 !important;
        -webkit-text-fill-color: #F1F5F9 !important;
        caret-color: #19C6A0 !important;
    }

    [data-testid="stChatInput"] {
        border: 1px solid #3A414E !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.18) !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #19C6A0 !important;
        box-shadow: 0 0 0 3px rgba(25, 198, 160, 0.10) !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #9AA5B5 !important;
        -webkit-text-fill-color: #9AA5B5 !important;
    }

    [data-testid="stChatInput"] button {
        background: #19C6A0 !important;
        color: #09251F !important;
        border: none !important;
        border-radius: 11px !important;
    }

    /* ---------- Buttons ---------- */

    .stButton > button,
    [data-testid="stSidebar"] .stButton > button {
        background: #252A34 !important;
        color: #E9EEF5 !important;
        border: 1px solid #3B424F !important;
        border-radius: 11px !important;
        min-height: 2.65rem;
        font-family: inherit !important;
        font-weight: 500 !important;
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        background: #303744 !important;
        border-color: #596474 !important;
        transform: translateY(-1px);
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(25, 198, 160, 0.15) !important;
    }

    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        color: #08251E !important;
        border-color: var(--accent) !important;
        font-weight: 700 !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
    }

    /* ---------- File uploader ---------- */

    [data-testid="stFileUploader"] {
        background: #1B1F27 !important;
        border: 1px solid #343B48 !important;
        border-radius: 15px !important;
        padding: 0.4rem !important;
    }

    [data-testid="stFileUploader"] section {
        background: #20252E !important;
        border: 1px dashed #4B5665 !important;
        border-radius: 11px !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--accent) !important;
    }

    [data-testid="stFileUploader"] button {
        background: #303744 !important;
        color: #F1F5F9 !important;
        border: 1px solid #46505E !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {
        color: #AAB4C3;
    }

    /* ---------- Evidence and source panels ---------- */

    [data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 13px !important;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        background: transparent !important;
        padding: 0.8rem 1rem !important;
    }

    [data-testid="stExpander"] summary:hover {
        background: var(--surface-hover) !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        color: #E7EDF5 !important;
        font-weight: 500;
    }

    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.75;
    }

    /* ---------- Alerts and status messages ---------- */

    [data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid #354353 !important;
    }

    [data-testid="stAlert"] p {
        line-height: 1.65;
    }

    [data-testid="stStatusWidget"] {
        background: var(--surface) !important;
    }

    /* ---------- Inputs and selection controls ---------- */

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
        background: #222731 !important;
        color: var(--text) !important;
        border: 1px solid #3A414E !important;
        border-radius: 10px !important;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(25, 198, 160, 0.10) !important;
    }

    /* ---------- Dividers, captions, and code ---------- */

    hr {
        border-color: #303641 !important;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }

    [data-testid="stCode"],
    pre {
        border-radius: 10px !important;
    }

    /* ---------- Scrollbars ---------- */

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg);
    }

    ::-webkit-scrollbar-thumb {
        background: #414957;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #596475;
    }

    /* ---------- Keep Streamlit chrome subtle ---------- */

    [data-testid="stToolbar"] {
        background: transparent !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ---------- Responsive adjustments ---------- */

    @media (max-width: 768px) {
        .block-container {
            padding: 1.3rem 1rem 6rem;
        }

        .welcome {
            padding-top: 7vh;
        }

        .welcome h1 {
            font-size: 1.8rem;
        }

        [data-testid="stChatInput"] {
            border-radius: 14px !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. SESSION STATE
# ============================================================

DEFAULTS = {
    "processed_document": None,
    "document_name": None,
    "messages": [],
    "processing_error": None,
}

for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# 5. DOCUMENT PROCESSING
# ============================================================

def process_uploaded_pdf(uploaded_file):
    """
    Save the PDF, extract its assets, build its CLIP image index,
    and initialize the existing multimodal RAG pipeline.
    """
    file_bytes = uploaded_file.getvalue()

    # A content hash prevents different PDFs with the same filename
    # from accidentally sharing an image index.
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
    safe_stem = Path(uploaded_file.name).stem
    document_id = f"{safe_stem}_{file_hash}"

    persistent_pdf = UPLOAD_DIR / f"{document_id}.pdf"
    persistent_pdf.write_bytes(file_bytes)

    document_image_root = IMAGE_ROOT / persistent_pdf.stem
    document_index_dir = IMAGE_INDEX_ROOT / persistent_pdf.stem

    # Existing ingestion module extracts text, tables, and images.
    document = ingest_document(
        pdf_path=persistent_pdf,
        image_root=IMAGE_ROOT,
        table_root=TABLE_ROOT,
    )

    # The current MultimodalRAGPipeline requires a persisted image
    # index, so we build it before initializing the pipeline.
    image_retriever = ImageRetriever(
        image_dir=document_image_root,
        build_index=True,
    )
    image_retriever.save_index(document_index_dir)

    # Initialize the project's actual end-to-end RAG pipeline.
    pipeline = MultimodalRAGPipeline(
        pdf_path=persistent_pdf,
        image_index_dir=document_index_dir,
        image_root=IMAGE_ROOT,
        table_root=TABLE_ROOT,
    )

    return {
        "source": uploaded_file.name,
        "document_id": document_id,
        "pdf_path": persistent_pdf,
        "document": document,
        "pipeline": pipeline,
    }


# ============================================================
# 6. SOURCE AND EVIDENCE DISPLAY
# ============================================================

def render_sources(sources):
    if not sources:
        return

    with st.expander("📚 Sources", expanded=False):
        for source in sources:
            filename = source.get("source") or "Unknown source"
            page = source.get("page")

            if page is not None:
                st.markdown(f"**{filename}** · Page {page}")
            else:
                st.markdown(f"**{filename}**")


def render_evidence(evidence):
    if not evidence:
        return

    with st.expander(
        f"🔎 Retrieved evidence ({len(evidence)} items)",
        expanded=False,
    ):
        for index, item in enumerate(evidence, start=1):
            modality = item.get("modality", "unknown").title()
            filename = item.get("source") or "Unknown source"
            page = item.get("page")

            heading = f"{index}. {modality}"

            if page is not None:
                heading += f" · Page {page}"

            st.markdown(f"**{heading}**")
            st.caption(filename)

            content = item.get("content", "")
            if content:
                st.write(content)

            image_path = item.get("image_path")
            if image_path and Path(image_path).is_file():
                st.image(
                    image_path,
                    caption=filename,
                    use_container_width=True,
                )

            scores = item.get("scores")
            if scores:
                readable_scores = {
                    key: value
                    for key, value in scores.items()
                    if value is not None
                }
                if readable_scores:
                    st.caption(f"Retrieval scores: {readable_scores}")

            if index < len(evidence):
                st.divider()


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            model = message.get("model")
            if model:
                st.caption(f"Generated by {model}")

            render_sources(message.get("sources", []))
            render_evidence(message.get("evidence", []))


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">✳</span> '
        'Multimodal RAG</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="muted">Your AI research assistant</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button("＋  New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### Knowledge source")

    uploaded_file = st.file_uploader(
        "Upload a research paper",
        type=["pdf"],
        help="Upload a scientific paper in PDF format.",
    )

    if uploaded_file is not None:
        st.caption(f"Selected: {uploaded_file.name}")

        if st.button(
            "Process paper",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Extracting content and building multimodal indexes..."
                ):
                    processed = process_uploaded_pdf(uploaded_file)

                st.session_state.processed_document = processed
                st.session_state.document_name = uploaded_file.name
                st.session_state.messages = []
                st.session_state.processing_error = None

                st.success("Multimodal RAG is ready!")
                st.rerun()

            except Exception as exc:
                st.session_state.processing_error = str(exc)
                st.error(f"Processing failed: {exc}")

    if st.session_state.processing_error:
        with st.expander("Processing error details"):
            st.code(st.session_state.processing_error)

    active = st.session_state.processed_document

    if active:
        st.divider()
        st.markdown("### Current paper")
        st.markdown(f"**{active['source']}**")

        document = active["document"]

        st.markdown(
            f'<div class="muted">'
            f'{len(document["text_pages"])} pages · '
            f'{len(active["pipeline"].text_retriever.chunks)} text chunks'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="muted">'
            f'{len(document["tables"])} tables extracted · '
            f'{len(document["images"])} images extracted'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    with st.expander("How it works"):
        st.markdown(
            """
            **Active pipeline**

            1. PDF ingestion
            2. Text chunking
            3. Dense + BM25 retrieval
            4. Reciprocal Rank Fusion
            5. CLIP image retrieval
            6. Table retrieval
            7. Evidence construction
            8. Gemini answer generation

            **Next improvement**

            Cross-encoder reranking can be integrated into the
            retrieval stage after validating the current pipeline.
            """
        )

    st.markdown(
        '<div class="muted">Local research workspace</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 8. MAIN CHAT WORKSPACE
# ============================================================

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome">
            <div class="symbol">✳</div>
            <h1>What would you like to explore?</h1>
            <p>Ask questions about your scientific paper.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.processed_document is None:
        st.info(
            "Upload a PDF in the sidebar and select **Process paper** "
            "to start asking questions."
        )
    else:
        st.caption(
            f"Ready to explore: {st.session_state.document_name}"
        )


# Render conversation history.
for message in st.session_state.messages:
    render_message(message)


# ============================================================
# 9. CHAT INPUT AND REAL RAG ANSWERS
# ============================================================

question = st.chat_input(
    "Ask anything about your paper...",
    disabled=st.session_state.processed_document is None,
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    try:
        with st.spinner(
            "Searching text, tables and figures, then generating an answer..."
        ):
            pipeline = (
                st.session_state.processed_document["pipeline"]
            )

            # Calls the existing project's retrieve + Gemini generation.
            result = pipeline.ask(question)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "evidence": result.get("evidence", []),
                "model": result.get("model"),
            }
        )

    except Exception as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "I couldn't complete the request. "
                    "Please check the error details below."
                ),
                "error": str(exc),
            }
        )

    st.rerun()


# Show errors associated with the latest assistant response.
if st.session_state.messages:
    last_message = st.session_state.messages[-1]

    if last_message.get("error"):
        with st.expander("Error details"):
            st.code(last_message["error"])
            
