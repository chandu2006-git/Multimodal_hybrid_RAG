
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
    /* Global backgrounds */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stHeader"],
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background-color: #212121 !important;
        color: #ececec !important;
    }

    [data-testid="stToolbar"] {
        background-color: transparent !important;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 8rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #171717 !important;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #303030;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ececec !important;
    }

    /* Typography */
    h1, h2, h3, h4, p, label {
        color: #ececec;
    }

    .brand {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.4px;
        margin-bottom: 0.35rem;
    }

    .muted {
        color: #a0a0a0;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    .welcome {
        text-align: center;
        padding-top: 13vh;
        padding-bottom: 2rem;
    }

    .welcome .symbol {
        font-size: 2.6rem;
        margin-bottom: 1rem;
    }

    .welcome h1 {
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.7px;
    }

    .welcome p {
        color: #a0a0a0;
    }

    /* Conversation */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    [data-testid="stChatMessageContent"] {
        color: #ececec !important;
        line-height: 1.75;
    }

    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li {
        color: #ececec !important;
    }

    /* Chat composer */
    [data-testid="stChatInput"] {
        background: #2f2f2f !important;
        border: 1px solid #505050 !important;
        border-radius: 24px !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #777777 !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #ececec !important;
        -webkit-text-fill-color: #ececec !important;
        caret-color: #ececec !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #a0a0a0 !important;
        -webkit-text-fill-color: #a0a0a0 !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        background: #ececec !important;
        color: #171717 !important;
        border-radius: 12px !important;
    }

    /* Inputs and uploader */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #2f2f2f !important;
        color: #ececec !important;
        border-color: #505050 !important;
    }

    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] section {
        background: #202020 !important;
        border-color: #505050 !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] button {
        color: #ececec !important;
        border-color: #505050 !important;
    }

    /* Buttons */
    .stButton > button {
        background: #2f2f2f !important;
        color: #ececec !important;
        border: 1px solid #454545 !important;
        border-radius: 10px !important;
        min-height: 2.5rem;
    }

    .stButton > button:hover {
        background: #3b3b3b !important;
        border-color: #666666 !important;
    }

    .stButton > button[kind="primary"] {
        background: #10a37f !important;
        color: white !important;
        border-color: #10a37f !important;
    }

    /* Evidence panels */
    [data-testid="stExpander"] {
        background: #262626 !important;
        border: 1px solid #414141 !important;
        border-radius: 12px !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        color: #ececec !important;
    }

    /* Dividers and captions */
    hr {
        border-color: #383838 !important;
    }

    [data-testid="stCaptionContainer"] {
        color: #a0a0a0 !important;
    }

    #MainMenu,
    footer {
        visibility: hidden;
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
        '<div class="brand">📚 Multimodal RAG</div>',
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
            
