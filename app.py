import os
import html
import streamlit as st
from anthropic import Anthropic
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi

import rag_core
from rag_core import MODEL, NO_MATCH_MESSAGE, SYSTEM_PROMPT, build_messages, tokenize

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
CHROMA_PATH = "/tmp/chroma_db" if os.path.exists("/tmp") else "./chroma_db"
POPPLER_PATH = r"C:\poppler\Library\bin" if os.name == "nt" else None

# On Linux/cloud, tesseract is on PATH (installed via packages.txt)
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DOCS_PATH = "./docs"

# ── Clients ──────────────────────────────────────────────────────────────────

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

if "embed_fn" not in st.session_state:
    st.session_state.embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2"
    )

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=st.session_state.embed_fn,
)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "loaded_docs" not in st.session_state:
    count = collection.count()
    st.session_state.loaded_docs = count > 0

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_bm25_index():
    """Build the BM25 index over ALL chunks in the collection."""
    all_data = collection.get(include=["documents", "metadatas"])
    if not all_data["documents"]:
        st.session_state.bm25_index = None
        return
    st.session_state.bm25_index = BM25Okapi([tokenize(doc) for doc in all_data["documents"]])
    st.session_state.bm25_chunks = all_data["documents"]
    st.session_state.bm25_metas = all_data["metadatas"]


if "bm25_index" not in st.session_state and collection.count() > 0:
    build_bm25_index()


def is_precise_doc(filename: str) -> bool:
    return any(k in filename.lower() for k in ("warranty", "contract", "agreement", "terms"))


def get_splitter(filename: str) -> RecursiveCharacterTextSplitter:
    if is_precise_doc(filename):
        return RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=150)
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)


def extract_text(filepath: str, filename: str) -> str:
    if filename.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    # PDF: extract text per page; OCR only the pages with no text layer (scanned)
    pages = []
    with fitz.open(filepath) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text.strip():
                images = convert_from_path(
                    filepath,
                    first_page=page_num,
                    last_page=page_num,
                    poppler_path=POPPLER_PATH,
                )
                text = "\n".join(pytesseract.image_to_string(img) for img in images)
            pages.append(text)
    return "\n".join(pages)


def load_documents(uploaded_files) -> int:
    added = 0
    for uf in uploaded_files:
        filepath = os.path.join(DOCS_PATH, uf.name)
        os.makedirs(DOCS_PATH, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(uf.getbuffer())

        text = extract_text(filepath, uf.name)
        splitter = get_splitter(uf.name)
        chunks = splitter.split_text(text)

        strategy = "precise" if is_precise_doc(uf.name) else "narrative"
        ids = [f"{uf.name}__chunk{i}" for i in range(len(chunks))]
        metas = [{"source": uf.name, "chunk_strategy": strategy} for _ in chunks]

        # Drop chunks from any previous version of this file before re-adding
        collection.delete(where={"source": uf.name})
        if chunks:
            collection.upsert(documents=chunks, ids=ids, metadatas=metas)
        added += len(chunks)

    build_bm25_index()
    return added


def get_sources() -> list[str]:
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    seen = set()
    sources = []
    for m in results["metadatas"]:
        s = m.get("source", "")
        if s and s not in seen:
            seen.add(s)
            sources.append(s)
    return sorted(sources)


def hybrid_search(
    query: str,
    selected_sources: list[str],
    all_sources: list[str],
) -> tuple[list[str], list[dict]]:
    return rag_core.hybrid_search(
        query,
        collection,
        st.session_state.bm25_index,
        st.session_state.bm25_chunks,
        st.session_state.bm25_metas,
        selected_sources,
        all_sources,
    )


def stream_response(
    context_chunks: list[str],
    sources_meta: list[dict],
    conversation_history: list[dict],
):
    """
    Generator — yields text chunks from the Anthropic streaming API.
    st.write_stream() collects them and returns the full string when done.
    """
    with anthropic_client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=build_messages(context_chunks, sources_meta, conversation_history),
    ) as stream:
        for text in stream.text_stream:
            yield text

def trim_history(messages: list, max_tokens: int = 100_000) -> list:
    """
    Drop oldest user+assistant pairs until the history fits within max_tokens.
    Never drops below 1 pair (the most recent exchange).
    Token estimate: total chars / 4 (rough but fast).
    """
    def token_estimate(msgs):
        return sum(len(m["content"]) for m in msgs) // 4

    while token_estimate(messages) > max_tokens and len(messages) >= 4:
        # pop index 0 twice — removes oldest user turn, then oldest assistant turn
        messages.pop(0)
        messages.pop(0)

    return messages


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Load documents", disabled=not uploaded_files):
        with st.spinner("Processing..."):
            n = load_documents(uploaded_files)
        st.success(f"Added {n} chunks.")
        st.session_state.loaded_docs = True
        st.rerun()

    st.divider()

    all_sources = get_sources()
    selected_sources = []

    if len(all_sources) >= 2:
        selected_sources = st.multiselect(
            "Filter by document",
            options=all_sources,
            default=all_sources,
        )
    elif len(all_sources) == 1:
        st.caption(f"Loaded: {all_sources[0]}")
        selected_sources = all_sources

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Main chat ─────────────────────────────────────────────────────────────────

st.title("RAG Chatbot")

if not st.session_state.loaded_docs:
    st.info("Upload documents in the sidebar to get started.")
    st.stop()

# Replay conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New user input
if prompt := st.chat_input("Ask a question about your documents..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve
    if st.session_state.get("bm25_index") is not None:
        context_chunks, sources_found = hybrid_search(
            prompt, selected_sources, all_sources
        )
    else:
        # No BM25 index means the collection is empty, so there is nothing to retrieve
        context_chunks, sources_found = [], []

    # Stream response
    with st.chat_message("assistant"):
        if context_chunks:
            api_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            api_history = trim_history(api_history)

            full_response = st.write_stream(
                stream_response(context_chunks, sources_found, api_history)
            )
        else:
            # Relevance gate in hybrid_search found nothing close enough
            full_response = NO_MATCH_MESSAGE
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Sources
    if sources_found:
        with st.expander("Sources"):
            for i, (chunk, meta) in enumerate(zip(context_chunks, sources_found)):
                source = meta.get("source", "?")
                preview = html.escape(chunk[:80].strip().replace("\n", " "))
                st.caption(f"**{source}** · chunk {i+1}")
                st.markdown(
                    f"<div style='font-size:12px;color:gray;padding:4px 8px;"
                    f"border-left:2px solid #444;margin-bottom:6px'>{preview}…</div>",
                    unsafe_allow_html=True
                )