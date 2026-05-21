import os
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

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
CHROMA_PATH = "/tmp/chroma_db" if os.path.exists("/tmp") else "./chroma_db"
POPPLER_PATH = r"C:\poppler\Library\bin" if os.name == "nt" else None

try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    pass

THRESHOLD = 1.5
N_RESULTS = 5
CHROMA_PATH = "./chroma_db"
DOCS_PATH = "./docs"

# ── Clients ──────────────────────────────────────────────────────────────────

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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

if "bm25_index" not in st.session_state and collection.count() > 0:
    from rank_bm25 import BM25Okapi
    all_data = collection.get(include=["documents", "metadatas"])
    tokenized = [doc.lower().split() for doc in all_data["documents"]]
    st.session_state.bm25_index = BM25Okapi(tokenized)
    st.session_state.bm25_chunks = all_data["documents"]
    st.session_state.bm25_metas  = all_data["metadatas"]
# ── Helpers ───────────────────────────────────────────────────────────────────

def get_splitter(filename: str) -> RecursiveCharacterTextSplitter:
    name = filename.lower()
    if any(k in name for k in ("warranty", "contract", "agreement", "terms")):
        return RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=150)
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)


def extract_text(filepath: str, filename: str) -> str:
    if filename.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    # PDF: try direct text extraction first
    doc = fitz.open(filepath)
    text = "".join(page.get_text() for page in doc)
    if text.strip():
        return text

    # OCR fallback for image-based PDFs
    images = convert_from_path(filepath, poppler_path=POPPLER_PATH if POPPLER_PATH else None)
    return "\n".join(pytesseract.image_to_string(img) for img in images)


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

        strategy = "precise" if "warranty" in uf.name.lower() else "narrative"
        ids = [f"{uf.name}__chunk{i}" for i in range(len(chunks))]
        metas = [{"source": uf.name, "chunk_strategy": strategy} for _ in chunks]

        collection.upsert(documents=chunks, ids=ids, metadatas=metas)
        added += len(chunks)

        # Build BM25 index over ALL chunks in the collection
        all_data = collection.get(include=["documents", "metadatas"])
        all_chunks = all_data["documents"]
        all_metas = all_data["metadatas"]

        tokenized = [doc.lower().split() for doc in all_chunks]
        st.session_state.bm25_index = BM25Okapi(tokenized)
        st.session_state.bm25_chunks = all_chunks
        st.session_state.bm25_metas = all_metas

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


def build_where_filter(selected: list[str], all_sources: list[str]):
    if not selected or set(selected) == set(all_sources):
        return None
    if len(selected) == 1:
        return {"source": selected[0]}
    return {"source": {"$in": selected}}

def hybrid_search(
    query: str,
    selected_sources: list[str],
    all_sources: list[str],
    n_results: int = N_RESULTS,
    k: int = 60,
) -> tuple[list[str], list[dict]]:
    """
    Combines BM25 (keyword) + ChromaDB (semantic) via Reciprocal Rank Fusion.
    Returns (context_chunks, sources_meta) — same shape as your existing retrieval.
    """

    # ── BM25 retrieval ────────────────────────────────────────────────────────
    tokenized_query = query.lower().split()
    bm25_scores = st.session_state.bm25_index.get_scores(tokenized_query)

    # Filter by selected sources before ranking
    all_chunks = st.session_state.bm25_chunks
    all_metas  = st.session_state.bm25_metas

    bm25_ranked = sorted(
        range(len(all_chunks)),
        key=lambda i: (
            bm25_scores[i]
            if (not selected_sources or all_metas[i].get("source") in selected_sources)
            else -1
        ),
        reverse=True,
    )[:n_results * 2]  # grab 2× for RRF headroom

    # ── ChromaDB retrieval ────────────────────────────────────────────────────
    where_filter = build_where_filter(selected_sources, all_sources)
    query_kwargs = dict(query_texts=[query], n_results=n_results * 2)
    if where_filter:
        query_kwargs["where"] = where_filter

    chroma_results = collection.query(**query_kwargs)
    chroma_docs      = chroma_results["documents"][0]
    chroma_distances = chroma_results["distances"][0]
    chroma_metas     = chroma_results["metadatas"][0]

    # Map chroma chunk text → its rank position
    chroma_rank = {doc: rank for rank, doc in enumerate(chroma_docs)}

    # ── RRF merge ─────────────────────────────────────────────────────────────
    # Collect all candidate chunks from both retrievers
    candidates = {}  # chunk_text → {"meta": ..., "rrf": 0.0}

    for rank, idx in enumerate(bm25_ranked):
        text = all_chunks[idx]
        if text not in candidates:
            candidates[text] = {"meta": all_metas[idx], "rrf": 0.0}
        candidates[text]["rrf"] += 1 / (k + rank)

    for rank, (doc, meta) in enumerate(zip(chroma_docs, chroma_metas)):
        if doc not in candidates:
            candidates[doc] = {"meta": meta, "rrf": 0.0}
        candidates[doc]["rrf"] += 1 / (k + rank)

    # Sort by RRF score, take top n_results
    ranked = sorted(candidates.items(), key=lambda x: x[1]["rrf"], reverse=True)
    top = ranked[:n_results]

    context_chunks = [text for text, _ in top]
    sources_meta   = [data["meta"] for _, data in top]

    return context_chunks, sources_meta


def stream_response(context_chunks: list[str], conversation_history: list[dict]):
    """
    Generator — yields text chunks from the Anthropic streaming API.
    st.write_stream() collects them and returns the full string when done.
    """
    system_prompt = (
    "You are a precise document assistant. Answer only from the provided context.\n"
    "Answer directly — do not start with 'Based on the provided context' or similar phrases.\n"
    "Be concise. No markdown formatting for simple factual answers.\n"
    "If the context doesn't contain the answer, say exactly: "
    "'The document does not mention this.'\n\n"
    "Context:\n" + "\n\n".join(context_chunks)
    )

    with anthropic_client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=conversation_history,
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
        # fallback to pure semantic if BM25 not built yet
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
                stream_response(context_chunks, api_history)
            )
        else:
            full_response = (
                "I couldn't find relevant information in the loaded documents "
                "for that question. Try rephrasing or loading more documents."
            )
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Sources
    if sources_found:
        with st.expander("Sources"):
            seen = set()
            for i, (chunk, meta) in enumerate(zip(context_chunks, sources_found)):
                source = meta.get("source", "?")
                # Build a unique key from source + chunk preview
                preview = chunk[:80].strip().replace("\n", " ")
                key = f"{source}_{i}"
                if key not in seen:
                    seen.add(key)
                    st.caption(f"**{source}** · chunk {i+1}")
                    st.markdown(
                        f"<div style='font-size:12px;color:gray;padding:4px 8px;"
                        f"border-left:2px solid #444;margin-bottom:6px'>{preview}…</div>",
                        unsafe_allow_html=True
                    )