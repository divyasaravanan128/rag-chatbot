import os
import html
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

import streamlit as st
from anthropic import Anthropic
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import dotenv_values
from rank_bm25 import BM25Okapi

import rag_core
from rag_core import MODEL, NO_MATCH_MESSAGE, SYSTEM_PROMPT, build_messages, tokenize


# ── Config ────────────────────────────────────────────────────────────────────

DOTENV = dotenv_values(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def get_setting(name: str, default):
    """
    First non-empty value from: environment, .env file, Streamlit secrets.
    .env is read directly because loading st.secrets copies every secret into
    os.environ, so an empty secret would otherwise blank a real .env value.
    """
    candidates = [os.getenv(name), DOTENV.get(name)]
    try:
        candidates.append(st.secrets.get(name))
    except Exception:  # no secrets file
        pass
    return next((v for v in candidates if v not in (None, "")), default)


ANTHROPIC_API_KEY = get_setting("ANTHROPIC_API_KEY", "")

# Usage caps: only questions that reach the Anthropic API count
MAX_QUESTIONS_PER_SESSION = int(get_setting("MAX_QUESTIONS_PER_SESSION", 20))
DAILY_QUESTION_CAP = int(get_setting("DAILY_QUESTION_CAP", 200))  # all users combined
MAX_HISTORY_TOKENS = 20_000  # conversation history sent per request

# Each browser session's documents live in memory only and are dropped
# after this long without activity
SESSION_TTL_SECONDS = int(get_setting("SESSION_TTL_SECONDS", 2 * 60 * 60))

# ── Shared resources (one per server process) ────────────────────────────────

@st.cache_resource
def get_embed_fn():
    return SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")


@st.cache_resource
def get_chroma_client():
    return chromadb.EphemeralClient()


@st.cache_resource
def get_session_registry() -> dict:
    return {"last_seen": {}, "lock": threading.Lock()}


@st.cache_resource
def get_daily_usage() -> dict:
    return {"date": None, "count": 0, "lock": threading.Lock()}


anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
chroma_client = get_chroma_client()

# ── Per-session document store ────────────────────────────────────────────────

if "collection_name" not in st.session_state:
    st.session_state.collection_name = f"session-{uuid.uuid4().hex}"


def touch_session_and_prune():
    """Mark this session active and delete collections of sessions idle past the TTL."""
    registry = get_session_registry()
    now = time.time()
    with registry["lock"]:
        registry["last_seen"][st.session_state.collection_name] = now
        for name, seen in list(registry["last_seen"].items()):
            if now - seen > SESSION_TTL_SECONDS:
                try:
                    chroma_client.delete_collection(name)
                except Exception:
                    pass
                del registry["last_seen"][name]


touch_session_and_prune()
collection = chroma_client.get_or_create_collection(
    name=st.session_state.collection_name,
    embedding_function=get_embed_fn(),
)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0

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


if collection.count() == 0:
    # New session, or this session's documents expired
    st.session_state.bm25_index = None
elif st.session_state.get("bm25_index") is None:
    build_bm25_index()


def load_documents(uploaded_files) -> int:
    """Uploads are written to a temp dir only for text extraction, then deleted."""
    added = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        for uf in uploaded_files:
            filename = os.path.basename(uf.name)
            filepath = os.path.join(tmp_dir, filename)
            with open(filepath, "wb") as f:
                f.write(uf.getbuffer())
            added += rag_core.add_document(collection, filepath, filename)

    build_bm25_index()
    return added


def clear_documents():
    chroma_client.delete_collection(st.session_state.collection_name)
    st.session_state.bm25_index = None
    st.session_state.messages = []


def questions_left() -> int:
    return max(0, MAX_QUESTIONS_PER_SESSION - st.session_state.questions_asked)


def reserve_api_call() -> str | None:
    """Count one question against the usage caps. Returns a message if a cap is reached."""
    if questions_left() == 0:
        return (
            f"This session has reached its limit of {MAX_QUESTIONS_PER_SESSION} questions."
        )
    usage = get_daily_usage()
    with usage["lock"]:
        today = datetime.now(timezone.utc).date()
        if usage["date"] != today:
            usage["date"], usage["count"] = today, 0
        if usage["count"] >= DAILY_QUESTION_CAP:
            return "The app has reached its daily question limit. Please try again tomorrow."
        usage["count"] += 1
    st.session_state.questions_asked += 1
    return None


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

def trim_history(messages: list, max_tokens: int = MAX_HISTORY_TOKENS) -> list:
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
        st.rerun()

    st.caption(
        "Documents are private to this browser session and are not saved. "
        "They're removed when the session ends or after 2 hours idle."
    )

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

    if all_sources and st.button("Remove all documents"):
        clear_documents()
        st.rerun()

    st.divider()
    st.caption(f"Questions left this session: {questions_left()} of {MAX_QUESTIONS_PER_SESSION}")

# ── Main chat ─────────────────────────────────────────────────────────────────

st.title("RAG Chatbot")

if collection.count() == 0:
    st.info("Upload documents in the sidebar to get started.")
    st.stop()

# Replay conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New user input
if prompt := st.chat_input(
    "Ask a question about your documents...",
    disabled=questions_left() == 0,
):

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
    limit_message = reserve_api_call() if context_chunks else None

    with st.chat_message("assistant"):
        if limit_message:
            full_response = limit_message
            sources_found = []
            st.markdown(full_response)
        elif context_chunks:
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

    if questions_left() == 0:
        st.info(f"That was the last of this session's {MAX_QUESTIONS_PER_SESSION} questions.")

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