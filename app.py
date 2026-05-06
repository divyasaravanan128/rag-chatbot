import os
import streamlit as st
from dotenv import load_dotenv
import chromadb

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Windows path fixes — must be before any PDF/OCR calls
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\Library\bin"

# Load API key
load_dotenv()

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")
st.write("Ask me anything about your documents.")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection_ready" not in st.session_state:
    st.session_state.collection_ready = False

# --- Initialize ChromaDB via session_state ---
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()
    st.session_state.collection = st.session_state.chroma_client.get_or_create_collection(name="my_docs")

collection = st.session_state.collection


# --- PDF loader with OCR fallback ---
def load_pdf(filepath):
    """
    Try PyPDFLoader first (text-based PDFs).
    If no text found, fall back to OCR via pytesseract + pdf2image.
    Handles scanned/image-based PDFs like the HomeLane warranty.
    """
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    total_text = "".join(p.page_content for p in pages)

    if len(total_text.strip()) > 50:
        return pages  # Normal text PDF — done

    # OCR fallback
    from pdf2image import convert_from_path
    st.sidebar.info("📷 Image-based PDF detected — running OCR. Please wait...")
    images = convert_from_path(filepath, dpi=200, poppler_path=POPPLER_PATH)
    docs = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": os.path.basename(filepath), "page": i + 1}
            ))
    return docs


# --- Load docs from docs/ folder ---
def load_docs_folder():
    docs_path = "docs"
    all_chunks = []
    all_ids = []
    chunk_id = 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    if not os.path.exists(docs_path):
        return [], []

    for filename in os.listdir(docs_path):
        filepath = os.path.join(docs_path, filename)

        if filename.endswith(".pdf"):
            pages = load_pdf(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
            pages = loader.load()
        else:
            continue

        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk.page_content,
                "source": filename
            })
            all_ids.append(f"doc_{chunk_id}")
            chunk_id += 1

    return all_chunks, all_ids


# Load docs/ folder into ChromaDB once per session
if not st.session_state.collection_ready:
    chunks, ids = load_docs_folder()
    if chunks and collection.count() == 0:
        collection.add(
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"]} for c in chunks],
            ids=ids
        )
        st.session_state.collection_ready = True
    elif collection.count() > 0:
        st.session_state.collection_ready = True


# --- Sidebar: file uploader ---
st.sidebar.title("📂 Upload Documents")
st.sidebar.write("Upload PDF or TXT files to add to the chatbot's knowledge.")

uploaded_files = st.sidebar.file_uploader(
    "Choose files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("📥 Load Documents"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        new_chunks = []
        new_ids = []
        start_id = collection.count()

        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            if uploaded_file.name.endswith(".pdf"):
                pages = load_pdf(temp_path)
            else:
                loader = TextLoader(temp_path)
                pages = loader.load()

            os.remove(temp_path)

            chunks_from_file = splitter.split_documents(pages)
            for i, chunk in enumerate(chunks_from_file):
                new_chunks.append({
                    "text": chunk.page_content,
                    "source": uploaded_file.name
                })
                new_ids.append(f"upload_{start_id + i}")

        if new_chunks:
            collection.add(
                documents=[c["text"] for c in new_chunks],
                metadatas=[{"source": c["source"]} for c in new_chunks],
                ids=new_ids
            )
            st.sidebar.success(f"✅ Added {len(new_chunks)} chunks from {len(uploaded_files)} file(s).")
            st.session_state.collection_ready = True
        else:
            st.sidebar.error("❌ No text extracted. Check poppler path or file quality.")


# --- Initialize Claude ---
llm = ChatAnthropic(model="claude-sonnet-4-5")


# --- Display chat history ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 Sources used"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['file']}**")
                    st.caption(src["snippet"])


# --- Chat input ---
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if not st.session_state.collection_ready or collection.count() == 0:
        answer = "No documents loaded yet. Please upload a file and click **Load Documents** first."
        sources = []
    else:
        results = collection.query(
            query_texts=[user_input],
            n_results=5
        )

        docs = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        THRESHOLD = 1.5
        best_distance = distances[0]

        if best_distance > THRESHOLD:
            answer = "I don't have enough relevant information in the documents to answer this."
            sources = []
        else:
            relevant = [
                (doc, meta, dist)
                for doc, meta, dist in zip(docs, metadatas, distances)
                if dist <= THRESHOLD
            ]

            context = "\n\n".join([r[0] for r in relevant])

            history_text = ""
            for msg in st.session_state.messages[:-1]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

            prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say you don't know.

Context:
{context}

Conversation so far:
{history_text}
User: {user_input}
Assistant:"""

            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content

            sources = []
            seen = set()
            for doc, meta, dist in relevant:
                fname = meta.get("source", "unknown")
                if fname not in seen:
                    seen.add(fname)
                    sources.append({
                        "file": fname,
                        "snippet": doc[:200] + "..." if len(doc) > 200 else doc
                    })

    st.chat_message("assistant").write(answer)
    if sources:
        with st.expander("📄 Sources used"):
            for src in sources:
                st.markdown(f"**{src['file']}**")
                st.caption(src["snippet"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })