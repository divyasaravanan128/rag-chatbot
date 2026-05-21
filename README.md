# RAG Chatbot

A retrieval-augmented generation chatbot that answers questions over uploaded documents. Built with a hybrid BM25 + semantic retrieval pipeline, streaming responses, and evaluated with RAGAS.

Live demo: rag-chatbot-divya-s.streamlit.app

---

# What it does

Upload a PDF or TXT file, ask questions in natural language, and get answers grounded in the document — with source chunk previews so responses remain verifiable.

Key behaviours:

- Refuses to answer from outside the document ("The document does not mention this.")
- Streams tokens live as they generate
- Filters retrieval by source document when multiple files are loaded
- Supports hybrid retrieval (BM25 + semantic search)
- Supports OCR locally for scanned/image-based PDFs when Tesseract + Poppler are installed

---

# Architecture

User query
    │
    ▼
┌─────────────────────────────────────────────┐
│              Hybrid Retrieval               │
│                                             │
│  BM25 (rank_bm25)    ChromaDB (semantic)    │
│  keyword scoring  +  cosine similarity      │
│         │                   │               │
│         └──── RRF merge ────┘               │
│              (k=60, rank-based)             │
└─────────────────────────────────────────────┘
    │
    ▼
Top-k chunks (N_RESULTS=5)
    │
    ▼
┌─────────────────────────────────────────────┐
│           Anthropic API (Claude)            │
│  system prompt: context + grounding rules   │
│  messages: trimmed conversation history     │
│  streaming: anthropic_client.messages.stream│
└─────────────────────────────────────────────┘
    │
    ▼
st.write_stream() → live token rendering

---

# Document ingestion pipeline

Upload (PDF / TXT)
    │
    ├── PyMuPDF (fitz) → direct text extraction
    │       │
    │       └── if insufficient text:
    │              pdf2image → pytesseract OCR fallback
    │
    ▼
RecursiveCharacterTextSplitter
    ├── warranty / contract / agreement → chunk_size=300, overlap=150
    └── everything else                → chunk_size=800, overlap=150
    │
    ▼
ChromaDB.upsert()          ← persisted to disk
BM25Okapi index rebuild    ← stored in session_state (RAM)

---

# Stack

| Layer | Technology | Why |
|---|---|---|
| LLM | Anthropic Claude Sonnet | Streaming API, strong instruction following |
| Embeddings | all-mpnet-base-v2 | Strong sentence-level semantic similarity |
| Vector store | ChromaDB (persistent) | Local, simple API, no external infra |
| Keyword search | rank_bm25 (BM25Okapi) | Catches exact identifiers semantic search misses |
| Retrieval merge | Reciprocal Rank Fusion | Rewards agreement across retrieval methods |
| UI | Streamlit | Fast iteration and streaming support |
| OCR | pytesseract + pdf2image | Local OCR support for scanned PDFs |
| Text splitting | langchain-text-splitters | Adaptive chunking by document type |

---

# Evaluation (RAGAS)

Evaluated on 12 hand-written Q&A pairs covering the warranty document.

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.787 | Are answers grounded in retrieved chunks? |
| Answer Relevancy | 0.692 | Do answers address the question asked? |
| Context Recall | 0.750 | Does retrieval fetch the right chunks? |

## Observations

- Faithfulness improved from 0.745 → 0.787 after tightening grounding instructions in the system prompt
- Answer Relevancy remains the weakest metric; response verbosity and unnecessary preambles reduce scores
- Context Recall plateau suggests chunk-boundary fragmentation in dense PDFs

---

# Known limitations

- Streamlit Cloud deployment currently does NOT support OCR because Tesseract and Poppler system binaries are unavailable in the hosted environment
- Scanned/image-only PDFs therefore cannot be processed in the live demo
- OCR functionality works locally when Tesseract OCR and Poppler are installed and configured
- BM25 index is RAM-only and rebuilds from ChromaDB after refresh/redeploy
- ChromaDB persistence on Streamlit Cloud uses ephemeral storage (`/tmp`)
- No cross-encoder reranking step after hybrid retrieval
- Retrieval quality for dense legal PDFs is sensitive to chunking strategy

---

# Future improvements

- Dockerized deployment with system-level OCR dependencies
- Cross-encoder reranking for higher context precision
- Persistent hosted vector database
- Token-aware conversation summarization
- Incremental BM25 indexing instead of full rebuilds
