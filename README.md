# RAG Chatbot

A retrieval-augmented generation chatbot that answers questions over uploaded documents. Built with a hybrid BM25 + semantic retrieval pipeline, streaming responses, and evaluated with RAGAS.

**Live demo:** [rag-chatbot-divya-s.streamlit.app]([https://rag-chatbot-divya-s.streamlit.app/])

---

## What it does

Upload a PDF or TXT file, ask questions in natural language, and get answers grounded in the document — with source chunk previews so you can verify every response.

Key behaviours:
- Refuses to answer from outside the document ("The document does not mention this.")
- Streams tokens live as they generate
- Filters retrieval by source document when multiple files are loaded
- Handles image-based PDFs via OCR fallback

---

## Architecture

```
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
```

### Document ingestion pipeline

```
Upload (PDF / TXT)
    │
    ├── PyMuPDF (fitz) → direct text extraction
    │       │
    │       └── if empty → pdf2image + pytesseract (OCR fallback unavailable)
    │
    ▼
RecursiveCharacterTextSplitter
    ├── warranty / contract / agreement → chunk_size=300, overlap=150
    └── everything else                → chunk_size=800, overlap=150
    │
    ▼
ChromaDB.upsert()          ← persisted to disk
BM25Okapi index rebuild    ← stored in session_state (RAM)
```

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| LLM | Anthropic Claude Sonnet | Streaming API, strong instruction following |
| Embeddings | all-mpnet-base-v2 | Strong sentence-level semantic similarity |
| Vector store | ChromaDB (persistent) | Local, no infra needed, simple API |
| Keyword search | rank_bm25 (BM25Okapi) | Catches exact identifiers semantic search misses |
| Retrieval merge | Reciprocal Rank Fusion | Scale-agnostic, rewards consistent signal |
| UI | Streamlit | Fast to build, st.write_stream for live tokens |
| OCR | pytesseract + pdf2image | Handles scanned/image-based PDFs |
| Text splitting | langchain-text-splitters | Adaptive chunking by document type |



## Evaluation (RAGAS)

Evaluated on 12 hand-written Q&A pairs covering the warranty document. Questions span specific lookups, summaries, out-of-scope queries (faithfulness tests), and reasoning questions.

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.787 | Are answers grounded in retrieved chunks? |
| Answer Relevancy | 0.692 | Do answers address the question asked? |
| Context Recall | 0.750 | Does retrieval fetch the right chunks? |

**Observations:**
- Faithfulness improved from 0.745 → 0.787 after tightening the system prompt to prohibit hallucination phrases
- Answer Relevancy is the weakest metric — verbose preambles ("Based on the provided context...") pull the score down; further system prompt tuning expected to push this toward 0.75+
- Context Recall plateau at 0.750 suggests chunk boundary splits in dense PDFs are the limiting factor; increasing chunk_size for warranty docs is the next lever

---


## Known limitations and next steps

- BM25 index is RAM-only — page refresh on Streamlit Cloud triggers a rebuild from ChromaDB (adds ~2s on first query after refresh)
- ChromaDB on Streamlit Cloud uses `/tmp` — documents must be re-uploaded each session; a cloud vector store (Pinecone, Weaviate) would fix this
- Answer Relevancy score (0.692) has room to improve — response formatting and preamble reduction
- OCR quality depends on scan resolution; low-quality scans produce poor chunks that no retrieval strategy can fully recover from
- No re-ranking step (cross-encoder) after hybrid retrieval — adding one would likely improve context precision

---
