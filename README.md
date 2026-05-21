# RAG Chatbot

A retrieval-augmented generation chatbot that answers questions over uploaded documents. Built with a hybrid BM25 + semantic retrieval pipeline, streaming responses, and evaluated with RAGAS.

**Live demo:** [your-app.streamlit.app](https://your-app.streamlit.app)

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
    │       └── if empty → pdf2image + pytesseract (OCR fallback)
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

---

## Design decisions

**Why hybrid search?**

Pure semantic search fails on exact identifiers — clause numbers, product codes, proper nouns. A query like "What does Annexure 1C say?" may return topically similar chunks while missing the chunk that literally contains "Annexure 1C". BM25 catches these exact matches via IDF-weighted term frequency. RRF merges both ranked lists without needing to reconcile incompatible score scales.

**Why RRF over weighted score combination?**

BM25 scores are unbounded floats; ChromaDB returns cosine distances (0–2). Adding or normalising them introduces arbitrary weighting assumptions. RRF discards raw scores entirely and uses only rank positions — a chunk that ranks well in both retrievers wins, regardless of absolute scores.

**Why adaptive chunk sizes?**

Legal and warranty documents are clause-dense. An 800-char chunk merges unrelated clauses, causing retrieval to return a chunk that's mostly irrelevant to the query. 300-char chunks keep individual clauses isolated. Narrative documents benefit from larger chunks that preserve context across sentences.

**Why trim at the call site, not inside stream_response()?**

Single responsibility. `stream_response()` is a pure generator — it receives history and chunks, yields tokens. Conversation state management (what to send, when to trim) belongs at the call site in the Streamlit chat loop. This makes both functions independently testable.

**Why rebuild BM25 on every ingest, not append?**

BM25's IDF scores depend on word frequency across the entire corpus. Adding new chunks changes these frequencies — a word that was rare may become common. Partial updates would produce incorrect scores. Full rebuild from all ChromaDB chunks is the only correct approach.

---

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

## Setup

**Prerequisites:** Python 3.11+, Tesseract OCR, Poppler

```bash
# Clone
git clone https://github.com/divyasaravanan128/rag-chatbot
cd rag-chatbot

# Environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Dependencies
pip install -r requirements.txt

# Environment variables
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

**Windows system dependencies:**
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Poppler: https://github.com/oschwartz10612/poppler-windows/releases

**Run:**
```bash
streamlit run app.py
```

---

## Project structure

```
rag-chatbot/
├── app.py                  # Main Streamlit app
├── reingest.py             # Utility — clears ChromaDB for fresh ingest
├── requirements.txt
├── packages.txt            # System deps for Streamlit Cloud
├── .env.example
├── eval/
│   ├── questions.py        # 12 Q&A pairs with ground truth
│   ├── run_eval.py         # RAGAS evaluation runner
│   └── results.csv         # Scores per question (git-ignored)
└── docs/                   # Uploaded documents (git-ignored)
```

---

## Known limitations and next steps

- BM25 index is RAM-only — page refresh on Streamlit Cloud triggers a rebuild from ChromaDB (adds ~2s on first query after refresh)
- ChromaDB on Streamlit Cloud uses `/tmp` — documents must be re-uploaded each session; a cloud vector store (Pinecone, Weaviate) would fix this
- Answer Relevancy score (0.692) has room to improve — response formatting and preamble reduction
- OCR quality depends on scan resolution; low-quality scans produce poor chunks that no retrieval strategy can fully recover from
- No re-ranking step (cross-encoder) after hybrid retrieval — adding one would likely improve context precision

---

## Built as part of an 8-week AI Engineering sprint

Weeks 1–4: environment setup, embeddings, ChromaDB retrieval, Streamlit chat UI
Week 5: streaming, context window management
Week 6: hybrid search (BM25 + RRF)
Week 7: RAGAS evaluation
Week 8: deployment
