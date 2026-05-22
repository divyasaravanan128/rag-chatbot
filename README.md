# RAG Chatbot

A retrieval-augmented generation chatbot that answers questions over uploaded documents. Hybrid BM25 + semantic retrieval, streaming responses, evaluated with RAGAS.

**Live demo:** [rag-chatbot-divya-s.streamlit.app](https://rag-chatbot-divya-s.streamlit.app)

---

## What it does

Upload a PDF or TXT file, ask questions, and get answers grounded in the document with source chunk previews.

- Refuses to answer outside the document
- Streams tokens live as they generate
- Filters retrieval by source when multiple files are loaded
- OCR supported locally (not available on Streamlit Cloud)

---

## Architecture

```
User query
    │
    ▼
BM25 (keyword)  +  ChromaDB (semantic)
         └──── RRF merge (k=60) ────┘
    │
    ▼
Top-k chunks → Anthropic API (Claude Sonnet)
    │
    ▼
st.write_stream() → live token rendering
```

**Ingestion:** PyMuPDF direct extraction → OCR fallback (local) → adaptive chunking → ChromaDB upsert + BM25 rebuild

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude Sonnet |
| Embeddings | all-mpnet-base-v2 |
| Vector store | ChromaDB (persistent) |
| Keyword search | rank_bm25 (BM25Okapi) |
| Retrieval merge | Reciprocal Rank Fusion |
| UI | Streamlit |
| Text splitting | langchain-text-splitters |

---

## Evaluation (RAGAS — 12 Q&A pairs)

| Metric | Score |
|---|---|
| Faithfulness | 0.787 |
| Answer Relevancy | 0.692 |
| Context Recall | 0.750 |

---

## Setup (local)

```bash
git clone https://github.com/divyasaravanan128/rag-chatbot
cd rag-chatbot
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
# Add ANTHROPIC_API_KEY to .env
streamlit run app.py
```

Windows: install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) for OCR support.

---

## Known limitations

- OCR not available on Streamlit Cloud (Tesseract/Poppler not supported in hosted environment)
- BM25 index is RAM-only — rebuilds from ChromaDB on page refresh
- ChromaDB uses ephemeral `/tmp` on Streamlit Cloud — documents must be re-uploaded each session
