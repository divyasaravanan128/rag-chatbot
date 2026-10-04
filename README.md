# RAG Chatbot

A retrieval-augmented generation chatbot that answers questions over uploaded documents. Hybrid BM25 + semantic retrieval, streaming responses, evaluated with RAGAS.

**Live demo:** [rag-chatbot-divya-s.streamlit.app](https://rag-chatbot-divya-s.streamlit.app)

---

## What it does

Upload PDF or TXT files, ask questions, and get answers grounded in the documents with source chunk previews.

- Answers only from retrieved chunks; replies "The document does not mention this." when the answer isn't there
- Streams tokens live as they generate
- Remembers the conversation for follow-up questions (oldest turns trimmed past ~100k tokens)
- Filters retrieval by document when multiple files are loaded
- Reads scanned PDFs: OCR runs per page, so scanned pages inside otherwise digital PDFs are captured too
- Re-uploading a file replaces its previous version
- Treats document text as data, not instructions, to resist prompt injection from uploaded files

---

## Architecture

```
User query
    │
    ▼
BM25 (keyword)  +  ChromaDB (semantic)
  filtered to selected documents
         └──── RRF merge (k=60) ────┘
    │
    ▼
Top-5 chunks → Claude Sonnet (temperature 0)
    │
    ▼
st.write_stream() → live token rendering
```

**Ingestion:** PyMuPDF text extraction → OCR for pages with no text layer → adaptive chunking → replace old chunks for that file → ChromaDB upsert → BM25 rebuild (once per upload batch)

**Adaptive chunking:** files with `warranty`, `contract`, `agreement` or `terms` in the name use 300-character chunks for precise clause lookup; everything else uses 800. Both use 150-character overlap.

**BM25 details:** query and chunks share one tokenizer (lowercase words, punctuation stripped), and only chunks with a real keyword match get RRF credit.

**Prompting:** the system prompt holds only the rules. Retrieved chunks go in the latest user message inside `<document>` tags, so instructions hidden in an uploaded file don't get system-level authority.

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude Sonnet |
| Embeddings | all-mpnet-base-v2 |
| Vector store | ChromaDB (persistent) |
| Keyword search | rank_bm25 (BM25Okapi) |
| Retrieval merge | Reciprocal Rank Fusion |
| OCR | Tesseract + Poppler (pdf2image) |
| UI | Streamlit |
| Text splitting | langchain-text-splitters |

---

## Evaluation (RAGAS — 12 Q&A pairs)

| Metric | Score |
|---|---|
| Faithfulness | 0.787 |
| Answer Relevancy | 0.692 |
| Context Recall | 0.750 |

These scores were measured before the October 2026 retrieval fixes (tokenizer, keyword-match filter). Re-run with `python eval/run_eval.py` from the repo root after loading the eval documents.

---

## Setup (local)

```bash
git clone https://github.com/divyasaravanan128/rag-chatbot
cd rag-chatbot
python -m venv venv && venv\Scripts\activate   # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
# Add ANTHROPIC_API_KEY to .env
streamlit run app.py
```

OCR needs Tesseract and Poppler:

- **Windows:** install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) to `C:\Program Files\Tesseract-OCR` and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) to `C:\poppler`
- **Linux:** `sudo apt install tesseract-ocr poppler-utils`
- **Streamlit Cloud:** installed automatically from `packages.txt`

On Streamlit Cloud, set `ANTHROPIC_API_KEY` in the app's secrets.

---

## Known limitations

- No relevance threshold: retrieval always returns the top 5 chunks, so the model, not the retriever, decides when the documents don't answer the question
- BM25 index lives in session memory and is rebuilt from ChromaDB when a new session starts
- On Streamlit Cloud, ChromaDB is stored in `/tmp`: documents are shared by everyone using the app and are lost when the app restarts or sleeps
