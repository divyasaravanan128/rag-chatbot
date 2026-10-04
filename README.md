# RAG Chatbot

A retrieval-augmented generation chatbot that answers questions over uploaded documents. Hybrid BM25 + semantic retrieval, streaming responses, evaluated with RAGAS.

**Live demo:** [rag-chatbot-divya-s.streamlit.app](https://rag-chatbot-divya-s.streamlit.app)

---

## What it does

Upload PDF or TXT files, ask questions, and get answers grounded in the documents with source chunk previews.

- Skips questions the documents don't cover: if no chunk is semantically close enough, it says so without calling the model
- Answers only from retrieved chunks; replies "The document does not mention this." when the answer isn't there
- Streams tokens live as they generate
- Keeps each user's documents private: every browser session has its own in-memory store, nothing is written to disk, and a session's documents are deleted after 2 hours idle (or with "Remove all documents")
- Caps API usage: 20 questions per session and 200 per day across all users (both configurable); off-topic questions stopped by the relevance check don't count
- Remembers the conversation for follow-up questions (oldest turns trimmed past ~20k tokens)
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
ChromaDB (semantic) ── best match too far? ──► "couldn't find relevant information"
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

**Relevance threshold:** if the closest chunk's cosine distance is above 0.65, no chunks are returned and the model isn't called. Calibrated on `Warranty.pdf`: answerable questions scored at most 0.58, off-topic ones at least 0.68. Questions close to the topic but not answered by the document (e.g. "warranty on an iPhone") still pass the threshold and are handled by the model's "does not mention" reply.

**Shared core:** retrieval, the threshold and the prompt live in `rag_core.py`, used by both the app and the eval, so the eval measures exactly what the app does.

**Prompting:** the system prompt holds only the rules. Retrieved chunks go in the latest user message inside `<document>` tags, so instructions hidden in an uploaded file don't get system-level authority.

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude Sonnet |
| Embeddings | all-mpnet-base-v2 |
| Vector store | ChromaDB (in-memory, one collection per session) |
| Keyword search | rank_bm25 (BM25Okapi) |
| Retrieval merge | Reciprocal Rank Fusion |
| OCR | Tesseract + Poppler (pdf2image) |
| UI | Streamlit |
| Text splitting | langchain-text-splitters |

---

## Evaluation (RAGAS — 12 Q&A pairs)

| Metric | Score |
|---|---|
| Faithfulness | 0.982 |
| Answer Relevancy | 0.781 |
| Context Recall | 0.917 |

Scores vary a little between runs (faithfulness was 0.932 on the previous run), and in this run the judge couldn't score faithfulness for 1 of the 12 answers. Answer Relevancy includes two out-of-scope questions (price, return policy) where the correct reply is "The document does not mention this."; RAGAS always scores that reply 0. Across the 10 answerable questions it is 0.938.

Run with `venv\Scripts\python eval\run_eval.py` from the repo root, with `Warranty.pdf` in `docs/`. The eval loads it with the app's own ingestion code (about 1.5 minutes including OCR) and writes per-question scores to `eval/results.csv`.

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

Optional settings (in `.env` locally or Streamlit secrets on Cloud):

| Setting | Default | Meaning |
|---|---|---|
| `MAX_QUESTIONS_PER_SESSION` | 20 | Questions one browser session can send to the API |
| `DAILY_QUESTION_CAP` | 200 | Questions per day (UTC) across all users |
| `SESSION_TTL_SECONDS` | 7200 | Idle time before a session's documents are deleted |

For a hard spending limit, also set a monthly spend limit for the API key's workspace in the Anthropic Console. The in-app caps reset if the app restarts.

---

## Known limitations

- The relevance threshold (0.65) was calibrated on one document and a small question set; other document types may need a different value
- OCR flattens tables, so table answers from scanned PDFs can mix up columns
- Documents aren't saved: refreshing the page starts a new session, so files must be uploaded again
- Usage counters live in server memory: they reset when the app restarts, and the per-session cap resets on page refresh (the daily cap is the real limit)
- All sessions share one server's memory, so many large uploads at once could exhaust RAM on Streamlit Cloud
