# RAG Chatbot

A retrieval-augmented generation chatbot that answers questions over uploaded documents. Hybrid BM25 + semantic retrieval, streaming responses, evaluated with RAGAS.

**Live demo:** [rag-chatbot-divya-s.streamlit.app](https://rag-chatbot-divya-s.streamlit.app)

---

## What it does

Upload PDF or TXT files, ask questions, and get answers grounded in the documents with source chunk previews.

- Answers only from the documents; replies "The document does not mention this." when the answer isn't there, and gives a partial answer when the documents cover part of the question
- Handles normal conversation: greetings, thanks, "what can you do?" and "which documents are loaded?"
- Summarizes whole documents: "summarize the document", "overview" or "key points" reads each selected document in order instead of a few search hits
- Streams tokens live as they generate
- Keeps each user's documents private: every browser session has its own in-memory store, nothing is written to disk, and a session's documents are deleted after 2 hours idle (or with "Remove all documents")
- Caps API usage: 20 questions per session and 200 per day across all users (both configurable)
- Remembers the conversation for follow-up questions (oldest turns trimmed past ~20k tokens)
- Filters retrieval by document when multiple files are loaded
- Reads scanned PDFs: OCR runs per page, so scanned pages inside otherwise digital PDFs are captured too
- Reads text files saved as UTF-8, UTF-16 (Windows Notepad/PowerShell) or Windows-1252
- Reports files that fail to load, without stopping the rest of the batch
- Re-uploading a file replaces its previous version
- Treats document text as data, not instructions, to resist prompt injection from uploaded files

---

## Architecture

```
User message
    │
    ├── summary request? ──► each selected document's chunks, in reading order
    │
    ▼
ChromaDB (semantic) ── best match too far? ──► no excerpts (greeting, small talk, off-topic)
    │
    ▼
BM25 (keyword)  +  ChromaDB (semantic)
  filtered to selected documents
         └──── RRF merge (k=60) ────┘
    │
    ▼
Top 5 chunks + their neighboring chunks
    │
    ▼
Claude Sonnet (temperature 0) → st.write_stream() live rendering
```

**Ingestion:** PyMuPDF text extraction → OCR for pages with no text layer → adaptive chunking → replace old chunks for that file → ChromaDB upsert → BM25 rebuild (once per upload batch)

**OCR:** scanned pages are rendered at 300 DPI with PyMuPDF and read by Tesseract. Lines Tesseract is less than 70% confident about, or with fewer than two letters or digits, are dropped: they're almost always smudges or table borders read as junk, and leaving them in splits real sentences across chunks.

**Adaptive chunking:** files with `warranty`, `contract`, `agreement` or `terms` in the name use 300-character chunks for precise clause lookup; everything else uses 800. Both use 150-character overlap. (500 and 800 for contracts were tested and scored worse.)

**Neighbor chunks:** each retrieved chunk brings the chunk before and after it from the same document, so tables, lists and sentences cut at a chunk boundary arrive whole.

**BM25 details:** query and chunks share one tokenizer (lowercase words, punctuation stripped), and only chunks with a real keyword match get RRF credit.

**Relevance threshold:** if the closest chunk's cosine distance is above 0.65, no excerpts are sent and the model replies conversationally or declines. Calibrated on `Warranty.pdf`: answerable questions scored at most 0.58, off-topic ones at least 0.68.

**Shared core:** ingestion, retrieval, the threshold and the prompt live in `rag_core.py`, used by both the app and the eval, so the eval measures exactly what the app does.

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
| OCR | Tesseract (pages rendered with PyMuPDF) |
| UI | Streamlit |
| Text splitting | langchain-text-splitters |

---

## Evaluation (RAGAS — 12 Q&A pairs on `Warranty.pdf`)

Average of 3 runs:

| Metric | Score | Runs |
|---|---|---|
| Faithfulness | 0.884 | 0.903, 0.875, 0.875 |
| Answer Relevancy | 0.882 | 0.886, 0.881, 0.877 |
| Context Recall | 0.889 | 0.917, 0.833, 0.917 |
| Out-of-scope questions declined | 2/2 | every run |

- **Out-of-scope questions** (price, return policy) have "The document does not mention this." as the correct reply. RAGAS scores that reply 0 for answer relevancy and erratically for faithfulness, so those two questions are scored on whether the bot declined, and faithfulness and answer relevancy are averaged over the 10 answerable questions.
- **Judge noise:** retrieval was identical in all 3 runs, yet single questions flipped between 0 and 1 (e.g. recall 0.917 vs 0.833). With 12 questions, one flip moves an average by up to 0.08, so compare averages over several runs.
- **Known misses:** two expected answers rely on the general warranty Disclaimer (wear and tear, misuse), which the search doesn't connect to questions about the wallpaper; and the judge can't verify answers that read across the flattened OCR table.

Run with `venv\Scripts\python eval\run_eval.py` from the repo root, with `Warranty.pdf` in `docs/`. The eval loads it with the app's own ingestion code and writes per-question scores to `eval/results.csv`.

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

OCR needs Tesseract:

- **Windows:** install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) to `C:\Program Files\Tesseract-OCR`
- **Linux:** `sudo apt install tesseract-ocr`
- **Streamlit Cloud:** installed automatically from `packages.txt`

If Tesseract is missing, the sidebar shows a warning and scanned PDFs report an error; digital PDFs and text files still load.

On Streamlit Cloud, set `ANTHROPIC_API_KEY` in the app's secrets.

Optional settings (in `.env` locally or Streamlit secrets on Cloud):

| Setting | Default | Meaning |
|---|---|---|
| `MAX_QUESTIONS_PER_SESSION` | 20 | Messages one browser session can send to the API |
| `DAILY_QUESTION_CAP` | 200 | Messages per day (UTC) across all users |
| `SESSION_TTL_SECONDS` | 7200 | Idle time before a session's documents are deleted |

For a hard spending limit, also set a monthly spend limit for the API key's workspace in the Anthropic Console. The in-app caps reset if the app restarts.

---

## Known limitations

- The relevance threshold (0.65) was calibrated on one document and a small question set; other document types may need a different value
- OCR flattens tables, so table answers from scanned PDFs can mix up columns
- General clauses (e.g. a warranty disclaimer) can be missed when a question names a specific product, because the clause never mentions it
- Summary detection uses keywords ("summarize", "overview", "key points", "what is this document about")
- Documents aren't saved: refreshing the page starts a new session, so files must be uploaded again
- Usage counters live in server memory: they reset when the app restarts, and the per-session cap resets on page refresh (the daily cap is the real limit)
- All sessions share one server's memory, so many large uploads at once could exhaust RAM on Streamlit Cloud
