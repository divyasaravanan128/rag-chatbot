# eval/run_eval.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic import Anthropic
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import pandas as pd

from eval.questions import eval_pairs
import rag_core
from rag_core import (
    MODEL, NO_MATCH_MESSAGE, SYSTEM_PROMPT, build_messages, hybrid_search, tokenize,
)

load_dotenv()

# ── Setup clients ─────────────────────────────────────────────────────────────

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

EVAL_DOCS = ["docs/Warranty.pdf"]  # the document eval/questions.py is written against

embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
collection = chromadb.EphemeralClient().get_or_create_collection(
    name="eval",
    embedding_function=embed_fn,
)

# Ingest with the same code the app uses
print("Loading documents...")
for path in EVAL_DOCS:
    rag_core.add_document(collection, path, os.path.basename(path))

# Build BM25 index from the collection
all_data = collection.get(include=["documents", "metadatas"])
all_chunks = all_data["documents"]
all_metas  = all_data["metadatas"]
bm25_index = BM25Okapi([tokenize(doc) for doc in all_chunks])

# ── Retrieval + answers (same code and prompt as app.py, via rag_core) ────────

def retrieve(question: str) -> tuple[list[str], list[dict]]:
    return hybrid_search(question, collection, bm25_index, all_chunks, all_metas)


def get_answer(question: str, context_chunks: list[str], sources_meta: list[dict]) -> str:
    if not context_chunks:
        return NO_MATCH_MESSAGE  # app shows this without calling the model
    response = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=build_messages(context_chunks, sources_meta, [{"role": "user", "content": question}]),
    )
    return response.content[0].text

# ── Build eval dataset ─────────────────────────────────────────────────────────

print("Generating answers and collecting contexts...")

rows = []
for pair in eval_pairs:
    question     = pair["question"]
    ground_truth = pair["ground_truth"]
    contexts, metas = retrieve(question)
    answer       = get_answer(question, contexts, metas)

    print(f"Q: {question[:60]}...")
    print(f"A: {answer[:80]}...\n")

    rows.append({
        "question":     question,
        "answer":       answer,
        "contexts":     contexts,
        "ground_truth": ground_truth,
    })

dataset = Dataset.from_list(rows)

# ── Run RAGAS ──────────────────────────────────────────────────────────────────

print("Running RAGAS evaluation...")

# Use Claude as the judge LLM
# temperature 0 for repeatable scores; room for long statement lists
ragas_llm = LangchainLLMWrapper(ChatAnthropic(
    model=MODEL,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0,
    max_tokens=4096,
))

# Use the same embedding model you already have locally — no OpenAI needed
ragas_embeddings = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2"
)

# Set the LLM and embeddings on each metric explicitly
for metric in [faithfulness, answer_relevancy, context_recall]:
    metric.llm = ragas_llm
    if hasattr(metric, "embeddings"):
        metric.embeddings = ragas_embeddings

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_recall],
)
# ── Save results ───────────────────────────────────────────────────────────────



df = results.to_pandas()
os.makedirs("eval", exist_ok=True)
df.to_csv("eval/results.csv", index=False)

print("\n── RAGAS Scores ──────────────────────────────")
print(f"Faithfulness:     {df['faithfulness'].mean():.3f}")
print(f"Answer Relevancy: {df['answer_relevancy'].mean():.3f}")
print(f"Context Recall:   {df['context_recall'].mean():.3f}")
print("\nDetailed results saved to eval/results.csv")