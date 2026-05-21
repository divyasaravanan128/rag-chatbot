# eval/run_eval.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic import Anthropic
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from datasets import Dataset
# imports — replace the ragas-related ones with:
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from anthropic import Anthropic as AnthropicClient
from dotenv import load_dotenv
import pandas as pd

from eval.questions import eval_pairs

load_dotenv()

# ── Setup clients (mirrors app.py) ────────────────────────────────────────────

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=embed_fn,
)

# Build BM25 index from existing collection
all_data = collection.get(include=["documents", "metadatas"])
all_chunks = all_data["documents"]
all_metas  = all_data["metadatas"]
tokenized  = [doc.lower().split() for doc in all_chunks]
bm25_index = BM25Okapi(tokenized)

# ── Hybrid search (mirrors app.py) ────────────────────────────────────────────

def hybrid_search(query: str, n_results: int = 5, k: int = 60):
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranked = sorted(range(len(all_chunks)), key=lambda i: bm25_scores[i], reverse=True)[:n_results * 2]

    chroma_results = collection.query(query_texts=[query], n_results=n_results * 2)
    chroma_docs = chroma_results["documents"][0]

    candidates = {}
    for rank, idx in enumerate(bm25_ranked):
        text = all_chunks[idx]
        candidates[text] = candidates.get(text, 0) + 1 / (k + rank)
    for rank, doc in enumerate(chroma_docs):
        candidates[doc] = candidates.get(doc, 0) + 1 / (k + rank)

    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [text for text, _ in ranked[:n_results]]

# ── Generate answers ───────────────────────────────────────────────────────────

def get_answer(question: str, context_chunks: list[str]) -> str:
    system_prompt = (
        "You are a helpful assistant. Answer questions based only on the provided context.\n"
        "If the context doesn't contain enough information, say so clearly.\n\n"
        "Context:\n" + "\n\n".join(context_chunks)
    )
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text

# ── Build eval dataset ─────────────────────────────────────────────────────────

print("Generating answers and collecting contexts...")

rows = []
for pair in eval_pairs:
    question     = pair["question"]
    ground_truth = pair["ground_truth"]
    contexts     = hybrid_search(question)
    answer       = get_answer(question, contexts)

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
ragas_llm = LangchainLLMWrapper(ChatAnthropic(
    model="claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
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