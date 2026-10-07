import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest
import litellm
from search import hybrid_search

load_dotenv()

# Initialize the lightweight local cross-encoder (already cached in /tmp)
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank_cache")
GENERATION_MODEL = "gemini/gemini-3.8-flash"


def rerank_passages(
    query: str, 
    documents: List[Dict[str, Any]], 
    top_n: int = 3, 
    min_score: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Stage 2: Cross-Encoder reranking using FlashRank.
    Filters candidate chunks by rank AND enforces a minimum score threshold.
    """
    if not documents:
        return []

    passages = [
        {"id": doc["metadata"].get("doc_id", str(doc["id"])), "text": doc["content"], "meta": doc["metadata"]}
        for doc in documents
    ]

    rerank_req = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_req)

    # Filter out low-confidence/zero matches, then take up to top_n
    valid_passages = [p for p in reranked if p["score"] >= min_score]
    return valid_passages[:top_n]


def generate_answer(query: str) -> Dict[str, Any]:
    """
    End-to-End Hybrid RAG Pipeline:
    1. Stage 1: Hybrid Search (pgvector + GIN + RRF) -> Top 5
    2. Stage 2: Local Reranker (FlashRank) -> Top 2
    3. Stage 3: LLM Generation with Grounded Citations
    """
    print(f"\nProcessing Query: '{query}'")
    print("-------------------------------------------------------")

    # 1. First-Stage Retrieval (High Recall)
    candidates = hybrid_search(query, limit=5)
    print(f"Stage 1 (Hybrid RRF): Retrieved {len(candidates)} candidates from Postgres.")

    # 2. Second-Stage Reranking (High Precision)
    reranked = rerank_passages(query, candidates, top_n=2)
    print(f"Stage 2 (FlashRank): Reranked down to top {len(reranked)} most relevant passages:")
    for r in reranked:
        print(f"  - [{r['id']}] Cross-Encoder Score: {round(r['score'], 4)}")

    # 3. Build Grounded Context
    context_blocks = []
    for r in reranked:
        context_blocks.append(f"Document [{r['id']}]:\n{r['text']}")
    context_str = "\n\n".join(context_blocks)

    # 4. LLM Generation
    system_prompt = (
        "You are an enterprise knowledge assistant. Answer the user query using ONLY "
        "the provided context. You must cite your sources inline using [Document ID]. "
        "If the context does not contain enough information to answer, state clearly "
        "that the information is unavailable in the knowledge base. Do not fabricate facts."
    )

    response = litellm.completion(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}"},
        ],
        num_retries=3,  # Auto-retry completion on network hiccups
    )

    answer_text = response.choices[0].message.content

    return {
        "query": query,
        "answer": answer_text,
        "sources": [r["id"] for r in reranked],
    }


if __name__ == "__main__":
    test_query = "What is the specific maintenance schedule and lubricant for our water pump?"
    result = generate_answer(test_query)

    print("\n--- Synthesized Answer with Grounded Citations ---")
    print(result["answer"])
    print(f"\nSources Cited: {result['sources']}")
