"""Reciprocal Rank Fusion (RRF) and reranking for Hybrid RAG."""

from typing import List, Dict


def reciprocal_rank_fusion(
    vector_results: List[dict],
    bm25_results: List[dict],
    k: int = 60,
    top_k: int = 5,
) -> List[dict]:
    print(f"[RRF] Input: {len(vector_results)} vector results + {len(bm25_results)} BM25 results (k={k}, top_k={top_k})")
    """
    Merge results from vector and BM25 search using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) for each result list the document appears in.
    Documents appearing in both lists get boosted scores.
    """
    doc_scores: Dict[str, dict] = {}

    # Process vector results
    for rank, result in enumerate(vector_results):
        doc_key = result["text"][:200]  # Use first 200 chars as key
        rrf_score = 1.0 / (k + rank + 1)
        if doc_key in doc_scores:
            doc_scores[doc_key]["rrf_score"] += rrf_score
            doc_scores[doc_key]["sources"].append("vector")
        else:
            doc_scores[doc_key] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "rrf_score": rrf_score,
                "vector_score": result.get("score", 0),
                "bm25_score": 0,
                "sources": ["vector"],
            }

    # Process BM25 results
    for rank, result in enumerate(bm25_results):
        doc_key = result["text"][:200]
        rrf_score = 1.0 / (k + rank + 1)
        if doc_key in doc_scores:
            doc_scores[doc_key]["rrf_score"] += rrf_score
            doc_scores[doc_key]["bm25_score"] = result.get("score", 0)
            if "bm25" not in doc_scores[doc_key]["sources"]:
                doc_scores[doc_key]["sources"].append("bm25")
        else:
            doc_scores[doc_key] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "rrf_score": rrf_score,
                "vector_score": 0,
                "bm25_score": result.get("score", 0),
                "sources": ["bm25"],
            }

    # Sort by RRF score and return top-k
    sorted_results = sorted(
        doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True
    )

    total_unique = len(sorted_results)
    both_count = sum(1 for r in sorted_results if len(r["sources"]) > 1)
    vector_only = sum(1 for r in sorted_results if r["sources"] == ["vector"])
    bm25_only = sum(1 for r in sorted_results if r["sources"] == ["bm25"])
    print(f"[RRF] Unique chunks: {total_unique} | Found by BOTH: {both_count} | Vector-only: {vector_only} | BM25-only: {bm25_only}")
    print(f"[RRF] Returning top {min(top_k, len(sorted_results))} results")

    return sorted_results[:top_k]
