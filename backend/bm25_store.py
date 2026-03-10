"""BM25-based sparse retrieval engine."""

import json
import os
from typing import List, Optional

from rank_bm25 import BM25Okapi

from config import CHROMA_PERSIST_DIR, TOP_K_BM25

BM25_INDEX_PATH = os.path.join(CHROMA_PERSIST_DIR, "bm25_index.json")


class BM25Store:
    """BM25-based keyword search store."""

    def __init__(self):
        self._documents: List[str] = []
        self._metadatas: List[dict] = []
        self._bm25: Optional[BM25Okapi] = None
        self._load_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def _load_index(self):
        """Load BM25 index from disk if available."""
        if os.path.exists(BM25_INDEX_PATH):
            with open(BM25_INDEX_PATH, "r") as f:
                data = json.load(f)
            self._documents = data.get("documents", [])
            self._metadatas = data.get("metadatas", [])
            if self._documents:
                tokenized = [self._tokenize(doc) for doc in self._documents]
                self._bm25 = BM25Okapi(tokenized)

    def _save_index(self):
        """Persist the BM25 index to disk."""
        with open(BM25_INDEX_PATH, "w") as f:
            json.dump({
                "documents": self._documents,
                "metadatas": self._metadatas,
            }, f)

    def add_documents(self, texts: List[str], metadatas: List[dict]):
        """Add documents to the BM25 index."""
        self._documents.extend(texts)
        self._metadatas.extend(metadatas)
        tokenized = [self._tokenize(doc) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)
        self._save_index()

    def search(self, query: str, top_k: int = TOP_K_BM25) -> List[dict]:
        """Perform BM25 keyword search."""
        print(f"[BM25_STORE] Searching with keywords: {self._tokenize(query)[:10]}...")
        if not self._bm25 or not self._documents:
            print(f"[BM25_STORE] Index empty — no results")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "text": self._documents[idx],
                    "metadata": self._metadatas[idx],
                    "score": float(scores[idx]),
                    "source": "bm25",
                })
        print(f"[BM25_STORE] Found {len(results)} results with score > 0 (scores: {[round(r['score'],4) for r in results[:5]]})")
        return results

    def get_doc_count(self) -> int:
        """Return the number of documents."""
        return len(self._documents)

    def clear(self):
        """Clear the entire index."""
        self._documents = []
        self._metadatas = []
        self._bm25 = None
        if os.path.exists(BM25_INDEX_PATH):
            os.remove(BM25_INDEX_PATH)
