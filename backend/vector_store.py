"""Vector store using ChromaDB for dense retrieval."""

import uuid
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR, TOP_K_VECTOR
from embeddings import get_embedding, get_embeddings_batch


class VectorStore:
    """ChromaDB-based vector store for semantic search."""

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="hybrid_rag_docs",
            metadata={"hnsw:space": "cosine"},
        )

    async def add_documents(
        self, texts: List[str], metadatas: List[dict], doc_name: str
    ):
        """Add document chunks with embeddings to the vector store."""
        print(f"[VECTOR_STORE] Generating embeddings for {len(texts)} chunks...")
        embeddings = await get_embeddings_batch(texts)

        # If some chunks were skipped during embedding, align texts/metadatas
        # get_embeddings_batch returns only successful embeddings in order
        if len(embeddings) < len(texts):
            print(f"[VECTOR_STORE] NOTE: {len(texts) - len(embeddings)} chunks skipped, storing {len(embeddings)} chunks")
            texts = texts[:len(embeddings)]
            metadatas = metadatas[:len(embeddings)]

        ids = [f"{doc_name}_{uuid.uuid4().hex[:8]}" for _ in texts]

        # ChromaDB has a batch limit, so add in batches of 100
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            print(f"[VECTOR_STORE] Adding batch {i//batch_size + 1} to ChromaDB ({end - i} chunks)...")
            self._collection.add(
                ids=ids[i:end],
                documents=texts[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )
        print(f"[VECTOR_STORE] All {len(texts)} chunks stored in ChromaDB (total: {self._collection.count()})")

    async def search(self, query: str, top_k: int = TOP_K_VECTOR) -> List[dict]:
        """Perform semantic search and return results with scores."""
        print(f"[VECTOR_STORE] Embedding query for semantic search...")
        query_embedding = await get_embedding(query)
        print(f"[VECTOR_STORE] Searching ChromaDB (top_k={top_k}, total_docs={self._collection.count()})...")
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                score = 1 - results["distances"][0][i]
                search_results.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": score,
                    "source": "vector",
                })
            print(f"[VECTOR_STORE] Found {len(search_results)} results (scores: {[round(r['score'],4) for r in search_results[:5]]})")
        else:
            print(f"[VECTOR_STORE] No results found")
        return search_results

    def get_doc_count(self) -> int:
        """Return the number of documents in the store."""
        return self._collection.count()

    def delete_collection(self):
        """Delete the entire collection."""
        self._client.delete_collection("hybrid_rag_docs")
        self._collection = self._client.get_or_create_collection(
            name="hybrid_rag_docs",
            metadata={"hnsw:space": "cosine"},
        )
