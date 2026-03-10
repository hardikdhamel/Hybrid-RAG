"""Hybrid RAG Engine - orchestrates the full pipeline."""

import json
import os
from typing import List

from vector_store import VectorStore
from bm25_store import BM25Store
from reranker import reciprocal_rank_fusion
from llm_service import generate_response, generate_response_stream
from document_loader import extract_text, chunk_text
from config import CHUNK_SIZE, CHUNK_OVERLAP, FINAL_TOP_K, UPLOAD_DIR, CHROMA_PERSIST_DIR

INGESTED_FILES_PATH = os.path.join(CHROMA_PERSIST_DIR, "ingested_files.json")


class HybridRAGEngine:
    """Main Hybrid RAG engine combining dense + sparse retrieval."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()
        self._ingested_files: List[str] = self._load_ingested_files()

    def _load_ingested_files(self) -> List[str]:
        """Load previously ingested file names from disk."""
        if os.path.exists(INGESTED_FILES_PATH):
            with open(INGESTED_FILES_PATH, "r") as f:
                return json.load(f)
        return []

    def _save_ingested_files(self):
        """Persist ingested file names to disk."""
        with open(INGESTED_FILES_PATH, "w") as f:
            json.dump(self._ingested_files, f)

    async def ingest_document(self, file_path: str, original_filename: str) -> dict:
        """
        Ingest a document: extract text, chunk it, and index in both stores.
        """
        print(f"\n[RAG_ENGINE] === Starting ingestion: {original_filename} ===")

        # Step 1: Extract text
        print(f"[RAG_ENGINE] Step 1/4: Extracting text...")
        text = extract_text(file_path)
        if not text.strip():
            print(f"[RAG_ENGINE] ERROR: No text extracted")
            return {"status": "error", "message": "No text could be extracted from the file."}
        print(f"[RAG_ENGINE] Step 1/4: DONE — {len(text)} chars extracted")

        # Step 2: Chunk the text
        print(f"[RAG_ENGINE] Step 2/4: Chunking text...")
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        if not chunks:
            print(f"[RAG_ENGINE] ERROR: No chunks generated")
            return {"status": "error", "message": "No chunks generated from the document."}
        print(f"[RAG_ENGINE] Step 2/4: DONE — {len(chunks)} chunks created")

        # Step 3: Prepare texts and metadata
        texts = [c["text"] for c in chunks]
        metadatas = [
            {"source": original_filename, "chunk_id": c["id"]}
            for c in chunks
        ]

        # Step 4: Index in both stores
        print(f"[RAG_ENGINE] Step 3/4: Generating embeddings & indexing in vector store...")
        await self.vector_store.add_documents(texts, metadatas, original_filename)
        print(f"[RAG_ENGINE] Step 3/4: DONE — Vector store indexed")

        print(f"[RAG_ENGINE] Step 4/4: Indexing in BM25 store...")
        self.bm25_store.add_documents(texts, metadatas)
        print(f"[RAG_ENGINE] Step 4/4: DONE — BM25 store indexed")

        self._ingested_files.append(original_filename)
        self._save_ingested_files()

        print(f"[RAG_ENGINE] === Ingestion complete: {original_filename} ===")
        return {
            "status": "success",
            "message": f"Ingested '{original_filename}' — {len(chunks)} chunks indexed.",
            "chunks": len(chunks),
            "filename": original_filename,
        }

    async def query(self, user_query: str) -> dict:
        """
        Run the full Hybrid RAG pipeline:
        1. Parallel retrieval (vector + BM25)
        2. Reciprocal Rank Fusion
        3. LLM generation with fused context
        """
        print(f"\n[RAG_ENGINE] === Query Pipeline Started ===")
        print(f"[RAG_ENGINE] Question: \"{user_query}\"")

        # Step 1: Parallel retrieval
        print(f"[RAG_ENGINE] Step 1/4: Running PARALLEL retrieval (Vector + BM25)...")

        print(f"[RAG_ENGINE]   → Vector search (semantic/dense)...")
        vector_results = await self.vector_store.search(user_query)
        print(f"[RAG_ENGINE]   ✓ Vector search returned {len(vector_results)} results")

        print(f"[RAG_ENGINE]   → BM25 search (keyword/sparse)...")
        bm25_results = self.bm25_store.search(user_query)
        print(f"[RAG_ENGINE]   ✓ BM25 search returned {len(bm25_results)} results")
        print(f"[RAG_ENGINE] Step 1/4: DONE")

        # Step 2: Reciprocal Rank Fusion
        print(f"[RAG_ENGINE] Step 2/4: Reciprocal Rank Fusion (merging results)...")
        fused_results = reciprocal_rank_fusion(
            vector_results, bm25_results, top_k=FINAL_TOP_K
        )
        print(f"[RAG_ENGINE] Step 2/4: DONE — {len(fused_results)} results after fusion")

        if not fused_results:
            print(f"[RAG_ENGINE] No results found — returning empty answer")
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                "sources": [],
                "retrieval_info": {
                    "vector_results": 0,
                    "bm25_results": 0,
                    "fused_results": 0,
                },
            }

        # Print RRF scoreboard
        print(f"\n[RAG_ENGINE] ┌─── RRF Scoreboard ───────────────────────────────────────┐")
        for i, r in enumerate(fused_results):
            src_file = r['metadata'].get('source', '?')
            chunk_id = r['metadata'].get('chunk_id', '?')
            found_by = ' + '.join(r['sources'])
            preview = r['text'][:80].replace('\n', ' ')
            print(f"[RAG_ENGINE] │ #{i+1}  RRF={r['rrf_score']:.4f}  vec={r['vector_score']:.4f}  bm25={r['bm25_score']:.4f}  found_by=[{found_by}]")
            print(f"[RAG_ENGINE] │     source={src_file} chunk={chunk_id}")
            print(f"[RAG_ENGINE] │     \"{preview}...\"")
        print(f"[RAG_ENGINE] └────────────────────────────────────────────────────────┘\n")

        # Step 3: Generate answer with LLM
        print(f"[RAG_ENGINE] Step 3/4: Sending {len(fused_results)} chunks to LLM for answer generation...")
        answer = await generate_response(user_query, fused_results)
        print(f"[RAG_ENGINE] Step 3/4: DONE — LLM generated {len(answer)} chars")

        # Step 4: Prepare source citations
        print(f"[RAG_ENGINE] Step 4/4: Preparing source citations...")
        sources = []
        for i, result in enumerate(fused_results):
            sources.append({
                "index": i + 1,
                "source_file": result["metadata"].get("source", "Unknown"),
                "chunk_id": result["metadata"].get("chunk_id", "?"),
                "text_preview": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                "rrf_score": round(result["rrf_score"], 4),
                "vector_score": round(result.get("vector_score", 0), 4),
                "bm25_score": round(result.get("bm25_score", 0), 4),
                "retrieval_sources": result["sources"],
            })
        print(f"[RAG_ENGINE] Step 4/4: DONE")
        print(f"[RAG_ENGINE] === Query Pipeline Complete ===")

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_info": {
                "vector_results": len(vector_results),
                "bm25_results": len(bm25_results),
                "fused_results": len(fused_results),
            },
        }

    async def query_stream(self, user_query: str):
        """Stream version of the query pipeline."""
        vector_results = await self.vector_store.search(user_query)
        bm25_results = self.bm25_store.search(user_query)

        fused_results = reciprocal_rank_fusion(
            vector_results, bm25_results, top_k=FINAL_TOP_K
        )

        if not fused_results:
            yield {
                "type": "answer",
                "content": "I couldn't find any relevant information in the uploaded documents.",
            }
            return

        # Stream LLM response
        async for token in generate_response_stream(user_query, fused_results):
            yield {"type": "token", "content": token}

        # Send sources at the end
        sources = []
        for i, result in enumerate(fused_results):
            sources.append({
                "index": i + 1,
                "source_file": result["metadata"].get("source", "Unknown"),
                "chunk_id": result["metadata"].get("chunk_id", "?"),
                "text_preview": result["text"][:200] + "...",
                "rrf_score": round(result["rrf_score"], 4),
                "retrieval_sources": result["sources"],
            })

        yield {"type": "sources", "content": sources}

    def get_stats(self) -> dict:
        """Get current system stats."""
        return {
            "vector_count": self.vector_store.get_doc_count(),
            "bm25_count": self.bm25_store.get_doc_count(),
            "ingested_files": self._ingested_files,
        }

    def reset(self):
        """Clear all indexed data."""
        self.vector_store.delete_collection()
        self.bm25_store.clear()
        self._ingested_files = []
        self._save_ingested_files()
