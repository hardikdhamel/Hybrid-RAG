"""Embedding service using Ollama's nomic-embed-text model."""

import asyncio
from typing import List

import httpx
import numpy as np

from config import OLLAMA_BASE_URL, EMBEDDING_MODEL

# Retry settings
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds (exponential: 2s, 4s, 8s)
EMBEDDING_TIMEOUT = 300.0  # 5 minutes per request
MAX_CHUNK_CHARS = 8000  # Truncate chunks beyond this to avoid Ollama overload


async def get_embedding(text: str, chunk_index: int = 0) -> List[float]:
    """Get embedding for a single text using Ollama with retry logic."""

    # Truncate extremely long chunks to prevent Ollama memory issues
    if len(text) > MAX_CHUNK_CHARS:
        print(f"[EMBEDDING] WARNING: Chunk {chunk_index} truncated from {len(text)} to {MAX_CHUNK_CHARS} chars")
        text = text[:MAX_CHUNK_CHARS]

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=EMBEDDING_TIMEOUT) as client:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": EMBEDDING_MODEL, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                if "embedding" not in data:
                    print(f"[EMBEDDING] ERROR: Ollama response missing 'embedding' key. Response: {str(data)[:200]}")
                    raise ValueError(f"Ollama did not return an embedding. Response keys: {list(data.keys())}")
                return data["embedding"]

        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY ** attempt  # 2s, 4s, 8s
                print(f"[EMBEDDING] ⚠ Chunk {chunk_index} failed (attempt {attempt}/{MAX_RETRIES}): {type(e).__name__}: {e}")
                print(f"[EMBEDDING]   Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                print(f"[EMBEDDING] ✖ Chunk {chunk_index} failed after {MAX_RETRIES} attempts: {type(e).__name__}: {e}")
                raise last_error


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts with retry support."""
    print(f"[EMBEDDING] Generating embeddings for {len(texts)} chunks using '{EMBEDDING_MODEL}'...")
    embeddings = []
    failed_chunks = []

    for i, text in enumerate(texts):
        try:
            emb = await get_embedding(text, chunk_index=i)
            embeddings.append(emb)
        except Exception as e:
            print(f"[EMBEDDING] ✖ Skipping chunk {i} after all retries failed: {e}")
            failed_chunks.append(i)
            continue

        if (i + 1) % 10 == 0 or (i + 1) == len(texts):
            print(f"[EMBEDDING]   Progress: {i+1}/{len(texts)} chunks embedded")

        # Small delay between requests to avoid overwhelming Ollama
        await asyncio.sleep(0.05)

    if failed_chunks:
        print(f"[EMBEDDING] WARNING: {len(failed_chunks)} chunks failed and were skipped: {failed_chunks[:20]}...")

    if not embeddings:
        raise RuntimeError("All embedding requests failed. Check if Ollama is running and the model is available.")

    print(f"[EMBEDDING] {len(embeddings)}/{len(texts)} embeddings generated (dim={len(embeddings[0])})")
    return embeddings
