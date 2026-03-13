"""Embedding service using Ollama's nomic-embed-text model."""

import asyncio
from typing import List

import httpx
import numpy as np

from config import OLLAMA_EMBEDDING_BASE_URL, EMBEDDING_MODEL

# Retry settings
MAX_RETRIES = 5
RETRY_BASE_DELAY = 4  # seconds (exponential) to survive Ngrok rate limits
EMBEDDING_TIMEOUT = 300.0  # 5 minutes per request
MAX_CHUNK_CHARS = 8000  # Truncate chunks beyond this to avoid Ollama overload

async def get_embedding(text: str, chunk_index: int = 0, client: httpx.AsyncClient = None) -> List[float]:
    """Get embedding for a single text using Ollama with retry logic."""

    # Truncate extremely long chunks to prevent Ollama memory issues
    if len(text) > MAX_CHUNK_CHARS:
        print(f"[EMBEDDING] WARNING: Chunk {chunk_index} truncated from {len(text)} to {MAX_CHUNK_CHARS} chars")
        text = text[:MAX_CHUNK_CHARS]

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if client is None:
                async with httpx.AsyncClient(timeout=EMBEDDING_TIMEOUT) as local_client:
                    response = await local_client.post(
                        f"{OLLAMA_EMBEDDING_BASE_URL}/api/embeddings",
                        json={"model": EMBEDDING_MODEL, "prompt": text},
                        headers={"ngrok-skip-browser-warning": "69420"},
                    )
                    response.raise_for_status()
                    data = response.json()
            else:
                response = await client.post(
                    f"{OLLAMA_EMBEDDING_BASE_URL}/api/embeddings",
                    json={"model": EMBEDDING_MODEL, "prompt": text},
                    headers={"ngrok-skip-browser-warning": "69420"},
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
                delay = RETRY_BASE_DELAY ** attempt  # 3s, 9s, 27s
                print(f"[EMBEDDING] ⚠ Chunk {chunk_index} failed (attempt {attempt}/{MAX_RETRIES}): {type(e).__name__}: {e}")
                print(f"[EMBEDDING]   Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                print(f"[EMBEDDING] ✖ Chunk {chunk_index} failed after {MAX_RETRIES} attempts: {type(e).__name__}: {e}")
                raise last_error


from typing import List, Optional

async def get_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """Get embeddings for a batch of texts with concurrent retry support. Returns None for failed chunks."""
    print(f"[EMBEDDING] Generating embeddings for {len(texts)} chunks using '{EMBEDDING_MODEL}' on remote GPU...")
    
    # Process concurrent requests using a semaphore
    # Ngrok Free Tier drops connections above ~20 concurrent, and Colab T4 struggles with massive batch sizes.
    concurrency_limit = 4 
    sem = asyncio.Semaphore(concurrency_limit)
    
    embeddings_dict = {}
    failed_chunks = []
    completed_count = 0
    total_texts = len(texts)

    async def _process_chunk(i: int, text: str, shared_client: httpx.AsyncClient):
        nonlocal completed_count
        async with sem:
            try:
                emb = await get_embedding(text, chunk_index=i, client=shared_client)
                embeddings_dict[i] = emb
            except Exception as e:
                print(f"[EMBEDDING] ✖ Skipping chunk {i} after all retries failed: {e}")
                embeddings_dict[i] = None
                failed_chunks.append(i)
            finally:
                completed_count += 1
                if completed_count % 10 == 0 or completed_count == total_texts:
                    print(f"[EMBEDDING]   Progress: {completed_count}/{total_texts} chunks embedded")
    
    # Use a single HTTP client for connection pooling
    limits = httpx.Limits(max_keepalive_connections=concurrency_limit, max_connections=concurrency_limit)
    async with httpx.AsyncClient(timeout=EMBEDDING_TIMEOUT, limits=limits) as shared_client:
        # Launch tasks concurrently
        tasks = [_process_chunk(i, text, shared_client) for i, text in enumerate(texts)]
        await asyncio.gather(*tasks)

    # Reconstruct in original order, including None for failed chunks
    embeddings = [embeddings_dict.get(i) for i in range(total_texts)]

    if failed_chunks:
        print(f"[EMBEDDING] WARNING: {len(failed_chunks)} chunks failed and were skipped: {failed_chunks[:20]}...")

    successful_count = total_texts - len(failed_chunks)
    if successful_count == 0:
        raise RuntimeError("All embedding requests failed. Check if Ollama/Ngrok is running and reachable.")

    print(f"[EMBEDDING] {successful_count}/{total_texts} embeddings successfully generated (dim={len(embeddings[0]) if successful_count > 0 else 0})")
    return embeddings
