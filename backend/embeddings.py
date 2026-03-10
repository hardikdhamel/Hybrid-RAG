"""Embedding service using Ollama's nomic-embed-text model."""

from typing import List

import httpx
import numpy as np

from config import OLLAMA_BASE_URL, EMBEDDING_MODEL


async def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text using Ollama."""
    async with httpx.AsyncClient(timeout=120.0) as client:
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


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts."""
    print(f"[EMBEDDING] Generating embeddings for {len(texts)} chunks using '{EMBEDDING_MODEL}'...")
    embeddings = []
    for i, text in enumerate(texts):
        emb = await get_embedding(text)
        embeddings.append(emb)
        if (i + 1) % 10 == 0 or (i + 1) == len(texts):
            print(f"[EMBEDDING]   Progress: {i+1}/{len(texts)} chunks embedded")
    print(f"[EMBEDDING] All {len(texts)} embeddings generated (dim={len(embeddings[0])})")
    return embeddings
