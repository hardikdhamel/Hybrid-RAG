"""LLM service using Ollama's gpt-oss:120b-cloud model."""

from typing import AsyncGenerator, List

import httpx

from config import OLLAMA_BASE_URL, LLM_MODEL


def build_prompt(query: str, context_chunks: List[dict]) -> str:
    """Build the RAG prompt with retrieved context."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("metadata", {}).get("source", "Unknown")
        chunk_id = chunk.get("metadata", {}).get("chunk_id", "?")
        context_parts.append(
            f"[Source {i}: {source} | Chunk {chunk_id}]\n{chunk['text']}"
        )

    context_str = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context below. 
If the context does not contain enough information to answer, say so clearly. Do not make up information.
Do NOT include any source citations or references like [Source 1] in your answer.
When writing mathematical formulas, use $$ for display math and $ for inline math. For example: $$E = mc^2$$ or inline $x^2$. Never use \\[ \\] or \\( \\) delimiters for math.

CONTEXT:
{context_str}

USER QUESTION: {query}

ANSWER:"""


async def generate_response(query: str, context_chunks: List[dict]) -> str:
    """Generate a response using the LLM with the retrieved context."""
    prompt = build_prompt(query, context_chunks)
    print(f"[LLM] Sending prompt to '{LLM_MODEL}' ({len(prompt)} chars, {len(context_chunks)} context chunks)...")

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 1024,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        answer = data["response"]
        # Print LLM timing if available
        if "total_duration" in data:
            duration_sec = data["total_duration"] / 1e9
            print(f"[LLM] Response received in {duration_sec:.2f}s ({len(answer)} chars)")
        else:
            print(f"[LLM] Response received ({len(answer)} chars)")
        return answer


async def generate_response_stream(query: str, context_chunks: List[dict]):
    """Generate a streaming response using the LLM."""
    prompt = build_prompt(query, context_chunks)

    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 1024,
                },
            },
        ) as response:
            import json
            async for line in response.aiter_lines():
                if line.strip():
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break
