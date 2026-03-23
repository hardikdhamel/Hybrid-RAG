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


async def generate_response(query: str, context_chunks: List[dict]) -> dict:
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
        
        # Extract Token & Timing stats from Ollama response
        prompt_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
        total_tokens = prompt_tokens + output_tokens
        
        print("\n" + "="*50)
        print(f"[LLM STATS] Model: {LLM_MODEL}")
        print(f"[LLM STATS] Input Tokens (Prompt): {prompt_tokens}")
        print(f"[LLM STATS] Output Tokens (Answer): {output_tokens}")
        print(f"[LLM STATS] Total Tokens: {total_tokens}")
        
        if "total_duration" in data:
            duration_sec = data["total_duration"] / 1e9
            print(f"[LLM STATS] Total Time: {duration_sec:.2f}s")
            if output_tokens > 0:
                tokens_per_sec = output_tokens / (data.get("eval_duration", 1) / 1e9)
                print(f"[LLM STATS] Generation Speed: {tokens_per_sec:.2f} tokens/sec")
        print("="*50 + "\n")
        
        return {
            "answer": answer,
            "tokens": {
                "prompt": prompt_tokens,
                "output": output_tokens,
                "total": total_tokens
            }
        }


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
                        # Yield token stats at the end
                        yield {
                            "type": "tokens",
                            "content": {
                                "prompt": data.get("prompt_eval_count", 0),
                                "output": data.get("eval_count", 0),
                                "total": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                            }
                        }
                        break

async def ask_llm_json(prompt: str) -> dict:
    """Ask the LLM a question and expect a JSON response."""
    import json
    
    print(f"[LLM] Sending JSON prompt to '{LLM_MODEL}' ({len(prompt)} chars)...")
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                # "format": "json" is supported by recent Ollama versions to guarantee JSON output
                "format": "json",
                "options": {
                    "temperature": 0.1,  # Low temp for deterministic JSON
                    "top_p": 0.9,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        answer = data["response"]
        
        try:
            return json.loads(answer)
        except json.JSONDecodeError as e:
            print(f"[LLM] WARNING: Failed to parse JSON response: {answer}")
            print(f"[LLM] Error: {e}")
            return {}
