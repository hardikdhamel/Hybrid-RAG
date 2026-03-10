"""FastAPI server for the Hybrid RAG system."""

import json
import os
import uuid
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import UPLOAD_DIR
from rag_engine import HybridRAGEngine

app = FastAPI(title="Hybrid RAG API", version="1.0.0")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG engine
engine = HybridRAGEngine()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list
    retrieval_info: dict


@app.get("/")
async def root():
    return {"message": "Hybrid RAG API is running", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document."""
    print(f"\n{'='*60}")
    print(f"[UPLOAD] Received file: {file.filename}")
    print(f"[UPLOAD] Content type: {file.content_type}")

    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        print(f"[UPLOAD] REJECTED — unsupported extension: {ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save uploaded file
    safe_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    content = await file.read()
    print(f"[UPLOAD] File size: {len(content)} bytes")

    with open(file_path, "wb") as f:
        f.write(content)
    print(f"[UPLOAD] Saved to: {file_path}")

    try:
        result = await engine.ingest_document(file_path, file.filename)
        if result["status"] == "error":
            print(f"[UPLOAD] Ingestion returned error: {result['message']}")
            raise HTTPException(status_code=400, detail=result["message"])
        print(f"[UPLOAD] SUCCESS — {result}")
        print(f"{'='*60}\n")
        return result
    except ValueError as e:
        print(f"[UPLOAD] ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"[UPLOAD] EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the Hybrid RAG system."""
    print(f"\n{'='*60}")
    print(f"[QUERY] Received question: \"{request.query}\"")

    if not request.query.strip():
        print(f"[QUERY] REJECTED — empty query")
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    stats = engine.get_stats()
    print(f"[QUERY] Knowledge base: {stats['vector_count']} vector chunks, {stats['bm25_count']} BM25 chunks")
    if stats["vector_count"] == 0:
        print(f"[QUERY] REJECTED — no documents uploaded")
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. Please upload documents first.",
        )

    try:
        result = await engine.query(request.query)
        print(f"[QUERY] Answer generated ({len(result['answer'])} chars), {len(result['sources'])} sources")
        print(f"{'='*60}\n")
        return result
    except Exception as e:
        print(f"[QUERY] EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    """Stream a response from the Hybrid RAG system."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    stats = engine.get_stats()
    if stats["vector_count"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet.",
        )

    async def event_generator():
        async for chunk in engine.query_stream(request.query):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    return engine.get_stats()


@app.post("/reset")
async def reset_system():
    """Reset all indexed data."""
    engine.reset()
    return {"status": "success", "message": "All data has been cleared."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
