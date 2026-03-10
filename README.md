# Hybrid RAG System

A production-grade Hybrid Retrieval-Augmented Generation system combining **dense (vector/semantic)** and **sparse (BM25/keyword)** retrieval with **Reciprocal Rank Fusion (RRF)** reranking.

## Architecture

```
User Query → Query Parsing → ┬─ Vector Search (ChromaDB + nomic-embed-text) ──┐
                              └─ Keyword Search (BM25)                         ├→ RRF Fusion → LLM (gpt-oss:120b-cloud) → Answer + Sources
                                                                               ┘
```

## Tech Stack

- **Backend**: FastAPI, ChromaDB, BM25 (rank-bm25), Ollama
- **Frontend**: React + Vite
- **LLM**: `gpt-oss:120b-cloud` via Ollama
- **Embeddings**: `nomic-embed-text:latest` via Ollama
- **Vector Store**: ChromaDB (persistent)
- **Sparse Retrieval**: BM25Okapi

## Prerequisites

- Python 3.10+
- Node.js 18+
- Ollama running with `gpt-oss:120b-cloud` and `nomic-embed-text:latest` models

## Quick Start

### 1. Start Ollama (ensure models are available)

```bash
ollama pull nomic-embed-text:latest
# Ensure gpt-oss:120b-cloud is available
```

### 2. Start Backend

```bash
cd "Hybrid RAG"
source venv/bin/activate
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start Frontend

```bash
cd "Hybrid RAG/frontend"
npm run dev
```

### 4. Open Browser

Navigate to `http://localhost:3000`

## API Endpoints

| Method | Endpoint        | Description                 |
| ------ | --------------- | --------------------------- |
| POST   | `/upload`       | Upload & ingest a document  |
| POST   | `/query`        | Query with hybrid retrieval |
| POST   | `/query/stream` | Stream response (SSE)       |
| GET    | `/stats`        | Get system statistics       |
| POST   | `/reset`        | Clear all indexed data      |
| GET    | `/health`       | Health check                |

## Supported File Types

- PDF (.pdf)
- Word Documents (.docx)
- Plain Text (.txt)
- Markdown (.md)
- CSV (.csv)
