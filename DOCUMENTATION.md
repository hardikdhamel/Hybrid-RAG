# Hybrid RAG System — Documentation

## What is This?

A question-answering AI that reads your documents (PDF, DOCX, TXT) and answers questions about them — but smarter than typical AI search.

**The "Hybrid" part:** Instead of using just one search method, it combines **two different search strategies** and merges their results for significantly better accuracy.

---

## How It Works (Simple Version)

```
You upload a document → System breaks it into small pieces → Stores them two ways

You ask a question →  Two search engines hunt for answers simultaneously
                   →  Results are merged & ranked
                   →  AI reads the best pieces and writes you an answer with sources
```

---

## The Two Search Strategies

| Strategy                     | What It Does                                       | Good At                               | Bad At                              |
| ---------------------------- | -------------------------------------------------- | ------------------------------------- | ----------------------------------- |
| **Vector Search** (Semantic) | Understands _meaning_ — "car" matches "automobile" | Finding conceptually related content  | Missing exact terms like IDs, codes |
| **BM25 Search** (Keyword)    | Matches _exact words_ — "Q4" finds "Q4"            | Precise term matching, names, numbers | Missing synonyms or rephrased ideas |

**Example:** You ask _"What's the Q4 marketing budget?"_

- Vector search finds chunks about _"fiscal year expenditure"_ and _"promotional spending"_
- BM25 search finds chunks containing exactly _"Q4"_, _"marketing"_, _"budget"_
- Together, they cover what each one alone would miss.

---

## Architecture

```
┌─────────────┐     ┌───────────────────────────────────────────────┐
│   React UI  │────▶│              FastAPI Backend                  │
│  (port 3000)│◀────│              (port 8000)                      │
└─────────────┘     │                                               │
                    │  ┌─────────────────────────────────────────┐  │
                    │  │         Hybrid RAG Engine               │  │
                    │  │                                         │  │
                    │  │  ┌──────────┐      ┌──────────┐         │  │
                    │  │  │ ChromaDB │      │  BM25    │         │  │
                    │  │  │ (Vector) │      │(Keyword) │         │  │
                    │  │  └────┬─────┘      └────┬─────┘         │  │
                    │  │       └──────┬──────────┘               │  │
                    │  │             RRF                         │  │
                    │  │          (Merger)                       │  │
                    │  │             │                           │  │
                    │  │         ┌───▼────┐                      │  │
                    │  │         │ Ollama │                      │  │
                    │  │         │  LLM   │                      │  │
                    │  │         └───┬────┘                      │  │
                    │  │             ▼                           │  │
                    │  │          Answer                         │  │
                    │  └─────────────────────────────────────────┘  │
                    └───────────────────────────────────────────────┘
```

---

## Key Concepts Explained

### 1. Chunking

Documents are split into overlapping pieces (~500 words each, 50-word overlap). The overlap ensures no idea gets cut in half at a boundary.

### 2. Embeddings (nomic-embed-text)

Each chunk is converted into a list of numbers (a vector) that captures its _meaning_. Similar meanings produce similar vectors. This is done by the `nomic-embed-text` model via Ollama.

### 3. BM25 (Best Matching 25)

A classic information retrieval algorithm. It scores documents based on how often your search terms appear, adjusted for document length. No AI needed — pure math on word frequencies.

### 4. Reciprocal Rank Fusion (RRF) ⭐

This is the secret sauce. When two search engines return ranked results, RRF merges them fairly:

```
RRF Score = 1/(k + rank_in_vector) + 1/(k + rank_in_bm25)
```

A document ranked high in **both** lists gets a big boost. A document found by only one engine still gets included. The constant `k=60` prevents top-ranked results from dominating too heavily.

### 5. ChromaDB

An embedded vector database. Stores chunks + their embeddings on disk. Supports fast cosine similarity search to find the closest-meaning chunks to your query.

---

## Workflow: Upload

```
PDF/DOCX/TXT  →  Extract raw text  →  Split into chunks
                                           │
                                    ┌──────┴──────┐
                                    ▼              ▼
                              Embed each      Store raw text
                              chunk via        in BM25 index
                              Ollama           (keyword search)
                                    │              │
                                    ▼              ▼
                               ChromaDB        JSON file
                              (vectors)       (bm25_index)
```

## Workflow: Query

```
User Question
      │
      ├──▶ Vector Search (ChromaDB) ──▶ Top 10 by meaning
      │
      └──▶ BM25 Search ───────────────▶ Top 10 by keywords
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                              Reciprocal Rank Fusion (RRF)
                                         │
                                    Top 5 merged results
                                         │
                                         ▼
                              Prompt + Context → Ollama LLM
                                         │
                                         ▼
                                Answer + Source Citations
```

---

## Project Structure

```
Hybrid RAG/
├── backend/
│   ├── main.py              # API server (FastAPI)
│   ├── rag_engine.py         # Orchestrator — runs the full pipeline
│   ├── document_loader.py    # PDF/DOCX/TXT extraction + chunking
│   ├── embeddings.py         # Ollama embedding calls
│   ├── vector_store.py       # ChromaDB (dense/semantic search)
│   ├── bm25_store.py         # BM25 (sparse/keyword search)
│   ├── reranker.py           # Reciprocal Rank Fusion
│   ├── llm_service.py        # Ollama LLM for answer generation
│   └── config.py             # All settings in one place
├── frontend/
│   └── src/
│       ├── App.jsx           # Main chat UI
│       ├── api.js            # API client
│       └── App.css           # Dark theme styles
├── venv/                     # Python virtual environment
└── start.sh                  # One-command startup
```

---

## Models Used

| Model                     | Purpose                           | Provider |
| ------------------------- | --------------------------------- | -------- |
| `gpt-oss:120b-cloud`      | Generates answers from context    | Ollama   |
| `nomic-embed-text:latest` | Converts text to semantic vectors | Ollama   |

---

## API Endpoints

| Endpoint        | Method | Purpose                              |
| --------------- | ------ | ------------------------------------ |
| `/upload`       | POST   | Upload & index a document            |
| `/query`        | POST   | Ask a question, get answer + sources |
| `/query/stream` | POST   | Same but with streaming (SSE)        |
| `/stats`        | GET    | Chunk counts, uploaded files         |
| `/reset`        | POST   | Clear all data                       |
| `/health`       | GET    | Server status check                  |

---

## Quick Start

```bash
# 1. Make sure Ollama is running with both models
ollama serve
ollama pull nomic-embed-text:latest

# 2. Start everything
cd "Hybrid RAG"
./start.sh
```

Open `http://localhost:3000` → Upload a document → Ask questions.
