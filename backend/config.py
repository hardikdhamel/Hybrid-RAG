"""Configuration for the Hybrid RAG system."""

import os

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_BASE_URL = os.getenv("OLLAMA_EMBEDDING_BASE_URL", "https://e92f-34-32-246-61.ngrok-free.app")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:120b-cloud")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval settings
TOP_K_VECTOR = 10
TOP_K_BM25 = 10
FINAL_TOP_K = 5

# ChromaDB settings
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
