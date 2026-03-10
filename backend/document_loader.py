"""Document loading and chunking utilities."""

import os
from typing import List

import fitz  # PyMuPDF
from docx import Document as DocxDocument


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    print(f"[DOC_LOADER] Opening PDF: {file_path}")
    text = ""
    with fitz.open(file_path) as doc:
        print(f"[DOC_LOADER] PDF has {len(doc)} pages")
        for i, page in enumerate(doc):
            page_text = page.get_text()
            text += page_text
            print(f"[DOC_LOADER]   Page {i+1}: {len(page_text)} chars extracted")
    print(f"[DOC_LOADER] Total extracted: {len(text)} chars")
    return text


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    print(f"[DOC_LOADER] Opening DOCX: {file_path}")
    doc = DocxDocument(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    print(f"[DOC_LOADER] DOCX has {len(paragraphs)} non-empty paragraphs")
    return "\n".join(paragraphs)


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(file_path: str) -> str:
    """Extract text based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    print(f"[DOC_LOADER] Detected extension: {ext}")
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext in (".txt", ".md", ".csv"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[dict]:
    """Split text into overlapping chunks."""
    chunks = []
    words = text.split()
    print(f"[CHUNKER] Total words: {len(words)}, chunk_size={chunk_size}, overlap={overlap}")
    if not words:
        print("[CHUNKER] No words found — returning empty")
        return chunks

    start = 0
    chunk_id = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)
        if chunk_text_str.strip():
            chunks.append({
                "id": chunk_id,
                "text": chunk_text_str,
            })
            chunk_id += 1
        start += chunk_size - overlap

    print(f"[CHUNKER] Generated {len(chunks)} chunks")
    return chunks
