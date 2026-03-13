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


def chunk_text(text: str, chunk_size_chars: int = 2500, overlap_chars: int = 250) -> List[dict]:
    """Split text into overlapping chunks based on character length."""
    chunks = []
    print(f"[CHUNKER] Total characters: {len(text)}, chunk_size_chars={chunk_size_chars}, overlap_chars={overlap_chars}")
    if not text.strip():
        print("[CHUNKER] No text found — returning empty")
        return chunks

    start = 0
    chunk_id = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size_chars
        
        # If we're not at the end of the text, try to find a natural break (space or newline)
        if end < text_len:
            # Look for the last space/newline within the last 10% of the chunk
            # to avoid cutting words in half
            search_start = max(start, end - int(chunk_size_chars * 0.1))
            last_space = max(text.rfind(' ', search_start, end), 
                             text.rfind('\n', search_start, end))
            
            if last_space != -1:
                end = last_space + 1  # Include the space
                
        chunk_text_str = text[start:end]
        if chunk_text_str.strip():
            chunks.append({
                "id": chunk_id,
                "text": chunk_text_str.strip(),
            })
            chunk_id += 1
            
        start = end - overlap_chars
        # Prevent infinite loops if overlap is somehow larger than the chunk advancement
        if start <= end - chunk_size_chars:
            start = end

    print(f"[CHUNKER] Generated {len(chunks)} chunks")
    return chunks
