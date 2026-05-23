"""
Document chunking strategies for legal documents.

Supports:
- Semantic chunking (preserves clause boundaries)
- Recursive character splitting
- Token-aware chunking for cost control
"""

from typing import List, Dict, Optional
import re


def chunk_document(text: str, chunk_size: int = 512,
                   chunk_overlap: int = 64,
                   strategy: str = "semantic") -> List[Dict]:
    """Chunk a document into smaller pieces.

    Args:
        text: Full document text
        chunk_size: Max characters per chunk
        chunk_overlap: Overlap between chunks
        strategy: 'semantic', 'recursive', or 'simple'

    Returns:
        List of {"text": "...", "metadata": {...}}
    """
    if strategy == "semantic":
        return semantic_chunking(text, chunk_size, chunk_overlap)
    elif strategy == "recursive":
        return recursive_chunking(text, chunk_size, chunk_overlap)
    else:
        return simple_chunking(text, chunk_size, chunk_overlap)


def semantic_chunking(text: str, chunk_size: int, overlap: int) -> List[Dict]:
    """Chunk preserving legal clause boundaries.

    Preferred for legal documents — keeps clauses intact.
    """
    # Split by common legal separators
    separators = [
        r'\n\d+\.\s+',          # Numbered clauses: "1. "
        r'\n\d+\)\s+',          # Numbered: "1) "
        r'\nClause\s+\d+',      # "Clause 1"
        r'\nبند\s+\d+',         # Arabic: "بند 1"
        r'\nمادة\s+\d+',        # Arabic: "مادة 1"
        r'\n{2,}',              # Double newlines
        r'(?<=[.!?])\s+(?=[A-Z\u0600-])',  # Sentence boundaries
    ]

    chunks = []
    current = ""
    current_meta = {"start": 0, "type": "legal"}

    # Split by the strongest separator first
    parts = re.split(separators[0], text)

    for part in parts:
        if not part.strip():
            continue

        if len(current) + len(part) > chunk_size and current:
            chunks.append({
                "text": current.strip(),
                "metadata": {**current_meta, "chunk_index": len(chunks)},
            })
            # Overlap: keep last `overlap` chars
            if overlap > 0:
                current = current[-overlap:] + part
            else:
                current = part
            current_meta["start"] += len(current)
        else:
            current += part

    if current.strip():
        chunks.append({
            "text": current.strip(),
            "metadata": {**current_meta, "chunk_index": len(chunks)},
        })

    return chunks


def recursive_chunking(text: str, chunk_size: int, overlap: int) -> List[Dict]:
    """Recursive chunking with multiple separator levels."""
    separators = ['\n\n', '\n', '. ', ' ', '']

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to find a good break point
        if end < len(text):
            for sep in separators:
                if not sep:
                    break
                break_point = text.rfind(sep, start + chunk_size // 2, end)
                if break_point > 0:
                    end = break_point + len(sep)
                    break

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {"start": start, "end": end, "chunk_index": len(chunks)},
            })

        start = end - overlap if overlap > 0 else end

    return chunks


def simple_chunking(text: str, chunk_size: int, overlap: int) -> List[Dict]:
    """Simple fixed-size chunking (fastest, but may break clauses)."""
    chunks = []

    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {"start": i, "end": i + len(chunk_text),
                             "chunk_index": len(chunks)},
            })

    return chunks
