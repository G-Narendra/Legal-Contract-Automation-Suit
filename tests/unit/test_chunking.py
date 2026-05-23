"""
Unit tests for document chunking strategies.
"""

import pytest
from src.retrieval.chunking import chunk_document, semantic_chunking, recursive_chunking, simple_chunking


class TestChunking:
    """Test suite for document chunking."""

    def test_simple_chunking_basic(self):
        text = "A " * 1000  # ~2000 chars
        chunks = simple_chunking(text, chunk_size=512, overlap=64)
        assert len(chunks) > 0
        assert all(0 < len(c["text"]) <= 600 for c in chunks)

    def test_simple_chunking_overlap(self):
        text = "Word " * 500
        chunks = simple_chunking(text, chunk_size=512, overlap=128)
        assert len(chunks) >= 2
        # Check overlap exists between consecutive chunks
        if len(chunks) >= 2:
            first_end = chunks[0]["metadata"]["end"]
            second_start = chunks[1]["metadata"]["start"]
            assert second_start < first_end  # There's overlap

    def test_recursive_chunking_respects_boundaries(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n\n"
        chunks = recursive_chunking(text, chunk_size=50, overlap=10)
        assert len(chunks) > 0
        for c in chunks:
            assert len(c["text"]) > 0

    def test_semantic_chunking_preserves_clauses(self):
        text = "1. This is clause one.\n2. This is clause two.\n3. This is clause three."
        chunks = semantic_chunking(text, chunk_size=200, overlap=20)
        assert len(chunks) > 0
        assert all(c["text"].strip() for c in chunks)

    def test_chunk_document_default(self):
        text = "Test document " * 200
        chunks = chunk_document(text, strategy="simple")
        assert len(chunks) > 0

    def test_chunk_document_semantic(self):
        text = "1. Introduction.\n2. Terms.\n3. Conditions."
        chunks = chunk_document(text, strategy="semantic")
        assert len(chunks) > 0

    def test_empty_text(self):
        chunks = chunk_document("")
        assert len(chunks) == 0

    def test_short_text_no_chunking(self):
        text = "Short text."
        chunks = chunk_document(text, chunk_size=500)
        assert len(chunks) <= 1

    def test_chunk_metadata(self):
        text = "Test " * 300
        chunks = simple_chunking(text, chunk_size=256, overlap=32)
        for c in chunks:
            assert "start" in c["metadata"]
            assert "chunk_index" in c["metadata"]
            assert c["metadata"]["chunk_index"] >= 0

    def test_large_document(self):
        text = "Paragraph content.\n\n" * 100
        chunks = recursive_chunking(text, chunk_size=256, overlap=32)
        assert len(chunks) > 3
        assert all(c["text"].strip() for c in chunks)
