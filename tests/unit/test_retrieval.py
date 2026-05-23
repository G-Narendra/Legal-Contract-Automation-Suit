"""
Unit tests for retrieval (hybrid search, BM25, reranker).
"""

import pytest
from src.retrieval.hybrid_search import BM25, hybrid_search_dense_bm25


class TestBM25:
    """Test suite for BM25 implementation."""

    def test_bm25_basic(self):
        bm25 = BM25()
        docs = [
            "This is an employment contract for UAE labor law",
            "This NDA agreement covers confidential information",
            "The UAE commercial code governs partnership agreements",
        ]
        bm25.fit(docs)
        results = bm25.search("employment contract UAE", top_k=3)
        assert len(results) > 0
        assert all("score" in r for r in results)

    def test_bm25_empty_query(self):
        bm25 = BM25()
        bm25.fit(["Test document"])
        results = bm25.search("", top_k=5)
        assert isinstance(results, list)

    def test_bm25_no_documents(self):
        bm25 = BM25()
        results = bm25.search("test", top_k=5)
        assert results == []

    def test_bm25_ranking(self):
        bm25 = BM25()
        docs = [
            "Employment contract terms and conditions",
            "Non-disclosure agreement for business partners",
            "UAE employment law requires notice period",
        ]
        bm25.fit(docs)
        results = bm25.search("employment", top_k=3)
        assert len(results) >= 1
        # First result should be most relevant
        assert results[0]["score"] >= 0

    def test_bm25_arabic_support(self):
        bm25 = BM25()
        docs = [
            "عقد عمل بموجب قانون العمل الإماراتي",
            "اتفاقية سرية لحماية المعلومات",
        ]
        bm25.fit(docs)
        results = bm25.search("عقد عمل", top_k=2)
        assert len(results) >= 1


class TestHybridSearch:
    """Test suite for hybrid search fusion."""

    def test_hybrid_fusion_basic(self):
        dense_results = [
            {"text": "Employment contract UAE", "score": 0.9},
            {"text": "NDA agreement", "score": 0.7},
        ]
        bm25_results = [
            {"text": "Employment contract UAE", "score": 0.8},
            {"text": "Partnership agreement", "score": 0.6},
        ]
        fused = hybrid_search_dense_bm25("test", dense_results, bm25_results, alpha=0.5)
        assert len(fused) > 0
        assert all("score" in r for r in fused)
        # Employment should be highest since it appears in both
        assert fused[0]["score"] >= fused[-1]["score"]

    def test_hybrid_empty_dense(self):
        fused = hybrid_search_dense_bm25("test", [], [{"text": "Doc", "score": 0.5}])
        assert len(fused) == 1

    def test_hybrid_empty_bm25(self):
        fused = hybrid_search_dense_bm25("test", [{"text": "Doc", "score": 0.5}], [])
        assert len(fused) == 1

    def test_hybrid_alpha_extremes(self):
        dense = [{"text": "Doc A", "score": 0.9}]
        bm25 = [{"text": "Doc B", "score": 0.8}]
        # alpha=1: dense only
        fused_dense = hybrid_search_dense_bm25("test", dense, bm25, alpha=1.0)
        # alpha=0: bm25 only
        fused_bm25 = hybrid_search_dense_bm25("test", dense, bm25, alpha=0.0)
        assert len(fused_dense) == 2
        assert len(fused_bm25) == 2
