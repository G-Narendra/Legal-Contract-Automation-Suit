"""
Hybrid search combining dense (semantic) and sparse (keyword) retrieval.

BM25 for keyword matching + vector search for semantic similarity.
"""

from typing import List, Dict
import math
from collections import Counter
import re


class BM25:
    """Simple BM25 implementation for keyword search.

    No external dependencies — pure Python.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self.avg_doc_len = 0

    def fit(self, documents: List[str]):
        """Fit BM25 on a corpus of documents."""
        self.documents = documents
        self.doc_freqs = [Counter(self._tokenize(doc)) for doc in documents]

        # Document frequency
        df = Counter()
        for freq in self.doc_freqs:
            for term in freq:
                df[term] += 1

        n_docs = len(documents)
        self.idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

        self.avg_doc_len = sum(len(self._tokenize(doc)) for doc in documents) / n_docs if n_docs > 0 else 100

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search documents by BM25 score."""
        query_terms = self._tokenize(query)
        scores = []

        for idx, freq in enumerate(self.doc_freqs):
            score = 0
            doc_len = sum(freq.values())

            for term in query_terms:
                if term in self.idf:
                    tf = freq.get(term, 0)
                    score += (self.idf[term] * tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                    )

            if score > 0:
                scores.append({
                    "index": idx,
                    "text": self.documents[idx][:200],
                    "score": score,
                })

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (English + Arabic)."""
        # Handle Arabic text
        arabic_pattern = r'[\u0600-\u06FF]+'
        english_pattern = r'[a-zA-Z]+'

        tokens = []
        for match in re.finditer(f'({arabic_pattern}|{english_pattern})', text.lower()):
            tokens.append(match.group())

        return tokens


def hybrid_search_dense_bm25(query: str, dense_results: List[Dict],
                               bm25_scores: List[Dict],
                               alpha: float = 0.5) -> List[Dict]:
    """Combine dense and BM25 results with weighted fusion.

    Args:
        query: Search query
        dense_results: Results from vector similarity search
        bm25_scores: Results from BM25 keyword search
        alpha: Weight for dense scores (0 = BM25 only, 1 = dense only)

    Returns:
        Fused and re-ranked results
    """
    # Normalize scores
    def normalize(results, key="score"):
        scores = [r.get(key, 0) for r in results]
        if not scores or max(scores) == 0:
            return results
        max_score = max(scores)
        for r in results:
            r["normalized_score"] = r.get(key, 0) / max_score
        return results

    dense_results = normalize(dense_results, "score")
    bm25_scores = normalize(bm25_scores, "score")

    # Map texts to combined scores
    combined = {}
    for r in dense_results:
        text = r.get("text", "")
        combined[text] = {
            "text": text,
            "dense_score": r.get("normalized_score", 0),
            "bm25_score": 0,
            "metadata": r.get("metadata", {}),
        }
    for r in bm25_scores:
        text = r.get("text", "")
        if text in combined:
            combined[text]["bm25_score"] = r.get("normalized_score", 0)
        else:
            combined[text] = {
                "text": text,
                "dense_score": 0,
                "bm25_score": r.get("normalized_score", 0),
                "metadata": {},
            }

    # Calculate fused scores
    results = []
    for text, data in combined.items():
        fused = alpha * data["dense_score"] + (1 - alpha) * data["bm25_score"]
        results.append({
            "text": text,
            "score": round(fused, 4),
            "dense_score": data["dense_score"],
            "bm25_score": data["bm25_score"],
            "metadata": data["metadata"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
