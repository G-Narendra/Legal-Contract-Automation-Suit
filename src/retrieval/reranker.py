"""
Cross-encoder reranker for improved retrieval precision.

Reranks initial search results using a cross-encoder model.
Falls back gracefully if model is not available.
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger("reranker")


class CrossEncoderReranker:
    """Cross-encoder based reranker for legal search results.

    Improves retrieval precision by scoring query-document pairs.
    Uses lightweight model (ms-marco-MiniLM) for speed.

    Cost-effective:
    - Only reranks top-N candidates (default: 20)
    - Falls back to identity reranking if model unavailable
    - Caches scores for repeated queries
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._cache = {}

    def rerank(self, query: str, candidates: List[Dict],
               top_k: int = 5) -> List[Dict]:
        """Rerank candidate documents by relevance to query.

        Args:
            query: The search query
            candidates: List of {"text": "...", "score": ...} from initial search
            top_k: Number of top results to return

        Returns:
            Reranked list of candidates with updated scores
        """
        if not candidates:
            return []

        if self._model is None:
            self._lazy_load()

        if self._model is None:
            # Fallback: return as-is (identity reranking)
            logger.info("Reranker not available — returning candidates as-is")
            return candidates[:top_k]

        # Prepare pairs
        pairs = [(query, c.get("text", "")) for c in candidates]

        # Check cache
        uncached_indices = []
        cached_scores = [None] * len(candidates)
        for i, pair in enumerate(pairs):
            cache_key = f"{pair[0][:100]}:{pair[1][:100]}"
            if cache_key in self._cache:
                cached_scores[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)

        # Score uncached pairs
        if uncached_indices:
            uncached_pairs = [pairs[i] for i in uncached_indices]
            try:
                scores = self._model.predict(uncached_pairs)
                for idx, score in zip(uncached_indices, scores.tolist()):
                    score_val = float(score)
                    cached_scores[idx] = score_val
                    cache_key = f"{pairs[idx][0][:100]}:{pairs[idx][1][:100]}"
                    self._cache[cache_key] = score_val
            except Exception as e:
                logger.warning(f"Reranker prediction error: {e}")
                return candidates[:top_k]

        # Update scores
        for i, candidate in enumerate(candidates):
            if cached_scores[i] is not None:
                candidate["reranker_score"] = cached_scores[i]
                candidate["score"] = cached_scores[i]

        # Sort by reranker score
        reranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return reranked[:top_k]

    def _lazy_load(self):
        """Lazy-load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info(f"Loaded reranker: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed — reranker disabled")
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}")

    def clear_cache(self):
        """Clear the scoring cache."""
        self._cache.clear()
