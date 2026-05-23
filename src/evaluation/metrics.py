"""
Evaluation metrics for legal contract automation.

Provides:
- Precision, recall, F1 for extraction
- Accuracy for classification
- Latency and cost tracking
- Quality scores for drafting
"""

from typing import Dict, List, Any


def calculate_extraction_metrics(expected: List[Dict],
                                  actual: List[Dict],
                                  key_field: str = "parties") -> Dict:
    """Calculate precision, recall, F1 for extracted fields."""
    expected_values = {str(e.get(key_field, "")) for e in expected if e.get(key_field)}
    actual_values = {str(a.get(key_field, "")) for a in actual if a.get(key_field)}

    true_positives = len(expected_values & actual_values)
    false_positives = len(actual_values - expected_values)
    false_negatives = len(expected_values - actual_values)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def calculate_accuracy(correct: int, total: int) -> Dict:
    """Calculate accuracy score."""
    return {
        "accuracy": round(correct / total, 3) if total > 0 else 0,
        "correct": correct,
        "total": total,
    }


def calculate_latency_stats(latencies: List[float]) -> Dict:
    """Calculate latency statistics."""
    if not latencies:
        return {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "min_ms": 0, "max_ms": 0}

    sorted_lats = sorted(latencies)
    n = len(sorted_lats)

    return {
        "avg_ms": round(sum(sorted_lats) / n, 1),
        "p50_ms": round(sorted_lats[int(n * 0.5)], 1),
        "p95_ms": round(sorted_lats[int(n * 0.95)], 1),
        "p99_ms": round(sorted_lats[int(n * 0.99)], 1),
        "min_ms": round(sorted_lats[0], 1),
        "max_ms": round(sorted_lats[-1], 1),
        "samples": n,
    }


def calculate_cost_efficiency(total_tokens: int, cache_hits: int,
                               total_requests: int) -> Dict:
    """Calculate cost efficiency metrics."""
    cost_per_1m_tokens = 0.075  # Approximate for Gemini Flash input
    estimated_cost = (total_tokens / 1_000_000) * cost_per_1m_tokens

    cache_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
    estimated_savings = estimated_cost * (cache_rate / 100)

    return {
        "estimated_cost_usd": round(estimated_cost, 4),
        "estimated_savings_usd": round(estimated_savings, 4),
        "cache_hit_rate": round(cache_rate, 1),
        "cost_per_request": round(estimated_cost / total_requests, 6) if total_requests > 0 else 0,
    }
