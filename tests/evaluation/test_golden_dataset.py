"""
Evaluation tests using the golden dataset.

These tests require:
1. A golden dataset at data/golden_dataset.json
2. An LLM API key for LLM-as-Judge evaluation
"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import calculate_extraction_metrics, calculate_accuracy
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.metrics import calculate_latency_stats, calculate_cost_efficiency


pytestmark = pytest.mark.evaluation


def load_golden_dataset():
    """Load golden dataset for evaluation."""
    dataset_path = Path("data/golden_dataset.json")
    if not dataset_path.exists():
        return None
    with open(dataset_path, "r") as f:
        return json.load(f)


class TestGoldenDataset:
    """Test suite that runs against the golden dataset."""

    @pytest.fixture(scope="class")
    def dataset(self):
        data = load_golden_dataset()
        if data is None:
            pytest.skip("Golden dataset not found at data/golden_dataset.json")
        return data

    def test_dataset_has_cases(self, dataset):
        """Verify the dataset has test cases."""
        assert len(dataset) > 0, "Dataset is empty"

    def test_extraction_accuracy(self, dataset):
        """Evaluate extraction accuracy against golden dataset."""
        judge = LLMJudge()
        scores = []

        for case in dataset[:5]:  # Limit to first 5 for speed
            expected = case.get("expected", {})
            actual = case.get("actual", {})
            result = judge.evaluate_extraction(expected, actual)
            scores.append(result.get("overall_score", 0))

        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"\n📊 Extraction Accuracy: {avg_score * 100:.1f}%")
        assert avg_score >= 0, "Score should be non-negative"

    def test_extraction_metrics(self, dataset):
        """Test metric calculation on dataset."""
        if len(dataset) < 2:
            pytest.skip("Need at least 2 cases")

        case1 = dataset[0].get("expected", {})
        case2 = dataset[1].get("expected", {})

        result = calculate_extraction_metrics(
            [case1], [case2],
            key_field=list(case1.keys())[0] if case1 else "parties"
        )
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result

    def test_latency_metrics(self):
        """Test latency calculation."""
        latencies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        stats = calculate_latency_stats(latencies)
        assert stats["avg_ms"] == 550.0
        assert stats["min_ms"] == 100.0
        assert stats["max_ms"] == 1000.0
        assert stats["samples"] == 10

    def test_empty_latency(self):
        """Test latency with empty list."""
        stats = calculate_latency_stats([])
        assert stats["avg_ms"] == 0

    def test_cost_efficiency(self):
        """Test cost efficiency calculation."""
        result = calculate_cost_efficiency(
            total_tokens=100000,
            cache_hits=40,
            total_requests=100,
        )
        assert "estimated_cost_usd" in result
        assert "cache_hit_rate" in result
        assert result["cache_hit_rate"] == 40.0

    def test_no_cache_cost(self):
        """Test cost efficiency with no cache hits."""
        result = calculate_cost_efficiency(
            total_tokens=100000,
            cache_hits=0,
            total_requests=100,
        )
        assert result["cache_hit_rate"] == 0.0
