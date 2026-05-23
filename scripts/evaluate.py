"""
Evaluate — Runs evaluation against the golden dataset.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --dataset data/golden_dataset.json --verbose
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import calculate_extraction_metrics
from src.evaluation.llm_judge import LLMJudge
from src.core.config import load_config
from src.core.llm_base import LLMFactory


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate system against golden dataset")
    parser.add_argument("--dataset", default="data/golden_dataset.json", help="Path to golden dataset")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"❌ Golden dataset not found: {dataset_path}")
        print("   Create data/golden_dataset.json with test cases.")
        sys.exit(1)

    with open(dataset_path, "r") as f:
        test_cases = json.load(f)

    print(f"📊 Evaluating {len(test_cases)} test cases from {dataset_path.name}")
    print("=" * 60)

    config = load_config()
    api_key = config.get("gemini_api_key", "")
    if api_key:
        llm = LLMFactory.create("google", api_key, config.get("model", "gemini-2.5-flash"))
        judge = LLMJudge(llm=llm)
    else:
        print("⚠️  No API key — using rule-based evaluation only")
        judge = LLMJudge()

    # Track overall metrics
    all_scores = []
    correct = 0
    total = len(test_cases)

    for i, case in enumerate(test_cases):
        expected = case.get("expected", {})
        actual = case.get("actual", {})
        case_name = case.get("name", f"Case {i+1}")

        # Evaluate extraction accuracy
        result = judge.evaluate_extraction(expected, actual)
        score = result.get("overall_score", 0)
        all_scores.append(score)

        if score >= 0.8:
            correct += 1

        if args.verbose:
            print(f"\n📋 {case_name}:")
            print(f"   Score: {score * 100:.1f}%")
            if result.get("issues"):
                for issue in result["issues"][:3]:
                    print(f"   ⚠️  {issue}")

    # Summary
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 60)
    print(f"📈 Evaluation Summary")
    print(f"   Average Score: {avg_score * 100:.1f}%")
    print(f"   Accuracy (≥80%): {accuracy * 100:.1f}%")
    print(f"   Passed: {correct}/{total}")
    print(f"   Method: {result.get('method', 'N/A')}")

    # Export results
    results_path = Path("data/evaluation_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "avg_score": round(avg_score, 3),
            "accuracy": round(accuracy, 3),
            "correct": correct,
            "total": total,
            "method": result.get("method", "N/A"),
        }, f, indent=2)

    print(f"\n✅ Results exported to {results_path}")


if __name__ == "__main__":
    main()
