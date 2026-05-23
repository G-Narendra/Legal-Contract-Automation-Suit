"""
Benchmark Script — Tests system performance and latency.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --iterations 20 --providers google,openai
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import load_config
from src.core.llm_base import LLMFactory
from src.utils.monitoring import MetricsCollector


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark legal automation suite")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per test")
    parser.add_argument("--providers", default="google", help="Comma-separated providers to test")
    return parser.parse_args()


def run_benchmark():
    args = parse_args()
    config = load_config()
    metrics = MetricsCollector()
    providers = [p.strip() for p in args.providers.split(",")]

    print(f"🚀 Benchmarking {len(providers)} providers × {args.iterations} iterations")
    print("=" * 60)

    test_prompts = [
        "Analyze this employment contract clause: The employee shall receive 30 days annual leave.",
        "Draft a non-disclosure agreement between Company A and Company B for UAE jurisdiction.",
        "List the key compliance requirements for UAE employment contracts under Federal Decree-Law No. 33 of 2021.",
    ]

    for provider in providers:
        print(f"\n📊 Testing Provider: {provider}")
        print("-" * 40)

        api_key = config.get(f"{provider}_api_key", "")
        if not api_key:
            print(f"⚠️  No API key for {provider}, skipping...")
            continue

        llm = LLMFactory.create(
            provider=provider,
            api_key=api_key,
            model=config.get("model", "gemini-2.5-flash"),
            temperature=0.1,
        )

        for prompt in test_prompts:
            latencies = []
            print(f"  Prompt: {prompt[:50]}...")

            for i in range(args.iterations):
                start = time.time()
                try:
                    result = llm.generate(prompt)
                    duration = (time.time() - start) * 1000
                    latencies.append(duration)
                    metrics.record(f"{provider}_benchmark", duration, tokens=len(prompt) // 4 + len(result) // 4)
                except Exception as e:
                    print(f"    ✗ Iteration {i+1}: Error - {e}")
                    continue

            if latencies:
                avg = sum(latencies) / len(latencies)
                p95 = sorted(latencies)[int(len(latencies) * 0.95)]
                print(f"    Avg: {avg:.0f}ms | P95: {p95:.0f}ms | Samples: {len(latencies)}")

    # Summary
    print("\n" + "=" * 60)
    print("📈 Benchmark Summary")
    print(metrics.get_summary())


if __name__ == "__main__":
    run_benchmark()
