"""
Monitoring and metrics collection for cost/performance tracking.

Tracks:
- Request counts per subsystem
- Latency distributions
- Cache hit rates
- Token consumption (estimated)
- Error rates
- Estimated API costs
"""

import time
from typing import Dict, List
from collections import defaultdict


class MetricsCollector:
    """Lightweight metrics collector for cost and performance tracking.

    Cost-effective design:
    - Tracks only last 100 samples per metric (memory efficient)
    - Estimates API costs based on token counts
    - Reports cache savings for cost optimization
    """

    def __init__(self):
        self._data = {
            "requests": defaultdict(int),
            "errors": defaultdict(int),
            "latencies": defaultdict(list),
            "cache_hits": defaultdict(int),
            "cache_misses": defaultdict(int),
            "tokens": defaultdict(int),
        }
        self.start_time = time.time()

    def record(self, subsystem: str, duration_ms: float,
               error: bool = False, cache_hit: bool = False,
               tokens: int = 0):
        """Record a metric observation."""
        self._data["requests"][subsystem] += 1
        if error:
            self._data["errors"][subsystem] += 1
        if cache_hit:
            self._data["cache_hits"][subsystem] += 1
        else:
            self._data["cache_misses"][subsystem] += 1

        latencies = self._data["latencies"][subsystem]
        latencies.append(duration_ms)
        if len(latencies) > 100:
            latencies.pop(0)

        self._data["tokens"][subsystem] += tokens

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        stats = {}
        for sub in self._data["requests"]:
            latencies = self._data["latencies"].get(sub, [])
            avg_lat = sum(latencies) / len(latencies) if latencies else 0
            reqs = self._data["requests"][sub]
            errs = self._data["errors"].get(sub, 0)
            hits = self._data["cache_hits"].get(sub, 0)
            misses = self._data["cache_misses"].get(sub, 0)
            total_cache = hits + misses
            cache_rate = (hits / total_cache * 100) if total_cache > 0 else 0

            stats[sub] = {
                "requests": reqs,
                "errors": errs,
                "error_rate": round(errs / reqs * 100, 1) if reqs > 0 else 0,
                "avg_latency_ms": round(avg_lat, 1),
                "cache_hit_rate": round(cache_rate, 1),
                "tokens_estimated": self._data["tokens"].get(sub, 0),
            }

        return stats

    def estimate_cost(self) -> Dict:
        """Estimate API costs based on token usage.

        Approximate rates:
        - Gemini Flash: $0.075/1M input, $0.30/1M output
        - Assumes 50/50 split input/output
        """
        stats = self.get_stats()
        total_tokens = sum(s["tokens_estimated"] for s in stats.values())

        # Rough cost estimation
        input_cost = (total_tokens * 0.5 / 1_000_000) * 0.075
        output_cost = (total_tokens * 0.5 / 1_000_000) * 0.30
        total_cost = input_cost + output_cost

        # Estimate savings from caching
        total_cache_hits = sum(
            s["requests"] * (s["cache_hit_rate"] / 100)
            for s in stats.values()
        )
        savings = total_cost * (total_cache_hits / max(
            sum(s["requests"] for s in stats.values()), 1
        )) if total_cost > 0 else 0

        return {
            "total_tokens_estimated": total_tokens,
            "estimated_cost_usd": round(total_cost, 4),
            "estimated_savings_usd": round(savings, 4),
            "uptime_hours": round((time.time() - self.start_time) / 3600, 1),
        }

    def get_summary(self) -> str:
        """Get a human-readable summary."""
        stats = self.get_stats()
        cost = self.estimate_cost()
        total_reqs = sum(s["requests"] for s in stats.values())
        total_errs = sum(s["errors"] for s in stats.values())

        lines = [
            f"📊 **Performance Summary**",
            f"Requests: {total_reqs} | Errors: {total_errs} | Uptime: {cost['uptime_hours']}h",
            f"Estimated Cost: ${cost['estimated_cost_usd']:.4f} | Savings: ${cost['estimated_savings_usd']:.4f}",
        ]

        for sub, s in sorted(stats.items()):
            lines.append(
                f"  {sub}: {s['requests']} reqs, {s['avg_latency_ms']}ms avg, "
                f"{s['cache_hit_rate']}% cache, {s['error_rate']}% err"
            )

        return "\n".join(lines)
