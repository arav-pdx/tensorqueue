"""
Latency tracking shared between worker (writer) and API (reader).

Samples are pushed into a capped Redis list so percentile stats are visible
cluster-wide (across all worker pods) without standing up a Prometheus
stack just to demo the project. In a real deployment, `record_prometheus`
below also exposes a proper Histogram on /metrics for Prometheus/HPA
(KEDA/custom-metrics) to scrape.
"""
from __future__ import annotations

import math

import redis

from .config import settings

LATENCY_KEY = "tensorqueue:latencies_ms"

try:
    from prometheus_client import Histogram

    INFERENCE_LATENCY = Histogram(
        "tensorqueue_inference_latency_ms",
        "End-to-end inference latency per job, in milliseconds",
        buckets=(5, 10, 20, 35, 50, 75, 100, 150, 250, 500, 1000, 2500),
    )
except ImportError:  # prometheus_client optional for the API-only image
    INFERENCE_LATENCY = None


def record_latency(r: redis.Redis, latency_ms: float) -> None:
    r.lpush(LATENCY_KEY, latency_ms)
    r.ltrim(LATENCY_KEY, 0, settings.latency_window_size - 1)
    if INFERENCE_LATENCY is not None:
        INFERENCE_LATENCY.observe(latency_ms)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def read_percentiles(r: redis.Redis) -> dict:
    raw = r.lrange(LATENCY_KEY, 0, -1)
    values = sorted(float(v) for v in raw)
    return {
        "p50_latency_ms": _percentile(values, 50) if values else None,
        "p95_latency_ms": _percentile(values, 95) if values else None,
        "p99_latency_ms": _percentile(values, 99) if values else None,
        "sample_count": len(values),
    }
