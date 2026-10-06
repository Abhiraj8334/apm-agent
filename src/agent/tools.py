"""
tools.py

Evidence-gathering tools the RCA agent can call while investigating an
anomaly. These are mocked/synthetic (no real log aggregator, deployment
system, or runbook wiki behind them) — the point is to demonstrate genuine
multi-step tool-calling behavior, not to wire up real infrastructure.
Swapping these for real implementations (an Elasticsearch query, a GitHub
deploys API call, a Confluence lookup) would not require changing the
agent's reasoning loop at all — that's the intended seam.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

# Static knowledge a real team would keep in a runbook wiki.
_RUNBOOK: dict[str, str] = {
    "cpu_spike": (
        "CPU spikes are most often caused by: (1) an inefficient query or "
        "N+1 pattern introduced in a recent deploy, (2) a retry storm from "
        "a downstream dependency timing out, or (3) a scheduled batch job "
        "overlapping with peak traffic. Check recent deploys and downstream "
        "dependency health first."
    ),
    "latency_spike": (
        "Latency spikes usually trace to: (1) connection pool exhaustion "
        "under load, (2) a slow downstream dependency, or (3) GC pauses "
        "under memory pressure. Correlate with CPU/memory in the same window."
    ),
    "memory_leak": (
        "Gradual memory growth typically indicates: (1) an unbounded cache "
        "or in-memory collection, (2) unclosed connections/file handles, or "
        "(3) a recent deploy that introduced a retained-object bug. Check "
        "deploy history against when the growth started."
    ),
    "error_burst": (
        "Error bursts commonly stem from: (1) a bad deploy, (2) a downstream "
        "dependency outage, or (3) a database connection pool exhausted by "
        "a traffic spike. Check deploy history and downstream status first."
    ),
}

# Deterministic-per-run synthetic deploy history so the agent has something
# concrete to correlate against.
_rng = random.Random(7)
_DEPLOY_HISTORY = [
    {"service": "checkout-api", "version": "v2.14.1", "minutes_ago": 42},
    {"service": "user-service", "version": "v1.8.0", "minutes_ago": 210},
    {"service": "checkout-api", "version": "v2.14.0", "minutes_ago": 340},
]


def check_runbook(anomaly_type: str) -> str:
    """Look up known causes and remediation guidance for a given anomaly type."""
    return _RUNBOOK.get(
        anomaly_type,
        f"No runbook entry found for anomaly type '{anomaly_type}'.",
    )


def query_recent_logs(metric_name: str, start_idx: int, end_idx: int) -> list[str]:
    """
    Return synthetic log lines that would plausibly appear during this
    anomaly window. In a real system this would query an aggregator
    (Elasticsearch, Loki, CloudWatch Logs) for the same time range.
    """
    window_size = end_idx - start_idx
    templates = {
        "latency_ms": [
            "WARN  connection-pool: waited 480ms for available connection",
            "ERROR downstream-client: timeout calling payments-service (2400ms)",
        ],
        "cpu_pct": [
            "WARN  scheduler: batch-reconciliation-job overlapping with peak traffic",
            "INFO  query-planner: full table scan detected on orders (12.4M rows)",
        ],
        "memory_pct": [
            "WARN  gc: full GC pause 1200ms, heap 94% utilized",
            "INFO  cache: in-memory session cache size grew to 2.1GB (no eviction configured)",
        ],
        "error_rate": [
            "ERROR db-pool: connection pool exhausted, 0 available of 50",
            "ERROR downstream-client: 503 from inventory-service",
        ],
        "throughput_rps": [
            "WARN  rate-limiter: shedding load, queue depth exceeded threshold",
        ],
    }
    pool = templates.get(metric_name, ["INFO  no metric-specific log pattern available"])
    n = min(len(pool), max(1, window_size // 5))
    return _rng.sample(pool, n) if n <= len(pool) else pool


def get_correlated_metrics(all_metrics: dict[str, float], exclude: str) -> dict[str, float]:
    """
    Given the current point's values across all metrics, return the ones
    other than the one that triggered the alert — lets the agent check
    whether other metrics moved together (a real correlation) or stayed
    normal (suggesting the anomaly is isolated to one metric).
    """
    return {k: v for k, v in all_metrics.items() if k != exclude}


def get_deployment_history(lookback_minutes: int = 360) -> list[dict]:
    """Return recent deploys within the lookback window, most recent first."""
    return sorted(
        (d for d in _DEPLOY_HISTORY if d["minutes_ago"] <= lookback_minutes),
        key=lambda d: d["minutes_ago"],
    )


TOOL_REGISTRY = {
    "check_runbook": check_runbook,
    "query_recent_logs": query_recent_logs,
    "get_correlated_metrics": get_correlated_metrics,
    "get_deployment_history": get_deployment_history,
}
