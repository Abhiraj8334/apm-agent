"""
run_rca.py

End-to-end demo: load metrics, find one flagged anomaly window, and run
the RCA agent against it using Gemini. Prints the full reasoning trace
followed by the final diagnosis.

Usage:
    python -m src.agent.run_rca
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.agent.agent import AnomalyContext, run_rca_agent
from src.agent.llm_client import GeminiClient

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "metrics.db"
LABELS_PATH = Path(__file__).resolve().parents[2] / "data" / "anomaly_labels.json"

METRIC_COLUMNS = ["latency_ms", "cpu_pct", "memory_pct", "error_rate", "throughput_rps"]


def load_first_labeled_anomaly() -> AnomalyContext:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM metrics ORDER BY idx", conn)
    conn.close()

    with open(LABELS_PATH) as f:
        labels = json.load(f)
    if not labels:
        raise RuntimeError("No anomaly labels found — run the simulator first.")

    label = labels[0]
    window = df[(df["idx"] >= label["start_idx"]) & (df["idx"] < label["end_idx"])]
    baseline = df[~df["idx"].between(label["start_idx"], label["end_idx"])]

    metric_col = {
        "cpu_spike": "cpu_pct",
        "latency_spike": "latency_ms",
        "memory_leak": "memory_pct",
        "error_burst": "error_rate",
    }.get(label["anomaly_type"], "cpu_pct")

    peak_row = window.loc[window[metric_col].idxmax()]

    return AnomalyContext(
        anomaly_type=label["anomaly_type"],
        metric=metric_col,
        start_idx=label["start_idx"],
        end_idx=label["end_idx"],
        peak_values={col: float(peak_row[col]) for col in METRIC_COLUMNS},
        baseline_values={col: float(baseline[col].mean()) for col in METRIC_COLUMNS},
    )


def main():
    load_dotenv()  # reads GEMINI_API_KEY (and optional GEMINI_MODEL) from .env
    ctx = load_first_labeled_anomaly()
    print(f"Investigating anomaly: {ctx.anomaly_type} on {ctx.metric} (idx {ctx.start_idx}-{ctx.end_idx})\n")

    print("=== Reasoning trace (live) ===")
    llm = GeminiClient()  # requires GEMINI_API_KEY in your .env
    result = run_rca_agent(ctx, llm)

    print("\n=== Final diagnosis ===")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Stopped because: {result.stopped_reason}")
    print(result.diagnosis)


if __name__ == "__main__":
    main()
