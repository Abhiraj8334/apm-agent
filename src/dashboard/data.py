"""
data.py

Data-loading functions for the dashboard, kept free of any Streamlit
import so they can be unit tested directly (Streamlit apps are awkward
to test in isolation; plain functions aren't).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "metrics.db"
LABELS_PATH = Path(__file__).resolve().parents[2] / "data" / "anomaly_labels.json"

METRIC_COLUMNS = ["latency_ms", "cpu_pct", "memory_pct", "error_rate", "throughput_rps"]


def data_available() -> bool:
    """True if the simulator has already been run and data exists on disk."""
    return DB_PATH.exists() and LABELS_PATH.exists()


def load_metrics_df() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found — run `python -m src.simulator.metrics_simulator` first."
        )
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM metrics ORDER BY idx", conn)
    conn.close()
    return df


def load_labels() -> list[dict]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"{LABELS_PATH} not found — run `python -m src.simulator.metrics_simulator` first."
        )
    with open(LABELS_PATH) as f:
        return json.load(f)


def label_summary_rows(labels: list[dict]) -> list[dict]:
    """
    Flatten labels into simple, display-ready rows: one per anomaly window,
    with a human-readable index range. Kept separate from the raw label
    dicts so the dashboard's selectbox can show something readable without
    duplicating this formatting logic inline in the Streamlit file.
    """
    return [
        {
            "id": label["anomaly_id"],
            "type": label["anomaly_type"],
            "window": f"{label['start_idx']}–{label['end_idx']}",
            "start_idx": label["start_idx"],
            "end_idx": label["end_idx"],
        }
        for label in labels
    ]
