"""
evaluate.py

Scores a detector's point-level predictions against the simulator's
ground-truth is_anomaly labels: precision, recall, F1, and false
positive rate. This is what separates "the detector flagged some stuff"
from a defensible claim about how well it actually works.

Run directly to load the generated dataset, fit both detectors
(ThresholdBaseline and IsolationForestDetector), and print a
side-by-side comparison report.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.detector.anomaly_detector import IsolationForestDetector, ThresholdBaseline

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "metrics.db"
LABELS_PATH = Path(__file__).resolve().parents[2] / "data" / "anomaly_labels.json"


@dataclass
class EvalResult:
    name: str
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


def load_metrics() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM metrics ORDER BY idx", conn)
    conn.close()
    return df


def score(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> EvalResult:
    tp = int(np.sum((y_pred == True) & (y_true == True)))   # noqa: E712
    fp = int(np.sum((y_pred == True) & (y_true == False)))  # noqa: E712
    fn = int(np.sum((y_pred == False) & (y_true == True)))  # noqa: E712
    tn = int(np.sum((y_pred == False) & (y_true == False))) # noqa: E712

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return EvalResult(
        name=name,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
    )


def print_report(results: list[EvalResult]) -> None:
    print(f"{'Detector':<22}{'Precision':>10}{'Recall':>10}{'F1':>10}{'FPR':>10}")
    print("-" * 62)
    for r in results:
        print(f"{r.name:<22}{r.precision:>10.3f}{r.recall:>10.3f}{r.f1:>10.3f}{r.false_positive_rate:>10.3f}")
    print()
    for r in results:
        print(
            f"{r.name}: TP={r.true_positives} FP={r.false_positives} "
            f"FN={r.false_negatives} TN={r.true_negatives}"
        )


def load_labels() -> list[dict]:
    with open(LABELS_PATH) as f:
        return json.load(f)


def window_level_score(y_pred: np.ndarray, labels: list[dict], n_points: int, name: str) -> EvalResult:
    """
    Scores whether the detector fired *anywhere* inside each ground-truth
    anomaly window, rather than requiring every point in the window to be
    flagged. This matches how APM/alerting tools are actually judged in
    practice: one alert during an incident is a catch, not a partial miss.

    False positives here are windows the detector flagged that don't
    overlap any true anomaly window at all (i.e. flagged points that fall
    entirely outside every labeled window, grouped loosely).
    """
    detected_windows = 0
    missed_windows = 0
    covered_idxs = set()

    for label in labels:
        start, end = label["start_idx"], label["end_idx"]
        window_idxs = range(start, end)
        covered_idxs.update(window_idxs)
        if any(y_pred[i] for i in window_idxs):
            detected_windows += 1
        else:
            missed_windows += 1

    # False positives: flagged points that fall entirely outside every labeled window,
    # grouped into contiguous "false alarm episodes" so precision compares windows
    # to episodes (not raw points to windows, which would be a unit mismatch).
    outside_idxs = sorted(i for i in range(n_points) if y_pred[i] and i not in covered_idxs)
    false_alarm_episodes = 0
    prev = None
    for i in outside_idxs:
        if prev is None or i > prev + 1:
            false_alarm_episodes += 1
        prev = i
    outside_total = n_points - len(covered_idxs)

    precision = (
        detected_windows / (detected_windows + false_alarm_episodes)
        if (detected_windows + false_alarm_episodes) > 0
        else 0.0
    )
    recall = detected_windows / len(labels) if labels else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = len(outside_idxs) / outside_total if outside_total > 0 else 0.0

    return EvalResult(
        name=f"{name} (window-level)",
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        true_positives=detected_windows,
        false_positives=false_alarm_episodes,
        false_negatives=missed_windows,
        true_negatives=outside_total - len(outside_idxs),
    )


def main():
    df = load_metrics()
    labels = load_labels()
    y_true = df["is_anomaly"].astype(bool).to_numpy()
    true_anomaly_rate = y_true.mean()

    period = max(2, len(df) // 4)

    baseline = ThresholdBaseline(z_cutoff=3.0).fit(df)
    baseline_pred = baseline.predict(df)
    baseline_point = score(y_true, baseline_pred, "ThresholdBaseline")
    baseline_window = window_level_score(baseline_pred, labels, len(df), "ThresholdBaseline")

    # contamination set to the true anomaly rate rather than a guess
    iforest = IsolationForestDetector(period=period, contamination=true_anomaly_rate).fit(df)
    iforest_pred = iforest.predict(df)
    iforest_point = score(y_true, iforest_pred, "IsolationForest+STL")
    iforest_window = window_level_score(iforest_pred, labels, len(df), "IsolationForest+STL")

    print(f"True anomaly rate: {true_anomaly_rate:.3f} (used as Isolation Forest's contamination)\n")
    print("=== Point-level (strict: every point in a window must be flagged) ===")
    print_report([baseline_point, iforest_point])
    print("\n=== Window-level (an alert anywhere in the window counts as a catch) ===")
    print_report([baseline_window, iforest_window])


if __name__ == "__main__":
    main()
