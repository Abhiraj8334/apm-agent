"""
metrics_simulator.py

Generates a realistic synthetic time series of application performance
metrics (latency, CPU, memory, error rate, throughput) with injectable
anomalies. Every injected anomaly is recorded with a ground-truth label
(type, start, end) so downstream detectors can be scored against a real
answer key instead of eyeballed.

Run directly to generate a default dataset:
    python -m src.simulator.metrics_simulator
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "metrics.db"
LABELS_PATH = Path(__file__).resolve().parents[2] / "data" / "anomaly_labels.json"


@dataclass
class AnomalyLabel:
    anomaly_id: int
    anomaly_type: str          # "cpu_spike" | "latency_spike" | "memory_leak" | "error_burst"
    start_idx: int
    end_idx: int
    description: str


@dataclass
class MetricPoint:
    idx: int
    timestamp: str
    latency_ms: float
    cpu_pct: float
    memory_pct: float
    error_rate: float          # fraction 0-1
    throughput_rps: float
    is_anomaly: bool           # ground truth, for evaluation only — not fed to the detector


class MetricsSimulator:
    """
    Generates a baseline "healthy" system with daily seasonality and
    natural noise, then injects a configurable set of anomaly types at
    random (non-overlapping) windows.
    """

    def __init__(self, n_points: int = 2000, seed: int = 42, interval_seconds: int = 30):
        self.n_points = n_points
        self.interval_seconds = interval_seconds
        self.rng = random.Random(seed)
        self.start_time = datetime.now(timezone.utc) - timedelta(seconds=n_points * interval_seconds)

    # ---------- baseline generation ----------

    def _seasonal(self, idx: int, period: int, amplitude: float) -> float:
        return amplitude * math.sin(2 * math.pi * idx / period)

    def _baseline_point(self, idx: int) -> dict:
        # Daily-ish seasonality compressed into the sim window + gaussian noise
        period = max(50, self.n_points // 4)
        latency = 120 + self._seasonal(idx, period, 15) + self.rng.gauss(0, 5)
        cpu = 35 + self._seasonal(idx, period, 10) + self.rng.gauss(0, 3)
        memory = 45 + (idx / self.n_points) * 2 + self.rng.gauss(0, 1.5)  # slow natural drift
        error_rate = max(0.0, 0.01 + self.rng.gauss(0, 0.003))
        throughput = 200 + self._seasonal(idx, period, 40) + self.rng.gauss(0, 10)
        return dict(
            latency_ms=max(1.0, latency),
            cpu_pct=min(100.0, max(0.0, cpu)),
            memory_pct=min(100.0, max(0.0, memory)),
            error_rate=min(1.0, error_rate),
            throughput_rps=max(0.0, throughput),
        )

    # ---------- anomaly injection ----------

    def _pick_windows(self, n_windows: int, min_len: int, max_len: int) -> list[tuple[int, int]]:
        """Pick non-overlapping [start, end) windows within the series."""
        windows = []
        attempts = 0
        while len(windows) < n_windows and attempts < 500:
            attempts += 1
            length = self.rng.randint(min_len, max_len)
            start = self.rng.randint(50, self.n_points - length - 50)
            end = start + length
            if all(end < w[0] - 20 or start > w[1] + 20 for w in windows):
                windows.append((start, end))
        return sorted(windows)

    def _apply_anomaly(self, point: dict, anomaly_type: str, progress: float) -> dict:
        """
        progress: 0.0 -> 1.0 position within the anomaly window, used so
        gradual anomalies (e.g. memory_leak) ramp rather than step-change.
        """
        p = dict(point)
        if anomaly_type == "cpu_spike":
            p["cpu_pct"] = min(100.0, p["cpu_pct"] + 45 + self.rng.gauss(0, 5))
            p["latency_ms"] += 30 * progress
        elif anomaly_type == "latency_spike":
            p["latency_ms"] += 250 + self.rng.gauss(0, 20)
            p["throughput_rps"] *= 0.6
        elif anomaly_type == "memory_leak":
            # Gradual ramp rather than instant jump — this is the "hard" case
            # for a naive threshold detector and the reason STL/drift-aware
            # detection matters later.
            p["memory_pct"] = min(99.0, p["memory_pct"] + 40 * progress)
        elif anomaly_type == "error_burst":
            p["error_rate"] = min(1.0, p["error_rate"] + 0.25 + self.rng.gauss(0, 0.05))
            p["throughput_rps"] *= 0.8
        return p

    def generate(
        self,
        anomaly_counts: dict[str, int] | None = None,
    ) -> tuple[list[MetricPoint], list[AnomalyLabel]]:
        anomaly_counts = anomaly_counts or {
            "cpu_spike": 3,
            "latency_spike": 3,
            "memory_leak": 2,
            "error_burst": 2,
        }

        # Build all injected windows up front so we can label ground truth precisely.
        labels: list[AnomalyLabel] = []
        window_map: dict[int, tuple[str, int, int]] = {}  # idx -> (type, start, end)
        anomaly_id = 0
        for a_type, count in anomaly_counts.items():
            min_len, max_len = (5, 15) if a_type != "memory_leak" else (20, 40)
            for (start, end) in self._pick_windows(count, min_len, max_len):
                anomaly_id += 1
                labels.append(
                    AnomalyLabel(
                        anomaly_id=anomaly_id,
                        anomaly_type=a_type,
                        start_idx=start,
                        end_idx=end,
                        description=f"{a_type} injected over idx [{start}, {end})",
                    )
                )
                for i in range(start, end):
                    window_map[i] = (a_type, start, end)

        points: list[MetricPoint] = []
        for idx in range(self.n_points):
            base = self._baseline_point(idx)
            is_anomaly = idx in window_map
            if is_anomaly:
                a_type, start, end = window_map[idx]
                progress = (idx - start) / max(1, (end - start))
                base = self._apply_anomaly(base, a_type, progress)

            ts = self.start_time + timedelta(seconds=idx * self.interval_seconds)
            points.append(
                MetricPoint(
                    idx=idx,
                    timestamp=ts.isoformat(),
                    is_anomaly=is_anomaly,
                    **base,
                )
            )

        return points, labels

    # ---------- persistence ----------

    def save(self, points: list[MetricPoint], labels: list[AnomalyLabel]) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DROP TABLE IF EXISTS metrics")
        conn.execute(
            """
            CREATE TABLE metrics (
                idx INTEGER PRIMARY KEY,
                timestamp TEXT,
                latency_ms REAL,
                cpu_pct REAL,
                memory_pct REAL,
                error_rate REAL,
                throughput_rps REAL,
                is_anomaly INTEGER
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO metrics VALUES (:idx, :timestamp, :latency_ms, :cpu_pct,
                                         :memory_pct, :error_rate, :throughput_rps, :is_anomaly)
            """,
            [{**asdict(p), "is_anomaly": int(p.is_anomaly)} for p in points],
        )
        conn.commit()
        conn.close()

        with open(LABELS_PATH, "w") as f:
            json.dump([asdict(l) for l in labels], f, indent=2)


def main():
    sim = MetricsSimulator(n_points=2000, seed=42)
    points, labels = sim.generate()
    sim.save(points, labels)
    print(f"Generated {len(points)} points, {len(labels)} injected anomaly windows.")
    print(f"  -> {DB_PATH}")
    print(f"  -> {LABELS_PATH}")
    anomaly_points = sum(1 for p in points if p.is_anomaly)
    print(f"  {anomaly_points} of {len(points)} points ({anomaly_points/len(points):.1%}) are anomalous.")


if __name__ == "__main__":
    main()
