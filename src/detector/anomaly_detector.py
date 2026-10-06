"""
anomaly_detector.py

Two detectors, so results can be compared rather than assumed:

1. ThresholdBaseline — flags a point anomalous if any metric's raw
   z-score exceeds a cutoff. Simple, naive, and the thing most
   beginner projects ship as their whole "anomaly detection" story.

2. IsolationForestDetector — trains on STL residuals (see
   preprocessing.py) across all metrics jointly, so it can catch
   *correlated* multivariate anomalies (e.g. CPU and latency both
   moving together) that a per-metric threshold would treat as two
   separate, weaker signals.

Both expose the same .fit(df) / .predict(df) -> np.ndarray[bool]
interface so the evaluation harness can score them identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.detector.preprocessing import build_residual_features

METRIC_COLUMNS = ["latency_ms", "cpu_pct", "memory_pct", "error_rate", "throughput_rps"]


class ThresholdBaseline:
    """Flags a point if any metric's z-score (vs the whole series' mean/std) exceeds `z_cutoff`."""

    def __init__(self, z_cutoff: float = 3.0):
        self.z_cutoff = z_cutoff
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "ThresholdBaseline":
        for col in METRIC_COLUMNS:
            self._means[col] = df[col].mean()
            self._stds[col] = df[col].std() or 1.0
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        flags = np.zeros(len(df), dtype=bool)
        for col in METRIC_COLUMNS:
            z = (df[col] - self._means[col]) / self._stds[col]
            flags |= (z.abs() > self.z_cutoff).to_numpy()
        return flags


class IsolationForestDetector:
    """
    Trains Isolation Forest on STL residuals across all metrics jointly.
    `contamination` is the assumed fraction of anomalous points — set it
    close to your best estimate of the true anomaly rate; too low and
    real anomalies get missed, too high and normal noise gets flagged.
    """

    def __init__(self, period: int, contamination: float = 0.07, random_state: int = 42):
        self.period = period
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self._residual_cache: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> "IsolationForestDetector":
        residuals = build_residual_features(df, period=self.period, columns=METRIC_COLUMNS)
        self._residual_cache = residuals
        self.model.fit(residuals)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # Reuse residuals computed during fit when predicting on the same df
        # (STL decomposition is not cheap to recompute unnecessarily).
        if self._residual_cache is not None and len(self._residual_cache) == len(df):
            residuals = self._residual_cache
        else:
            residuals = build_residual_features(df, period=self.period, columns=METRIC_COLUMNS)
        predictions = self.model.predict(residuals)  # -1 = anomaly, 1 = normal
        return predictions == -1

    def anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Lower (more negative) score = more anomalous. Useful for ranking/alerting."""
        if self._residual_cache is not None and len(self._residual_cache) == len(df):
            residuals = self._residual_cache
        else:
            residuals = build_residual_features(df, period=self.period, columns=METRIC_COLUMNS)
        return self.model.score_samples(residuals)
