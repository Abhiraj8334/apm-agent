"""
preprocessing.py

Removes trend and seasonality from each metric series before anomaly
detection, using STL (Seasonal-Trend decomposition using LOESS).

Why this matters: a naive threshold or Isolation Forest run on raw
metric values will treat normal daily/cyclical swings as suspicious.
By decomposing each series into trend + seasonal + residual and scoring
anomalies on the *residual*, normal seasonal variation is no longer
mistaken for an anomaly — only genuine deviations from the expected
pattern remain.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.seasonal import STL


def stl_residuals(series: pd.Series, period: int) -> pd.Series:
    """
    Decompose `series` into trend + seasonal + residual via STL and
    return just the residual component.

    period: the expected seasonal cycle length, in number of points.
    Must be >= 2 and, per statsmodels' STL requirement, less than half
    the series length.
    """
    period = max(2, min(period, len(series) // 2 - 1))
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    return result.resid


def build_residual_features(df: pd.DataFrame, period: int, columns: list[str]) -> pd.DataFrame:
    """
    Apply STL residual extraction to each named column of df and return
    a new DataFrame of residuals only (same index as df), one column
    per input metric.
    """
    residuals = {}
    for col in columns:
        residuals[col] = stl_residuals(df[col], period=period)
    return pd.DataFrame(residuals, index=df.index)
