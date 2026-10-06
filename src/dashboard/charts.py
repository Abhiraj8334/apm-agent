"""
charts.py

Matplotlib figure builders for the dashboard, kept free of Streamlit so
they can be tested directly: given a DataFrame and some flags/labels, do
we get a Figure with the right number of series and shaded regions? That
is checkable without ever importing streamlit.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_metric_timeseries_figure(
    df: pd.DataFrame,
    metric: str,
    labels: list[dict],
    detector_flags: dict[str, np.ndarray] | None = None,
) -> plt.Figure:
    """
    Plot one metric over time, with:
      - ground-truth anomaly windows shaded (from `labels`)
      - optionally, each detector's flagged points marked (from `detector_flags`,
        a dict of {detector_name: boolean array aligned to df.index})
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["idx"], df[metric], linewidth=1, color="#1f77b4", label=metric)

    for label in labels:
        ax.axvspan(label["start_idx"], label["end_idx"], color="red", alpha=0.15)

    if detector_flags:
        markers = ["o", "x", "^", "s"]
        colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
        for i, (name, flags) in enumerate(detector_flags.items()):
            flagged_idx = df["idx"][flags]
            flagged_vals = df[metric][flags]
            ax.scatter(
                flagged_idx,
                flagged_vals,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                s=25,
                label=f"{name} flagged",
                zorder=5,
            )

    ax.set_xlabel("time index")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} over time (shaded = true anomaly window)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def build_evaluation_bar_figure(eval_rows: list[dict]) -> plt.Figure:
    """
    eval_rows: list of {"name": str, "precision": float, "recall": float, "f1": float}
    Grouped bar chart comparing detectors on the three metrics.
    """
    names = [r["name"] for r in eval_rows]
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in eval_rows]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xticks(x + width)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Detector comparison")
    ax.legend()
    fig.tight_layout()
    return fig
