import numpy as np

from src.simulator.metrics_simulator import MetricsSimulator
from src.dashboard.data import label_summary_rows
from src.dashboard.charts import build_metric_timeseries_figure, build_evaluation_bar_figure


def _labeled_dataframe_and_labels():
    import pandas as pd

    sim = MetricsSimulator(n_points=500, seed=3)
    points, labels = sim.generate()
    df = pd.DataFrame([p.__dict__ for p in points])
    label_dicts = [l.__dict__ for l in labels]
    return df, label_dicts


def test_label_summary_rows_formats_window_readably():
    rows = label_summary_rows(
        [{"anomaly_id": 1, "anomaly_type": "cpu_spike", "start_idx": 10, "end_idx": 20, "description": "x"}]
    )
    assert rows[0]["window"] == "10–20"
    assert rows[0]["type"] == "cpu_spike"
    assert rows[0]["id"] == 1


def test_timeseries_figure_shades_one_region_per_label():
    df, labels = _labeled_dataframe_and_labels()
    fig = build_metric_timeseries_figure(df, "cpu_pct", labels)
    ax = fig.axes[0]
    assert len(ax.patches) == len(labels)


def test_timeseries_figure_adds_one_scatter_series_per_detector():
    df, labels = _labeled_dataframe_and_labels()
    flags = {
        "DetectorA": (df["cpu_pct"] > 70).to_numpy(),
        "DetectorB": (df["cpu_pct"] > 60).to_numpy(),
    }
    fig = build_metric_timeseries_figure(df, "cpu_pct", labels, flags)
    ax = fig.axes[0]
    assert len(ax.collections) == 2


def test_timeseries_figure_with_no_detector_flags_has_no_scatter():
    df, labels = _labeled_dataframe_and_labels()
    fig = build_metric_timeseries_figure(df, "cpu_pct", labels, detector_flags=None)
    ax = fig.axes[0]
    assert len(ax.collections) == 0


def test_evaluation_bar_figure_has_one_bar_group_per_detector():
    rows = [
        {"name": "A", "precision": 1.0, "recall": 0.8, "f1": 0.9},
        {"name": "B", "precision": 0.6, "recall": 0.6, "f1": 0.6},
    ]
    fig = build_evaluation_bar_figure(rows)
    ax = fig.axes[0]
    assert len(ax.patches) == 6
