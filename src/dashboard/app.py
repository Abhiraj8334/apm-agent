"""
app.py

Streamlit dashboard for APM-Agent. Ties together all three phases:
  - visualizes the simulated metrics with ground-truth anomaly windows
  - runs both detectors (ThresholdBaseline, IsolationForest+STL) and shows
    their evaluation numbers (point- and window-level) side by side
  - lets you pick one flagged anomaly and run the RCA agent against it live,
    showing the reasoning trace step by step

Run with:
    streamlit run src/dashboard/app.py

This file is intentionally thin: data loading lives in data.py, chart
building in charts.py, and both are plain functions tested without
Streamlit. This file just wires them into the UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts this script's own folder on sys.path, not the project
# root, so "from src.dashboard..." fails unless we add the root ourselves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import streamlit as st
from dotenv import load_dotenv

from src.dashboard.data import (
    METRIC_COLUMNS,
    data_available,
    load_metrics_df,
    load_labels,
    label_summary_rows,
)
from src.dashboard.charts import build_metric_timeseries_figure, build_evaluation_bar_figure
from src.detector.anomaly_detector import IsolationForestDetector, ThresholdBaseline
from src.detector.evaluate import score, window_level_score
from src.agent.agent import AnomalyContext, run_rca_agent
from src.agent.llm_client import GeminiClient

load_dotenv()

st.set_page_config(page_title="APM-Agent Dashboard", layout="wide")
st.title("APM-Agent: AI-Driven Performance Monitoring")
st.caption(
    "Synthetic metrics → ML anomaly detection (evaluated against ground truth) "
    "→ LLM root-cause agent. See the README for the honest findings behind these numbers."
)

if not data_available():
    st.warning(
        "No data found. Run `python -m src.simulator.metrics_simulator` in a terminal first, "
        "then refresh this page."
    )
    st.stop()


@st.cache_data
def _load():
    df = load_metrics_df()
    labels = load_labels()
    return df, labels


@st.cache_data
def _run_detectors(df: "pd.DataFrame"):  # noqa: F821 - pd imported transitively via df's type
    import pandas as pd  # local import keeps the top-level import list minimal for a thin UI file

    y_true = df["is_anomaly"].astype(bool).to_numpy()
    period = max(2, len(df) // 4)

    baseline = ThresholdBaseline(z_cutoff=3.0).fit(df)
    baseline_pred = baseline.predict(df)

    iforest = IsolationForestDetector(period=period, contamination=y_true.mean()).fit(df)
    iforest_pred = iforest.predict(df)

    return baseline_pred, iforest_pred


df, labels = _load()
label_rows = label_summary_rows(labels)

st.header("1. Metrics and detected anomalies")

col1, col2 = st.columns([1, 3])
with col1:
    metric = st.selectbox("Metric to plot", METRIC_COLUMNS, index=1)
    show_detectors = st.checkbox("Overlay detector flags", value=True)

try:
    baseline_pred, iforest_pred = _run_detectors(df)
    detector_flags = (
        {"ThresholdBaseline": baseline_pred, "IsolationForest+STL": iforest_pred}
        if show_detectors
        else None
    )
except Exception as e:  # noqa: BLE001 - surface any detector error plainly in the UI rather than crashing the page
    st.error(f"Detector failed to run: {e}")
    detector_flags = None
    baseline_pred = iforest_pred = None

with col2:
    fig = build_metric_timeseries_figure(df, metric, labels, detector_flags)
    st.pyplot(fig)

st.header("2. Detector evaluation (point-level and window-level)")

if baseline_pred is not None:
    y_true = df["is_anomaly"].astype(bool).to_numpy()

    point_results = [
        score(y_true, baseline_pred, "ThresholdBaseline"),
        score(y_true, iforest_pred, "IsolationForest+STL"),
    ]
    window_results = [
        window_level_score(baseline_pred, labels, len(df), "ThresholdBaseline"),
        window_level_score(iforest_pred, labels, len(df), "IsolationForest+STL"),
    ]

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        st.subheader("Point-level")
        st.table(
            [{"detector": r.name, "precision": f"{r.precision:.2f}", "recall": f"{r.recall:.2f}", "f1": f"{r.f1:.2f}"} for r in point_results]
        )
        st.pyplot(build_evaluation_bar_figure([{"name": r.name, "precision": r.precision, "recall": r.recall, "f1": r.f1} for r in point_results]))
    with eval_col2:
        st.subheader("Window-level")
        st.table(
            [{"detector": r.name, "precision": f"{r.precision:.2f}", "recall": f"{r.recall:.2f}", "f1": f"{r.f1:.2f}"} for r in window_results]
        )
        st.pyplot(build_evaluation_bar_figure([{"name": r.name, "precision": r.precision, "recall": r.recall, "f1": r.f1} for r in window_results]))

    st.info(
        "ThresholdBaseline outperforms IsolationForest+STL on this synthetic dataset. "
        "See the README's 'Honest finding' section for why — this isn't hidden or explained away."
    )

st.header("3. Investigate an anomaly with the RCA agent")

selected_label = st.selectbox(
    "Pick an anomaly window to investigate",
    options=label_rows,
    format_func=lambda r: f"#{r['id']} — {r['type']} (idx {r['window']})",
)

if st.button("Investigate with AI agent"):
    window_df = df[(df["idx"] >= selected_label["start_idx"]) & (df["idx"] < selected_label["end_idx"])]
    baseline_df = df[~df["idx"].between(selected_label["start_idx"], selected_label["end_idx"])]

    metric_for_type = {
        "cpu_spike": "cpu_pct",
        "latency_spike": "latency_ms",
        "memory_leak": "memory_pct",
        "error_burst": "error_rate",
    }.get(selected_label["type"], "cpu_pct")

    peak_row = window_df.loc[window_df[metric_for_type].idxmax()]

    ctx = AnomalyContext(
        anomaly_type=selected_label["type"],
        metric=metric_for_type,
        start_idx=selected_label["start_idx"],
        end_idx=selected_label["end_idx"],
        peak_values={col: float(peak_row[col]) for col in METRIC_COLUMNS},
        baseline_values={col: float(baseline_df[col].mean()) for col in METRIC_COLUMNS},
    )

    trace_placeholder = st.empty()
    steps_shown: list[str] = []

    try:
        llm = GeminiClient()
    except (ImportError, ValueError) as e:
        st.error(f"Could not start the LLM client: {e}")
    else:
        with st.spinner("Agent investigating..."):
            result = run_rca_agent(ctx, llm)

        st.subheader("Reasoning trace")
        for step in result.trace:
            with st.expander(f"Step {step.step}: {step.action}"):
                st.write(f"**Thought:** {step.thought or '(empty)'}")
                st.write(f"**Observation:** {step.observation}")

        st.subheader("Final diagnosis")
        st.metric("Confidence", f"{result.confidence:.0%}")
        st.write(f"Stopped because: `{result.stopped_reason}`")
        st.write(result.diagnosis)

