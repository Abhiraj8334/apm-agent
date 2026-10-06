import numpy as np

from src.simulator.metrics_simulator import MetricsSimulator
from src.detector.anomaly_detector import IsolationForestDetector, ThresholdBaseline
from src.detector.evaluate import score


def _labeled_dataframe():
    """Generate a small labeled dataset in-memory (no DB round-trip needed for tests)."""
    import pandas as pd

    sim = MetricsSimulator(n_points=1000, seed=7)
    points, _ = sim.generate()
    df = pd.DataFrame([p.__dict__ for p in points])
    return df


def test_threshold_baseline_detects_some_true_anomalies():
    df = _labeled_dataframe()
    y_true = df["is_anomaly"].to_numpy()

    baseline = ThresholdBaseline(z_cutoff=3.0).fit(df)
    y_pred = baseline.predict(df)

    result = score(y_true, y_pred, "ThresholdBaseline")
    assert result.recall > 0.3, f"baseline recall too low: {result.recall}"


def test_isolation_forest_detects_some_true_anomalies():
    df = _labeled_dataframe()
    y_true = df["is_anomaly"].to_numpy()

    period = max(2, len(df) // 4)
    detector = IsolationForestDetector(period=period, contamination=0.07).fit(df)
    y_pred = detector.predict(df)

    result = score(y_true, y_pred, "IsolationForest+STL")
    assert result.recall > 0.3, f"isolation forest recall too low: {result.recall}"
    # Not a strict correctness bound, just a regression guard: if this detector
    # starts flagging almost everything or almost nothing, something broke.
    flagged_fraction = y_pred.mean()
    assert 0.01 < flagged_fraction < 0.5, f"suspicious flagged fraction: {flagged_fraction}"


def test_score_function_matches_known_confusion_matrix():
    y_true = np.array([True, True, False, False, True])
    y_pred = np.array([True, False, False, True, True])
    result = score(y_true, y_pred, "test")
    # TP: idx0, idx4 = 2 ; FP: idx3 = 1 ; FN: idx1 = 1 ; TN: idx2 = 1
    assert result.true_positives == 2
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.true_negatives == 1
    assert abs(result.precision - (2 / 3)) < 1e-9
    assert abs(result.recall - (2 / 3)) < 1e-9


def test_window_level_score_counts_a_single_hit_as_full_window_detection():
    from src.detector.evaluate import window_level_score

    # 20 points, one true anomaly window [5,10), detector only catches idx 7
    y_pred = np.zeros(20, dtype=bool)
    y_pred[7] = True
    labels = [{"start_idx": 5, "end_idx": 10}]

    result = window_level_score(y_pred, labels, n_points=20, name="test")
    assert result.true_positives == 1  # window counted as detected
    assert result.false_negatives == 0
    assert result.recall == 1.0


def test_window_level_score_groups_false_positives_into_episodes():
    from src.detector.evaluate import window_level_score

    # No true anomalies at all; detector fires at two separate, non-adjacent points
    y_pred = np.zeros(20, dtype=bool)
    y_pred[2] = True
    y_pred[15] = True
    labels: list[dict] = []

    result = window_level_score(y_pred, labels, n_points=20, name="test")
    # Two isolated flags -> two separate false-alarm episodes, not one
    assert result.false_positives == 2
