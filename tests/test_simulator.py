from src.simulator.metrics_simulator import MetricsSimulator


def test_generate_produces_requested_point_count():
    sim = MetricsSimulator(n_points=500, seed=1)
    points, _ = sim.generate()
    assert len(points) == 500


def test_labels_match_flagged_points():
    """Every point inside a labeled window must be flagged is_anomaly,
    and every point outside all windows must NOT be flagged. This is the
    ground truth the evaluation harness (Phase 3) will score against, so
    it has to be exactly right."""
    sim = MetricsSimulator(n_points=1000, seed=2)
    points, labels = sim.generate()

    labeled_idxs = set()
    for label in labels:
        for i in range(label.start_idx, label.end_idx):
            labeled_idxs.add(i)

    for p in points:
        expected = p.idx in labeled_idxs
        assert p.is_anomaly == expected, f"idx {p.idx}: expected {expected}, got {p.is_anomaly}"


def test_no_overlapping_anomaly_windows():
    sim = MetricsSimulator(n_points=2000, seed=42)
    _, labels = sim.generate()
    windows = sorted((l.start_idx, l.end_idx) for l in labels)
    for i in range(len(windows) - 1):
        assert windows[i][1] <= windows[i + 1][0], "anomaly windows must not overlap"


def test_anomaly_values_are_more_extreme_than_baseline():
    """Sanity check that injected anomalies actually perturb the metric
    they claim to affect — catches silent no-op injection bugs."""
    sim = MetricsSimulator(n_points=2000, seed=42)
    points, labels = sim.generate()

    cpu_spikes = [l for l in labels if l.anomaly_type == "cpu_spike"]
    assert cpu_spikes, "expected at least one cpu_spike in default config"

    baseline_cpu = [p.cpu_pct for p in points if not p.is_anomaly]
    avg_baseline_cpu = sum(baseline_cpu) / len(baseline_cpu)

    for label in cpu_spikes:
        spike_cpu = [
            p.cpu_pct for p in points if label.start_idx <= p.idx < label.end_idx
        ]
        avg_spike_cpu = sum(spike_cpu) / len(spike_cpu)
        assert avg_spike_cpu > avg_baseline_cpu + 20, (
            f"cpu_spike window {label.anomaly_id} not meaningfully elevated "
            f"({avg_spike_cpu:.1f} vs baseline {avg_baseline_cpu:.1f})"
        )
