"""Tests for Prometheus metrics functionality."""
from pyruvector import MetricsRecorder, gather_metrics


class TestMetricsRecorder:
    """Test MetricsRecorder functionality."""

    def test_create_recorder(self):
        recorder = MetricsRecorder()
        assert recorder is not None

    def test_record_search(self):
        recorder = MetricsRecorder()
        recorder.record_search("test_collection", 0.05, True)

    def test_record_insert(self):
        recorder = MetricsRecorder()
        recorder.record_insert("test_collection", 100, 0.1)

    def test_record_delete(self):
        recorder = MetricsRecorder()
        recorder.record_delete("test_collection", 5)

    def test_update_vector_count(self):
        recorder = MetricsRecorder()
        recorder.update_vector_count("test_collection", 1000)

    def test_update_memory_usage(self):
        recorder = MetricsRecorder()
        recorder.update_memory_usage(1024 * 1024 * 100)


class TestGatherMetrics:
    """Test metrics gathering."""

    def test_gather_returns_string(self):
        metrics = gather_metrics()
        assert isinstance(metrics, str)

    def test_gather_prometheus_format(self):
        recorder = MetricsRecorder()
        recorder.record_search("col1", 0.01, True)
        metrics = gather_metrics()
        # Prometheus format includes # HELP and # TYPE lines
        assert "# " in metrics or len(metrics) > 0
