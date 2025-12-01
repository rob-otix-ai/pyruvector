use lazy_static::lazy_static;
use prometheus::{CounterVec, Encoder, Gauge, GaugeVec, HistogramVec, Opts, Registry, TextEncoder};
use pyo3::prelude::*;

lazy_static! {
    /// Global Prometheus registry for all metrics
    static ref REGISTRY: Registry = {
        let registry = Registry::new();

        // Register all metrics
        registry.register(Box::new(SEARCH_REQUESTS.clone())).unwrap();
        registry.register(Box::new(SEARCH_LATENCY.clone())).unwrap();
        registry.register(Box::new(SEARCH_ERRORS.clone())).unwrap();
        registry.register(Box::new(INSERT_REQUESTS.clone())).unwrap();
        registry.register(Box::new(INSERT_LATENCY.clone())).unwrap();
        registry.register(Box::new(DELETE_REQUESTS.clone())).unwrap();
        registry.register(Box::new(VECTORS_TOTAL.clone())).unwrap();
        registry.register(Box::new(MEMORY_USAGE.clone())).unwrap();

        registry
    };

    /// Counter for search requests per collection
    static ref SEARCH_REQUESTS: CounterVec = CounterVec::new(
        Opts::new("pyruvector_search_requests_total", "Total number of search requests")
            .namespace("pyruvector"),
        &["collection", "status"]
    ).unwrap();

    /// Histogram for search latency per collection
    static ref SEARCH_LATENCY: HistogramVec = HistogramVec::new(
        prometheus::HistogramOpts::new(
            "pyruvector_search_duration_seconds",
            "Search request duration in seconds"
        )
        .namespace("pyruvector")
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
        &["collection"]
    ).unwrap();

    /// Counter for search errors per collection
    static ref SEARCH_ERRORS: CounterVec = CounterVec::new(
        Opts::new("pyruvector_search_errors_total", "Total number of search errors")
            .namespace("pyruvector"),
        &["collection"]
    ).unwrap();

    /// Counter for insert requests per collection
    static ref INSERT_REQUESTS: CounterVec = CounterVec::new(
        Opts::new("pyruvector_insert_requests_total", "Total number of insert requests")
            .namespace("pyruvector"),
        &["collection"]
    ).unwrap();

    /// Histogram for insert latency per collection
    static ref INSERT_LATENCY: HistogramVec = HistogramVec::new(
        prometheus::HistogramOpts::new(
            "pyruvector_insert_duration_seconds",
            "Insert request duration in seconds"
        )
        .namespace("pyruvector")
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]),
        &["collection"]
    ).unwrap();

    /// Counter for delete requests per collection
    static ref DELETE_REQUESTS: CounterVec = CounterVec::new(
        Opts::new("pyruvector_delete_requests_total", "Total number of delete requests")
            .namespace("pyruvector"),
        &["collection"]
    ).unwrap();

    /// Gauge for total number of vectors per collection
    static ref VECTORS_TOTAL: GaugeVec = GaugeVec::new(
        Opts::new("pyruvector_vectors_total", "Total number of vectors stored")
            .namespace("pyruvector"),
        &["collection"]
    ).unwrap();

    /// Gauge for memory usage in bytes
    static ref MEMORY_USAGE: Gauge = Gauge::new(
        "pyruvector_memory_usage_bytes",
        "Memory usage in bytes"
    ).unwrap();
}

/// Python class for recording metrics
#[pyclass]
#[derive(Clone)]
pub struct MetricsRecorder;

impl Default for MetricsRecorder {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl MetricsRecorder {
    /// Create a new MetricsRecorder instance
    #[new]
    pub fn new() -> Self {
        MetricsRecorder
    }

    /// Record a search operation
    ///
    /// # Arguments
    /// * `collection` - Name of the collection being searched
    /// * `duration_seconds` - Duration of the search in seconds
    /// * `success` - Whether the search was successful
    #[pyo3(text_signature = "($self, collection, duration_seconds, success)")]
    pub fn record_search(
        &self,
        collection: String,
        duration_seconds: f64,
        success: bool,
    ) -> PyResult<()> {
        let status = if success { "success" } else { "error" };

        // Increment search counter
        SEARCH_REQUESTS
            .with_label_values(&[&collection, status])
            .inc();

        // Record latency
        SEARCH_LATENCY
            .with_label_values(&[&collection])
            .observe(duration_seconds);

        // Increment error counter if failed
        if !success {
            SEARCH_ERRORS.with_label_values(&[&collection]).inc();
        }

        Ok(())
    }

    /// Record an insert operation
    ///
    /// # Arguments
    /// * `collection` - Name of the collection being inserted into
    /// * `count` - Number of vectors inserted
    /// * `duration_seconds` - Duration of the insert in seconds
    #[pyo3(text_signature = "($self, collection, count, duration_seconds)")]
    pub fn record_insert(
        &self,
        collection: String,
        count: usize,
        duration_seconds: f64,
    ) -> PyResult<()> {
        // Increment insert counter by count
        INSERT_REQUESTS
            .with_label_values(&[&collection])
            .inc_by(count as f64);

        // Record latency
        INSERT_LATENCY
            .with_label_values(&[&collection])
            .observe(duration_seconds);

        Ok(())
    }

    /// Record a delete operation
    ///
    /// # Arguments
    /// * `collection` - Name of the collection being deleted from
    /// * `count` - Number of vectors deleted
    #[pyo3(text_signature = "($self, collection, count)")]
    pub fn record_delete(&self, collection: String, count: usize) -> PyResult<()> {
        // Increment delete counter by count
        DELETE_REQUESTS
            .with_label_values(&[&collection])
            .inc_by(count as f64);

        Ok(())
    }

    /// Update the total vector count for a collection
    ///
    /// # Arguments
    /// * `collection` - Name of the collection
    /// * `count` - Total number of vectors in the collection
    #[pyo3(text_signature = "($self, collection, count)")]
    pub fn update_vector_count(&self, collection: String, count: usize) -> PyResult<()> {
        VECTORS_TOTAL
            .with_label_values(&[&collection])
            .set(count as f64);

        Ok(())
    }

    /// Update the memory usage
    ///
    /// # Arguments
    /// * `bytes` - Memory usage in bytes
    #[pyo3(text_signature = "($self, bytes)")]
    pub fn update_memory_usage(&self, bytes: usize) -> PyResult<()> {
        MEMORY_USAGE.set(bytes as f64);

        Ok(())
    }

    /// Reset all metrics for a specific collection
    ///
    /// # Arguments
    /// * `collection` - Name of the collection to reset metrics for
    #[pyo3(text_signature = "($self, collection)")]
    pub fn reset_collection_metrics(&self, collection: String) -> PyResult<()> {
        // Reset vector count to 0
        VECTORS_TOTAL.with_label_values(&[&collection]).set(0.0);

        Ok(())
    }
}

/// Gather all metrics in Prometheus text format
///
/// Returns a string containing all metrics in Prometheus exposition format
#[pyfunction]
#[pyo3(text_signature = "()")]
pub fn gather_metrics() -> PyResult<String> {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();

    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to encode metrics: {}",
            e
        ))
    })?;

    String::from_utf8(buffer).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to convert metrics to string: {}",
            e
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recorder_creation() {
        let recorder = MetricsRecorder::new();
        assert!(recorder
            .record_search("test".to_string(), 0.5, true)
            .is_ok());
    }

    #[test]
    fn test_gather_metrics() {
        let recorder = MetricsRecorder::new();
        recorder
            .record_search("test".to_string(), 0.1, true)
            .unwrap();

        let metrics_text = gather_metrics().unwrap();
        assert!(metrics_text.contains("pyruvector_search_requests_total"));
        assert!(metrics_text.contains("pyruvector_search_duration_seconds"));
    }
}
