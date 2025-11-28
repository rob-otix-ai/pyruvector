use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyType};
use serde_json::Value;
use std::collections::HashMap;

/// Search result returned from vector database queries
#[pyclass]
#[derive(Clone, Debug)]
pub struct SearchResult {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub score: f32,
    pub metadata: HashMap<String, Value>,
    /// Optional vector data (populated when available)
    pub vector: Option<Vec<f32>>,
}

#[pymethods]
impl SearchResult {
    // Note: Constructor is used internally in Rust code, not exposed to Python
    // SearchResult instances are created by search operations

    #[getter]
    fn metadata(&self, py: Python) -> PyResult<PyObject> {
        hashmap_to_pydict(py, &self.metadata)
    }

    #[getter]
    fn vector(&self) -> Option<Vec<f32>> {
        self.vector.clone()
    }

    fn __repr__(&self) -> String {
        let vector_info = if let Some(ref v) = self.vector {
            format!(", vector_dims={}", v.len())
        } else {
            String::new()
        };
        format!(
            "SearchResult(id='{}', score={:.4}{}, metadata={})",
            self.id,
            self.score,
            vector_info,
            serde_json::to_string(&self.metadata).unwrap_or_else(|_| "{}".to_string())
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Database statistics
#[pyclass]
#[derive(Clone, Debug)]
pub struct DBStats {
    #[pyo3(get)]
    pub vector_count: usize,
    #[pyo3(get)]
    pub dimensions: usize,
    #[pyo3(get)]
    pub estimated_memory_bytes: usize,
}

#[pymethods]
impl DBStats {
    #[new]
    pub fn new(vector_count: usize, dimensions: usize, estimated_memory_bytes: usize) -> Self {
        Self {
            vector_count,
            dimensions,
            estimated_memory_bytes,
        }
    }

    /// Alias for vector_count for backward compatibility
    #[getter]
    fn count(&self) -> usize {
        self.vector_count
    }

    /// Alias for estimated_memory_bytes for backward compatibility
    #[getter]
    fn memory_usage_bytes(&self) -> usize {
        self.estimated_memory_bytes
    }

    fn __repr__(&self) -> String {
        format!(
            "DBStats(vector_count={}, dimensions={}, estimated_memory_bytes={})",
            self.vector_count, self.dimensions, self.estimated_memory_bytes
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Get memory usage in human-readable format (MB)
    #[getter]
    fn memory_usage_mb(&self) -> f64 {
        self.estimated_memory_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Distance metric for vector similarity computation
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

#[pymethods]
impl DistanceMetric {
    #[classmethod]
    fn cosine(_cls: &PyType) -> Self {
        DistanceMetric::Cosine
    }

    #[classmethod]
    fn euclidean(_cls: &PyType) -> Self {
        DistanceMetric::Euclidean
    }

    #[classmethod]
    fn dot_product(_cls: &PyType) -> Self {
        DistanceMetric::DotProduct
    }

    #[classmethod]
    fn manhattan(_cls: &PyType) -> Self {
        DistanceMetric::Manhattan
    }

    fn __repr__(&self) -> String {
        match self {
            DistanceMetric::Cosine => "DistanceMetric.Cosine".to_string(),
            DistanceMetric::Euclidean => "DistanceMetric.Euclidean".to_string(),
            DistanceMetric::DotProduct => "DistanceMetric.DotProduct".to_string(),
            DistanceMetric::Manhattan => "DistanceMetric.Manhattan".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self {
            DistanceMetric::Cosine => "Cosine".to_string(),
            DistanceMetric::Euclidean => "Euclidean".to_string(),
            DistanceMetric::DotProduct => "DotProduct".to_string(),
            DistanceMetric::Manhattan => "Manhattan".to_string(),
        }
    }
}

impl Default for DistanceMetric {
    fn default() -> Self {
        DistanceMetric::Cosine
    }
}

/// Quantization type for memory optimization
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantizationType {
    None,
    Scalar,
    Product,
    Binary,
}

#[pymethods]
impl QuantizationType {
    #[classmethod]
    fn none(_cls: &PyType) -> Self {
        QuantizationType::None
    }

    #[classmethod]
    fn scalar(_cls: &PyType) -> Self {
        QuantizationType::Scalar
    }

    #[classmethod]
    fn product(_cls: &PyType) -> Self {
        QuantizationType::Product
    }

    #[classmethod]
    fn binary(_cls: &PyType) -> Self {
        QuantizationType::Binary
    }

    fn __repr__(&self) -> String {
        match self {
            QuantizationType::None => "QuantizationType.None".to_string(),
            QuantizationType::Scalar => "QuantizationType.Scalar".to_string(),
            QuantizationType::Product => "QuantizationType.Product".to_string(),
            QuantizationType::Binary => "QuantizationType.Binary".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self {
            QuantizationType::None => "None".to_string(),
            QuantizationType::Scalar => "Scalar".to_string(),
            QuantizationType::Product => "Product".to_string(),
            QuantizationType::Binary => "Binary".to_string(),
        }
    }
}

impl Default for QuantizationType {
    fn default() -> Self {
        QuantizationType::None
    }
}

/// HNSW (Hierarchical Navigable Small World) index configuration
#[pyclass]
#[derive(Clone, Debug)]
pub struct HNSWConfig {
    #[pyo3(get, set)]
    pub m: usize,
    #[pyo3(get, set)]
    pub ef_construction: usize,
    #[pyo3(get, set)]
    pub ef_search: usize,
    #[pyo3(get, set)]
    pub max_elements: Option<usize>,
}

#[pymethods]
impl HNSWConfig {
    #[new]
    #[pyo3(signature = (m=16, ef_construction=200, ef_search=50, max_elements=None))]
    pub fn new(
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        max_elements: Option<usize>,
    ) -> Self {
        Self {
            m,
            ef_construction,
            ef_search,
            max_elements,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HNSWConfig(m={}, ef_construction={}, ef_search={}, max_elements={:?})",
            self.m, self.ef_construction, self.ef_search, self.max_elements
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl Default for HNSWConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            max_elements: None,
        }
    }
}

/// Quantization configuration for memory optimization
#[pyclass]
#[derive(Clone, Debug)]
pub struct QuantizationConfig {
    #[pyo3(get, set)]
    pub quantization_type: QuantizationType,
    #[pyo3(get, set)]
    pub subspaces: Option<usize>,
    #[pyo3(get, set)]
    pub bits: Option<usize>,
}

#[pymethods]
impl QuantizationConfig {
    #[new]
    #[pyo3(signature = (quantization_type=None, subspaces=None, bits=None))]
    pub fn new(
        quantization_type: Option<QuantizationType>,
        subspaces: Option<usize>,
        bits: Option<usize>,
    ) -> Self {
        Self {
            quantization_type: quantization_type.unwrap_or_default(),
            subspaces,
            bits,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantizationConfig(quantization_type={}, subspaces={:?}, bits={:?})",
            self.quantization_type.__str__(),
            self.subspaces,
            self.bits
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quantization_type: QuantizationType::None,
            subspaces: None,
            bits: None,
        }
    }
}

/// Database configuration options
#[pyclass]
#[derive(Clone, Debug)]
pub struct DbOptions {
    #[pyo3(get, set)]
    pub dimensions: usize,
    #[pyo3(get, set)]
    pub distance_metric: DistanceMetric,
    #[pyo3(get, set)]
    pub storage_path: Option<String>,
    #[pyo3(get, set)]
    pub hnsw_config: HNSWConfig,
    #[pyo3(get, set)]
    pub quantization: QuantizationConfig,
}

#[pymethods]
impl DbOptions {
    #[new]
    #[pyo3(signature = (dimensions, distance_metric=None, storage_path=None, hnsw_config=None, quantization=None))]
    pub fn new(
        dimensions: usize,
        distance_metric: Option<DistanceMetric>,
        storage_path: Option<String>,
        hnsw_config: Option<HNSWConfig>,
        quantization: Option<QuantizationConfig>,
    ) -> Self {
        Self {
            dimensions,
            distance_metric: distance_metric.unwrap_or_default(),
            storage_path,
            hnsw_config: hnsw_config.unwrap_or_default(),
            quantization: quantization.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DbOptions(dimensions={}, distance_metric={}, storage_path={:?}, hnsw_config={}, quantization={})",
            self.dimensions,
            self.distance_metric.__str__(),
            self.storage_path,
            self.hnsw_config.__repr__(),
            self.quantization.__repr__()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Collection statistics
#[pyclass]
#[derive(Clone, Debug)]
pub struct CollectionStats {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub vector_count: usize,
    #[pyo3(get)]
    pub dimensions: usize,
    #[pyo3(get)]
    pub distance_metric: DistanceMetric,
    #[pyo3(get)]
    pub memory_usage_bytes: usize,
}

#[pymethods]
impl CollectionStats {
    #[new]
    pub fn new(
        name: String,
        vector_count: usize,
        dimensions: usize,
        distance_metric: DistanceMetric,
        memory_usage_bytes: usize,
    ) -> Self {
        Self {
            name,
            vector_count,
            dimensions,
            distance_metric,
            memory_usage_bytes,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CollectionStats(name='{}', vector_count={}, dimensions={}, distance_metric={}, memory_usage_bytes={})",
            self.name,
            self.vector_count,
            self.dimensions,
            self.distance_metric.__str__(),
            self.memory_usage_bytes
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Get memory usage in human-readable format (MB)
    #[getter]
    fn memory_usage_mb(&self) -> f64 {
        self.memory_usage_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Health status of the database
#[pyclass]
#[derive(Clone, Debug)]
pub struct HealthStatus {
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub vector_count: usize,
    #[pyo3(get)]
    pub memory_usage_bytes: usize,
    #[pyo3(get)]
    pub uptime_seconds: f64,
}

#[pymethods]
impl HealthStatus {
    #[new]
    pub fn new(
        status: String,
        vector_count: usize,
        memory_usage_bytes: usize,
        uptime_seconds: f64,
    ) -> Self {
        Self {
            status,
            vector_count,
            memory_usage_bytes,
            uptime_seconds,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HealthStatus(status='{}', vector_count={}, memory_usage_bytes={}, uptime_seconds={:.2})",
            self.status,
            self.vector_count,
            self.memory_usage_bytes,
            self.uptime_seconds
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Get memory usage in human-readable format (MB)
    #[getter]
    fn memory_usage_mb(&self) -> f64 {
        self.memory_usage_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get uptime in human-readable format (hours)
    #[getter]
    fn uptime_hours(&self) -> f64 {
        self.uptime_seconds / 3600.0
    }
}

/// Convert Rust HashMap<String, Value> to Python dict
pub fn hashmap_to_pydict(py: Python, map: &HashMap<String, Value>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    for (key, value) in map.iter() {
        let py_value = json_value_to_pyobject(py, value)?;
        dict.set_item(key, py_value)?;
    }

    Ok(dict.into())
}

/// Convert serde_json::Value to PyObject
pub fn json_value_to_pyobject(py: Python, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.into_py(py)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py(py))
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_py(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py(py))
            } else {
                Ok(py.None())
            }
        }
        Value::String(s) => Ok(PyString::new(py, s).into()),
        Value::Array(arr) => {
            let list: PyResult<Vec<PyObject>> = arr
                .iter()
                .map(|v| json_value_to_pyobject(py, v))
                .collect();
            Ok(list?.into_py(py))
        }
        Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (key, val) in obj.iter() {
                let py_val = json_value_to_pyobject(py, val)?;
                dict.set_item(key, py_val)?;
            }
            Ok(dict.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_result_creation() {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::String("test".to_string()));

        let result = SearchResult {
            id: "id1".to_string(),
            score: 0.95,
            metadata,
            vector: None,
        };
        assert_eq!(result.id, "id1");
        assert!((result.score - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_db_stats_creation() {
        let stats = DBStats::new(1000, 384, 1024 * 1024);
        assert_eq!(stats.vector_count, 1000);
        assert_eq!(stats.dimensions, 384);
        assert_eq!(stats.estimated_memory_bytes, 1024 * 1024);
    }

    #[test]
    fn test_db_stats_memory_mb() {
        let stats = DBStats::new(1000, 384, 2 * 1024 * 1024);
        assert!((stats.memory_usage_mb() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_search_result_repr() {
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), Value::String("value".to_string()));

        let result = SearchResult {
            id: "test_id".to_string(),
            score: 0.1235,
            metadata,
            vector: None,
        };
        let repr = result.__repr__();
        assert!(repr.contains("test_id"));
        assert!(repr.contains("0.1235"));
    }

    #[test]
    fn test_db_stats_repr() {
        let stats = DBStats::new(500, 256, 512000);
        let repr = stats.__repr__();
        assert!(repr.contains("500"));
        assert!(repr.contains("256"));
        assert!(repr.contains("512000"));
    }
}
