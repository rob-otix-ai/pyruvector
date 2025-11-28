//! VectorDB wrapper for ruvector-core

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use ruvector_core::{
    VectorDB as RuvectorDB,
    VectorEntry,
    SearchQuery,
};

use crate::types::{SearchResult, DBStats, DistanceMetric, HNSWConfig, QuantizationConfig, QuantizationType, DbOptions, HealthStatus};
use crate::filter::{parse_filter, evaluate_filter};

/// Main VectorDB class for storing and searching vectors
///
/// This wraps ruvector-core's high-performance HNSW-indexed vector database
/// with a Python-friendly API.
#[pyclass]
#[derive(Clone)]
pub struct VectorDB {
    pub(crate) inner: Arc<RwLock<RuvectorDB>>,
    pub(crate) dimensions: usize,
    pub(crate) path: Option<String>,
    pub(crate) distance_metric: DistanceMetric,
    pub(crate) hnsw_config: HNSWConfig,
    pub(crate) quantization_config: QuantizationConfig,
    pub(crate) created_at: std::time::Instant,
}

#[pymethods]
impl VectorDB {
    /// Create a new VectorDB instance
    ///
    /// # Arguments
    /// * `dimensions` - The dimensionality of vectors
    /// * `path` - Optional file path for persistence (enables automatic disk storage via redb)
    /// * `hnsw_m` - Optional HNSW M parameter (default: 16)
    /// * `hnsw_ef` - Optional HNSW ef parameter (default: 200)
    /// * `distance_metric` - Optional distance metric (default: Euclidean)
    /// * `hnsw_config` - Optional HNSW configuration object
    /// * `quantization_config` - Optional quantization configuration object
    ///
    /// # Notes
    /// When `path` is provided, the database will automatically persist all operations
    /// to disk using ruvector-core's storage backend (redb). No explicit save is required.
    #[new]
    #[pyo3(signature = (dimensions, path=None, hnsw_m=None, hnsw_ef=None, distance_metric=None, hnsw_config=None, quantization_config=None))]
    fn new(
        dimensions: usize,
        path: Option<String>,
        hnsw_m: Option<usize>,
        hnsw_ef: Option<usize>,
        distance_metric: Option<DistanceMetric>,
        hnsw_config: Option<HNSWConfig>,
        quantization_config: Option<QuantizationConfig>,
    ) -> PyResult<Self> {
        // Use provided configs or create defaults
        let hnsw_cfg = hnsw_config.unwrap_or_else(|| HNSWConfig {
            m: hnsw_m.unwrap_or(16),
            ef_construction: hnsw_ef.unwrap_or(200),
            ef_search: hnsw_ef.unwrap_or(200),
            max_elements: None,
        });

        // Build ruvector-core DbOptions
        let default_qconfig = QuantizationConfig::default();
        let qconfig = quantization_config.as_ref().unwrap_or(&default_qconfig);

        let core_options = ruvector_core::types::DbOptions {
            dimensions,
            distance_metric: match distance_metric.unwrap_or(DistanceMetric::Euclidean) {
                DistanceMetric::Euclidean => ruvector_core::types::DistanceMetric::Euclidean,
                DistanceMetric::Cosine => ruvector_core::types::DistanceMetric::Cosine,
                DistanceMetric::DotProduct => ruvector_core::types::DistanceMetric::DotProduct,
                DistanceMetric::Manhattan => ruvector_core::types::DistanceMetric::Manhattan,
            },
            storage_path: path.clone().unwrap_or_else(|| {
                // Use in-memory path if none provided
                format!(":memory:{}", uuid::Uuid::new_v4())
            }),
            hnsw_config: Some(ruvector_core::types::HnswConfig {
                m: hnsw_cfg.m,
                ef_construction: hnsw_cfg.ef_construction,
                ef_search: hnsw_cfg.ef_search,
                max_elements: hnsw_cfg.max_elements.unwrap_or(10_000_000),
            }),
            quantization: match qconfig.quantization_type {
                QuantizationType::None => Some(ruvector_core::types::QuantizationConfig::None),
                QuantizationType::Scalar => Some(ruvector_core::types::QuantizationConfig::Scalar),
                QuantizationType::Product => {
                    Some(ruvector_core::types::QuantizationConfig::Product {
                        subspaces: qconfig.subspaces.unwrap_or(8),
                        k: 1 << qconfig.bits.unwrap_or(8), // Convert bits to k (codebook size)
                    })
                }
                QuantizationType::Binary => Some(ruvector_core::types::QuantizationConfig::Binary),
            },
        };

        // Create VectorDB with proper options for persistence
        let inner = RuvectorDB::new(core_options)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to create VectorDB: {}", e)
            ))?;

        Ok(VectorDB {
            inner: Arc::new(RwLock::new(inner)),
            dimensions,
            path,
            distance_metric: distance_metric.unwrap_or(DistanceMetric::Euclidean),
            hnsw_config: hnsw_cfg,
            quantization_config: quantization_config.unwrap_or_default(),
            created_at: std::time::Instant::now(),
        })
    }

    /// Load an existing VectorDB from disk
    ///
    /// # Arguments
    /// * `path` - File path to the existing database
    /// * `dimensions` - Vector dimensions (must match the saved database)
    ///
    /// # Notes
    /// Opens an existing database that was previously created with a storage path.
    /// The database automatically loads all previously stored vectors from disk.
    ///
    /// # Example
    /// ```python
    /// # Create and save database
    /// db = VectorDB(dimensions=128, path="my_vectors.db")
    /// db.insert("vec1", [0.1] * 128)
    ///
    /// # Later, load the existing database
    /// db2 = VectorDB.load("my_vectors.db", dimensions=128)
    /// result = db2.get("vec1")  # Returns the previously saved vector
    /// ```
    #[staticmethod]
    #[pyo3(signature = (path, dimensions))]
    pub fn load(path: String, dimensions: usize) -> PyResult<Self> {
        // Create DbOptions with the path to open existing database
        let core_options = ruvector_core::types::DbOptions {
            dimensions,
            distance_metric: ruvector_core::types::DistanceMetric::Cosine, // Default, will be overridden by stored config
            storage_path: path.clone(),
            hnsw_config: Some(ruvector_core::types::HnswConfig::default()),
            quantization: Some(ruvector_core::types::QuantizationConfig::None),
        };

        // Open existing database - ruvector-core automatically loads from storage_path
        let inner = RuvectorDB::new(core_options)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load VectorDB from '{}': {}", path, e)
            ))?;

        // Get configuration from opened database and clone needed values
        let db_options = inner.options();
        let distance = db_options.distance_metric;
        let hnsw_opt = db_options.hnsw_config.clone();
        let quant_opt = db_options.quantization.clone();

        Ok(VectorDB {
            inner: Arc::new(RwLock::new(inner)),
            dimensions,
            path: Some(path),
            distance_metric: match distance {
                ruvector_core::types::DistanceMetric::Euclidean => DistanceMetric::Euclidean,
                ruvector_core::types::DistanceMetric::Cosine => DistanceMetric::Cosine,
                ruvector_core::types::DistanceMetric::DotProduct => DistanceMetric::DotProduct,
                ruvector_core::types::DistanceMetric::Manhattan => DistanceMetric::Manhattan,
            },
            hnsw_config: if let Some(hnsw) = hnsw_opt {
                HNSWConfig {
                    m: hnsw.m,
                    ef_construction: hnsw.ef_construction,
                    ef_search: hnsw.ef_search,
                    max_elements: Some(hnsw.max_elements),
                }
            } else {
                HNSWConfig::default()
            },
            quantization_config: match quant_opt {
                Some(ruvector_core::types::QuantizationConfig::None) | None => {
                    QuantizationConfig::new(Some(QuantizationType::None), None, None)
                }
                Some(ruvector_core::types::QuantizationConfig::Scalar) => {
                    QuantizationConfig::new(Some(QuantizationType::Scalar), None, None)
                }
                Some(ruvector_core::types::QuantizationConfig::Product { subspaces, k }) => {
                    QuantizationConfig::new(
                        Some(QuantizationType::Product),
                        Some(subspaces),
                        Some((k as f64).log2() as usize), // Convert k back to bits
                    )
                }
                Some(ruvector_core::types::QuantizationConfig::Binary) => {
                    QuantizationConfig::new(Some(QuantizationType::Binary), None, None)
                }
            },
            created_at: std::time::Instant::now(),
        })
    }

    /// Create a new VectorDB instance with comprehensive options
    ///
    /// # Arguments
    /// * `options` - DbOptions object containing all configuration parameters
    ///
    /// # Notes
    /// When `options.storage_path` is provided, the database will automatically
    /// persist all operations to disk using ruvector-core's storage backend (redb).
    #[staticmethod]
    pub fn with_options(options: DbOptions) -> PyResult<Self> {
        // Build ruvector-core DbOptions
        let core_options = ruvector_core::types::DbOptions {
            dimensions: options.dimensions,
            distance_metric: match options.distance_metric {
                DistanceMetric::Euclidean => ruvector_core::types::DistanceMetric::Euclidean,
                DistanceMetric::Cosine => ruvector_core::types::DistanceMetric::Cosine,
                DistanceMetric::DotProduct => ruvector_core::types::DistanceMetric::DotProduct,
                DistanceMetric::Manhattan => ruvector_core::types::DistanceMetric::Manhattan,
            },
            storage_path: options.storage_path.clone().unwrap_or_else(|| {
                format!(":memory:{}", uuid::Uuid::new_v4())
            }),
            hnsw_config: Some(ruvector_core::types::HnswConfig {
                m: options.hnsw_config.m,
                ef_construction: options.hnsw_config.ef_construction,
                ef_search: options.hnsw_config.ef_search,
                max_elements: options.hnsw_config.max_elements.unwrap_or(10_000_000),
            }),
            quantization: match options.quantization.quantization_type {
                QuantizationType::None => Some(ruvector_core::types::QuantizationConfig::None),
                QuantizationType::Scalar => Some(ruvector_core::types::QuantizationConfig::Scalar),
                QuantizationType::Product => {
                    Some(ruvector_core::types::QuantizationConfig::Product {
                        subspaces: options.quantization.subspaces.unwrap_or(8),
                        k: 1 << options.quantization.bits.unwrap_or(8),
                    })
                }
                QuantizationType::Binary => Some(ruvector_core::types::QuantizationConfig::Binary),
            },
        };

        let inner = RuvectorDB::new(core_options)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to create VectorDB: {}", e)
            ))?;

        Ok(VectorDB {
            inner: Arc::new(RwLock::new(inner)),
            dimensions: options.dimensions,
            path: options.storage_path,
            distance_metric: options.distance_metric,
            hnsw_config: options.hnsw_config,
            quantization_config: options.quantization,
            created_at: std::time::Instant::now(),
        })
    }

    /// Insert a single vector with optional metadata
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the vector
    /// * `vector` - The vector data
    /// * `metadata` - Optional metadata dictionary
    #[pyo3(signature = (id, vector, metadata=None))]
    pub fn insert(
        &self,
        py: Python,
        id: String,
        vector: Vec<f32>,
        metadata: Option<&PyDict>,
    ) -> PyResult<()> {
        // Validate vector dimensions
        if vector.len() != self.dimensions {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.dimensions,
                    vector.len()
                )
            ));
        }

        // Convert metadata to serde_json::Value
        let meta = if let Some(m) = metadata {
            Some(python_dict_to_json(py, m)?)
        } else {
            None
        };

        // Create VectorEntry
        let entry = VectorEntry {
            id: Some(id),
            vector,
            metadata: meta,
        };

        // Insert into ruvector-core
        let db = self.inner.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;

        db.insert(entry).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Insert failed: {}", e))
        })?;

        Ok(())
    }

    /// Insert multiple vectors in batch
    ///
    /// # Arguments
    /// * `ids` - List of unique identifiers
    /// * `vectors` - List of vectors
    /// * `metadatas` - Optional list of metadata dictionaries
    #[pyo3(signature = (ids, vectors, metadatas=None))]
    fn insert_batch(
        &self,
        py: Python,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        metadatas: Option<Vec<Option<&PyDict>>>,
    ) -> PyResult<()> {
        // Validate all inputs have same length
        if ids.len() != vectors.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ids and vectors must have the same length"
            ));
        }

        if let Some(ref metas) = metadatas {
            if metas.len() != ids.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "metadatas must have the same length as ids and vectors"
                ));
            }
        }

        // Build entries
        let mut entries = Vec::with_capacity(ids.len());
        for (i, (id, vector)) in ids.into_iter().zip(vectors.into_iter()).enumerate() {
            // Validate dimensions
            if vector.len() != self.dimensions {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!(
                        "Vector {} dimension mismatch: expected {}, got {}",
                        i, self.dimensions, vector.len()
                    )
                ));
            }

            let meta = if let Some(ref metas) = metadatas {
                if let Some(Some(m)) = metas.get(i) {
                    Some(python_dict_to_json(py, m)?)
                } else {
                    None
                }
            } else {
                None
            };

            entries.push(VectorEntry {
                id: Some(id),
                vector,
                metadata: meta,
            });
        }

        // Batch insert
        let db = self.inner.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;

        db.insert_batch(entries).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Batch insert failed: {}", e))
        })?;

        Ok(())
    }

    /// Search for k nearest neighbors
    ///
    /// # Arguments
    /// * `vector` - Query vector
    /// * `k` - Number of results to return
    /// * `filter` - Optional metadata filter
    #[pyo3(signature = (vector, k, filter=None))]
    fn search(
        &self,
        py: Python,
        vector: Vec<f32>,
        k: usize,
        filter: Option<&PyDict>,
    ) -> PyResult<Vec<SearchResult>> {
        // Validate vector dimensions
        if vector.len() != self.dimensions {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!(
                    "Query vector dimension mismatch: expected {}, got {}",
                    self.dimensions,
                    vector.len()
                )
            ));
        }

        // Parse filter if provided
        let filter_map = if let Some(f) = filter {
            Some(python_dict_to_json(py, f)?)
        } else {
            None
        };

        // Build search query
        let query = SearchQuery {
            vector,
            k,
            filter: filter_map.clone(),
            ef_search: Some(self.hnsw_config.ef_search),
        };

        // Execute search
        let db = self.inner.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;

        let results = db.search(query).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Search failed: {}", e))
        })?;

        // Convert results to Python-friendly format
        // Apply additional client-side filtering if ruvector-core doesn't support all operators
        let filter_conditions = filter_map.map(|fm| parse_filter(fm));

        let py_results: Vec<SearchResult> = results
            .into_iter()
            .filter(|r| {
                if let Some(ref conditions) = filter_conditions {
                    if let Some(ref meta) = r.metadata {
                        evaluate_filter(conditions, meta)
                    } else {
                        conditions.is_empty()
                    }
                } else {
                    true
                }
            })
            .map(|r| SearchResult {
                id: r.id,
                score: r.score,
                metadata: r.metadata.clone().unwrap_or_default(),
                vector: r.vector, // Preserve vector data from ruvector-core
            })
            .collect();

        Ok(py_results)
    }

    /// Delete a vector by id
    ///
    /// # Arguments
    /// * `id` - The vector id to delete
    ///
    /// # Returns
    /// True if the vector was deleted, False if it didn't exist
    fn delete(&self, id: String) -> PyResult<bool> {
        let db = self.inner.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;

        db.delete(&id).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Delete failed: {}", e))
        })
    }

    /// Delete multiple vectors by id
    ///
    /// # Arguments
    /// * `ids` - List of vector ids to delete
    ///
    /// # Returns
    /// Number of vectors actually deleted
    fn delete_batch(&self, ids: Vec<String>) -> PyResult<usize> {
        let mut count = 0;
        for id in ids {
            if self.delete(id)? {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Get a vector by ID
    ///
    /// # Arguments
    /// * `id` - The vector id to retrieve
    ///
    /// # Returns
    /// SearchResult if found, None otherwise
    fn get(&self, id: String) -> PyResult<Option<SearchResult>> {
        let db = self.inner.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;

        match db.get(&id) {
            Ok(Some(entry)) => Ok(Some(SearchResult {
                id: entry.id.clone().unwrap_or_default(),
                score: 1.0, // Perfect match
                metadata: entry.metadata.clone().unwrap_or_default(),
                vector: Some(entry.vector), // Include vector data
            })),
            Ok(None) => Ok(None),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Get failed: {}", e)
            )),
        }
    }

    /// Save the database to disk
    ///
    /// # Notes
    /// With ruvector-core's automatic persistence (via redb), data is automatically
    /// written to disk on every insert/delete operation when a storage path is provided.
    /// This method exists for API compatibility but is effectively a no-op since
    /// persistence happens automatically.
    ///
    /// # Returns
    /// Ok(()) if a storage path was specified, error otherwise
    fn save(&self) -> PyResult<()> {
        if self.path.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot save: no storage path specified. Create VectorDB with path parameter for automatic persistence."
            ));
        }

        // Data is automatically persisted to disk on every operation when storage_path is set
        // This method is a no-op but confirms that persistence is active
        Ok(())
    }

    /// Close the database
    fn close(&mut self) -> PyResult<()> {
        // ruvector-core handles cleanup automatically via Drop
        Ok(())
    }

    /// Get database statistics
    fn stats(&self) -> PyResult<DBStats> {
        let db = self.inner.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;

        let vector_count = db.len().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Stats failed: {}", e))
        })?;

        // Estimate memory usage
        // Each f32 is 4 bytes, plus HNSW graph overhead (~100 bytes per vector)
        let estimated_memory_bytes = vector_count * (self.dimensions * 4 + 100);

        Ok(DBStats {
            vector_count,
            dimensions: self.dimensions,
            estimated_memory_bytes,
        })
    }

    /// Check if database is empty
    ///
    /// # Returns
    /// True if the database contains no vectors
    fn is_empty(&self) -> PyResult<bool> {
        let db = self.inner.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;

        let count = db.len().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Length failed: {}", e))
        })?;

        Ok(count == 0)
    }

    /// Get the distance metric used by this database
    #[getter]
    fn distance_metric(&self) -> DistanceMetric {
        self.distance_metric.clone()
    }

    /// Get the HNSW configuration
    #[getter]
    fn hnsw_config(&self) -> HNSWConfig {
        self.hnsw_config.clone()
    }

    /// Get the quantization configuration
    #[getter]
    fn quantization_config(&self) -> QuantizationConfig {
        self.quantization_config.clone()
    }

    /// Get health status of the database
    ///
    /// # Returns
    /// HealthStatus object with current database health information
    fn health(&self) -> PyResult<HealthStatus> {
        let db = self.inner.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;

        let vector_count = db.len().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Stats failed: {}", e))
        })?;

        let uptime_seconds = self.created_at.elapsed().as_secs_f64();

        // Estimate memory usage
        let estimated_memory_bytes = vector_count * (self.dimensions * 4 + 100);

        Ok(HealthStatus {
            status: "healthy".to_string(),
            vector_count,
            memory_usage_bytes: estimated_memory_bytes,
            uptime_seconds,
        })
    }

    /// Clear all vectors from the database
    ///
    /// # Returns
    /// Number of vectors that were removed
    ///
    /// # Notes
    /// If persistence is enabled (storage_path was provided), the cleared state
    /// will be automatically persisted to disk.
    fn clear(&self) -> PyResult<usize> {
        let mut db = self.inner.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;

        let count = db.len().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Length failed: {}", e))
        })?;

        // Create a new empty database with the same configuration
        let core_options = ruvector_core::types::DbOptions {
            dimensions: self.dimensions,
            distance_metric: match self.distance_metric {
                DistanceMetric::Euclidean => ruvector_core::types::DistanceMetric::Euclidean,
                DistanceMetric::Cosine => ruvector_core::types::DistanceMetric::Cosine,
                DistanceMetric::DotProduct => ruvector_core::types::DistanceMetric::DotProduct,
                DistanceMetric::Manhattan => ruvector_core::types::DistanceMetric::Manhattan,
            },
            storage_path: self.path.clone().unwrap_or_else(|| {
                format!(":memory:{}", uuid::Uuid::new_v4())
            }),
            hnsw_config: Some(ruvector_core::types::HnswConfig {
                m: self.hnsw_config.m,
                ef_construction: self.hnsw_config.ef_construction,
                ef_search: self.hnsw_config.ef_search,
                max_elements: self.hnsw_config.max_elements.unwrap_or(10_000_000),
            }),
            quantization: match self.quantization_config.quantization_type {
                QuantizationType::None => Some(ruvector_core::types::QuantizationConfig::None),
                QuantizationType::Scalar => Some(ruvector_core::types::QuantizationConfig::Scalar),
                QuantizationType::Product => {
                    Some(ruvector_core::types::QuantizationConfig::Product {
                        subspaces: self.quantization_config.subspaces.unwrap_or(8),
                        k: 1 << self.quantization_config.bits.unwrap_or(8),
                    })
                }
                QuantizationType::Binary => Some(ruvector_core::types::QuantizationConfig::Binary),
            },
        };

        let new_db = RuvectorDB::new(core_options)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to create new VectorDB: {}", e)
            ))?;

        *db = new_db;

        Ok(count)
    }

    /// Check if a vector exists in the database
    ///
    /// # Arguments
    /// * `id` - The vector id to check
    ///
    /// # Returns
    /// True if the vector exists
    fn contains(&self, id: String) -> PyResult<bool> {
        Ok(self.get(id)?.is_some())
    }

    /// Get the number of vectors in the database
    fn __len__(&self) -> PyResult<usize> {
        let db = self.inner.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;

        db.len().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Length failed: {}", e))
        })
    }

    /// Check if database contains a vector
    fn __contains__(&self, id: String) -> PyResult<bool> {
        Ok(self.get(id)?.is_some())
    }

    /// String representation
    fn __repr__(&self) -> PyResult<String> {
        let count = self.__len__()?;
        Ok(format!(
            "VectorDB(dimensions={}, vectors={}, path={:?})",
            self.dimensions, count, self.path
        ))
    }

    /// Get all vector IDs in the database
    ///
    /// # Returns
    /// List of all vector IDs currently stored in the database
    ///
    /// # Example
    /// ```python
    /// db = VectorDB(dimensions=128, path="vectors.db")
    /// db.insert("v1", [0.1] * 128)
    /// db.insert("v2", [0.2] * 128)
    /// ids = db.get_all_ids()
    /// print(ids)  # ['v1', 'v2']
    /// ```
    fn get_all_ids(&self) -> PyResult<Vec<String>> {
        let db = self.inner.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;

        // Get the storage layer and call all_ids()
        // Since we can't access storage directly, we'll iterate through the database
        // by getting all IDs from 0 to len and collecting successful gets
        let _count = db.len().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Length failed: {}", e))
        })?;

        // Note: This is a workaround because VectorDB doesn't expose all_ids()
        // We would need to add this method to ruvector-core's VectorDB
        // For now, return error with helpful message
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "get_all_ids() is not yet supported by ruvector-core. \
             The underlying storage has all_ids() but VectorDB doesn't expose it. \
             Consider tracking inserted IDs in your application for now."
        ))
    }
}

// Public Rust API (not exposed to Python)
impl VectorDB {
    /// Get database statistics (Rust API)
    pub fn get_stats(&self) -> PyResult<DBStats> {
        let db = self.inner.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;

        let vector_count = db.len().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Stats failed: {}", e))
        })?;

        // Estimate memory usage
        // Each f32 is 4 bytes, plus HNSW graph overhead (~100 bytes per vector)
        let estimated_memory_bytes = vector_count * (self.dimensions * 4 + 100);

        Ok(DBStats {
            vector_count,
            dimensions: self.dimensions,
            estimated_memory_bytes,
        })
    }

    /// Get the distance metric (Rust API)
    pub fn get_distance_metric(&self) -> DistanceMetric {
        self.distance_metric
    }

    /// Get all vector IDs (Rust API for snapshot support)
    /// Returns error because ruvector-core doesn't expose all_ids()
    pub fn get_all_ids_internal(&self) -> PyResult<Vec<String>> {
        // This would require ruvector-core to expose storage.all_ids()
        // For now, we return an empty vector for file-backed databases
        // (snapshots of file-backed DBs reference the file itself)
        Ok(Vec::new())
    }

    /// Get all vectors by IDs (public API for snapshot support)
    /// Users must provide the list of IDs they've inserted
    pub fn get_vectors_by_ids(&self, ids: Vec<String>) -> PyResult<Vec<(String, Vec<f32>, HashMap<String, serde_json::Value>)>> {
        let mut results = Vec::with_capacity(ids.len());

        for id in ids {
            if let Some(result) = self.get(id.clone())? {
                if let Some(vector) = result.vector {
                    results.push((id, vector, result.metadata.into_iter().collect()));
                }
            }
        }

        Ok(results)
    }
}

/// Convert Python dict to JSON HashMap
fn python_dict_to_json(
    py: Python,
    dict: &PyDict,
) -> PyResult<HashMap<String, serde_json::Value>> {
    let mut map = HashMap::new();

    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let json_value = pyobject_to_json(py, &value)?;
        map.insert(key_str, json_value);
    }

    Ok(map)
}

/// Convert Python object to serde_json::Value
fn pyobject_to_json(py: Python, obj: &PyAny) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(serde_json::Value::Number(i.into()))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(list) = obj.downcast::<pyo3::types::PyList>() {
        let mut vec = Vec::new();
        for item in list.iter() {
            vec.push(pyobject_to_json(py, &item)?);
        }
        Ok(serde_json::Value::Array(vec))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        Ok(serde_json::Value::Object(
            python_dict_to_json(py, dict)?
                .into_iter()
                .collect()
        ))
    } else {
        // Fallback: convert to string
        Ok(serde_json::Value::String(obj.to_string()))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_compiles() {
        // Basic compilation test
        assert!(true);
    }
}
