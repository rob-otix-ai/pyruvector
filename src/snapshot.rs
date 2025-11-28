//! Snapshot and backup functionality for vector databases
//!
//! This module wraps ruvector-snapshot and provides Python bindings for
//! creating, managing, and restoring database snapshots.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use std::path::PathBuf;
use std::sync::Arc;

use ruvector_snapshot::{
    SnapshotManager as RuvectorSnapshotManager,
    Snapshot as RuvectorSnapshot,
    SnapshotData as RuvectorSnapshotData,
    VectorRecord as RuvectorVectorRecord,
    LocalStorage,
    SnapshotStorage,
};

use crate::db::VectorDB;
use crate::types::DistanceMetric;

/// Snapshot metadata information for Python
#[pyclass]
#[derive(Clone, Debug)]
pub struct SnapshotInfo {
    /// Snapshot ID
    #[pyo3(get)]
    pub id: String,

    /// Collection/database name
    #[pyo3(get)]
    pub name: String,

    /// ISO 8601 timestamp when snapshot was created
    #[pyo3(get)]
    pub created_at: String,

    /// Number of vectors in the snapshot
    #[pyo3(get)]
    pub vector_count: usize,

    /// Vector dimensions
    #[pyo3(get)]
    pub dimensions: usize,

    /// Snapshot size in bytes (compressed)
    #[pyo3(get)]
    pub size_bytes: u64,

    /// SHA-256 checksum for data integrity
    #[pyo3(get)]
    pub checksum: String,

    /// Optional description
    #[pyo3(get)]
    pub description: Option<String>,
}

impl From<RuvectorSnapshot> for SnapshotInfo {
    fn from(snapshot: RuvectorSnapshot) -> Self {
        Self {
            id: snapshot.id.clone(),
            name: snapshot.collection_name,
            created_at: snapshot.created_at.to_rfc3339(),
            vector_count: snapshot.vectors_count,
            dimensions: 0, // Will be populated from config
            size_bytes: snapshot.size_bytes,
            checksum: snapshot.checksum,
            description: None,
        }
    }
}

#[pymethods]
impl SnapshotInfo {
    /// Get snapshot size in human-readable format (MB)
    #[getter]
    fn size_mb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get snapshot size in human-readable format (GB)
    #[getter]
    fn size_gb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "SnapshotInfo(id='{}', name='{}', created_at='{}', vector_count={}, dimensions={}, size_mb={:.2}, checksum='{}')",
            self.id,
            self.name,
            self.created_at,
            self.vector_count,
            self.dimensions,
            self.size_mb(),
            &self.checksum[..8.min(self.checksum.len())]
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Manager for creating and restoring vector database snapshots
///
/// This wraps ruvector-snapshot's SnapshotManager and provides automatic
/// compression, checksumming, and persistence.
#[pyclass]
pub struct SnapshotManager {
    storage_path: PathBuf,
    inner: Arc<RuvectorSnapshotManager>,
    runtime: Arc<tokio::runtime::Runtime>,
}

// Helper function to build SnapshotData using JSON serialization
// This is necessary because CollectionConfig and related types are not re-exported
fn build_snapshot_data(
    collection_name: String,
    dimensions: usize,
    metric: &DistanceMetric,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,
    vectors: Vec<RuvectorVectorRecord>,
) -> Result<RuvectorSnapshotData, String> {
    use serde_json::json;

    // Convert our metric to string for JSON
    let metric_str = match metric {
        DistanceMetric::Euclidean => "Euclidean",
        DistanceMetric::Cosine => "Cosine",
        DistanceMetric::DotProduct => "DotProduct",
        DistanceMetric::Manhattan => "Euclidean", // Fallback
    };

    // Build JSON representation
    let snapshot_json = json!({
        "metadata": {
            "id": uuid::Uuid::new_v4().to_string(),
            "collection_name": collection_name,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "version": env!("CARGO_PKG_VERSION"),
        },
        "config": {
            "dimension": dimensions,
            "metric": metric_str,
            "hnsw_config": {
                "m": hnsw_m,
                "ef_construction": hnsw_ef_construction,
                "ef_search": hnsw_ef_search,
            }
        },
        "vectors": vectors.iter().map(|v| {
            json!({
                "id": v.id,
                "vector": v.vector,
                "payload_json": v.payload().and_then(|p| serde_json::to_string(&p).ok()),
            })
        }).collect::<Vec<_>>()
    });

    // Deserialize into SnapshotData
    serde_json::from_value(snapshot_json)
        .map_err(|e| format!("Failed to construct snapshot data: {}", e))
}

#[pymethods]
impl SnapshotManager {
    /// Create a new SnapshotManager
    ///
    /// # Arguments
    /// * `storage_path` - Directory path where snapshots will be stored
    ///
    /// # Example
    /// ```python
    /// manager = SnapshotManager("/path/to/snapshots")
    /// ```
    #[new]
    pub fn new(storage_path: String) -> PyResult<Self> {
        let path = PathBuf::from(&storage_path);

        // Create storage directory if it doesn't exist
        if !path.exists() {
            std::fs::create_dir_all(&path).map_err(|e| {
                PyIOError::new_err(format!("Failed to create storage directory: {}", e))
            })?;
        }

        // Create LocalStorage backend
        let storage = Box::new(LocalStorage::new(path.clone())) as Box<dyn SnapshotStorage>;

        // Create the ruvector SnapshotManager
        let inner = RuvectorSnapshotManager::new(storage);

        // Create tokio runtime for async operations
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create async runtime: {}", e))
            })?;

        Ok(Self {
            storage_path: path,
            inner: Arc::new(inner),
            runtime: Arc::new(runtime),
        })
    }

    /// Create a snapshot of a VectorDB with user-provided vector IDs
    ///
    /// # Arguments
    /// * `db` - VectorDB instance to snapshot
    /// * `name` - Name for the snapshot (used as collection_name)
    /// * `vector_ids` - List of vector IDs to include in snapshot (required due to API limitation)
    /// * `description` - Optional description
    ///
    /// # Returns
    /// SnapshotInfo object with snapshot metadata
    ///
    /// # Example
    /// ```python
    /// db = VectorDB(dimensions=128, path="my_vectors.db")
    ///
    /// # Track IDs as you insert vectors
    /// ids = []
    /// ids.append("v1")
    /// db.insert("v1", [0.1] * 128)
    /// ids.append("v2")
    /// db.insert("v2", [0.2] * 128)
    ///
    /// # Create snapshot with tracked IDs
    /// manager = SnapshotManager("/snapshots")
    /// info = manager.create_snapshot_with_ids(db, "backup-2024", ids, description="Daily backup")
    /// print(f"Created snapshot: {info.id} ({info.size_mb:.2f} MB)")
    /// ```
    ///
    /// # Note
    /// You must track vector IDs in your application because ruvector-core
    /// doesn't expose an iterator over stored vectors.
    #[pyo3(signature = (db, name, vector_ids, description=None))]
    pub fn create_snapshot_with_ids(
        &self,
        db: &VectorDB,
        name: String,
        vector_ids: Vec<String>,
        description: Option<String>,
    ) -> PyResult<SnapshotInfo> {
        // Validate snapshot name
        if name.is_empty() || name.contains('/') || name.contains('\\') {
            return Err(PyValueError::new_err("Invalid snapshot name"));
        }

        // Get database configuration
        let stats = db.get_stats()?;
        let distance_metric = db.get_distance_metric();

        // Extract vectors using provided IDs
        let vectors = self.extract_vectors_with_ids(db, vector_ids)?;

        if vectors.is_empty() {
            return Err(PyValueError::new_err(
                "Cannot create snapshot: No valid vectors found. \n\
                 Ensure the provided vector_ids exist in the database."
            ));
        }

        // Build snapshot data using the helper
        let snapshot_data = build_snapshot_data(
            name.clone(),
            stats.dimensions,
            &distance_metric,
            db.hnsw_config.m,
            db.hnsw_config.ef_construction,
            db.hnsw_config.ef_search,
            vectors,
        ).map_err(|e| PyRuntimeError::new_err(format!("Failed to build snapshot: {}", e)))?;

        // Store description in metadata if needed
        let _description = description; // Store for later use if needed

        // Create the snapshot using the async runtime
        let inner = Arc::clone(&self.inner);
        let snapshot = self.runtime.block_on(async move {
            inner.create_snapshot(snapshot_data).await
        }).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create snapshot: {}", e))
        })?;

        // Convert to Python-friendly SnapshotInfo
        let mut info = SnapshotInfo::from(snapshot);
        info.dimensions = stats.dimensions;

        Ok(info)
    }

    /// Create a snapshot of a VectorDB (deprecated - use create_snapshot_with_ids)
    ///
    /// # Arguments
    /// * `db` - VectorDB instance to snapshot
    /// * `name` - Name for the snapshot (used as collection_name)
    /// * `description` - Optional description
    ///
    /// # Returns
    /// SnapshotInfo object with snapshot metadata
    ///
    /// # Note
    /// This method is deprecated and will fail with an error explaining the limitation.
    /// Use create_snapshot_with_ids() instead and track your vector IDs.
    #[pyo3(signature = (db, name, description=None))]
    pub fn create_snapshot(
        &self,
        db: &VectorDB,
        name: String,
        description: Option<String>,
    ) -> PyResult<SnapshotInfo> {
        // Validate snapshot name
        if name.is_empty() || name.contains('/') || name.contains('\\') {
            return Err(PyValueError::new_err("Invalid snapshot name"));
        }

        // Get database configuration
        let stats = db.get_stats()?;
        let distance_metric = db.get_distance_metric();

        // Extract vectors from the database
        let vectors = self.extract_vectors_from_db(db)?;

        // Build snapshot data using the helper
        let snapshot_data = build_snapshot_data(
            name.clone(),
            stats.dimensions,
            &distance_metric,
            db.hnsw_config.m,
            db.hnsw_config.ef_construction,
            db.hnsw_config.ef_search,
            vectors,
        ).map_err(|e| PyRuntimeError::new_err(format!("Failed to build snapshot: {}", e)))?;

        // Store description in metadata if needed
        let _description = description; // Store for later use if needed

        // Create the snapshot using the async runtime
        let inner = Arc::clone(&self.inner);
        let snapshot = self.runtime.block_on(async move {
            inner.create_snapshot(snapshot_data).await
        }).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create snapshot: {}", e))
        })?;

        // Convert to Python-friendly SnapshotInfo
        let mut info = SnapshotInfo::from(snapshot);
        info.dimensions = stats.dimensions;

        Ok(info)
    }

    /// List all available snapshots
    ///
    /// # Returns
    /// List of SnapshotInfo objects
    ///
    /// # Example
    /// ```python
    /// manager = SnapshotManager("/snapshots")
    /// for snapshot in manager.list_snapshots():
    ///     print(f"{snapshot.name}: {snapshot.vector_count} vectors")
    /// ```
    pub fn list_snapshots(&self) -> PyResult<Vec<SnapshotInfo>> {
        let inner = Arc::clone(&self.inner);
        let snapshots = self.runtime.block_on(async move {
            inner.list_snapshots().await
        }).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to list snapshots: {}", e))
        })?;

        Ok(snapshots.into_iter().map(SnapshotInfo::from).collect())
    }

    /// Restore a VectorDB from a snapshot
    ///
    /// # Arguments
    /// * `snapshot_id` - ID of the snapshot to restore
    ///
    /// # Returns
    /// Restored VectorDB instance
    ///
    /// # Example
    /// ```python
    /// manager = SnapshotManager("/snapshots")
    /// db = manager.restore_snapshot("abc-123-def-456")
    /// print(f"Restored {len(db)} vectors")
    /// ```
    pub fn restore_snapshot(&self, snapshot_id: String) -> PyResult<VectorDB> {
        let inner = Arc::clone(&self.inner);
        let snapshot_id_clone = snapshot_id.clone();

        let snapshot_data = self.runtime.block_on(async move {
            inner.restore_snapshot(&snapshot_id_clone).await
        }).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to restore snapshot: {}", e))
        })?;

        // Convert ruvector-snapshot types back to VectorDB
        self.create_db_from_snapshot_data(snapshot_data)
    }

    /// Delete a snapshot
    ///
    /// # Arguments
    /// * `snapshot_id` - ID of the snapshot to delete
    ///
    /// # Returns
    /// True if snapshot was deleted
    ///
    /// # Example
    /// ```python
    /// manager = SnapshotManager("/snapshots")
    /// manager.delete_snapshot("abc-123-def-456")
    /// ```
    pub fn delete_snapshot(&self, snapshot_id: String) -> PyResult<bool> {
        let inner = Arc::clone(&self.inner);
        let snapshot_id_clone = snapshot_id.clone();

        self.runtime.block_on(async move {
            inner.delete_snapshot(&snapshot_id_clone).await
        }).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to delete snapshot: {}", e))
        })?;

        Ok(true)
    }

    /// Get information about a specific snapshot
    ///
    /// # Arguments
    /// * `snapshot_id` - ID of the snapshot
    ///
    /// # Returns
    /// SnapshotInfo if found, None otherwise
    ///
    /// # Example
    /// ```python
    /// manager = SnapshotManager("/snapshots")
    /// info = manager.get_snapshot_info("abc-123-def-456")
    /// if info:
    ///     print(f"Snapshot contains {info.vector_count} vectors")
    /// ```
    pub fn get_snapshot_info(&self, snapshot_id: String) -> PyResult<Option<SnapshotInfo>> {
        let inner = Arc::clone(&self.inner);
        let snapshot_id_clone = snapshot_id.clone();

        let snapshot = self.runtime.block_on(async move {
            inner.get_snapshot_info(&snapshot_id_clone).await
        });

        match snapshot {
            Ok(s) => Ok(Some(SnapshotInfo::from(s))),
            Err(_) => Ok(None),
        }
    }

    /// Get total size of all snapshots in bytes
    #[getter]
    fn total_size_bytes(&self) -> PyResult<u64> {
        let inner = Arc::clone(&self.inner);

        self.runtime.block_on(async move {
            inner.total_size().await
        }).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get total size: {}", e))
        })
    }

    /// Get total size of all snapshots in MB
    #[getter]
    fn total_size_mb(&self) -> PyResult<f64> {
        Ok(self.total_size_bytes()? as f64 / (1024.0 * 1024.0))
    }

    /// Get number of snapshots
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.list_snapshots()?.len())
    }

    fn __repr__(&self) -> PyResult<String> {
        let count = self.__len__()?;
        let total_mb = self.total_size_mb()?;
        Ok(format!(
            "SnapshotManager(path='{}', snapshots={}, total_size_mb={:.2})",
            self.storage_path.display(),
            count,
            total_mb
        ))
    }
}

// Private implementation methods
impl SnapshotManager {
    /// Extract vectors from a VectorDB instance using provided IDs
    ///
    /// This method retrieves vectors from the database using the provided list of IDs.
    /// It's used by both create_snapshot_with_ids() and the deprecated create_snapshot().
    fn extract_vectors_with_ids(&self, db: &VectorDB, vector_ids: Vec<String>) -> PyResult<Vec<RuvectorVectorRecord>> {
        if vector_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Use the public API method to get vectors
        let results = db.get_vectors_by_ids(vector_ids)?;

        let mut vectors = Vec::with_capacity(results.len());

        for (id, vector_data, metadata) in results {
            // Convert metadata HashMap to serde_json::Value if present
            let payload = if !metadata.is_empty() {
                Some(serde_json::Value::Object(
                    metadata.into_iter().collect()
                ))
            } else {
                None
            };

            // Create RuvectorVectorRecord
            vectors.push(RuvectorVectorRecord::new(
                id,
                vector_data,
                payload,
            ));
        }

        Ok(vectors)
    }

    /// Extract vectors from a VectorDB instance (deprecated)
    ///
    /// # Current Limitation
    /// ruvector-core's VectorDB does not expose an iterator or all_ids() method,
    /// so we cannot extract vectors from the database. The underlying storage layer
    /// (VectorStorage) has an all_ids() method, but it's not exposed through VectorDB.
    ///
    /// # Workaround for File-backed Databases
    /// For file-backed databases, snapshots should ideally reference the database file
    /// itself rather than duplicating all vectors. However, ruvector-snapshot requires
    /// actual vector data to create snapshots.
    ///
    /// # Solutions
    /// 1. **Track IDs externally** - Keep a list of inserted vector IDs in your application
    ///    and use create_snapshot_with_ids() method
    /// 2. **File-backed databases** - For persistence, use the `path` parameter when creating
    ///    VectorDB. The database automatically persists to disk.
    /// 3. **Future enhancement** - Request ruvector-core to expose storage.all_ids() or
    ///    add an iterator to VectorDB
    fn extract_vectors_from_db(&self, db: &VectorDB) -> PyResult<Vec<RuvectorVectorRecord>> {
        // Attempt to get all IDs (currently returns empty for file-backed DBs)
        let ids = db.get_all_ids_internal()?;

        if ids.is_empty() {
            // Check if database actually has vectors
            let stats = db.get_stats()?;

            if stats.vector_count > 0 {
                // Database has vectors but we can't access them
                if db.path.is_some() {
                    // File-backed database
                    return Err(PyValueError::new_err(
                        "Cannot create snapshot: ruvector-core doesn't expose vector iteration. \n\
                         \n\
                         WORKAROUND: Use create_snapshot_with_ids() instead:\n\
                         \n\
                         # Track vector IDs as you insert\n\
                         ids = []\n\
                         db.insert('v1', vector1)  # ids.append('v1')\n\
                         db.insert('v2', vector2)  # ids.append('v2')\n\
                         \n\
                         # Create snapshot with tracked IDs\n\
                         manager.create_snapshot_with_ids(db, 'backup', ids)\n\
                         \n\
                         For persistence, file-backed databases (created with 'path' parameter) \n\
                         automatically save to disk - no snapshot needed for basic persistence."
                    ));
                } else {
                    // In-memory database
                    return Err(PyValueError::new_err(
                        "Cannot snapshot in-memory database: ruvector-core doesn't expose vector iteration. \n\
                         \n\
                         SOLUTIONS:\n\
                         1. Use create_snapshot_with_ids() and track vector IDs in your application\n\
                         2. Use a file-backed database (provide 'path' parameter) for automatic persistence\n\
                         3. Request ruvector-core to expose storage.all_ids() method"
                    ));
                }
            } else {
                // Empty database - snapshots cannot be empty per ruvector-snapshot requirements
                return Err(PyValueError::new_err(
                    "Cannot create snapshot of empty database. \n\
                     Insert at least one vector before creating a snapshot."
                ));
            }
        }

        // If we have IDs, use the new method
        self.extract_vectors_with_ids(db, ids)
    }

    /// Create a VectorDB from snapshot data
    fn create_db_from_snapshot_data(&self, data: RuvectorSnapshotData) -> PyResult<VectorDB> {
        use crate::types::{DbOptions, HNSWConfig, QuantizationConfig};

        // Since we can't access config fields directly (they're not re-exported),
        // serialize to JSON and extract the values
        let snapshot_json = serde_json::to_value(&data)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize snapshot: {}", e)))?;

        let dimensions = snapshot_json["config"]["dimension"]
            .as_u64()
            .ok_or_else(|| PyRuntimeError::new_err("Missing dimension in snapshot"))?
            as usize;

        let metric_str = snapshot_json["config"]["metric"]
            .as_str()
            .ok_or_else(|| PyRuntimeError::new_err("Missing metric in snapshot"))?;

        // Convert metric string to DistanceMetric
        let distance_metric = match metric_str {
            "Euclidean" => DistanceMetric::Euclidean,
            "Cosine" => DistanceMetric::Cosine,
            "DotProduct" => DistanceMetric::DotProduct,
            _ => DistanceMetric::Cosine, // Default fallback
        };

        // Extract HNSW config
        let hnsw_config = if let Some(hnsw) = snapshot_json["config"]["hnsw_config"].as_object() {
            HNSWConfig {
                m: hnsw.get("m").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
                ef_construction: hnsw.get("ef_construction").and_then(|v| v.as_u64()).unwrap_or(200) as usize,
                ef_search: hnsw.get("ef_search").and_then(|v| v.as_u64()).unwrap_or(200) as usize,
                max_elements: None,
            }
        } else {
            HNSWConfig::default()
        };

        // Create new VectorDB with the configuration
        let options = DbOptions::new(
            dimensions,
            Some(distance_metric),
            None, // No path - create in-memory
            Some(hnsw_config),
            Some(QuantizationConfig::default()),
        );

        let db = VectorDB::with_options(options)?;

        // Insert all vectors from the snapshot
        let vectors = data.vectors;
        for record in vectors {
            // Convert serde_json::Value to PyDict if needed
            Python::with_gil(|py| {
                let metadata = if let Some(payload) = record.payload() {
                    let dict = PyDict::new(py);
                    if let Some(obj) = payload.as_object() {
                        for (key, value) in obj {
                            let py_value = match value {
                                serde_json::Value::String(s) => s.to_object(py),
                                serde_json::Value::Number(n) => {
                                    if let Some(i) = n.as_i64() {
                                        i.to_object(py)
                                    } else if let Some(f) = n.as_f64() {
                                        f.to_object(py)
                                    } else {
                                        continue;
                                    }
                                }
                                serde_json::Value::Bool(b) => b.to_object(py),
                                _ => continue,
                            };
                            dict.set_item(key, py_value)?;
                        }
                    }
                    Some(dict)
                } else {
                    None
                };

                db.insert(py, record.id.clone(), record.vector, metadata)
            })?;
        }

        Ok(db)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_info_creation() {
        use chrono::Utc;

        let ruvector_snapshot = RuvectorSnapshot {
            id: "test-123".to_string(),
            collection_name: "test-collection".to_string(),
            created_at: Utc::now(),
            vectors_count: 1000,
            checksum: "abc123def456".to_string(),
            size_bytes: 1024 * 1024,
        };

        let info = SnapshotInfo::from(ruvector_snapshot);
        assert_eq!(info.id, "test-123");
        assert_eq!(info.vector_count, 1000);
        assert!((info.size_mb() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_manager_creation() {
        let temp_dir = std::env::temp_dir().join("pyruvector-snapshot-test");
        let result = SnapshotManager::new(temp_dir.to_string_lossy().to_string());
        assert!(result.is_ok());

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
    }
}
