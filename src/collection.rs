//! Collection management for multi-tenancy support

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::db::VectorDB;
use crate::types::{CollectionStats, DbOptions, DistanceMetric, HNSWConfig, QuantizationConfig};

/// Manages multiple named vector collections
#[pyclass]
pub struct CollectionManager {
    base_path: Option<PathBuf>,
    collections: Arc<RwLock<HashMap<String, VectorDB>>>,
    aliases: Arc<RwLock<HashMap<String, String>>>, // alias -> collection name
}

#[pymethods]
impl CollectionManager {
    #[new]
    #[pyo3(signature = (base_path=None))]
    fn new(base_path: Option<String>) -> PyResult<Self> {
        Ok(Self {
            base_path: base_path.map(PathBuf::from),
            collections: Arc::new(RwLock::new(HashMap::new())),
            aliases: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Create a new collection
    #[pyo3(signature = (name, dimensions, distance_metric=None))]
    fn create_collection(
        &self,
        name: String,
        dimensions: usize,
        distance_metric: Option<DistanceMetric>,
    ) -> PyResult<()> {
        // Validate collection name
        if name.is_empty() {
            return Err(PyValueError::new_err("Collection name cannot be empty"));
        }

        // Check if collection already exists
        {
            let collections = self
                .collections
                .read()
                .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

            if collections.contains_key(&name) {
                return Err(PyValueError::new_err(format!(
                    "Collection '{}' already exists",
                    name
                )));
            }
        }

        // Create the VectorDB with specified options
        let options = DbOptions {
            dimensions,
            distance_metric: distance_metric.unwrap_or_default(),
            storage_path: None,
            hnsw_config: HNSWConfig::default(),
            quantization: QuantizationConfig::default(),
        };

        let db = VectorDB::with_options(options)?;

        // Insert into collections
        {
            let mut collections = self
                .collections
                .write()
                .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

            collections.insert(name.clone(), db);
        }

        Ok(())
    }

    /// Get an existing collection by name or alias
    fn get_collection(&self, name: String) -> PyResult<VectorDB> {
        // Resolve alias if exists
        let collection_name = {
            let aliases = self
                .aliases
                .read()
                .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

            aliases.get(&name).cloned().unwrap_or(name.clone())
        };

        // Get the collection
        let collections = self
            .collections
            .read()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        collections.get(&collection_name).cloned().ok_or_else(|| {
            PyKeyError::new_err(format!("Collection '{}' not found", collection_name))
        })
    }

    /// List all collection names
    fn list_collections(&self) -> PyResult<Vec<String>> {
        let collections = self
            .collections
            .read()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let mut names: Vec<String> = collections.keys().cloned().collect();
        names.sort();
        Ok(names)
    }

    /// Delete a collection
    fn delete_collection(&self, name: String) -> PyResult<bool> {
        // Remove any aliases pointing to this collection
        {
            let mut aliases = self
                .aliases
                .write()
                .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

            let aliases_to_remove: Vec<String> = aliases
                .iter()
                .filter(|(_, target)| *target == &name)
                .map(|(alias, _)| alias.clone())
                .collect();

            for alias in aliases_to_remove {
                aliases.remove(&alias);
            }
        }

        // Remove the collection
        let mut collections = self
            .collections
            .write()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        Ok(collections.remove(&name).is_some())
    }

    /// Get collection statistics
    fn get_stats(&self, name: String) -> PyResult<CollectionStats> {
        // Resolve alias if exists
        let collection_name = {
            let aliases = self
                .aliases
                .read()
                .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

            aliases.get(&name).cloned().unwrap_or(name.clone())
        };

        // Get the collection
        let collections = self
            .collections
            .read()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let db = collections.get(&collection_name).ok_or_else(|| {
            PyKeyError::new_err(format!("Collection '{}' not found", collection_name))
        })?;

        // Get database statistics
        let db_stats = db.get_stats()?;

        // Create CollectionStats from DBStats
        Ok(CollectionStats::new(
            collection_name,
            db_stats.vector_count,
            db_stats.dimensions,
            db.get_distance_metric(),
            db_stats.estimated_memory_bytes,
        ))
    }

    /// Create an alias for a collection
    fn create_alias(&self, alias: String, collection: String) -> PyResult<()> {
        // Validate inputs
        if alias.is_empty() {
            return Err(PyValueError::new_err("Alias name cannot be empty"));
        }
        if collection.is_empty() {
            return Err(PyValueError::new_err("Collection name cannot be empty"));
        }

        // Check if collection exists
        {
            let collections = self
                .collections
                .read()
                .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

            if !collections.contains_key(&collection) {
                return Err(PyKeyError::new_err(format!(
                    "Collection '{}' not found",
                    collection
                )));
            }
        }

        // Create or update alias
        {
            let mut aliases = self
                .aliases
                .write()
                .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

            aliases.insert(alias, collection);
        }

        Ok(())
    }

    /// Delete an alias
    fn delete_alias(&self, alias: String) -> PyResult<bool> {
        let mut aliases = self
            .aliases
            .write()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        Ok(aliases.remove(&alias).is_some())
    }

    /// List all aliases with their target collections
    fn list_aliases(&self) -> PyResult<Vec<(String, String)>> {
        let aliases = self
            .aliases
            .read()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let mut result: Vec<(String, String)> = aliases
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        result.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(result)
    }

    /// Check if collection exists
    fn has_collection(&self, name: String) -> PyResult<bool> {
        let collections = self
            .collections
            .read()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        Ok(collections.contains_key(&name))
    }

    fn __repr__(&self) -> PyResult<String> {
        let collections = self
            .collections
            .read()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let aliases = self
            .aliases
            .read()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let base_path_str = match &self.base_path {
            Some(path) => format!("base_path='{}', ", path.display()),
            None => String::new(),
        };

        Ok(format!(
            "CollectionManager({}collections={}, aliases={})",
            base_path_str,
            collections.len(),
            aliases.len()
        ))
    }

    fn __len__(&self) -> PyResult<usize> {
        let collections = self
            .collections
            .read()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        Ok(collections.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_get_collection() {
        let manager = CollectionManager::new(None).unwrap();

        // Create collection
        manager
            .create_collection("test".to_string(), 128, None)
            .unwrap();

        // Get collection
        let db = manager.get_collection("test".to_string()).unwrap();
        assert!(db.len().unwrap() == 0);

        // Check exists
        assert!(manager.has_collection("test".to_string()).unwrap());
    }

    #[test]
    fn test_list_collections() {
        let manager = CollectionManager::new(None).unwrap();

        manager
            .create_collection("col1".to_string(), 128, None)
            .unwrap();
        manager
            .create_collection("col2".to_string(), 256, None)
            .unwrap();

        let collections = manager.list_collections().unwrap();
        assert_eq!(collections, vec!["col1", "col2"]);
    }

    #[test]
    fn test_delete_collection() {
        let manager = CollectionManager::new(None).unwrap();

        manager
            .create_collection("test".to_string(), 128, None)
            .unwrap();
        assert!(manager.has_collection("test".to_string()).unwrap());

        let deleted = manager.delete_collection("test".to_string()).unwrap();
        assert!(deleted);
        assert!(!manager.has_collection("test".to_string()).unwrap());
    }

    #[test]
    fn test_aliases() {
        let manager = CollectionManager::new(None).unwrap();

        manager
            .create_collection("original".to_string(), 128, None)
            .unwrap();
        manager
            .create_alias("alias1".to_string(), "original".to_string())
            .unwrap();

        // Get via alias
        let db = manager.get_collection("alias1".to_string()).unwrap();
        assert!(db.len().unwrap() == 0);

        // List aliases
        let aliases = manager.list_aliases().unwrap();
        assert_eq!(
            aliases,
            vec![("alias1".to_string(), "original".to_string())]
        );

        // Delete alias
        assert!(manager.delete_alias("alias1".to_string()).unwrap());
        assert_eq!(manager.list_aliases().unwrap().len(), 0);
    }

    #[test]
    fn test_delete_collection_removes_aliases() {
        let manager = CollectionManager::new(None).unwrap();

        manager
            .create_collection("test".to_string(), 128, None)
            .unwrap();
        manager
            .create_alias("alias1".to_string(), "test".to_string())
            .unwrap();
        manager
            .create_alias("alias2".to_string(), "test".to_string())
            .unwrap();

        manager.delete_collection("test".to_string()).unwrap();

        // Aliases should be removed
        assert_eq!(manager.list_aliases().unwrap().len(), 0);
    }

    #[test]
    fn test_duplicate_collection_error() {
        let manager = CollectionManager::new(None).unwrap();

        manager
            .create_collection("test".to_string(), 128, None)
            .unwrap();
        let result = manager.create_collection("test".to_string(), 128, None);

        assert!(result.is_err());
    }

    #[test]
    fn test_get_nonexistent_collection() {
        let manager = CollectionManager::new(None).unwrap();
        let result = manager.get_collection("nonexistent".to_string());

        assert!(result.is_err());
    }

    #[test]
    fn test_alias_to_nonexistent_collection() {
        let manager = CollectionManager::new(None).unwrap();
        let result = manager.create_alias("alias".to_string(), "nonexistent".to_string());

        assert!(result.is_err());
    }
}
