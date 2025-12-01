//! PyO3 wrapper for ruvector-tiny-dancer-core and ruvector-router-core
//!
//! This module provides Python bindings for:
//! - FastGRNN neural routing from ruvector-tiny-dancer-core
//! - VectorDB with HNSW indexing from ruvector-router-core
//! - Feature engineering and training infrastructure

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// Import from ruvector-tiny-dancer-core
use ruvector_tiny_dancer_core::{
    types::RouterConfig as TDRouterConfig, Candidate as TDCandidate, Router as TinyDancerRouter,
    RoutingRequest as TDRoutingRequest, RoutingResponse as TDRoutingResponse,
};

// Import from ruvector-router-core
use ruvector_router_core::{
    types::VectorDbConfig, DistanceMetric, SearchQuery, VectorDB as RuvectorDB, VectorEntry,
};

/// Candidate model/endpoint for LLM routing
///
/// Python wrapper for the Tiny Dancer Candidate type.
/// Represents a target LLM endpoint that requests can be routed to.
///
/// # Example
/// ```python
/// from pyruvector import Candidate
///
/// candidate = Candidate(
///     id="gpt-3.5-turbo",
///     embedding=[0.1, 0.2, ...],  # 384-768 dimensions
///     metadata={"model": "gpt-3.5", "provider": "openai"},
///     cost_per_1m_tokens=1.50
/// )
/// ```
#[pyclass]
#[derive(Clone, Debug)]
pub struct Candidate {
    /// Internal Tiny Dancer candidate
    inner: TDCandidate,
}

#[pymethods]
impl Candidate {
    #[new]
    #[pyo3(signature = (id, embedding, metadata=None, _cost_per_1m_tokens=1.0))]
    fn new(
        id: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        _cost_per_1m_tokens: f32,
    ) -> PyResult<Self> {
        // Convert Python metadata to JSON metadata
        let json_metadata: HashMap<String, serde_json::Value> = metadata
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| (k, serde_json::Value::String(v)))
            .collect();

        let inner = TDCandidate {
            id,
            embedding,
            metadata: json_metadata,
            created_at: chrono::Utc::now().timestamp(),
            access_count: 0,
            success_rate: 0.95, // Default success rate
        };

        Ok(Self { inner })
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    fn embedding(&self) -> Vec<f32> {
        self.inner.embedding.clone()
    }

    #[getter]
    fn access_count(&self) -> u64 {
        self.inner.access_count
    }

    #[getter]
    fn success_rate(&self) -> f32 {
        self.inner.success_rate
    }

    fn __repr__(&self) -> String {
        format!(
            "Candidate(id='{}', embedding_dim={}, success_rate={:.2})",
            self.inner.id,
            self.inner.embedding.len(),
            self.inner.success_rate
        )
    }
}

/// Configuration for the neural router
///
/// Python wrapper for Tiny Dancer RouterConfig.
///
/// # Example
/// ```python
/// from pyruvector import RouterConfig
///
/// config = RouterConfig(
///     model_path="./models/fastgrnn.safetensors",
///     confidence_threshold=0.85,
///     enable_circuit_breaker=True
/// )
/// ```
#[pyclass]
#[derive(Clone, Debug)]
pub struct RouterConfig {
    inner: TDRouterConfig,
}

#[pymethods]
impl RouterConfig {
    #[new]
    #[pyo3(signature = (
        model_path="./models/fastgrnn.safetensors",
        confidence_threshold=0.85,
        max_uncertainty=0.15,
        enable_circuit_breaker=true
    ))]
    fn new(
        model_path: &str,
        confidence_threshold: f32,
        max_uncertainty: f32,
        enable_circuit_breaker: bool,
    ) -> Self {
        let inner = TDRouterConfig {
            model_path: model_path.to_string(),
            confidence_threshold,
            max_uncertainty,
            enable_circuit_breaker,
            circuit_breaker_threshold: 5,
            enable_quantization: true,
            database_path: None,
        };

        Self { inner }
    }

    #[getter]
    fn model_path(&self) -> String {
        self.inner.model_path.clone()
    }

    #[getter]
    fn confidence_threshold(&self) -> f32 {
        self.inner.confidence_threshold
    }

    #[getter]
    fn max_uncertainty(&self) -> f32 {
        self.inner.max_uncertainty
    }

    fn __repr__(&self) -> String {
        format!(
            "RouterConfig(confidence_threshold={:.2}, max_uncertainty={:.2})",
            self.inner.confidence_threshold, self.inner.max_uncertainty
        )
    }
}

/// Request to be routed to an optimal LLM candidate
///
/// Python wrapper for Tiny Dancer RoutingRequest.
///
/// # Example
/// ```python
/// from pyruvector import RoutingRequest
///
/// request = RoutingRequest(
///     query_embedding=[0.1, 0.2, ...],  # 384-768 dimensions
///     candidates=[candidate1, candidate2],
///     metadata={"user_id": "123", "priority": "high"}
/// )
/// ```
#[pyclass]
#[derive(Clone, Debug)]
pub struct RoutingRequest {
    #[pyo3(get, set)]
    pub query_embedding: Vec<f32>,
    pub candidates: Vec<Candidate>,
}

#[pymethods]
impl RoutingRequest {
    #[new]
    #[pyo3(signature = (query_embedding, candidates, _metadata=None))]
    fn new(
        query_embedding: Vec<f32>,
        candidates: Vec<Candidate>,
        _metadata: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            query_embedding,
            candidates,
        }
    }

    #[getter]
    fn candidates(&self) -> Vec<Candidate> {
        self.candidates.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RoutingRequest(embedding_dim={}, candidates={})",
            self.query_embedding.len(),
            self.candidates.len()
        )
    }
}

/// Response from the neural router with routing decision
///
/// Python wrapper for Tiny Dancer RoutingResponse.
///
/// # Example
/// ```python
/// response = router.route(request)
/// for decision in response.decisions:
///     print(f"Candidate: {decision.candidate_id}")
///     print(f"Confidence: {decision.confidence:.2%}")
/// ```
#[pyclass]
#[derive(Clone, Debug)]
pub struct RoutingResponse {
    inner: TDRoutingResponse,
}

#[pymethods]
impl RoutingResponse {
    #[getter]
    fn decisions(&self) -> Vec<RoutingDecision> {
        self.inner
            .decisions
            .iter()
            .map(|d| RoutingDecision {
                candidate_id: d.candidate_id.clone(),
                confidence: d.confidence,
                use_lightweight: d.use_lightweight,
                uncertainty: d.uncertainty,
            })
            .collect()
    }

    #[getter]
    fn inference_time_us(&self) -> u64 {
        self.inner.inference_time_us
    }

    #[getter]
    fn candidates_processed(&self) -> usize {
        self.inner.candidates_processed
    }

    fn __repr__(&self) -> String {
        format!(
            "RoutingResponse(decisions={}, inference_time_us={})",
            self.inner.decisions.len(),
            self.inner.inference_time_us
        )
    }
}

/// Individual routing decision for a candidate
#[pyclass]
#[derive(Clone, Debug)]
pub struct RoutingDecision {
    #[pyo3(get)]
    pub candidate_id: String,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub use_lightweight: bool,
    #[pyo3(get)]
    pub uncertainty: f32,
}

#[pymethods]
impl RoutingDecision {
    fn __repr__(&self) -> String {
        format!(
            "RoutingDecision(candidate='{}', confidence={:.2}, use_lightweight={})",
            self.candidate_id, self.confidence, self.use_lightweight
        )
    }
}

/// Training dataset for the neural router
///
/// Simplified training dataset for the router.
/// Note: Full training support requires external tools due to module visibility.
///
/// # Example
/// ```python
/// from pyruvector import TrainingDataset
///
/// dataset = TrainingDataset(
///     features=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
///     labels=[1.0, 0.0]
/// )
/// ```
#[pyclass]
#[derive(Clone, Debug)]
pub struct TrainingDataset {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<f32>,
}

#[pymethods]
impl TrainingDataset {
    #[new]
    fn new(features: Vec<Vec<f32>>, labels: Vec<f32>) -> PyResult<Self> {
        if features.len() != labels.len() {
            return Err(PyValueError::new_err(
                "Features and labels must have the same length",
            ));
        }
        if features.is_empty() {
            return Err(PyValueError::new_err("Dataset cannot be empty"));
        }

        Ok(Self { features, labels })
    }

    fn __len__(&self) -> usize {
        self.features.len()
    }

    fn __repr__(&self) -> String {
        format!("TrainingDataset(examples={})", self.features.len())
    }
}

/// Configuration for training the neural router
///
/// Simplified training configuration.
///
/// # Example
/// ```python
/// from pyruvector import TrainingConfig
///
/// config = TrainingConfig(
///     epochs=100,
///     learning_rate=0.001,
///     batch_size=32
/// )
/// ```
#[pyclass]
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    #[pyo3(get, set)]
    pub epochs: usize,
    #[pyo3(get, set)]
    pub learning_rate: f32,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub validation_split: f32,
}

#[pymethods]
impl TrainingConfig {
    #[new]
    #[pyo3(signature = (
        epochs=100,
        learning_rate=0.001,
        batch_size=32,
        validation_split=0.2
    ))]
    fn new(epochs: usize, learning_rate: f32, batch_size: usize, validation_split: f32) -> Self {
        Self {
            epochs,
            learning_rate,
            batch_size,
            validation_split,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainingConfig(epochs={}, lr={:.4}, batch_size={})",
            self.epochs, self.learning_rate, self.batch_size
        )
    }
}

/// Metrics from training the neural router
///
/// Python wrapper for Tiny Dancer TrainingMetrics.
#[pyclass]
#[derive(Clone, Debug)]
pub struct TrainingMetrics {
    #[pyo3(get)]
    pub epoch: usize,
    #[pyo3(get)]
    pub train_loss: f32,
    #[pyo3(get)]
    pub val_loss: f32,
    #[pyo3(get)]
    pub train_accuracy: f32,
    #[pyo3(get)]
    pub val_accuracy: f32,
}

#[pymethods]
impl TrainingMetrics {
    fn __repr__(&self) -> String {
        format!(
            "TrainingMetrics(epoch={}, train_loss={:.4}, val_loss={:.4}, val_accuracy={:.2})",
            self.epoch, self.train_loss, self.val_loss, self.val_accuracy
        )
    }
}

/// Neural Router for LLM request routing optimization
///
/// This is the main "Tiny Dancer" neural routing system that wraps the
/// ruvector-tiny-dancer-core Router implementation. It uses a FastGRNN
/// model for sub-millisecond routing decisions.
///
/// # Example
/// ```python
/// from pyruvector import NeuralRouter, RouterConfig, Candidate, RoutingRequest
///
/// config = RouterConfig(confidence_threshold=0.85)
/// router = NeuralRouter(config)
///
/// # Add candidates
/// candidate1 = Candidate(
///     id="gpt-3.5-turbo",
///     embedding=[0.1] * 384,
///     cost_per_1m_tokens=1.50
/// )
/// candidate2 = Candidate(
///     id="gpt-4",
///     embedding=[0.2] * 384,
///     cost_per_1m_tokens=30.0
/// )
///
/// # Route a request
/// request = RoutingRequest(
///     query_embedding=[0.15] * 384,
///     candidates=[candidate1, candidate2]
/// )
///
/// response = router.route(request)
/// print(f"Selected: {response.decisions[0].candidate_id}")
/// ```
#[pyclass]
pub struct NeuralRouter {
    router: Arc<RwLock<TinyDancerRouter>>,
}

#[pymethods]
impl NeuralRouter {
    #[new]
    fn new(config: RouterConfig) -> PyResult<Self> {
        let router = TinyDancerRouter::new(config.inner)
            .map_err(|e| PyValueError::new_err(format!("Failed to create router: {}", e)))?;

        Ok(Self {
            router: Arc::new(RwLock::new(router)),
        })
    }

    /// Route a request to the optimal candidate
    ///
    /// Uses the FastGRNN neural model and feature engineering to select
    /// the best candidate based on semantic similarity, cost, and learned patterns.
    ///
    /// # Arguments
    /// * `request` - The routing request with query embedding and candidates
    ///
    /// # Returns
    /// A RoutingResponse with routing decisions sorted by confidence
    fn route(&self, request: RoutingRequest) -> PyResult<RoutingResponse> {
        let router = self.router.read().unwrap();

        // Convert Python candidates to Tiny Dancer candidates
        let td_candidates: Vec<TDCandidate> =
            request.candidates.iter().map(|c| c.inner.clone()).collect();

        // Create Tiny Dancer routing request
        let td_request = TDRoutingRequest {
            query_embedding: request.query_embedding.clone(),
            candidates: td_candidates,
            metadata: None,
        };

        // Perform routing
        let td_response = router
            .route(td_request)
            .map_err(|e| PyValueError::new_err(format!("Routing failed: {}", e)))?;

        Ok(RoutingResponse { inner: td_response })
    }

    /// Train the neural routing model on a dataset
    ///
    /// Note: Full training functionality requires using the ruvector-tiny-dancer-core
    /// crate directly due to module visibility constraints. This is a stub that
    /// returns a placeholder result.
    ///
    /// For production training, use the command-line tools or Rust API directly.
    ///
    /// # Arguments
    /// * `dataset` - Training dataset with features and labels
    /// * `config` - Training configuration (epochs, learning rate, etc.)
    ///
    /// # Returns
    /// List of TrainingMetrics for each epoch (placeholder)
    fn train(
        &self,
        _dataset: &TrainingDataset,
        _config: &TrainingConfig,
    ) -> PyResult<Vec<TrainingMetrics>> {
        // Return placeholder metrics
        // TODO: Expose training API when ruvector-tiny-dancer-core makes it public
        Err(PyValueError::new_err(
            "Training is not yet supported in the Python bindings. \
             Use the ruvector-tiny-dancer-core Rust crate directly for training.",
        ))
    }

    /// Reload the model from disk
    fn reload_model(&self) -> PyResult<()> {
        let router = self.router.read().unwrap();
        router
            .reload_model()
            .map_err(|e| PyValueError::new_err(format!("Failed to reload model: {}", e)))?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        "NeuralRouter(backend='ruvector-tiny-dancer-core')".to_string()
    }
}

/// High-performance vector database with HNSW indexing
///
/// Python wrapper for ruvector-router-core VectorDB.
/// Provides semantic search with SIMD-optimized distance calculations
/// and multiple quantization techniques.
///
/// # Example
/// ```python
/// from pyruvector import VectorDatabase
///
/// db = VectorDatabase(
///     dimensions=384,
///     metric="cosine",
///     storage_path="./vectors.db"
/// )
///
/// # Insert vectors
/// db.insert(
///     id="doc1",
///     vector=[0.1, 0.2, ...],
///     metadata={"title": "Document 1"}
/// )
///
/// # Search
/// results = db.search(
///     query=[0.15, 0.25, ...],
///     k=10
/// )
/// ```
#[pyclass]
pub struct VectorDatabase {
    db: Arc<RuvectorDB>,
}

#[pymethods]
impl VectorDatabase {
    #[new]
    #[pyo3(signature = (
        dimensions=384,
        metric="cosine",
        storage_path="./vectors.db",
        max_elements=1000000
    ))]
    fn new(
        dimensions: usize,
        metric: &str,
        storage_path: &str,
        max_elements: usize,
    ) -> PyResult<Self> {
        let distance_metric = match metric {
            "cosine" => DistanceMetric::Cosine,
            "euclidean" => DistanceMetric::Euclidean,
            "dotproduct" | "dot" => DistanceMetric::DotProduct,
            "manhattan" => DistanceMetric::Manhattan,
            _ => return Err(PyValueError::new_err(format!("Unknown metric: {}", metric))),
        };

        let config = VectorDbConfig {
            dimensions,
            max_elements,
            distance_metric,
            storage_path: storage_path.to_string(),
            ..Default::default()
        };

        let db = RuvectorDB::new(config)
            .map_err(|e| PyValueError::new_err(format!("Failed to create database: {}", e)))?;

        Ok(Self { db: Arc::new(db) })
    }

    /// Insert a vector entry
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the vector
    /// * `vector` - Vector data (must match database dimensions)
    /// * `metadata` - Optional metadata dictionary
    fn insert(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<String> {
        // Convert metadata to JSON
        let json_metadata: HashMap<String, serde_json::Value> = metadata
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| (k, serde_json::Value::String(v)))
            .collect();

        let entry = VectorEntry {
            id: id.clone(),
            vector,
            metadata: json_metadata,
            timestamp: chrono::Utc::now().timestamp(),
        };

        self.db
            .insert(entry)
            .map_err(|e| PyValueError::new_err(format!("Insert failed: {}", e)))
    }

    /// Search for similar vectors
    ///
    /// # Arguments
    /// * `query` - Query vector (must match database dimensions)
    /// * `k` - Number of results to return
    /// * `threshold` - Optional distance threshold
    ///
    /// # Returns
    /// List of (id, score, metadata) tuples sorted by similarity
    fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        threshold: Option<f32>,
    ) -> PyResult<Vec<(String, f32, HashMap<String, String>)>> {
        let search_query = SearchQuery {
            vector: query,
            k,
            filters: None,
            threshold,
            ef_search: None,
        };

        let results = self
            .db
            .search(search_query)
            .map_err(|e| PyValueError::new_err(format!("Search failed: {}", e)))?;

        // Convert results to Python tuples
        let py_results: Vec<(String, f32, HashMap<String, String>)> = results
            .into_iter()
            .map(|r| {
                let metadata: HashMap<String, String> = r
                    .metadata
                    .into_iter()
                    .filter_map(|(k, v)| {
                        if let serde_json::Value::String(s) = v {
                            Some((k, s))
                        } else {
                            None
                        }
                    })
                    .collect();

                (r.id, r.score, metadata)
            })
            .collect();

        Ok(py_results)
    }

    /// Delete a vector by ID
    fn delete(&self, id: &str) -> PyResult<bool> {
        self.db
            .delete(id)
            .map_err(|e| PyValueError::new_err(format!("Delete failed: {}", e)))
    }

    /// Get total number of vectors
    fn count(&self) -> PyResult<usize> {
        self.db
            .count()
            .map_err(|e| PyValueError::new_err(format!("Count failed: {}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "VectorDatabase(backend='ruvector-router-core', vectors={})",
            self.db.count().unwrap_or(0)
        )
    }
}
