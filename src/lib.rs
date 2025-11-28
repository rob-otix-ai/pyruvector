#![allow(non_local_definitions)]

//! pyruvector - High-performance vector database with SIMD acceleration
//!
//! A Python binding for ruvector, providing efficient vector similarity search
//! with HNSW indexing and quantization support.

use pyo3::prelude::*;

mod advanced_filter;
mod cluster;
mod collection;
mod db;
mod error;
mod filter;
mod gnn;
mod graph;
mod metrics;
mod router;
mod snapshot;
mod types;

use advanced_filter::{FilterBuilder, FilterEvaluator, IndexType, PayloadIndexManager};
use cluster::{
    ClusterConfig, ClusterManager, ClusterNode, ClusterStats, NodeStatus,
    Replica, ReplicaRole, ReplicaSet, ShardInfo, ShardStatus, SyncMode,
};
use collection::CollectionManager;
use db::VectorDB;
use gnn::{
    BasicGNNLayer, cosine_similarity, info_nce_loss, GNNConfig, GNNModel, OptimizerType,
    PyTrainConfig, ReplayBuffer, RuvectorLayerWrapper, SchedulerType, Tensor,
    TrainingMetrics as GNNTrainingMetrics,
};
use graph::{Edge, GraphDB, Hyperedge, IsolationLevel, Node, QueryResult, Transaction};
use metrics::MetricsRecorder;
use router::{
    Candidate, NeuralRouter, RouterConfig, RoutingDecision, RoutingRequest,
    RoutingResponse, TrainingConfig as RouterTrainingConfig,
    TrainingDataset, TrainingMetrics as RouterTrainingMetrics,
    VectorDatabase,
};
use snapshot::{SnapshotInfo, SnapshotManager};
use types::{
    CollectionStats, DBStats, DbOptions, DistanceMetric, HealthStatus, HNSWConfig,
    QuantizationConfig, QuantizationType, SearchResult,
};

/// Get the pyruvector version
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Get library information
#[pyfunction]
fn info() -> String {
    format!(
        "pyruvector v{} - Python bindings for ruvector\n\
         Powered by ruvector-core with HNSW indexing and SIMD acceleration\n\
         \n\
         Features:\n\
         - SIMD-accelerated distance calculations\n\
         - HNSW indexing for fast similarity search\n\
         - Quantization (4-32x memory reduction)\n\
         - Multi-collection support\n\
         - Advanced payload filtering with indices\n\
         - Prometheus metrics export\n\
         - Snapshot backup and restore\n\
         - Graph database with Cypher queries\n\
         - Graph Neural Networks (GNN)\n\
         - Distributed clustering and sharding\n\
         - Data replication\n\
         - Neural LLM routing optimization\n\
         - Batch operations\n\
         - Thread-safe operations",
        env!("CARGO_PKG_VERSION")
    )
}

/// pyruvector - High-performance vector database with SIMD acceleration
///
/// This module provides Python bindings to the ruvector Rust library,
/// offering efficient vector similarity search capabilities.
///
/// # Classes
///
/// ## Vector Database
/// * `VectorDB` - Main vector database class for storing and searching vectors
/// * `CollectionManager` - Manage multiple vector collections
/// * `SearchResult` - Result object containing search matches
/// * `DBStats` - Database statistics and metrics
/// * `CollectionStats` - Individual collection statistics
/// * `HealthStatus` - System health information
/// * `DistanceMetric` - Distance metric enumeration
/// * `QuantizationType` - Quantization type enumeration
/// * `HNSWConfig` - HNSW index configuration
/// * `QuantizationConfig` - Quantization configuration
/// * `DbOptions` - Database options
///
/// ## Advanced Filtering
/// * `PayloadIndexManager` - Manage payload indices for filtering
/// * `IndexType` - Index type enumeration
/// * `FilterBuilder` - Build complex filter queries
/// * `FilterEvaluator` - Evaluate filter expressions
///
/// ## Metrics & Snapshots
/// * `MetricsRecorder` - Record and export Prometheus metrics
/// * `SnapshotManager` - Create and restore database snapshots (wraps ruvector-snapshot)
/// * `SnapshotInfo` - Snapshot metadata information
///
/// ## Graph Database
/// * `GraphDB` - Graph database for complex relational data
/// * `Node` - Graph node representation
/// * `Edge` - Graph edge representation
/// * `Hyperedge` - Multi-node hyperedge
/// * `Transaction` - ACID transaction support
/// * `IsolationLevel` - Transaction isolation level
/// * `QueryResult` - Cypher query results
///
/// ## Graph Neural Networks (GNN)
/// * `GNNModel` - Graph Neural Network model
/// * `GNNLayer` - Individual GNN layer
/// * `RuvectorLayer` - HNSW-based GNN layer with attention and GRU (wraps ruvector-gnn)
/// * `GNNConfig` - GNN configuration
/// * `TrainConfig` - GNN training configuration
/// * `GNNTrainingMetrics` - GNN training metrics
/// * `OptimizerType` - Neural network optimizer
/// * `SchedulerType` - Learning rate scheduler
/// * `ReplayBuffer` - Experience replay buffer
/// * `Tensor` - Multi-dimensional tensor
///
/// ## Distributed Clustering
/// * `ClusterManager` - Manage distributed cluster
/// * `ClusterConfig` - Cluster configuration
/// * `ClusterNode` - Individual cluster node
/// * `ClusterStats` - Cluster statistics
/// * `NodeStatus` - Node health status
/// * `ShardInfo` - Shard information
/// * `ShardStatus` - Shard health status
/// * `ReplicaSet` - Data replica set
/// * `Replica` - Individual replica
/// * `ReplicaRole` - Replica role (primary/secondary)
/// * `SyncMode` - Replication sync mode
///
/// ## Neural LLM Router
/// * `NeuralRouter` - Neural-optimized LLM routing
/// * `RouterConfig` - Router configuration
/// * `Candidate` - LLM routing candidate
/// * `RoutingRequest` - Routing request
/// * `RoutingResponse` - Routing response
/// * `RoutingDecision` - Individual routing decision
/// * `TrainingDataset` - Router training dataset
/// * `RouterTrainingConfig` - Router training configuration
/// * `RouterTrainingMetrics` - Router training metrics
///
/// # Example
///
/// ```python
/// from pyruvector import VectorDB, CollectionManager
///
/// # Create a database for 128-dimensional vectors
/// db = VectorDB(dimension=128)
///
/// # Add vectors
/// db.add(id="vec1", vector=[0.1] * 128, metadata={"label": "example"})
///
/// # Search for similar vectors
/// results = db.search(query=[0.1] * 128, k=10)
/// for result in results:
///     print(f"ID: {result.id}, Score: {result.score}")
///
/// # Multi-collection management
/// manager = CollectionManager()
/// manager.create_collection("embeddings", dimension=384)
/// manager.add_to_collection("embeddings", "doc1", [0.1] * 384)
/// ```
#[pymodule]
fn pyruvector(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "rUv")?;
    m.add(
        "__description__",
        "High-performance vector database with SIMD acceleration",
    )?;

    // Core classes
    m.add_class::<VectorDB>()?;
    m.add_class::<CollectionManager>()?;
    m.add_class::<SearchResult>()?;
    m.add_class::<DBStats>()?;

    // Configuration types
    m.add_class::<DistanceMetric>()?;
    m.add_class::<QuantizationType>()?;
    m.add_class::<HNSWConfig>()?;
    m.add_class::<QuantizationConfig>()?;
    m.add_class::<DbOptions>()?;

    // Stats types
    m.add_class::<CollectionStats>()?;
    m.add_class::<HealthStatus>()?;

    // Advanced filtering
    m.add_class::<PayloadIndexManager>()?;
    m.add_class::<IndexType>()?;
    m.add_class::<FilterBuilder>()?;
    m.add_class::<FilterEvaluator>()?;

    // Metrics
    m.add_class::<MetricsRecorder>()?;
    m.add_function(wrap_pyfunction!(metrics::gather_metrics, m)?)?;

    // Snapshot/backup (wraps ruvector-snapshot)
    m.add_class::<SnapshotManager>()?;
    m.add_class::<SnapshotInfo>()?;

    // Graph database
    m.add_class::<GraphDB>()?;
    m.add_class::<Node>()?;
    m.add_class::<Edge>()?;
    m.add_class::<Hyperedge>()?;
    m.add_class::<Transaction>()?;
    m.add_class::<IsolationLevel>()?;
    m.add_class::<QueryResult>()?;

    // GNN (Graph Neural Networks)
    m.add_class::<GNNModel>()?;
    m.add_class::<BasicGNNLayer>()?;
    m.add_class::<RuvectorLayerWrapper>()?;
    m.add_class::<GNNConfig>()?;
    m.add_class::<PyTrainConfig>()?;
    m.add_class::<GNNTrainingMetrics>()?;
    m.add_class::<OptimizerType>()?;
    m.add_class::<SchedulerType>()?;
    m.add_class::<ReplayBuffer>()?;
    m.add_class::<Tensor>()?;

    // Cluster/Distributed
    m.add_class::<ClusterManager>()?;
    m.add_class::<ClusterConfig>()?;
    m.add_class::<ClusterNode>()?;
    m.add_class::<ClusterStats>()?;
    m.add_class::<NodeStatus>()?;
    m.add_class::<ShardInfo>()?;
    m.add_class::<ShardStatus>()?;
    m.add_class::<ReplicaSet>()?;
    m.add_class::<Replica>()?;
    m.add_class::<ReplicaRole>()?;
    m.add_class::<SyncMode>()?;

    // Router/AI Routing
    m.add_class::<NeuralRouter>()?;
    m.add_class::<RouterConfig>()?;
    m.add_class::<Candidate>()?;
    m.add_class::<RoutingRequest>()?;
    m.add_class::<RoutingResponse>()?;
    m.add_class::<RoutingDecision>()?;
    m.add_class::<VectorDatabase>()?;
    m.add_class::<TrainingDataset>()?;
    m.add_class::<RouterTrainingConfig>()?;
    m.add_class::<RouterTrainingMetrics>()?;

    // GNN utility functions
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(info_nce_loss, m)?)?;

    // Module functions
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(info, m)?)?;

    // Add module documentation
    m.add(
        "__doc__",
        "pyruvector - High-performance vector database with SIMD acceleration\n\n\
         A Python binding for ruvector, providing efficient vector similarity search\n\
         with HNSW indexing and quantization support.\n\n\
         Features:\n\
         - SIMD-accelerated distance calculations\n\
         - HNSW (Hierarchical Navigable Small World) indexing\n\
         - Quantization for memory efficiency (4-32x reduction)\n\
         - Multi-collection support with CollectionManager\n\
         - Advanced payload filtering with indices\n\
         - Prometheus metrics export\n\
         - Snapshot backup and restore\n\
         - Graph database with Cypher queries\n\
         - Graph Neural Networks (GNN)\n\
         - Distributed clustering and sharding\n\
         - Data replication\n\
         - Neural LLM routing optimization\n\
         - Batch operations\n\
         - Thread-safe operations\n\n\
         Requires Python 3.9 or higher.",
    )?;

    Ok(())
}
