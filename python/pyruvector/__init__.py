"""
pyruvector - A distributed vector database that learns.

Store embeddings, query with Cypher, scale horizontally with Raft consensus,
and let the index improve itself through Graph Neural Networks.

Python bindings for rUvector - the Rust vector database ecosystem.
Developed by Rob-otix Ai Ltd.

Core Features:
    - Vector Search: HNSW index, <1ms latency, SIMD acceleration
    - Cypher Queries: Neo4j-style graph queries (MATCH, WHERE, CREATE, RETURN)
    - GNN Layers: Neural network on index topology - search improves with usage
    - Hyperedges: Connect 3+ nodes for complex relationships
    - Metadata Filtering: Combine semantic + structured search
    - Collections: Namespace isolation for multi-tenancy

Distributed Systems:
    - Raft Consensus: Leader election, log replication, strong consistency
    - Auto-Sharding: Consistent hashing, shard migration
    - Multi-Master Replication: Write to any node, conflict resolution
    - Snapshots: Point-in-time backups with incremental support

AI & ML:
    - Tensor Compression: 2-32x memory reduction (Scalar/Product/Binary)
    - Semantic Router: Route queries to optimal endpoints (Tiny Dancer)
    - Adaptive Routing: Learn optimal strategies, minimize latency

Think of it as: Pinecone + Neo4j + PyTorch + etcd in one Python package.

Example:
    >>> from pyruvector import VectorDB, DistanceMetric
    >>> db = VectorDB(dimensions=384, distance_metric=DistanceMetric.cosine())
    >>> db.insert("doc1", [0.1] * 384, {"title": "Example"})
    >>> results = db.search([0.1] * 384, k=5)

    # Multi-tenancy with CollectionManager
    >>> from pyruvector import CollectionManager
    >>> manager = CollectionManager()
    >>> manager.create_collection("docs", dimensions=384)

    # Graph database with Cypher-style operations
    >>> from pyruvector import GraphDB
    >>> graph = GraphDB()
    >>> node_id = graph.create_node("Person", {"name": "Alice"})

Requires Python 3.9 or higher.
"""

from ._pyruvector import (
    # Core classes
    VectorDB,
    CollectionManager,
    SearchResult,
    DBStats,

    # Configuration types
    DistanceMetric,
    QuantizationType,
    HNSWConfig,
    QuantizationConfig,
    DbOptions,

    # Stats types
    CollectionStats,
    HealthStatus,

    # Advanced filtering
    PayloadIndexManager,
    IndexType,
    FilterBuilder,
    FilterEvaluator,

    # Metrics
    MetricsRecorder,
    gather_metrics,

    # Snapshot
    SnapshotManager,
    SnapshotInfo,

    # Graph database
    GraphDB,
    Node,
    Edge,
    Hyperedge,
    Transaction,
    IsolationLevel,
    QueryResult,

    # GNN (Graph Neural Networks)
    GNNModel,
    BasicGNNLayer,
    RuvectorLayer,
    GNNConfig,
    PyTrainConfig,
    OptimizerType,
    SchedulerType,
    ReplayBuffer,
    Tensor,
    TrainingMetrics as GNNTrainingMetrics,

    # Cluster/Distributed
    ClusterManager,
    ClusterConfig,
    ClusterNode,
    ClusterStats,
    NodeStatus,
    ShardInfo,
    ShardStatus,
    ReplicaSet,
    Replica,
    ReplicaRole,
    SyncMode,

    # Router/AI Routing
    NeuralRouter,
    RouterConfig,
    Candidate,
    RoutingRequest,
    RoutingResponse,
    RoutingDecision,
    VectorDatabase,
    TrainingDataset,
    TrainingConfig as RouterTrainingConfig,

    # GNN utility functions
    cosine_similarity,
    info_nce_loss,

    # Module functions
    version,
    info,
)

# Aliases for backward compatibility
GNNLayer = BasicGNNLayer
TrainConfig = PyTrainConfig
RoutingMetrics = GNNTrainingMetrics  # Router metrics

__all__ = [
    # Core
    "VectorDB",
    "CollectionManager",
    "SearchResult",
    "DBStats",

    # Configuration
    "DistanceMetric",
    "QuantizationType",
    "HNSWConfig",
    "QuantizationConfig",
    "DbOptions",

    # Stats
    "CollectionStats",
    "HealthStatus",

    # Advanced filtering
    "PayloadIndexManager",
    "IndexType",
    "FilterBuilder",
    "FilterEvaluator",

    # Metrics
    "MetricsRecorder",
    "gather_metrics",

    # Snapshot
    "SnapshotManager",
    "SnapshotInfo",

    # Graph database
    "GraphDB",
    "Node",
    "Edge",
    "Hyperedge",
    "Transaction",
    "IsolationLevel",
    "QueryResult",

    # GNN
    "GNNModel",
    "GNNLayer",
    "BasicGNNLayer",
    "RuvectorLayer",
    "GNNConfig",
    "TrainConfig",
    "PyTrainConfig",
    "GNNTrainingMetrics",
    "OptimizerType",
    "SchedulerType",
    "ReplayBuffer",
    "Tensor",
    "cosine_similarity",
    "info_nce_loss",

    # Cluster
    "ClusterManager",
    "ClusterConfig",
    "ClusterNode",
    "ClusterStats",
    "NodeStatus",
    "ShardInfo",
    "ShardStatus",
    "ReplicaSet",
    "Replica",
    "ReplicaRole",
    "SyncMode",

    # Router
    "NeuralRouter",
    "RouterConfig",
    "Candidate",
    "RoutingRequest",
    "RoutingResponse",
    "RoutingDecision",
    "RoutingMetrics",
    "VectorDatabase",
    "TrainingDataset",
    "RouterTrainingConfig",

    # Functions
    "version",
    "info",
]

__version__ = version()
