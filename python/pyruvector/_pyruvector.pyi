"""
Type stubs for pyruvector native bindings.

This file provides complete type information for IDE support and type checking.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union, Tuple

# ============================================================================
# Filter Types
# ============================================================================

class FilterOperator(TypedDict, total=False):
    """Filter operator for metadata queries.

    Attributes:
        eq: Exact match (equals)
        ne: Not equals
        gt: Greater than
        gte: Greater than or equal
        lt: Less than
        lte: Less than or equal
        in_: Value in list
        nin: Value not in list
        exists: Field exists
        regex: Regular expression match
    """
    eq: Any
    ne: Any
    gt: Union[int, float]
    gte: Union[int, float]
    lt: Union[int, float]
    lte: Union[int, float]
    in_: List[Any]
    nin: List[Any]
    exists: bool
    regex: str

# ============================================================================
# Distance Metrics
# ============================================================================

class DistanceMetric:
    """Distance metric for vector similarity computation.

    Available metrics:
    - Cosine: Cosine similarity (normalized dot product)
    - Euclidean: L2 distance
    - DotProduct: Dot product similarity
    - Manhattan: L1 distance
    """
    Cosine: DistanceMetric
    Euclidean: DistanceMetric
    DotProduct: DistanceMetric
    Manhattan: DistanceMetric

    @classmethod
    def cosine(cls) -> DistanceMetric:
        """Create cosine distance metric."""
        ...

    @classmethod
    def euclidean(cls) -> DistanceMetric:
        """Create Euclidean (L2) distance metric."""
        ...

    @classmethod
    def dot_product(cls) -> DistanceMetric:
        """Create dot product distance metric."""
        ...

    @classmethod
    def manhattan(cls) -> DistanceMetric:
        """Create Manhattan (L1) distance metric."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

# ============================================================================
# Quantization Types
# ============================================================================

class QuantizationType:
    """Quantization type for memory optimization.

    Available types:
    - None: No quantization (full precision)
    - Scalar: Scalar quantization (4-8x memory reduction)
    - Product: Product quantization (8-32x memory reduction)
    - Binary: Binary quantization (32x memory reduction)
    """
    None_: QuantizationType
    Scalar: QuantizationType
    Product: QuantizationType
    Binary: QuantizationType

    @classmethod
    def none(cls) -> QuantizationType:
        """No quantization."""
        ...

    @classmethod
    def scalar(cls) -> QuantizationType:
        """Scalar quantization (4-8x compression)."""
        ...

    @classmethod
    def product(cls) -> QuantizationType:
        """Product quantization (8-32x compression)."""
        ...

    @classmethod
    def binary(cls) -> QuantizationType:
        """Binary quantization (32x compression)."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

# ============================================================================
# Configuration Classes
# ============================================================================

class HNSWConfig:
    """HNSW (Hierarchical Navigable Small World) index configuration.

    Attributes:
        m: Maximum number of connections per layer (default: 16)
            Higher values = better recall, more memory
        ef_construction: Index construction quality (default: 200)
            Higher values = better quality index, slower indexing
        ef_search: Search quality (default: 50)
            Higher values = better recall, slower search
        max_elements: Maximum number of elements (optional)
    """
    m: int
    ef_construction: int
    ef_search: int
    max_elements: Optional[int]

    def __init__(
        self,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_elements: Optional[int] = None,
    ) -> None:
        """Initialize HNSW configuration.

        Args:
            m: Maximum connections per layer (default: 16)
            ef_construction: Construction quality (default: 200)
            ef_search: Search quality (default: 50)
            max_elements: Maximum number of elements (optional)
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class QuantizationConfig:
    """Quantization configuration for memory optimization.

    Attributes:
        quantization_type: Type of quantization to use
        subspaces: Number of subspaces for product quantization (optional)
        bits: Number of bits per code (optional)
    """
    quantization_type: QuantizationType
    subspaces: Optional[int]
    bits: Optional[int]

    def __init__(
        self,
        quantization_type: Optional[QuantizationType] = None,
        subspaces: Optional[int] = None,
        bits: Optional[int] = None,
    ) -> None:
        """Initialize quantization configuration.

        Args:
            quantization_type: Type of quantization (default: None)
            subspaces: Number of subspaces for product quantization
            bits: Number of bits per code
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class DbOptions:
    """Database configuration options.

    Attributes:
        dimensions: Vector dimensionality
        distance_metric: Distance metric for similarity
        storage_path: Optional path for persistence
        hnsw_config: HNSW index configuration
        quantization: Quantization configuration
    """
    dimensions: int
    distance_metric: DistanceMetric
    storage_path: Optional[str]
    hnsw_config: HNSWConfig
    quantization: QuantizationConfig

    def __init__(
        self,
        dimensions: int,
        distance_metric: Optional[DistanceMetric] = None,
        storage_path: Optional[str] = None,
        hnsw_config: Optional[HNSWConfig] = None,
        quantization: Optional[QuantizationConfig] = None,
    ) -> None:
        """Initialize database options.

        Args:
            dimensions: Vector dimensionality (required)
            distance_metric: Distance metric (default: Cosine)
            storage_path: Storage path for persistence
            hnsw_config: HNSW configuration (default: HNSWConfig())
            quantization: Quantization configuration (default: QuantizationConfig())
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

# ============================================================================
# Statistics and Status Classes
# ============================================================================

class SearchResult:
    """Result from a vector search operation.

    Attributes:
        id: Document identifier
        score: Similarity score (0.0 to 1.0, higher is more similar)
        metadata: Document metadata dictionary
    """
    id: str
    score: float
    metadata: Dict[str, Any]

    def __init__(self, id: str, score: float, metadata: Dict[str, Any]) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class DBStats:
    """Database statistics.

    Attributes:
        vector_count: Total number of vectors in database
        dimensions: Vector dimensionality
        estimated_memory_bytes: Approximate memory usage in bytes
        count: Alias for vector_count (backward compatibility)
        memory_usage_bytes: Alias for estimated_memory_bytes (backward compatibility)
        memory_usage_mb: Memory usage in megabytes
    """
    vector_count: int
    dimensions: int
    estimated_memory_bytes: int
    count: int  # Alias for vector_count
    memory_usage_bytes: int  # Alias for estimated_memory_bytes

    def __init__(
        self,
        vector_count: int,
        dimensions: int,
        estimated_memory_bytes: int
    ) -> None: ...

    @property
    def memory_usage_mb(self) -> float:
        """Get memory usage in megabytes."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class CollectionStats:
    """Individual collection statistics.

    Attributes:
        name: Collection name
        vector_count: Number of vectors in collection
        dimensions: Vector dimensionality
        distance_metric: Distance metric used
        memory_usage_bytes: Memory usage in bytes
        memory_usage_mb: Memory usage in megabytes
    """
    name: str
    vector_count: int
    dimensions: int
    distance_metric: DistanceMetric
    memory_usage_bytes: int

    def __init__(
        self,
        name: str,
        vector_count: int,
        dimensions: int,
        distance_metric: DistanceMetric,
        memory_usage_bytes: int,
    ) -> None: ...

    @property
    def memory_usage_mb(self) -> float:
        """Get memory usage in megabytes."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class HealthStatus:
    """System health information.

    Attributes:
        status: Health status string ("healthy", "degraded", "down")
        vector_count: Total number of vectors
        memory_usage_bytes: Memory usage in bytes
        uptime_seconds: Uptime in seconds
        memory_usage_mb: Memory usage in megabytes
        uptime_hours: Uptime in hours
    """
    status: str
    vector_count: int
    memory_usage_bytes: int
    uptime_seconds: float

    def __init__(
        self,
        status: str,
        vector_count: int,
        memory_usage_bytes: int,
        uptime_seconds: float,
    ) -> None: ...

    @property
    def memory_usage_mb(self) -> float:
        """Get memory usage in megabytes."""
        ...

    @property
    def uptime_hours(self) -> float:
        """Get uptime in hours."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

# ============================================================================
# Main Database Classes
# ============================================================================

class VectorDB:
    """High-performance vector database with HNSW indexing and automatic persistence.

    The database supports automatic persistence to disk when a path is provided.
    All changes (inserts, updates, deletes) are automatically persisted.

    Example:
        >>> # In-memory database
        >>> db = VectorDB(dimensions=384)
        >>> db.insert("doc1", [0.1] * 384, {"title": "Example"})
        >>> results = db.search([0.1] * 384, k=5)
        >>> for result in results:
        ...     print(f"{result.id}: {result.score}")
        >>>
        >>> # Persistent database
        >>> db = VectorDB(dimensions=384, path="/data/vectors.db")
        >>> db.insert("doc1", [0.1] * 384)  # Auto-persisted
        >>>
        >>> # Load existing database
        >>> db = VectorDB.load("/data/vectors.db")
    """

    def __init__(
        self,
        dimensions: int,
        path: Optional[str] = None,
        hnsw_m: Optional[int] = None,
        hnsw_ef: Optional[int] = None,
        distance_metric: Optional[DistanceMetric] = None,
        hnsw_config: Optional[HNSWConfig] = None,
        quantization_config: Optional[QuantizationConfig] = None,
    ) -> None:
        """Initialize a new vector database.

        Args:
            dimensions: Vector dimensionality (must be positive)
            path: Optional storage path for automatic persistence.
                If provided, all changes are automatically persisted to disk.
                The database can be reopened with VectorDB.load(path).
            hnsw_m: HNSW M parameter (default: 16)
            hnsw_ef: HNSW ef parameter (default: 200)
            distance_metric: Distance metric (default: Euclidean)
            hnsw_config: Complete HNSW configuration (overrides hnsw_m/hnsw_ef)
            quantization_config: Quantization configuration

        Raises:
            ValueError: If dimensions <= 0 or parameters out of valid range
            RuntimeError: If database cannot be created
        """
        ...

    @staticmethod
    def load(path: str) -> VectorDB:
        """Load an existing database from disk.

        Args:
            path: Path to the database file

        Returns:
            VectorDB instance loaded from disk

        Raises:
            FileNotFoundError: If the database file doesn't exist
            RuntimeError: If the database cannot be opened or is corrupted

        Example:
            >>> # Load previously saved database
            >>> db = VectorDB.load("/data/vectors.db")
            >>> print(f"Loaded {len(db)} vectors")
        """
        ...

    @staticmethod
    def with_options(options: DbOptions) -> VectorDB:
        """Create a VectorDB with configuration options.

        Args:
            options: DbOptions object containing all configuration

        Returns:
            Configured VectorDB instance

        Raises:
            RuntimeError: If database creation fails
        """
        ...

    def insert(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a vector with metadata.

        Args:
            id: Unique identifier for the vector
            vector: Vector data (must match database dimensions)
            metadata: Optional metadata dictionary (default: {})

        Raises:
            ValueError: If vector dimensions don't match database
            RuntimeError: If insertion fails
        """
        ...

    def insert_batch(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """Insert multiple vectors efficiently.

        Args:
            ids: List of unique identifiers
            vectors: List of vectors
            metadatas: Optional list of metadata dictionaries

        Raises:
            ValueError: If lengths don't match or vector dimensions are wrong
            RuntimeError: If batch insertion fails
        """
        ...

    def search(
        self,
        vector: List[float],
        k: int = 10,
        *,
        filter: Optional[Dict[str, Union[Any, FilterOperator]]] = None,
    ) -> List[SearchResult]:
        """Search for k nearest neighbors.

        Args:
            vector: Query vector (must match database dimensions)
            k: Number of results to return (default: 10)
            filter: Optional metadata filter (default: None)
                Examples:
                - {"category": "news"} - exact match
                - {"score": {"gte": 0.8}} - range query
                - {"tags": {"in_": ["python", "rust"]}} - value in list

        Returns:
            List of SearchResult objects ordered by similarity (best first)

        Raises:
            ValueError: If query dimensions don't match database
            RuntimeError: If search fails
        """
        ...

    def get(self, id: str) -> Optional[SearchResult]:
        """Retrieve a vector by ID.

        Args:
            id: Document identifier

        Returns:
            SearchResult if found, None otherwise
            Note: score will be 1.0 for exact match retrieval
        """
        ...

    def delete(self, id: str) -> bool:
        """Delete a vector by ID.

        Args:
            id: Document identifier

        Returns:
            True if vector was deleted, False if not found
        """
        ...

    def delete_batch(self, ids: List[str]) -> int:
        """Delete multiple vectors by ID.

        Args:
            ids: List of document identifiers

        Returns:
            Number of vectors actually deleted
        """
        ...

    def stats(self) -> DBStats:
        """Get database statistics.

        Returns:
            DBStats object with count, dimensions, and memory usage
        """
        ...

    def get_stats(self) -> CollectionStats:
        """Get collection statistics.

        Returns:
            CollectionStats object with detailed collection information
        """
        ...

    def is_empty(self) -> bool:
        """Check if database is empty.

        Returns:
            True if database contains no vectors
        """
        ...

    @property
    def distance_metric(self) -> DistanceMetric:
        """Get the distance metric used by this database."""
        ...

    @property
    def hnsw_config(self) -> HNSWConfig:
        """Get the HNSW configuration."""
        ...

    @property
    def quantization_config(self) -> QuantizationConfig:
        """Get the quantization configuration."""
        ...

    def health(self) -> HealthStatus:
        """Get health status of the database.

        Returns:
            HealthStatus object with current database health information
        """
        ...

    def clear(self) -> int:
        """Remove all vectors from the database.

        Returns:
            Number of vectors that were removed

        Note:
            This operation cannot be undone.
        """
        ...

    def contains(self, id: str) -> bool:
        """Check if a vector exists in the database.

        Args:
            id: Vector identifier to check

        Returns:
            True if the vector exists
        """
        ...

    def save(self) -> None:
        """Manually save database to disk.

        Note: This method is provided for API compatibility. When a path is
        specified during database creation, all changes are automatically
        persisted to disk. Calling save() explicitly is not necessary for
        persistent databases.

        For in-memory databases (created without a path), this method will
        raise an error.

        Raises:
            ValueError: If no path was specified at creation
            RuntimeError: If save operation fails

        Example:
            >>> # Not needed for persistent databases (auto-saved)
            >>> db = VectorDB(dimensions=384, path="/data/vectors.db")
            >>> db.insert("doc1", [0.1] * 384)  # Already persisted
            >>>
            >>> # In-memory databases cannot be saved
            >>> db = VectorDB(dimensions=384)
            >>> db.save()  # Raises ValueError
        """
        ...

    def close(self) -> None:
        """Close the database and release resources."""
        ...

    def __len__(self) -> int:
        """Get number of vectors in database.

        Returns:
            Total vector count
        """
        ...

    def __contains__(self, id: str) -> bool:
        """Check if vector ID exists in database.

        Args:
            id: Document identifier

        Returns:
            True if vector exists, False otherwise
        """
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# Collection Manager
# ============================================================================

class CollectionManager:
    """Manages multiple named vector collections for multi-tenancy.

    Example:
        >>> manager = CollectionManager(base_path="/data/vectors")
        >>> manager.create_collection("embeddings", dimensions=384)
        >>> db = manager.get_collection("embeddings")
        >>> db.insert("doc1", [0.1] * 384)
    """

    def __init__(self, base_path: Optional[str] = None) -> None:
        """Initialize collection manager.

        Args:
            base_path: Optional base directory for storing collections
        """
        ...

    def create_collection(
        self,
        name: str,
        dimensions: int,
        distance_metric: Optional[DistanceMetric] = None,
    ) -> None:
        """Create a new collection.

        Args:
            name: Collection name (must be unique)
            dimensions: Vector dimensionality
            distance_metric: Distance metric to use (default: Cosine)

        Raises:
            ValueError: If collection name is empty or already exists
        """
        ...

    def get_collection(self, name: str) -> VectorDB:
        """Get an existing collection by name or alias.

        Args:
            name: Collection name or alias

        Returns:
            VectorDB instance for the collection

        Raises:
            KeyError: If collection not found
        """
        ...

    def list_collections(self) -> List[str]:
        """List all collection names.

        Returns:
            Sorted list of collection names
        """
        ...

    def delete_collection(self, name: str) -> bool:
        """Delete a collection and all its aliases.

        Args:
            name: Collection name to delete

        Returns:
            True if collection was deleted, False if not found
        """
        ...

    def get_stats(self, name: str) -> CollectionStats:
        """Get statistics for a collection.

        Args:
            name: Collection name or alias

        Returns:
            CollectionStats object

        Raises:
            KeyError: If collection not found
        """
        ...

    def create_alias(self, alias: str, collection: str) -> None:
        """Create an alias for a collection.

        Args:
            alias: Alias name
            collection: Target collection name

        Raises:
            ValueError: If alias or collection name is empty
            KeyError: If collection doesn't exist
        """
        ...

    def delete_alias(self, alias: str) -> bool:
        """Delete an alias.

        Args:
            alias: Alias name to delete

        Returns:
            True if alias was deleted, False if not found
        """
        ...

    def list_aliases(self) -> List[tuple[str, str]]:
        """List all aliases with their target collections.

        Returns:
            Sorted list of (alias, collection) tuples
        """
        ...

    def has_collection(self, name: str) -> bool:
        """Check if collection exists.

        Args:
            name: Collection name to check

        Returns:
            True if collection exists
        """
        ...

    def __len__(self) -> int:
        """Get number of collections.

        Returns:
            Number of collections managed
        """
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# Advanced Filtering Types
# ============================================================================

class IndexType:
    """Payload index type for optimized filtering.

    Available types:
    - Keyword: String exact match indexing
    - Integer: Integer range indexing
    - Float: Float range indexing
    - Boolean: Boolean value indexing
    - Geo: Geographic location indexing
    - Text: Full-text search indexing
    """
    Keyword: IndexType
    Integer: IndexType
    Float: IndexType
    Boolean: IndexType
    Geo: IndexType
    Text: IndexType

    @classmethod
    def keyword(cls) -> IndexType:
        """Create keyword index type."""
        ...

    @classmethod
    def integer(cls) -> IndexType:
        """Create integer index type."""
        ...

    @classmethod
    def float_(cls) -> IndexType:
        """Create float index type."""
        ...

    @classmethod
    def boolean(cls) -> IndexType:
        """Create boolean index type."""
        ...

    @classmethod
    def geo(cls) -> IndexType:
        """Create geo index type."""
        ...

    @classmethod
    def text(cls) -> IndexType:
        """Create text index type."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class PayloadIndexManager:
    """Manages payload indices for accelerated filtering.

    Provides index creation, management, and payload indexing for
    faster metadata filtering during vector searches.

    Example:
        >>> manager = PayloadIndexManager()
        >>> manager.create_index("category", IndexType.keyword())
        >>> manager.create_index("price", IndexType.float_())
        >>> manager.index_payload("doc1", {"category": "books", "price": 29.99})
    """

    def __init__(self) -> None:
        """Initialize payload index manager."""
        ...

    def create_index(self, field_name: str, index_type: IndexType) -> None:
        """Create an index for a payload field.

        Args:
            field_name: Field name to index
            index_type: Type of index to create

        Raises:
            ValueError: If field name is empty or index already exists
        """
        ...

    def drop_index(self, field_name: str) -> None:
        """Drop an existing index.

        Args:
            field_name: Field name to drop index for

        Raises:
            KeyError: If index doesn't exist
        """
        ...

    def list_indices(self) -> List[str]:
        """List all indexed field names.

        Returns:
            Sorted list of indexed field names
        """
        ...

    def index_payload(self, vector_id: str, payload: Dict[str, Any]) -> None:
        """Index a payload for a vector.

        Args:
            vector_id: Vector identifier
            payload: Payload dictionary to index
        """
        ...

    def remove_payload(self, vector_id: str) -> None:
        """Remove payload index for a vector.

        Args:
            vector_id: Vector identifier
        """
        ...

    def __repr__(self) -> str: ...

class FilterBuilder:
    """Fluent builder for complex filter expressions.

    Provides a chainable API for building sophisticated filter queries
    with support for comparison operators, logical operators, and
    specialized queries (geo, text, etc.).

    Example:
        >>> filter = (FilterBuilder()
        ...     .eq("category", "electronics")
        ...     .gte("price", 100)
        ...     .lte("price", 500)
        ...     .build())
        >>> results = db.search(vector, k=10, filter=filter)
    """

    def __init__(self) -> None:
        """Initialize filter builder."""
        ...

    def eq(self, field: str, value: Any) -> FilterBuilder:
        """Add equality filter.

        Args:
            field: Field name
            value: Value to match

        Returns:
            Self for chaining
        """
        ...

    def ne(self, field: str, value: Any) -> FilterBuilder:
        """Add not-equals filter.

        Args:
            field: Field name
            value: Value to exclude

        Returns:
            Self for chaining
        """
        ...

    def gt(self, field: str, value: Any) -> FilterBuilder:
        """Add greater-than filter.

        Args:
            field: Field name
            value: Minimum value (exclusive)

        Returns:
            Self for chaining
        """
        ...

    def gte(self, field: str, value: Any) -> FilterBuilder:
        """Add greater-than-or-equal filter.

        Args:
            field: Field name
            value: Minimum value (inclusive)

        Returns:
            Self for chaining
        """
        ...

    def lt(self, field: str, value: Any) -> FilterBuilder:
        """Add less-than filter.

        Args:
            field: Field name
            value: Maximum value (exclusive)

        Returns:
            Self for chaining
        """
        ...

    def lte(self, field: str, value: Any) -> FilterBuilder:
        """Add less-than-or-equal filter.

        Args:
            field: Field name
            value: Maximum value (inclusive)

        Returns:
            Self for chaining
        """
        ...

    def in_values(self, field: str, values: List[Any]) -> FilterBuilder:
        """Add value-in-list filter.

        Args:
            field: Field name
            values: List of acceptable values

        Returns:
            Self for chaining
        """
        ...

    def contains(self, field: str, value: str) -> FilterBuilder:
        """Add string contains filter.

        Args:
            field: Field name
            value: Substring to search for

        Returns:
            Self for chaining
        """
        ...

    def geo_radius(self, field: str, lat: float, lon: float, radius_km: float) -> FilterBuilder:
        """Add geographic radius filter.

        Args:
            field: Field name (should be geo-indexed)
            lat: Latitude in degrees
            lon: Longitude in degrees
            radius_km: Radius in kilometers

        Returns:
            Self for chaining
        """
        ...

    def text_match(self, field: str, query: str) -> FilterBuilder:
        """Add full-text search filter.

        Args:
            field: Field name (should be text-indexed)
            query: Search query

        Returns:
            Self for chaining
        """
        ...

    def and_(self, filters: List[FilterBuilder]) -> FilterBuilder:
        """Combine filters with logical AND.

        Args:
            filters: List of filter builders

        Returns:
            Self for chaining
        """
        ...

    def or_(self, filters: List[FilterBuilder]) -> FilterBuilder:
        """Combine filters with logical OR.

        Args:
            filters: List of filter builders

        Returns:
            Self for chaining
        """
        ...

    def not_(self, filter: FilterBuilder) -> FilterBuilder:
        """Negate a filter with logical NOT.

        Args:
            filter: Filter builder to negate

        Returns:
            Self for chaining
        """
        ...

    def build(self) -> Dict[str, Any]:
        """Build the final filter dictionary.

        Returns:
            Filter dictionary ready for use in search queries
        """
        ...

    def __repr__(self) -> str: ...

class FilterEvaluator:
    """Evaluates filters against indexed payloads.

    Efficiently evaluates complex filter expressions using indexed
    payload data for fast pre-filtering during search operations.

    Example:
        >>> manager = PayloadIndexManager()
        >>> evaluator = FilterEvaluator(manager)
        >>> filter = FilterBuilder().gte("score", 0.8).build()
        >>> matching_ids = evaluator.evaluate(filter)
    """

    def __init__(self, index_manager: PayloadIndexManager) -> None:
        """Initialize filter evaluator.

        Args:
            index_manager: Payload index manager to use
        """
        ...

    def evaluate(self, filter: Dict[str, Any]) -> List[str]:
        """Evaluate filter and return matching vector IDs.

        Args:
            filter: Filter dictionary (from FilterBuilder.build())

        Returns:
            List of vector IDs matching the filter
        """
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# Metrics Types
# ============================================================================

class MetricsRecorder:
    """Records Prometheus-compatible metrics.

    Tracks database operations, performance, and resource usage
    in a format compatible with Prometheus monitoring.

    Example:
        >>> recorder = MetricsRecorder()
        >>> recorder.record_search("embeddings", 0.025, True)
        >>> recorder.record_insert("embeddings", 100, 0.450)
        >>> metrics = gather_metrics()
    """

    def __init__(self) -> None:
        """Initialize metrics recorder."""
        ...

    def record_search(self, collection: str, duration_seconds: float, success: bool) -> None:
        """Record a search operation.

        Args:
            collection: Collection name
            duration_seconds: Search duration in seconds
            success: Whether search was successful
        """
        ...

    def record_insert(self, collection: str, count: int, duration_seconds: float) -> None:
        """Record an insert operation.

        Args:
            collection: Collection name
            count: Number of vectors inserted
            duration_seconds: Insert duration in seconds
        """
        ...

    def record_delete(self, collection: str, count: int) -> None:
        """Record a delete operation.

        Args:
            collection: Collection name
            count: Number of vectors deleted
        """
        ...

    def update_vector_count(self, collection: str, count: int) -> None:
        """Update vector count gauge.

        Args:
            collection: Collection name
            count: Current vector count
        """
        ...

    def update_memory_usage(self, bytes: int) -> None:
        """Update memory usage gauge.

        Args:
            bytes: Current memory usage in bytes
        """
        ...

    def __repr__(self) -> str: ...

def gather_metrics() -> str:
    """Returns all metrics in Prometheus text format.

    Returns:
        Prometheus-formatted metrics text

    Example:
        >>> metrics = gather_metrics()
        >>> print(metrics)
        # HELP pyruvector_search_duration_seconds Search duration
        # TYPE pyruvector_search_duration_seconds histogram
        pyruvector_search_duration_seconds_bucket{collection="embeddings",le="0.01"} 45
        ...
    """
    ...

# ============================================================================
# Snapshot Types
# ============================================================================

class SnapshotCompression:
    """Compression options for snapshots.

    Available types:
    - None: No compression (fastest, largest)
    - Gzip: Gzip compression (medium speed, good compression)
    - Lz4: LZ4 compression (fast, medium compression)
    """
    None_: SnapshotCompression
    Gzip: SnapshotCompression
    Lz4: SnapshotCompression

    @classmethod
    def none(cls) -> SnapshotCompression:
        """No compression."""
        ...

    @classmethod
    def gzip(cls) -> SnapshotCompression:
        """Gzip compression."""
        ...

    @classmethod
    def lz4(cls) -> SnapshotCompression:
        """LZ4 compression."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class SnapshotInfo:
    """Snapshot metadata.

    Attributes:
        name: Snapshot name
        created_at: Creation timestamp (ISO 8601)
        vector_count: Number of vectors in snapshot
        dimensions: Vector dimensionality
        size_bytes: Snapshot file size in bytes
        checksum: Snapshot checksum (SHA-256)
    """
    name: str
    created_at: str
    vector_count: int
    dimensions: int
    size_bytes: int
    checksum: str

    def __init__(
        self,
        name: str,
        created_at: str,
        vector_count: int,
        dimensions: int,
        size_bytes: int,
        checksum: str,
    ) -> None:
        """Initialize snapshot info.

        Args:
            name: Snapshot name
            created_at: Creation timestamp
            vector_count: Vector count
            dimensions: Vector dimensions
            size_bytes: File size in bytes
            checksum: Checksum string
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class SnapshotManager:
    """Manages database snapshots for backup and restore.

    Provides snapshot creation, restoration, and management for
    database backup and disaster recovery.

    Example:
        >>> manager = SnapshotManager("/data/snapshots")
        >>> info = manager.create_snapshot(db, "backup-2025-01-28", SnapshotCompression.gzip())
        >>> print(f"Created snapshot: {info.name} ({info.size_bytes} bytes)")
        >>>
        >>> # Later restore
        >>> restored_db = manager.restore_snapshot("backup-2025-01-28")
    """

    def __init__(self, storage_path: str) -> None:
        """Initialize snapshot manager.

        Args:
            storage_path: Directory path for storing snapshots
        """
        ...

    def create_snapshot(
        self,
        db: VectorDB,
        name: str,
        compression: SnapshotCompression = ...,
    ) -> SnapshotInfo:
        """Create a snapshot of a database.

        Args:
            db: Database to snapshot
            name: Snapshot name
            compression: Compression type (default: None)

        Returns:
            SnapshotInfo with snapshot metadata

        Raises:
            ValueError: If snapshot name is invalid or already exists
            RuntimeError: If snapshot creation fails
        """
        ...

    def list_snapshots(self) -> List[SnapshotInfo]:
        """List all available snapshots.

        Returns:
            List of SnapshotInfo objects sorted by creation time
        """
        ...

    def restore_snapshot(self, name: str) -> VectorDB:
        """Restore a database from a snapshot.

        Args:
            name: Snapshot name to restore

        Returns:
            Restored VectorDB instance

        Raises:
            KeyError: If snapshot not found
            RuntimeError: If restoration fails
        """
        ...

    def delete_snapshot(self, name: str) -> bool:
        """Delete a snapshot.

        Args:
            name: Snapshot name to delete

        Returns:
            True if snapshot was deleted, False if not found
        """
        ...

    def get_snapshot_info(self, name: str) -> Optional[SnapshotInfo]:
        """Get snapshot metadata.

        Args:
            name: Snapshot name

        Returns:
            SnapshotInfo if found, None otherwise
        """
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# Graph Database Types
# ============================================================================

class IsolationLevel:
    """Transaction isolation level for graph database operations.

    Available levels:
    - ReadUncommitted: No isolation, fastest but least safe
    - ReadCommitted: Prevents dirty reads
    - RepeatableRead: Prevents non-repeatable reads
    - Serializable: Full isolation, slowest but safest
    """
    ReadUncommitted: IsolationLevel
    ReadCommitted: IsolationLevel
    RepeatableRead: IsolationLevel
    Serializable: IsolationLevel

    @classmethod
    def read_uncommitted(cls) -> IsolationLevel:
        """No isolation level."""
        ...

    @classmethod
    def read_committed(cls) -> IsolationLevel:
        """Read committed isolation level."""
        ...

    @classmethod
    def repeatable_read(cls) -> IsolationLevel:
        """Repeatable read isolation level."""
        ...

    @classmethod
    def serializable(cls) -> IsolationLevel:
        """Serializable isolation level."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Node:
    """Graph node with properties and embeddings.

    Attributes:
        id: Unique node identifier
        labels: List of node labels/types
        properties: Node properties dictionary
        embedding: Optional vector embedding
    """
    id: str
    labels: List[str]
    properties: Dict[str, Any]
    embedding: Optional[List[float]]

    def __init__(
        self,
        id: str,
        labels: List[str],
        properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Initialize a graph node.

        Args:
            id: Unique node identifier
            labels: List of node labels
            properties: Optional properties dictionary
            embedding: Optional vector embedding
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Edge:
    """Graph edge connecting two nodes.

    Attributes:
        id: Unique edge identifier
        source: Source node ID
        target: Target node ID
        relationship: Edge relationship type
        properties: Edge properties dictionary
        weight: Optional edge weight
    """
    id: str
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any]
    weight: Optional[float]

    def __init__(
        self,
        id: str,
        source: str,
        target: str,
        relationship: str,
        properties: Optional[Dict[str, Any]] = None,
        weight: Optional[float] = None,
    ) -> None:
        """Initialize a graph edge.

        Args:
            id: Unique edge identifier
            source: Source node ID
            target: Target node ID
            relationship: Relationship type
            properties: Optional properties dictionary
            weight: Optional edge weight
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Hyperedge:
    """Hyperedge connecting multiple nodes.

    Attributes:
        id: Unique hyperedge identifier
        nodes: List of node IDs
        relationship: Hyperedge relationship type
        properties: Hyperedge properties dictionary
    """
    id: str
    nodes: List[str]
    relationship: str
    properties: Dict[str, Any]

    def __init__(
        self,
        id: str,
        nodes: List[str],
        relationship: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a hyperedge.

        Args:
            id: Unique hyperedge identifier
            nodes: List of node IDs to connect
            relationship: Relationship type
            properties: Optional properties dictionary
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class QueryResult:
    """Result from a graph query.

    Attributes:
        nodes: List of matching nodes
        edges: List of matching edges
        metadata: Query metadata and statistics
    """
    nodes: List[Node]
    edges: List[Edge]
    metadata: Dict[str, Any]

    def __init__(
        self,
        nodes: List[Node],
        edges: List[Edge],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Transaction:
    """Graph database transaction for ACID operations.

    Example:
        >>> tx = graph.begin_transaction(IsolationLevel.serializable())
        >>> tx.add_node(Node("n1", ["Person"], {"name": "Alice"}))
        >>> tx.add_edge(Edge("e1", "n1", "n2", "KNOWS"))
        >>> tx.commit()
    """

    def add_node(self, node: Node) -> None:
        """Add a node to the transaction.

        Args:
            node: Node to add

        Raises:
            RuntimeError: If transaction is not active
        """
        ...

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the transaction.

        Args:
            edge: Edge to add

        Raises:
            RuntimeError: If transaction is not active
        """
        ...

    def add_hyperedge(self, hyperedge: Hyperedge) -> None:
        """Add a hyperedge to the transaction.

        Args:
            hyperedge: Hyperedge to add

        Raises:
            RuntimeError: If transaction is not active
        """
        ...

    def commit(self) -> None:
        """Commit the transaction.

        Raises:
            RuntimeError: If commit fails
        """
        ...

    def rollback(self) -> None:
        """Rollback the transaction."""
        ...

    def __repr__(self) -> str: ...

class GraphDB:
    """Graph database with vector embeddings and ACID transactions.

    Supports property graphs with nodes, edges, and hyperedges, along with
    optional vector embeddings for semantic search.

    Example:
        >>> graph = GraphDB(dimensions=384)
        >>> graph.add_node(Node("n1", ["Person"], {"name": "Alice"}, [0.1] * 384))
        >>> graph.add_edge(Edge("e1", "n1", "n2", "KNOWS"))
        >>> results = graph.query_neighbors("n1", max_depth=2)
    """

    def __init__(
        self,
        dimensions: Optional[int] = None,
        path: Optional[str] = None,
    ) -> None:
        """Initialize a graph database.

        Args:
            dimensions: Optional vector dimensionality for embeddings
            path: Optional storage path for persistence

        Raises:
            ValueError: If dimensions <= 0
            RuntimeError: If database cannot be created
        """
        ...

    @staticmethod
    def load(path: str) -> GraphDB:
        """Load an existing graph database from disk.

        Args:
            path: Path to the database file

        Returns:
            GraphDB instance loaded from disk

        Raises:
            FileNotFoundError: If the database file doesn't exist
            RuntimeError: If the database cannot be opened
        """
        ...

    def add_node(self, node: Node) -> None:
        """Add or update a node.

        Args:
            node: Node to add

        Raises:
            ValueError: If node data is invalid
            RuntimeError: If operation fails
        """
        ...

    def add_edge(self, edge: Edge) -> None:
        """Add or update an edge.

        Args:
            edge: Edge to add

        Raises:
            ValueError: If edge data is invalid
            RuntimeError: If operation fails
        """
        ...

    def add_hyperedge(self, hyperedge: Hyperedge) -> None:
        """Add or update a hyperedge.

        Args:
            hyperedge: Hyperedge to add

        Raises:
            ValueError: If hyperedge data is invalid
            RuntimeError: If operation fails
        """
        ...

    def get_node(self, id: str) -> Optional[Node]:
        """Retrieve a node by ID.

        Args:
            id: Node identifier

        Returns:
            Node if found, None otherwise
        """
        ...

    def get_edge(self, id: str) -> Optional[Edge]:
        """Retrieve an edge by ID.

        Args:
            id: Edge identifier

        Returns:
            Edge if found, None otherwise
        """
        ...

    def delete_node(self, id: str) -> bool:
        """Delete a node and its connected edges.

        Args:
            id: Node identifier

        Returns:
            True if node was deleted, False if not found
        """
        ...

    def delete_edge(self, id: str) -> bool:
        """Delete an edge.

        Args:
            id: Edge identifier

        Returns:
            True if edge was deleted, False if not found
        """
        ...

    def query_neighbors(
        self,
        node_id: str,
        max_depth: int = 1,
        relationship: Optional[str] = None,
    ) -> QueryResult:
        """Query neighbors of a node.

        Args:
            node_id: Starting node ID
            max_depth: Maximum traversal depth (default: 1)
            relationship: Optional relationship filter

        Returns:
            QueryResult with neighbors and edges

        Raises:
            KeyError: If node not found
            RuntimeError: If query fails
        """
        ...

    def query_by_properties(
        self,
        properties: Dict[str, Any],
        labels: Optional[List[str]] = None,
    ) -> QueryResult:
        """Query nodes by properties.

        Args:
            properties: Properties to match
            labels: Optional label filters

        Returns:
            QueryResult with matching nodes
        """
        ...

    def semantic_search(
        self,
        embedding: List[float],
        k: int = 10,
        labels: Optional[List[str]] = None,
    ) -> List[Node]:
        """Search nodes by semantic similarity.

        Args:
            embedding: Query embedding vector
            k: Number of results (default: 10)
            labels: Optional label filters

        Returns:
            List of similar nodes ordered by similarity

        Raises:
            ValueError: If dimensions don't match
            RuntimeError: If search fails
        """
        ...

    def begin_transaction(
        self,
        isolation_level: Optional[IsolationLevel] = None,
    ) -> Transaction:
        """Begin a new transaction.

        Args:
            isolation_level: Isolation level (default: ReadCommitted)

        Returns:
            New transaction object

        Raises:
            RuntimeError: If transaction cannot be started
        """
        ...

    def save(self) -> None:
        """Save the graph database to disk.

        Raises:
            ValueError: If no path was specified
            RuntimeError: If save fails
        """
        ...

    def close(self) -> None:
        """Close the database and release resources."""
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# GNN (Graph Neural Network) Types
# ============================================================================

class OptimizerType:
    """Optimizer type for neural network training.

    Available types:
    - Adam: Adaptive Moment Estimation
    - SGD: Stochastic Gradient Descent
    - RMSprop: Root Mean Square Propagation
    """
    Adam: OptimizerType
    SGD: OptimizerType
    RMSprop: OptimizerType

    @classmethod
    def adam(cls) -> OptimizerType:
        """Adam optimizer."""
        ...

    @classmethod
    def sgd(cls) -> OptimizerType:
        """SGD optimizer."""
        ...

    @classmethod
    def rmsprop(cls) -> OptimizerType:
        """RMSprop optimizer."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class SchedulerType:
    """Learning rate scheduler type.

    Available types:
    - StepLR: Step-based decay
    - ExponentialLR: Exponential decay
    - CosineAnnealing: Cosine annealing
    """
    StepLR: SchedulerType
    ExponentialLR: SchedulerType
    CosineAnnealing: SchedulerType

    @classmethod
    def step_lr(cls, step_size: int, gamma: float = 0.1) -> SchedulerType:
        """Step learning rate scheduler.

        Args:
            step_size: Period of learning rate decay
            gamma: Multiplicative factor of learning rate decay
        """
        ...

    @classmethod
    def exponential_lr(cls, gamma: float) -> SchedulerType:
        """Exponential learning rate scheduler.

        Args:
            gamma: Multiplicative factor of learning rate decay
        """
        ...

    @classmethod
    def cosine_annealing(cls, t_max: int) -> SchedulerType:
        """Cosine annealing scheduler.

        Args:
            t_max: Maximum number of iterations
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Tensor:
    """Multi-dimensional tensor for neural network operations.

    Attributes:
        shape: Tensor shape
        data: Tensor data as nested lists
    """
    shape: List[int]
    data: List[Any]

    def __init__(self, data: List[Any], shape: Optional[List[int]] = None) -> None:
        """Initialize a tensor.

        Args:
            data: Tensor data
            shape: Optional shape (inferred if not provided)
        """
        ...

    def reshape(self, shape: List[int]) -> Tensor:
        """Reshape the tensor.

        Args:
            shape: New shape

        Returns:
            Reshaped tensor
        """
        ...

    def to_list(self) -> List[Any]:
        """Convert tensor to nested lists.

        Returns:
            Nested list representation
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class GNNLayer:
    """Graph Neural Network layer configuration.

    Attributes:
        layer_type: Type of GNN layer (GCN, GAT, GraphSAGE, etc.)
        in_features: Input feature dimension
        out_features: Output feature dimension
        activation: Activation function
        dropout: Dropout rate
    """
    layer_type: str
    in_features: int
    out_features: int
    activation: Optional[str]
    dropout: float

    def __init__(
        self,
        layer_type: str,
        in_features: int,
        out_features: int,
        activation: Optional[str] = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialize GNN layer configuration.

        Args:
            layer_type: Layer type (gcn, gat, graphsage, gin)
            in_features: Input dimension
            out_features: Output dimension
            activation: Optional activation function (relu, tanh, sigmoid)
            dropout: Dropout rate (0.0 to 1.0)
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class GNNConfig:
    """Graph Neural Network configuration.

    Attributes:
        layers: List of GNN layers
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of GNN layers
    """
    layers: List[GNNLayer]
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        layer_type: str = "gcn",
    ) -> None:
        """Initialize GNN configuration.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of layers (default: 2)
            layer_type: Default layer type (default: gcn)
        """
        ...

    def add_layer(self, layer: GNNLayer) -> None:
        """Add a layer to the configuration.

        Args:
            layer: GNN layer to add
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class TrainConfig:
    """Training configuration for GNN models.

    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        optimizer: Optimizer type
        scheduler: Optional learning rate scheduler
        early_stopping: Early stopping patience
    """
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: OptimizerType
    scheduler: Optional[SchedulerType]
    early_stopping: Optional[int]

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        optimizer: Optional[OptimizerType] = None,
        scheduler: Optional[SchedulerType] = None,
        early_stopping: Optional[int] = None,
    ) -> None:
        """Initialize training configuration.

        Args:
            epochs: Number of epochs (default: 100)
            batch_size: Batch size (default: 32)
            learning_rate: Learning rate (default: 0.001)
            optimizer: Optimizer type (default: Adam)
            scheduler: Optional scheduler
            early_stopping: Early stopping patience
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ReplayBuffer:
    """Experience replay buffer for reinforcement learning.

    Stores transitions for training GNN models with RL.

    Example:
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> batch = buffer.sample(batch_size=32)
    """

    def __init__(self, capacity: int) -> None:
        """Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
        """
        ...

    def push(
        self,
        state: Tensor,
        action: int,
        reward: float,
        next_state: Tensor,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        ...

    def sample(self, batch_size: int) -> List[Tuple[Tensor, int, float, Tensor, bool]]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of (state, action, reward, next_state, done) tuples

        Raises:
            ValueError: If batch_size > buffer size
        """
        ...

    def __len__(self) -> int:
        """Get current buffer size."""
        ...

    def __repr__(self) -> str: ...

class GNNModel:
    """Graph Neural Network model for node classification and link prediction.

    Example:
        >>> config = GNNConfig(input_dim=128, hidden_dim=64, output_dim=10)
        >>> model = GNNModel(config)
        >>> model.train(graph, train_config)
        >>> predictions = model.predict(graph, node_ids)
    """

    def __init__(self, config: GNNConfig) -> None:
        """Initialize GNN model.

        Args:
            config: GNN configuration

        Raises:
            ValueError: If configuration is invalid
        """
        ...

    def train(
        self,
        graph: GraphDB,
        train_config: TrainConfig,
        train_nodes: Optional[List[str]] = None,
        labels: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[float]]:
        """Train the GNN model.

        Args:
            graph: Graph database with training data
            train_config: Training configuration
            train_nodes: Optional list of training node IDs
            labels: Optional node labels for supervision

        Returns:
            Training history with losses and metrics

        Raises:
            RuntimeError: If training fails
        """
        ...

    def predict(
        self,
        graph: GraphDB,
        node_ids: List[str],
    ) -> Dict[str, Tensor]:
        """Predict node embeddings or labels.

        Args:
            graph: Graph database
            node_ids: List of node IDs to predict

        Returns:
            Dictionary mapping node IDs to predictions

        Raises:
            RuntimeError: If prediction fails
        """
        ...

    def get_embeddings(self, graph: GraphDB) -> Dict[str, List[float]]:
        """Get learned node embeddings.

        Args:
            graph: Graph database

        Returns:
            Dictionary mapping node IDs to embeddings
        """
        ...

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Save path

        Raises:
            RuntimeError: If save fails
        """
        ...

    @staticmethod
    def load(path: str) -> GNNModel:
        """Load model from disk.

        Args:
            path: Model path

        Returns:
            Loaded GNN model

        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If load fails
        """
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# Cluster/Distributed Types
# ============================================================================

class NodeStatus:
    """Cluster node status.

    Available statuses:
    - Active: Node is active and available
    - Inactive: Node is inactive
    - Failed: Node has failed
    """
    Active: NodeStatus
    Inactive: NodeStatus
    Failed: NodeStatus

    @classmethod
    def active(cls) -> NodeStatus:
        """Active status."""
        ...

    @classmethod
    def inactive(cls) -> NodeStatus:
        """Inactive status."""
        ...

    @classmethod
    def failed(cls) -> NodeStatus:
        """Failed status."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ReplicaRole:
    """Replica role in a replica set.

    Available roles:
    - Primary: Primary replica (read/write)
    - Secondary: Secondary replica (read-only)
    - Arbiter: Arbiter (voting only)
    """
    Primary: ReplicaRole
    Secondary: ReplicaRole
    Arbiter: ReplicaRole

    @classmethod
    def primary(cls) -> ReplicaRole:
        """Primary role."""
        ...

    @classmethod
    def secondary(cls) -> ReplicaRole:
        """Secondary role."""
        ...

    @classmethod
    def arbiter(cls) -> ReplicaRole:
        """Arbiter role."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class SyncMode:
    """Replication synchronization mode.

    Available modes:
    - Async: Asynchronous replication
    - Sync: Synchronous replication
    - SemiSync: Semi-synchronous replication
    """
    Async: SyncMode
    Sync: SyncMode
    SemiSync: SyncMode

    @classmethod
    def async_(cls) -> SyncMode:
        """Async mode."""
        ...

    @classmethod
    def sync(cls) -> SyncMode:
        """Sync mode."""
        ...

    @classmethod
    def semi_sync(cls) -> SyncMode:
        """Semi-sync mode."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ShardStatus:
    """Shard status.

    Available statuses:
    - Healthy: Shard is healthy
    - Degraded: Shard is degraded
    - Unavailable: Shard is unavailable
    """
    Healthy: ShardStatus
    Degraded: ShardStatus
    Unavailable: ShardStatus

    @classmethod
    def healthy(cls) -> ShardStatus:
        """Healthy status."""
        ...

    @classmethod
    def degraded(cls) -> ShardStatus:
        """Degraded status."""
        ...

    @classmethod
    def unavailable(cls) -> ShardStatus:
        """Unavailable status."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Replica:
    """Replica information.

    Attributes:
        node_id: Node ID hosting the replica
        role: Replica role
        lag_ms: Replication lag in milliseconds
    """
    node_id: str
    role: ReplicaRole
    lag_ms: int

    def __init__(self, node_id: str, role: ReplicaRole, lag_ms: int = 0) -> None:
        """Initialize replica info.

        Args:
            node_id: Node ID
            role: Replica role
            lag_ms: Replication lag (default: 0)
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ReplicaSet:
    """Replica set configuration.

    Attributes:
        replicas: List of replicas
        sync_mode: Synchronization mode
    """
    replicas: List[Replica]
    sync_mode: SyncMode

    def __init__(self, replicas: List[Replica], sync_mode: SyncMode) -> None:
        """Initialize replica set.

        Args:
            replicas: List of replicas
            sync_mode: Synchronization mode
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ShardInfo:
    """Shard information.

    Attributes:
        shard_id: Shard identifier
        range_start: Shard range start (hash)
        range_end: Shard range end (hash)
        replica_set: Replica set for this shard
        status: Shard status
    """
    shard_id: str
    range_start: int
    range_end: int
    replica_set: ReplicaSet
    status: ShardStatus

    def __init__(
        self,
        shard_id: str,
        range_start: int,
        range_end: int,
        replica_set: ReplicaSet,
        status: ShardStatus,
    ) -> None:
        """Initialize shard info.

        Args:
            shard_id: Shard ID
            range_start: Range start
            range_end: Range end
            replica_set: Replica set
            status: Shard status
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ClusterNode:
    """Cluster node information.

    Attributes:
        node_id: Unique node identifier
        address: Node network address
        status: Node status
        shards: List of shard IDs hosted by this node
    """
    node_id: str
    address: str
    status: NodeStatus
    shards: List[str]

    def __init__(
        self,
        node_id: str,
        address: str,
        status: NodeStatus,
        shards: Optional[List[str]] = None,
    ) -> None:
        """Initialize cluster node.

        Args:
            node_id: Node ID
            address: Network address
            status: Node status
            shards: List of shard IDs
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ClusterStats:
    """Cluster statistics.

    Attributes:
        total_nodes: Total number of nodes
        active_nodes: Number of active nodes
        total_shards: Total number of shards
        healthy_shards: Number of healthy shards
        total_vectors: Total vectors across cluster
        total_memory_bytes: Total memory usage
    """
    total_nodes: int
    active_nodes: int
    total_shards: int
    healthy_shards: int
    total_vectors: int
    total_memory_bytes: int

    def __init__(
        self,
        total_nodes: int,
        active_nodes: int,
        total_shards: int,
        healthy_shards: int,
        total_vectors: int,
        total_memory_bytes: int,
    ) -> None: ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ClusterConfig:
    """Cluster configuration.

    Attributes:
        num_shards: Number of shards
        replication_factor: Replication factor
        sync_mode: Replication sync mode
    """
    num_shards: int
    replication_factor: int
    sync_mode: SyncMode

    def __init__(
        self,
        num_shards: int = 3,
        replication_factor: int = 2,
        sync_mode: Optional[SyncMode] = None,
    ) -> None:
        """Initialize cluster configuration.

        Args:
            num_shards: Number of shards (default: 3)
            replication_factor: Replication factor (default: 2)
            sync_mode: Sync mode (default: Async)
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ClusterManager:
    """Manages distributed vector database cluster.

    Provides sharding, replication, and distributed query coordination.

    Example:
        >>> config = ClusterConfig(num_shards=3, replication_factor=2)
        >>> cluster = ClusterManager(config)
        >>> cluster.add_node("node1", "127.0.0.1:5000")
        >>> cluster.insert("doc1", [0.1] * 384, {"title": "Example"})
        >>> results = cluster.search([0.1] * 384, k=10)
    """

    def __init__(self, config: ClusterConfig) -> None:
        """Initialize cluster manager.

        Args:
            config: Cluster configuration

        Raises:
            ValueError: If configuration is invalid
        """
        ...

    def add_node(self, node_id: str, address: str) -> None:
        """Add a node to the cluster.

        Args:
            node_id: Unique node identifier
            address: Node network address

        Raises:
            ValueError: If node already exists
            RuntimeError: If operation fails
        """
        ...

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the cluster.

        Args:
            node_id: Node identifier

        Returns:
            True if node was removed, False if not found
        """
        ...

    def get_node(self, node_id: str) -> Optional[ClusterNode]:
        """Get node information.

        Args:
            node_id: Node identifier

        Returns:
            ClusterNode if found, None otherwise
        """
        ...

    def list_nodes(self) -> List[ClusterNode]:
        """List all nodes in the cluster.

        Returns:
            List of cluster nodes
        """
        ...

    def insert(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a vector into the cluster.

        Args:
            id: Vector identifier
            vector: Vector data
            metadata: Optional metadata

        Raises:
            RuntimeError: If insertion fails
        """
        ...

    def search(
        self,
        vector: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search across the cluster.

        Args:
            vector: Query vector
            k: Number of results (default: 10)
            filter: Optional metadata filter

        Returns:
            List of search results

        Raises:
            RuntimeError: If search fails
        """
        ...

    def get_stats(self) -> ClusterStats:
        """Get cluster statistics.

        Returns:
            ClusterStats object
        """
        ...

    def get_shard_info(self, shard_id: str) -> Optional[ShardInfo]:
        """Get shard information.

        Args:
            shard_id: Shard identifier

        Returns:
            ShardInfo if found, None otherwise
        """
        ...

    def list_shards(self) -> List[ShardInfo]:
        """List all shards.

        Returns:
            List of shard information
        """
        ...

    def rebalance(self) -> None:
        """Rebalance shards across nodes.

        Raises:
            RuntimeError: If rebalancing fails
        """
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# Router/AI Routing Types
# ============================================================================

class Candidate:
    """Routing candidate (model or endpoint).

    Attributes:
        id: Candidate identifier
        name: Candidate name
        endpoint: Endpoint URL or identifier
        capabilities: List of capabilities
        cost: Cost per request
        latency_ms: Average latency
    """
    id: str
    name: str
    endpoint: str
    capabilities: List[str]
    cost: float
    latency_ms: float

    def __init__(
        self,
        id: str,
        name: str,
        endpoint: str,
        capabilities: Optional[List[str]] = None,
        cost: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        """Initialize routing candidate.

        Args:
            id: Candidate ID
            name: Candidate name
            endpoint: Endpoint URL
            capabilities: List of capabilities
            cost: Cost per request
            latency_ms: Average latency
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class RoutingRequest:
    """Request for routing decision.

    Attributes:
        query: Query text or embedding
        context: Additional context
        constraints: Routing constraints
    """
    query: Union[str, List[float]]
    context: Dict[str, Any]
    constraints: Dict[str, Any]

    def __init__(
        self,
        query: Union[str, List[float]],
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize routing request.

        Args:
            query: Query (text or embedding)
            context: Additional context
            constraints: Routing constraints
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class RoutingResponse:
    """Response from routing decision.

    Attributes:
        candidate: Selected candidate
        confidence: Confidence score (0-1)
        reasoning: Routing reasoning/explanation
        alternatives: Alternative candidates
    """
    candidate: Candidate
    confidence: float
    reasoning: str
    alternatives: List[Tuple[Candidate, float]]

    def __init__(
        self,
        candidate: Candidate,
        confidence: float,
        reasoning: str = "",
        alternatives: Optional[List[Tuple[Candidate, float]]] = None,
    ) -> None:
        """Initialize routing response.

        Args:
            candidate: Selected candidate
            confidence: Confidence score
            reasoning: Routing reasoning
            alternatives: Alternative candidates with scores
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class RoutingMetrics:
    """Routing performance metrics.

    Attributes:
        total_requests: Total requests routed
        accuracy: Routing accuracy
        avg_latency_ms: Average latency
        cost_saved: Total cost saved
    """
    total_requests: int
    accuracy: float
    avg_latency_ms: float
    cost_saved: float

    def __init__(
        self,
        total_requests: int = 0,
        accuracy: float = 0.0,
        avg_latency_ms: float = 0.0,
        cost_saved: float = 0.0,
    ) -> None: ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class TrainingDataset:
    """Training dataset for router.

    Attributes:
        examples: List of (request, candidate_id) pairs
        metadata: Dataset metadata
    """
    examples: List[Tuple[RoutingRequest, str]]
    metadata: Dict[str, Any]

    def __init__(
        self,
        examples: Optional[List[Tuple[RoutingRequest, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize training dataset.

        Args:
            examples: Training examples
            metadata: Dataset metadata
        """
        ...

    def add_example(self, request: RoutingRequest, candidate_id: str) -> None:
        """Add a training example.

        Args:
            request: Routing request
            candidate_id: Correct candidate ID
        """
        ...

    def __len__(self) -> int:
        """Get number of examples."""
        ...

    def __repr__(self) -> str: ...

class RouterConfig:
    """Neural router configuration.

    Attributes:
        embedding_dim: Embedding dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of neural layers
        learning_rate: Learning rate
    """
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    learning_rate: float

    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 3,
        learning_rate: float = 0.001,
    ) -> None:
        """Initialize router configuration.

        Args:
            embedding_dim: Embedding dimension (default: 384)
            hidden_dim: Hidden dimension (default: 256)
            num_layers: Number of layers (default: 3)
            learning_rate: Learning rate (default: 0.001)
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class NeuralRouter:
    """Neural network-based intelligent request router.

    Routes requests to optimal models or endpoints based on learned patterns.

    Example:
        >>> config = RouterConfig(embedding_dim=384)
        >>> router = NeuralRouter(config)
        >>> router.add_candidate(Candidate("gpt4", "GPT-4", "api.openai.com"))
        >>> router.add_candidate(Candidate("claude", "Claude", "api.anthropic.com"))
        >>> response = router.route(RoutingRequest("Explain quantum physics"))
    """

    def __init__(self, config: RouterConfig) -> None:
        """Initialize neural router.

        Args:
            config: Router configuration

        Raises:
            ValueError: If configuration is invalid
        """
        ...

    def add_candidate(self, candidate: Candidate) -> None:
        """Add a routing candidate.

        Args:
            candidate: Candidate to add

        Raises:
            ValueError: If candidate already exists
        """
        ...

    def remove_candidate(self, candidate_id: str) -> bool:
        """Remove a routing candidate.

        Args:
            candidate_id: Candidate ID

        Returns:
            True if removed, False if not found
        """
        ...

    def list_candidates(self) -> List[Candidate]:
        """List all candidates.

        Returns:
            List of candidates
        """
        ...

    def route(self, request: RoutingRequest) -> RoutingResponse:
        """Route a request to the optimal candidate.

        Args:
            request: Routing request

        Returns:
            Routing response with selected candidate

        Raises:
            RuntimeError: If routing fails
        """
        ...

    def train(self, dataset: TrainingDataset, epochs: int = 100) -> Dict[str, List[float]]:
        """Train the router on a dataset.

        Args:
            dataset: Training dataset
            epochs: Number of training epochs (default: 100)

        Returns:
            Training history with losses and metrics

        Raises:
            RuntimeError: If training fails
        """
        ...

    def get_metrics(self) -> RoutingMetrics:
        """Get routing performance metrics.

        Returns:
            RoutingMetrics object
        """
        ...

    def save(self, path: str) -> None:
        """Save router model to disk.

        Args:
            path: Save path

        Raises:
            RuntimeError: If save fails
        """
        ...

    @staticmethod
    def load(path: str) -> NeuralRouter:
        """Load router model from disk.

        Args:
            path: Model path

        Returns:
            Loaded router

        Raises:
            FileNotFoundError: If model not found
            RuntimeError: If load fails
        """
        ...

    def __repr__(self) -> str: ...

# ============================================================================
# Module Functions
# ============================================================================

def version() -> str:
    """Get the pyruvector version string.

    Returns:
        Version string (e.g., "0.1.0")
    """
    ...

def info() -> str:
    """Get library information and features.

    Returns:
        Multi-line string with library information
    """
    ...

# ============================================================================
# Module Metadata
# ============================================================================

__version__: str
__author__: str
__description__: str
