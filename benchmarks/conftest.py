"""
Pytest configuration and fixtures for pyruvector benchmarks.

Provides reusable fixtures for:
- Vector generation
- Database setup/teardown
- Performance measurement utilities
- Test markers for selective execution
"""

import os
import pytest
import numpy as np
from typing import List, Dict

# ============================================================================
# Import Real pyruvector - Fail Fast if Not Built
# ============================================================================

try:
    from pyruvector import VectorDB, DistanceMetric, HNSWConfig, QuantizationConfig, QuantizationType
except ImportError as e:
    raise ImportError(
        "pyruvector must be built before running benchmarks.\n"
        "Run: maturin develop --release\n"
        f"Original error: {e}"
    ) from e


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers for benchmark categorization."""
    config.addinivalue_line(
        "markers", "quick: Quick benchmarks suitable for CI (< 30s total)"
    )
    config.addinivalue_line(
        "markers", "slow: Comprehensive benchmarks (> 2 minutes)"
    )
    config.addinivalue_line(
        "markers", "insert: Vector insertion benchmarks"
    )
    config.addinivalue_line(
        "markers", "search: Search and retrieval benchmarks"
    )
    config.addinivalue_line(
        "markers", "hnsw: HNSW indexing benchmarks"
    )
    config.addinivalue_line(
        "markers", "quantization: Quantization performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "persistence: Save/load operation benchmarks"
    )
    config.addinivalue_line(
        "markers", "metadata: Metadata filtering benchmarks"
    )
    config.addinivalue_line(
        "markers", "scaling: Large-scale benchmarks (100K+ vectors)"
    )


def pytest_collection_modifyitems(config, items):
    """
    Auto-skip slow tests in quick mode.

    Set BENCHMARK_QUICK=1 to run only quick tests.
    """
    if os.environ.get("BENCHMARK_QUICK"):
        skip_slow = pytest.mark.skip(reason="Skipped in quick benchmark mode")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# ============================================================================
# Environment Configuration
# ============================================================================

@pytest.fixture(scope="session")
def benchmark_config():
    """
    Get benchmark configuration from environment variables.

    Environment Variables:
        BENCHMARK_DIMS: Default vector dimensions (default: 128)
        BENCHMARK_SMALL: Small dataset size (default: 1000)
        BENCHMARK_MEDIUM: Medium dataset size (default: 10000)
        BENCHMARK_LARGE: Large dataset size (default: 100000)
        BENCHMARK_QUICK: Run quick benchmarks only (0 or 1)

    Returns:
        Dictionary with benchmark configuration
    """
    return {
        "dimensions": int(os.environ.get("BENCHMARK_DIMS", 128)),
        "small": int(os.environ.get("BENCHMARK_SMALL", 1000)),
        "medium": int(os.environ.get("BENCHMARK_MEDIUM", 10000)),
        "large": int(os.environ.get("BENCHMARK_LARGE", 100000)),
        "quick_mode": bool(os.environ.get("BENCHMARK_QUICK", 0)),
    }


# ============================================================================
# Vector Generation Fixtures
# ============================================================================

@pytest.fixture
def vector_generator():
    """
    Factory fixture for generating test vectors.

    Returns:
        Callable that generates normalized random vectors

    Example:
        vectors = vector_generator(dimension=128, count=1000)
    """
    def _generate(dimension: int, count: int, normalize: bool = True, seed: int = 42) -> List[List[float]]:
        """
        Generate random vectors for testing.

        Args:
            dimension: Vector dimensionality
            count: Number of vectors to generate
            normalize: Whether to L2-normalize vectors (default: True)
            seed: Random seed for reproducibility (default: 42)

        Returns:
            List of vectors as lists of floats
        """
        np.random.seed(seed)
        vectors = np.random.randn(count, dimension).astype(np.float32)

        if normalize:
            # L2 normalization
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            vectors = vectors / norms

        return vectors.tolist()

    return _generate


@pytest.fixture
def metadata_generator():
    """
    Factory fixture for generating test metadata.

    Returns:
        Callable that generates metadata dictionaries

    Example:
        metadata = metadata_generator(count=1000)
    """
    def _generate(count: int, categories: int = 5, seed: int = 42) -> List[Dict]:
        """
        Generate test metadata dictionaries.

        Args:
            count: Number of metadata dicts to generate
            categories: Number of unique categories (default: 5)
            seed: Random seed for reproducibility (default: 42)

        Returns:
            List of metadata dictionaries
        """
        np.random.seed(seed)
        metadata_list = []

        for i in range(count):
            metadata_list.append({
                "id": i,
                "category": f"cat_{i % categories}",
                "score": float(np.random.random()),
                "active": bool(i % 2 == 0),
                "tags": [f"tag_{i % 10}", f"tag_{i % 3}"],
                "name": f"item_{i}",
                "priority": int(np.random.randint(1, 6)),
            })

        return metadata_list

    return _generate


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def empty_db(tmp_path) -> VectorDB:
    """
    Create an empty VectorDB with default configuration.

    Args:
        tmp_path: Pytest temporary directory fixture

    Yields:
        Empty VectorDB instance (128 dimensions, Cosine distance)
    """
    db = VectorDB(dimensions=128, distance_metric=DistanceMetric.Cosine)
    yield db
    # Cleanup handled by Python GC


@pytest.fixture
def populated_db_small(vector_generator, metadata_generator) -> VectorDB:
    """
    Create a VectorDB with 1K vectors (quick tests).

    Yields:
        VectorDB with 1,000 vectors
    """
    dimensions = 128
    count = 1000

    db = VectorDB(dimensions=dimensions, distance_metric=DistanceMetric.Cosine)
    vectors = vector_generator(dimensions, count)
    metadata = metadata_generator(count)

    # Create IDs for batch insertion
    ids = [f"vec_{i}" for i in range(count)]
    # Convert metadata list to list of dicts for insert_batch
    metadata_dicts = [dict(m) if m else None for m in metadata]

    db.insert_batch(ids, vectors, metadata_dicts)

    yield db


@pytest.fixture
def populated_db_medium(vector_generator, metadata_generator) -> VectorDB:
    """
    Create a VectorDB with 10K vectors (standard tests).

    Yields:
        VectorDB with 10,000 vectors
    """
    dimensions = 128
    count = 10000

    db = VectorDB(dimensions=dimensions, distance_metric=DistanceMetric.Cosine)
    vectors = vector_generator(dimensions, count)
    metadata = metadata_generator(count)

    # Create IDs for batch insertion
    ids = [f"vec_{i}" for i in range(count)]
    # Convert metadata list to list of dicts for insert_batch
    metadata_dicts = [dict(m) if m else None for m in metadata]

    db.insert_batch(ids, vectors, metadata_dicts)

    yield db


@pytest.fixture
def populated_db_large(vector_generator, metadata_generator) -> VectorDB:
    """
    Create a VectorDB with 100K vectors (scaling tests).

    Note: Marked as slow, only runs in full benchmark suite.

    Yields:
        VectorDB with 100,000 vectors
    """
    dimensions = 128
    count = 100000

    db = VectorDB(dimensions=dimensions, distance_metric=DistanceMetric.Cosine)
    vectors = vector_generator(dimensions, count)
    metadata = metadata_generator(count)

    # Create IDs for batch insertion
    ids = [f"vec_{i}" for i in range(count)]
    # Convert metadata list to list of dicts for insert_batch
    metadata_dicts = [dict(m) if m else None for m in metadata]

    db.insert_batch(ids, vectors, metadata_dicts)

    yield db


@pytest.fixture
def db_factory(tmp_path):
    """
    Factory fixture for creating custom VectorDB instances.

    Returns:
        Callable that creates VectorDB with specified configuration

    Example:
        db = db_factory(dimensions=256, distance_metric=DistanceMetric.Euclidean)
    """
    created_dbs = []

    def _create(
        dimensions: int = 128,
        distance_metric: DistanceMetric = DistanceMetric.Cosine,
        hnsw_config: HNSWConfig = None,
        quantization_config: QuantizationConfig = None,
    ) -> VectorDB:
        """
        Create VectorDB with custom configuration.

        Args:
            dimensions: Vector dimensionality
            distance_metric: Distance metric to use
            hnsw_config: Optional HNSW configuration
            quantization_config: Optional quantization configuration

        Returns:
            Configured VectorDB instance
        """
        db = VectorDB(
            dimensions=dimensions,
            distance_metric=distance_metric,
            hnsw_config=hnsw_config,
            quantization_config=quantization_config,
        )
        created_dbs.append(db)
        return db

    yield _create

    # Cleanup
    created_dbs.clear()


# ============================================================================
# HNSW Configuration Fixtures
# ============================================================================

@pytest.fixture
def hnsw_configs():
    """
    Provide common HNSW configurations for testing.

    Returns:
        Dictionary of named HNSW configurations
    """
    return {
        "default": HNSWConfig(m=16, ef_construction=200, ef_search=50),
        "high_recall": HNSWConfig(m=32, ef_construction=400, ef_search=200),
        "fast_build": HNSWConfig(m=8, ef_construction=100, ef_search=30),
        "balanced": HNSWConfig(m=16, ef_construction=200, ef_search=100),
    }


# ============================================================================
# Quantization Configuration Fixtures
# ============================================================================

@pytest.fixture
def quantization_configs():
    """
    Provide common quantization configurations for testing.

    Returns:
        Dictionary of named quantization configurations
    """
    return {
        "none": None,
        "scalar_8bit": QuantizationConfig(
            quantization_type=QuantizationType.Scalar,
            bits=8
        ),
        "product_8sub": QuantizationConfig(
            quantization_type=QuantizationType.Product,
            num_subvectors=8
        ),
        "binary": QuantizationConfig(
            quantization_type=QuantizationType.Binary
        ),
    }


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def temp_db_path(tmp_path):
    """
    Provide a temporary file path for database persistence tests.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path object for temporary database file
    """
    return tmp_path / "benchmark_db.pyruvector"


@pytest.fixture
def query_vectors(vector_generator):
    """
    Generate query vectors for search benchmarks.

    Returns:
        List of 100 normalized query vectors (128 dimensions)
    """
    return vector_generator(dimension=128, count=100, seed=999)


# ============================================================================
# Benchmark Helper Fixtures
# ============================================================================

@pytest.fixture
def assert_performance():
    """
    Helper fixture for asserting performance baselines.

    Returns:
        Callable that checks if operation meets performance threshold

    Example:
        assert_performance(result, max_time_ms=10.0)
    """
    def _assert(benchmark_result, max_time_ms: float, operation_name: str = "Operation"):
        """
        Assert that benchmark result is within performance threshold.

        Args:
            benchmark_result: pytest-benchmark result object
            max_time_ms: Maximum acceptable time in milliseconds
            operation_name: Name of operation for error message

        Raises:
            AssertionError: If performance threshold is exceeded
        """
        stats = benchmark_result.stats
        mean_time_ms = stats.mean * 1000  # Convert to ms

        assert mean_time_ms < max_time_ms, (
            f"{operation_name} exceeded performance threshold: "
            f"{mean_time_ms:.2f}ms > {max_time_ms:.2f}ms"
        )

    return _assert


@pytest.fixture
def skip_if_quick_mode(benchmark_config):
    """
    Skip test if running in quick benchmark mode.

    Use as decorator or in test body:
        skip_if_quick_mode()
    """
    def _skip():
        if benchmark_config["quick_mode"]:
            pytest.skip("Skipped in quick benchmark mode")

    return _skip
