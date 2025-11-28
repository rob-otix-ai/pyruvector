"""Tests for configuration types and options."""

import pytest
from pyruvector import (
    VectorDB,
    DistanceMetric,
    QuantizationType,
    HNSWConfig,
    QuantizationConfig,
    DbOptions,
    HealthStatus,
    version,
    info,
)


class TestDistanceMetric:
    """Test suite for DistanceMetric configuration."""

    def test_cosine_metric(self):
        """Test cosine distance metric creation."""
        metric = DistanceMetric.cosine()
        assert metric is not None
        assert "cosine" in str(metric).lower() or "Cosine" in repr(metric)

    def test_euclidean_metric(self):
        """Test euclidean distance metric creation."""
        metric = DistanceMetric.euclidean()
        assert metric is not None

    def test_dot_product_metric(self):
        """Test dot product distance metric creation."""
        metric = DistanceMetric.dot_product()
        assert metric is not None

    def test_manhattan_metric(self):
        """Test manhattan distance metric creation."""
        metric = DistanceMetric.manhattan()
        assert metric is not None


class TestQuantizationType:
    """Test suite for QuantizationType configuration."""

    def test_none_quantization(self):
        """Test no quantization type."""
        q = QuantizationType.none()
        assert q is not None

    def test_scalar_quantization(self):
        """Test scalar quantization type."""
        q = QuantizationType.scalar()
        assert q is not None

    def test_product_quantization(self):
        """Test product quantization type."""
        q = QuantizationType.product()
        assert q is not None

    def test_binary_quantization(self):
        """Test binary quantization type."""
        q = QuantizationType.binary()
        assert q is not None


class TestHNSWConfig:
    """Test suite for HNSW index configuration."""

    def test_default_config(self):
        """Test HNSW config with default values."""
        config = HNSWConfig()
        assert config.m == 16
        assert config.ef_construction == 200
        assert config.ef_search == 50
        assert config.max_elements is None

    def test_custom_config(self):
        """Test HNSW config with custom values."""
        config = HNSWConfig(
            m=32,
            ef_construction=400,
            ef_search=100,
            max_elements=100000
        )
        assert config.m == 32
        assert config.ef_construction == 400
        assert config.ef_search == 100
        assert config.max_elements == 100000

    def test_config_repr(self):
        """Test HNSW config string representation."""
        config = HNSWConfig()
        repr_str = repr(config)
        assert "HNSWConfig" in repr_str

    def test_partial_config(self):
        """Test HNSW config with partial custom values."""
        config = HNSWConfig(m=24, ef_search=75)
        assert config.m == 24
        assert config.ef_construction == 200  # Default
        assert config.ef_search == 75
        assert config.max_elements is None  # Default


class TestQuantizationConfig:
    """Test suite for quantization configuration."""

    def test_default_config(self):
        """Test quantization config with defaults."""
        config = QuantizationConfig()
        # Should default to no quantization
        assert config.quantization_type is not None

    def test_product_quantization_config(self):
        """Test product quantization configuration."""
        config = QuantizationConfig(
            quantization_type=QuantizationType.product(),
            subspaces=8,
            bits=8
        )
        assert config.subspaces == 8
        assert config.bits == 8

    def test_scalar_quantization_config(self):
        """Test scalar quantization configuration."""
        config = QuantizationConfig(
            quantization_type=QuantizationType.scalar(),
            bits=8
        )
        assert config.bits == 8

    def test_binary_quantization_config(self):
        """Test binary quantization configuration."""
        config = QuantizationConfig(
            quantization_type=QuantizationType.binary()
        )
        assert config.quantization_type is not None


class TestDbOptions:
    """Test suite for database options."""

    def test_create_options(self):
        """Test basic DbOptions creation."""
        options = DbOptions(
            dimensions=384,
            distance_metric=DistanceMetric.cosine(),
        )
        assert options.dimensions == 384

    def test_full_options(self):
        """Test DbOptions with all parameters."""
        options = DbOptions(
            dimensions=768,
            distance_metric=DistanceMetric.euclidean(),
            storage_path="/tmp/test.db",
            hnsw_config=HNSWConfig(m=32),
            quantization=QuantizationConfig(),
        )
        assert options.dimensions == 768
        assert options.storage_path == "/tmp/test.db"

    def test_options_with_hnsw(self):
        """Test DbOptions with custom HNSW configuration."""
        hnsw = HNSWConfig(m=48, ef_construction=500)
        options = DbOptions(
            dimensions=512,
            distance_metric=DistanceMetric.dot_product(),
            hnsw_config=hnsw
        )
        assert options.dimensions == 512
        assert options.hnsw_config.m == 48

    def test_options_with_quantization(self):
        """Test DbOptions with quantization."""
        quant = QuantizationConfig(
            quantization_type=QuantizationType.product(),
            subspaces=8,
            bits=8
        )
        options = DbOptions(
            dimensions=256,
            distance_metric=DistanceMetric.cosine(),
            quantization=quant
        )
        assert options.dimensions == 256
        assert options.quantization.subspaces == 8


class TestVectorDBWithConfig:
    """Test suite for VectorDB with various configurations."""

    def test_create_with_distance_metric(self):
        """Test VectorDB creation with distance metric."""
        db = VectorDB(dimensions=4, distance_metric=DistanceMetric.cosine())
        assert db.distance_metric is not None

    def test_create_with_hnsw_config(self):
        """Test VectorDB creation with HNSW config."""
        config = HNSWConfig(m=32, ef_search=100)
        db = VectorDB(dimensions=4, hnsw_config=config)
        assert db.hnsw_config.m == 32

    def test_create_with_multiple_metrics(self):
        """Test VectorDB with different distance metrics."""
        db_cosine = VectorDB(dimensions=4, distance_metric=DistanceMetric.cosine())
        db_euclidean = VectorDB(dimensions=4, distance_metric=DistanceMetric.euclidean())
        db_manhattan = VectorDB(dimensions=4, distance_metric=DistanceMetric.manhattan())

        assert db_cosine.distance_metric is not None
        assert db_euclidean.distance_metric is not None
        assert db_manhattan.distance_metric is not None

    def test_is_empty(self):
        """Test is_empty method."""
        db = VectorDB(dimensions=4)
        assert db.is_empty()

        db.insert("a", [1.0, 0.0, 0.0, 0.0])
        assert not db.is_empty()

    def test_contains(self):
        """Test contains method."""
        db = VectorDB(dimensions=4)
        db.insert("a", [1.0, 0.0, 0.0, 0.0])

        assert db.contains("a")
        assert not db.contains("b")

    def test_clear(self):
        """Test clear method."""
        db = VectorDB(dimensions=4)
        db.insert("a", [1.0, 0.0, 0.0, 0.0])
        db.insert("b", [0.0, 1.0, 0.0, 0.0])

        assert len(db) == 2
        db.clear()
        assert len(db) == 0
        assert db.is_empty()

    def test_health(self):
        """Test health status reporting."""
        db = VectorDB(dimensions=4)
        health = db.health()

        assert health.status == "healthy"
        assert health.vector_count == 0
        assert health.uptime_seconds >= 0

    def test_health_with_vectors(self):
        """Test health status with vectors inserted."""
        db = VectorDB(dimensions=4)
        db.insert("a", [1.0, 0.0, 0.0, 0.0])
        db.insert("b", [0.0, 1.0, 0.0, 0.0])

        health = db.health()
        assert health.status == "healthy"
        assert health.vector_count == 2

    def test_hnsw_config_persistence(self):
        """Test that HNSW config persists in database."""
        config = HNSWConfig(m=24, ef_construction=300, ef_search=80)
        db = VectorDB(dimensions=4, hnsw_config=config)

        retrieved_config = db.hnsw_config
        assert retrieved_config.m == 24
        assert retrieved_config.ef_construction == 300
        assert retrieved_config.ef_search == 80


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_version(self):
        """Test version function."""
        v = version()
        assert isinstance(v, str)
        assert len(v) > 0

        # Should be semver format (major.minor.patch)
        parts = v.split(".")
        assert len(parts) >= 2

    def test_info(self):
        """Test info function."""
        i = info()
        assert isinstance(i, str)
        assert "pyruvector" in i.lower()

    def test_version_format(self):
        """Test version string format validity."""
        v = version()
        parts = v.split(".")

        # Check that major and minor are numeric
        assert parts[0].isdigit()
        assert parts[1].isdigit()


class TestHealthStatus:
    """Test suite for HealthStatus type."""

    def test_health_status_attributes(self):
        """Test HealthStatus has required attributes."""
        db = VectorDB(dimensions=4)
        health = db.health()

        # Check all required attributes exist
        assert hasattr(health, "status")
        assert hasattr(health, "vector_count")
        assert hasattr(health, "uptime_seconds")

    def test_health_status_types(self):
        """Test HealthStatus attribute types."""
        db = VectorDB(dimensions=4)
        health = db.health()

        assert isinstance(health.status, str)
        assert isinstance(health.vector_count, int)
        assert isinstance(health.uptime_seconds, (int, float))

    def test_health_status_values(self):
        """Test HealthStatus attribute value ranges."""
        db = VectorDB(dimensions=4)
        health = db.health()

        # Vector count should be non-negative
        assert health.vector_count >= 0

        # Uptime should be non-negative
        assert health.uptime_seconds >= 0

        # Status should be a valid status string
        assert health.status in ["healthy", "degraded", "unhealthy"]


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_zero_dimensions_raises(self):
        """Test that zero dimensions raises an error."""
        with pytest.raises((ValueError, RuntimeError)):
            VectorDB(dimensions=0)

    def test_negative_dimensions_raises(self):
        """Test that negative dimensions raises an error."""
        with pytest.raises((ValueError, RuntimeError)):
            VectorDB(dimensions=-1)

    def test_hnsw_invalid_m(self):
        """Test HNSW config with invalid m value."""
        with pytest.raises((ValueError, RuntimeError)):
            HNSWConfig(m=0)

    def test_hnsw_invalid_ef_construction(self):
        """Test HNSW config with invalid ef_construction value."""
        with pytest.raises((ValueError, RuntimeError)):
            HNSWConfig(ef_construction=0)

    def test_large_dimensions(self):
        """Test VectorDB with large dimensions."""
        db = VectorDB(dimensions=2048)
        assert db.dimensions == 2048

    def test_clear_empty_db(self):
        """Test clearing an already empty database."""
        db = VectorDB(dimensions=4)
        assert db.is_empty()

        db.clear()  # Should not raise
        assert db.is_empty()

    def test_contains_on_empty_db(self):
        """Test contains on empty database."""
        db = VectorDB(dimensions=4)
        assert not db.contains("nonexistent")
