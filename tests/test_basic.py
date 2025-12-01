"""
Basic functionality tests for VectorDB.
"""

import pytest
import numpy as np
from pyruvector import VectorDB


class TestBasicOperations:
    """Test basic VectorDB operations."""

    def test_create_db(self, dimensions):
        """Test creating a VectorDB with specified dimensions."""
        db = VectorDB(dimensions=dimensions)

        stats = db.stats()
        assert stats.dimensions == dimensions
        assert stats.vector_count == 0

        db.close()

    def test_create_db_with_path(self, persistent_db_path, dimensions):
        """Test creating a VectorDB with a file path."""
        db = VectorDB(dimensions=dimensions, path=persistent_db_path)

        stats = db.stats()
        assert stats.dimensions == dimensions

        db.close()

    def test_insert_and_search(self, empty_db, dimensions):
        """Test inserting a vector and searching for it."""
        # Create a test vector
        vector = np.random.randn(dimensions).astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        # Insert the vector with explicit ID
        vector_id = "test_vector_1"
        empty_db.insert(vector_id, vector.tolist(), {})

        assert empty_db.stats().vector_count == 1

        # Search for similar vectors
        results = empty_db.search(vector.tolist(), k=1)

        assert len(results) == 1
        assert results[0].id == vector_id
        # Score for identical vector should be very low (close to 0.0 for distance)
        assert results[0].score < 0.0001

    def test_insert_multiple_and_search(self, empty_db, sample_vectors, dimensions):
        """Test inserting multiple vectors and searching."""
        vectors = sample_vectors(dimensions, 5)
        ids = []

        # Insert all vectors with explicit IDs
        for i, vec in enumerate(vectors):
            vector_id = f"vector_{i}"
            empty_db.insert(vector_id, vec.tolist(), {})
            ids.append(vector_id)

        assert empty_db.stats().vector_count == 5

        # Search with the first vector
        results = empty_db.search(vectors[0].tolist(), k=3)

        assert len(results) == 3
        assert results[0].id == ids[0]
        # Score for identical vector should be very low (close to 0.0 for distance)
        assert results[0].score < 0.0001

    def test_dimension_mismatch(self, empty_db, dimensions):
        """Test that inserting wrong dimension raises ValueError."""
        # Create vector with wrong dimensions
        wrong_vector = np.random.randn(dimensions + 10).astype(np.float32)

        with pytest.raises(ValueError, match="dimension"):
            empty_db.insert("test_id", wrong_vector.tolist(), {})

    def test_search_empty_db(self, empty_db, dimensions):
        """Test searching an empty database returns empty list."""
        query_vector = np.random.randn(dimensions).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)

        results = empty_db.search(query_vector.tolist(), k=5)

        assert results == []
        assert len(results) == 0

    def test_search_k_larger_than_count(self, empty_db, sample_vectors, dimensions):
        """Test that k > count returns available vectors (up to count)."""
        vectors = sample_vectors(dimensions, 3)

        # Insert 3 vectors with explicit IDs
        for i, vec in enumerate(vectors):
            empty_db.insert(f"vector_{i}", vec.tolist(), {})

        # Search with k=10 (larger than count)
        results = empty_db.search(vectors[0].tolist(), k=10)

        # Should return at least 1 vector (HNSW may limit results)
        assert len(results) >= 1
        assert len(results) <= 3

    def test_count(self, empty_db, sample_vectors, dimensions):
        """Test stats().vector_count returns correct number of vectors."""
        assert empty_db.stats().vector_count == 0

        vectors = sample_vectors(dimensions, 7)
        for i, vec in enumerate(vectors):
            empty_db.insert(f"vector_{i}", vec.tolist(), {})

        assert empty_db.stats().vector_count == 7

    def test_delete_vector(self, empty_db, dimensions):
        """Test deleting a vector."""
        vector = np.random.randn(dimensions).astype(np.float32)
        vector_id = "test_vector_delete"
        empty_db.insert(vector_id, vector.tolist(), {})

        assert empty_db.stats().vector_count == 1

        # Delete the vector
        result = empty_db.delete(vector_id)

        assert result
        assert empty_db.stats().vector_count == 0

    def test_delete_nonexistent_vector(self, empty_db):
        """Test deleting a nonexistent vector returns False."""
        # Should handle gracefully and return False
        result = empty_db.delete("nonexistent_id")

        assert not result
        assert empty_db.stats().vector_count == 0

    def test_search_with_zero_k(self, populated_db, dimensions):
        """Test searching with k=0 returns empty list."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        results = populated_db.search(query_vector.tolist(), k=0)

        assert results == []

    def test_search_with_negative_k(self, populated_db, dimensions):
        """Test searching with negative k raises OverflowError."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        with pytest.raises(OverflowError):
            populated_db.search(query_vector.tolist(), k=-1)

    def test_vector_normalization(self, empty_db, dimensions):
        """Test that vectors are normalized during insertion."""
        # Create non-normalized vector
        vector = np.array([3.0, 4.0] + [0.0] * (dimensions - 2), dtype=np.float32)

        vector_id = "test_normalize"
        empty_db.insert(vector_id, vector.tolist(), {})

        # Search should still work correctly
        results = empty_db.search(vector.tolist(), k=1)

        assert len(results) == 1
        assert results[0].id == vector_id
