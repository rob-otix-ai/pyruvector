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

        assert db.dimensions == dimensions
        assert db.count() == 0

        db.close()

    def test_create_db_with_path(self, persistent_db_path, dimensions):
        """Test creating a VectorDB with a file path."""
        db = VectorDB(dimensions=dimensions, path=persistent_db_path)

        assert db.dimensions == dimensions
        assert db.path == persistent_db_path

        db.close()

    def test_insert_and_search(self, empty_db, dimensions):
        """Test inserting a vector and searching for it."""
        # Create a test vector
        vector = np.random.randn(dimensions).astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        # Insert the vector
        vector_id = empty_db.insert(vector)

        assert vector_id is not None
        assert empty_db.count() == 1

        # Search for similar vectors
        results = empty_db.search(vector, k=1)

        assert len(results) == 1
        assert results[0].id == vector_id
        # Distance to itself should be very close to 0
        assert results[0].distance < 0.0001

    def test_insert_multiple_and_search(self, empty_db, sample_vectors, dimensions):
        """Test inserting multiple vectors and searching."""
        vectors = sample_vectors(dimensions, 5)
        ids = []

        # Insert all vectors
        for vec in vectors:
            vector_id = empty_db.insert(vec)
            ids.append(vector_id)

        assert empty_db.count() == 5

        # Search with the first vector
        results = empty_db.search(vectors[0], k=3)

        assert len(results) == 3
        assert results[0].id == ids[0]
        assert results[0].distance < 0.0001

    def test_dimension_mismatch(self, empty_db, dimensions):
        """Test that inserting wrong dimension raises ValueError."""
        # Create vector with wrong dimensions
        wrong_vector = np.random.randn(dimensions + 10).astype(np.float32)

        with pytest.raises(ValueError, match="dimension"):
            empty_db.insert(wrong_vector)

    def test_search_empty_db(self, empty_db, dimensions):
        """Test searching an empty database returns empty list."""
        query_vector = np.random.randn(dimensions).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)

        results = empty_db.search(query_vector, k=5)

        assert results == []
        assert len(results) == 0

    def test_search_k_larger_than_count(self, empty_db, sample_vectors, dimensions):
        """Test that k > count returns all available vectors."""
        vectors = sample_vectors(dimensions, 3)

        # Insert 3 vectors
        for vec in vectors:
            empty_db.insert(vec)

        # Search with k=10 (larger than count)
        results = empty_db.search(vectors[0], k=10)

        # Should return all 3 vectors
        assert len(results) == 3

    def test_count(self, empty_db, sample_vectors, dimensions):
        """Test count() returns correct number of vectors."""
        assert empty_db.count() == 0

        vectors = sample_vectors(dimensions, 7)
        for vec in vectors:
            empty_db.insert(vec)

        assert empty_db.count() == 7

    def test_delete_vector(self, empty_db, dimensions):
        """Test deleting a vector."""
        vector = np.random.randn(dimensions).astype(np.float32)
        vector_id = empty_db.insert(vector)

        assert empty_db.count() == 1

        # Delete the vector
        empty_db.delete(vector_id)

        assert empty_db.count() == 0

    def test_delete_nonexistent_vector(self, empty_db):
        """Test deleting a nonexistent vector doesn't raise error."""
        # Should handle gracefully
        empty_db.delete("nonexistent_id")

        assert empty_db.count() == 0

    def test_search_with_zero_k(self, populated_db, dimensions):
        """Test searching with k=0 returns empty list."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        results = populated_db.search(query_vector, k=0)

        assert results == []

    def test_search_with_negative_k(self, populated_db, dimensions):
        """Test searching with negative k raises ValueError."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        with pytest.raises(ValueError, match="k"):
            populated_db.search(query_vector, k=-1)

    def test_vector_normalization(self, empty_db, dimensions):
        """Test that vectors are normalized during insertion."""
        # Create non-normalized vector
        vector = np.array([3.0, 4.0] + [0.0] * (dimensions - 2), dtype=np.float32)

        vector_id = empty_db.insert(vector)

        # Search should still work correctly
        results = empty_db.search(vector, k=1)

        assert len(results) == 1
        assert results[0].id == vector_id
