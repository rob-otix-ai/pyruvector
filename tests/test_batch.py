"""
Tests for batch operations in VectorDB.
"""

import pytest
import numpy as np
from pyruvector import VectorDB


class TestBatchOperations:
    """Test batch insert, delete, and update operations."""

    def test_batch_insert(self, empty_db, sample_vectors, sample_metadata, dimensions):
        """Test inserting multiple vectors at once."""
        vectors = sample_vectors(dimensions, 10)
        metadata = sample_metadata(10)

        # Batch insert
        vector_ids = empty_db.batch_insert(vectors, metadata=metadata)

        assert len(vector_ids) == 10
        assert empty_db.count() == 10
        assert all(vid is not None for vid in vector_ids)

        # Verify all vectors are searchable
        results = empty_db.search(vectors[0], k=10)
        assert len(results) == 10

    def test_batch_insert_without_metadata(self, empty_db, sample_vectors, dimensions):
        """Test batch insert without metadata."""
        vectors = sample_vectors(dimensions, 5)

        vector_ids = empty_db.batch_insert(vectors)

        assert len(vector_ids) == 5
        assert empty_db.count() == 5

    def test_batch_insert_mismatched_lengths(self, empty_db, sample_vectors, sample_metadata, dimensions):
        """Test that mismatched vectors/metadata lengths raises error."""
        vectors = sample_vectors(dimensions, 5)
        metadata = sample_metadata(3)  # Fewer metadata items

        with pytest.raises(ValueError, match="length"):
            empty_db.batch_insert(vectors, metadata=metadata)

    def test_batch_insert_empty_list(self, empty_db):
        """Test batch insert with empty list."""
        vector_ids = empty_db.batch_insert([])

        assert vector_ids == []
        assert empty_db.count() == 0

    def test_batch_insert_dimension_validation(self, empty_db, dimensions):
        """Test that batch insert validates all vector dimensions."""
        # Create vectors with mixed dimensions
        vectors = [
            np.random.randn(dimensions).astype(np.float32),
            np.random.randn(dimensions).astype(np.float32),
            np.random.randn(dimensions + 5).astype(np.float32),  # Wrong dimension
        ]

        with pytest.raises(ValueError, match="dimension"):
            empty_db.batch_insert(vectors)

    def test_batch_delete(self, empty_db, sample_vectors, dimensions):
        """Test deleting multiple vectors at once."""
        vectors = sample_vectors(dimensions, 10)

        # Insert vectors
        vector_ids = empty_db.batch_insert(vectors)

        assert empty_db.count() == 10

        # Delete first 5 vectors
        ids_to_delete = vector_ids[:5]
        empty_db.batch_delete(ids_to_delete)

        assert empty_db.count() == 5

        # Verify deleted vectors are gone
        results = empty_db.search(vectors[0], k=10)
        result_ids = {r.id for r in results}

        for deleted_id in ids_to_delete:
            assert deleted_id not in result_ids

    def test_batch_delete_empty_list(self, populated_db):
        """Test batch delete with empty list."""
        initial_count = populated_db.count()

        populated_db.batch_delete([])

        assert populated_db.count() == initial_count

    def test_batch_delete_nonexistent_ids(self, populated_db):
        """Test batch delete with some nonexistent IDs."""
        initial_count = populated_db.count()

        # Mix of nonexistent IDs
        ids_to_delete = ["nonexistent1", "nonexistent2", "nonexistent3"]

        # Should handle gracefully
        populated_db.batch_delete(ids_to_delete)

        # Count should remain same
        assert populated_db.count() == initial_count

    def test_large_batch_insert(self, empty_db, sample_vectors, dimensions):
        """Test inserting 1000 vectors in a batch."""
        vectors = sample_vectors(dimensions, 1000)

        vector_ids = empty_db.batch_insert(vectors)

        assert len(vector_ids) == 1000
        assert empty_db.count() == 1000

        # Verify searchability
        results = empty_db.search(vectors[0], k=10)
        assert len(results) == 10

    def test_large_batch_with_metadata(self, empty_db, sample_vectors, sample_metadata, dimensions):
        """Test large batch insert with metadata."""
        vectors = sample_vectors(dimensions, 1000)
        metadata = sample_metadata(1000)

        vector_ids = empty_db.batch_insert(vectors, metadata=metadata)

        assert len(vector_ids) == 1000

        # Verify metadata is preserved
        results = empty_db.search(vectors[0], k=5)

        for result in results:
            assert result.metadata is not None
            assert "id" in result.metadata

    def test_batch_insert_performance(self, empty_db, sample_vectors, dimensions):
        """Test that batch insert is more efficient than individual inserts."""
        import time

        vectors = sample_vectors(dimensions, 100)

        # Time individual inserts
        db1 = VectorDB(dimensions=dimensions)
        start_individual = time.time()
        for vec in vectors:
            db1.insert(vec)
        individual_time = time.time() - start_individual
        db1.close()

        # Time batch insert
        start_batch = time.time()
        empty_db.batch_insert(vectors)
        batch_time = time.time() - start_batch

        # Batch should be faster (or at least not significantly slower)
        # Allow some margin for overhead
        assert batch_time <= individual_time * 1.5

    def test_batch_update_metadata(self, empty_db, sample_vectors, sample_metadata, dimensions):
        """Test batch updating metadata for multiple vectors."""
        vectors = sample_vectors(dimensions, 5)
        metadata = sample_metadata(5)

        # Insert vectors
        vector_ids = empty_db.batch_insert(vectors, metadata=metadata)

        # Prepare updated metadata
        updated_metadata = [
            {"id": i, "status": "updated", "version": 2}
            for i in range(5)
        ]

        # Batch update
        empty_db.batch_update_metadata(vector_ids, updated_metadata)

        # Verify updates
        results = empty_db.search(vectors[0], k=5)

        for result in results:
            assert result.metadata["status"] == "updated"
            assert result.metadata["version"] == 2

    def test_batch_operations_atomicity(self, empty_db, sample_vectors, dimensions):
        """Test that batch operations are atomic (all or nothing)."""
        vectors = sample_vectors(dimensions, 3)

        # Create invalid batch with one wrong dimension
        invalid_vectors = vectors + [np.random.randn(dimensions + 10).astype(np.float32)]

        initial_count = empty_db.count()

        with pytest.raises(ValueError):
            empty_db.batch_insert(invalid_vectors)

        # Database should remain unchanged
        assert empty_db.count() == initial_count

    def test_batch_search(self, populated_db, sample_vectors, dimensions):
        """Test searching with multiple query vectors at once."""
        query_vectors = sample_vectors(dimensions, 3)

        # Batch search
        all_results = populated_db.batch_search(query_vectors, k=5)

        assert len(all_results) == 3

        for results in all_results:
            assert len(results) <= 5
            for result in results:
                assert hasattr(result, 'id')
                assert hasattr(result, 'distance')
                assert hasattr(result, 'metadata')

    def test_mixed_batch_operations(self, empty_db, sample_vectors, dimensions):
        """Test combining multiple batch operations."""
        vectors = sample_vectors(dimensions, 20)

        # Insert first batch
        ids_batch1 = empty_db.batch_insert(vectors[:10])
        assert empty_db.count() == 10

        # Insert second batch
        ids_batch2 = empty_db.batch_insert(vectors[10:])
        assert empty_db.count() == 20

        # Delete half
        empty_db.batch_delete(ids_batch1[:5] + ids_batch2[:5])
        assert empty_db.count() == 10
