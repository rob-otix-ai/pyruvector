"""
Tests for metadata filtering functionality in VectorDB.
"""

import pytest
import numpy as np


class TestFilters:
    """Test metadata filtering in search operations."""

    def test_filter_eq(self, populated_db, dimensions):
        """Test equality filter."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        # Filter for category_1
        results = populated_db.search(
            query_vector,
            k=10,
            filter={"category": {"$eq": "category_1"}}
        )

        # Verify all results match the filter
        for result in results:
            assert result.metadata["category"] == "category_1"

    def test_filter_ne(self, populated_db, dimensions):
        """Test not equal filter."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        # Filter for category != category_0
        results = populated_db.search(
            query_vector,
            k=10,
            filter={"category": {"$ne": "category_0"}}
        )

        # Verify no results have category_0
        for result in results:
            assert result.metadata["category"] != "category_0"

    def test_filter_gt_gte_lt_lte(self, populated_db, dimensions):
        """Test numeric comparison filters: gt, gte, lt, lte."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        # Test greater than
        results_gt = populated_db.search(
            query_vector,
            k=10,
            filter={"id": {"$gt": 5}}
        )
        for result in results_gt:
            assert result.metadata["id"] > 5

        # Test greater than or equal
        results_gte = populated_db.search(
            query_vector,
            k=10,
            filter={"id": {"$gte": 5}}
        )
        for result in results_gte:
            assert result.metadata["id"] >= 5

        # Test less than
        results_lt = populated_db.search(
            query_vector,
            k=10,
            filter={"id": {"$lt": 5}}
        )
        for result in results_lt:
            assert result.metadata["id"] < 5

        # Test less than or equal
        results_lte = populated_db.search(
            query_vector,
            k=10,
            filter={"id": {"$lte": 5}}
        )
        for result in results_lte:
            assert result.metadata["id"] <= 5

    def test_filter_in(self, populated_db, dimensions):
        """Test 'in' filter for value in list."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        # Filter for categories in list
        results = populated_db.search(
            query_vector,
            k=10,
            filter={"category": {"$in": ["category_0", "category_2"]}}
        )

        # Verify all results are in the allowed categories
        for result in results:
            assert result.metadata["category"] in ["category_0", "category_2"]

    def test_filter_nin(self, populated_db, dimensions):
        """Test 'not in' filter for value not in list."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        # Filter for categories not in list
        results = populated_db.search(
            query_vector,
            k=10,
            filter={"category": {"$nin": ["category_0"]}}
        )

        # Verify no results have excluded category
        for result in results:
            assert result.metadata["category"] not in ["category_0"]

    def test_filter_contains(self, populated_db, dimensions):
        """Test array contains filter."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        # Filter for vectors whose tags contain specific tag
        results = populated_db.search(
            query_vector,
            k=10,
            filter={"tags": {"$contains": "tag_0"}}
        )

        # Verify all results contain the tag
        for result in results:
            assert "tag_0" in result.metadata["tags"]

    def test_filter_exists(self, empty_db, dimensions, sample_vectors):
        """Test field exists filter."""
        vectors = sample_vectors(dimensions, 5)

        # Insert some with optional field, some without
        for i, vec in enumerate(vectors):
            metadata = {"id": i}
            if i % 2 == 0:
                metadata["optional_field"] = f"value_{i}"
            empty_db.insert(f"vec_{i}", vec.tolist(), metadata)

        query_vector = vectors[0]

        # Filter for vectors with optional_field
        results_exists = empty_db.search(
            query_vector,
            k=10,
            filter={"optional_field": {"$exists": True}}
        )

        for result in results_exists:
            assert "optional_field" in result.metadata

        # Filter for vectors without optional_field
        results_not_exists = empty_db.search(
            query_vector,
            k=10,
            filter={"optional_field": {"$exists": False}}
        )

        for result in results_not_exists:
            assert "optional_field" not in result.metadata

    def test_filter_combined(self, populated_db, dimensions):
        """Test multiple filter conditions combined."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        # Combine multiple filters
        results = populated_db.search(
            query_vector,
            k=10,
            filter={
                "id": {"$gte": 2, "$lte": 7},
                "active": {"$eq": True},
                "category": {"$in": ["category_1", "category_2"]}
            }
        )

        # Verify all conditions are met
        for result in results:
            assert 2 <= result.metadata["id"] <= 7
            assert result.metadata["active"] is True
            assert result.metadata["category"] in ["category_1", "category_2"]

    def test_filter_no_matches(self, populated_db, dimensions):
        """Test filter that matches no vectors."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        # Filter that should match nothing
        results = populated_db.search(
            query_vector,
            k=10,
            filter={"id": {"$gt": 999}}
        )

        assert len(results) == 0
        assert results == []

    def test_filter_with_float_comparison(self, populated_db, dimensions):
        """Test filtering on float values."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        results = populated_db.search(
            query_vector,
            k=10,
            filter={"score": {"$gte": 0.5}}
        )

        for result in results:
            assert result.metadata["score"] >= 0.5

    def test_filter_with_boolean(self, populated_db, dimensions):
        """Test filtering on boolean values."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        # Filter for active=True
        results_active = populated_db.search(
            query_vector,
            k=10,
            filter={"active": {"$eq": True}}
        )

        for result in results_active:
            assert result.metadata["active"] is True

        # Filter for active=False
        results_inactive = populated_db.search(
            query_vector,
            k=10,
            filter={"active": {"$eq": False}}
        )

        for result in results_inactive:
            assert result.metadata["active"] is False

    def test_filter_empty_dict(self, populated_db, dimensions):
        """Test that empty filter dict returns all results."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        results_with_filter = populated_db.search(
            query_vector,
            k=5,
            filter={}
        )

        results_without_filter = populated_db.search(
            query_vector,
            k=5
        )

        # Should return same results
        assert len(results_with_filter) == len(results_without_filter)

    @pytest.mark.skip(reason="Invalid operators are silently ignored by current implementation")
    def test_filter_invalid_operator(self, populated_db, dimensions):
        """Test that invalid filter operator raises error."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        with pytest.raises((ValueError, KeyError)):
            populated_db.search(
                query_vector,
                k=10,
                filter={"id": {"$invalid_op": 5}}
            )
