"""
Tests for metadata functionality in VectorDB.
"""

import pytest
import numpy as np
from pyruvector import VectorDB


class TestMetadata:
    """Test metadata storage and retrieval."""

    def test_insert_with_metadata(self, empty_db, dimensions):
        """Test inserting a vector with metadata."""
        vector = np.random.randn(dimensions).astype(np.float32)
        metadata = {"name": "test_vector", "category": "test"}

        vector_id = empty_db.insert(vector, metadata=metadata)

        assert vector_id is not None

        # Search and verify metadata is returned
        results = empty_db.search(vector, k=1)

        assert len(results) == 1
        assert results[0].metadata == metadata
        assert results[0].metadata["name"] == "test_vector"
        assert results[0].metadata["category"] == "test"

    def test_metadata_types(self, empty_db, dimensions, sample_vectors):
        """Test various metadata types: string, int, float, bool, list, None."""
        vectors = sample_vectors(dimensions, 6)

        metadata_list = [
            {"type": "string", "value": "hello world"},
            {"type": "integer", "value": 42},
            {"type": "float", "value": 3.14159},
            {"type": "boolean", "value": True},
            {"type": "list", "value": [1, 2, 3, "four"]},
            {"type": "none", "value": None}
        ]

        # Insert vectors with different metadata types
        for vec, meta in zip(vectors, metadata_list):
            empty_db.insert(vec, metadata=meta)

        # Verify all metadata is preserved
        results = empty_db.search(vectors[0], k=6)

        assert len(results) == 6

        # Check each metadata type
        result_metadata = {r.metadata["type"]: r.metadata["value"] for r in results}

        assert result_metadata["string"] == "hello world"
        assert result_metadata["integer"] == 42
        assert result_metadata["float"] == 3.14159
        assert result_metadata["boolean"] is True
        assert result_metadata["list"] == [1, 2, 3, "four"]
        assert result_metadata["none"] is None

    def test_search_returns_metadata(self, populated_db, dimensions):
        """Test that search results include metadata."""
        query_vector = np.random.randn(dimensions).astype(np.float32)

        results = populated_db.search(query_vector, k=5)

        assert len(results) == 5

        for result in results:
            assert hasattr(result, 'metadata')
            assert isinstance(result.metadata, dict)
            assert "id" in result.metadata
            assert "category" in result.metadata
            assert "score" in result.metadata

    def test_nested_metadata(self, empty_db, dimensions):
        """Test support for nested metadata dictionaries."""
        vector = np.random.randn(dimensions).astype(np.float32)

        nested_metadata = {
            "user": {
                "name": "John Doe",
                "age": 30,
                "preferences": {
                    "theme": "dark",
                    "notifications": True
                }
            },
            "tags": ["important", "reviewed"],
            "metrics": {
                "views": 100,
                "likes": 25
            }
        }

        vector_id = empty_db.insert(vector, metadata=nested_metadata)

        # Retrieve and verify nested structure
        results = empty_db.search(vector, k=1)

        assert len(results) == 1
        assert results[0].metadata["user"]["name"] == "John Doe"
        assert results[0].metadata["user"]["preferences"]["theme"] == "dark"
        assert results[0].metadata["metrics"]["views"] == 100
        assert "important" in results[0].metadata["tags"]

    def test_empty_metadata(self, empty_db, dimensions):
        """Test inserting with empty metadata dict."""
        vector = np.random.randn(dimensions).astype(np.float32)

        vector_id = empty_db.insert(vector, metadata={})

        results = empty_db.search(vector, k=1)

        assert len(results) == 1
        assert results[0].metadata == {}

    def test_no_metadata(self, empty_db, dimensions):
        """Test inserting without metadata."""
        vector = np.random.randn(dimensions).astype(np.float32)

        vector_id = empty_db.insert(vector)

        results = empty_db.search(vector, k=1)

        assert len(results) == 1
        # Metadata should be None or empty dict
        assert results[0].metadata is None or results[0].metadata == {}

    def test_update_metadata(self, empty_db, dimensions):
        """Test updating metadata for existing vector."""
        vector = np.random.randn(dimensions).astype(np.float32)
        original_metadata = {"status": "draft", "version": 1}

        vector_id = empty_db.insert(vector, metadata=original_metadata)

        # Update metadata
        updated_metadata = {"status": "published", "version": 2}
        empty_db.update_metadata(vector_id, updated_metadata)

        # Verify update
        results = empty_db.search(vector, k=1)

        assert results[0].metadata["status"] == "published"
        assert results[0].metadata["version"] == 2

    def test_metadata_with_special_characters(self, empty_db, dimensions):
        """Test metadata with special characters and unicode."""
        vector = np.random.randn(dimensions).astype(np.float32)

        special_metadata = {
            "unicode": "こんにちは 🌍",
            "special_chars": "!@#$%^&*()",
            "escaped": "line1\nline2\ttab",
            "quotes": 'He said "hello"'
        }

        vector_id = empty_db.insert(vector, metadata=special_metadata)

        results = empty_db.search(vector, k=1)

        assert results[0].metadata["unicode"] == "こんにちは 🌍"
        assert results[0].metadata["special_chars"] == "!@#$%^&*()"
        assert "\n" in results[0].metadata["escaped"]
        assert '"hello"' in results[0].metadata["quotes"]
