"""Test persistence functionality with ruvector-core's automatic storage."""

import os
import tempfile
import pytest
from pyruvector import VectorDB


def test_basic_persistence():
    """Test that vectors persist across database instances."""
    # Create a temporary file for the database
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        db_path = tmp.name

    try:
        # Create database and insert vectors
        db = VectorDB(dimensions=3, path=db_path)

        # Insert some test vectors
        db.insert("vec1", [1.0, 0.0, 0.0], {"label": "x-axis"})
        db.insert("vec2", [0.0, 1.0, 0.0], {"label": "y-axis"})
        db.insert("vec3", [0.0, 0.0, 1.0], {"label": "z-axis"})

        # Verify vectors exist
        assert len(db) == 3
        assert db.contains("vec1")

        # Close the database
        db.close()
        del db

        # Load the database again
        db2 = VectorDB.load(db_path, dimensions=3)

        # Verify all vectors are still there
        assert len(db2) == 3
        assert db2.contains("vec1")
        assert db2.contains("vec2")
        assert db2.contains("vec3")

        # Verify we can retrieve the vectors
        result = db2.get("vec1")
        assert result is not None
        assert result.metadata["label"] == "x-axis"

        # Verify search still works
        results = db2.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0].id == "vec1"

    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_save_method_compatibility():
    """Test that save() method works for API compatibility."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        db_path = tmp.name

    try:
        # Create database with path
        db = VectorDB(dimensions=2, path=db_path)
        db.insert("test", [1.0, 2.0])

        # save() should succeed (even though it's a no-op)
        db.save()

        # Verify data persisted automatically
        db.close()
        del db

        db2 = VectorDB.load(db_path, dimensions=2)
        assert db2.contains("test")

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
