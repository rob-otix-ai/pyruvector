"""
Pytest configuration and shared fixtures for pyruvector tests.
"""

import pytest
import numpy as np
from pyruvector import VectorDB


@pytest.fixture
def dimensions():
    """Standard dimension size for test vectors."""
    return 128


@pytest.fixture
def sample_vectors():
    """Generate sample test vectors."""
    def _generate(dimensions: int, count: int = 10):
        """
        Generate normalized random vectors.

        Args:
            dimensions: Vector dimension size
            count: Number of vectors to generate

        Returns:
            List of numpy arrays
        """
        vectors = []
        for _ in range(count):
            vec = np.random.randn(dimensions).astype(np.float32)
            # Normalize
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
        return vectors
    return _generate


@pytest.fixture
def sample_metadata():
    """Generate sample metadata dictionaries."""
    def _generate(count: int = 10):
        """
        Generate test metadata.

        Args:
            count: Number of metadata dicts to generate

        Returns:
            List of metadata dictionaries
        """
        metadata_list = []
        for i in range(count):
            metadata_list.append({
                "id": i,
                "category": f"category_{i % 3}",
                "score": float(i * 0.1),
                "active": i % 2 == 0,
                "tags": [f"tag_{i}", f"tag_{i % 5}"],
                "name": f"item_{i}"
            })
        return metadata_list
    return _generate


@pytest.fixture
def empty_db(tmp_path, dimensions):
    """Create an empty VectorDB instance."""
    db = VectorDB(dimensions=dimensions)
    yield db
    db.close()


@pytest.fixture
def populated_db(tmp_path, dimensions, sample_vectors, sample_metadata):
    """Create a VectorDB with sample data."""
    db = VectorDB(dimensions=dimensions)
    vectors = sample_vectors(dimensions, 10)
    metadata = sample_metadata(10)

    for i, (vec, meta) in enumerate(zip(vectors, metadata)):
        db.insert(f"vec_{i}", vec.tolist(), meta)

    yield db
    db.close()


@pytest.fixture
def persistent_db_path(tmp_path):
    """Provide a temporary path for persistent database testing."""
    db_path = tmp_path / "test_db.pyruvector"
    return str(db_path)


@pytest.fixture
def create_db():
    """Factory fixture for creating VectorDB instances."""
    dbs = []

    def _create(dimensions: int, path: str = None):
        db = VectorDB(dimensions=dimensions)
        dbs.append(db)
        return db

    yield _create

    # Cleanup
    for db in dbs:
        try:
            db.close()
        except Exception:
            pass
