"""Tests for CollectionManager multi-tenancy support."""

import pytest
from pyruvector import CollectionManager, DistanceMetric, CollectionStats


class TestCollectionManagerBasics:
    def test_create_manager(self):
        manager = CollectionManager()
        assert manager is not None

    def test_create_manager_with_path(self, tmp_path):
        manager = CollectionManager(base_path=str(tmp_path))
        assert manager is not None

    def test_manager_repr(self):
        manager = CollectionManager()
        repr_str = repr(manager)
        assert "CollectionManager" in repr_str

    def test_manager_len_empty(self):
        manager = CollectionManager()
        assert len(manager) == 0


class TestCollectionCRUD:
    def test_create_collection(self):
        manager = CollectionManager()
        manager.create_collection("test", dimensions=128)
        assert manager.has_collection("test")

    def test_create_collection_with_metric(self):
        manager = CollectionManager()
        manager.create_collection(
            "test",
            dimensions=128,
            distance_metric=DistanceMetric.euclidean()
        )
        assert manager.has_collection("test")

    def test_list_collections(self):
        manager = CollectionManager()
        manager.create_collection("col1", dimensions=64)
        manager.create_collection("col2", dimensions=128)
        collections = manager.list_collections()
        assert "col1" in collections
        assert "col2" in collections
        assert len(collections) == 2

    def test_delete_collection(self):
        manager = CollectionManager()
        manager.create_collection("test", dimensions=128)
        assert manager.has_collection("test")
        result = manager.delete_collection("test")
        assert result is True
        assert not manager.has_collection("test")

    def test_delete_nonexistent_collection(self):
        manager = CollectionManager()
        result = manager.delete_collection("nonexistent")
        assert result is False

    def test_duplicate_collection_error(self):
        manager = CollectionManager()
        manager.create_collection("test", dimensions=128)
        with pytest.raises(ValueError):  # Should raise error for duplicate
            manager.create_collection("test", dimensions=256)


class TestCollectionAccess:
    def test_get_collection(self):
        manager = CollectionManager()
        manager.create_collection("test", dimensions=4)
        db = manager.get_collection("test")
        assert db is not None

    def test_get_nonexistent_collection(self):
        manager = CollectionManager()
        with pytest.raises(KeyError):
            manager.get_collection("nonexistent")

    def test_use_collection(self):
        manager = CollectionManager()
        manager.create_collection("test", dimensions=4)
        db = manager.get_collection("test")
        db.insert("vec1", [1.0, 0.0, 0.0, 0.0], {"label": "test"})
        results = db.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0].id == "vec1"


class TestCollectionStats:
    def test_get_stats(self):
        manager = CollectionManager()
        manager.create_collection("test", dimensions=128)
        stats = manager.get_stats("test")
        assert isinstance(stats, CollectionStats)
        assert stats.name == "test"
        assert stats.dimensions == 128
        assert stats.vector_count == 0

    def test_stats_after_insert(self):
        manager = CollectionManager()
        manager.create_collection("test", dimensions=4)
        db = manager.get_collection("test")
        db.insert("a", [1.0, 0.0, 0.0, 0.0])
        db.insert("b", [0.0, 1.0, 0.0, 0.0])
        stats = manager.get_stats("test")
        assert stats.vector_count == 2

    def test_stats_nonexistent(self):
        manager = CollectionManager()
        with pytest.raises(KeyError):
            manager.get_stats("nonexistent")


class TestAliases:
    def test_create_alias(self):
        manager = CollectionManager()
        manager.create_collection("documents", dimensions=128)
        manager.create_alias("docs", "documents")
        # Should be able to access via alias
        db = manager.get_collection("docs")
        assert db is not None

    def test_list_aliases(self):
        manager = CollectionManager()
        manager.create_collection("col1", dimensions=64)
        manager.create_collection("col2", dimensions=128)
        manager.create_alias("alias1", "col1")
        manager.create_alias("alias2", "col2")
        aliases = manager.list_aliases()
        assert len(aliases) == 2
        # Should be list of (alias, collection) tuples
        alias_dict = dict(aliases)
        assert alias_dict["alias1"] == "col1"
        assert alias_dict["alias2"] == "col2"

    def test_delete_alias(self):
        manager = CollectionManager()
        manager.create_collection("test", dimensions=128)
        manager.create_alias("t", "test")
        result = manager.delete_alias("t")
        assert result is True
        # Alias should be gone but collection remains
        assert manager.has_collection("test")
        with pytest.raises(KeyError):
            manager.get_collection("t")  # Alias no longer resolves

    def test_alias_to_nonexistent_collection(self):
        manager = CollectionManager()
        with pytest.raises(KeyError):
            manager.create_alias("alias", "nonexistent")

    def test_overwrite_alias(self):
        manager = CollectionManager()
        manager.create_collection("col1", dimensions=64)
        manager.create_collection("col2", dimensions=128)
        manager.create_alias("active", "col1")
        manager.create_alias("active", "col2")  # Should overwrite
        aliases = dict(manager.list_aliases())
        assert aliases["active"] == "col2"


class TestMultipleTenants:
    def test_isolated_collections(self):
        manager = CollectionManager()
        manager.create_collection("tenant_a", dimensions=4)
        manager.create_collection("tenant_b", dimensions=4)

        db_a = manager.get_collection("tenant_a")
        db_b = manager.get_collection("tenant_b")

        db_a.insert("doc1", [1.0, 0.0, 0.0, 0.0])
        db_b.insert("doc2", [0.0, 1.0, 0.0, 0.0])

        assert len(db_a) == 1
        assert len(db_b) == 1

        # Search in tenant_a should not find tenant_b's data
        results_a = db_a.search([1.0, 0.0, 0.0, 0.0], k=10)
        assert len(results_a) == 1
        assert results_a[0].id == "doc1"
