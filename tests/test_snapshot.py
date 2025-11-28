"""Tests for snapshot backup/restore functionality."""
import pytest
import tempfile
import os
from pyruvector import (
    VectorDB,
    SnapshotManager,
    SnapshotInfo,
    SnapshotCompression,
)


class TestSnapshotManager:
    """Test SnapshotManager functionality."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def populated_db(self):
        db = VectorDB(dimensions=128)
        for i in range(10):
            db.insert(f"vec_{i}", [float(i % 10) / 10.0] * 128, {"index": i})
        return db

    def test_create_manager(self, temp_dir):
        manager = SnapshotManager(temp_dir)
        assert manager is not None

    def test_list_empty_snapshots(self, temp_dir):
        manager = SnapshotManager(temp_dir)
        snapshots = manager.list_snapshots()
        assert snapshots == []

    def test_create_snapshot(self, temp_dir, populated_db):
        manager = SnapshotManager(temp_dir)
        info = manager.create_snapshot(populated_db, "backup_001")
        assert info.name == "backup_001"
        assert info.vector_count == 10
        assert info.dimensions == 128

    def test_list_snapshots_after_create(self, temp_dir, populated_db):
        manager = SnapshotManager(temp_dir)
        manager.create_snapshot(populated_db, "backup_001")
        manager.create_snapshot(populated_db, "backup_002")
        snapshots = manager.list_snapshots()
        names = [s.name for s in snapshots]
        assert "backup_001" in names
        assert "backup_002" in names

    def test_get_snapshot_info(self, temp_dir, populated_db):
        manager = SnapshotManager(temp_dir)
        manager.create_snapshot(populated_db, "test_snap")
        info = manager.get_snapshot_info("test_snap")
        assert info is not None
        assert info.name == "test_snap"

    def test_get_nonexistent_snapshot(self, temp_dir):
        manager = SnapshotManager(temp_dir)
        info = manager.get_snapshot_info("does_not_exist")
        assert info is None

    def test_delete_snapshot(self, temp_dir, populated_db):
        manager = SnapshotManager(temp_dir)
        manager.create_snapshot(populated_db, "to_delete")
        result = manager.delete_snapshot("to_delete")
        assert result is True
        assert manager.get_snapshot_info("to_delete") is None

    def test_restore_snapshot(self, temp_dir, populated_db):
        manager = SnapshotManager(temp_dir)
        manager.create_snapshot(populated_db, "restore_test")

        restored_db = manager.restore_snapshot("restore_test")
        assert len(restored_db) == 10


class TestSnapshotCompression:
    """Test snapshot compression options."""

    def test_compression_none(self):
        comp = SnapshotCompression.none()
        assert comp is not None

    def test_compression_gzip(self):
        comp = SnapshotCompression.gzip()
        assert comp is not None

    def test_compression_lz4(self):
        comp = SnapshotCompression.lz4()
        assert comp is not None
