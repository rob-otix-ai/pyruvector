"""Tests for snapshot backup/restore functionality."""
import pytest
import tempfile
from pyruvector import (
    VectorDB,
    SnapshotManager,
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
        ids = []
        for i in range(10):
            vec_id = f"vec_{i}"
            db.insert(vec_id, [float(i % 10) / 10.0] * 128, {"index": i})
            ids.append(vec_id)
        # Return tuple of (db, ids) for tests to access
        return db, ids

    def test_create_manager(self, temp_dir):
        manager = SnapshotManager(temp_dir)
        assert manager is not None

    def test_list_empty_snapshots(self, temp_dir):
        manager = SnapshotManager(temp_dir)
        snapshots = manager.list_snapshots()
        assert snapshots == []

    def test_create_snapshot(self, temp_dir, populated_db):
        db, ids = populated_db
        manager = SnapshotManager(temp_dir)
        info = manager.create_snapshot_with_ids(db, "backup_001", ids)
        assert info.name == "backup_001"
        assert info.vector_count == 10
        assert info.dimensions == 128

    def test_list_snapshots_after_create(self, temp_dir, populated_db):
        db, ids = populated_db
        manager = SnapshotManager(temp_dir)
        manager.create_snapshot_with_ids(db, "backup_001", ids)
        manager.create_snapshot_with_ids(db, "backup_002", ids)
        snapshots = manager.list_snapshots()
        names = [s.name for s in snapshots]
        assert "backup_001" in names
        assert "backup_002" in names

    def test_get_snapshot_info(self, temp_dir, populated_db):
        db, ids = populated_db
        manager = SnapshotManager(temp_dir)
        created_info = manager.create_snapshot_with_ids(db, "test_snap", ids)
        # Use the snapshot ID from created info
        info = manager.get_snapshot_info(created_info.id)
        assert info is not None
        assert info.name == "test_snap"

    def test_get_nonexistent_snapshot(self, temp_dir):
        manager = SnapshotManager(temp_dir)
        info = manager.get_snapshot_info("does_not_exist")
        assert info is None

    def test_delete_snapshot(self, temp_dir, populated_db):
        db, ids = populated_db
        manager = SnapshotManager(temp_dir)
        created_info = manager.create_snapshot_with_ids(db, "to_delete", ids)
        result = manager.delete_snapshot(created_info.id)
        assert result is True
        assert manager.get_snapshot_info(created_info.id) is None

    def test_restore_snapshot(self, temp_dir, populated_db):
        db, ids = populated_db
        manager = SnapshotManager(temp_dir)
        created_info = manager.create_snapshot_with_ids(db, "restore_test", ids)

        restored_db = manager.restore_snapshot(created_info.id)
        assert len(restored_db) == 10


