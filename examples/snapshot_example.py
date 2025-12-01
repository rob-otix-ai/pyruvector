#!/usr/bin/env python3
"""
Example demonstrating snapshot/backup functionality for pyruvector.

This example shows how to:
1. Create a vector database
2. Add vectors with metadata
3. Create snapshots for backup
4. List and manage snapshots
5. Restore from snapshots
"""

import pyruvector
import random
import time


def generate_random_vector(dimensions):
    """Generate a random normalized vector."""
    vector = [random.gauss(0, 1) for _ in range(dimensions)]
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector]


def main():
    print("=" * 60)
    print("PyRuVector Snapshot/Backup Example")
    print("=" * 60)
    print()

    # Configuration
    dimensions = 128
    num_vectors = 1000
    snapshot_dir = "/tmp/pyruvector_snapshots"

    # Step 1: Create a VectorDB and populate it
    print(f"Step 1: Creating VectorDB with {dimensions} dimensions...")
    db = pyruvector.VectorDB(dimensions=dimensions)

    print(f"Step 2: Adding {num_vectors} random vectors...")
    start_time = time.time()

    # Track IDs for snapshot creation
    vector_ids = []
    for i in range(num_vectors):
        vector = generate_random_vector(dimensions)
        metadata = {
            "index": i,
            "category": f"cat_{i % 10}",
            "timestamp": time.time(),
        }
        vec_id = f"vec_{i}"
        db.insert(id=vec_id, vector=vector, metadata=metadata)
        vector_ids.append(vec_id)

    elapsed = time.time() - start_time
    print(f"Added {num_vectors} vectors in {elapsed:.2f}s ({num_vectors / elapsed:.0f} vec/s)")
    print()

    # Step 3: Create a SnapshotManager
    print(f"Step 3: Creating SnapshotManager at '{snapshot_dir}'...")
    manager = pyruvector.SnapshotManager(snapshot_dir)
    print(f"SnapshotManager created: {manager}")
    print()

    # Step 4: Create snapshots
    print("Step 4: Creating snapshots...")

    # Snapshot 1: Full backup
    print("  Creating full backup snapshot...")
    start_time = time.time()
    info1 = manager.create_snapshot_with_ids(
        db,
        name="backup-full",
        vector_ids=vector_ids,
        description="Full backup of all vectors",
    )
    elapsed = time.time() - start_time
    print(f"  Created '{info1.name}' in {elapsed:.2f}s")
    print(f"    - Vectors: {info1.vector_count}")
    print(f"    - Size: {info1.size_mb:.2f} MB")
    print(f"    - Checksum: {info1.checksum[:16]}...")
    print()

    # Snapshot 2: Partial backup (first half)
    print("  Creating partial backup (first 500 vectors)...")
    start_time = time.time()
    info2 = manager.create_snapshot_with_ids(
        db,
        name="backup-partial",
        vector_ids=vector_ids[:500],
        description="Partial backup with first 500 vectors",
    )
    elapsed = time.time() - start_time
    print(f"  Created '{info2.name}' in {elapsed:.2f}s")
    print(f"    - Vectors: {info2.vector_count}")
    print(f"    - Size: {info2.size_mb:.2f} MB")
    print()

    # Step 5: List all snapshots
    print("Step 5: Listing all snapshots...")
    snapshots = manager.list_snapshots()
    print(f"Found {len(snapshots)} snapshot(s):")
    for snap in snapshots:
        print(f"  - {snap.name}")
        print(f"    Created: {snap.created_at}")
        print(f"    Vectors: {snap.vector_count}")
        print(f"    Size: {snap.size_mb:.2f} MB")
        if snap.description:
            print(f"    Description: {snap.description}")
        print()

    # Step 6: Get snapshot info
    print("Step 6: Getting specific snapshot info...")
    info = manager.get_snapshot_info("backup-full")
    if info:
        print("Snapshot 'backup-full' details:")
        print(f"  - Created at: {info.created_at}")
        print(f"  - Dimensions: {info.dimensions}")
        print(f"  - Vector count: {info.vector_count}")
        print(f"  - Size: {info.size_mb:.2f} MB ({info.size_gb:.4f} GB)")
        print(f"  - Checksum: {info.checksum}")
        print()

    # Step 7: Check if snapshot exists
    print("Step 7: Checking snapshot existence...")
    exists = "backup-full" in manager
    print(f"Snapshot 'backup-full' exists: {exists}")
    exists = "nonexistent-backup" in manager
    print(f"Snapshot 'nonexistent-backup' exists: {exists}")
    print()

    # Step 8: Get manager statistics
    print("Step 8: Manager statistics...")
    print(f"Total snapshots: {len(manager)}")
    print(f"Total storage used: {manager.total_size_mb:.2f} MB")
    print()

    # Step 9: Restore from snapshot
    print("Step 9: Restoring database from snapshot...")
    print("  Restoring from 'backup-full'...")
    start_time = time.time()
    restored_db = manager.restore_snapshot("backup-full")
    elapsed = time.time() - start_time
    print(f"  Database restored in {elapsed:.2f}s")
    print(f"  Restored database: {restored_db}")
    print()

    # Step 10: Verify restored database
    print("Step 10: Verifying restored database...")
    stats = restored_db.stats()
    print("Restored database stats:")
    print(f"  - Vectors: {stats.vector_count}")
    print(f"  - Dimensions: {stats.dimensions}")
    print(f"  - Memory usage: {stats.memory_usage_mb:.2f} MB")
    print()

    # Step 11: Delete a snapshot
    print("Step 11: Deleting a snapshot...")
    deleted = manager.delete_snapshot("backup-partial")
    if deleted:
        print("Snapshot 'backup-partial' deleted successfully")
        print(f"Remaining snapshots: {len(manager)}")
    print()

    # Step 12: Create a timestamped snapshot
    print("Step 12: Creating timestamped snapshot...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    snapshot_name = f"auto_backup_{timestamp}"
    info = manager.create_snapshot_with_ids(
        db,
        name=snapshot_name,
        vector_ids=vector_ids,
        description=f"Automated backup created at {timestamp}",
    )
    print(f"Created automated snapshot: {info.name}")
    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Original database: {len(db)} vectors, {dimensions} dimensions")
    print(f"Snapshots created: {len(manager)}")
    print(f"Total storage: {manager.total_size_mb:.2f} MB")
    print()

    final_snapshots = manager.list_snapshots()
    print("Final snapshot list:")
    for snap in final_snapshots:
        print(f"  - {snap.name} ({snap.size_mb:.2f} MB)")

    print()
    print("Snapshot example completed successfully!")


if __name__ == "__main__":
    main()
