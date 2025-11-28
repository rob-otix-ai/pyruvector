#!/usr/bin/env python3
"""
Filtered search example for pyruvector.

This example demonstrates:
- Various filter operators ($eq, $ne, $gt, $gte, $lt, $lte)
- Array operators ($in, $nin, $contains)
- Range queries
- Combined filters
"""

from pyruvector import VectorDB


def main():
    print("=== pyruvector Filtered Search Example ===\n")

    # Create database and insert vectors with rich metadata
    db = VectorDB(dimension=4)

    print("Inserting vectors with metadata...")
    db.insert([1.0, 0.0, 0.0, 0.0], {
        "category": "A",
        "score": 0.9,
        "status": "active",
        "tags": ["important", "verified"]
    })
    db.insert([0.9, 0.1, 0.0, 0.0], {
        "category": "A",
        "score": 0.7,
        "status": "active",
        "tags": ["verified"]
    })
    db.insert([0.0, 1.0, 0.0, 0.0], {
        "category": "B",
        "score": 0.5,
        "status": "pending",
        "tags": ["important"]
    })
    db.insert([0.0, 0.0, 1.0, 0.0], {
        "category": "C",
        "score": 0.3,
        "status": "archived",
        "tags": ["old"]
    })
    db.insert([0.5, 0.5, 0.0, 0.0], {
        "category": "A",
        "score": 0.85,
        "status": "active",
        "tags": ["important", "verified", "premium"]
    })

    print(f"Inserted {db.count()} vectors\n")

    query = [1.0, 0.0, 0.0, 0.0]

    # Example 1: Equality filter
    print("=" * 70)
    print("Example 1: Filter by exact category ($eq)")
    print("-" * 70)

    results = db.search(query, k=5, filter={"category": {"$eq": "A"}})
    print(f"Found {len(results)} vectors in category 'A':")
    for r in results:
        print(f"  ID {r['id']}: category={r['metadata']['category']}, distance={r['distance']:.4f}")

    # Example 2: Range query
    print("\n" + "=" * 70)
    print("Example 2: Filter by score range ($gte)")
    print("-" * 70)

    results = db.search(query, k=5, filter={"score": {"$gte": 0.7}})
    print(f"Found {len(results)} vectors with score >= 0.7:")
    for r in results:
        print(f"  ID {r['id']}: score={r['metadata']['score']}, distance={r['distance']:.4f}")

    # Example 3: Not equal
    print("\n" + "=" * 70)
    print("Example 3: Exclude archived items ($ne)")
    print("-" * 70)

    results = db.search(query, k=5, filter={"status": {"$ne": "archived"}})
    print(f"Found {len(results)} non-archived vectors:")
    for r in results:
        print(f"  ID {r['id']}: status={r['metadata']['status']}, distance={r['distance']:.4f}")

    # Example 4: In array
    print("\n" + "=" * 70)
    print("Example 4: Filter by multiple categories ($in)")
    print("-" * 70)

    results = db.search(query, k=5, filter={"category": {"$in": ["A", "B"]}})
    print(f"Found {len(results)} vectors in categories A or B:")
    for r in results:
        print(f"  ID {r['id']}: category={r['metadata']['category']}, distance={r['distance']:.4f}")

    # Example 5: Array contains
    print("\n" + "=" * 70)
    print("Example 5: Filter by tag ($contains)")
    print("-" * 70)

    results = db.search(query, k=5, filter={"tags": {"$contains": "important"}})
    print(f"Found {len(results)} vectors tagged 'important':")
    for r in results:
        print(f"  ID {r['id']}: tags={r['metadata']['tags']}, distance={r['distance']:.4f}")

    # Example 6: Combined filters
    print("\n" + "=" * 70)
    print("Example 6: Multiple conditions (AND logic)")
    print("-" * 70)

    results = db.search(query, k=5, filter={
        "category": {"$eq": "A"},
        "score": {"$gte": 0.8},
        "status": {"$eq": "active"}
    })
    print(f"Found {len(results)} active category A vectors with score >= 0.8:")
    for r in results:
        meta = r['metadata']
        print(f"  ID {r['id']}: category={meta['category']}, score={meta['score']}, status={meta['status']}")

    # Example 7: Less than filter
    print("\n" + "=" * 70)
    print("Example 7: Filter by score threshold ($lt)")
    print("-" * 70)

    results = db.search(query, k=5, filter={"score": {"$lt": 0.6}})
    print(f"Found {len(results)} vectors with score < 0.6:")
    for r in results:
        print(f"  ID {r['id']}: score={r['metadata']['score']}, distance={r['distance']:.4f}")


if __name__ == "__main__":
    main()
