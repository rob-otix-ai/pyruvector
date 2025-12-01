#!/usr/bin/env python3
"""
Basic usage example for pyruvector.

This example demonstrates:
- Creating a VectorDB
- Inserting vectors with metadata
- Searching for similar vectors
- Retrieving results
"""

from pyruvector import VectorDB


def main():
    print("=== pyruvector Basic Usage Example ===\n")

    # Create a new vector database with 4-dimensional vectors
    db = VectorDB(dimension=4)
    print("Created vector database with dimension=4\n")

    # Insert some vectors with metadata
    print("Inserting vectors...")
    id1 = db.insert([1.0, 0.0, 0.0, 0.0], {"name": "vector_1", "type": "unit_x"})
    db.insert([0.0, 1.0, 0.0, 0.0], {"name": "vector_2", "type": "unit_y"})
    db.insert([0.0, 0.0, 1.0, 0.0], {"name": "vector_3", "type": "unit_z"})
    db.insert([0.7, 0.7, 0.0, 0.0], {"name": "vector_4", "type": "mixed"})

    print(f"Inserted {db.count()} vectors\n")

    # Search for vectors similar to [0.9, 0.1, 0.0, 0.0]
    query = [0.9, 0.1, 0.0, 0.0]
    print(f"Searching for vectors similar to {query}...")

    results = db.search(query, k=3)

    print(f"\nTop {len(results)} results:")
    print("-" * 60)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Vector ID: {result['id']}")
        print(f"   Distance: {result['distance']:.4f}")
        print(f"   Vector: {result['vector']}")
        print(f"   Metadata: {result['metadata']}")

    # Retrieve a specific vector by ID
    print("\n" + "=" * 60)
    print(f"\nRetrieving vector with ID {id1}...")
    vector_data = db.get(id1)

    if vector_data:
        print(f"Vector: {vector_data['vector']}")
        print(f"Metadata: {vector_data['metadata']}")

    print(f"\nTotal vectors in database: {db.count()}")


if __name__ == "__main__":
    main()
