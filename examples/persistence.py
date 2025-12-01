#!/usr/bin/env python3
"""
Persistence example for pyruvector.

This example demonstrates:
- Creating a database with a file path
- Inserting and saving data
- Closing and reopening the database
- Loading persisted data
"""

import os
from pyruvector import VectorDB


def main():
    print("=== pyruvector Persistence Example ===\n")

    db_path = "example_vectors.db"

    # Clean up if file exists from previous run
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database file: {db_path}\n")

    # Part 1: Create and populate database
    print("Part 1: Creating and populating database")
    print("-" * 60)

    db = VectorDB(dimension=4, path=db_path)
    print(f"Created database with path: {db_path}")

    # Insert some vectors
    print("\nInserting vectors...")
    db.insert([1.0, 0.0, 0.0, 0.0], {"name": "vector_1", "type": "unit_x"})
    db.insert([0.0, 1.0, 0.0, 0.0], {"name": "vector_2", "type": "unit_y"})
    db.insert([0.0, 0.0, 1.0, 0.0], {"name": "vector_3", "type": "unit_z"})
    db.insert([0.7, 0.7, 0.0, 0.0], {"name": "vector_4", "type": "mixed"})
    db.insert([0.5, 0.5, 0.5, 0.0], {"name": "vector_5", "type": "mixed"})

    print(f"Inserted {db.count()} vectors")

    # Save to disk
    print(f"\nSaving database to {db_path}...")
    db.save(db_path)
    print("Database saved successfully")

    # Show initial search results
    query = [1.0, 0.0, 0.0, 0.0]
    results = db.search(query, k=3)

    print(f"\nInitial search results for {query}:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['metadata']['name']} - distance: {r['distance']:.4f}")

    # Part 2: Load database from disk
    print("\n" + "=" * 60)
    print("Part 2: Loading database from disk")
    print("-" * 60)

    print(f"\nLoading database from {db_path}...")
    loaded_db = VectorDB.load(db_path)
    print("Database loaded successfully")
    print(f"Vector count: {loaded_db.count()}")

    # Search in loaded database
    print("\nSearch results after loading:")
    results = loaded_db.search(query, k=3)

    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['metadata']['name']} - distance: {r['distance']:.4f}")

    # Part 3: Modify and re-save
    print("\n" + "=" * 60)
    print("Part 3: Modifying and re-saving")
    print("-" * 60)

    print("\nAdding more vectors to loaded database...")
    loaded_db.insert([0.0, 0.0, 0.0, 1.0], {"name": "vector_6", "type": "unit_w"})
    loaded_db.insert([0.25, 0.25, 0.25, 0.25], {"name": "vector_7", "type": "uniform"})

    print(f"New vector count: {loaded_db.count()}")

    print("\nSaving updated database...")
    loaded_db.save(db_path)
    print("Database saved successfully")

    # Part 4: Verify persistence
    print("\n" + "=" * 60)
    print("Part 4: Verifying persistence")
    print("-" * 60)

    final_db = VectorDB.load(db_path)
    print(f"\nFinal database vector count: {final_db.count()}")

    results = final_db.search(query, k=5)
    print("\nFinal search results (top 5):")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['metadata']['name']} - distance: {r['distance']:.4f}")

    # Cleanup
    print("\n" + "=" * 60)
    print(f"\nCleaning up: removing {db_path}")
    os.remove(db_path)
    print("Done!")


if __name__ == "__main__":
    main()
