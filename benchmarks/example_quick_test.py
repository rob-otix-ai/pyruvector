#!/usr/bin/env python3
"""
Quick example showing how to run pyruvector benchmarks.

This is a standalone script that doesn't require numpy or other dependencies
to demonstrate basic usage.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fail-fast import - no mocks, no fallbacks
from pyruvector import VectorDB


def simple_benchmark():
    """Run a simple benchmark without numpy."""
    print("=" * 80)
    print("PYRUVECTOR SIMPLE BENCHMARK")
    print("=" * 80)

    # Test configuration
    dimensions = 128
    vector_count = 1000
    batch_size = 100
    k = 10

    print("\nConfiguration:")
    print(f"  Dimensions: {dimensions}")
    print(f"  Vectors: {vector_count:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  k: {k}")

    # Create database
    print("\nCreating database...")
    db = VectorDB(dimensions=dimensions)

    # Generate simple test vectors (no numpy needed)
    def generate_vector(i, dim):
        """Generate a simple test vector."""
        vec = [0.0] * dim
        # Create a pattern based on index
        for j in range(dim):
            vec[j] = float((i + j) % 10) / 10.0
        # Normalize
        magnitude = sum(x * x for x in vec) ** 0.5
        return [x / magnitude for x in vec]

    # Benchmark insertion
    print(f"\nInserting {vector_count:,} vectors...")
    start_time = time.time()

    for i in range(vector_count):
        vec = generate_vector(i, dimensions)
        db.insert(f"vec_{i}", vec, {"id": i, "batch": i // batch_size})

    insert_time = time.time() - start_time
    throughput = vector_count / insert_time

    print(f"  ✓ Inserted in {insert_time:.2f}s ({throughput:,.0f} vectors/sec)")

    # Benchmark search
    print(f"\nBenchmarking search (k={k})...")
    num_queries = 100
    search_times = []

    for i in range(num_queries):
        query = generate_vector(i * 10, dimensions)
        start = time.perf_counter()
        db.search(query, k=k)
        latency = (time.perf_counter() - start) * 1000  # Convert to ms
        search_times.append(latency)

    avg_latency = sum(search_times) / len(search_times)
    min_latency = min(search_times)
    max_latency = max(search_times)

    print(f"  ✓ Avg: {avg_latency:.2f}ms | Min: {min_latency:.2f}ms | Max: {max_latency:.2f}ms")

    # Database info
    print("\nDatabase stats:")
    print(f"  Total vectors: {db.len():,}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    return {
        "insert_throughput": throughput,
        "insert_time": insert_time,
        "search_latency_avg": avg_latency,
        "search_latency_min": min_latency,
        "search_latency_max": max_latency,
    }


if __name__ == "__main__":
    try:
        results = simple_benchmark()
        print("\nTo run the full benchmark suite with database comparison:")
        print("  1. Install dependencies: pip install numpy qdrant-client")
        print("  2. Run: python3 -m benchmarks.run_benchmarks --quick")
        print("\nOr for detailed comparison:")
        print("  python3 -m benchmarks.run_benchmarks --db all --vectors 10000 50000")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
