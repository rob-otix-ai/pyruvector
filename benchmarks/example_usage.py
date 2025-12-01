"""
Example usage of the PyRuvector benchmark framework.

Demonstrates how to use the benchmark suites to measure performance
of your vector database implementation.
"""

import sys
from pathlib import Path

# Add parent directory to path to import pyruvector
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks import Timer, MemoryTracker, MetricsCollector, VectorGenerator, BenchmarkReporter
from benchmarks.suites import InsertionBenchmark, SearchBenchmark, MixedWorkloadBenchmark

# Import real pyruvector - fail fast if not built
try:
    from pyruvector import VectorDB
except ImportError as e:
    raise ImportError(
        "pyruvector must be built before running benchmarks.\n"
        "Run: maturin develop --release\n"
        f"Original error: {e}"
    ) from e


def example_basic_usage():
    """Example of basic timer and memory tracking."""
    print("\n" + "="*70)
    print("Example 1: Basic Timer and Memory Tracking")
    print("="*70)

    # Using Timer
    with Timer() as t:
        sum(range(1000000))

    print(f"Computation took: {t.duration_ms:.2f}ms")

    # Using MemoryTracker
    tracker = MemoryTracker()
    tracker.set_baseline()

    # Allocate some memory

    delta = tracker.delta()
    print(f"Memory delta: {delta['rss_mb']:.2f}MB")


def example_metrics_collector():
    """Example of collecting and analyzing metrics."""
    print("\n" + "="*70)
    print("Example 2: Metrics Collection")
    print("="*70)

    collector = MetricsCollector()

    # Simulate some operations
    import time
    for i in range(100):
        with Timer() as t:
            time.sleep(0.001 + (i % 10) * 0.0001)  # Variable latency
        collector.record(t.duration_ms)

    # Get statistics
    summary = collector.summary()
    print(f"Mean latency: {summary['mean']:.2f}ms")
    print(f"Median latency: {summary['median']:.2f}ms")
    print(f"P95 latency: {summary['p95']:.2f}ms")
    print(f"P99 latency: {summary['p99']:.2f}ms")
    print(f"Min/Max: {summary['min']:.2f}ms / {summary['max']:.2f}ms")


def example_vector_generation():
    """Example of vector data generation."""
    print("\n" + "="*70)
    print("Example 3: Vector Generation")
    print("="*70)

    generator = VectorGenerator(dimensions=128, seed=42)

    # Generate random vectors
    random_vecs = generator.random(count=1000, normalize=True)
    print(f"Generated {len(random_vecs)} random vectors of dimension {random_vecs.shape[1]}")

    # Generate clustered vectors
    clustered_vecs, labels = generator.clustered(count=1000)
    print(f"Generated {len(clustered_vecs)} clustered vectors")
    print(f"Number of unique clusters: {len(set(labels))}")

    # Analyze distribution
    info = generator.distribution_info(clustered_vecs)
    print(f"Average vector norm: {info['norm_mean']:.4f}")
    print(f"Average pairwise distance: {info['dist_mean']:.4f}")


def example_insertion_benchmark():
    """Example of running insertion benchmarks."""
    print("\n" + "="*70)
    print("Example 4: Insertion Benchmark")
    print("="*70)

    # Create benchmark suite
    bench = InsertionBenchmark(
        database_class=VectorDB,
        dimensions=128,
        seed=42
    )

    # Run single insert benchmark
    print("\nRunning single insert benchmark...")
    result = bench.run_single_insert(count=100)

    print(f"Mean latency: {result.metrics['mean']:.2f}ms")
    print(f"Throughput: {result.metrics['throughput']:.0f} ops/sec")

    # Run batch insert benchmark
    print("\nRunning batch insert benchmark...")
    result = bench.run_batch_insert(batch_size=10, num_batches=10)

    print(f"Mean latency: {result.metrics['mean']:.2f}ms")
    print(f"Throughput: {result.metrics['throughput']:.0f} vectors/sec")


def example_search_benchmark():
    """Example of running search benchmarks."""
    print("\n" + "="*70)
    print("Example 5: Search Benchmark")
    print("="*70)

    bench = SearchBenchmark(
        database_class=VectorDB,
        dimensions=128,
        seed=42
    )

    # Run search scaling benchmark
    print("\nRunning search benchmark with 1000 vectors...")
    result = bench.run_search_scaling(database_size=1000, num_queries=50, k=10)

    print(f"Mean search latency: {result.metrics['mean']:.2f}ms")
    print(f"P95 latency: {result.metrics['p95']:.2f}ms")
    print(f"QPS: {result.metrics['qps']:.0f}")


def example_mixed_workload():
    """Example of running mixed workload benchmarks."""
    print("\n" + "="*70)
    print("Example 6: Mixed Workload Benchmark")
    print("="*70)

    bench = MixedWorkloadBenchmark(
        database_class=VectorDB,
        dimensions=128,
        seed=42
    )

    # Run read-heavy workload
    print("\nRunning read-heavy workload (90% search)...")
    result = bench.run_read_heavy(initial_size=1000, num_operations=100)

    print(f"Total throughput: {result.metrics['total_throughput']:.0f} ops/sec")
    print(f"Search mean: {result.metrics['search_mean_ms']:.2f}ms")
    print(f"Insert mean: {result.metrics['insert_mean_ms']:.2f}ms")


def example_reporting():
    """Example of generating reports."""
    print("\n" + "="*70)
    print("Example 7: Reporting Results")
    print("="*70)

    reporter = BenchmarkReporter()

    # Run a few benchmarks
    bench = InsertionBenchmark(VectorDB, dimensions=128)

    result1 = bench.run_single_insert(count=100)
    result2 = bench.run_batch_insert(batch_size=10, num_batches=10)

    reporter.add_result(result1)
    reporter.add_result(result2)

    # Print table
    print("\nResults Table:")
    reporter.print_table(metrics_to_show=['mean', 'p95', 'throughput'])

    # Export to JSON
    output_dir = Path(__file__).parent
    reporter.export_json(output_dir / "results.json")
    print(f"\nResults exported to: {output_dir / 'results.json'}")

    # Export to Markdown
    reporter.export_markdown(output_dir / "results.md", title="Benchmark Results")
    print(f"Report exported to: {output_dir / 'results.md'}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PyRuvector Benchmark Framework - Example Usage")
    print("="*70)

    try:
        example_basic_usage()
        example_metrics_collector()
        example_vector_generation()
        example_insertion_benchmark()
        example_search_benchmark()
        example_mixed_workload()
        example_reporting()

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
