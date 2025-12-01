"""
Insertion benchmark suite for pyruvector.

Tests various insertion scenarios:
- Single vector insertion
- Batch insertion with different batch sizes
- Insertion with metadata
- Concurrent insertion (if supported)
- Memory growth during insertion
"""

import sys
import time
import traceback
from typing import Dict, Any, List
import numpy as np
import psutil
import os

# Import benchmark utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.metrics import BenchmarkResult, MetricsCollector, MemoryTracker

# Import real pyruvector - fail fast if not built
try:
    from pyruvector import VectorDB
except ImportError as e:
    raise ImportError(
        "pyruvector must be built before running benchmarks.\n"
        "Run: maturin develop --release\n"
        f"Original error: {e}"
    ) from e


def _create_db(dimension: int):
    """Create a vector database instance."""
    return VectorDB(dimensions=dimension)


def _generate_vectors(count: int, dimension: int) -> np.ndarray:
    """Generate random test vectors."""
    return np.random.randn(count, dimension).astype(np.float32)


def _generate_metadata(count: int) -> List[Dict[str, Any]]:
    """Generate random metadata."""
    return [
        {
            'id': i,
            'category': f'cat_{i % 10}',
            'timestamp': time.time(),
        }
        for i in range(count)
    ]


def benchmark_single_insert(config: Dict[str, Any], metrics: MetricsCollector) -> Dict[str, Any]:
    """Benchmark single vector insertion performance."""
    dimension = config.get('dimension', 128)
    num_inserts = config.get('num_inserts', 1000)

    db = _create_db(dimension)
    vectors = _generate_vectors(num_inserts, dimension)

    # Warm-up
    warmup_vectors = _generate_vectors(10, dimension)
    for vec in warmup_vectors:
        db.insert(vec)

    # Recreate database for actual benchmark
    db = _create_db(dimension)

    # Benchmark
    latencies = []
    for vec in vectors:
        start = time.perf_counter()
        db.insert(vec)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)

    return {
        'mean_latency_ms': np.mean(latencies),
        'median_latency_ms': np.median(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'max_latency_ms': np.max(latencies),
        'throughput_ops_per_sec': 1000 / np.mean(latencies),
    }


def benchmark_batch_insert(config: Dict[str, Any], metrics: MetricsCollector) -> Dict[str, Any]:
    """Benchmark batch insertion with various batch sizes."""
    dimension = config.get('dimension', 128)
    total_vectors = config.get('total_vectors', 10000)
    batch_sizes = config.get('batch_sizes', [10, 50, 100, 500, 1000])

    results = {}

    for batch_size in batch_sizes:
        db = _create_db(dimension)
        num_batches = total_vectors // batch_size

        # Warm-up
        warmup_vecs = _generate_vectors(batch_size, dimension)
        db.insert_batch(warmup_vecs)

        # Recreate database for actual benchmark
        db = _create_db(dimension)

        # Benchmark
        batch_latencies = []
        for _ in range(num_batches):
            vectors = _generate_vectors(batch_size, dimension)

            start = time.perf_counter()
            db.insert_batch(vectors)
            latency = (time.perf_counter() - start) * 1000  # ms
            batch_latencies.append(latency)

        results[f'batch_{batch_size}'] = {
            'mean_latency_ms': np.mean(batch_latencies),
            'median_latency_ms': np.median(batch_latencies),
            'throughput_vectors_per_sec': (batch_size * 1000) / np.mean(batch_latencies),
            'vectors_per_batch': batch_size,
        }

    return results


def benchmark_insert_with_metadata(config: Dict[str, Any], metrics: MetricsCollector) -> Dict[str, Any]:
    """Benchmark insertion with metadata."""
    dimension = config.get('dimension', 128)
    num_inserts = config.get('num_inserts', 1000)

    db = _create_db(dimension)
    vectors = _generate_vectors(num_inserts, dimension)
    metadata_list = _generate_metadata(num_inserts)

    # Warm-up
    warmup_vecs = _generate_vectors(10, dimension)
    warmup_meta = _generate_metadata(10)
    for vec, meta in zip(warmup_vecs, warmup_meta):
        db.insert(vec, meta)

    # Recreate database for actual benchmark
    db = _create_db(dimension)

    # Benchmark
    latencies = []
    for vec, meta in zip(vectors, metadata_list):
        start = time.perf_counter()
        db.insert(vec, meta)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)

    return {
        'mean_latency_ms': np.mean(latencies),
        'median_latency_ms': np.median(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'throughput_ops_per_sec': 1000 / np.mean(latencies),
    }


def benchmark_memory_growth(config: Dict[str, Any], metrics: MetricsCollector) -> Dict[str, Any]:
    """Benchmark memory growth during insertion."""
    dimension = config.get('dimension', 128)
    num_inserts = config.get('num_inserts', 10000)
    sample_interval = config.get('sample_interval', 100)

    db = _create_db(dimension)
    vectors = _generate_vectors(num_inserts, dimension)

    process = psutil.Process()
    memory_samples = []
    vector_counts = []

    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

    for i, vec in enumerate(vectors):
        db.insert(vec)

        if i % sample_interval == 0:
            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_samples.append(current_memory)
            vector_counts.append(i + 1)

    final_memory = process.memory_info().rss / (1024 * 1024)  # MB

    # Calculate memory per vector
    total_growth = final_memory - initial_memory
    memory_per_vector = total_growth / num_inserts if num_inserts > 0 else 0

    return {
        'initial_memory_mb': initial_memory,
        'final_memory_mb': final_memory,
        'total_growth_mb': total_growth,
        'memory_per_vector_kb': memory_per_vector * 1024,
        'memory_samples': memory_samples,
        'vector_counts': vector_counts,
    }


def run(config: Dict[str, Any]) -> BenchmarkResult:
    """
    Run the insertion benchmark suite.

    Args:
        config: Configuration dictionary with test parameters

    Returns:
        BenchmarkResult with all test results and metrics
    """
    metrics = MetricsCollector()
    memory_tracker = MemoryTracker()

    suite_name = 'insertion'
    results = {}
    errors = []

    print(f"\n{'='*60}")
    print("Running Insertion Benchmark Suite")
    print(f"{'='*60}\n")

    # Track overall suite execution
    suite_start = time.perf_counter()
    memory_tracker.start()

    tests = [
        ('single_insert', benchmark_single_insert, 'Single Vector Insertion'),
        ('batch_insert', benchmark_batch_insert, 'Batch Insertion'),
        ('insert_with_metadata', benchmark_insert_with_metadata, 'Insertion with Metadata'),
        ('memory_growth', benchmark_memory_growth, 'Memory Growth'),
    ]

    for test_name, test_func, test_desc in tests:
        print(f"Running: {test_desc}...", end=' ', flush=True)

        try:
            test_start = time.perf_counter()
            test_results = test_func(config, metrics)
            test_duration = time.perf_counter() - test_start

            results[test_name] = test_results
            metrics.record_operation(test_name, test_duration)

            print(f"✓ ({test_duration:.2f}s)")

        except Exception as e:
            error_msg = f"{test_name}: {str(e)}\n{traceback.format_exc()}"
            errors.append(error_msg)
            print("✗ FAILED")
            print(f"  Error: {str(e)}")

    suite_duration = time.perf_counter() - suite_start
    memory_stats = memory_tracker.stop()

    # Create benchmark result
    benchmark_result = BenchmarkResult(
        suite_name=suite_name,
        config=config,
        results=results,
        metrics=metrics.get_summary(),
        memory_stats=memory_stats,
        duration=suite_duration,
        timestamp=time.time(),
        errors=errors if errors else None,
    )

    print(f"\n{'='*60}")
    print(f"Suite completed in {suite_duration:.2f}s")
    print(f"Peak memory: {memory_stats['peak_mb']:.2f} MB")
    print(f"{'='*60}\n")

    return benchmark_result
