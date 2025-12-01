"""
Mixed workload benchmark suite for pyruvector.

Tests real-world usage patterns:
- Read-heavy workload (90% search, 10% insert)
- Write-heavy workload (90% insert, 10% search)
- Balanced workload (50% search, 50% insert)
- Concurrent operations
- Sustained throughput
"""

import sys
import time
import traceback
from typing import Dict, Any
import numpy as np
import os
from collections import defaultdict

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


def benchmark_read_heavy(config: Dict[str, Any], metrics: MetricsCollector) -> Dict[str, Any]:
    """
    Benchmark read-heavy workload (90% search, 10% insert).
    Simulates a production recommendation system or search engine.
    """
    dimension = config.get('dimension', 128)
    initial_vectors = config.get('initial_vectors', 10000)
    num_operations = config.get('num_operations', 1000)
    k = config.get('k', 10)

    # Setup database with initial data
    db = _create_db(dimension)
    vectors = _generate_vectors(initial_vectors, dimension)
    db.insert_batch(vectors)

    # Pre-generate operations
    insert_vectors = _generate_vectors(num_operations, dimension)
    query_vectors = _generate_vectors(num_operations, dimension)

    # Generate operation sequence (90% read, 10% write)
    operations = ['search'] * 90 + ['insert'] * 10
    np.random.shuffle(operations)
    operations = operations[:100]  # Take first 100 for pattern

    # Scale to num_operations
    full_operations = (operations * (num_operations // 100 + 1))[:num_operations]

    # Warm-up
    for _ in range(10):
        db.search(query_vectors[0], k=k)

    # Execute mixed workload
    insert_latencies = []
    search_latencies = []
    insert_idx = 0
    search_idx = 0

    overall_start = time.perf_counter()

    for op in full_operations:
        if op == 'insert':
            vec = insert_vectors[insert_idx % len(insert_vectors)]
            start = time.perf_counter()
            db.insert(vec)
            latency = (time.perf_counter() - start) * 1000
            insert_latencies.append(latency)
            insert_idx += 1
        else:  # search
            query = query_vectors[search_idx % len(query_vectors)]
            start = time.perf_counter()
            db.search(query, k=k)
            latency = (time.perf_counter() - start) * 1000
            search_latencies.append(latency)
            search_idx += 1

    overall_duration = time.perf_counter() - overall_start

    return {
        'total_operations': num_operations,
        'search_operations': len(search_latencies),
        'insert_operations': len(insert_latencies),
        'search_mean_latency_ms': np.mean(search_latencies),
        'search_p95_latency_ms': np.percentile(search_latencies, 95),
        'insert_mean_latency_ms': np.mean(insert_latencies),
        'insert_p95_latency_ms': np.percentile(insert_latencies, 95),
        'overall_throughput_ops_per_sec': num_operations / overall_duration,
        'duration_seconds': overall_duration,
    }


def benchmark_write_heavy(config: Dict[str, Any], metrics: MetricsCollector) -> Dict[str, Any]:
    """
    Benchmark write-heavy workload (90% insert, 10% search).
    Simulates initial data loading or bulk ingestion scenarios.
    """
    dimension = config.get('dimension', 128)
    initial_vectors = config.get('initial_vectors', 1000)
    num_operations = config.get('num_operations', 1000)
    k = config.get('k', 10)

    # Setup database with minimal initial data
    db = _create_db(dimension)
    vectors = _generate_vectors(initial_vectors, dimension)
    db.insert_batch(vectors)

    # Pre-generate operations
    insert_vectors = _generate_vectors(num_operations, dimension)
    query_vectors = _generate_vectors(num_operations, dimension)

    # Generate operation sequence (10% read, 90% write)
    operations = ['search'] * 10 + ['insert'] * 90
    np.random.shuffle(operations)
    operations = operations[:100]

    # Scale to num_operations
    full_operations = (operations * (num_operations // 100 + 1))[:num_operations]

    # Execute mixed workload
    insert_latencies = []
    search_latencies = []
    insert_idx = 0
    search_idx = 0

    overall_start = time.perf_counter()

    for op in full_operations:
        if op == 'insert':
            vec = insert_vectors[insert_idx % len(insert_vectors)]
            start = time.perf_counter()
            db.insert(vec)
            latency = (time.perf_counter() - start) * 1000
            insert_latencies.append(latency)
            insert_idx += 1
        else:  # search
            query = query_vectors[search_idx % len(query_vectors)]
            start = time.perf_counter()
            db.search(query, k=k)
            latency = (time.perf_counter() - start) * 1000
            search_latencies.append(latency)
            search_idx += 1

    overall_duration = time.perf_counter() - overall_start

    return {
        'total_operations': num_operations,
        'search_operations': len(search_latencies),
        'insert_operations': len(insert_latencies),
        'search_mean_latency_ms': np.mean(search_latencies) if search_latencies else 0,
        'search_p95_latency_ms': np.percentile(search_latencies, 95) if search_latencies else 0,
        'insert_mean_latency_ms': np.mean(insert_latencies),
        'insert_p95_latency_ms': np.percentile(insert_latencies, 95),
        'overall_throughput_ops_per_sec': num_operations / overall_duration,
        'duration_seconds': overall_duration,
    }


def benchmark_balanced(config: Dict[str, Any], metrics: MetricsCollector) -> Dict[str, Any]:
    """
    Benchmark balanced workload (50% search, 50% insert).
    Simulates typical application with continuous data ingestion and queries.
    """
    dimension = config.get('dimension', 128)
    initial_vectors = config.get('initial_vectors', 5000)
    num_operations = config.get('num_operations', 1000)
    k = config.get('k', 10)

    # Setup database
    db = _create_db(dimension)
    vectors = _generate_vectors(initial_vectors, dimension)
    db.insert_batch(vectors)

    # Pre-generate operations
    insert_vectors = _generate_vectors(num_operations, dimension)
    query_vectors = _generate_vectors(num_operations, dimension)

    # Generate balanced operation sequence
    operations = ['search'] * 50 + ['insert'] * 50
    np.random.shuffle(operations)
    operations = operations[:100]

    # Scale to num_operations
    full_operations = (operations * (num_operations // 100 + 1))[:num_operations]

    # Execute mixed workload
    insert_latencies = []
    search_latencies = []
    insert_idx = 0
    search_idx = 0

    overall_start = time.perf_counter()

    for op in full_operations:
        if op == 'insert':
            vec = insert_vectors[insert_idx % len(insert_vectors)]
            start = time.perf_counter()
            db.insert(vec)
            latency = (time.perf_counter() - start) * 1000
            insert_latencies.append(latency)
            insert_idx += 1
        else:  # search
            query = query_vectors[search_idx % len(query_vectors)]
            start = time.perf_counter()
            db.search(query, k=k)
            latency = (time.perf_counter() - start) * 1000
            search_latencies.append(latency)
            search_idx += 1

    overall_duration = time.perf_counter() - overall_start

    return {
        'total_operations': num_operations,
        'search_operations': len(search_latencies),
        'insert_operations': len(insert_latencies),
        'search_mean_latency_ms': np.mean(search_latencies),
        'search_p95_latency_ms': np.percentile(search_latencies, 95),
        'insert_mean_latency_ms': np.mean(insert_latencies),
        'insert_p95_latency_ms': np.percentile(insert_latencies, 95),
        'overall_throughput_ops_per_sec': num_operations / overall_duration,
        'duration_seconds': overall_duration,
    }


def benchmark_sustained_throughput(config: Dict[str, Any], metrics: MetricsCollector) -> Dict[str, Any]:
    """
    Benchmark sustained throughput over time.
    Tests system stability under continuous load.
    """
    dimension = config.get('dimension', 128)
    initial_vectors = config.get('initial_vectors', 5000)
    duration_seconds = config.get('duration_seconds', 10)
    k = config.get('k', 10)

    # Setup database
    db = _create_db(dimension)
    vectors = _generate_vectors(initial_vectors, dimension)
    db.insert_batch(vectors)

    # Pre-generate operations
    insert_vectors = _generate_vectors(10000, dimension)
    query_vectors = _generate_vectors(10000, dimension)

    # Mixed operations (70% search, 30% insert)
    operations = ['search'] * 70 + ['insert'] * 30
    np.random.shuffle(operations)

    # Run for specified duration
    insert_idx = 0
    search_idx = 0
    op_idx = 0

    operation_counts = defaultdict(int)
    latencies_by_second = defaultdict(list)

    start_time = time.perf_counter()
    end_time = start_time + duration_seconds

    total_ops = 0

    while time.perf_counter() < end_time:
        op = operations[op_idx % len(operations)]
        current_time = time.perf_counter()
        elapsed_seconds = int(current_time - start_time)

        if op == 'insert':
            vec = insert_vectors[insert_idx % len(insert_vectors)]
            op_start = time.perf_counter()
            db.insert(vec)
            latency = (time.perf_counter() - op_start) * 1000
            latencies_by_second[elapsed_seconds].append(latency)
            insert_idx += 1
            operation_counts['insert'] += 1
        else:  # search
            query = query_vectors[search_idx % len(query_vectors)]
            op_start = time.perf_counter()
            db.search(query, k=k)
            latency = (time.perf_counter() - op_start) * 1000
            latencies_by_second[elapsed_seconds].append(latency)
            search_idx += 1
            operation_counts['search'] += 1

        op_idx += 1
        total_ops += 1

    actual_duration = time.perf_counter() - start_time

    # Calculate per-second throughput
    throughput_by_second = {
        sec: len(latencies) / 1.0 if latencies else 0
        for sec, latencies in latencies_by_second.items()
    }

    all_latencies = [lat for lats in latencies_by_second.values() for lat in lats]

    return {
        'duration_seconds': actual_duration,
        'total_operations': total_ops,
        'search_operations': operation_counts['search'],
        'insert_operations': operation_counts['insert'],
        'overall_throughput_ops_per_sec': total_ops / actual_duration,
        'mean_latency_ms': np.mean(all_latencies),
        'p95_latency_ms': np.percentile(all_latencies, 95),
        'p99_latency_ms': np.percentile(all_latencies, 99),
        'min_throughput_ops_per_sec': min(throughput_by_second.values()) if throughput_by_second else 0,
        'max_throughput_ops_per_sec': max(throughput_by_second.values()) if throughput_by_second else 0,
        'throughput_by_second': dict(throughput_by_second),
    }


def run(config: Dict[str, Any]) -> BenchmarkResult:
    """
    Run the mixed workload benchmark suite.

    Args:
        config: Configuration dictionary with test parameters

    Returns:
        BenchmarkResult with all test results and metrics
    """
    metrics = MetricsCollector()
    memory_tracker = MemoryTracker()

    suite_name = 'mixed'
    results = {}
    errors = []

    print(f"\n{'='*60}")
    print("Running Mixed Workload Benchmark Suite")
    print(f"{'='*60}\n")

    # Track overall suite execution
    suite_start = time.perf_counter()
    memory_tracker.start()

    tests = [
        ('read_heavy', benchmark_read_heavy, 'Read-Heavy Workload (90% search)'),
        ('write_heavy', benchmark_write_heavy, 'Write-Heavy Workload (90% insert)'),
        ('balanced', benchmark_balanced, 'Balanced Workload (50/50)'),
        ('sustained_throughput', benchmark_sustained_throughput, 'Sustained Throughput'),
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
