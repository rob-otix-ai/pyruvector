"""
Search benchmark suite for pyruvector.

Tests various search scenarios:
- Basic KNN search
- Search with different k values
- Filtered search with metadata
- Search latency percentiles
- Recall accuracy vs brute force
"""

import sys
import time
import traceback
from typing import Dict, Any, List
import numpy as np
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
            'score': float(i % 100),
        }
        for i in range(count)
    ]


def _calculate_recall(retrieved: List[int], ground_truth: List[int]) -> float:
    """Calculate recall@k."""
    if not ground_truth:
        return 0.0

    retrieved_set = set(retrieved)
    ground_truth_set = set(ground_truth)

    intersection = len(retrieved_set & ground_truth_set)
    return intersection / len(ground_truth_set)


def benchmark_basic_knn(config: Dict[str, Any], metrics: MetricsCollector) -> Dict[str, Any]:
    """Benchmark basic KNN search performance."""
    dimension = config.get('dimension', 128)
    num_vectors = config.get('num_vectors', 10000)
    num_queries = config.get('num_queries', 100)
    k = config.get('k', 10)

    # Setup database
    db = _create_db(dimension)
    vectors = _generate_vectors(num_vectors, dimension)
    db.insert_batch(vectors)

    # Generate query vectors
    query_vectors = _generate_vectors(num_queries, dimension)

    # Warm-up
    for _ in range(5):
        db.search(query_vectors[0], k=k)

    # Benchmark
    latencies = []
    for query in query_vectors:
        start = time.perf_counter()
        db.search(query, k=k)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)

    return {
        'mean_latency_ms': np.mean(latencies),
        'median_latency_ms': np.median(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'max_latency_ms': np.max(latencies),
        'qps': 1000 / np.mean(latencies),
        'num_vectors': num_vectors,
        'k': k,
    }


def benchmark_varying_k(config: Dict[str, Any], metrics: MetricsCollector) -> Dict[str, Any]:
    """Benchmark search with different k values."""
    dimension = config.get('dimension', 128)
    num_vectors = config.get('num_vectors', 10000)
    num_queries = config.get('num_queries', 50)
    k_values = config.get('k_values', [1, 10, 50, 100])

    # Setup database
    db = _create_db(dimension)
    vectors = _generate_vectors(num_vectors, dimension)
    db.insert_batch(vectors)

    query_vectors = _generate_vectors(num_queries, dimension)

    results = {}

    for k in k_values:
        # Warm-up
        for _ in range(3):
            db.search(query_vectors[0], k=k)

        # Benchmark
        latencies = []
        for query in query_vectors:
            start = time.perf_counter()
            db.search(query, k=k)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)

        results[f'k_{k}'] = {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'qps': 1000 / np.mean(latencies),
        }

    return results


def benchmark_filtered_search(config: Dict[str, Any], metrics: MetricsCollector) -> Dict[str, Any]:
    """Benchmark search with metadata filtering."""
    dimension = config.get('dimension', 128)
    num_vectors = config.get('num_vectors', 10000)
    num_queries = config.get('num_queries', 50)
    k = config.get('k', 10)

    # Setup database with metadata
    db = _create_db(dimension)
    vectors = _generate_vectors(num_vectors, dimension)
    metadata_list = _generate_metadata(num_vectors)
    db.insert_batch(vectors, metadata_list)

    query_vectors = _generate_vectors(num_queries, dimension)

    # Define filters
    filters = {
        'no_filter': lambda meta: True,
        'category_filter': lambda meta: meta.get('category', '') in ['cat_0', 'cat_1', 'cat_2'],
        'score_filter': lambda meta: meta.get('score', 0) > 50,
    }

    results = {}

    for filter_name, filter_func in filters.items():
        # Warm-up
        for _ in range(3):
            db.search(query_vectors[0], k=k, filter_func=filter_func)

        # Benchmark
        latencies = []
        result_counts = []

        for query in query_vectors:
            start = time.perf_counter()
            search_results = db.search(query, k=k, filter_func=filter_func)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
            result_counts.append(len(search_results))

        results[filter_name] = {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'avg_results_returned': np.mean(result_counts),
            'qps': 1000 / np.mean(latencies),
        }

    return results


def benchmark_recall_accuracy(config: Dict[str, Any], metrics: MetricsCollector) -> Dict[str, Any]:
    """Benchmark recall accuracy compared to brute force."""
    dimension = config.get('dimension', 128)
    num_vectors = config.get('num_vectors', 5000)
    num_queries = config.get('num_queries', 50)
    k = config.get('k', 10)

    # Setup database
    db = _create_db(dimension)
    vectors = _generate_vectors(num_vectors, dimension)
    db.insert_batch(vectors)

    query_vectors = _generate_vectors(num_queries, dimension)

    # Calculate ground truth using brute force
    vectors_array = np.array([vectors[i] for i in range(num_vectors)])

    recall_scores = []
    latencies = []

    for query in query_vectors:
        # Ground truth (brute force)
        distances = np.linalg.norm(vectors_array - query, axis=1)
        ground_truth_indices = np.argsort(distances)[:k].tolist()

        # Database search
        start = time.perf_counter()
        results = db.search(query, k=k)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)

        # Extract indices from results
        retrieved_indices = [idx for idx, _ in results]

        # Calculate recall
        recall = _calculate_recall(retrieved_indices, ground_truth_indices)
        recall_scores.append(recall)

    return {
        'mean_recall': np.mean(recall_scores),
        'min_recall': np.min(recall_scores),
        'max_recall': np.max(recall_scores),
        'std_recall': np.std(recall_scores),
        'mean_latency_ms': np.mean(latencies),
        'num_vectors': num_vectors,
        'k': k,
    }


def benchmark_latency_percentiles(config: Dict[str, Any], metrics: MetricsCollector) -> Dict[str, Any]:
    """Benchmark detailed latency percentiles."""
    dimension = config.get('dimension', 128)
    num_vectors = config.get('num_vectors', 10000)
    num_queries = config.get('num_queries', 1000)
    k = config.get('k', 10)

    # Setup database
    db = _create_db(dimension)
    vectors = _generate_vectors(num_vectors, dimension)
    db.insert_batch(vectors)

    query_vectors = _generate_vectors(num_queries, dimension)

    # Warm-up
    for _ in range(10):
        db.search(query_vectors[0], k=k)

    # Benchmark
    latencies = []
    for query in query_vectors:
        start = time.perf_counter()
        db.search(query, k=k)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)

    percentiles = [50, 90, 95, 99, 99.9]
    percentile_values = np.percentile(latencies, percentiles)

    return {
        'mean_latency_ms': np.mean(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        **{f'p{p}_latency_ms': v for p, v in zip(percentiles, percentile_values)},
        'qps': 1000 / np.mean(latencies),
    }


def run(config: Dict[str, Any]) -> BenchmarkResult:
    """
    Run the search benchmark suite.

    Args:
        config: Configuration dictionary with test parameters

    Returns:
        BenchmarkResult with all test results and metrics
    """
    metrics = MetricsCollector()
    memory_tracker = MemoryTracker()

    suite_name = 'search'
    results = {}
    errors = []

    print(f"\n{'='*60}")
    print("Running Search Benchmark Suite")
    print(f"{'='*60}\n")

    # Track overall suite execution
    suite_start = time.perf_counter()
    memory_tracker.start()

    tests = [
        ('basic_knn', benchmark_basic_knn, 'Basic KNN Search'),
        ('varying_k', benchmark_varying_k, 'Search with Varying K'),
        ('filtered_search', benchmark_filtered_search, 'Filtered Search'),
        ('recall_accuracy', benchmark_recall_accuracy, 'Recall Accuracy'),
        ('latency_percentiles', benchmark_latency_percentiles, 'Latency Percentiles'),
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
