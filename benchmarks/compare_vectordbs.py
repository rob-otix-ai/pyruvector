#!/usr/bin/env python3
"""
Vector Database Comparison Benchmark

Compares pyruvector against other vector databases (Qdrant if available).
Tests insertion throughput, search latency, memory usage, and recall accuracy.
"""

import time
import json
import sys
import traceback
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Fail-fast imports - no mocks, no fallbacks
try:
    from pyruvector import VectorDB  # noqa: F401 - DistanceMetric, HNSWConfig, DbOptions available for future use
except ImportError as e:
    print("Error: pyruvector must be built before running benchmarks.")
    print("Run: maturin develop --release")
    print(f"Original error: {e}")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required for benchmarks")
    print("Install with: pip install numpy")
    sys.exit(1)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    db_name: str
    vector_count: int
    dimensions: int
    batch_size: int
    k: int  # Number of neighbors to search for
    warmup_queries: int = 10
    test_queries: int = 100
    iterations: int = 3


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    db_name: str
    vector_count: int
    dimensions: int
    batch_size: int

    # Insertion metrics
    insert_throughput: float  # vectors/second
    insert_time: float  # seconds

    # Search metrics
    search_latency_k10: float  # milliseconds
    search_latency_k100: float  # milliseconds

    # Memory metrics
    memory_usage_mb: float

    # Accuracy metrics
    recall_at_10: float
    recall_at_100: float

    # Additional info
    config: Dict[str, Any]
    error: Optional[str] = None


class VectorGenerator:
    """Generates clustered test vectors for realistic benchmarks."""

    def __init__(self, dimensions: int, num_clusters: int = 10, seed: int = 42):
        self.dimensions = dimensions
        self.num_clusters = num_clusters
        self.rng = np.random.RandomState(seed)

        # Generate cluster centers
        self.centers = self.rng.randn(num_clusters, dimensions).astype(np.float32)
        # Normalize centers
        norms = np.linalg.norm(self.centers, axis=1, keepdims=True)
        self.centers = self.centers / (norms + 1e-8)

    def generate_batch(self, count: int) -> np.ndarray:
        """Generate a batch of clustered vectors."""
        # Assign each vector to a random cluster
        cluster_assignments = self.rng.randint(0, self.num_clusters, count)

        # Generate vectors around cluster centers with some noise
        vectors = np.zeros((count, self.dimensions), dtype=np.float32)
        for i in range(count):
            center = self.centers[cluster_assignments[i]]
            noise = self.rng.randn(self.dimensions).astype(np.float32) * 0.1
            vectors[i] = center + noise

        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-8)

        return vectors

    def generate_query(self) -> np.ndarray:
        """Generate a single query vector."""
        # Query is a random cluster center with small noise
        center_idx = self.rng.randint(0, self.num_clusters)
        center = self.centers[center_idx]
        noise = self.rng.randn(self.dimensions).astype(np.float32) * 0.05
        query = center + noise

        # Normalize
        query = query / (np.linalg.norm(query) + 1e-8)
        return query


class PyruVectorBenchmark:
    """Benchmark implementation for pyruvector."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.db = None
        self.generator = VectorGenerator(config.dimensions)
        self.inserted_vectors = []
        self.ground_truth = {}

    def setup(self):
        """Initialize the database."""
        self.db = VectorDB(dimensions=self.config.dimensions)

    def insert_vectors(self) -> float:
        """Insert vectors and return time taken."""
        start_time = time.time()

        remaining = self.config.vector_count
        vector_id = 0

        while remaining > 0:
            batch_size = min(self.config.batch_size, remaining)
            vectors = self.generator.generate_batch(batch_size)

            for vec in vectors:
                self.db.insert(
                    f"vec_{vector_id}",
                    vec.tolist(),
                    {"id": vector_id, "batch": vector_id // self.config.batch_size}
                )
                self.inserted_vectors.append(vec)
                vector_id += 1

            remaining -= batch_size

        return time.time() - start_time

    def warmup(self):
        """Perform warmup queries."""
        for _ in range(self.config.warmup_queries):
            query = self.generator.generate_query()
            self.db.search(query.tolist(), k=10)

    def compute_ground_truth(self, query: np.ndarray, k: int) -> List[int]:
        """Compute ground truth nearest neighbors using brute force."""
        # Compute cosine similarity with all vectors
        similarities = np.dot(self.inserted_vectors, query)
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices.tolist()

    def benchmark_search(self, k: int) -> Tuple[float, float]:
        """Benchmark search and return (latency_ms, recall)."""
        latencies = []
        recalls = []

        for _ in range(self.config.test_queries):
            query = self.generator.generate_query()

            # Compute ground truth if not cached
            query_key = query.tobytes()
            if query_key not in self.ground_truth:
                self.ground_truth[query_key] = self.compute_ground_truth(query, k)

            ground_truth = set(self.ground_truth[query_key])

            # Time the search
            start = time.perf_counter()
            results = self.db.search(query.tolist(), k=k)
            latency = (time.perf_counter() - start) * 1000  # Convert to ms

            latencies.append(latency)

            # Compute recall
            result_ids = [int(r.id.split('_')[1]) for r in results]
            recall = len(set(result_ids) & ground_truth) / k
            recalls.append(recall)

        return np.mean(latencies), np.mean(recalls)

    def get_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Rough estimation based on vector count and dimensions
        vector_size_bytes = self.config.dimensions * 4  # float32
        metadata_overhead = 100  # bytes per vector (rough estimate)
        hnsw_overhead = 200  # bytes per vector for HNSW graph (rough estimate)

        total_bytes = self.config.vector_count * (
            vector_size_bytes + metadata_overhead + hnsw_overhead
        )

        return total_bytes / (1024 * 1024)  # Convert to MB

    def cleanup(self):
        """Clean up resources."""
        self.db = None
        self.inserted_vectors = []
        self.ground_truth = {}


class QdrantBenchmark:
    """Benchmark implementation for Qdrant (if available)."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.client = None
        self.collection_name = "benchmark_collection"
        self.generator = VectorGenerator(config.dimensions)
        self.inserted_vectors = []
        self.ground_truth = {}

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            self.QdrantClient = QdrantClient
            self.Distance = Distance
            self.VectorParams = VectorParams
            self.PointStruct = PointStruct
            self.available = True
        except ImportError:
            self.available = False

    def setup(self):
        """Initialize Qdrant."""
        if not self.available:
            raise RuntimeError("Qdrant not available")

        self.client = self.QdrantClient(":memory:")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.VectorParams(
                size=self.config.dimensions,
                distance=self.Distance.COSINE,
            ),
        )

    def insert_vectors(self) -> float:
        """Insert vectors and return time taken."""
        start_time = time.time()

        remaining = self.config.vector_count
        vector_id = 0

        while remaining > 0:
            batch_size = min(self.config.batch_size, remaining)
            vectors = self.generator.generate_batch(batch_size)

            points = []
            for i, vec in enumerate(vectors):
                self.inserted_vectors.append(vec)
                points.append(
                    self.PointStruct(
                        id=vector_id,
                        vector=vec.tolist(),
                        payload={"id": vector_id, "batch": vector_id // self.config.batch_size}
                    )
                )
                vector_id += 1

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            remaining -= batch_size

        return time.time() - start_time

    def warmup(self):
        """Perform warmup queries."""
        for _ in range(self.config.warmup_queries):
            query = self.generator.generate_query()
            self.client.search(
                collection_name=self.collection_name,
                query_vector=query.tolist(),
                limit=10,
            )

    def compute_ground_truth(self, query: np.ndarray, k: int) -> List[int]:
        """Compute ground truth nearest neighbors using brute force."""
        similarities = np.dot(self.inserted_vectors, query)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices.tolist()

    def benchmark_search(self, k: int) -> Tuple[float, float]:
        """Benchmark search and return (latency_ms, recall)."""
        latencies = []
        recalls = []

        for _ in range(self.config.test_queries):
            query = self.generator.generate_query()

            # Compute ground truth if not cached
            query_key = query.tobytes()
            if query_key not in self.ground_truth:
                self.ground_truth[query_key] = self.compute_ground_truth(query, k)

            ground_truth = set(self.ground_truth[query_key])

            # Time the search
            start = time.perf_counter()
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query.tolist(),
                limit=k,
            )
            latency = (time.perf_counter() - start) * 1000

            latencies.append(latency)

            # Compute recall
            result_ids = [r.id for r in results]
            recall = len(set(result_ids) & ground_truth) / k
            recalls.append(recall)

        return np.mean(latencies), np.mean(recalls)

    def get_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Similar estimation as pyruvector
        vector_size_bytes = self.config.dimensions * 4
        metadata_overhead = 100
        hnsw_overhead = 200

        total_bytes = self.config.vector_count * (
            vector_size_bytes + metadata_overhead + hnsw_overhead
        )

        return total_bytes / (1024 * 1024)

    def cleanup(self):
        """Clean up resources."""
        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
        self.client = None
        self.inserted_vectors = []
        self.ground_truth = {}


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    print(f"\n{'='*80}")
    print(f"Benchmarking {config.db_name}")
    print(f"Vectors: {config.vector_count:,} | Dimensions: {config.dimensions} | Batch: {config.batch_size}")
    print(f"{'='*80}")

    # Select benchmark implementation
    if config.db_name == "pyruvector":
        benchmark = PyruVectorBenchmark(config)
    elif config.db_name == "qdrant":
        benchmark = QdrantBenchmark(config)
        if not benchmark.available:
            return BenchmarkResult(
                db_name=config.db_name,
                vector_count=config.vector_count,
                dimensions=config.dimensions,
                batch_size=config.batch_size,
                insert_throughput=0.0,
                insert_time=0.0,
                search_latency_k10=0.0,
                search_latency_k100=0.0,
                memory_usage_mb=0.0,
                recall_at_10=0.0,
                recall_at_100=0.0,
                config={},
                error="Qdrant not installed (install with: pip install qdrant-client)",
            )
    else:
        raise ValueError(f"Unknown database: {config.db_name}")

    try:
        # Setup
        print("Setting up database...")
        benchmark.setup()

        # Insert vectors
        print(f"Inserting {config.vector_count:,} vectors...")
        insert_time = benchmark.insert_vectors()
        insert_throughput = config.vector_count / insert_time
        print(f"  ✓ Inserted in {insert_time:.2f}s ({insert_throughput:,.0f} vectors/sec)")

        # Warmup
        print(f"Warming up with {config.warmup_queries} queries...")
        benchmark.warmup()

        # Benchmark search with k=10
        print(f"Benchmarking search (k=10) with {config.test_queries} queries...")
        latency_k10, recall_k10 = benchmark.benchmark_search(k=10)
        print(f"  ✓ Latency: {latency_k10:.2f}ms | Recall: {recall_k10:.1%}")

        # Benchmark search with k=100
        print(f"Benchmarking search (k=100) with {config.test_queries} queries...")
        latency_k100, recall_k100 = benchmark.benchmark_search(k=100)
        print(f"  ✓ Latency: {latency_k100:.2f}ms | Recall: {recall_k100:.1%}")

        # Memory usage
        memory_mb = benchmark.get_memory_usage()
        print(f"Memory usage: ~{memory_mb:.1f} MB")

        # Cleanup
        benchmark.cleanup()

        return BenchmarkResult(
            db_name=config.db_name,
            vector_count=config.vector_count,
            dimensions=config.dimensions,
            batch_size=config.batch_size,
            insert_throughput=insert_throughput,
            insert_time=insert_time,
            search_latency_k10=latency_k10,
            search_latency_k100=latency_k100,
            memory_usage_mb=memory_mb,
            recall_at_10=recall_k10,
            recall_at_100=recall_k100,
            config=asdict(config),
        )

    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return BenchmarkResult(
            db_name=config.db_name,
            vector_count=config.vector_count,
            dimensions=config.dimensions,
            batch_size=config.batch_size,
            insert_throughput=0.0,
            insert_time=0.0,
            search_latency_k10=0.0,
            search_latency_k100=0.0,
            memory_usage_mb=0.0,
            recall_at_10=0.0,
            recall_at_100=0.0,
            config=asdict(config),
            error=str(e),
        )


def print_results_table(results: List[BenchmarkResult]):
    """Print results in a formatted table."""
    print("\n" + "="*120)
    print("BENCHMARK RESULTS")
    print("="*120)

    # Header
    header = (
        f"{'Database':<12} | "
        f"{'Vectors':<10} | "
        f"{'Dims':<6} | "
        f"{'Insert (v/s)':<13} | "
        f"{'Search k=10':<12} | "
        f"{'Search k=100':<13} | "
        f"{'Recall@10':<10} | "
        f"{'Memory (MB)':<12}"
    )
    print(header)
    print("-" * 120)

    # Rows
    for r in results:
        if r.error:
            row = f"{r.db_name:<12} | {r.vector_count:>10,} | {r.dimensions:>6} | ERROR: {r.error}"
        else:
            row = (
                f"{r.db_name:<12} | "
                f"{r.vector_count:>10,} | "
                f"{r.dimensions:>6} | "
                f"{r.insert_throughput:>13,.0f} | "
                f"{r.search_latency_k10:>11.2f}ms | "
                f"{r.search_latency_k100:>12.2f}ms | "
                f"{r.recall_at_10:>9.1%} | "
                f"{r.memory_usage_mb:>11.1f}"
            )
        print(row)

    print("="*120)


def save_results_json(results: List[BenchmarkResult], output_file: Path):
    """Save results to JSON file."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [asdict(r) for r in results],
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def run_comparison(
    databases: List[str],
    vector_counts: List[int],
    dimensions_list: List[int],
    batch_sizes: List[int],
    output_file: Optional[Path] = None,
):
    """Run comprehensive comparison across all configurations."""
    all_results = []

    for db_name in databases:
        for vector_count in vector_counts:
            for dimensions in dimensions_list:
                for batch_size in batch_sizes:
                    config = BenchmarkConfig(
                        db_name=db_name,
                        vector_count=vector_count,
                        dimensions=dimensions,
                        batch_size=batch_size,
                        k=10,
                        warmup_queries=10,
                        test_queries=100,
                        iterations=3,
                    )

                    result = run_benchmark(config)
                    all_results.append(result)

    # Print summary table
    print_results_table(all_results)

    # Save to JSON if requested
    if output_file:
        save_results_json(all_results, output_file)

    return all_results


if __name__ == "__main__":
    # Example usage
    results = run_comparison(
        databases=["pyruvector", "qdrant"],
        vector_counts=[10000, 50000],
        dimensions_list=[128, 384],
        batch_sizes=[100, 1000],
        output_file=Path("benchmark_results.json"),
    )
