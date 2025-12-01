"""
Vector data generation utilities for benchmarking.

Provides random and clustered vector generation to simulate realistic
embedding workloads.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ClusterConfig:
    """Configuration for clustered vector generation."""

    num_clusters: int = 5
    cluster_std: float = 0.1
    cluster_separation: float = 2.0


class VectorGenerator:
    """
    Generates test vectors for benchmarking.

    Supports both random uniform vectors and clustered vectors that simulate
    realistic embedding distributions.
    """

    def __init__(self, dimensions: int, seed: Optional[int] = None) -> None:
        """
        Initialize vector generator.

        Args:
            dimensions: Vector dimensionality
            seed: Random seed for reproducibility
        """
        self.dimensions = dimensions
        self.rng = np.random.RandomState(seed)

    def random(self, count: int, normalize: bool = True) -> np.ndarray:
        """
        Generate random vectors uniformly distributed in [-1, 1].

        Args:
            count: Number of vectors to generate
            normalize: Whether to L2-normalize the vectors

        Returns:
            Array of shape (count, dimensions)
        """
        vectors = self.rng.uniform(-1.0, 1.0, size=(count, self.dimensions))

        if normalize:
            vectors = self._normalize(vectors)

        return vectors.astype(np.float32)

    def clustered(
        self,
        count: int,
        config: Optional[ClusterConfig] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate clustered vectors that simulate realistic embeddings.

        Creates vectors grouped around cluster centers with Gaussian noise,
        simulating the distribution of real-world embedding spaces.

        Args:
            count: Number of vectors to generate
            config: Cluster configuration (uses defaults if None)
            normalize: Whether to L2-normalize the vectors

        Returns:
            Tuple of (vectors, labels) where:
                - vectors: Array of shape (count, dimensions)
                - labels: Array of shape (count,) with cluster assignments
        """
        if config is None:
            config = ClusterConfig()

        # Generate cluster centers
        centers = self._generate_cluster_centers(
            config.num_clusters,
            config.cluster_separation
        )

        # Assign vectors to clusters
        cluster_assignments = self.rng.choice(
            config.num_clusters,
            size=count
        )

        # Generate vectors around cluster centers
        vectors = np.zeros((count, self.dimensions), dtype=np.float32)

        for i in range(count):
            cluster_idx = cluster_assignments[i]
            center = centers[cluster_idx]

            # Add Gaussian noise around cluster center
            noise = self.rng.normal(
                loc=0.0,
                scale=config.cluster_std,
                size=self.dimensions
            )

            vectors[i] = center + noise

        if normalize:
            vectors = self._normalize(vectors)

        return vectors.astype(np.float32), cluster_assignments

    def sequential_queries(
        self,
        database: np.ndarray,
        num_queries: int,
        perturbation: float = 0.05
    ) -> np.ndarray:
        """
        Generate query vectors as perturbed versions of database vectors.

        Useful for testing recall rates with known ground truth.

        Args:
            database: Database vectors to perturb
            num_queries: Number of query vectors to generate
            perturbation: Standard deviation of Gaussian perturbation

        Returns:
            Array of shape (num_queries, dimensions)
        """
        if len(database) == 0:
            raise ValueError("Database cannot be empty")

        # Randomly select vectors from database
        indices = self.rng.choice(len(database), size=num_queries)
        base_vectors = database[indices]

        # Add small perturbation
        noise = self.rng.normal(
            loc=0.0,
            scale=perturbation,
            size=(num_queries, self.dimensions)
        )

        queries = base_vectors + noise
        return self._normalize(queries).astype(np.float32)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2-normalize vectors.

        Args:
            vectors: Input vectors of shape (n, d)

        Returns:
            Normalized vectors of shape (n, d)
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        return vectors / norms

    def _generate_cluster_centers(
        self,
        num_clusters: int,
        separation: float
    ) -> np.ndarray:
        """
        Generate well-separated cluster centers.

        Args:
            num_clusters: Number of cluster centers
            separation: Minimum separation between centers

        Returns:
            Array of shape (num_clusters, dimensions)
        """
        centers = []

        for _ in range(num_clusters):
            if len(centers) == 0:
                # First center at origin with small offset
                center = self.rng.normal(0, 0.1, size=self.dimensions)
            else:
                # Generate center far from existing centers
                max_attempts = 100
                for _ in range(max_attempts):
                    candidate = self.rng.normal(0, separation, size=self.dimensions)

                    # Check distance to existing centers
                    min_dist = min(
                        np.linalg.norm(candidate - c)
                        for c in centers
                    )

                    if min_dist >= separation:
                        center = candidate
                        break
                else:
                    # Fallback: use candidate anyway
                    center = candidate

            centers.append(center)

        return np.array(centers, dtype=np.float32)

    def distribution_info(self, vectors: np.ndarray) -> dict:
        """
        Analyze vector distribution characteristics.

        Args:
            vectors: Input vectors to analyze

        Returns:
            Dictionary with distribution statistics
        """
        norms = np.linalg.norm(vectors, axis=1)

        # Pairwise distances (sample if too many)
        sample_size = min(1000, len(vectors))
        sample_indices = self.rng.choice(len(vectors), size=sample_size, replace=False)
        sample_vectors = vectors[sample_indices]

        # Calculate pairwise distances for sample
        distances = []
        for i in range(len(sample_vectors)):
            for j in range(i + 1, len(sample_vectors)):
                dist = np.linalg.norm(sample_vectors[i] - sample_vectors[j])
                distances.append(dist)

        distances = np.array(distances)

        return {
            "count": len(vectors),
            "dimensions": vectors.shape[1],
            "norm_mean": float(np.mean(norms)),
            "norm_std": float(np.std(norms)),
            "dist_mean": float(np.mean(distances)),
            "dist_std": float(np.std(distances)),
            "dist_min": float(np.min(distances)),
            "dist_max": float(np.max(distances))
        }
