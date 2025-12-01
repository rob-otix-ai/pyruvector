"""
Performance metrics collection and analysis utilities.

Provides timing, memory tracking, percentile calculations, and throughput
measurements for benchmark operations.
"""

import time
import psutil
import statistics
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimingResult:
    """Result of a timing measurement."""

    operation: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory usage percentage
    timestamp: float = field(default_factory=time.time)


class Timer:
    """
    High-precision timer context manager for measuring operation duration.

    Example:
        >>> with Timer() as t:
        ...     expensive_operation()
        >>> print(f"Took {t.duration_ms:.2f}ms")
    """

    def __init__(self) -> None:
        """Initialize timer."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_ms: float = 0.0

    def __enter__(self) -> "Timer":
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and calculate duration."""
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            self.duration_ms = (self.end_time - self.start_time) * 1000.0

    @contextmanager
    @staticmethod
    def measure(operation: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager that yields a TimingResult.

        Args:
            operation: Name of the operation being timed
            metadata: Additional metadata to attach to the result

        Yields:
            TimingResult with the measured duration
        """
        timer = Timer()
        timer.__enter__()
        try:
            yield timer
        finally:
            timer.__exit__(None, None, None)
            TimingResult(
                operation=operation,
                duration_ms=timer.duration_ms,
                metadata=metadata or {}
            )


class MemoryTracker:
    """
    Tracks memory usage of the current process.

    Uses psutil to monitor RSS (Resident Set Size), VMS (Virtual Memory Size),
    and memory percentage.
    """

    def __init__(self) -> None:
        """Initialize memory tracker."""
        self.process = psutil.Process()
        self.baseline: Optional[MemorySnapshot] = None

    def snapshot(self) -> MemorySnapshot:
        """
        Capture current memory usage snapshot.

        Returns:
            MemorySnapshot with current memory metrics
        """
        mem_info = self.process.memory_info()
        mem_percent = self.process.memory_percent()

        return MemorySnapshot(
            rss_mb=mem_info.rss / (1024 * 1024),
            vms_mb=mem_info.vms / (1024 * 1024),
            percent=mem_percent
        )

    def set_baseline(self) -> MemorySnapshot:
        """
        Set baseline memory usage for delta calculations.

        Returns:
            Baseline MemorySnapshot
        """
        self.baseline = self.snapshot()
        return self.baseline

    def delta(self) -> Optional[Dict[str, float]]:
        """
        Calculate memory delta from baseline.

        Returns:
            Dictionary with memory deltas in MB, or None if no baseline set
        """
        if self.baseline is None:
            return None

        current = self.snapshot()
        return {
            "rss_mb": current.rss_mb - self.baseline.rss_mb,
            "vms_mb": current.vms_mb - self.baseline.vms_mb,
            "percent": current.percent - self.baseline.percent
        }

    @contextmanager
    def track(self):
        """
        Context manager for tracking memory delta across an operation.

        Yields:
            Dictionary with memory delta after operation completes
        """
        self.set_baseline()
        yield
        delta = self.delta()
        return delta


class MetricsCollector:
    """
    Collects and analyzes performance metrics.

    Calculates percentiles (p50, p95, p99), throughput, and statistical
    summaries of timing measurements.
    """

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.timings: List[float] = []
        self.metadata: List[Dict[str, Any]] = []

    def record(self, duration_ms: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a timing measurement.

        Args:
            duration_ms: Duration in milliseconds
            metadata: Optional metadata to attach
        """
        self.timings.append(duration_ms)
        self.metadata.append(metadata or {})

    def clear(self) -> None:
        """Clear all recorded metrics."""
        self.timings.clear()
        self.metadata.clear()

    def percentile(self, p: float) -> float:
        """
        Calculate percentile of recorded timings.

        Args:
            p: Percentile to calculate (0-100)

        Returns:
            Percentile value in milliseconds

        Raises:
            ValueError: If no timings recorded
        """
        if not self.timings:
            raise ValueError("No timings recorded")

        sorted_timings = sorted(self.timings)
        index = int((p / 100.0) * len(sorted_timings))
        index = min(index, len(sorted_timings) - 1)
        return sorted_timings[index]

    def percentiles(self, ps: List[float] = [50, 95, 99]) -> Dict[str, float]:
        """
        Calculate multiple percentiles.

        Args:
            ps: List of percentiles to calculate

        Returns:
            Dictionary mapping percentile to value
        """
        return {f"p{int(p)}": self.percentile(p) for p in ps}

    def mean(self) -> float:
        """
        Calculate mean of recorded timings.

        Returns:
            Mean duration in milliseconds

        Raises:
            ValueError: If no timings recorded
        """
        if not self.timings:
            raise ValueError("No timings recorded")
        return statistics.mean(self.timings)

    def median(self) -> float:
        """
        Calculate median of recorded timings.

        Returns:
            Median duration in milliseconds

        Raises:
            ValueError: If no timings recorded
        """
        if not self.timings:
            raise ValueError("No timings recorded")
        return statistics.median(self.timings)

    def stdev(self) -> float:
        """
        Calculate standard deviation of recorded timings.

        Returns:
            Standard deviation in milliseconds

        Raises:
            ValueError: If fewer than 2 timings recorded
        """
        if len(self.timings) < 2:
            raise ValueError("Need at least 2 timings for standard deviation")
        return statistics.stdev(self.timings)

    def min_max(self) -> Dict[str, float]:
        """
        Get minimum and maximum timings.

        Returns:
            Dictionary with min and max values

        Raises:
            ValueError: If no timings recorded
        """
        if not self.timings:
            raise ValueError("No timings recorded")
        return {
            "min": min(self.timings),
            "max": max(self.timings)
        }

    def throughput(self, operations: int, total_time_ms: float) -> float:
        """
        Calculate throughput (operations per second).

        Args:
            operations: Number of operations performed
            total_time_ms: Total time in milliseconds

        Returns:
            Operations per second
        """
        if total_time_ms <= 0:
            return 0.0
        return (operations / total_time_ms) * 1000.0

    def summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistical summary.

        Returns:
            Dictionary with all statistical metrics

        Raises:
            ValueError: If no timings recorded
        """
        if not self.timings:
            raise ValueError("No timings recorded")

        summary = {
            "count": len(self.timings),
            "mean": self.mean(),
            "median": self.median(),
            **self.min_max()
        }

        if len(self.timings) >= 2:
            summary["stdev"] = self.stdev()

        # Add percentiles
        try:
            summary.update(self.percentiles())
        except ValueError:
            pass

        return summary
