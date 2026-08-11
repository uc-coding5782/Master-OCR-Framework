"""Benchmark system for OCR performance evaluation."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    duration_seconds: float
    memory_mb: float
    success: bool
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""

    results: list[BenchmarkResult] = field(default_factory=list)
    total_duration: float = 0.0
    average_duration: float = 0.0
    success_rate: float = 0.0

    def __post_init__(self) -> None:
        """Calculate aggregate statistics."""
        if self.results:
            self.total_duration = sum(r.duration_seconds for r in self.results)
            self.average_duration = self.total_duration / len(self.results)
            successful = sum(1 for r in self.results if r.success)
            self.success_rate = successful / len(self.results)


class Benchmark:
    """Benchmark OCR pipeline performance and resource usage."""

    def __init__(self) -> None:
        """Initialize the benchmark system."""
        self._results: list[BenchmarkResult] = []

    def run(
        self,
        name: str,
        func: Callable[[], any],
        iterations: int = 1,
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Args:
            name: Name of the benchmark.
            func: Function to benchmark.
            iterations: Number of iterations to run.

        Returns:
            BenchmarkResult with timing and memory metrics.
        """
        import tracemalloc

        # Start memory tracking
        tracemalloc.start()

        try:
            # Time the execution
            start_time = time.time()

            for _ in range(iterations):
                result = func()

            duration = time.time() - start_time

            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            memory_mb = peak / (1024 * 1024)

            benchmark_result = BenchmarkResult(
                name=name,
                duration_seconds=duration,
                memory_mb=memory_mb,
                success=True,
                metadata={"iterations": iterations},
            )

        except Exception as exc:
            benchmark_result = BenchmarkResult(
                name=name,
                duration_seconds=0.0,
                memory_mb=0.0,
                success=False,
                error=str(exc),
            )

        finally:
            tracemalloc.stop()

        self._results.append(benchmark_result)
        return benchmark_result

    def run_batch(
        self,
        benchmarks: list[tuple[str, Callable[[], any], int]],
    ) -> BenchmarkReport:
        """Run multiple benchmarks.

        Args:
            benchmarks: List of (name, func, iterations) tuples.

        Returns:
            BenchmarkReport with aggregated results.
        """
        for name, func, iterations in benchmarks:
            self.run(name, func, iterations)

        return BenchmarkReport(results=list(self._results))

    def clear(self) -> None:
        """Clear all benchmark results."""
        self._results.clear()

    def get_report(self) -> BenchmarkReport:
        """Get current benchmark report.

        Returns:
            BenchmarkReport with aggregated statistics.
        """
        return BenchmarkReport(results=list(self._results))
