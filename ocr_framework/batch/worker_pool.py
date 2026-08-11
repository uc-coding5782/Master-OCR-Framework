"""Worker pool for parallel batch processing."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from typing import TYPE_CHECKING

from ocr_framework.types import JobStatus

if TYPE_CHECKING:
    from ocr_framework.models.job import BatchJob


class WorkerPool:
    """Manage parallel execution of batch jobs using a thread pool.

    The WorkerPool provides controlled parallelism for batch processing,
    allowing configuration of worker count and error handling behavior.
    """

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize the worker pool.

        Args:
            max_workers: Maximum number of worker threads.
        """
        self._max_workers = max_workers

    def execute(
        self,
        jobs: list[BatchJob],
        worker_func: Callable[[BatchJob], BatchJob],
        continue_on_error: bool = True,
    ) -> list[BatchJob]:
        """Execute batch jobs in parallel.

        Args:
            jobs: List of batch jobs to execute.
            worker_func: Function to execute for each job.
            continue_on_error: If True, continue processing other jobs if one fails.

        Returns:
            List of processed batch jobs with updated status.
        """
        if self._max_workers <= 1:
            # Sequential execution for single worker
            return self._execute_sequential(jobs, worker_func, continue_on_error)
        else:
            # Parallel execution for multiple workers
            return self._execute_parallel(jobs, worker_func, continue_on_error)

    def _execute_sequential(
        self,
        jobs: list[BatchJob],
        worker_func: Callable[[BatchJob], BatchJob],
        continue_on_error: bool,
    ) -> list[BatchJob]:
        """Execute jobs sequentially.

        Args:
            jobs: List of batch jobs to execute.
            worker_func: Function to execute for each job.
            continue_on_error: If True, continue processing other jobs if one fails.

        Returns:
            List of processed batch jobs.
        """
        results = []
        for job in jobs:
            try:
                result = worker_func(job)
                results.append(result)
            except Exception as exc:
                if not continue_on_error:
                    raise
                # Mark job as failed and continue
                job.status = JobStatus.FAILED
                job.error = str(exc)
                results.append(job)
        return results

    def _execute_parallel(
        self,
        jobs: list[BatchJob],
        worker_func: Callable[[BatchJob], BatchJob],
        continue_on_error: bool,
    ) -> list[BatchJob]:
        """Execute jobs in parallel using thread pool.

        Args:
            jobs: List of batch jobs to execute.
            worker_func: Function to execute for each job.
            continue_on_error: If True, continue processing other jobs if one fails.

        Returns:
            List of processed batch jobs.
        """
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(worker_func, job): job for job in jobs
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    if not continue_on_error:
                        # Cancel remaining futures and re-raise
                        for f in future_to_job:
                            f.cancel()
                        raise
                    # Mark job as failed and continue
                    job.status = JobStatus.FAILED
                    job.error = str(exc)
                    results.append(job)

        return results
