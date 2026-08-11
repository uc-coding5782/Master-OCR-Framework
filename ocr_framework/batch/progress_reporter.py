"""Progress reporting for batch processing operations.

Provides console progress bars via tqdm when available, with a plain-text
fallback. Emits structured progress updates as jobs complete.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ocr_framework.types import JobStatus

if TYPE_CHECKING:
    from ocr_framework.models.job import BatchJob

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


@dataclass
class ProgressUpdate:
    """Progress update for a single batch job."""

    job_id: str
    filename: str
    status: JobStatus
    current: int
    total: int
    elapsed_seconds: float
    error: str | None = None


class ProgressReporter:
    """Report progress during batch processing operations.

    The ProgressReporter provides real-time updates on batch job progress,
    including a progress bar (via tqdm when available), completion
    percentage, elapsed time, and individual job status.
    """

    def __init__(self, silent: bool = False) -> None:
        """Initialize the progress reporter.

        Args:
            silent: If True, suppress console output.
        """
        self._silent = silent
        self._start_time: float | None = None
        self._current: int = 0
        self._total: int = 0
        self._pbar: "tqdm | None" = None

    def start(self, total: int) -> None:
        """Initialize progress tracking for a batch operation.

        Args:
            total: Total number of jobs to process.
        """
        self._start_time = time.time()
        self._current = 0
        self._total = total

        if self._silent:
            return

        if TQDM_AVAILABLE and total > 0:
            self._pbar = tqdm(
                total=total,
                desc="Batch OCR",
                unit="file",
                leave=True,
            )
        else:
            print(f"Starting batch processing: {total} files")
            print("-" * 50)

    def update(self, job: "BatchJob") -> None:
        """Report progress for a completed job.

        Args:
            job: The batch job that was just processed.
        """
        self._current += 1

        if self._start_time is None:
            elapsed = 0.0
        else:
            elapsed = time.time() - self._start_time

        update = ProgressUpdate(
            job_id=job.job_id,
            filename=job.input_path.name,
            status=job.status,
            current=self._current,
            total=self._total,
            elapsed_seconds=elapsed,
            error=job.error,
        )

        self._report(update)

    def _report(self, update: ProgressUpdate) -> None:
        """Generate a progress report.

        Args:
            update: Progress update to report.
        """
        if self._silent:
            return

        if self._pbar is not None:
            # Update tqdm progress bar with status postfix
            self._pbar.set_postfix_str(
                f"{update.status.value}: {update.filename}"
            )
            self._pbar.update(1)
            if update.error:
                self._pbar.write(f"  Error: {update.error}")
        else:
            percentage = (update.current / update.total) * 100 if update.total > 0 else 0.0
            status_emoji = self._get_status_emoji(update.status)

            print(
                f"[{update.current}/{update.total}] "
                f"{percentage:5.1f}% | "
                f"{status_emoji} {update.filename} | "
                f"{update.status.value} | "
                f"{update.elapsed_seconds:.1f}s"
            )

            if update.error:
                print(f"  Error: {update.error}")

    def _get_status_emoji(self, status: JobStatus) -> str:
        """Get emoji representation for job status.

        Args:
            status: Job status.

        Returns:
            Emoji character representing the status.
        """
        emoji_map = {
            JobStatus.PENDING: "⏳",
            JobStatus.RUNNING: "🔄",
            JobStatus.COMPLETED: "✅",
            JobStatus.FAILED: "❌",
        }
        return emoji_map.get(status, "❓")

    def complete(self, succeeded: int, failed: int) -> None:
        """Report batch completion summary.

        Args:
            succeeded: Number of successful jobs.
            failed: Number of failed jobs.
        """
        if self._silent:
            return

        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        if self._start_time is not None:
            elapsed = time.time() - self._start_time
        else:
            elapsed = 0.0

        print("-" * 50)
        print(f"Batch processing complete in {elapsed:.1f}s")
        print(f"  Succeeded: {succeeded}")
        print(f"  Failed: {failed}")
        print(f"  Total: {succeeded + failed}")