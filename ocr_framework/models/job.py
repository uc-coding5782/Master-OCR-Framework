"""Batch job and reporting models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ocr_framework.types import JobStatus, Metadata


@dataclass
class BatchJob:
    """A single unit of work within a batch OCR run."""

    job_id: str
    input_path: Path
    status: JobStatus = JobStatus.PENDING
    output_path: Path | None = None
    error: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass
class BatchReport:
    """Summary report for a completed batch run."""

    jobs: list[BatchJob] = field(default_factory=list)
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total == 0:
            self.total = len(self.jobs)
        if self.succeeded == 0 and self.failed == 0 and self.jobs:
            self.succeeded = sum(
                1 for job in self.jobs if job.status == JobStatus.COMPLETED
            )
            self.failed = sum(
                1 for job in self.jobs if job.status == JobStatus.FAILED
            )
