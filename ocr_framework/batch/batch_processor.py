"""Batch processing orchestration for OCR operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from ocr_framework.batch.output_manager import OutputManager
from ocr_framework.batch.progress_reporter import ProgressReporter
from ocr_framework.batch.worker_pool import WorkerPool
from ocr_framework.exceptions import PipelineError
from ocr_framework.models.job import BatchJob, BatchReport
from ocr_framework.pipeline.runner import PipelineRunner
from ocr_framework.types import JobStatus

if TYPE_CHECKING:
    from ocr_framework.config.schema import FrameworkConfig
    from ocr_framework.models.page_result import DocumentResult


logger = logging.getLogger(__name__)


class BatchProcessor:
    """Orchestrate batch OCR processing on multiple files.

    The BatchProcessor manages the complete lifecycle of batch operations,
    including job creation, parallel execution, progress reporting, and
    result aggregation.
    """

    def __init__(
        self,
        pipeline: PipelineRunner,
        config: FrameworkConfig,
        output_dir: Path,
        silent: bool = False,
    ) -> None:
        """Initialize the batch processor.

        Args:
            pipeline: Configured pipeline runner for individual file processing.
            config: Framework configuration.
            output_dir: Target output directory for results.
            silent: If True, suppress console output.
        """
        self._pipeline = pipeline
        self._config = config
        self._output_manager = OutputManager(output_dir, output_format=config.batch.output_format)
        self._progress_reporter = ProgressReporter(silent=silent)
        self._worker_pool = WorkerPool(max_workers=config.batch.workers)

    def process_directory(self, input_dir: Path) -> BatchReport:
        """Process all supported files in a directory.

        Args:
            input_dir: Directory containing input files.

        Returns:
            BatchReport with processing results and statistics.

        Raises:
            PipelineError: If input directory is invalid.
        """
        if not input_dir.exists():
            raise PipelineError(f"Input directory does not exist: {input_dir}")

        if not input_dir.is_dir():
            raise PipelineError(f"Input path is not a directory: {input_dir}")

        # Discover supported files
        jobs = self._discover_jobs(input_dir)

        if not jobs:
            logger.warning(f"No supported files found in {input_dir}")
            return BatchReport()

        # Process jobs
        return self._process_jobs(jobs)

    def process_files(self, input_files: list[Path]) -> BatchReport:
        """Process a specific list of files.

        Args:
            input_files: List of input file paths.

        Returns:
            BatchReport with processing results and statistics.
        """
        # Create jobs from file list
        jobs = [
            BatchJob(
                job_id=str(uuid4()),
                input_path=file_path,
            )
            for file_path in input_files
        ]

        return self._process_jobs(jobs)

    def _discover_jobs(self, input_dir: Path) -> list[BatchJob]:
        """Discover supported files in input directory.

        Args:
            input_dir: Directory to scan for files.

        Returns:
            List of BatchJob objects for discovered files.
        """
        jobs = []
        supported_extensions = {
            ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".pdf",
        }

        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                jobs.append(
                    BatchJob(
                        job_id=str(uuid4()),
                        input_path=file_path,
                    )
                )

        # Sort for deterministic processing order
        jobs.sort(key=lambda job: job.input_path.name)

        return jobs

    def _process_jobs(self, jobs: list[BatchJob]) -> BatchReport:
        """Process batch jobs with configured worker pool.

        Args:
            jobs: List of batch jobs to process.

        Returns:
            BatchReport with processing results.
        """
        self._progress_reporter.start(len(jobs))

        # Execute jobs in parallel
        processed_jobs = self._worker_pool.execute(
            jobs=jobs,
            worker_func=self._process_single_job,
            continue_on_error=self._config.batch.continue_on_error,
        )

        # Build report
        report = BatchReport(jobs=processed_jobs)

        self._progress_reporter.complete(report.succeeded, report.failed)

        return report

    def _process_single_job(self, job: BatchJob) -> BatchJob:
        """Process a single batch job.

        Args:
            job: Batch job to process.

        Returns:
            Updated batch job with processing results.
        """
        job.status = JobStatus.RUNNING

        try:
            # Run OCR pipeline
            result = self._pipeline.run(job.input_path)

            # Export result
            output_path = self._output_manager.get_output_path(job.input_path)
            self._export_result(result, output_path)

            # Update job status
            job.status = JobStatus.COMPLETED
            job.output_path = output_path

        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            logger.error(f"Failed to process {job.input_path}: {exc}")

        finally:
            # Report progress
            self._progress_reporter.update(job)

        return job

    def _export_result(self, result: DocumentResult, output_path: Path) -> None:
        """Export OCR result to file.

        Args:
            result: Document result to export.
            output_path: Target output path.

        Raises:
            PipelineError: If export fails.
        """
        try:
            from ocr_framework.exporters.txt_exporter import TXTExporter

            exporter = TXTExporter()
            exporter.export(result, output_path)
        except Exception as exc:
            raise PipelineError(f"Failed to export result to {output_path}: {exc}") from exc
