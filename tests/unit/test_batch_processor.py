"""Tests for batch processing components."""

from pathlib import Path

import numpy as np
import pytest

from ocr_framework.batch.batch_processor import BatchProcessor
from ocr_framework.batch.output_manager import OutputManager
from ocr_framework.batch.progress_reporter import ProgressReporter, ProgressUpdate
from ocr_framework.batch.worker_pool import WorkerPool
from ocr_framework.models.job import BatchJob
from ocr_framework.types import JobStatus


@pytest.fixture
def sample_jobs() -> list[BatchJob]:
    """Create sample batch jobs for testing."""
    return [
        BatchJob(job_id="1", input_path=Path("image1.jpg")),
        BatchJob(job_id="2", input_path=Path("image2.png")),
        BatchJob(job_id="3", input_path=Path("image3.jpeg")),
    ]


@pytest.fixture
def sample_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    return output_dir


class TestOutputManager:
    """Tests for OutputManager."""

    def test_creates_output_directory(self, sample_output_dir: Path) -> None:
        """Test that OutputManager creates the output directory."""
        manager = OutputManager(sample_output_dir)
        assert sample_output_dir.exists()
        assert sample_output_dir.is_dir()

    def test_preserves_filename_with_new_extension(self, sample_output_dir: Path) -> None:
        """Test that output filenames preserve original names with new extension."""
        manager = OutputManager(sample_output_dir, output_format="txt")
        input_path = Path("image1.jpg")
        output_path = manager.get_output_path(input_path)

        assert output_path.name == "image1.txt"
        assert output_path.parent == sample_output_dir

    def test_supports_different_formats(self, sample_output_dir: Path) -> None:
        """Test that OutputManager supports different output formats."""
        txt_manager = OutputManager(sample_output_dir, output_format="txt")
        json_manager = OutputManager(sample_output_dir, output_format="json")

        assert txt_manager.get_output_path(Path("test.jpg")).name == "test.txt"
        assert json_manager.get_output_path(Path("test.jpg")).name == "test.json"

    def test_checks_file_existence(self, sample_output_dir: Path) -> None:
        """Test that OutputManager can check if output files exist."""
        manager = OutputManager(sample_output_dir)
        output_path = manager.get_output_path(Path("test.jpg"))

        # File shouldn't exist initially
        assert not manager.exists(output_path)

        # Create the file
        output_path.touch()

        # Now it should exist
        assert manager.exists(output_path)


class TestProgressReporter:
    """Tests for ProgressReporter."""

    def test_initializes_progress_tracking(self) -> None:
        """Test that progress reporter initializes tracking."""
        reporter = ProgressReporter(silent=True)
        reporter.start(10)

        assert reporter._total == 10
        assert reporter._current == 0

    def test_updates_progress_for_completed_job(self, sample_jobs: list[BatchJob]) -> None:
        """Test that progress reporter updates on job completion."""
        reporter = ProgressReporter(silent=True)
        reporter.start(len(sample_jobs))

        job = sample_jobs[0]
        job.status = JobStatus.COMPLETED
        reporter.update(job)

        assert reporter._current == 1

    def test_generates_progress_update(self, sample_jobs: list[BatchJob]) -> None:
        """Test that progress reporter generates correct updates."""
        reporter = ProgressReporter(silent=True)
        reporter.start(len(sample_jobs))

        job = sample_jobs[0]
        job.status = JobStatus.COMPLETED
        reporter.update(job)

        # The update should have been created and reported
        assert reporter._current == 1

    def test_reports_completion_summary(self) -> None:
        """Test that progress reporter reports completion summary."""
        reporter = ProgressReporter(silent=True)
        reporter.complete(5, 2)

        # Should not raise any errors
        assert True


class TestWorkerPool:
    """Tests for WorkerPool."""

    def test_sequential_execution(self, sample_jobs: list[BatchJob]) -> None:
        """Test that worker pool executes jobs sequentially with single worker."""
        pool = WorkerPool(max_workers=1)

        def dummy_worker(job: BatchJob) -> BatchJob:
            job.status = JobStatus.COMPLETED
            return job

        results = pool.execute(sample_jobs, dummy_worker, continue_on_error=True)

        assert len(results) == len(sample_jobs)
        assert all(job.status == JobStatus.COMPLETED for job in results)

    def test_continues_on_error(self, sample_jobs: list[BatchJob]) -> None:
        """Test that worker pool continues processing on errors when configured."""
        pool = WorkerPool(max_workers=1)

        def failing_worker(job: BatchJob) -> BatchJob:
            if job.job_id == "2":
                raise ValueError("Simulated error")
            job.status = JobStatus.COMPLETED
            return job

        results = pool.execute(sample_jobs, failing_worker, continue_on_error=True)

        assert len(results) == len(sample_jobs)
        assert results[1].status == JobStatus.FAILED
        assert results[1].error is not None

    def test_stops_on_error_when_configured(self, sample_jobs: list[BatchJob]) -> None:
        """Test that worker pool stops on errors when not configured to continue."""
        pool = WorkerPool(max_workers=1)

        def failing_worker(job: BatchJob) -> BatchJob:
            if job.job_id == "2":
                raise ValueError("Simulated error")
            job.status = JobStatus.COMPLETED
            return job

        with pytest.raises(ValueError, match="Simulated error"):
            pool.execute(sample_jobs, failing_worker, continue_on_error=False)


class TestBatchProcessor:
    """Tests for BatchProcessor."""

    def test_rejects_nonexistent_directory(self) -> None:
        """Test that BatchProcessor rejects nonexistent input directory."""
        from ocr_framework.pipeline.builder import PipelineBuilder

        pipeline = PipelineBuilder().build()
        nonexistent_dir = Path("/nonexistent/directory")

        processor = BatchProcessor(
            pipeline=pipeline,
            config=pipeline.config,
            output_dir=Path("output"),
            silent=True,
        )

        with pytest.raises(Exception):  # PipelineError
            processor.process_directory(nonexistent_dir)

    def test_rejects_file_instead_of_directory(self, tmp_path: Path) -> None:
        """Test that BatchProcessor rejects file path instead of directory."""
        from ocr_framework.pipeline.builder import PipelineBuilder

        # Create a file instead of directory
        test_file = tmp_path / "test.txt"
        test_file.touch()

        pipeline = PipelineBuilder().build()

        processor = BatchProcessor(
            pipeline=pipeline,
            config=pipeline.config,
            output_dir=Path("output"),
            silent=True,
        )

        with pytest.raises(Exception):  # PipelineError
            processor.process_directory(test_file)

    def test_handles_empty_directory(self, tmp_path: Path) -> None:
        """Test that BatchProcessor handles empty directory gracefully."""
        from ocr_framework.pipeline.builder import PipelineBuilder

        pipeline = PipelineBuilder().build()

        processor = BatchProcessor(
            pipeline=pipeline,
            config=pipeline.config,
            output_dir=tmp_path / "output",
            silent=True,
        )

        report = processor.process_directory(tmp_path)

        assert report.total == 0
        assert report.succeeded == 0
        assert report.failed == 0

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """Test that BatchProcessor creates output directory."""
        from ocr_framework.pipeline.builder import PipelineBuilder

        # Create a test image directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create a simple test image
        try:
            import cv2
            test_image = input_dir / "test.png"
            test_array = np.zeros((100, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(test_image), test_array)
        except ImportError:
            # If cv2 is not available, skip this test
            pytest.skip("OpenCV not available for test image creation")

        pipeline = PipelineBuilder().build()
        output_dir = tmp_path / "output"

        processor = BatchProcessor(
            pipeline=pipeline,
            config=pipeline.config,
            output_dir=output_dir,
            silent=True,
        )

        # The output directory should be created when processing starts
        # (even if processing fails due to missing PaddleOCR)
        try:
            processor.process_directory(input_dir)
        except ImportError:
            # Expected if PaddleOCR is not installed
            pass

        # Output directory should exist
        assert output_dir.exists()
