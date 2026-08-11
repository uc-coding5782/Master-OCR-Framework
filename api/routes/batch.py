"""Batch OCR endpoint."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.dependencies import get_pipeline
from api.schemas import BatchJobResponse, BatchResponse
from ocr_framework import create_batch_processor
from ocr_framework.exceptions import OCRFrameworkError

router = APIRouter(tags=["Batch"])


@router.post("/ocr/batch", response_model=BatchResponse)
async def run_batch(
    files: list[UploadFile] = File(..., description="Files to process"),
    language: str = Form(default="en", description="OCR language code"),
    workers: int = Form(default=2, ge=1, le=16, description="Parallel workers"),
) -> BatchResponse:
    """Process multiple uploaded files in batch.

    Accepts a collection of image or PDF files and processes them in
    parallel using the configured worker count. Returns a summary with
    per-file status and any errors encountered.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    pipeline = get_pipeline()
    pipeline.config.language = language
    pipeline.config.batch.workers = workers

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = Path(tmp_dir) / "input"
        output_dir = Path(tmp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_paths: list[Path] = []
        for upload in files:
            suffix = Path(upload.filename or "file.png").suffix.lower()
            dest = input_dir / (upload.filename or f"file{suffix}")
            dest.write_bytes(await upload.read())
            input_paths.append(dest)

        processor = create_batch_processor(pipeline, str(output_dir))
        try:
            report = processor.process_files(input_paths)
        except OCRFrameworkError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    jobs = [
        BatchJobResponse(
            job_id=job.job_id,
            input_path=job.input_path.name,
            output_path=str(job.output_path.name) if job.output_path else None,
            status=job.status.value,
            error=job.error,
        )
        for job in report.jobs
    ]

    return BatchResponse(
        total=len(jobs),
        succeeded=report.succeeded,
        failed=report.failed,
        jobs=jobs,
    )