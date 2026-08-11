"""Pydantic schemas for the OCR HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    gpu_available: bool


class OCRLineResponse(BaseModel):
    """Single recognized line in an API response."""

    text: str
    confidence: float
    engine: str = ""
    language: str | None = None
    bounding_box: dict[str, float] | None = None


class PageResponse(BaseModel):
    """OCR results for a single page."""

    page_index: int
    lines: list[OCRLineResponse]
    engine_used: str = ""
    aggregate_confidence: float = 0.0
    timings: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OCRResponse(BaseModel):
    """OCR result response."""

    page_count: int
    pages: list[PageResponse]
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchJobResponse(BaseModel):
    """Single batch job result."""

    job_id: str
    input_path: str
    output_path: str | None = None
    status: str
    error: str | None = None


class BatchResponse(BaseModel):
    """Batch processing response."""

    total: int
    succeeded: int
    failed: int
    jobs: list[BatchJobResponse]


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str
    error_type: str = "OCRFrameworkError"


class VersionResponse(BaseModel):
    """Version information response."""

    framework_version: str
    api_version: str
