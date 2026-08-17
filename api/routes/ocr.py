"""OCR endpoint for single image processing."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.dependencies import get_pipeline
from api.schemas import OCRLineResponse, OCRResponse, PageResponse
from ocr_framework.exceptions import OCRFrameworkError

router = APIRouter(tags=["OCR"])

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


@router.post("/ocr/image", response_model=OCRResponse)
async def run_ocr(
    file: UploadFile = File(..., description="Image file to process"),
    language: str = Form(default="en", description="OCR language code"),
    include_boxes: bool = Form(default=True, description="Include bounding boxes"),
) -> OCRResponse:
    """Run OCR on a single uploaded image.

    Accepts PNG, JPEG, BMP, TIFF, and WebP images and returns recognized
    text with per-line confidence scores and optional bounding boxes.
    """
    suffix = Path(file.filename or "upload.png").suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}",
        )

    from starlette.concurrency import run_in_threadpool

    pipeline = get_pipeline()
    pipeline.config.language = language

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        result = await run_in_threadpool(pipeline.run, tmp_path)
    except OCRFrameworkError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {exc}") from exc
    finally:
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    pages = []
    for page in result.pages:
        lines = []
        for line in page.lines:
            bbox = None
            if include_boxes and line.bbox is not None:
                if hasattr(line.bbox, "x_min"):
                    bbox = {
                        "x_min": line.bbox.x_min,
                        "y_min": line.bbox.y_min,
                        "x_max": line.bbox.x_max,
                        "y_max": line.bbox.y_max,
                    }
                elif hasattr(line.bbox, "points") and line.bbox.points:
                    xs = [p.x for p in line.bbox.points]
                    ys = [p.y for p in line.bbox.points]
                    bbox = {
                        "x_min": min(xs),
                        "y_min": min(ys),
                        "x_max": max(xs),
                        "y_max": max(ys),
                    }
            lines.append(
                OCRLineResponse(
                    text=line.text,
                    confidence=line.confidence,
                    engine=line.engine_name,
                    language=line.language,
                    bounding_box=bbox,
                )
            )
        pages.append(
            PageResponse(
                page_index=page.page_index,
                lines=lines,
                engine_used=page.engine_used,
                aggregate_confidence=page.aggregate_confidence,
                timings=page.timings,
                metadata=page.metadata,
            )
        )

    return OCRResponse(
        page_count=result.page_count,
        pages=pages,
        metadata=result.metadata,
    )