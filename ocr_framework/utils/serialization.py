"""Serialization helpers for OCR results."""

from __future__ import annotations

from typing import Any

from ocr_framework.models.page_result import DocumentResult


def document_result_to_dict(result: DocumentResult, include_boxes: bool = True) -> dict[str, Any]:
    """Convert a DocumentResult to a JSON-serializable dictionary."""
    pages_data = []
    for page_result in result.pages:
        lines_data = []
        for line in page_result.lines:
            line_data: dict[str, Any] = {
                "text": line.text,
                "confidence": line.confidence,
                "engine": line.engine_name,
                "language": line.language,
            }
            if include_boxes and line.bbox is not None:
                line_data["bounding_box"] = {
                    "x_min": line.bbox.x_min,
                    "y_min": line.bbox.y_min,
                    "x_max": line.bbox.x_max,
                    "y_max": line.bbox.y_max,
                }
            lines_data.append(line_data)

        pages_data.append(
            {
                "page_index": page_result.page_index,
                "lines": lines_data,
                "engine_used": page_result.engine_used,
                "aggregate_confidence": page_result.aggregate_confidence,
                "timings": page_result.timings,
                "metadata": page_result.metadata,
            }
        )

    return {
        "page_count": result.page_count,
        "pages": pages_data,
        "metadata": result.metadata,
    }
