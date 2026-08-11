"""Confidence-based filtering postprocessing filter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ocr_framework.models.page_result import PageResult

if TYPE_CHECKING:
    from ocr_framework.pipeline.context import PipelineContext


class ConfidenceFilter:
    """Filter OCR lines based on confidence scores."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        """Initialize the confidence filter.

        Args:
            min_confidence: Minimum confidence threshold. Lines below this
                confidence will be removed.
        """
        self._min_confidence = min_confidence

    @property
    def name(self) -> str:
        """Return the filter identifier."""
        return "confidence_filter"

    def filter(self, page_result: PageResult, context: "PipelineContext") -> PageResult:
        """Filter lines based on confidence threshold.

        Args:
            page_result: Page result to filter.
            context: Pipeline execution context.

        Returns:
            Filtered page result with only high-confidence lines.
        """
        _ = context  # Reserved for future use

        # Filter lines by confidence
        filtered_lines = [
            line for line in page_result.lines
            if line.confidence >= self._min_confidence
        ]

        # Update page result with filtered lines
        page_result.lines = filtered_lines

        # Update aggregate confidence
        if filtered_lines:
            page_result.aggregate_confidence = (
                sum(line.confidence for line in filtered_lines) / len(filtered_lines)
            )
        else:
            page_result.aggregate_confidence = 0.0

        return page_result
