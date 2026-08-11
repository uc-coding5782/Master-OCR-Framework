"""Confidence manager for OCR quality assessment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ocr_framework.models.page_result import PageResult

if TYPE_CHECKING:
    from ocr_framework.config.schema import FrameworkConfig


class ConfidenceManager:
    """Manage confidence-based decisions for OCR processing.

    The ConfidenceManager evaluates OCR confidence scores and makes
    decisions about whether results are acceptable, need reprocessing,
    or should trigger fallback mechanisms.
    """

    def __init__(self, config: FrameworkConfig) -> None:
        """Initialize the confidence manager.

        Args:
            config: Framework configuration with routing settings.
        """
        self._config = config

    def is_acceptable(self, page_result: PageResult) -> bool:
        """Determine if OCR results are acceptable.

        Args:
            page_result: OCR result to evaluate.

        Returns:
            True if confidence is acceptable, False otherwise.
        """
        return page_result.aggregate_confidence >= self._config.routing.min_confidence

    def is_high_quality(self, page_result: PageResult) -> bool:
        """Determine if OCR results are high quality.

        Args:
            page_result: OCR result to evaluate.

        Returns:
            True if confidence is high (>0.8), False otherwise.
        """
        return page_result.aggregate_confidence >= 0.8

    def is_low_quality(self, page_result: PageResult) -> bool:
        """Determine if OCR results are low quality.

        Args:
            page_result: OCR result to evaluate.

        Returns:
            True if confidence is low (<0.5), False otherwise.
        """
        return page_result.aggregate_confidence < 0.5

    def get_confidence_summary(self, page_result: PageResult) -> dict:
        """Get a summary of confidence metrics.

        Args:
            page_result: OCR result to evaluate.

        Returns:
            Dictionary with confidence summary statistics.
        """
        if not page_result.lines:
            return {
                "aggregate": 0.0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "line_count": 0,
            }

        confidences = [line.confidence for line in page_result.lines]

        return {
            "aggregate": page_result.aggregate_confidence,
            "min": min(confidences),
            "max": max(confidences),
            "mean": sum(confidences) / len(confidences),
            "line_count": len(confidences),
        }

    def filter_low_confidence_lines(
        self,
        page_result: PageResult,
        threshold: float | None = None,
    ) -> list:
        """Filter out low-confidence lines from the result.

        Args:
            page_result: OCR result to filter.
            threshold: Confidence threshold (uses config default if None).

        Returns:
            List of high-confidence lines.
        """
        threshold = threshold or self._config.routing.min_confidence
        return [line for line in page_result.lines if line.confidence >= threshold]
