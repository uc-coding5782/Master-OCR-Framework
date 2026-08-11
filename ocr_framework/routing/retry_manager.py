"""Retry manager for OCR engine fallback and retry logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ocr_framework.models.page_result import PageResult

if TYPE_CHECKING:
    from ocr_framework.config.schema import FrameworkConfig


class RetryManager:
    """Manage retry logic and fallback between OCR engines.

    The RetryManager coordinates fallback to alternative engines when
    the primary engine produces low-confidence results or fails.
    """

    def __init__(self, config: FrameworkConfig) -> None:
        """Initialize the retry manager.

        Args:
            config: Framework configuration with routing settings.
        """
        self._config = config

    def should_retry(self, page_result: PageResult, attempt: int) -> bool:
        """Determine if OCR should be retried with a different engine.

        Args:
            page_result: Current OCR result.
            attempt: Current attempt number.

        Returns:
            True if retry should occur, False otherwise.
        """
        # Don't retry if we've exceeded max attempts
        max_attempts = len(self._config.routing.fallback_chain) + 1
        if attempt >= max_attempts:
            return False

        # Retry if confidence is too low
        if page_result.aggregate_confidence < self._config.routing.min_confidence:
            return True

        # Retry if we have empty results
        if not page_result.lines:
            return True

        return False

    def get_next_engine(self, current_engine: str, attempt: int) -> str:
        """Get the next engine to try in the fallback chain.

        Args:
            current_engine: Currently used engine.
            attempt: Current attempt number.

        Returns:
            Next engine identifier to try.
        """
        fallback_chain = self._config.routing.fallback_chain

        # On first retry, use first fallback
        if attempt == 1 and fallback_chain:
            return fallback_chain[0]

        # On subsequent retries, cycle through fallbacks
        if fallback_chain:
            index = (attempt - 1) % len(fallback_chain)
            return fallback_chain[index]

        # Fallback to primary if no fallbacks available
        return self._config.routing.primary_engine

    def get_max_retries(self) -> int:
        """Get the maximum number of retry attempts.

        Returns:
            Maximum retry attempts.
        """
        return len(self._config.routing.fallback_chain)

    def record_attempt(
        self,
        engine: str,
        attempt: int,
        confidence: float,
        success: bool,
    ) -> dict:
        """Record a retry attempt for tracking.

        Args:
            engine: Engine used for this attempt.
            attempt: Attempt number.
            confidence: Confidence score achieved.
            success: Whether the attempt was successful.

        Returns:
            Attempt record dictionary.
        """
        return {
            "engine": engine,
            "attempt": attempt,
            "confidence": confidence,
            "success": success,
        }
