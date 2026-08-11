"""Engine selector for automatic OCR engine choice."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ocr_framework.models.page_result import PageResult

if TYPE_CHECKING:
    from ocr_framework.config.schema import FrameworkConfig


class EngineSelector:
    """Select the appropriate OCR engine based on page characteristics.

    The EngineSelector analyzes page results and selects the best engine
    for processing based on confidence scores, engine capabilities, and
    document characteristics.
    """

    def __init__(self, config: FrameworkConfig) -> None:
        """Initialize the engine selector.

        Args:
            config: Framework configuration with routing settings.
        """
        self._config = config

    def select_engine(
        self,
        page_result: PageResult | None = None,
        image_metadata: dict | None = None,
    ) -> str:
        """Select the best OCR engine for the given page.

        Args:
            page_result: Previous OCR result (if available) for reprocessing.
            image_metadata: Image metadata for heuristic selection.

        Returns:
            Selected engine identifier (e.g., 'paddle', 'trocr').
        """
        # If we have a previous result with good confidence, stick with that engine
        if page_result and page_result.aggregate_confidence >= self._config.routing.min_confidence:
            return page_result.engine_used or self._config.routing.primary_engine

        # Default to primary engine
        return self._config.routing.primary_engine

    def should_fallback(
        self,
        page_result: PageResult,
        current_engine: str,
    ) -> bool:
        """Determine if we should fallback to a different engine.

        Args:
            page_result: Current OCR result.
            current_engine: Currently used engine.

        Returns:
            True if fallback should occur, False otherwise.
        """
        # Fallback if confidence is below threshold
        if page_result.aggregate_confidence < self._config.routing.min_confidence:
            return True

        # Fallback if current engine is not primary and confidence is marginal
        if current_engine != self._config.routing.primary_engine:
            marginal_threshold = self._config.routing.min_confidence + 0.1
            if page_result.aggregate_confidence < marginal_threshold:
                return True

        return False

    def get_fallback_engine(self, current_engine: str) -> str:
        """Get the next fallback engine in the chain.

        Args:
            current_engine: Currently used engine.

        Returns:
            Next engine identifier, or empty string if no fallback available.
        """
        fallback_chain = self._config.routing.fallback_chain

        # If current engine is primary, return first fallback
        if current_engine == self._config.routing.primary_engine:
            if fallback_chain:
                return fallback_chain[0]
            return ""

        # If current engine is in fallback chain, return next in chain
        try:
            current_index = fallback_chain.index(current_engine)
            if current_index + 1 < len(fallback_chain):
                return fallback_chain[current_index + 1]
        except ValueError:
            pass

        return ""
