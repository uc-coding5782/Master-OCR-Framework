"""Abstract interface for OCR postprocessors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ocr_framework.models.page_result import PageResult

if TYPE_CHECKING:
    from ocr_framework.pipeline.context import PipelineContext


class PostProcessor(ABC):
    """Refine recognized OCR output for a single page."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique postprocessor identifier.

        Returns:
            A stable string name used for logging and audit trails.
        """

    @abstractmethod
    def process(
        self,
        page_result: PageResult,
        context: PipelineContext,
    ) -> PageResult:
        """Apply postprocessing to a page result.

        Args:
            page_result: Raw or partially processed OCR output for one page.
            context: Shared pipeline execution context.

        Returns:
            The refined page result.
        """
