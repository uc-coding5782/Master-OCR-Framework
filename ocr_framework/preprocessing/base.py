"""Abstract interface for image preprocessors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ocr_framework.models.image import ImagePayload

if TYPE_CHECKING:
    from ocr_framework.pipeline.context import PipelineContext


class Preprocessor(ABC):
    """Transform an image to improve downstream OCR accuracy.

    Preprocessors must be stateless with respect to pipeline execution. Any
    configuration should be provided at construction time.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique preprocessor identifier.

        Returns:
            A stable string name used for logging and audit trails.
        """

    @abstractmethod
    def process(self, image: ImagePayload, context: PipelineContext) -> ImagePayload:
        """Apply preprocessing to an image.

        Args:
            image: Input page image.
            context: Shared pipeline execution context.

        Returns:
            The transformed image payload.

        Raises:
            PreprocessingError: If preprocessing fails.
        """
