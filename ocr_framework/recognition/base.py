"""Abstract interface for text recognizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ocr_framework.models.image import ImagePayload
from ocr_framework.models.ocr_result import DetectionRegion, OCRLine

if TYPE_CHECKING:
    from ocr_framework.pipeline.context import PipelineContext


class TextRecognizer(ABC):
    """Recognize text content from detected regions.

    Recognizers consume detection geometry and return line-level OCR output.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique recognizer identifier.

        Returns:
            A stable string name used for logging and audit trails.
        """

    @abstractmethod
    def recognize(
        self,
        image: ImagePayload,
        regions: list[DetectionRegion],
        context: PipelineContext,
    ) -> list[OCRLine]:
        """Recognize text for the supplied detection regions.

        Args:
            image: Preprocessed or raw page image.
            regions: Detection regions to transcribe.
            context: Shared pipeline execution context.

        Returns:
            Recognized OCR lines.

        Raises:
            RecognitionError: If recognition fails.
        """
