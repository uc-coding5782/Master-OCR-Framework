"""Abstract interface for text detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ocr_framework.models.image import ImagePayload
from ocr_framework.models.ocr_result import DetectionRegion

if TYPE_CHECKING:
    from ocr_framework.pipeline.context import PipelineContext


class TextDetector(ABC):
    """Locate text regions within an image.

    Detectors are responsible for geometry only. Text transcription belongs to
    ``TextRecognizer`` implementations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique detector identifier.

        Returns:
            A stable string name used for logging and audit trails.
        """

    @abstractmethod
    def detect(
        self,
        image: ImagePayload,
        context: PipelineContext,
    ) -> list[DetectionRegion]:
        """Detect text regions in an image.

        Args:
            image: Preprocessed or raw page image.
            context: Shared pipeline execution context.

        Returns:
            Detected regions ordered arbitrarily; reading order is applied later.

        Raises:
            DetectionError: If detection fails.
        """
