"""Composite preprocessor that chains multiple preprocessing steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ocr_framework.exceptions import PreprocessingError
from ocr_framework.models.image import ImagePayload
from ocr_framework.preprocessing.base import Preprocessor

if TYPE_CHECKING:
    from ocr_framework.pipeline.context import PipelineContext


class CompositePreprocessor(Preprocessor):
    """Preprocessor that applies multiple steps in sequence."""

    def __init__(self, steps: list[Any]) -> None:
        """Initialize the composite preprocessor.

        Args:
            steps: List of preprocessing step objects (e.g., DenoiseStep, DeskewStep).
        """
        self._steps = steps

    @property
    def name(self) -> str:
        """Return the preprocessor identifier."""
        return "composite"

    def process(self, image: ImagePayload, context: PipelineContext) -> ImagePayload:
        """Apply all preprocessing steps in sequence.

        Args:
            image: Input image payload.
            context: Pipeline execution context.

        Returns:
            Processed image payload after all steps.

        Raises:
            PreprocessingError: If any preprocessing step fails.
        """
        processed_image = image

        for step in self._steps:
            try:
                processed_image = step.process(processed_image, context)
            except PreprocessingError:
                raise
            except Exception as exc:
                raise PreprocessingError(
                    f"Preprocessing step '{step.name}' failed: {exc}"
                ) from exc

        return processed_image
