"""Denoising preprocessing step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from ocr_framework.exceptions import PreprocessingError
from ocr_framework.models.image import ImagePayload

if TYPE_CHECKING:
    from ocr_framework.pipeline.context import PipelineContext


class DenoiseStep:
    """Remove noise from images while preserving edges."""

    def __init__(
        self,
        h: int = 10,
        h_color: int = 10,
        template_window_size: int = 7,
        search_window_size: int = 21,
    ) -> None:
        """Initialize the denoising step.

        Args:
            h: Parameter regulating filter strength for luminance component.
            h_color: Parameter regulating filter strength for color component.
            template_window_size: Size of template patch for computing weights.
            search_window_size: Size of search window for computing weights.
        """
        self._h = h
        self._h_color = h_color
        self._template_window_size = template_window_size
        self._search_window_size = search_window_size

    @property
    def name(self) -> str:
        """Return the step identifier."""
        return "denoise"

    def process(self, image: ImagePayload, context: PipelineContext) -> ImagePayload:
        """Apply denoising to the image.

        Args:
            image: Input image payload.
            context: Pipeline execution context.

        Returns:
            Denoised image payload.

        Raises:
            PreprocessingError: If denoising fails.
        """
        _ = context  # Reserved for future use

        try:
            # Apply fastNlMeansDenoisingColored
            denoised = cv2.fastNlMeansDenoisingColored(
                image.data,
                None,
                self._h,
                self._h_color,
                self._template_window_size,
                self._search_window_size,
            )

            # Create new ImagePayload with denoised data
            return ImagePayload(
                data=denoised,
                width=image.width,
                height=image.height,
                channels=image.channels,
                color_space=image.color_space,
                dpi=image.dpi,
                metadata={**image.metadata, "denoised": True},
            )

        except Exception as exc:
            raise PreprocessingError(f"Denoising failed: {exc}") from exc
