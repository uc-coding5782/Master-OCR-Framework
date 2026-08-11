"""Contrast enhancement preprocessing step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from ocr_framework.exceptions import PreprocessingError
from ocr_framework.models.image import ImagePayload

if TYPE_CHECKING:
    from ocr_framework.pipeline.context import PipelineContext


class ContrastStep:
    """Apply CLAHE (adaptive histogram equalization) to boost text contrast."""

    def __init__(
        self,
        clip_limit: float = 2.5,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        """Initialize the contrast enhancement step.

        Args:
            clip_limit: Threshold for contrast limiting.
            tile_grid_size: Size of grid for histogram equalization.
        """
        self._clip_limit = clip_limit
        self._tile_grid_size = tile_grid_size

    @property
    def name(self) -> str:
        """Return the step identifier."""
        return "contrast"

    def process(self, image: ImagePayload, context: PipelineContext) -> ImagePayload:
        """Apply contrast enhancement to the image.

        Args:
            image: Input image payload.
            context: Pipeline execution context.

        Returns:
            Contrast-enhanced image payload.

        Raises:
            PreprocessingError: If contrast enhancement fails.
        """
        _ = context  # Reserved for future use

        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image.data, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self._clip_limit,
                tileGridSize=self._tile_grid_size,
            )
            l = clahe.apply(l)

            # Merge channels and convert back to BGR
            merged = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

            # Create new ImagePayload with enhanced data
            return ImagePayload(
                data=enhanced,
                width=image.width,
                height=image.height,
                channels=image.channels,
                color_space=image.color_space,
                dpi=image.dpi,
                metadata={**image.metadata, "contrast_enhanced": True},
            )

        except Exception as exc:
            raise PreprocessingError(f"Contrast enhancement failed: {exc}") from exc
