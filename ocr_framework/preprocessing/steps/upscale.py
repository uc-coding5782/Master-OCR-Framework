"""Upscaling preprocessing step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from ocr_framework.exceptions import PreprocessingError
from ocr_framework.models.image import ImagePayload

if TYPE_CHECKING:
    from ocr_framework.pipeline.context import PipelineContext


class UpscaleStep:
    """Upscale small/low-res images to improve OCR accuracy."""

    def __init__(self, min_dim: int = 1000) -> None:
        """Initialize the upscaling step.

        Args:
            min_dim: Minimum dimension (width or height) in pixels.
                Images smaller than this will be upscaled.
        """
        self._min_dim = min_dim

    @property
    def name(self) -> str:
        """Return the step identifier."""
        return "upscale"

    def process(self, image: ImagePayload, context: PipelineContext) -> ImagePayload:
        """Apply upscaling to the image if needed.

        Args:
            image: Input image payload.
            context: Pipeline execution context.

        Returns:
            Upscaled image payload (or original if already large enough).

        Raises:
            PreprocessingError: If upscaling fails.
        """
        _ = context  # Reserved for future use

        try:
            h, w = image.data.shape[:2]

            # Skip if already large enough
            if min(h, w) >= self._min_dim:
                return image

            # Calculate scale factor
            scale = self._min_dim / min(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Resize using cubic interpolation
            upscaled = cv2.resize(
                image.data,
                (new_w, new_h),
                interpolation=cv2.INTER_CUBIC,
            )

            # Create new ImagePayload with upscaled data
            return ImagePayload(
                data=upscaled,
                width=new_w,
                height=new_h,
                channels=image.channels,
                color_space=image.color_space,
                dpi=image.dpi,
                metadata={
                    **image.metadata,
                    "upscaled": True,
                    "scale_factor": float(scale),
                },
            )

        except Exception as exc:
            raise PreprocessingError(f"Upscaling failed: {exc}") from exc
