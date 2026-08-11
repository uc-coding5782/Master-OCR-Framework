"""Deskewing preprocessing step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from ocr_framework.exceptions import PreprocessingError
from ocr_framework.models.image import ImagePayload

if TYPE_CHECKING:
    from ocr_framework.pipeline.context import PipelineContext


class DeskewStep:
    """Detect and correct rotation/skew in images."""

    def __init__(self, min_angle_threshold: float = 0.5) -> None:
        """Initialize the deskewing step.

        Args:
            min_angle_threshold: Minimum angle in degrees to apply correction.
                Smaller angles are skipped to avoid unnecessary interpolation.
        """
        self._min_angle_threshold = min_angle_threshold

    @property
    def name(self) -> str:
        """Return the step identifier."""
        return "deskew"

    def process(self, image: ImagePayload, context: PipelineContext) -> ImagePayload:
        """Apply deskewing to the image.

        Args:
            image: Input image payload.
            context: Pipeline execution context.

        Returns:
            Deskewed image payload.

        Raises:
            PreprocessingError: If deskewing fails.
        """
        _ = context  # Reserved for future use

        try:
            # Convert to grayscale for angle detection
            gray = cv2.cvtColor(image.data, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)

            # Threshold for contour detection
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # Get coordinates of non-zero pixels
            coords = np.column_stack(np.where(thresh > 0))

            # Skip if not enough information
            if len(coords) < 10:
                return image

            # Get minimum area rectangle
            angle = cv2.minAreaRect(coords)[-1]

            # Normalize angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            # Skip tiny corrections
            if abs(angle) < self._min_angle_threshold:
                return image

            # Rotate the image
            (h, w) = image.data.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image.data,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

            # Create new ImagePayload with deskewed data
            return ImagePayload(
                data=rotated,
                width=image.width,
                height=image.height,
                channels=image.channels,
                color_space=image.color_space,
                dpi=image.dpi,
                metadata={**image.metadata, "deskewed": True, "deskew_angle": float(angle)},
            )

        except Exception as exc:
            raise PreprocessingError(f"Deskewing failed: {exc}") from exc
