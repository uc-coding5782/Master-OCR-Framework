"""Image payload model used across preprocessing and OCR stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ocr_framework.types import ColorSpace, Metadata


@dataclass
class ImagePayload:
    """In-memory image representation passed between pipeline stages.

    Attributes:
        data: Raw image array, typically ``uint8`` with shape ``(H, W, C)``.
        width: Image width in pixels.
        height: Image height in pixels.
        channels: Number of color channels.
        color_space: Declared color space of ``data``.
        dpi: Optional dots-per-inch resolution metadata.
        metadata: Additional image-level metadata.
    """

    data: NDArray[Any]
    width: int
    height: int
    channels: int
    color_space: ColorSpace = ColorSpace.BGR
    dpi: int | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate image dimension invariants.

        Raises:
            ValueError: If width, height, or channels are not positive.
        """
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive.")
        if self.channels <= 0:
            raise ValueError("Image channel count must be positive.")
