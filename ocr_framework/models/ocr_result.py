"""OCR detection and recognition result models."""

from __future__ import annotations

from dataclasses import dataclass, field

from ocr_framework.models.bbox import BoundingBox, Polygon
from ocr_framework.types import RegionType


@dataclass
class DetectionRegion:
    """A text or layout region detected in an image.

    Attributes:
        bbox: Region geometry as a box or polygon.
        confidence: Detector confidence score in ``[0.0, 1.0]``.
        region_type: Semantic classification of the detected region.
    """

    bbox: BoundingBox | Polygon
    confidence: float
    region_type: RegionType = RegionType.TEXT


@dataclass
class OCRToken:
    """A single recognized token with geometry and confidence.

    Attributes:
        text: Recognized token text.
        confidence: Recognizer confidence score in ``[0.0, 1.0]``.
        bbox: Token geometry as a box or polygon.
    """

    text: str
    confidence: float
    bbox: BoundingBox | Polygon


@dataclass
class OCRLine:
    """A recognized line of text with optional token-level detail.

    Attributes:
        text: Full line text content.
        confidence: Line-level confidence score in ``[0.0, 1.0]``.
        bbox: Line geometry as a box or polygon.
        engine_name: Name of the recognizer that produced this line.
        language: Optional BCP-47 or engine-specific language code.
        tokens: Optional token-level recognition details.
    """

    text: str
    confidence: float
    bbox: BoundingBox | Polygon
    engine_name: str = ""
    language: str | None = None
    tokens: list[OCRToken] = field(default_factory=list)
