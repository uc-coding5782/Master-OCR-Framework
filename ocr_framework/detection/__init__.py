"""Text detection implementations."""

from ocr_framework.detection.base import TextDetector
from ocr_framework.detection.paddle_detector import PaddleDetector

__all__ = [
    "TextDetector",
    "PaddleDetector",
]
