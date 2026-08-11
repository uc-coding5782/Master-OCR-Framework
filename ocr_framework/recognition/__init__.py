"""Text recognition implementations."""

from ocr_framework.recognition.base import TextRecognizer
from ocr_framework.recognition.paddle_recognizer import PaddleRecognizer
from ocr_framework.recognition.trocr_recognizer import TrOCRRecognizer

__all__ = [
    "TextRecognizer",
    "PaddleRecognizer",
    "TrOCRRecognizer",
]
