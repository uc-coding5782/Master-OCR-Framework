"""OCR engine interfaces and implementations."""

from ocr_framework.engines.capabilities import EngineCapabilities
from ocr_framework.engines.detector import TextDetector
from ocr_framework.engines.recognizer import TextRecognizer

__all__ = ["EngineCapabilities", "TextDetector", "TextRecognizer"]
