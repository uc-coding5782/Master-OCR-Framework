"""Document intelligence module."""

from ocr_framework.intelligence.document_classifier import DocumentClassifier
from ocr_framework.intelligence.document_intelligence import DocumentIntelligence
from ocr_framework.intelligence.handwriting_detector import HandwritingDetector
from ocr_framework.intelligence.language_detector import LanguageDetector

__all__ = [
    "DocumentClassifier",
    "DocumentIntelligence",
    "HandwritingDetector",
    "LanguageDetector",
]
