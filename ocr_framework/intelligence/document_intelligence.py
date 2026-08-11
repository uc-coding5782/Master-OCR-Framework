"""Comprehensive document intelligence analyzer."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocr_framework.intelligence.document_classifier import DocumentClassifier
    from ocr_framework.intelligence.handwriting_detector import HandwritingDetector
    from ocr_framework.intelligence.language_detector import LanguageDetector


class DocumentIntelligence:
    """Comprehensive document intelligence analysis.

    The DocumentIntelligence combines multiple analysis components to provide
    insights about document type, handwriting presence, and language.
    """

    def __init__(self) -> None:
        """Initialize the document intelligence analyzer."""
        self._handwriting_detector: HandwritingDetector | None = None
        self._language_detector: LanguageDetector | None = None
        self._document_classifier: DocumentClassifier | None = None

    def analyze_image(self, image: np.ndarray) -> dict:
        """Analyze image for document intelligence.

        Args:
            image: Input image as numpy array.

        Returns:
            Dictionary with image-based analysis results.
        """
        if self._handwriting_detector is None:
            from ocr_framework.intelligence.handwriting_detector import HandwritingDetector
            self._handwriting_detector = HandwritingDetector()

        try:
            is_handwriting = self._handwriting_detector.detect(image)
            handwriting_confidence = self._handwriting_detector.get_confidence(image)
        except ImportError:
            is_handwriting = False
            handwriting_confidence = 0.0

        return {
            "handwriting": {
                "detected": is_handwriting,
                "confidence": handwriting_confidence,
            },
        }

    def analyze_text(self, text: str) -> dict:
        """Analyze text for document intelligence.

        Args:
            text: Input text string.

        Returns:
            Dictionary with text-based analysis results.
        """
        if self._language_detector is None:
            from ocr_framework.intelligence.language_detector import LanguageDetector
            self._language_detector = LanguageDetector()

        if self._document_classifier is None:
            from ocr_framework.intelligence.document_classifier import DocumentClassifier
            self._document_classifier = DocumentClassifier()

        language = self._language_detector.detect(text)
        language_confidence = self._language_detector.get_confidence(text, language)
        doc_type = self._document_classifier.classify(text)
        doc_type_confidence = self._document_classifier.get_confidence(text, doc_type)

        return {
            "language": {
                "detected": language,
                "confidence": language_confidence,
            },
            "document_type": {
                "detected": doc_type,
                "confidence": doc_type_confidence,
            },
        }

    def analyze(self, image: np.ndarray, text: str) -> dict:
        """Perform comprehensive document intelligence analysis.

        Args:
            image: Input image as numpy array.
            text: Extracted text from the image.

        Returns:
            Dictionary with all analysis results.
        """
        image_analysis = self.analyze_image(image)
        text_analysis = self.analyze_text(text)

        return {
            **image_analysis,
            **text_analysis,
        }

    def get_recommendations(self, image: np.ndarray, text: str) -> dict:
        """Get OCR recommendations based on document intelligence.

        Args:
            image: Input image as numpy array.
            text: Extracted text from the image.

        Returns:
            Dictionary with OCR recommendations.
        """
        analysis = self.analyze(image, text)

        recommendations = {
            "use_trocr": False,
            "use_paddle": True,
            "language": analysis["language"]["detected"],
            "document_type": analysis["document_type"]["detected"],
        }

        # Recommend TrOCR for handwriting
        if analysis["handwriting"]["detected"]:
            recommendations["use_trocr"] = True
            recommendations["use_paddle"] = False

        return recommendations
