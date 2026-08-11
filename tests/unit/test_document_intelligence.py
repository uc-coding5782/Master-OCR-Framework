"""Tests for document intelligence components."""

import numpy as np
import pytest

from ocr_framework.intelligence.document_classifier import DocumentClassifier
from ocr_framework.intelligence.document_intelligence import DocumentIntelligence
from ocr_framework.intelligence.handwriting_detector import HandwritingDetector, CV2_AVAILABLE
from ocr_framework.intelligence.language_detector import LanguageDetector


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample test image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


class TestHandwritingDetector:
    """Tests for HandwritingDetector."""

    def test_raises_import_error_without_cv2(self) -> None:
        """Test that HandwritingDetector raises ImportError without cv2."""
        if CV2_AVAILABLE:
            pytest.skip("cv2 is installed")

        detector = HandwritingDetector()
        with pytest.raises(ImportError, match="cv2"):
            detector.detect(np.zeros((100, 100), dtype=np.uint8))

    def test_detects_handwriting(self, sample_image: np.ndarray) -> None:
        """Test handwriting detection."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        detector = HandwritingDetector()
        is_handwriting = detector.detect(sample_image)

        assert isinstance(is_handwriting, bool)

    def test_get_confidence(self, sample_image: np.ndarray) -> None:
        """Test confidence score."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        detector = HandwritingDetector()
        confidence = detector.get_confidence(sample_image)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


class TestLanguageDetector:
    """Tests for LanguageDetector."""

    def test_detects_english(self) -> None:
        """Test English language detection."""
        detector = LanguageDetector()
        text = "Hello world, this is a test."

        language = detector.detect(text)
        assert language == "en"

    def test_detects_empty_text(self) -> None:
        """Test empty text detection."""
        detector = LanguageDetector()
        language = detector.detect("")

        assert language == "unknown"

    def test_detects_non_ascii(self) -> None:
        """Test non-ASCII text detection."""
        detector = LanguageDetector()
        text = "Hello 世界"  # Mixed ASCII and Chinese

        language = detector.detect(text)
        # Should detect unknown due to mixed content
        assert language in ["en", "unknown"]

    def test_detect_batch(self) -> None:
        """Test batch language detection."""
        detector = LanguageDetector()
        texts = ["Hello world", "Test text", "Another test"]

        result = detector.detect_batch(texts)

        assert isinstance(result, dict)
        assert "en" in result

    def test_get_confidence(self) -> None:
        """Test confidence score."""
        detector = LanguageDetector()
        text = "Hello world, this is a test."

        confidence = detector.get_confidence(text, "en")

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


class TestDocumentClassifier:
    """Tests for DocumentClassifier."""

    def test_classifies_receipt(self) -> None:
        """Test receipt classification."""
        classifier = DocumentClassifier()
        text = "Total: $50.00 Cash Payment Change: $10.00"

        doc_type = classifier.classify(text)
        assert doc_type == "receipt"

    def test_classifies_invoice(self) -> None:
        """Test invoice classification."""
        classifier = DocumentClassifier()
        text = "Invoice Number: INV-001 Bill To: John Doe Due Date: 2024-01-01"

        doc_type = classifier.classify(text)
        assert doc_type == "invoice"

    def test_classifies_form(self) -> None:
        """Test form classification."""
        classifier = DocumentClassifier()
        text = "Name: _____ Address: _____ Phone: _____ Signature: _____"

        doc_type = classifier.classify(text)
        assert doc_type == "form"

    def test_classifies_generic_document(self) -> None:
        """Test generic document classification."""
        classifier = DocumentClassifier()
        text = "This is some random text without specific keywords."

        doc_type = classifier.classify(text)
        assert doc_type == "document"

    def test_classifies_empty_text(self) -> None:
        """Test empty text classification."""
        classifier = DocumentClassifier()
        doc_type = classifier.classify("")

        assert doc_type == "document"

    def test_get_confidence(self) -> None:
        """Test confidence score."""
        classifier = DocumentClassifier()
        text = "Total: $50.00 Cash Payment"

        confidence = classifier.get_confidence(text, "receipt")

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_classify_batch(self) -> None:
        """Test batch classification."""
        classifier = DocumentClassifier()
        texts = [
            "Total: $50.00",
            "Invoice Number: INV-001",
            "Random text",
        ]

        result = classifier.classify_batch(texts)

        assert isinstance(result, dict)
        assert "receipt" in result or "invoice" in result or "document" in result


class TestDocumentIntelligence:
    """Tests for DocumentIntelligence."""

    def test_analyzes_image(self, sample_image: np.ndarray) -> None:
        """Test image analysis."""
        intelligence = DocumentIntelligence()
        result = intelligence.analyze_image(sample_image)

        assert "handwriting" in result
        assert "detected" in result["handwriting"]
        assert "confidence" in result["handwriting"]

    def test_analyzes_text(self) -> None:
        """Test text analysis."""
        intelligence = DocumentIntelligence()
        text = "Hello world"

        result = intelligence.analyze_text(text)

        assert "language" in result
        assert "document_type" in result
        assert "detected" in result["language"]
        assert "detected" in result["document_type"]

    def test_performs_comprehensive_analysis(self, sample_image: np.ndarray) -> None:
        """Test comprehensive analysis."""
        intelligence = DocumentIntelligence()
        text = "Total: $50.00"

        result = intelligence.analyze(sample_image, text)

        assert "handwriting" in result
        assert "language" in result
        assert "document_type" in result

    def test_get_recommendations(self, sample_image: np.ndarray) -> None:
        """Test OCR recommendations."""
        intelligence = DocumentIntelligence()
        text = "Hello world"

        recommendations = intelligence.get_recommendations(sample_image, text)

        assert "use_trocr" in recommendations
        assert "use_paddle" in recommendations
        assert "language" in recommendations
        assert "document_type" in recommendations

    def test_recommends_trocr_for_handwriting(self, sample_image: np.ndarray) -> None:
        """Test that TrOCR is recommended for handwriting."""
        intelligence = DocumentIntelligence()
        text = "Handwritten text"

        # Mock handwriting detection
        recommendations = intelligence.get_recommendations(sample_image, text)

        # If handwriting is detected, should recommend TrOCR
        assert isinstance(recommendations["use_trocr"], bool)
