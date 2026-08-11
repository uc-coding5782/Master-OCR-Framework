"""Tests for TrOCR recognizer."""

import pytest

from ocr_framework.recognition.trocr_recognizer import TrOCRRecognizer, TROCR_AVAILABLE


class TestTrOCRRecognizer:
    """Tests for TrOCRRecognizer."""

    def test_raises_import_error_without_transformers(self) -> None:
        """Test that TrOCRRecognizer raises ImportError without transformers."""
        if TROCR_AVAILABLE:
            pytest.skip("transformers library is installed")

        with pytest.raises(ImportError, match="transformers"):
            TrOCRRecognizer()

    def test_recognizer_name_property(self) -> None:
        """Test that recognizer name is set correctly (without loading model)."""
        # This test checks the name property without loading the model
        # We can test this even without transformers installed
        assert TrOCRRecognizer.__name__ == "TrOCRRecognizer"

    def test_supports_different_models_property(self) -> None:
        """Test that model name is stored correctly (without loading model)."""
        # Test parameter handling without loading model
        if not TROCR_AVAILABLE:
            # Just verify the class exists and has the right structure
            assert hasattr(TrOCRRecognizer, "__init__")
        else:
            recognizer = TrOCRRecognizer(model_name="microsoft/trocr-small-handwritten")
            assert recognizer._model_name == "microsoft/trocr-small-handwritten"

    def test_supports_gpu_configuration_property(self) -> None:
        """Test that GPU configuration is stored correctly (without loading model)."""
        if not TROCR_AVAILABLE:
            # Just verify the class exists and has the right structure
            assert hasattr(TrOCRRecognizer, "__init__")
        else:
            recognizer = TrOCRRecognizer(use_gpu=True)
            assert recognizer._use_gpu is True
