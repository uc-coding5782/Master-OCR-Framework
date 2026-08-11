"""Tests for image quality analysis components."""

import numpy as np
import pytest

from ocr_framework.quality.blur_estimator import BlurEstimator, CV2_AVAILABLE
from ocr_framework.quality.brightness_analyzer import BrightnessAnalyzer
from ocr_framework.quality.contrast_analyzer import ContrastAnalyzer
from ocr_framework.quality.noise_estimator import NoiseEstimator
from ocr_framework.quality.quality_analyzer import QualityAnalyzer
from ocr_framework.quality.resolution_analyzer import ResolutionAnalyzer
from ocr_framework.quality.rotation_detector import RotationDetector


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample test image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale() -> np.ndarray:
    """Create a sample grayscale test image."""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


class TestBlurEstimator:
    """Tests for BlurEstimator."""

    def test_raises_import_error_without_cv2(self) -> None:
        """Test that BlurEstimator raises ImportError without cv2."""
        if CV2_AVAILABLE:
            pytest.skip("cv2 is installed")

        estimator = BlurEstimator()
        with pytest.raises(ImportError, match="cv2"):
            estimator.estimate(np.zeros((100, 100), dtype=np.uint8))

    def test_estimates_blur_score(self, sample_image: np.ndarray) -> None:
        """Test blur score estimation."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        estimator = BlurEstimator()
        score = estimator.estimate(sample_image)

        assert isinstance(score, float)
        assert score >= 0.0

    def test_determines_blurry(self, sample_image: np.ndarray) -> None:
        """Test blurry detection."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        estimator = BlurEstimator()
        is_blurry = estimator.is_blurry(sample_image, threshold=10000.0)

        # Random image should not be blurry with high threshold
        assert isinstance(is_blurry, bool)

    def test_get_quality_rating(self, sample_image: np.ndarray) -> None:
        """Test quality rating."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        estimator = BlurEstimator()
        rating = estimator.get_quality_rating(sample_image)

        assert rating in ["excellent", "good", "fair", "poor"]


class TestNoiseEstimator:
    """Tests for NoiseEstimator."""

    def test_raises_import_error_without_cv2(self) -> None:
        """Test that NoiseEstimator raises ImportError without cv2."""
        if CV2_AVAILABLE:
            pytest.skip("cv2 is installed")

        estimator = NoiseEstimator()
        with pytest.raises(ImportError, match="cv2"):
            estimator.estimate(np.zeros((100, 100), dtype=np.uint8))

    def test_estimates_noise_level(self, sample_image: np.ndarray) -> None:
        """Test noise level estimation."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        estimator = NoiseEstimator()
        level = estimator.estimate(sample_image)

        assert isinstance(level, float)
        assert level >= 0.0

    def test_determines_noisy(self, sample_image: np.ndarray) -> None:
        """Test noisy detection."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        estimator = NoiseEstimator()
        is_noisy = estimator.is_noisy(sample_image, threshold=100.0)

        assert isinstance(is_noisy, bool)

    def test_get_quality_rating(self, sample_image: np.ndarray) -> None:
        """Test quality rating."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        estimator = NoiseEstimator()
        rating = estimator.get_quality_rating(sample_image)

        assert rating in ["excellent", "good", "fair", "poor"]


class TestBrightnessAnalyzer:
    """Tests for BrightnessAnalyzer."""

    def test_raises_import_error_without_cv2(self) -> None:
        """Test that BrightnessAnalyzer raises ImportError without cv2."""
        if CV2_AVAILABLE:
            pytest.skip("cv2 is installed")

        analyzer = BrightnessAnalyzer()
        with pytest.raises(ImportError, match="cv2"):
            analyzer.analyze(np.zeros((100, 100), dtype=np.uint8))

    def test_analyzes_brightness(self, sample_image: np.ndarray) -> None:
        """Test brightness analysis."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        analyzer = BrightnessAnalyzer()
        brightness = analyzer.analyze(sample_image)

        assert isinstance(brightness, float)
        assert 0.0 <= brightness <= 255.0

    def test_determines_too_dark(self, sample_image: np.ndarray) -> None:
        """Test dark detection."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        analyzer = BrightnessAnalyzer()
        is_dark = analyzer.is_too_dark(sample_image, threshold=250.0)

        assert isinstance(is_dark, bool)

    def test_determines_too_bright(self, sample_image: np.ndarray) -> None:
        """Test bright detection."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        analyzer = BrightnessAnalyzer()
        is_bright = analyzer.is_too_bright(sample_image, threshold=5.0)

        assert isinstance(is_bright, bool)

    def test_get_quality_rating(self, sample_image: np.ndarray) -> None:
        """Test quality rating."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        analyzer = BrightnessAnalyzer()
        rating = analyzer.get_quality_rating(sample_image)

        assert rating in ["excellent", "good", "fair", "poor"]


class TestContrastAnalyzer:
    """Tests for ContrastAnalyzer."""

    def test_raises_import_error_without_cv2(self) -> None:
        """Test that ContrastAnalyzer raises ImportError without cv2."""
        if CV2_AVAILABLE:
            pytest.skip("cv2 is installed")

        analyzer = ContrastAnalyzer()
        with pytest.raises(ImportError, match="cv2"):
            analyzer.analyze(np.zeros((100, 100), dtype=np.uint8))

    def test_analyzes_contrast(self, sample_image: np.ndarray) -> None:
        """Test contrast analysis."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        analyzer = ContrastAnalyzer()
        contrast = analyzer.analyze(sample_image)

        assert isinstance(contrast, float)
        assert contrast >= 0.0

    def test_determines_low_contrast(self, sample_image: np.ndarray) -> None:
        """Test low contrast detection."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        analyzer = ContrastAnalyzer()
        is_low = analyzer.is_low_contrast(sample_image, threshold=200.0)

        assert isinstance(is_low, bool)

    def test_get_quality_rating(self, sample_image: np.ndarray) -> None:
        """Test quality rating."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        analyzer = ContrastAnalyzer()
        rating = analyzer.get_quality_rating(sample_image)

        assert rating in ["excellent", "good", "fair", "poor"]


class TestRotationDetector:
    """Tests for RotationDetector."""

    def test_raises_import_error_without_cv2(self) -> None:
        """Test that RotationDetector raises ImportError without cv2."""
        if CV2_AVAILABLE:
            pytest.skip("cv2 is installed")

        detector = RotationDetector()
        with pytest.raises(ImportError, match="cv2"):
            detector.detect(np.zeros((100, 100), dtype=np.uint8))

    def test_detects_rotation(self, sample_image: np.ndarray) -> None:
        """Test rotation detection."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        detector = RotationDetector()
        angle = detector.detect(sample_image)

        assert isinstance(angle, float)
        assert -90.0 <= angle <= 90.0

    def test_determines_rotated(self, sample_image: np.ndarray) -> None:
        """Test rotated detection."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        detector = RotationDetector()
        is_rotated = detector.is_rotated(sample_image, threshold=100.0)

        assert isinstance(is_rotated, bool)

    def test_get_rotation_direction(self, sample_image: np.ndarray) -> None:
        """Test rotation direction."""
        if not CV2_AVAILABLE:
            pytest.skip("cv2 not available")

        detector = RotationDetector()
        direction = detector.get_rotation_direction(sample_image)

        assert direction in ["clockwise", "counterclockwise", "none"]


class TestResolutionAnalyzer:
    """Tests for ResolutionAnalyzer."""

    def test_analyzes_resolution(self, sample_image: np.ndarray) -> None:
        """Test resolution analysis."""
        analyzer = ResolutionAnalyzer()
        metrics = analyzer.analyze(sample_image)

        assert metrics["width"] == 100
        assert metrics["height"] == 100
        assert metrics["total_pixels"] == 10000
        assert metrics["aspect_ratio"] == 1.0

    def test_determines_low_resolution(self, sample_image: np.ndarray) -> None:
        """Test low resolution detection."""
        analyzer = ResolutionAnalyzer()
        is_low = analyzer.is_low_resolution(sample_image, min_width=200, min_height=200)

        assert is_low is True

    def test_get_quality_rating(self, sample_image: np.ndarray) -> None:
        """Test quality rating."""
        analyzer = ResolutionAnalyzer()
        rating = analyzer.get_quality_rating(sample_image)

        assert rating in ["excellent", "good", "fair", "poor"]

    def test_get_dpi(self, sample_image: np.ndarray) -> None:
        """Test DPI estimation."""
        analyzer = ResolutionAnalyzer()
        dpi = analyzer.get_dpi(sample_image, physical_width_mm=210.0)

        assert isinstance(dpi, float)
        assert dpi >= 0.0


class TestQualityAnalyzer:
    """Tests for QualityAnalyzer."""

    def test_performs_comprehensive_analysis(self, sample_image: np.ndarray) -> None:
        """Test comprehensive quality analysis."""
        analyzer = QualityAnalyzer()
        metrics = analyzer.analyze(sample_image)

        assert "blur" in metrics
        assert "noise" in metrics
        assert "brightness" in metrics
        assert "contrast" in metrics
        assert "rotation" in metrics
        assert "resolution" in metrics

    def test_get_overall_quality(self, sample_image: np.ndarray) -> None:
        """Test overall quality rating."""
        analyzer = QualityAnalyzer()
        rating = analyzer.get_overall_quality(sample_image)

        assert rating in ["excellent", "good", "fair", "poor", "unavailable"]

    def test_needs_preprocessing(self, sample_image: np.ndarray) -> None:
        """Test preprocessing recommendations."""
        analyzer = QualityAnalyzer()
        recommendations = analyzer.needs_preprocessing(sample_image)

        assert "denoise" in recommendations
        assert "deskew" in recommendations
        assert "contrast_enhancement" in recommendations
        assert "brightness_adjustment" in recommendations
        assert "upscale" in recommendations

        # All should be boolean
        for key, value in recommendations.items():
            assert isinstance(value, bool)
