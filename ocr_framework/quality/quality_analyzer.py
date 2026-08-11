"""Comprehensive image quality analyzer."""

from __future__ import annotations

import numpy as np

from ocr_framework.quality.blur_estimator import BlurEstimator
from ocr_framework.quality.brightness_analyzer import BrightnessAnalyzer
from ocr_framework.quality.contrast_analyzer import ContrastAnalyzer
from ocr_framework.quality.noise_estimator import NoiseEstimator
from ocr_framework.quality.resolution_analyzer import ResolutionAnalyzer
from ocr_framework.quality.rotation_detector import RotationDetector


class QualityAnalyzer:
    """Comprehensive image quality analysis for OCR suitability.

    The QualityAnalyzer combines multiple quality metrics to provide
    an overall assessment of image quality for OCR processing.
    """

    def __init__(self) -> None:
        """Initialize the quality analyzer."""
        self._blur_estimator = BlurEstimator()
        self._noise_estimator = NoiseEstimator()
        self._brightness_analyzer = BrightnessAnalyzer()
        self._contrast_analyzer = ContrastAnalyzer()
        self._rotation_detector = RotationDetector()
        self._resolution_analyzer = ResolutionAnalyzer()

    def analyze(self, image: np.ndarray) -> dict:
        """Perform comprehensive quality analysis.

        Args:
            image: Input image as numpy array.

        Returns:
            Dictionary with all quality metrics.
        """
        try:
            blur_score = self._blur_estimator.estimate(image)
            blur_rating = self._blur_estimator.get_quality_rating(image)
        except ImportError:
            blur_score = 0.0
            blur_rating = "unavailable"

        try:
            noise_level = self._noise_estimator.estimate(image)
            noise_rating = self._noise_estimator.get_quality_rating(image)
        except ImportError:
            noise_level = 0.0
            noise_rating = "unavailable"

        try:
            brightness = self._brightness_analyzer.analyze(image)
            brightness_rating = self._brightness_analyzer.get_quality_rating(image)
        except ImportError:
            brightness = 0.0
            brightness_rating = "unavailable"

        try:
            contrast = self._contrast_analyzer.analyze(image)
            contrast_rating = self._contrast_analyzer.get_quality_rating(image)
        except ImportError:
            contrast = 0.0
            contrast_rating = "unavailable"

        try:
            rotation = self._rotation_detector.detect(image)
            rotation_direction = self._rotation_detector.get_rotation_direction(image)
        except ImportError:
            rotation = 0.0
            rotation_direction = "unavailable"

        resolution_metrics = self._resolution_analyzer.analyze(image)
        resolution_rating = self._resolution_analyzer.get_quality_rating(image)

        return {
            "blur": {
                "score": blur_score,
                "rating": blur_rating,
            },
            "noise": {
                "level": noise_level,
                "rating": noise_rating,
            },
            "brightness": {
                "score": brightness,
                "rating": brightness_rating,
            },
            "contrast": {
                "score": contrast,
                "rating": contrast_rating,
            },
            "rotation": {
                "angle": rotation,
                "direction": rotation_direction,
            },
            "resolution": {
                **resolution_metrics,
                "rating": resolution_rating,
            },
        }

    def get_overall_quality(self, image: np.ndarray) -> str:
        """Get overall quality rating.

        Args:
            image: Input image as numpy array.

        Returns:
            Overall quality rating: 'excellent', 'good', 'fair', or 'poor'.
        """
        metrics = self.analyze(image)

        # Count ratings
        ratings = [
            metrics["blur"]["rating"],
            metrics["noise"]["rating"],
            metrics["brightness"]["rating"],
            metrics["contrast"]["rating"],
            metrics["resolution"]["rating"],
        ]

        # Skip unavailable ratings
        ratings = [r for r in ratings if r != "unavailable"]

        if not ratings:
            return "unavailable"

        # If any rating is poor, overall is poor
        if "poor" in ratings:
            return "poor"

        # If any rating is fair, overall is fair
        if "fair" in ratings:
            return "fair"

        # If all are good or excellent
        if all(r in ["good", "excellent"] for r in ratings):
            return "good" if "good" in ratings else "excellent"

        return "fair"

    def needs_preprocessing(self, image: np.ndarray) -> dict:
        """Determine which preprocessing steps are needed.

        Args:
            image: Input image as numpy array.

        Returns:
            Dictionary indicating which preprocessing steps are recommended.
        """
        metrics = self.analyze(image)

        recommendations = {
            "denoise": False,
            "deskew": False,
            "contrast_enhancement": False,
            "brightness_adjustment": False,
            "upscale": False,
        }

        # Check if denoising is needed
        if metrics["noise"]["rating"] in ["fair", "poor"]:
            recommendations["denoise"] = True

        # Check if deskewing is needed
        if abs(metrics["rotation"]["angle"]) > 5:
            recommendations["deskew"] = True

        # Check if contrast enhancement is needed
        if metrics["contrast"]["rating"] in ["fair", "poor"]:
            recommendations["contrast_enhancement"] = True

        # Check if brightness adjustment is needed
        if metrics["brightness"]["rating"] in ["fair", "poor"]:
            recommendations["brightness_adjustment"] = True

        # Check if upscaling is needed
        if metrics["resolution"]["rating"] in ["fair", "poor"]:
            recommendations["upscale"] = True

        return recommendations
