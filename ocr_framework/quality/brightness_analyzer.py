"""Brightness analysis for image quality assessment."""

from __future__ import annotations

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class BrightnessAnalyzer:
    """Analyze image brightness using mean pixel intensity."""

    def analyze(self, image: np.ndarray) -> float:
        """Analyze brightness using mean pixel intensity.

        Args:
            image: Input image as numpy array (grayscale or BGR).

        Returns:
            Brightness score (0-255, where 128 is neutral).

        Raises:
            ImportError: If cv2 is not available.
        """
        if not CV2_AVAILABLE:
            raise ImportError("cv2 is required for brightness analysis")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate mean brightness
        brightness = np.mean(gray)

        return float(brightness)

    def is_too_dark(self, image: np.ndarray, threshold: float = 50.0) -> bool:
        """Determine if image is too dark.

        Args:
            image: Input image as numpy array.
            threshold: Darkness threshold (below this is too dark).

        Returns:
            True if image is too dark, False otherwise.
        """
        brightness = self.analyze(image)
        return brightness < threshold

    def is_too_bright(self, image: np.ndarray, threshold: float = 200.0) -> bool:
        """Determine if image is too bright.

        Args:
            image: Input image as numpy array.
            threshold: Brightness threshold (above this is too bright).

        Returns:
            True if image is too bright, False otherwise.
        """
        brightness = self.analyze(image)
        return brightness > threshold

    def get_quality_rating(self, image: np.ndarray) -> str:
        """Get quality rating based on brightness.

        Args:
            image: Input image as numpy array.

        Returns:
            Quality rating: 'excellent', 'good', 'fair', or 'poor'.
        """
        brightness = self.analyze(image)

        if 100 <= brightness <= 155:
            return "excellent"
        elif 80 <= brightness <= 175:
            return "good"
        elif 50 <= brightness <= 200:
            return "fair"
        else:
            return "poor"
