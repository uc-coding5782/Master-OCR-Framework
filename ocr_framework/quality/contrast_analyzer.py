"""Contrast analysis for image quality assessment."""

from __future__ import annotations

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class ContrastAnalyzer:
    """Analyze image contrast using standard deviation of pixel intensities."""

    def analyze(self, image: np.ndarray) -> float:
        """Analyze contrast using standard deviation.

        Args:
            image: Input image as numpy array (grayscale or BGR).

        Returns:
            Contrast score (higher is more contrast).

        Raises:
            ImportError: If cv2 is not available.
        """
        if not CV2_AVAILABLE:
            raise ImportError("cv2 is required for contrast analysis")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate contrast using standard deviation
        contrast = np.std(gray)

        return float(contrast)

    def is_low_contrast(self, image: np.ndarray, threshold: float = 40.0) -> bool:
        """Determine if image has low contrast.

        Args:
            image: Input image as numpy array.
            threshold: Contrast threshold (below this is low contrast).

        Returns:
            True if image has low contrast, False otherwise.
        """
        contrast = self.analyze(image)
        return contrast < threshold

    def get_quality_rating(self, image: np.ndarray) -> str:
        """Get quality rating based on contrast.

        Args:
            image: Input image as numpy array.

        Returns:
            Quality rating: 'excellent', 'good', 'fair', or 'poor'.
        """
        contrast = self.analyze(image)

        if contrast > 80:
            return "excellent"
        elif contrast > 60:
            return "good"
        elif contrast > 40:
            return "fair"
        else:
            return "poor"
