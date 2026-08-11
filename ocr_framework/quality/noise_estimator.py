"""Noise estimation for image quality assessment."""

from __future__ import annotations

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class NoiseEstimator:
    """Estimate image noise using standard deviation of high-frequency components."""

    def estimate(self, image: np.ndarray) -> float:
        """Estimate noise level using standard deviation.

        Args:
            image: Input image as numpy array (grayscale or BGR).

        Returns:
            Noise level (higher is more noisy).

        Raises:
            ImportError: If cv2 is not available.
        """
        if not CV2_AVAILABLE:
            raise ImportError("cv2 is required for noise estimation")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate noise using standard deviation of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = np.std(laplacian)

        return float(noise_level)

    def is_noisy(self, image: np.ndarray, threshold: float = 15.0) -> bool:
        """Determine if image is noisy.

        Args:
            image: Input image as numpy array.
            threshold: Noise threshold (above this is considered noisy).

        Returns:
            True if image is noisy, False otherwise.
        """
        noise_level = self.estimate(image)
        return noise_level > threshold

    def get_quality_rating(self, image: np.ndarray) -> str:
        """Get quality rating based on noise level.

        Args:
            image: Input image as numpy array.

        Returns:
            Quality rating: 'excellent', 'good', 'fair', or 'poor'.
        """
        noise_level = self.estimate(image)

        if noise_level < 5:
            return "excellent"
        elif noise_level < 10:
            return "good"
        elif noise_level < 15:
            return "fair"
        else:
            return "poor"
