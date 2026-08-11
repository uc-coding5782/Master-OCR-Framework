"""Blur estimation for image quality assessment."""

from __future__ import annotations

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class BlurEstimator:
    """Estimate image blur using Laplacian variance.

    The Laplacian variance method measures the amount of high-frequency
    content in an image. Lower values indicate more blur.
    """

    def estimate(self, image: np.ndarray) -> float:
        """Estimate blur score using Laplacian variance.

        Args:
            image: Input image as numpy array (grayscale or BGR).

        Returns:
            Blur score (higher is sharper, lower is more blurred).

        Raises:
            ImportError: If cv2 is not available.
        """
        if not CV2_AVAILABLE:
            raise ImportError("cv2 is required for blur estimation")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Return variance
        return float(laplacian.var())

    def is_blurry(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        """Determine if image is blurry.

        Args:
            image: Input image as numpy array.
            threshold: Blur threshold (below this is considered blurry).

        Returns:
            True if image is blurry, False otherwise.
        """
        blur_score = self.estimate(image)
        return blur_score < threshold

    def get_quality_rating(self, image: np.ndarray) -> str:
        """Get quality rating based on blur score.

        Args:
            image: Input image as numpy array.

        Returns:
            Quality rating: 'excellent', 'good', 'fair', or 'poor'.
        """
        blur_score = self.estimate(image)

        if blur_score > 500:
            return "excellent"
        elif blur_score > 200:
            return "good"
        elif blur_score > 100:
            return "fair"
        else:
            return "poor"
