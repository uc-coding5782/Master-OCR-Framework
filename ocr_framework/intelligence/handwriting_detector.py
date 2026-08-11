"""Handwriting detection for document intelligence."""

from __future__ import annotations

import numpy as np
from collections import Counter

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class HandwritingDetector:
    """Detect handwritten text using heuristics.

    This detector uses image analysis heuristics to distinguish between
    printed and handwritten text. It can be extended with ML models
    for higher accuracy.
    """

    def detect(self, image: np.ndarray) -> bool:
        """Detect if image contains handwritten text.

        Args:
            image: Input image as numpy array.

        Returns:
            True if handwriting is detected, False otherwise.

        Raises:
            ImportError: If cv2 is not available.
        """
        if not CV2_AVAILABLE:
            raise ImportError("cv2 is required for handwriting detection")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate edge density (handwriting tends to have more irregular edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

        # Calculate variance in stroke width (handwriting is more variable)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_variance = np.var(sobel_x) + np.var(sobel_y)

        # Heuristic: handwriting typically has higher edge density and gradient variance
        is_handwriting = edge_density > 0.05 and gradient_variance > 1000

        return bool(is_handwriting)

    def get_confidence(self, image: np.ndarray) -> float:
        """Get confidence score for handwriting detection.

        Args:
            image: Input image as numpy array.

        Returns:
            Confidence score (0.0-1.0).

        Raises:
            ImportError: If cv2 is not available.
        """
        if not CV2_AVAILABLE:
            raise ImportError("cv2 is required for handwriting detection")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

        # Normalize to 0-1 range
        confidence = min(edge_density * 10, 1.0)

        return float(confidence)
