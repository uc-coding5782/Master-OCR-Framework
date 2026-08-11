"""Rotation detection for image quality assessment."""

from __future__ import annotations

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class RotationDetector:
    """Detect image rotation using text orientation analysis."""

    def detect(self, image: np.ndarray) -> float:
        """Detect rotation angle using text orientation.

        Args:
            image: Input image as numpy array (grayscale or BGR).

        Returns:
            Rotation angle in degrees (0 = upright, positive = clockwise).

        Raises:
            ImportError: If cv2 is not available.
        """
        if not CV2_AVAILABLE:
            raise ImportError("cv2 is required for rotation detection")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find all contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Find minimum area rectangle for each contour
        angles = []
        for contour in contours:
            if len(contour) > 5:
                rect = cv2.minAreaRect(contour)
                angle = rect[-1]
                angles.append(angle)

        if not angles:
            return 0.0

        # Use median angle to be robust to outliers
        rotation = np.median(angles)

        # Normalize angle to [-90, 90]
        if rotation < -45:
            rotation += 90
        elif rotation > 45:
            rotation -= 90

        return float(rotation)

    def is_rotated(self, image: np.ndarray, threshold: float = 5.0) -> bool:
        """Determine if image is significantly rotated.

        Args:
            image: Input image as numpy array.
            threshold: Rotation threshold in degrees.

        Returns:
            True if image is rotated beyond threshold, False otherwise.
        """
        rotation = self.detect(image)
        return abs(rotation) > threshold

    def get_rotation_direction(self, image: np.ndarray) -> str:
        """Get rotation direction.

        Args:
            image: Input image as numpy array.

        Returns:
            Rotation direction: 'clockwise', 'counterclockwise', or 'none'.
        """
        rotation = self.detect(image)

        if rotation > 5:
            return "clockwise"
        elif rotation < -5:
            return "counterclockwise"
        else:
            return "none"
