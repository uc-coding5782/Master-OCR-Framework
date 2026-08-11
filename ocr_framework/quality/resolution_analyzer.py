"""Resolution analysis for image quality assessment."""

from __future__ import annotations

import numpy as np


class ResolutionAnalyzer:
    """Analyze image resolution for OCR suitability."""

    def analyze(self, image: np.ndarray) -> dict:
        """Analyze image resolution metrics.

        Args:
            image: Input image as numpy array.

        Returns:
            Dictionary with resolution metrics.
        """
        height, width = image.shape[:2]

        return {
            "width": width,
            "height": height,
            "total_pixels": width * height,
            "aspect_ratio": width / height if height > 0 else 0.0,
        }

    def is_low_resolution(self, image: np.ndarray, min_width: int = 300, min_height: int = 300) -> bool:
        """Determine if image has low resolution.

        Args:
            image: Input image as numpy array.
            min_width: Minimum acceptable width.
            min_height: Minimum acceptable height.

        Returns:
            True if image has low resolution, False otherwise.
        """
        metrics = self.analyze(image)
        return metrics["width"] < min_width or metrics["height"] < min_height

    def get_quality_rating(self, image: np.ndarray) -> str:
        """Get quality rating based on resolution.

        Args:
            image: Input image as numpy array.

        Returns:
            Quality rating: 'excellent', 'good', 'fair', or 'poor'.
        """
        metrics = self.analyze(image)
        total_pixels = metrics["total_pixels"]

        if total_pixels > 2000000:  # > 2MP
            return "excellent"
        elif total_pixels > 1000000:  # > 1MP
            return "good"
        elif total_pixels > 500000:  # > 0.5MP
            return "fair"
        else:
            return "poor"

    def get_dpi(self, image: np.ndarray, physical_width_mm: float = 210.0) -> float:
        """Estimate DPI based on physical dimensions.

        Args:
            image: Input image as numpy array.
            physical_width_mm: Physical width in millimeters (default A4 width).

        Returns:
            Estimated DPI.
        """
        metrics = self.analyze(image)
        width_pixels = metrics["width"]

        # Convert mm to inches
        width_inches = physical_width_mm / 25.4

        # Calculate DPI
        dpi = width_pixels / width_inches if width_inches > 0 else 0.0

        return dpi
