"""Adaptive preprocessing based on image quality analysis."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ocr_framework.models.image import ImagePayload
from ocr_framework.pipeline.context import PipelineContext
from ocr_framework.preprocessing.base import Preprocessor
from ocr_framework.preprocessing.composite import CompositePreprocessor
from ocr_framework.preprocessing.steps.contrast import ContrastStep
from ocr_framework.preprocessing.steps.deskew import DeskewStep
from ocr_framework.preprocessing.steps.denoise import DenoiseStep
from ocr_framework.preprocessing.steps.upscale import UpscaleStep

if TYPE_CHECKING:
    from ocr_framework.quality.quality_analyzer import QualityAnalyzer


class AdaptivePreprocessor(Preprocessor):
    """Adaptive preprocessor that selects steps based on image quality.

    The AdaptivePreprocessor analyzes image quality and dynamically
    selects which preprocessing steps to apply, avoiding unnecessary
    processing for images that don't need it.
    """

    def __init__(self, min_dim: int = 1000) -> None:
        """Initialize the adaptive preprocessor.

        Args:
            min_dim: Minimum dimension for upscaling.
        """
        self._min_dim = min_dim
        self._quality_analyzer: QualityAnalyzer | None = None
        self._steps_cache: dict[str, Preprocessor] = {}

    @property
    def name(self) -> str:
        """Return the preprocessor identifier."""
        return "adaptive_preprocessor"

    def process(self, image: ImagePayload, context: PipelineContext) -> ImagePayload:
        """Process image with adaptive preprocessing based on quality analysis.

        Args:
            image: Input image payload.
            context: Pipeline execution context.

        Returns:
            Processed image payload.
        """
        # Lazy load quality analyzer
        if self._quality_analyzer is None:
            from ocr_framework.quality.quality_analyzer import QualityAnalyzer
            self._quality_analyzer = QualityAnalyzer()

        # Analyze image quality
        recommendations = self._quality_analyzer.needs_preprocessing(image.data)

        # Build dynamic preprocessing pipeline
        steps = self._build_pipeline(recommendations)

        if not steps:
            # No preprocessing needed
            return image

        # Create composite preprocessor
        composite = CompositePreprocessor(steps=steps)

        # Process the image
        return composite.process(image, context)

    def _build_pipeline(self, recommendations: dict) -> list[Preprocessor]:
        """Build preprocessing pipeline based on recommendations.

        Args:
            recommendations: Preprocessing recommendations from quality analysis.

        Returns:
            List of preprocessing steps to apply.
        """
        steps = []

        # Upscale if needed
        if recommendations.get("upscale", False):
            steps.append(self._get_step("upscale"))

        # Denoise if needed
        if recommendations.get("denoise", False):
            steps.append(self._get_step("denoise"))

        # Deskew if needed
        if recommendations.get("deskew", False):
            steps.append(self._get_step("deskew"))

        # Adjust contrast if needed
        if recommendations.get("contrast_enhancement", False):
            steps.append(self._get_step("contrast"))

        return steps

    def _get_step(self, step_name: str) -> Preprocessor:
        """Get or create a preprocessing step (cached).

        Args:
            step_name: Name of the preprocessing step.

        Returns:
            Preprocessor instance.
        """
        if step_name not in self._steps_cache:
            if step_name == "upscale":
                self._steps_cache[step_name] = UpscaleStep(min_dim=self._min_dim)
            elif step_name == "denoise":
                self._steps_cache[step_name] = DenoiseStep()
            elif step_name == "deskew":
                self._steps_cache[step_name] = DeskewStep(min_angle_threshold=0.5)
            elif step_name == "contrast":
                self._steps_cache[step_name] = ContrastStep()
            else:
                raise ValueError(f"Unknown preprocessing step: {step_name}")

        return self._steps_cache[step_name]

    def get_quality_report(self, image: ImagePayload) -> dict:
        """Get quality analysis report for the image.

        Args:
            image: Input image payload.

        Returns:
            Quality analysis report.
        """
        if self._quality_analyzer is None:
            from ocr_framework.quality.quality_analyzer import QualityAnalyzer
            self._quality_analyzer = QualityAnalyzer()
        return self._quality_analyzer.analyze(image.data)

    def get_applied_steps(self, image: ImagePayload) -> list[str]:
        """Get list of preprocessing steps that would be applied.

        Args:
            image: Input image payload.

        Returns:
            List of step names that would be applied.
        """
        if self._quality_analyzer is None:
            from ocr_framework.quality.quality_analyzer import QualityAnalyzer
            self._quality_analyzer = QualityAnalyzer()
        recommendations = self._quality_analyzer.needs_preprocessing(image.data)
        steps = self._build_pipeline(recommendations)
        return [step.name for step in steps]
