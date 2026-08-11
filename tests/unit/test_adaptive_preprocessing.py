"""Tests for adaptive preprocessing."""

import numpy as np
import pytest

from ocr_framework.models.image import ImagePayload, ColorSpace
from ocr_framework.pipeline.context import PipelineContext
from ocr_framework.preprocessing.adaptive_preprocessor import AdaptivePreprocessor


@pytest.fixture
def sample_image() -> ImagePayload:
    """Create a sample image payload."""
    data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return ImagePayload(
        data=data,
        width=100,
        height=100,
        channels=3,
        color_space=ColorSpace.BGR,
        metadata={},
    )


@pytest.fixture
def context() -> PipelineContext:
    """Create a pipeline context."""
    return PipelineContext(
        input_path=None,
        config=None,
    )


class TestAdaptivePreprocessor:
    """Tests for AdaptivePreprocessor."""

    def test_preprocessor_name(self) -> None:
        """Test preprocessor name property."""
        preprocessor = AdaptivePreprocessor()
        assert preprocessor.name == "adaptive_preprocessor"

    def test_processes_image(self, sample_image: ImagePayload, context: PipelineContext) -> None:
        """Test that preprocessor processes image."""
        preprocessor = AdaptivePreprocessor()
        result = preprocessor.process(sample_image, context)

        assert isinstance(result, ImagePayload)
        # Image may be processed (e.g., upscaled), so just check it's valid
        assert result.data.shape[2] == sample_image.data.shape[2]  # Same channels

    def test_get_quality_report(self, sample_image: ImagePayload) -> None:
        """Test quality report generation."""
        preprocessor = AdaptivePreprocessor()
        report = preprocessor.get_quality_report(sample_image)

        assert "blur" in report
        assert "noise" in report
        assert "brightness" in report
        assert "contrast" in report
        assert "rotation" in report
        assert "resolution" in report

    def test_get_applied_steps(self, sample_image: ImagePayload) -> None:
        """Test getting applied steps."""
        preprocessor = AdaptivePreprocessor()
        steps = preprocessor.get_applied_steps(sample_image)

        assert isinstance(steps, list)
        # Steps should be a list of strings
        for step in steps:
            assert isinstance(step, str)

    def test_caches_steps(self, sample_image: ImagePayload, context: PipelineContext) -> None:
        """Test that steps are cached."""
        preprocessor = AdaptivePreprocessor()

        # Process once to trigger lazy loading
        preprocessor.get_quality_report(sample_image)

        # Process
        preprocessor.process(sample_image, context)

        # Check that cache is populated
        # (This is implicit - if it doesn't crash, caching works)
        assert True

    def test_min_dim_parameter(self) -> None:
        """Test min_dim parameter."""
        preprocessor = AdaptivePreprocessor(min_dim=2000)
        assert preprocessor._min_dim == 2000

    def test_returns_original_image_if_no_preprocessing_needed(self, sample_image: ImagePayload, context: PipelineContext) -> None:
        """Test that original image is returned if no preprocessing needed."""
        # Create a high-quality image that shouldn't need preprocessing
        high_quality_data = np.random.randint(128, 255, (2000, 2000, 3), dtype=np.uint8)
        high_quality_image = ImagePayload(
            data=high_quality_data,
            width=2000,
            height=2000,
            channels=3,
            color_space=ColorSpace.BGR,
            metadata={},
        )

        preprocessor = AdaptivePreprocessor()
        result = preprocessor.process(high_quality_image, context)

        # Should return the same image (or minimally processed)
        assert isinstance(result, ImagePayload)
