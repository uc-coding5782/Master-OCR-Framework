"""Tests for preprocessing steps."""

import numpy as np
import pytest

from ocr_framework.models.image import ImagePayload
from ocr_framework.pipeline.context import PipelineContext
from ocr_framework.preprocessing.composite import CompositePreprocessor
from ocr_framework.preprocessing.steps.contrast import ContrastStep
from ocr_framework.preprocessing.steps.deskew import DeskewStep
from ocr_framework.preprocessing.steps.denoise import DenoiseStep
from ocr_framework.preprocessing.steps.upscale import UpscaleStep
from ocr_framework.types import ColorSpace


@pytest.fixture
def sample_image() -> ImagePayload:
    """Create a sample image payload for testing."""
    data = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    return ImagePayload(
        data=data,
        width=200,
        height=100,
        channels=3,
        color_space=ColorSpace.BGR,
    )


@pytest.fixture
def sample_context() -> PipelineContext:
    """Create a sample pipeline context for testing."""
    from ocr_framework.config.schema import FrameworkConfig
    from pathlib import Path

    return PipelineContext(
        input_path=Path("dummy.png"),
        config=FrameworkConfig(),
    )


def test_denoise_step_properties() -> None:
    """Test DenoiseStep properties."""
    step = DenoiseStep()
    assert step.name == "denoise"


def test_denoise_step_processes_image(sample_image: ImagePayload, sample_context: PipelineContext) -> None:
    """Test that DenoiseStep processes an image."""
    step = DenoiseStep()
    result = step.process(sample_image, sample_context)

    assert isinstance(result, ImagePayload)
    assert result.width == sample_image.width
    assert result.height == sample_image.height
    assert result.channels == sample_image.channels
    assert result.metadata.get("denoised") is True


def test_deskew_step_properties() -> None:
    """Test DeskewStep properties."""
    step = DeskewStep()
    assert step.name == "deskew"


def test_deskew_step_processes_image(sample_image: ImagePayload, sample_context: PipelineContext) -> None:
    """Test that DeskewStep processes an image."""
    step = DeskewStep()
    result = step.process(sample_image, sample_context)

    assert isinstance(result, ImagePayload)
    assert result.width == sample_image.width
    assert result.height == sample_image.height
    assert result.channels == sample_image.channels


def test_contrast_step_properties() -> None:
    """Test ContrastStep properties."""
    step = ContrastStep()
    assert step.name == "contrast"


def test_contrast_step_processes_image(sample_image: ImagePayload, sample_context: PipelineContext) -> None:
    """Test that ContrastStep processes an image."""
    step = ContrastStep()
    result = step.process(sample_image, sample_context)

    assert isinstance(result, ImagePayload)
    assert result.width == sample_image.width
    assert result.height == sample_image.height
    assert result.channels == sample_image.channels
    assert result.metadata.get("contrast_enhanced") is True


def test_upscale_step_properties() -> None:
    """Test UpscaleStep properties."""
    step = UpscaleStep()
    assert step.name == "upscale"


def test_upscale_step_processes_small_image(sample_context: PipelineContext) -> None:
    """Test that UpscaleStep upscales small images."""
    data = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
    small_image = ImagePayload(
        data=data,
        width=100,
        height=50,
        channels=3,
        color_space=ColorSpace.BGR,
    )

    step = UpscaleStep(min_dim=1000)
    result = step.process(small_image, sample_context)

    assert isinstance(result, ImagePayload)
    assert result.width >= 100  # Should be upscaled
    assert result.height >= 50
    assert result.metadata.get("upscaled") is True
    assert result.metadata.get("scale_factor") > 1.0


def test_upscale_step_skips_large_images(sample_image: ImagePayload, sample_context: PipelineContext) -> None:
    """Test that UpscaleStep skips images that are already large enough."""
    step = UpscaleStep(min_dim=50)
    result = step.process(sample_image, sample_context)

    assert isinstance(result, ImagePayload)
    assert result.width == sample_image.width
    assert result.height == sample_image.height
    assert result.metadata.get("upscaled") is None  # Should not be upscaled


def test_composite_preprocessor_properties() -> None:
    """Test CompositePreprocessor properties."""
    steps = [DenoiseStep(), ContrastStep()]
    preprocessor = CompositePreprocessor(steps=steps)
    assert preprocessor.name == "composite"


def test_composite_preprocessor_chains_steps(sample_image: ImagePayload, sample_context: PipelineContext) -> None:
    """Test that CompositePreprocessor chains multiple steps."""
    steps = [DenoiseStep(), ContrastStep()]
    preprocessor = CompositePreprocessor(steps=steps)
    result = preprocessor.process(sample_image, sample_context)

    assert isinstance(result, ImagePayload)
    assert result.metadata.get("denoised") is True
    assert result.metadata.get("contrast_enhanced") is True
