"""Tests for PipelineBuilder with concrete components."""

from pathlib import Path

import pytest

from ocr_framework.pipeline.builder import PipelineBuilder
from ocr_framework.pipeline.runner import PipelineRunner


def test_pipeline_builder_returns_runner() -> None:
    """Test that PipelineBuilder returns a PipelineRunner."""
    builder = PipelineBuilder()
    runner = builder.build()
    assert isinstance(runner, PipelineRunner)


def test_pipeline_builder_with_profile() -> None:
    """Test that PipelineBuilder can set a profile."""
    builder = PipelineBuilder()
    builder.with_profile("document")
    runner = builder.build()
    assert isinstance(runner, PipelineRunner)
    assert runner.config.profile == "document"


def test_pipeline_builder_with_language() -> None:
    """Test that PipelineBuilder can set language."""
    builder = PipelineBuilder()
    builder.with_language("fr")
    runner = builder.build()
    assert isinstance(runner, PipelineRunner)
    assert runner.config.language == "fr"


def test_pipeline_builder_with_paddle_ocr() -> None:
    """Test that PipelineBuilder can wire PaddleOCR components."""
    try:
        builder = PipelineBuilder()
        builder.with_language("en")
        builder.with_paddle_ocr()
        runner = builder.build()

        assert isinstance(runner, PipelineRunner)
        assert runner.components.loader is not None
        assert runner.components.preprocessor is not None
        assert runner.components.detector is not None
        assert runner.components.recognizer is not None
        assert runner.components.postprocessor is not None
    except ImportError:
        pytest.skip("PaddleOCR not installed")


def test_pipeline_builder_with_custom_components() -> None:
    """Test that PipelineBuilder can accept custom components."""
    from ocr_framework.pipeline.components import PipelineComponents
    from ocr_framework.loaders.image_loader import ImageLoader

    builder = PipelineBuilder()
    components = PipelineComponents(loader=ImageLoader())
    builder.with_components(components)
    runner = builder.build()

    assert isinstance(runner, PipelineRunner)
    assert runner.components.loader is not None
    assert runner.components.loader is components.loader
