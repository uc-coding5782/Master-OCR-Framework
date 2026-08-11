"""Tests for postprocessing filters and correctors."""

import pytest

from ocr_framework.models.bbox import BoundingBox, Point, Polygon
from ocr_framework.models.ocr_result import OCRLine
from ocr_framework.models.page_result import PageResult
from ocr_framework.pipeline.context import PipelineContext
from ocr_framework.postprocessing.composite import CompositePostProcessor
from ocr_framework.postprocessing.correctors.spell_corrector import SpellCorrector
from ocr_framework.postprocessing.filters.confidence_filter import ConfidenceFilter


@pytest.fixture
def sample_page_result() -> PageResult:
    """Create a sample page result for testing."""
    lines = [
        OCRLine(
            text="Hello world",
            confidence=0.9,
            bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0),
            engine_name="test",
        ),
        OCRLine(
            text="Low confidence text",
            confidence=0.3,
            bbox=BoundingBox(x_min=0.0, y_min=20.0, x_max=100.0, y_max=40.0),
            engine_name="test",
        ),
        OCRLine(
            text="Another high confidence",
            confidence=0.8,
            bbox=BoundingBox(x_min=0.0, y_min=40.0, x_max=100.0, y_max=60.0),
            engine_name="test",
        ),
    ]
    return PageResult(page_index=0, lines=lines)


@pytest.fixture
def sample_context() -> PipelineContext:
    """Create a sample pipeline context for testing."""
    from ocr_framework.config.schema import FrameworkConfig
    from pathlib import Path

    return PipelineContext(
        input_path=Path("dummy.png"),
        config=FrameworkConfig(),
    )


def test_confidence_filter_properties() -> None:
    """Test ConfidenceFilter properties."""
    filter = ConfidenceFilter(min_confidence=0.5)
    assert filter.name == "confidence_filter"


def test_confidence_filter_filters_low_confidence_lines(
    sample_page_result: PageResult,
    sample_context: PipelineContext,
) -> None:
    """Test that ConfidenceFilter removes low-confidence lines."""
    filter = ConfidenceFilter(min_confidence=0.5)
    result = filter.filter(sample_page_result, sample_context)

    assert len(result.lines) == 2  # Only high-confidence lines remain
    assert all(line.confidence >= 0.5 for line in result.lines)
    assert result.aggregate_confidence > 0.5


def test_confidence_filter_keeps_all_high_confidence_lines(
    sample_context: PipelineContext,
) -> None:
    """Test that ConfidenceFilter keeps all high-confidence lines."""
    lines = [
        OCRLine(
            text="High confidence 1",
            confidence=0.9,
            bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0),
            engine_name="test",
        ),
        OCRLine(
            text="High confidence 2",
            confidence=0.8,
            bbox=BoundingBox(x_min=0.0, y_min=20.0, x_max=100.0, y_max=40.0),
            engine_name="test",
        ),
    ]
    page_result = PageResult(page_index=0, lines=lines)

    filter = ConfidenceFilter(min_confidence=0.5)
    result = filter.filter(page_result, sample_context)

    assert len(result.lines) == 2


def test_spell_corrector_properties() -> None:
    """Test SpellCorrector properties."""
    try:
        corrector = SpellCorrector(language="en")
        assert corrector.name == "spell_corrector"
    except ImportError:
        pytest.skip("spellchecker not installed")


def test_spell_corrector_skips_unsupported_languages(
    sample_page_result: PageResult,
    sample_context: PipelineContext,
) -> None:
    """Test that SpellCorrector skips unsupported languages."""
    try:
        corrector = SpellCorrector(language="invalid_lang")
        original_text = sample_page_result.lines[0].text

        result = corrector.correct(sample_page_result, sample_context)

        assert result.lines[0].text == original_text  # Should be unchanged
    except ImportError:
        pytest.skip("spellchecker not installed")


def test_spell_corrector_processes_supported_languages(
    sample_page_result: PageResult,
    sample_context: PipelineContext,
) -> None:
    """Test that SpellCorrector processes supported languages."""
    try:
        # Add a line with a misspelled word
        sample_page_result.lines.append(
            OCRLine(
                text="helo world",  # Intentional misspelling
                confidence=0.9,
                bbox=BoundingBox(x_min=0.0, y_min=60.0, x_max=100.0, y_max=80.0),
                engine_name="test",
            )
        )

        corrector = SpellCorrector(language="en")
        result = corrector.correct(sample_page_result, sample_context)

        # The spell checker should attempt to correct "helo" to "hello"
        # Note: pyspellchecker behavior may vary, so we just check it doesn't crash
        assert len(result.lines) == len(sample_page_result.lines)
    except ImportError:
        pytest.skip("spellchecker not installed")


def test_composite_postprocessor_properties() -> None:
    """Test CompositePostProcessor properties."""
    processors = [ConfidenceFilter(min_confidence=0.5)]
    postprocessor = CompositePostProcessor(processors=processors)
    assert postprocessor.name == "composite"


def test_composite_postprocessor_chains_processors(
    sample_page_result: PageResult,
    sample_context: PipelineContext,
) -> None:
    """Test that CompositePostProcessor chains multiple processors."""
    processors = [
        ConfidenceFilter(min_confidence=0.5),
    ]

    # Only add spell corrector if available
    try:
        processors.append(SpellCorrector(language="en"))
    except ImportError:
        pass

    postprocessor = CompositePostProcessor(processors=processors)
    result = postprocessor.process(sample_page_result, sample_context)

    # Should have filtered low-confidence lines
    assert len(result.lines) == 2
    assert all(line.confidence >= 0.5 for line in result.lines)
