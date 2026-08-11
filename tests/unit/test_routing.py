"""Tests for intelligent engine routing components."""

import pytest

from ocr_framework.config.schema import FrameworkConfig, RoutingConfig
from ocr_framework.models.bbox import BoundingBox
from ocr_framework.models.ocr_result import OCRLine
from ocr_framework.models.page_result import PageResult
from ocr_framework.routing.confidence_manager import ConfidenceManager
from ocr_framework.routing.engine_selector import EngineSelector
from ocr_framework.routing.retry_manager import RetryManager


@pytest.fixture
def framework_config() -> FrameworkConfig:
    """Create a test framework configuration."""
    config = FrameworkConfig()
    config.routing = RoutingConfig(
        primary_engine="paddle",
        fallback_chain=["trocr"],
        min_confidence=0.6,
    )
    return config


@pytest.fixture
def sample_page_result() -> PageResult:
    """Create a sample page result for testing."""
    lines = [
        OCRLine(
            text="Hello world",
            confidence=0.9,
            bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0),
            engine_name="paddle",
        ),
        OCRLine(
            text="Test text",
            confidence=0.4,
            bbox=BoundingBox(x_min=0.0, y_min=20.0, x_max=100.0, y_max=40.0),
            engine_name="paddle",
        ),
    ]
    page_result = PageResult(page_index=0, lines=lines)
    page_result.aggregate_confidence = 0.65  # Average of 0.9 and 0.4
    return page_result


class TestEngineSelector:
    """Tests for EngineSelector."""

    def test_selects_primary_engine_by_default(self, framework_config: FrameworkConfig) -> None:
        """Test that primary engine is selected by default."""
        selector = EngineSelector(framework_config)
        engine = selector.select_engine()

        assert engine == "paddle"

    def test_selects_primary_with_good_confidence(self, framework_config: FrameworkConfig) -> None:
        """Test that engine with good confidence is kept."""
        selector = EngineSelector(framework_config)
        page_result = PageResult(
            page_index=0,
            lines=[
                OCRLine(
                    text="Test",
                    confidence=0.8,
                    bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0),
                    engine_name="paddle",
                )
            ],
        )

        engine = selector.select_engine(page_result=page_result)
        assert engine == "paddle"

    def test_should_fallback_on_low_confidence(self, framework_config: FrameworkConfig, sample_page_result: PageResult) -> None:
        """Test that fallback is triggered on low confidence."""
        selector = EngineSelector(framework_config)
        sample_page_result.aggregate_confidence = 0.4

        should_fallback = selector.should_fallback(sample_page_result, "paddle")
        assert should_fallback is True

    def test_should_not_fallback_on_high_confidence(self, framework_config: FrameworkConfig) -> None:
        """Test that fallback is not triggered on high confidence."""
        selector = EngineSelector(framework_config)
        page_result = PageResult(
            page_index=0,
            lines=[
                OCRLine(
                    text="Test",
                    confidence=0.9,
                    bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0),
                    engine_name="paddle",
                )
            ],
        )
        page_result.aggregate_confidence = 0.9

        should_fallback = selector.should_fallback(page_result, "paddle")
        assert should_fallback is False

    def test_get_fallback_engine(self, framework_config: FrameworkConfig) -> None:
        """Test fallback engine selection."""
        # Ensure fallback chain is set before creating selector
        framework_config.routing.fallback_chain = ["trocr"]
        selector = EngineSelector(framework_config)

        fallback = selector.get_fallback_engine("paddle")
        assert fallback == "trocr"

    def test_get_fallback_exhausted(self, framework_config: FrameworkConfig) -> None:
        """Test that empty string is returned when fallback chain is exhausted."""
        selector = EngineSelector(framework_config)

        fallback = selector.get_fallback_engine("trocr")
        assert fallback == ""


class TestConfidenceManager:
    """Tests for ConfidenceManager."""

    def test_is_acceptable_with_good_confidence(self, framework_config: FrameworkConfig) -> None:
        """Test that high confidence is acceptable."""
        manager = ConfidenceManager(framework_config)
        page_result = PageResult(
            page_index=0,
            lines=[
                OCRLine(
                    text="Test",
                    confidence=0.8,
                    bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0),
                    engine_name="paddle",
                )
            ],
        )
        page_result.aggregate_confidence = 0.8

        assert manager.is_acceptable(page_result) is True

    def test_is_not_acceptable_with_low_confidence(self, framework_config: FrameworkConfig) -> None:
        """Test that low confidence is not acceptable."""
        manager = ConfidenceManager(framework_config)
        page_result = PageResult(
            page_index=0,
            lines=[
                OCRLine(
                    text="Test",
                    confidence=0.4,
                    bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0),
                    engine_name="paddle",
                )
            ],
        )
        page_result.aggregate_confidence = 0.4

        assert manager.is_acceptable(page_result) is False

    def test_is_high_quality(self, framework_config: FrameworkConfig) -> None:
        """Test high quality detection."""
        manager = ConfidenceManager(framework_config)
        page_result = PageResult(
            page_index=0,
            lines=[
                OCRLine(
                    text="Test",
                    confidence=0.9,
                    bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0),
                    engine_name="paddle",
                )
            ],
        )
        page_result.aggregate_confidence = 0.9

        assert manager.is_high_quality(page_result) is True

    def test_is_low_quality(self, framework_config: FrameworkConfig) -> None:
        """Test low quality detection."""
        manager = ConfidenceManager(framework_config)
        page_result = PageResult(
            page_index=0,
            lines=[
                OCRLine(
                    text="Test",
                    confidence=0.3,
                    bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0),
                    engine_name="paddle",
                )
            ],
        )
        page_result.aggregate_confidence = 0.3

        assert manager.is_low_quality(page_result) is True

    def test_get_confidence_summary(self, framework_config: FrameworkConfig) -> None:
        """Test confidence summary generation."""
        manager = ConfidenceManager(framework_config)
        page_result = PageResult(
            page_index=0,
            lines=[
                OCRLine(
                    text="Test 1",
                    confidence=0.8,
                    bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0),
                    engine_name="paddle",
                ),
                OCRLine(
                    text="Test 2",
                    confidence=0.6,
                    bbox=BoundingBox(x_min=0.0, y_min=20.0, x_max=100.0, y_max=40.0),
                    engine_name="paddle",
                ),
            ],
        )
        page_result.aggregate_confidence = 0.7

        summary = manager.get_confidence_summary(page_result)

        assert summary["aggregate"] == 0.7
        assert summary["min"] == 0.6
        assert summary["max"] == 0.8
        assert summary["mean"] == 0.7
        assert summary["line_count"] == 2

    def test_get_confidence_summary_empty(self, framework_config: FrameworkConfig) -> None:
        """Test confidence summary with empty result."""
        manager = ConfidenceManager(framework_config)
        page_result = PageResult(page_index=0, lines=[])

        summary = manager.get_confidence_summary(page_result)

        assert summary["aggregate"] == 0.0
        assert summary["line_count"] == 0

    def test_filter_low_confidence_lines(self, framework_config: FrameworkConfig) -> None:
        """Test filtering low-confidence lines."""
        manager = ConfidenceManager(framework_config)
        page_result = PageResult(
            page_index=0,
            lines=[
                OCRLine(
                    text="High confidence",
                    confidence=0.9,
                    bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0),
                    engine_name="paddle",
                ),
                OCRLine(
                    text="Low confidence",
                    confidence=0.3,
                    bbox=BoundingBox(x_min=0.0, y_min=20.0, x_max=100.0, y_max=40.0),
                    engine_name="paddle",
                ),
            ],
        )

        filtered = manager.filter_low_confidence_lines(page_result)
        assert len(filtered) == 1
        assert filtered[0].text == "High confidence"


class TestRetryManager:
    """Tests for RetryManager."""

    def test_should_retry_on_low_confidence(self, framework_config: FrameworkConfig) -> None:
        """Test that retry is triggered on low confidence."""
        manager = RetryManager(framework_config)
        page_result = PageResult(
            page_index=0,
            lines=[
                OCRLine(
                    text="Test",
                    confidence=0.3,
                    bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0),
                    engine_name="paddle",
                )
            ],
        )
        page_result.aggregate_confidence = 0.3

        assert manager.should_retry(page_result, attempt=0) is True

    def test_should_not_retry_on_high_confidence(self, framework_config: FrameworkConfig) -> None:
        """Test that retry is not triggered on high confidence."""
        manager = RetryManager(framework_config)
        page_result = PageResult(
            page_index=0,
            lines=[
                OCRLine(
                    text="Test",
                    confidence=0.9,
                    bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0),
                    engine_name="paddle",
                )
            ],
        )
        page_result.aggregate_confidence = 0.9

        assert manager.should_retry(page_result, attempt=0) is False

    def test_should_not_retry_on_empty_results(self, framework_config: FrameworkConfig) -> None:
        """Test that retry is triggered on empty results."""
        manager = RetryManager(framework_config)
        page_result = PageResult(page_index=0, lines=[])

        assert manager.should_retry(page_result, attempt=0) is True

    def test_should_not_retry_exceeded_attempts(self, framework_config: FrameworkConfig) -> None:
        """Test that retry stops after max attempts."""
        manager = RetryManager(framework_config)
        page_result = PageResult(
            page_index=0,
            lines=[
                OCRLine(
                    text="Test",
                    confidence=0.3,
                    bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0),
                    engine_name="paddle",
                )
            ],
        )

        assert manager.should_retry(page_result, attempt=10) is False

    def test_get_next_engine(self, framework_config: FrameworkConfig) -> None:
        """Test getting next engine in fallback chain."""
        manager = RetryManager(framework_config)

        next_engine = manager.get_next_engine("paddle", attempt=1)
        assert next_engine == "trocr"

    def test_get_next_engine_cyclic(self, framework_config: FrameworkConfig) -> None:
        """Test that fallback chain cycles."""
        manager = RetryManager(framework_config)

        # Should cycle through fallbacks
        engine1 = manager.get_next_engine("paddle", attempt=1)
        engine2 = manager.get_next_engine("paddle", attempt=2)

        assert engine1 == "trocr"
        assert engine2 == "trocr"  # Only one fallback, so it cycles

    def test_get_max_retries(self, framework_config: FrameworkConfig) -> None:
        """Test getting max retry attempts."""
        manager = RetryManager(framework_config)

        max_retries = manager.get_max_retries()
        assert max_retries == 1  # One fallback in test config

    def test_record_attempt(self, framework_config: FrameworkConfig) -> None:
        """Test recording retry attempts."""
        manager = RetryManager(framework_config)

        record = manager.record_attempt(
            engine="paddle",
            attempt=1,
            confidence=0.5,
            success=False,
        )

        assert record["engine"] == "paddle"
        assert record["attempt"] == 1
        assert record["confidence"] == 0.5
        assert record["success"] is False
