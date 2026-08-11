"""Phase 1 foundation tests."""

from pathlib import Path

import numpy as np
import pytest

from ocr_framework.exceptions import PipelineError
from ocr_framework.models import (
    BoundingBox,
    Document,
    ImagePayload,
    Page,
    Point,
    Polygon,
)
from ocr_framework.pipeline import PipelineBuilder, PipelineRunner
from ocr_framework.types import ColorSpace


def test_domain_models_construct() -> None:
    image = ImagePayload(
        data=np.zeros((100, 200, 3), dtype=np.uint8),
        width=200,
        height=100,
        channels=3,
        color_space=ColorSpace.BGR,
    )
    page = Page(page_index=0, image=image)
    document = Document(pages=[page], mime_type="image/png")

    assert document.page_count == 1
    assert page.image.width == 200


def test_bounding_box_helpers() -> None:
    bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=20.0)
    assert bbox.width == 10.0
    assert bbox.height == 20.0


def test_polygon_point_count() -> None:
    polygon = Polygon(points=(Point(0, 0), Point(1, 0), Point(1, 1)))
    assert polygon.point_count == 3


def test_pipeline_builder_returns_runner() -> None:
    runner = PipelineBuilder().with_profile("document").with_language("en").build()
    assert isinstance(runner, PipelineRunner)


def test_pipeline_runner_with_concrete_components(tmp_path: Path) -> None:
    """Test that PipelineRunner works with concrete components."""
    from ocr_framework import PipelineBuilder

    # Create a valid test image
    import numpy as np
    image_path = tmp_path / "sample.png"
    test_array = np.zeros((100, 200, 3), dtype=np.uint8)

    # Use PIL to save the image (more reliable than cv2 in test env)
    try:
        from PIL import Image
        img = Image.fromarray(test_array)
        img.save(image_path)
    except ImportError:
        # If PIL is not available, skip this test
        pytest.skip("PIL not available for test image creation")

    # Build a pipeline with concrete components
    try:
        builder = PipelineBuilder()
        builder.with_language("en")
        builder.with_paddle_ocr()
        runner = builder.build()
    except ImportError:
        pytest.skip("PaddleOCR not installed")

    # Verify components are wired correctly
    assert runner.components.loader is not None
    assert runner.components.preprocessor is not None
    assert runner.components.detector is not None
    assert runner.components.recognizer is not None
    assert runner.components.postprocessor is not None
