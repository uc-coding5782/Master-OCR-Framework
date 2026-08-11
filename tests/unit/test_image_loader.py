"""Tests for ImageLoader."""

from pathlib import Path

import numpy as np
import pytest

from ocr_framework.exceptions import LoaderError
from ocr_framework.loaders.image_loader import ImageLoader
from ocr_framework.models.image import ImagePayload
from ocr_framework.types import ColorSpace


def test_image_loader_supports_common_formats(tmp_path: Path) -> None:
    """Test that ImageLoader supports common image formats."""
    loader = ImageLoader()

    supported_extensions = [
        ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"
    ]

    for ext in supported_extensions:
        test_file = tmp_path / f"test{ext}"
        test_file.touch()
        assert loader.supports(test_file), f"Should support {ext}"


def test_image_loader_does_not_support_unsupported_formats(tmp_path: Path) -> None:
    """Test that ImageLoader rejects unsupported formats."""
    loader = ImageLoader()

    unsupported_extensions = [".pdf", ".docx", ".txt", ".json"]

    for ext in unsupported_extensions:
        test_file = tmp_path / f"test{ext}"
        test_file.touch()
        assert not loader.supports(test_file), f"Should not support {ext}"


def test_image_loader_loads_valid_image(tmp_path: Path) -> None:
    """Test that ImageLoader can load a valid image."""
    loader = ImageLoader()

    # Create a simple test image using OpenCV
    test_image = tmp_path / "test.png"
    test_array = np.zeros((100, 200, 3), dtype=np.uint8)
    import cv2
    cv2.imwrite(str(test_image), test_array)

    document = loader.load(test_image)

    assert document.page_count == 1
    assert document.source_path == test_image
    assert document.mime_type == "image/png"

    page = document.pages[0]
    assert page.page_index == 0
    assert isinstance(page.image, ImagePayload)
    assert page.image.width == 200
    assert page.image.height == 100
    assert page.image.channels == 3
    assert page.image.color_space == ColorSpace.BGR


def test_image_loader_raises_on_nonexistent_file() -> None:
    """Test that ImageLoader raises LoaderError for nonexistent files."""
    loader = ImageLoader()
    nonexistent = Path("/nonexistent/path/image.png")

    with pytest.raises(LoaderError):
        loader.load(nonexistent)
