"""Image file loader implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ocr_framework.exceptions import LoaderError
from ocr_framework.loaders.base import DocumentLoader
from ocr_framework.models.document import Document, Page
from ocr_framework.models.image import ImagePayload
from ocr_framework.types import ColorSpace, Metadata


class ImageLoader(DocumentLoader):
    """Load single image files into framework Document objects.

    Supports common image formats including PNG, JPEG, BMP, and TIFF.
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

    def supports(self, path: Path) -> bool:
        """Check if the loader supports the given file path.

        Args:
            path: Candidate input file path.

        Returns:
            True if the file has a supported image extension.
        """
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, path: Path, options: Metadata | None = None) -> Document:
        """Load an image file into a Document.

        Args:
            path: Input image file path.
            options: Optional loader-specific settings (currently unused).

        Returns:
            A Document with a single Page containing the loaded image.

        Raises:
            LoaderError: If the image cannot be loaded.
        """
        _ = options  # Reserved for future use

        if not path.exists():
            raise LoaderError(f"Image file not found: {path}")

        try:
            # Load image using OpenCV
            img_array = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img_array is None:
                raise LoaderError(f"Failed to read image file: {path}")

            # Get image dimensions
            height, width = img_array.shape[:2]
            channels = img_array.shape[2] if len(img_array.shape) == 3 else 1

            # Create ImagePayload
            image = ImagePayload(
                data=img_array,
                width=width,
                height=height,
                channels=channels,
                color_space=ColorSpace.BGR,  # OpenCV loads as BGR
            )

            # Create Page
            page = Page(page_index=0, image=image)

            # Create Document
            document = Document(
                pages=[page],
                source_path=path,
                mime_type=self._get_mime_type(path),
            )

            return document

        except LoaderError:
            raise
        except Exception as exc:
            raise LoaderError(f"Failed to load image {path}: {exc}") from exc

    def _get_mime_type(self, path: Path) -> str:
        """Get MIME type for the image file.

        Args:
            path: Image file path.

        Returns:
            MIME type string.
        """
        ext = path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".webp": "image/webp",
        }
        return mime_map.get(ext, "application/octet-stream")
