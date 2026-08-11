"""PDF document model for multi-page document support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from numpy.typing import NDArray

from ocr_framework.types import Metadata


@dataclass
class PDFPage:
    """A single page from a PDF document.

    Attributes:
        page_number: Zero-based page index.
        image: Page image as numpy array.
        width: Page width in pixels.
        height: Page height in pixels.
        dpi: DPI resolution if available.
        rotation: Rotation angle in degrees.
        metadata: Additional page-level metadata.
    """

    page_number: int
    image: NDArray[Any]
    width: int
    height: int
    dpi: int | None = None
    rotation: float = 0.0
    metadata: Metadata = field(default_factory=dict)


@dataclass
class PDFDocument:
    """A multi-page PDF document.

    Attributes:
        path: Path to the original PDF file.
        page_count: Total number of pages.
        pages: List of PDFPage objects.
        metadata: Document-level metadata.
    """

    path: str
    page_count: int
    pages: list[PDFPage] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate document invariants."""
        if self.page_count != len(self.pages):
            raise ValueError(f"Page count mismatch: {self.page_count} != {len(self.pages)}")

    def get_page(self, page_number: int) -> PDFPage:
        """Get a specific page by number.

        Args:
            page_number: Zero-based page index.

        Returns:
            PDFPage object.

        Raises:
            IndexError: If page number is out of range.
        """
        if page_number < 0 or page_number >= self.page_count:
            raise IndexError(f"Page number {page_number} out of range (0-{self.page_count - 1})")
        return self.pages[page_number]

    def get_pages_range(self, start: int, end: int) -> list[PDFPage]:
        """Get a range of pages.

        Args:
            start: Start page number (inclusive).
            end: End page number (exclusive).

        Returns:
            List of PDFPage objects.
        """
        if start < 0 or end > self.page_count or start >= end:
            raise IndexError(f"Invalid page range: {start}-{end}")
        return self.pages[start:end]
