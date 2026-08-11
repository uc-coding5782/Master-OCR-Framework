"""PDF loader for multi-page document support."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from ocr_framework.exceptions import PipelineError
from ocr_framework.pdf.models import PDFDocument, PDFPage

if TYPE_CHECKING:
    from ocr_framework.config.schema import FrameworkConfig


class PDFLoader:
    """Load PDF and TIFF documents for OCR processing.

    The PDFLoader supports both native PDFs and scanned PDFs,
    extracting each page as an image for OCR processing.
    """

    def __init__(self, config: FrameworkConfig | None = None) -> None:
        """Initialize the PDF loader.

        Args:
            config: Framework configuration with PDF settings.

        Raises:
            ImportError: If PyMuPDF is not installed.
        """
        if not FITZ_AVAILABLE:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF support. "
                "Install it with: pip install pymupdf"
            )

        if not PIL_AVAILABLE:
            raise ImportError(
                "Pillow is required for PDF support. "
                "Install it with: pip install Pillow"
            )

        self._config = config

    def load(self, file_path: Path | str) -> PDFDocument:
        """Load a PDF or TIFF document.

        Args:
            file_path: Path to the PDF or TIFF file.

        Returns:
            PDFDocument with extracted pages.

        Raises:
            PipelineError: If file cannot be loaded.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise PipelineError(f"File not found: {file_path}")

        try:
            # Open the document
            doc = fitz.open(str(file_path))

            # Extract pages
            pages = []
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)

                # Get page image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                image = pix.tobytes("png")

                # Convert to numpy array
                image_array = np.array(PILImage.open(BytesIO(image)))

                # Get page metadata
                width = image_array.shape[1]
                height = image_array.shape[0]
                dpi = self._get_page_dpi(page)

                pdf_page = PDFPage(
                    page_number=page_num,
                    image=image_array,
                    width=width,
                    height=height,
                    dpi=dpi,
                    rotation=page.rotation,
                    metadata={
                        "media_box": page.mediabox,
                        "crop_box": page.cropbox,
                    },
                )
                pages.append(pdf_page)

            doc.close()

            return PDFDocument(
                path=str(file_path),
                page_count=len(pages),
                pages=pages,
                metadata={
                    "format": self._detect_format(file_path),
                },
            )

        except Exception as exc:
            raise PipelineError(f"Failed to load PDF: {exc}") from exc

    def _get_page_dpi(self, page) -> int | None:
        """Get DPI for a page.

        Args:
            page: PyMuPDF page object.

        Returns:
            DPI value or None if not available.
        """
        try:
            dpi = page.get_dpi()
            return int(dpi[0]) if dpi else None
        except Exception:
            return None

    def _detect_format(self, file_path: Path) -> str:
        """Detect the document format.

        Args:
            file_path: Path to the file.

        Returns:
            Format string (e.g., 'pdf', 'tiff').
        """
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        elif suffix in [".tif", ".tiff"]:
            return "tiff"
        else:
            return "unknown"

    def load_page(self, file_path: Path | str, page_number: int) -> PDFPage:
        """Load a single page from a PDF document.

        Args:
            file_path: Path to the PDF file.
            page_number: Zero-based page index.

        Returns:
            PDFPage object.

        Raises:
            PipelineError: If page cannot be loaded.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise PipelineError(f"File not found: {file_path}")

        try:
            doc = fitz.open(str(file_path))

            if page_number < 0 or page_number >= doc.page_count:
                doc.close()
                raise PipelineError(f"Page number {page_number} out of range (0-{doc.page_count - 1})")

            page = doc.load_page(page_number)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image = pix.tobytes("png")

            image_array = np.array(PILImage.open(BytesIO(image)))

            width = image_array.shape[1]
            height = image_array.shape[0]
            dpi = self._get_page_dpi(page)

            pdf_page = PDFPage(
                page_number=page_number,
                image=image_array,
                width=width,
                height=height,
                dpi=dpi,
                rotation=page.rotation,
                metadata={
                    "media_box": page.mediabox,
                    "crop_box": page.cropbox,
                },
            )

            doc.close()

            return pdf_page

        except Exception as exc:
            raise PipelineError(f"Failed to load page {page_number}: {exc}") from exc
