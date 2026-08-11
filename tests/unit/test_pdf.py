"""Tests for PDF loading and processing."""

import numpy as np
import pytest
from pathlib import Path

from ocr_framework.loaders.pdf_loader import PDFLoader
from ocr_framework.pdf.models import PDFDocument, PDFPage
from ocr_framework.pdf.pdf_loader import PDFLoader as PDFDocumentLoader, FITZ_AVAILABLE, PIL_AVAILABLE


class TestPDFDocument:
    """Tests for PDF document model."""

    def test_creates_pdf_page(self) -> None:
        """Test PDF page creation."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        page = PDFPage(
            page_number=0,
            image=image,
            width=100,
            height=100,
            dpi=300,
            rotation=0.0,
            metadata={"test": "value"},
        )

        assert page.page_number == 0
        assert page.width == 100
        assert page.height == 100
        assert page.dpi == 300
        assert page.rotation == 0.0
        assert page.metadata == {"test": "value"}

    def test_creates_pdf_document(self) -> None:
        """Test PDF document creation."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        pages = [
            PDFPage(page_number=0, image=image, width=100, height=100),
            PDFPage(page_number=1, image=image, width=100, height=100),
        ]

        doc = PDFDocument(
            path="test.pdf",
            page_count=2,
            pages=pages,
            metadata={"format": "pdf"},
        )

        assert doc.path == "test.pdf"
        assert doc.page_count == 2
        assert len(doc.pages) == 2

    def test_validates_page_count(self) -> None:
        """Test page count validation."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        pages = [
            PDFPage(page_number=0, image=image, width=100, height=100),
        ]

        with pytest.raises(ValueError, match="Page count mismatch"):
            PDFDocument(
                path="test.pdf",
                page_count=2,  # Wrong count
                pages=pages,
            )

    def test_get_page(self) -> None:
        """Test getting a specific page."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        pages = [
            PDFPage(page_number=0, image=image, width=100, height=100),
            PDFPage(page_number=1, image=image, width=100, height=100),
        ]

        doc = PDFDocument(path="test.pdf", page_count=2, pages=pages)

        page = doc.get_page(0)
        assert page.page_number == 0

    def test_get_page_out_of_range(self) -> None:
        """Test getting page out of range."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        pages = [
            PDFPage(page_number=0, image=image, width=100, height=100),
        ]

        doc = PDFDocument(path="test.pdf", page_count=1, pages=pages)

        with pytest.raises(IndexError):
            doc.get_page(5)

    def test_get_pages_range(self) -> None:
        """Test getting a range of pages."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        pages = [
            PDFPage(page_number=0, image=image, width=100, height=100),
            PDFPage(page_number=1, image=image, width=100, height=100),
            PDFPage(page_number=2, image=image, width=100, height=100),
        ]

        doc = PDFDocument(path="test.pdf", page_count=3, pages=pages)

        range_pages = doc.get_pages_range(0, 2)
        assert len(range_pages) == 2
        assert range_pages[0].page_number == 0
        assert range_pages[1].page_number == 1


class TestPDFDocumentLoader:
    """Tests for PDF document loader."""

    def test_raises_import_error_without_fitz(self) -> None:
        """Test that PDFLoader raises ImportError without PyMuPDF."""
        if FITZ_AVAILABLE:
            pytest.skip("PyMuPDF is installed")

        with pytest.raises(ImportError, match="PyMuPDF"):
            PDFDocumentLoader()

    def test_raises_import_error_without_pil(self) -> None:
        """Test that PDFLoader raises ImportError without Pillow."""
        if PIL_AVAILABLE:
            pytest.skip("Pillow is installed")

        with pytest.raises(ImportError, match="Pillow"):
            PDFDocumentLoader()

    def test_supports_pdf_extension(self) -> None:
        """Test PDF extension detection."""
        if not FITZ_AVAILABLE or not PIL_AVAILABLE:
            pytest.skip("Dependencies not available")

        loader = PDFDocumentLoader()
        assert loader._detect_format(Path("test.pdf")) == "pdf"

    def test_supports_tiff_extension(self) -> None:
        """Test TIFF extension detection."""
        if not FITZ_AVAILABLE or not PIL_AVAILABLE:
            pytest.skip("Dependencies not available")

        loader = PDFDocumentLoader()
        assert loader._detect_format(Path("test.tif")) == "tiff"
        assert loader._detect_format(Path("test.tiff")) == "tiff"

    def test_detects_unknown_format(self) -> None:
        """Test unknown format detection."""
        if not FITZ_AVAILABLE or not PIL_AVAILABLE:
            pytest.skip("Dependencies not available")

        loader = PDFDocumentLoader()
        assert loader._detect_format(Path("test.txt")) == "unknown"


class TestPDFLoader:
    """Tests for PDF loader integration."""

    def test_supports_pdf_files(self) -> None:
        """Test that PDFLoader supports PDF files."""
        if not FITZ_AVAILABLE or not PIL_AVAILABLE:
            pytest.skip("Dependencies not available")

        loader = PDFLoader()
        assert loader.supports(Path("test.pdf"))
        assert loader.supports(Path("test.PDF"))

    def test_supports_tiff_files(self) -> None:
        """Test that PDFLoader supports TIFF files."""
        if not FITZ_AVAILABLE or not PIL_AVAILABLE:
            pytest.skip("Dependencies not available")

        loader = PDFLoader()
        assert loader.supports(Path("test.tif"))
        assert loader.supports(Path("test.tiff"))

    def test_does_not_support_other_files(self) -> None:
        """Test that PDFLoader does not support other files."""
        if not FITZ_AVAILABLE or not PIL_AVAILABLE:
            pytest.skip("Dependencies not available")

        loader = PDFLoader()
        assert not loader.supports(Path("test.png"))
        assert not loader.supports(Path("test.jpg"))

    def test_get_mime_type_pdf(self) -> None:
        """Test MIME type for PDF."""
        if not FITZ_AVAILABLE or not PIL_AVAILABLE:
            pytest.skip("Dependencies not available")

        loader = PDFLoader()
        mime = loader._get_mime_type(Path("test.pdf"))
        assert mime == "application/pdf"

    def test_get_mime_type_tiff(self) -> None:
        """Test MIME type for TIFF."""
        if not FITZ_AVAILABLE or not PIL_AVAILABLE:
            pytest.skip("Dependencies not available")

        loader = PDFLoader()
        mime = loader._get_mime_type(Path("test.tif"))
        assert mime == "image/tiff"
