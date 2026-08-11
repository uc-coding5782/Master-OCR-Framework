"""Tests for exporters."""

import json
import pytest
from pathlib import Path

from ocr_framework.exceptions import ExportError
from ocr_framework.exporters.docx_exporter import DOCXExporter, DOCX_AVAILABLE
from ocr_framework.exporters.json_exporter import JSONExporter
from ocr_framework.exporters.searchable_pdf_exporter import SearchablePDFExporter, FITZ_AVAILABLE
from ocr_framework.exporters.txt_exporter import TXTExporter
from ocr_framework.models.page_result import DocumentResult, PageResult
from ocr_framework.models.ocr_result import OCRLine
from ocr_framework.models.bbox import BoundingBox


@pytest.fixture
def sample_document_result() -> DocumentResult:
    """Create a sample document result."""
    lines = [
        OCRLine(text="Line 1", confidence=0.95, bbox=BoundingBox(x_min=10.0, y_min=10.0, x_max=110.0, y_max=30.0)),
        OCRLine(text="Line 2", confidence=0.90, bbox=BoundingBox(x_min=10.0, y_min=40.0, x_max=110.0, y_max=60.0)),
    ]

    page = PageResult(
        page_index=0,
        lines=lines,
        metadata={"test": "value"},
    )

    return DocumentResult(
        document=None,
        pages=[page],
        metadata={"document": "test"},
    )


class TestTXTExporter:
    """Tests for TXTExporter."""

    def test_exporter_format(self) -> None:
        """Test exporter format identifier."""
        exporter = TXTExporter()
        assert exporter.format == "txt"

    def test_exports_to_txt(self, sample_document_result: DocumentResult, tmp_path: Path) -> None:
        """Test TXT export."""
        exporter = TXTExporter()
        destination = tmp_path / "output.txt"

        report = exporter.export(sample_document_result, destination)

        assert report.success
        assert report.format == "txt"
        assert destination.exists()

    def test_creates_parent_directory(self, sample_document_result: DocumentResult, tmp_path: Path) -> None:
        """Test that parent directory is created."""
        exporter = TXTExporter()
        destination = tmp_path / "subdir" / "output.txt"

        report = exporter.export(sample_document_result, destination)

        assert report.success
        assert destination.exists()

    def test_handles_multiple_pages(self, tmp_path: Path) -> None:
        """Test handling of multiple pages."""
        lines1 = [OCRLine(text="Page 1 Line 1", confidence=0.95, bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0))]
        lines2 = [OCRLine(text="Page 2 Line 1", confidence=0.95, bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=20.0))]

        page1 = PageResult(page_index=0, lines=lines1)
        page2 = PageResult(page_index=1, lines=lines2)

        doc = DocumentResult(document=None, pages=[page1, page2], metadata={})

        exporter = TXTExporter()
        destination = tmp_path / "output.txt"

        report = exporter.export(doc, destination)

        assert report.success
        assert destination.exists()

        content = destination.read_text(encoding="utf-8")
        assert "--- Page 1 ---" in content
        assert "--- Page 2 ---" in content


class TestJSONExporter:
    """Tests for JSONExporter."""

    def test_exporter_format(self) -> None:
        """Test exporter format identifier."""
        exporter = JSONExporter()
        assert exporter.format == "json"

    def test_exports_to_json(self, sample_document_result: DocumentResult, tmp_path: Path) -> None:
        """Test JSON export."""
        exporter = JSONExporter()
        destination = tmp_path / "output.json"

        report = exporter.export(sample_document_result, destination)

        assert report.success
        assert report.format == "json"
        assert destination.exists()

        # Check content
        with open(destination, encoding="utf-8") as f:
            data = json.load(f)

        assert "pages" in data
        assert len(data["pages"]) == 1
        assert data["pages"][0]["lines"][0]["text"] == "Line 1"
        assert data["pages"][0]["lines"][0]["confidence"] == 0.95
        assert "bounding_box" in data["pages"][0]["lines"][0]
        assert data["pages"][0]["lines"][0]["bounding_box"]["x_min"] == 10.0

    def test_includes_metadata(self, sample_document_result: DocumentResult, tmp_path: Path) -> None:
        """Test that metadata is included in JSON."""
        exporter = JSONExporter()
        destination = tmp_path / "output.json"

        exporter.export(sample_document_result, destination)

        with open(destination, encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["document"] == "test"
        assert data["pages"][0]["metadata"]["test"] == "value"


class TestDOCXExporter:
    """Tests for DOCXExporter."""

    def test_exporter_format(self) -> None:
        """Test exporter format identifier."""
        exporter = DOCXExporter()
        assert exporter.format == "docx"

    def test_raises_import_error_without_docx(self) -> None:
        """Test that DOCXExporter raises ImportError without python-docx."""
        if DOCX_AVAILABLE:
            pytest.skip("python-docx is installed")

        exporter = DOCXExporter()
        with pytest.raises((ImportError, ExportError), match="python-docx"):
            exporter.export(
                DocumentResult(document=None, pages=[], metadata={}),
                Path("output.docx"),
            )

    def test_exports_to_docx(self, sample_document_result: DocumentResult, tmp_path: Path) -> None:
        """Test DOCX export."""
        if not DOCX_AVAILABLE:
            pytest.skip("python-docx not available")

        exporter = DOCXExporter()
        destination = tmp_path / "output.docx"

        report = exporter.export(sample_document_result, destination)

        assert report.success
        assert report.format == "docx"
        assert destination.exists()


class TestSearchablePDFExporter:
    """Tests for SearchablePDFExporter."""

    def test_exporter_format(self) -> None:
        """Test exporter format identifier."""
        exporter = SearchablePDFExporter()
        assert exporter.format == "searchable_pdf"

    def test_raises_import_error_without_fitz(self) -> None:
        """Test that SearchablePDFExporter raises ImportError without PyMuPDF."""
        if FITZ_AVAILABLE:
            pytest.skip("PyMuPDF is installed")

        exporter = SearchablePDFExporter()
        with pytest.raises((ImportError, ExportError), match="PyMuPDF"):
            exporter.export(
                DocumentResult(document=None, pages=[], metadata={}),
                Path("output.pdf"),
            )

    def test_requires_source_pdf_option(self, sample_document_result: DocumentResult, tmp_path: Path) -> None:
        """Test that source_pdf option is required."""
        if not FITZ_AVAILABLE:
            pytest.skip("PyMuPDF not available")

        exporter = SearchablePDFExporter()
        destination = tmp_path / "output.pdf"

        with pytest.raises(Exception, match="source_pdf"):
            exporter.export(sample_document_result, destination, options={})
