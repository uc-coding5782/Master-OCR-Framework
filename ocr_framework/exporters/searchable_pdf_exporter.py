"""Searchable PDF exporter for OCR results."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

from ocr_framework.exporters.base import Exporter
from ocr_framework.exceptions import ExportError
from ocr_framework.models.export_payload import ExportReport
from ocr_framework.models.page_result import DocumentResult

if TYPE_CHECKING:
    from ocr_framework.types import Metadata


class SearchablePDFExporter(Exporter):
    """Export OCR results to searchable PDF format.

    The SearchablePDFExporter creates PDF files with invisible text overlay,
    making them searchable while preserving the original document appearance.
    """

    @property
    def format(self) -> str:
        """Return the export format identifier."""
        return "searchable_pdf"

    def export(
        self,
        result: DocumentResult,
        destination: Path,
        options: Metadata | None = None,
    ) -> ExportReport:
        """Export document result to a searchable PDF file.

        Args:
            result: OCR output to serialize.
            destination: Target output path.
            options: Optional exporter-specific settings. May include:
                - source_pdf: Path to original PDF file (required for PDF input)

        Returns:
            An ExportReport describing the export operation.

        Raises:
            ExportError: If export fails or PyMuPDF is not available.
        """
        if not FITZ_AVAILABLE:
            raise ExportError(
                "PyMuPDF (fitz) is required for searchable PDF export. "
                "Install it with: pip install pymupdf"
            )

        try:
            # Ensure parent directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Get source PDF path from options
            source_pdf = options.get("source_pdf") if options else None
            if not source_pdf:
                raise ExportError("source_pdf option is required for searchable PDF export")

            # Open source PDF
            doc = fitz.open(source_pdf)

            # Add text overlay to each page
            self._add_text_overlay(doc, result)

            # Save to destination
            doc.save(str(destination))
            doc.close()

            # Create export report
            report = ExportReport(
                format=self.format,
                destination=str(destination),
                success=True,
            )

            return report

        except IOError as exc:
            raise ExportError(f"Failed to write searchable PDF: {exc}") from exc
        except Exception as exc:
            raise ExportError(f"Searchable PDF export failed: {exc}") from exc

    def _add_text_overlay(self, doc: fitz.Document, result: DocumentResult) -> None:
        """Add invisible text overlay to PDF pages.

        Args:
            doc: PyMuPDF document object.
            result: Document result with OCR text.
        """
        for page_result in result.pages:
            if page_result.page_index >= doc.page_count:
                continue

            page = doc.load_page(page_result.page_index)

            # Add invisible text for each line
            for line in page_result.lines:
                # Get bounding box if available
                if line.bbox:
                    rect = fitz.Rect(
                        line.bbox.x_min,
                        line.bbox.y_min,
                        line.bbox.x_max,
                        line.bbox.y_max,
                    )
                else:
                    # Use full page if no bounding box
                    rect = page.rect

                # Insert invisible text
                # Text is rendered at 0 opacity (invisible) but still searchable
                page.insert_text(
                    rect.tl,
                    line.text,
                    fontsize=11,
                    color=(0, 0, 0, 0),  # Black with 0 opacity = invisible
                )
