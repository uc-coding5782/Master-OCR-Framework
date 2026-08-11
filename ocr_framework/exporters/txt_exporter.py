"""TXT exporter for OCR results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ocr_framework.exporters.base import Exporter
from ocr_framework.exceptions import ExportError
from ocr_framework.models.export_payload import ExportReport
from ocr_framework.models.page_result import DocumentResult

if TYPE_CHECKING:
    from ocr_framework.types import Metadata


class TXTExporter(Exporter):
    """Export OCR results to plain text format.

    The TXTExporter produces simple text files with recognized content,
    suitable for human reading and basic text processing.
    """

    @property
    def format(self) -> str:
        """Return the export format identifier."""
        return "txt"

    def export(
        self,
        result: DocumentResult,
        destination: Path,
        options: Metadata | None = None,
    ) -> ExportReport:
        """Export document result to a text file.

        Args:
            result: OCR output to serialize.
            destination: Target output path.
            options: Optional exporter-specific settings.

        Returns:
            An ExportReport describing the export operation.

        Raises:
            ExportError: If export fails.
        """
        _ = options  # Reserved for future use

        try:
            # Ensure parent directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Build text content
            lines = self._build_text_content(result)

            # Write to file
            with open(destination, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            # Create export report
            report = ExportReport(
                format=self.format,
                destination=str(destination),
                success=True,
            )

            return report

        except IOError as exc:
            raise ExportError(f"Failed to write TXT file: {exc}") from exc
        except Exception as exc:
            raise ExportError(f"TXT export failed: {exc}") from exc

    def _build_text_content(self, result: DocumentResult) -> list[str]:
        """Build text content from document result.

        Args:
            result: Document result to convert to text.

        Returns:
            List of text lines.
        """
        lines = []

        for page_result in result.pages:
            # Add page separator if multiple pages
            if len(result.pages) > 1:
                lines.append(f"--- Page {page_result.page_index + 1} ---")

            # Add each line of text
            for line in page_result.lines:
                lines.append(line.text)

            # Add blank line between pages
            if len(result.pages) > 1:
                lines.append("")

        return lines
