"""JSON exporter for OCR results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ocr_framework.exporters.base import Exporter
from ocr_framework.exceptions import ExportError
from ocr_framework.models.export_payload import ExportReport
from ocr_framework.models.page_result import DocumentResult

if TYPE_CHECKING:
    from ocr_framework.types import Metadata


class JSONExporter(Exporter):
    """Export OCR results to JSON format.

    The JSONExporter produces structured JSON files with complete OCR results,
    including text, bounding boxes, confidence scores, and metadata.
    """

    @property
    def format(self) -> str:
        """Return the export format identifier."""
        return "json"

    def export(
        self,
        result: DocumentResult,
        destination: Path,
        options: Metadata | None = None,
    ) -> ExportReport:
        """Export document result to a JSON file.

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

            # Build JSON content
            data = self._build_json_content(result)

            # Write to file
            with open(destination, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Create export report
            report = ExportReport(
                format=self.format,
                destination=str(destination),
                success=True,
            )

            return report

        except IOError as exc:
            raise ExportError(f"Failed to write JSON file: {exc}") from exc
        except Exception as exc:
            raise ExportError(f"JSON export failed: {exc}") from exc

    def _build_json_content(self, result: DocumentResult) -> dict:
        """Build JSON content from document result.

        Args:
            result: Document result to convert to JSON.

        Returns:
            Dictionary with structured OCR data.
        """
        pages_data = []

        for page_result in result.pages:
            lines_data = []

            for line in page_result.lines:
                line_data = {
                    "text": line.text,
                    "confidence": line.confidence,
                }

                # Add bounding box if available
                if line.bbox:
                    line_data["bounding_box"] = {
                        "x_min": line.bbox.x_min,
                        "y_min": line.bbox.y_min,
                        "x_max": line.bbox.x_max,
                        "y_max": line.bbox.y_max,
                    }

                lines_data.append(line_data)

            page_data = {
                "page_index": page_result.page_index,
                "lines": lines_data,
            }

            # Add page-level metadata if available
            if page_result.metadata:
                page_data["metadata"] = page_result.metadata

            pages_data.append(page_data)

        data = {
            "pages": pages_data,
            "page_count": len(result.pages),
            "metadata": result.metadata or {},
        }

        return data
