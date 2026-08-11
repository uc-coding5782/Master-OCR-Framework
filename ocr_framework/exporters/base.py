"""Abstract interface for result exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ocr_framework.models.export_payload import ExportReport
from ocr_framework.models.page_result import DocumentResult
from ocr_framework.types import Metadata


class Exporter(ABC):
    """Serialize ``DocumentResult`` objects to external formats.

    Exporters are output adapters only and must not perform OCR processing.
    """

    @property
    @abstractmethod
    def format(self) -> str:
        """Return the export format identifier.

        Returns:
            A short format name such as ``json`` or ``txt``.
        """

    @abstractmethod
    def export(
        self,
        result: DocumentResult,
        destination: Path,
        options: Metadata | None = None,
    ) -> ExportReport:
        """Write a document result to disk.

        Args:
            result: OCR output to serialize.
            destination: Target output path.
            options: Optional exporter-specific settings.

        Returns:
            An ``ExportReport`` describing the export operation.

        Raises:
            ExportError: If export fails.
        """
