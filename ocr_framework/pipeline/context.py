"""Mutable execution context passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ocr_framework.config.schema import FrameworkConfig
from ocr_framework.models.document import Document, Page
from ocr_framework.models.export_payload import ExportReport
from ocr_framework.models.image import ImagePayload
from ocr_framework.models.ocr_result import DetectionRegion, OCRLine
from ocr_framework.models.page_result import DocumentResult, PageResult, RoutingDecision
from ocr_framework.types import Metadata


@dataclass
class PipelineContext:
    """Shared mutable state for a single pipeline execution.

    ``PipelineRunner`` populates this object as each stage executes. Stage
    methods read prior results and write their own outputs back into the
    context for subsequent stages.

    Attributes:
        input_path: Source file path being processed.
        config: Active framework configuration.
        export_destination: Optional export target path set before export.
        document: Loaded document populated by ``load()``.
        current_page_index: Zero-based index of the page being processed.
        current_page: Active page reference for per-page stages.
        processed_image: Image produced by ``preprocess()``.
        signals: Image-quality or routing signals collected during execution.
        detections: Detection regions produced by ``detect()``.
        recognized_lines: OCR lines produced by ``recognize()``.
        current_page_result: Page result built before ``postprocess()``.
        page_results: Completed page results accumulated across the document.
        routing_decisions: Routing audit records for the current execution.
        document_result: Final assembled document result.
        export_report: Export report populated by ``export()``.
        metrics: Timing and operational metrics.
        metadata: Additional execution metadata.
    """

    input_path: Path
    config: FrameworkConfig
    export_destination: Path | None = None
    document: Document | None = None
    current_page_index: int = 0
    current_page: Page | None = None
    processed_image: ImagePayload | None = None
    signals: Metadata = field(default_factory=dict)
    detections: list[DetectionRegion] = field(default_factory=list)
    recognized_lines: list[OCRLine] = field(default_factory=list)
    current_page_result: PageResult | None = None
    page_results: list[PageResult] = field(default_factory=list)
    routing_decisions: list[RoutingDecision] = field(default_factory=list)
    document_result: DocumentResult | None = None
    export_report: ExportReport | None = None
    metrics: Metadata = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        """Return the number of pages in the loaded document.

        Returns:
            Page count, or ``0`` if no document has been loaded yet.
        """
        if self.document is None:
            return 0
        return self.document.page_count

    @property
    def active_image(self) -> ImagePayload | None:
        """Return the best available image for OCR stages.

        Returns:
            The preprocessed image when available, otherwise the current page
            image. Returns ``None`` if no page is active.
        """
        if self.processed_image is not None:
            return self.processed_image
        if self.current_page is not None:
            return self.current_page.image
        return None
