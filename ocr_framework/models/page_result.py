"""Aggregated page- and document-level OCR results."""

from __future__ import annotations

from dataclasses import dataclass, field

from ocr_framework.models.document import Document
from ocr_framework.models.ocr_result import DetectionRegion, OCRLine
from ocr_framework.types import Metadata


@dataclass
class RoutingDecision:
    """Audit record describing an engine routing choice.

    Attributes:
        primary_engine: Selected primary engine identifier.
        fallback_engines: Ordered fallback engine identifiers.
        reason: Human-readable explanation for the routing decision.
        max_retries: Maximum number of retries allowed for this decision.
    """

    primary_engine: str
    fallback_engines: list[str] = field(default_factory=list)
    reason: str = ""
    max_retries: int = 0


@dataclass
class PageResult:
    """OCR output for a single document page.

    Attributes:
        page_index: Zero-based index of the processed page.
        lines: Recognized text lines in reading order.
        detections: Raw detection regions produced before recognition.
        engine_used: Recognizer engine identifier for this page.
        routing_decisions: Routing audit records applied to this page.
        aggregate_confidence: Mean or weighted confidence for the page.
        timings: Stage-level timing metrics in seconds.
        metadata: Additional page result metadata.
    """

    page_index: int
    lines: list[OCRLine] = field(default_factory=list)
    detections: list[DetectionRegion] = field(default_factory=list)
    engine_used: str = ""
    routing_decisions: list[RoutingDecision] = field(default_factory=list)
    aggregate_confidence: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)


@dataclass
class DocumentResult:
    """Complete OCR output for an entire document.

    Attributes:
        document: Source document that was processed.
        pages: Ordered page results for the document.
        metadata: Additional document result metadata.
    """

    document: Document
    pages: list[PageResult] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        """Return the number of processed pages.

        Returns:
            Total processed page count.
        """
        return len(self.pages)
