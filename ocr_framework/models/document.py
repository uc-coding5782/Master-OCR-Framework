"""Document and page domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from ocr_framework.models.image import ImagePayload
from ocr_framework.types import Metadata


@dataclass
class Page:
    """A single page within a loaded document.

    Attributes:
        page_index: Zero-based page index within the parent document.
        image: Raw page image payload.
        preprocessing_trace: Ordered audit records of preprocessing steps.
        metadata: Additional page-level metadata.
    """

    page_index: int
    image: ImagePayload
    preprocessing_trace: list[Metadata] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)


@dataclass
class Document:
    """A loaded input document composed of one or more pages.

    Attributes:
        pages: Ordered list of document pages.
        document_id: Unique identifier for this document instance.
        source_path: Original file path, if loaded from disk.
        mime_type: MIME type of the source document.
        metadata: Additional document-level metadata.
    """

    pages: list[Page]
    document_id: str = field(default_factory=lambda: str(uuid4()))
    source_path: Path | None = None
    mime_type: str = "application/octet-stream"
    metadata: Metadata = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        """Return the number of pages in the document.

        Returns:
            Total page count.
        """
        return len(self.pages)
