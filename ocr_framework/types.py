"""Shared type aliases and enumerations."""

from __future__ import annotations

from enum import Enum
from typing import Any, TypeAlias

Metadata: TypeAlias = dict[str, Any]


class ColorSpace(str, Enum):
    """Supported color space identifiers for image payloads."""

    BGR = "BGR"
    RGB = "RGB"
    GRAY = "GRAY"


class ExportFormat(str, Enum):
    """Supported export format identifiers."""

    TXT = "txt"
    JSON = "json"
    CSV = "csv"
    DOCX = "docx"
    SEARCHABLE_PDF = "searchable_pdf"


class PreprocessingMode(str, Enum):
    """Preprocessing execution modes."""

    NONE = "none"
    FIXED = "fixed"
    INTELLIGENT = "intelligent"


class JobStatus(str, Enum):
    """Batch job lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RegionType(str, Enum):
    """Detection region classification."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    UNKNOWN = "unknown"
