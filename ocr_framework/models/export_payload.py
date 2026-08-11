"""Export result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ocr_framework.types import Metadata


@dataclass
class ExportReport:
    """Result of an export operation."""

    format: str
    destination: Path
    success: bool
    message: str = ""
    metadata: Metadata = field(default_factory=dict)
