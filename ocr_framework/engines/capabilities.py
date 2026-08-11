"""Shared engine capability metadata.

Concrete engine metadata will be populated in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EngineCapabilities:
    """Describe supported features of an OCR engine implementation."""

    name: str
    languages: tuple[str, ...] = field(default_factory=tuple)
    supports_handwriting: bool = False
    supports_gpu: bool = False
    supports_layout: bool = False
    supports_detection: bool = True
    supports_recognition: bool = True
