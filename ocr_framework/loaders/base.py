"""Abstract interface for document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ocr_framework.models.document import Document
from ocr_framework.types import Metadata


class DocumentLoader(ABC):
    """Load files from disk into framework ``Document`` objects.

    Implementations must declare supported file types via ``supports`` and
    convert supported inputs into one or more ``Page`` instances.
    """

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Indicate whether this loader can handle the given path.

        Args:
            path: Candidate input file path.

        Returns:
            ``True`` if the loader can process the file, otherwise ``False``.
        """

    @abstractmethod
    def load(self, path: Path, options: Metadata | None = None) -> Document:
        """Load a document from disk.

        Args:
            path: Input file path.
            options: Optional loader-specific settings.

        Returns:
            A populated ``Document`` instance.

        Raises:
            LoaderError: If the file cannot be loaded.
        """
