"""Output directory and filename management for batch processing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ocr_framework.exceptions import PipelineError

if TYPE_CHECKING:
    from ocr_framework.models.job import BatchJob


class OutputManager:
    """Manage output directory creation and filename preservation for batch jobs.

    The OutputManager ensures that output directories exist and generates
    appropriate output filenames while preserving the original input filenames.
    """

    def __init__(self, output_dir: Path, output_format: str = "txt") -> None:
        """Initialize the output manager.

        Args:
            output_dir: Target output directory path.
            output_format: File extension for output files (e.g., 'txt', 'json').

        Raises:
            PipelineError: If the output directory cannot be created.
        """
        self._output_dir = Path(output_dir)
        self._output_format = output_format.lower().lstrip(".")
        self._ensure_output_directory()

    def _ensure_output_directory(self) -> None:
        """Create the output directory if it doesn't exist.

        Raises:
            PipelineError: If directory creation fails.
        """
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PipelineError(f"Failed to create output directory {self._output_dir}: {exc}") from exc

    def get_output_path(self, input_path: Path) -> Path:
        """Generate output path for a given input file.

        Args:
            input_path: Input file path.

        Returns:
            Output file path with preserved base name and new extension.
        """
        # Preserve the original filename but change extension
        output_name = input_path.stem + f".{self._output_format}"
        return self._output_dir / output_name

    def get_output_dir(self) -> Path:
        """Return the output directory path.

        Returns:
            The output directory path.
        """
        return self._output_dir

    def exists(self, output_path: Path) -> bool:
        """Check if an output file already exists.

        Args:
            output_path: Output file path to check.

        Returns:
            True if the file exists, False otherwise.
        """
        return output_path.exists()
