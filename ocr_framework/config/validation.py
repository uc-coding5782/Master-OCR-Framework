"""Configuration validation utilities."""

from __future__ import annotations

from ocr_framework.config.schema import FrameworkConfig
from ocr_framework.exceptions import ConfigurationError
from ocr_framework.types import ExportFormat, PreprocessingMode


def validate_config(config: FrameworkConfig) -> None:
    """Validate a framework configuration object.

    Args:
        config: Configuration to validate.

    Raises:
        ConfigurationError: If any field is invalid.
    """
    errors: list[str] = []

    if not config.language:
        errors.append("language must not be empty")

    if config.batch.workers < 1:
        errors.append("batch.workers must be >= 1")

    if not 0.0 <= config.postprocessing.min_confidence <= 1.0:
        errors.append("postprocessing.min_confidence must be between 0.0 and 1.0")

    if not 0.0 <= config.routing.min_confidence <= 1.0:
        errors.append("routing.min_confidence must be between 0.0 and 1.0")

    if config.loader.pdf_dpi < 72:
        errors.append("loader.pdf_dpi must be >= 72")

    valid_prep_modes = {m.value for m in PreprocessingMode}
    if config.preprocessing.mode.value not in valid_prep_modes:
        errors.append(f"preprocessing.mode must be one of {sorted(valid_prep_modes)}")

    valid_formats = {f.value for f in ExportFormat}
    for fmt in config.export.formats:
        if fmt.value not in valid_formats:
            errors.append(f"export.formats contains invalid format: {fmt.value}")

    if errors:
        raise ConfigurationError("Invalid configuration:\n  - " + "\n  - ".join(errors))
