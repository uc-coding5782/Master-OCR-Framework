"""Typed exceptions for the OCR framework."""


class OCRFrameworkError(Exception):
    """Base exception for all framework errors."""


class ConfigurationError(OCRFrameworkError):
    """Raised when configuration is invalid or cannot be loaded."""


class LoaderError(OCRFrameworkError):
    """Raised when document loading fails."""


class PreprocessingError(OCRFrameworkError):
    """Raised when image preprocessing fails."""


class DetectionError(OCRFrameworkError):
    """Raised when text detection fails."""


class RecognitionError(OCRFrameworkError):
    """Raised when text recognition fails."""


class ExportError(OCRFrameworkError):
    """Raised when result export fails."""


class PipelineError(OCRFrameworkError):
    """Raised when pipeline orchestration fails."""


class RoutingError(OCRFrameworkError):
    """Raised when engine routing fails."""


class PluginError(OCRFrameworkError):
    """Raised when plugin registration or execution fails."""
