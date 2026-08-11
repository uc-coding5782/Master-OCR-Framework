"""Default configuration values."""

from ocr_framework.config.schema import FrameworkConfig


def default_config() -> FrameworkConfig:
    """Return a framework configuration with default values."""
    return FrameworkConfig()
