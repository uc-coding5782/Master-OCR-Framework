"""Configuration package for the OCR framework."""

from ocr_framework.config.defaults import default_config
from ocr_framework.config.loader import load_config
from ocr_framework.config.profiles import PROFILE_NAMES, get_profile
from ocr_framework.config.schema import (
    BatchConfig,
    ExportConfig,
    FrameworkConfig,
    LoaderConfig,
    PostprocessingConfig,
    PreprocessingConfig,
    RoutingConfig,
)

__all__ = [
    "BatchConfig",
    "ExportConfig",
    "FrameworkConfig",
    "LoaderConfig",
    "PostprocessingConfig",
    "PreprocessingConfig",
    "PROFILE_NAMES",
    "RoutingConfig",
    "default_config",
    "get_profile",
    "load_config",
]
