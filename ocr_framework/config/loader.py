"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path

import yaml

from ocr_framework.config.defaults import default_config
from ocr_framework.config.settings import load_env_settings
from ocr_framework.config.validation import validate_config
from ocr_framework.config.schema import (
    BatchConfig,
    ExportConfig,
    FrameworkConfig,
    LoaderConfig,
    PostprocessingConfig,
    PreprocessingConfig,
    RoutingConfig,
)
from ocr_framework.exceptions import ConfigurationError
from ocr_framework.types import ExportFormat, PreprocessingMode


def load_config(path: Path | None = None, validate: bool = True) -> FrameworkConfig:
    """Load framework configuration from an optional path.

    When ``path`` is not provided, checks ``OCR_CONFIG_PATH`` environment
    variable, then falls back to defaults merged with environment overrides.

    Args:
        path: Optional path to YAML configuration file. If not provided,
            returns default configuration.
        validate: Whether to validate the loaded configuration.

    Returns:
        A FrameworkConfig object loaded from the file or defaults.

    Raises:
        ConfigurationError: If the configuration file cannot be loaded or parsed.
    """
    env = load_env_settings()

    if path is None and env.config_path is not None:
        path = env.config_path

    if path is None:
        config = default_config()
    else:
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data is None:
                config = default_config()
            else:
                config = _parse_config(data)

        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Failed to parse YAML configuration: {exc}") from exc
        except Exception as exc:
            raise ConfigurationError(f"Failed to load configuration: {exc}") from exc

    config = _apply_env_overrides(config, env)

    if validate:
        validate_config(config)

    return config


def _apply_env_overrides(config: FrameworkConfig, env) -> FrameworkConfig:
    """Apply environment variable overrides to configuration."""
    if env.profile:
        config.profile = env.profile
    if env.use_gpu is not None:
        config.use_gpu = env.use_gpu
    return config


def _parse_config(data: dict) -> FrameworkConfig:
    """Parse configuration dictionary into FrameworkConfig object.

    Args:
        data: Configuration dictionary from YAML.

    Returns:
        A FrameworkConfig object.
    """
    config = default_config()

    # Parse top-level fields
    if "profile" in data:
        config.profile = data["profile"]
    if "language" in data:
        config.language = data["language"]
    if "use_gpu" in data:
        config.use_gpu = data["use_gpu"]

    # Parse loader config
    if "loader" in data:
        loader_data = data["loader"]
        config.loader = LoaderConfig(
            pdf_dpi=loader_data.get("pdf_dpi", config.loader.pdf_dpi),
        )

    # Parse preprocessing config
    if "preprocessing" in data:
        prep_data = data["preprocessing"]
        config.preprocessing = PreprocessingConfig(
            mode=PreprocessingMode(prep_data.get("mode", config.preprocessing.mode)),
            steps=prep_data.get("steps", config.preprocessing.steps),
            intelligent_thresholds=prep_data.get(
                "intelligent_thresholds", config.preprocessing.intelligent_thresholds
            ),
        )

    # Parse routing config
    if "routing" in data:
        routing_data = data["routing"]
        config.routing = RoutingConfig(
            primary_engine=routing_data.get("primary_engine", config.routing.primary_engine),
            fallback_chain=routing_data.get("fallback_chain", config.routing.fallback_chain),
            min_confidence=routing_data.get("min_confidence", config.routing.min_confidence),
            rules=routing_data.get("rules", config.routing.rules),
        )

    # Parse postprocessing config
    if "postprocessing" in data:
        post_data = data["postprocessing"]
        config.postprocessing = PostprocessingConfig(
            steps=post_data.get("steps", config.postprocessing.steps),
            spell_check_langs=post_data.get(
                "spell_check_langs", config.postprocessing.spell_check_langs
            ),
            min_confidence=post_data.get("min_confidence", config.postprocessing.min_confidence),
        )

    # Parse export config
    if "export" in data:
        export_data = data["export"]
        formats = export_data.get("formats", config.export.formats)
        parsed_formats = [ExportFormat(f) if isinstance(f, str) else f for f in formats]

        config.export = ExportConfig(
            formats=parsed_formats,
            include_boxes=export_data.get("include_boxes", config.export.include_boxes),
            output_dir=export_data.get("output_dir", config.export.output_dir),
        )

    # Parse batch config
    if "batch" in data:
        batch_data = data["batch"]
        config.batch = BatchConfig(
            workers=batch_data.get("workers", config.batch.workers),
            continue_on_error=batch_data.get(
                "continue_on_error", config.batch.continue_on_error
            ),
            output_format=batch_data.get("output_format", config.batch.output_format),
        )

    # Parse metadata
    if "metadata" in data:
        config.metadata = data["metadata"]

    return config
