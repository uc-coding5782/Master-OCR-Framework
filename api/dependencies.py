"""Shared FastAPI dependencies."""

from __future__ import annotations

from functools import lru_cache

from ocr_framework import __version__, create_paddle_pipeline
from ocr_framework.config.loader import load_config
from ocr_framework.config.settings import load_env_settings
from ocr_framework.observability.logging import configure_logging
from ocr_framework.pipeline.runner import PipelineRunner
from ocr_framework.utils.gpu import is_gpu_available, resolve_use_gpu


def setup_logging() -> None:
    """Initialize logging from environment settings."""
    env = load_env_settings()
    configure_logging(
        level=env.log_level,
        log_file=env.log_file,
        json_format=env.log_json,
    )


@lru_cache(maxsize=1)
def get_pipeline() -> PipelineRunner:
    """Return a cached pipeline instance for API requests."""
    setup_logging()
    config = load_config()
    use_gpu = resolve_use_gpu(config.use_gpu)
    return create_paddle_pipeline(language=config.language, use_gpu=use_gpu)


def get_version() -> str:
    """Return the framework version."""
    return __version__


def get_gpu_status() -> bool:
    """Return whether GPU acceleration is available."""
    return is_gpu_available()
