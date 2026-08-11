"""Environment-based settings for the OCR framework."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentSettings:
    """Settings loaded from environment variables."""

    log_level: str
    log_file: Path | None
    log_json: bool
    use_gpu: bool | None
    config_path: Path | None
    profile: str
    api_host: str
    api_port: int
    api_workers: int


def load_env_settings() -> EnvironmentSettings:
    """Load settings from environment variables."""
    log_file_raw = os.environ.get("OCR_LOG_FILE")
    config_path_raw = os.environ.get("OCR_CONFIG_PATH")

    use_gpu: bool | None = None
    gpu_env = os.environ.get("OCR_USE_GPU", "").lower()
    if gpu_env in ("1", "true", "yes"):
        use_gpu = True
    elif gpu_env in ("0", "false", "no"):
        use_gpu = False

    return EnvironmentSettings(
        log_level=os.environ.get("OCR_LOG_LEVEL", "INFO").upper(),
        log_file=Path(log_file_raw) if log_file_raw else None,
        log_json=os.environ.get("OCR_LOG_JSON", "").lower() in ("1", "true", "yes"),
        use_gpu=use_gpu,
        config_path=Path(config_path_raw) if config_path_raw else None,
        profile=os.environ.get("OCR_PROFILE", "default"),
        api_host=os.environ.get("OCR_API_HOST", "0.0.0.0"),
        api_port=int(os.environ.get("OCR_API_PORT", "8000")),
        api_workers=int(os.environ.get("OCR_API_WORKERS", "1")),
    )
