"""Structured logging configuration for the OCR framework."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured log output."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        for key in ("job_id", "stage", "duration_ms", "engine", "confidence", "path"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def configure_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    json_format: bool = False,
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> None:
    """Configure framework-wide logging.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path for rotating file log output.
        json_format: Use JSON formatting on console when True.
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of rotated log files to retain.
    """
    root = logging.getLogger("ocr_framework")
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter())
    root.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(StructuredFormatter())
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ocr_framework namespace."""
    if not name.startswith("ocr_framework"):
        name = f"ocr_framework.{name}"
    return logging.getLogger(name)
