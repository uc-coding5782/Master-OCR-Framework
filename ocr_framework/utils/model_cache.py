"""Thread-safe model instance cache for OCR engines."""

from __future__ import annotations

import threading
from typing import Any


class ModelCache:
    """Cache heavy OCR model instances keyed by configuration."""

    _instances: dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def get_or_create(cls, key: str, factory: Any) -> Any:
        """Return a cached model or create one with ``factory``.

        Args:
            key: Unique cache key for the model configuration.
            factory: Callable that creates the model when missing.

        Returns:
            Cached or newly created model instance.
        """
        if key in cls._instances:
            return cls._instances[key]

        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = factory()
            return cls._instances[key]

    @classmethod
    def clear(cls) -> None:
        """Clear all cached model instances."""
        with cls._lock:
            cls._instances.clear()

    @classmethod
    def size(cls) -> int:
        """Return the number of cached model instances."""
        return len(cls._instances)


def paddle_cache_key(lang: str, use_angle_cls: bool, use_gpu: bool) -> str:
    """Build a cache key for PaddleOCR instances."""
    return f"paddle:{lang}:{use_angle_cls}:{use_gpu}"
