"""Thread-safe model cache for performance optimization."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, TypeVar

from ocr_framework.performance.models import CacheStats, ModelCacheEntry

T = TypeVar("T")


class ModelCache:
    """Thread-safe cache for model instances.

    The ModelCache provides thread-safe access to cached models with
    automatic cleanup and statistics tracking.
    """

    def __init__(self, max_size: int = 10) -> None:
        """Initialize the model cache.

        Args:
            max_size: Maximum number of models to cache.
        """
        self._max_size = max_size
        self._cache: dict[str, ModelCacheEntry] = {}
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0

    def get(self, key: str) -> Any | None:
        """Get a model from the cache.

        Args:
            key: Cache key for the model.

        Returns:
            Cached model instance or None if not found.
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is not None:
                entry.last_used = time.time()
                entry.access_count += 1
                self._hit_count += 1
                return entry.model

            self._miss_count += 1
            return None

    def set(self, key: str, model: Any, size_bytes: int | None = None) -> None:
        """Store a model in the cache.

        Args:
            key: Cache key for the model.
            model: Model instance to cache.
            size_bytes: Approximate memory size of the model.
        """
        with self._lock:
            # Evict oldest entry if cache is full
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_oldest()

            self._cache[key] = ModelCacheEntry(
                model=model,
                last_used=time.time(),
                access_count=1,
                size_bytes=size_bytes,
            )

    def get_or_create(self, key: str, factory: Callable[[], T]) -> T:
        """Get a model from cache or create it using the factory function.

        Args:
            key: Cache key for the model.
            factory: Function to create the model if not cached.

        Returns:
            Model instance (from cache or newly created).
        """
        model = self.get(key)

        if model is not None:
            return model

        # Create new model
        new_model = factory()
        self.set(key, new_model)

        return new_model

    def remove(self, key: str) -> bool:
        """Remove a model from the cache.

        Args:
            key: Cache key for the model.

        Returns:
            True if model was removed, False if not found.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all models from the cache."""
        with self._lock:
            self._cache.clear()

    def _evict_oldest(self) -> None:
        """Evict the least recently used model from the cache."""
        if not self._cache:
            return

        # Find entry with oldest last_used time
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_used)
        del self._cache[oldest_key]

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with current cache metrics.
        """
        with self._lock:
            total_size = sum(entry.size_bytes or 0 for entry in self._cache.values())
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_requests if total_requests > 0 else 0.0

            return CacheStats(
                total_entries=len(self._cache),
                total_size_bytes=total_size,
                hit_count=self._hit_count,
                miss_count=self._miss_count,
                hit_rate=hit_rate,
            )

    def size(self) -> int:
        """Get the current number of cached models.

        Returns:
            Number of cached models.
        """
        with self._lock:
            return len(self._cache)
