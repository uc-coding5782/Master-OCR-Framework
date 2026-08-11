"""Performance-related data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeviceInfo:
    """Information about available compute devices.

    Attributes:
        gpu_available: Whether GPU acceleration is available.
        device_count: Number of available GPU devices.
        device_type: Type of device (e.g., 'cuda', 'cpu').
        platform: Operating system platform.
        metadata: Additional device metadata.
    """

    gpu_available: bool
    device_count: int
    device_type: str
    platform: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCacheEntry:
    """Entry in the model cache.

    Attributes:
        model: The cached model instance.
        last_used: Timestamp of last usage.
        access_count: Number of times the model was accessed.
        size_bytes: Approximate memory size of the model.
    """

    model: Any
    last_used: float
    access_count: int
    size_bytes: int | None = None


@dataclass
class CacheStats:
    """Statistics about the model cache.

    Attributes:
        total_entries: Total number of cached models.
        total_size_bytes: Total memory usage in bytes.
        hit_count: Number of cache hits.
        miss_count: Number of cache misses.
        hit_rate: Cache hit rate (0.0-1.0).
    """

    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    hit_rate: float
