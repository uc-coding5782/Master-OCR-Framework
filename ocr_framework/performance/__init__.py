"""Performance optimization module."""

from ocr_framework.performance.gpu_detector import GPUDetector
from ocr_framework.performance.model_cache import ModelCache
from ocr_framework.performance.models import CacheStats, DeviceInfo, ModelCacheEntry
from ocr_framework.performance.resource_manager import ResourceManager

__all__ = [
    "GPUDetector",
    "ModelCache",
    "CacheStats",
    "DeviceInfo",
    "ModelCacheEntry",
    "ResourceManager",
]
