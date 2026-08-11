"""Tests for performance optimization components."""

import pytest

from ocr_framework.performance.gpu_detector import GPUDetector
from ocr_framework.performance.model_cache import ModelCache
from ocr_framework.performance.models import DeviceInfo, ModelCacheEntry
from ocr_framework.performance.resource_manager import ResourceManager


class TestGPUDetector:
    """Tests for GPUDetector."""

    def test_detects_device_info(self) -> None:
        """Test device info detection."""
        detector = GPUDetector()
        info = detector.detect()

        assert isinstance(info, DeviceInfo)
        assert isinstance(info.gpu_available, bool)
        assert isinstance(info.device_count, int)
        assert info.device_count >= 0
        assert info.device_type in ["cuda", "cpu"]
        assert isinstance(info.platform, str)

    def test_get_default_device(self) -> None:
        """Test default device selection."""
        detector = GPUDetector()
        device = detector.get_default_device()

        assert device in ["cuda:0", "cpu"]

    def test_is_available(self) -> None:
        """Test GPU availability check."""
        detector = GPUDetector()
        available = detector.is_available()

        assert isinstance(available, bool)


class TestModelCache:
    """Tests for ModelCache."""

    def test_cache_is_empty_initially(self) -> None:
        """Test that cache starts empty."""
        cache = ModelCache()
        assert cache.size() == 0

    def test_cache_set_and_get(self) -> None:
        """Test setting and getting models."""
        cache = ModelCache()
        model = {"key": "value"}

        cache.set("test", model)
        assert cache.size() == 1

        retrieved = cache.get("test")
        assert retrieved == model

    def test_cache_get_or_create(self) -> None:
        """Test get_or_create factory pattern."""
        cache = ModelCache()
        call_count = 0

        def factory() -> dict:
            nonlocal call_count
            call_count += 1
            return {"key": "value"}

        # First call should create
        model1 = cache.get_or_create("test", factory)
        assert call_count == 1

        # Second call should use cache
        model2 = cache.get_or_create("test", factory)
        assert call_count == 1
        assert model1 == model2

    def test_cache_remove(self) -> None:
        """Test removing models from cache."""
        cache = ModelCache()
        model = {"key": "value"}

        cache.set("test", model)
        assert cache.size() == 1

        removed = cache.remove("test")
        assert removed is True
        assert cache.size() == 0

    def test_cache_clear(self) -> None:
        """Test clearing the cache."""
        cache = ModelCache()

        cache.set("test1", {"key": "value1"})
        cache.set("test2", {"key": "value2"})
        assert cache.size() == 2

        cache.clear()
        assert cache.size() == 0

    def test_cache_stats(self) -> None:
        """Test cache statistics."""
        cache = ModelCache()
        model = {"key": "value"}

        cache.set("test", model)
        cache.get("test")
        cache.get("nonexistent")

        stats = cache.get_stats()

        assert stats.total_entries == 1
        assert stats.hit_count == 1
        assert stats.miss_count == 1
        assert stats.hit_rate == 0.5

    def test_cache_max_size_eviction(self) -> None:
        """Test cache eviction when max size is reached."""
        cache = ModelCache(max_size=2)

        cache.set("test1", {"key": "value1"})
        cache.set("test2", {"key": "value2"})
        assert cache.size() == 2

        # Third entry should evict oldest
        cache.set("test3", {"key": "value3"})
        assert cache.size() == 2
        assert cache.get("test1") is None
        assert cache.get("test2") is not None
        assert cache.get("test3") is not None


class TestResourceManager:
    """Tests for ResourceManager."""

    def test_register_resource(self) -> None:
        """Test resource registration."""
        manager = ResourceManager()

        # Use an object that supports weakref
        class TestObject:
            pass

        resource = TestObject()
        manager.register(resource)

    def test_cleanup(self) -> None:
        """Test cleanup callback."""
        manager = ResourceManager()
        cleanup_called = False

        def cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        manager.register(None, cleanup)
        manager.cleanup()

        assert cleanup_called

    def test_force_gc(self) -> None:
        """Test forcing garbage collection."""
        manager = ResourceManager()
        manager.force_gc()
        # Should not raise any errors

    def test_get_memory_usage(self) -> None:
        """Test memory usage statistics."""
        manager = ResourceManager()
        stats = manager.get_memory_usage()

        assert "objects" in stats
        assert "gc_collections" in stats
        assert isinstance(stats["objects"], int)
        assert isinstance(stats["gc_collections"], int)
