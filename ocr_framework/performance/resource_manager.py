"""Resource cleanup and memory management."""

from __future__ import annotations

import gc
import threading
import weakref
from typing import Any, Callable


class ResourceManager:
    """Manage resource cleanup and memory optimization.

    The ResourceManager tracks resources and ensures proper cleanup
    when they are no longer needed.
    """

    def __init__(self) -> None:
        """Initialize the resource manager."""
        self._resources: list[weakref.ref] = []
        self._lock = threading.RLock()
        self._cleanup_callbacks: list[Callable[[], None]] = []

    def register(self, resource: Any, cleanup: Callable[[], None] | None = None) -> None:
        """Register a resource for cleanup.

        Args:
            resource: Resource to track.
            cleanup: Optional cleanup callback.
        """
        with self._lock:
            if cleanup is not None:
                self._cleanup_callbacks.append(cleanup)

            # Use weakref to avoid preventing garbage collection (only for objects that support it)
            try:
                ref = weakref.ref(resource, self._on_resource_deleted)
                self._resources.append(ref)
            except TypeError:
                # Object doesn't support weakref (e.g., dict, None, str)
                # Just track it by not using weakref - still cleanup will work
                pass

    def _on_resource_deleted(self, ref: weakref.ref) -> None:
        """Handle resource deletion.

        Args:
            ref: Weak reference to the deleted resource.
        """
        with self._lock:
            if ref in self._resources:
                self._resources.remove(ref)

    def cleanup(self) -> None:
        """Perform cleanup of all registered resources."""
        with self._lock:
            # Run cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception:
                    pass

            self._cleanup_callbacks.clear()

            # Force garbage collection
            gc.collect()

    def force_gc(self) -> None:
        """Force garbage collection."""
        gc.collect()

    def get_memory_usage(self) -> dict[str, int]:
        """Get current memory usage statistics.

        Returns:
            Dictionary with memory usage metrics.
        """
        import sys

        return {
            "objects": len(gc.get_objects()),
            "gc_collections": len(gc.get_stats()),
        }
