"""GPU detection and device management."""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocr_framework.performance.models import DeviceInfo


class GPUDetector:
    """Detect GPU availability and capabilities for OCR acceleration.

    The GPUDetector checks for CUDA, ROCm, and other GPU acceleration
    options to inform model loading decisions.
    """

    def __init__(self) -> None:
        """Initialize the GPU detector."""
        self._cuda_available = self._check_cuda()
        self._device_count = self._get_device_count()

    def detect(self) -> "DeviceInfo":
        """Detect available GPU devices.

        Returns:
            DeviceInfo with GPU availability and device count.
        """
        from ocr_framework.performance.models import DeviceInfo

        return DeviceInfo(
            gpu_available=self._cuda_available,
            device_count=self._device_count,
            device_type="cuda" if self._cuda_available else "cpu",
            platform=platform.system(),
        )

    def _check_cuda(self) -> bool:
        """Check if CUDA is available.

        Returns:
            True if CUDA is available, False otherwise.
        """
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_device_count(self) -> int:
        """Get the number of available GPU devices.

        Returns:
            Number of GPU devices (0 if none available).
        """
        if not self._cuda_available:
            return 0

        try:
            import torch

            return torch.cuda.device_count()
        except Exception:
            return 0

    def get_default_device(self) -> str:
        """Get the default device for model execution.

        Returns:
            Device string (e.g., 'cuda:0' or 'cpu').
        """
        if self._cuda_available and self._device_count > 0:
            return "cuda:0"
        return "cpu"

    def is_available(self) -> bool:
        """Check if GPU acceleration is available.

        Returns:
            True if GPU is available, False otherwise.
        """
        return self._cuda_available
