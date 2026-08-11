"""GPU detection and device selection utilities."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def is_gpu_available() -> bool:
    """Check whether a CUDA-capable GPU is available.

    Returns:
        True when CUDA is available via Paddle or PyTorch.
    """
    if os.environ.get("OCR_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return False

    try:
        import paddle

        return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        pass

    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        pass

    return False


def resolve_use_gpu(requested: bool | None = None) -> bool:
    """Resolve GPU usage from explicit request or environment.

    Args:
        requested: Explicit GPU preference. When None, reads ``OCR_USE_GPU``.

    Returns:
        Whether GPU acceleration should be enabled.
    """
    if requested is not None:
        return requested and is_gpu_available()

    env_value = os.environ.get("OCR_USE_GPU", "").lower()
    if env_value in ("1", "true", "yes"):
        return is_gpu_available()
    if env_value in ("0", "false", "no"):
        return False

    return False
