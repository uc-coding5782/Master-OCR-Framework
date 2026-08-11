"""Health and version endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from api.dependencies import get_gpu_status, get_version
from api.schemas import HealthResponse, VersionResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health and runtime information.

    Indicates whether the service is reachable and reports GPU availability
    so clients can adjust their workload accordingly.
    """
    return HealthResponse(
        status="ok",
        version=get_version(),
        gpu_available=get_gpu_status(),
    )


@router.get("/version", response_model=VersionResponse)
async def version_info() -> VersionResponse:
    """Return framework and API version information."""
    return VersionResponse(
        framework_version=get_version(),
        api_version="1.0.0",
    )
