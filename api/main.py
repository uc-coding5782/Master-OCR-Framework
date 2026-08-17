"""FastAPI OCR service.

Production-grade HTTP API for the OCR framework. Exposes endpoints for
single-image OCR, multi-page PDF/TIFF OCR, and batch processing, plus
health and version endpoints. OpenAPI documentation is auto-generated and
available at ``/docs`` (Swagger UI) and ``/redoc`` (ReDoc).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

if TYPE_CHECKING:
    from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    """Manage application lifecycle.

    Initializes logging and pipeline on startup; releases resources on
    shutdown.

    Args:
        app: FastAPI application instance.
    """
    # Startup: initialize logging (pipeline is lazily cached on first request)
    from api.dependencies import setup_logging

    setup_logging()
    yield
    # Shutdown: clear cached pipeline and model cache
    try:
        from api.dependencies import get_pipeline

        get_pipeline.cache_clear()
    except Exception:
        pass
    try:
        from ocr_framework.utils.model_cache import ModelCache

        ModelCache.clear()
    except Exception:
        pass


def create_app() -> "FastAPI":
    """Create and configure the FastAPI application.

    Registers all route modules (health, ocr, pdf, batch) and wires up
    exception handlers for framework errors.

    Returns:
        Configured FastAPI application.

    Raises:
        ImportError: If FastAPI is not installed.
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for the API service. "
            "Install it with: pip install fastapi uvicorn python-multipart"
        )

    from pathlib import Path
    from fastapi import FastAPI, Form, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    from api.routes import batch as batch_routes
    from api.routes import health as health_routes
    from api.routes import ocr as ocr_routes
    from api.routes import pdf as pdf_routes
    from api.schemas import ErrorResponse
    from ocr_framework import __version__
    from ocr_framework.exceptions import OCRFrameworkError

    app = FastAPI(
        title="OCR Framework API",
        description=(
            "Production-grade OCR framework with intelligent engine routing, "
            "adaptive preprocessing, image quality analysis, document "
            "intelligence, batch processing, PDF support, and multi-format "
            "export."
        ),
        version=__version__,
        lifespan=lifespan,
        contact={
            "name": "OCR Framework",
            "url": "https://github.com/ocr-framework/ocr-framework",
        },
        license_info={
            "name": "MIT",
        },
    )

    # Enable CORS for flexible client interactions
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files directory
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"message": "OCR Framework API operational. Web UI assets not found."})

    @app.post("/ocr/download", include_in_schema=False)
    async def download_transcript(
        content: str = Form(...),
        filename: str = Form(default="transcript.txt"),
        format: str = Form(default="txt"),
    ):
        from fastapi.responses import Response

        media_types = {
            "txt": "text/plain; charset=utf-8",
            "md": "text/markdown; charset=utf-8",
            "csv": "text/csv; charset=utf-8",
            "json": "application/json; charset=utf-8",
        }
        media_type = media_types.get(format, "text/plain; charset=utf-8")
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        return Response(
            content=content.encode("utf-8"),
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # Register route modules
    app.include_router(health_routes.router)
    app.include_router(ocr_routes.router)
    app.include_router(pdf_routes.router)
    app.include_router(batch_routes.router)

    # Framework error handler -> 422
    @app.exception_handler(OCRFrameworkError)
    async def framework_error_handler(request: Request, exc: OCRFrameworkError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(detail=str(exc), error_type=type(exc).__name__).model_dump(),
        )

    # HTTP exception pass-through (FastAPI handles this, but ensure consistent shape)
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=str(exc.detail), error_type="HTTPException").model_dump(),
        )

    # Catch-all for unexpected errors -> 500
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail=f"Internal server error: {exc}",
                error_type=type(exc).__name__,
            ).model_dump(),
        )

    return app


# Create global app instance when FastAPI is available
app = create_app() if FASTAPI_AVAILABLE else None


def main() -> None:
    """Run the FastAPI server using uvicorn."""
    if not FASTAPI_AVAILABLE:
        print("FastAPI is not installed. Install it with: pip install fastapi uvicorn")
        return

    import uvicorn

    from ocr_framework.config.settings import load_env_settings

    env = load_env_settings()
    uvicorn.run(
        "api.main:app",
        host=env.api_host,
        port=env.api_port,
        workers=env.api_workers,
    )


if __name__ == "__main__":
    main()