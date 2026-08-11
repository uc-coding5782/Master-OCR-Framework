"""Typer-based CLI for the OCR framework."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from ocr_framework import __version__, create_batch_processor, create_paddle_pipeline
from ocr_framework.config.loader import load_config
from ocr_framework.config.settings import load_env_settings
from ocr_framework.exceptions import OCRFrameworkError
from ocr_framework.observability.logging import configure_logging, get_logger
from ocr_framework.utils.gpu import is_gpu_available, resolve_use_gpu
from ocr_framework.utils.serialization import document_result_to_dict

app = typer.Typer(
    name="ocr",
    help="OCR Framework — production-ready OCR pipeline CLI.",
    add_completion=False,
    no_args_is_help=True,
)
logger = get_logger("cli")


def _setup_cli_logging(verbose: bool, quiet: bool) -> None:
    env = load_env_settings()
    level = "DEBUG" if verbose else ("ERROR" if quiet else env.log_level)
    configure_logging(level=level, log_file=env.log_file, json_format=env.log_json)


def _handle_error(exc: Exception) -> None:
    if isinstance(exc, OCRFrameworkError):
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
    else:
        typer.secho(f"Unexpected error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command("run")
def run_ocr(
    image: Path = typer.Argument(..., help="Path to input image or PDF", exists=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    profile: str = typer.Option("default", "--profile", "-p", help="Configuration profile"),
    language: str = typer.Option("en", "--lang", "-l", help="OCR language code"),
    gpu: bool = typer.Option(False, "--gpu", help="Enable GPU acceleration"),
    routing: bool = typer.Option(False, "--routing", help="Use intelligent engine routing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-error output"),
) -> None:
    """Run OCR on a single image or document."""
    _setup_cli_logging(verbose, quiet)

    try:
        config = load_config()
        config.profile = profile
        config.language = language
        config.use_gpu = resolve_use_gpu(gpu)

        if not quiet:
            typer.echo(f"Processing: {image}")
            if config.use_gpu:
                typer.echo("GPU acceleration: enabled")
            elif gpu and not is_gpu_available():
                typer.secho("Warning: GPU requested but not available", fg=typer.colors.YELLOW)

        pipeline = create_paddle_pipeline(language=language, use_gpu=config.use_gpu)

        if routing:
            result = pipeline.run_with_routing(image, export_destination=output)
        else:
            result = pipeline.run(image, export_destination=output)

        if output is None and not quiet:
            for page in result.pages:
                for line in page.lines:
                    typer.echo(line.text)

        if not quiet:
            typer.secho(
                f"Done — {result.page_count} page(s), "
                f"{sum(len(p.lines) for p in result.pages)} line(s)",
                fg=typer.colors.GREEN,
            )

    except Exception as exc:
        _handle_error(exc)


@app.command("batch")
def run_batch(
    input_dir: Path = typer.Argument(..., help="Input directory", exists=True),
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", "-o", help="Output directory"),
    profile: str = typer.Option("batch", "--profile", "-p", help="Configuration profile"),
    language: str = typer.Option("en", "--lang", "-l", help="OCR language code"),
    workers: int = typer.Option(4, "--workers", "-w", help="Parallel workers", min=1, max=16),
    gpu: bool = typer.Option(False, "--gpu", help="Enable GPU acceleration"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output"),
) -> None:
    """Process all supported files in a directory."""
    _setup_cli_logging(verbose, quiet)

    try:
        config = load_config()
        config.profile = profile
        config.language = language
        config.use_gpu = resolve_use_gpu(gpu)
        config.batch.workers = workers

        pipeline = create_paddle_pipeline(language=language, use_gpu=config.use_gpu)
        processor = create_batch_processor(pipeline, str(output_dir), silent=quiet)

        if not quiet:
            typer.echo(f"Batch processing: {input_dir} → {output_dir} ({workers} workers)")

        report = processor.process_directory(input_dir)

        if quiet:
            typer.echo(f"{report.succeeded}/{report.succeeded + report.failed} succeeded")
        elif report.failed > 0:
            typer.secho(
                f"Completed with {report.failed} failure(s)",
                fg=typer.colors.YELLOW,
            )

        if report.failed > 0 and not config.batch.continue_on_error:
            raise typer.Exit(code=1)

    except Exception as exc:
        _handle_error(exc)


@app.command("serve")
def serve_api(
    host: str = typer.Option("0.0.0.0", "--host", help="API host"),
    port: int = typer.Option(8000, "--port", help="API port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
) -> None:
    """Start the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        typer.secho(
            "uvicorn is required. Install with: pip install ocr-framework[api]",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Starting OCR API on http://{host}:{port}")
    typer.echo(f"OpenAPI docs: http://{host}:{port}/docs")
    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


@app.command("version")
def show_version() -> None:
    """Show framework version and GPU status."""
    typer.echo(f"OCR Framework v{__version__}")
    typer.echo(f"GPU available: {is_gpu_available()}")


@app.command("config")
def show_config(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path"),
) -> None:
    """Show active configuration."""
    import json

    try:
        config = load_config(config_path)
        typer.echo(json.dumps(
            {
                "profile": config.profile,
                "language": config.language,
                "use_gpu": config.use_gpu,
                "batch_workers": config.batch.workers,
                "preprocessing_mode": config.preprocessing.mode.value,
                "export_formats": [f.value for f in config.export.formats],
            },
            indent=2,
        ))
    except Exception as exc:
        _handle_error(exc)


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
