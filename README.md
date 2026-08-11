# Master OCR Framework

> Production-ready, extensible multilingual OCR pipeline framework featuring PaddleOCR (v2 & v3 compatibility), Microsoft TrOCR, intelligent engine routing, adaptive preprocessing, and FastAPI web service.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/tests-184%20passed-brightgreen.svg)](#testing)

---

## ✨ Features

- **Multilingual & Multi-Engine OCR**: Supports PaddleOCR (printed text across 80+ languages) and Microsoft TrOCR (handwriting recognition).
- **PaddleOCR v2 & v3 Compatibility**: Robust fallback handling for modern PaddleOCR v3.x and PaddleX pipelines.
- **Punctuation-Preserving Spell Correction**: Automated spell-checking that safely preserves leading/trailing punctuation marks and formatting.
- **Intelligent Engine Routing**: Automatic engine selection and confidence-based fallback chains.
- **Adaptive Preprocessing**: Denoising, deskewing, adaptive contrast enhancement (CLAHE), and automatic upscaling for low-resolution images.
- **Image Quality Analysis**: Blur, brightness, contrast, noise, and resolution estimation.
- **Document Intelligence**: Handwriting detection, language detection, and document classification.
- **Batch Processing**: Multithreaded directory processing with progress tracking.
- **PDF & Image Support**: Supports Multi-page PDF, TIFF, PNG, JPG, BMP formats.
- **Export Formats**: Export extracted text and bounding boxes to TXT, JSON, DOCX, and Searchable PDF.
- **FastAPI REST API**: High-performance API with interactive OpenAPI / Swagger UI documentation.
- **CLI & Quick Script**: Typer-based CLI for production workflows + standalone `main.py` for instant execution.

---

## 🚀 Quick Start

### 1. Installation

Clone your repository and install dependencies:

```bash
git clone https://github.com/uc-coding5782/Master-OCR-Framework.git
cd Master-OCR-Framework

# Install core framework
pip install -e .

# Install PaddleOCR dependencies
pip install -e ".[paddle]"

# Install dev & test dependencies
pip install -e ".[dev]"
```

### 2. Standalone Script Usage (`main.py`)

Run high-accuracy OCR directly on any image:

```bash
# Basic English OCR
python main.py --image document.png

# Specify language (e.g., hindi, french, german, spanish, japanese, korean, chinese)
python main.py --image sample.jpg --lang hindi

# List all supported language shortcuts
python main.py --list-langs

# Save output to text file with custom confidence threshold
python main.py --image scan.jpg --min-confidence 0.65 --output result.txt
```

### 3. CLI Usage

```bash
# Single file OCR
ocr run document.png --lang en

# Export to JSON
ocr run scan.pdf --format json --output result.json

# Batch process a directory
ocr batch ./scans --output-dir ./results --workers 8

# Start the REST API server
ocr serve --port 8000
```

### 4. Python API Usage

```python
from ocr_framework import create_paddle_pipeline

# Create pipeline
pipeline = create_paddle_pipeline(language="en", use_gpu=False)

# Execute OCR
result = pipeline.run("document.png")

# Inspect lines & confidence scores
for page in result.pages:
    for line in page.lines:
        print(f"[{line.confidence:.2f}] {line.text}")
```

### 5. REST API Usage

Start the FastAPI server:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

```bash
# Perform OCR via REST API
curl -X POST "http://localhost:8000/ocr/image" \
  -F "file=@document.png" \
  -F "language=en"
```

---

## 🏗️ Architecture & Project Layout

```text
Master-OCR-Framework/
├── main.py                 # Standalone CLI entry point for fast OCR execution
├── ocr_framework/          # Core framework package
│   ├── config/             # YAML profile loader & Pydantic schemas
│   ├── detection/          # Text detection (PaddleDetector)
│   ├── recognition/        # Text recognition (Paddle, TrOCR)
│   ├── preprocessing/      # Adaptive image preprocessing pipeline
│   ├── postprocessing/     # Confidence filtering & spell correction
│   ├── quality/            # Image quality analyzers (blur, noise, etc.)
│   ├── intelligence/       # Handwriting/language detection
│   ├── routing/            # Engine selector & fallback manager
│   ├── pipeline/           # Pipeline builder & runner orchestration
│   ├── batch/              # Multithreaded batch processing
│   ├── loaders/            # Image and PDF document loaders
│   ├── exporters/          # TXT, JSON, DOCX, Searchable PDF exporters
│   └── models/             # Domain models (OCRLine, PageResult, BoundingBox)
├── api/                    # FastAPI web service & endpoints
├── cli/                    # Typer-based production CLI
├── configs/                # Pre-configured YAML profiles (default, receipt, etc.)
├── tests/                  # Unit and integration test suite
├── pyproject.toml          # Project metadata, dependencies, build settings
└── pyrightconfig.json      # IDE language server import paths
```

---

## ⚙️ Configuration Profiles

Pre-configured YAML profiles in `configs/`:

| Profile | Description |
|---|---|
| `default` | General printed text |
| `document` | High-resolution scanned documents |
| `receipt` | Low-contrast receipts and invoices |
| `handwriting` | Handwritten documents (TrOCR engine) |
| `batch` | Directory batch processing |

---

## 🧪 Testing

Run unit and integration tests using `pytest`:

```bash
python -m pytest
```

---

## 📜 License

Distributed under the [MIT License](LICENSE).