"""Document loader interfaces and implementations."""

from ocr_framework.loaders.base import DocumentLoader
from ocr_framework.loaders.image_loader import ImageLoader
from ocr_framework.loaders.pdf_loader import PDFLoader

__all__ = [
    "DocumentLoader",
    "ImageLoader",
    "PDFLoader",
]
