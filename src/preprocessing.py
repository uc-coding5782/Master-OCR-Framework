"""
Image preprocessing pipeline for OCR.
Improves accuracy by cleaning up images before they hit the OCR engine.
"""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return img


def denoise(img: np.ndarray) -> np.ndarray:
    """Remove noise while preserving edges (important for text)."""
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def deskew(img: np.ndarray) -> np.ndarray:
    """Detect and correct rotation/skew in the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return img  # not enough info to deskew safely

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Skip tiny corrections that aren't worth the interpolation cost
    if abs(angle) < 0.5:
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE (adaptive histogram equalization) to boost text contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def upscale_if_small(img: np.ndarray, min_dim: int = 1000) -> np.ndarray:
    """Upscale small/low-res images so text is easier to recognize."""
    h, w = img.shape[:2]
    if min(h, w) >= min_dim:
        return img
    scale = min_dim / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def preprocess(path: str) -> np.ndarray:
    """Full preprocessing pipeline: load -> upscale -> denoise -> deskew -> contrast."""
    img = load_image(path)
    img = upscale_if_small(img)
    img = denoise(img)
    img = deskew(img)
    img = enhance_contrast(img)
    return img
