"""Basic image preprocessing utilities used by the API.

These implementations are intentionally small and dependency-light so the
project can run in minimal environments. Replace with production-grade
implementations when available.
"""
from typing import Any
import cv2
import numpy as np


def deskew_image(img: Any) -> Any:
    """Return the input image unchanged (placeholder for deskewing)."""
    return img


def enhance_contrast(img: Any) -> Any:
    """Apply a simple CLAHE contrast enhancement on the L channel.

    Accepts an OpenCV image (BGR) and returns same-type image.
    """
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except Exception:
        return img


def image_quality_score(img: Any) -> float:
    """Return a simple heuristic image quality score between 0.0 and 1.0.

    Current heuristic is based on image variance. This is intentionally
    lightweight.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(np.clip(np.var(gray) / 1000.0, 0.0, 1.0))
    except Exception:
        return 0.0
