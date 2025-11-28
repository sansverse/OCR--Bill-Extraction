"""Simple OCR service wrapper.

Provides `extract_text_with_easyocr(img_cv)` which tries to use `easyocr` if
available; otherwise returns an empty list. The function returns a list of
dicts with `text` and `bbox` keys to match the rest of the codebase.
"""
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def _easyocr_to_output(result):
    # easyocr result element: (bbox, text, confidence)
    bbox, text, conf = result
    # bbox is list of 4 points; convert to simple [x, y, w, h] using min/max
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return {"text": text, "bbox": [x_min, y_min, x_max - x_min, y_max - y_min], "conf": conf}


def extract_text_with_easyocr(img_cv) -> List[Dict]:
    """Run OCR on an OpenCV image (BGR) and return standardized results.

    If `easyocr` is not installed, returns an empty list so the server can
    still start and endpoints return an empty extraction.
    """
    try:
        import easyocr
    except Exception:  # easyocr not installed or import error
        logger.warning("easyocr not available — returning empty OCR results")
        return []

    # Convert image to RGB for easyocr (easyocr expects RGB numpy array)
    try:
        import cv2
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    except Exception:
        img_rgb = img_cv

    reader = easyocr.Reader(["en"], gpu=False)
    raw = reader.readtext(img_rgb)
    return [_easyocr_to_output(r) for r in raw]
