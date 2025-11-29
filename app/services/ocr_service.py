"""Simple OCR service wrapper.

Provides `extract_text_with_easyocr(img_cv)` which tries to use `easyocr` if
available; otherwise returns an empty list. The function returns a list of
dicts with `text` and `bbox` keys to match the rest of the codebase.
"""
from typing import List, Dict
import logging
import numpy as np
import cv2

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


def extract_text_with_easyocr(image_cv: np.ndarray) -> List[Dict]:
    """Run OCR on an OpenCV image (BGR) and return standardized results.

    If `easyocr` is not installed, returns an empty list so the server can
    still start and endpoints return an empty extraction.
    """
    try:
        import easyocr
    except Exception:
        logger.warning("easyocr not available — returning empty OCR results")
        return []

    try:
        img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    except Exception:
        img_rgb = image_cv

    reader = easyocr.Reader(["en"], gpu=False)
    raw = reader.readtext(img_rgb)

    ocr_data: List[Dict] = []
    for bbox, text, conf in raw:
        if not text:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        ocr_data.append({
            "text": text.strip(),
            "confidence": float(conf),
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
        })

    logger.info(f"EasyOCR extracted {len(ocr_data)} regions")
    for item in ocr_data[:20]:
        print(f"TEXT: {item['text'][:50]:50s} | CONF: {item['confidence']:.2f} | BBOX: {item['bbox']}")

    return ocr_data
